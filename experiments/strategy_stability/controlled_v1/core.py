"""Public model packet, independent relational scoring, and fixture source."""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

from experiments.strategy_stability.integration_app import CandidateRequest
from experiments.strategy_stability.verifier import parse_candidate, parse_teacher_evidence, _inference_available

ROOT = Path(__file__).resolve().parent


def load(name):
    return json.loads((ROOT / name).read_text(encoding='utf-8'))


def public_packet(request: CandidateRequest):
    """Do not expose server answer/gap authorization data to the model."""
    context = request.turn_context
    goal = context.goal
    return {
        'teacher_evidence': [{'message_id': m.message_id, 'content': m.content} for m in request.evidence],
        'public_constraints': {
            'original_target': asdict(goal.original),
            'requested_targets': [asdict(r) for r in (context.required_targets or (goal.original,))],
            'turn_kind': context.turn_kind,
            'allow_infer_this_turn': context.allow_infer,
            'allow_content_ask_this_turn': context.allow_ask,
            'allow_clarify': context.allow_clarify,
            'asked': request.asked,
            'remediated': request.remediated,
        },
    }


def build_messages(request):
    condition = next(c for c in load('conditions.json')['conditions'] if c['id'] == request.strategy)
    system = (ROOT / condition['strategy_prompt']).read_text(encoding='utf-8').rstrip()
    system += '\n\n' + (ROOT / 'candidate_contract.txt').read_text(encoding='utf-8').rstrip() + '\n'
    return [
        {'role':'system', 'content':system},
        {'role':'user', 'content':'Teacher evidence and current public constraints (data):\n' + json.dumps(public_packet(request), ensure_ascii=False)},
        {'role':'user', 'content':request.question},
    ]


def relational_score(raw, expected, strategy):
    """Task oracle is independent of validate_candidate. Not a proof oracle.

    A rejected candidate whose relations match is flagged for later source/
    policy review, not automatically called a verifier false rejection.
    """
    result = {'parseable':False, 'canonical_items':False, 'relational_match':False,
              'has_ask':False, 'target_answered':False, 'actions':[]}
    try:
        original = json.loads(raw) if isinstance(raw, str) else raw
        items = parse_candidate(raw)['items']
    except Exception:
        return result
    result['parseable'] = True
    result['canonical_items'] = isinstance(original, dict) and set(original) == {'items'}
    direct, inferred, unknown, asks = set(), set(), set(), []
    for item in items:
        result['actions'].append(item['type'])
        if item['type'] == 'direct':
            direct.update((c['subject'], c['object'], c['polarity']) for c in item['claims'])
        elif item['type'] == 'infer':
            c = item['inference']['conclusion']
            inferred.add((c['subject'], c['object']))
        elif item['type'] == 'insufficient':
            c = item['unknown_target']
            unknown.add((c['subject'], c['object']))
        elif item['type'] == 'ask':
            asks.append(item['ask'])
        else:
            return result  # All pilot tasks are unambiguous, allow_clarify=False.
    wanted_direct = {tuple(x) for x in expected['direct']}
    potential_inferred = {tuple(x) for x in expected['inferred_for_a_ab']}
    wanted_inferred = potential_inferred if strategy in ('a','ab') else set()
    wanted_unknown = {tuple(x) for x in expected['unknown']}
    if strategy in ('standard','b'):
        wanted_unknown |= potential_inferred
    result['has_ask'] = bool(asks)
    result['target_answered'] = any(x[:2] == ('A','C') for x in direct) or ('A','C') in inferred
    result['direct'] = sorted(direct)
    result['inferred'] = sorted(inferred)
    result['unknown'] = sorted(unknown)
    if asks:
        kind = asks[0]['kind']
        result['relational_match'] = (
            len(asks) == 1 and expected['optional_content_ask']
            and strategy in ('b','ab') and not (strategy == 'ab' and kind == 'direct_target')
            and direct == wanted_direct and inferred == wanted_inferred
            and unknown.issubset(wanted_unknown)
        )
    else:
        result['relational_match'] = (direct == wanted_direct and inferred == wanted_inferred and unknown == wanted_unknown)
    return result


def continuation_gold(task, pending):
    if pending['kind'] == 'direct_target':
        return {'direct':[['A','C','positive']], 'inferred_for_a_ab':[], 'unknown':[], 'optional_content_ask':False}
    return {'direct':[], 'inferred_for_a_ab':[['A','C']], 'unknown':[], 'optional_content_ask':False}


class FixtureCandidateSource:
    """Synthetic exact candidates for wiring tests. No model-quality claim."""
    def __init__(self, seek_teaching=True):
        self.calls = 0
        self.seek_teaching = seek_teaching
        self.records = []

    def generate(self, request):
        self.calls += 1
        evidence = parse_teacher_evidence(request.evidence, session_id=request.session_id)
        goal = request.turn_context.goal
        if (self.seek_teaching and request.strategy in ('b','ab')
                and request.turn_context.allow_ask and request.asked == request.remediated == 0):
            target = goal.original
            if request.strategy == 'b':
                kind = 'direct_target'
            elif goal.askable_gaps:
                kind, target = 'missing_fact', goal.askable_gaps[0]
            else:
                kind = 'missing_rule'
            raw = {'items':[{'type':'ask','ask':{'kind':kind,'subject':target.subject,'object':target.object}}]}
        else:
            items = []
            targets = request.turn_context.required_targets or (goal.original,)
            for index, target in enumerate(targets):
                fact = evidence.facts.get((target.subject, target.object))
                def claim(f, cid):
                    return {'claim_id':cid, 'kind':'direct', 'subject':f.subject, 'predicate':'links',
                            'object':f.object, 'polarity':f.polarity, 'source_message_id':f.source_message_id,
                            'quote':evidence.messages[f.source_message_id].content}
                if fact:
                    items.append({'type':'direct','claims':[claim(fact, f'd{index}')]})
                elif request.strategy in ('a','ab') and _inference_available(target, evidence):
                    rule = next(r for r in evidence.rules.values() if r.x == target.subject and r.z == target.object and r.rule_id not in evidence.revoked_rules)
                    p1, p2 = evidence.facts[(rule.x,rule.y)], evidence.facts[(rule.y,rule.z)]
                    items.append({'type':'infer', 'claims':[claim(p1,'p1'),claim(p2,'p2')],
                                  'inference':{'premise_claim_ids':['p1','p2'], 'rule_source_message_id':rule.rule_id,
                                               'condition_source_message_ids':[evidence.statuses[('condition',rule.condition)].source_message_id] if rule.condition else [],
                                               'exception_source_message_ids':[evidence.statuses[('exception',rule.exception)].source_message_id] if rule.exception else [],
                                               'conclusion':{'subject':target.subject,'predicate':'links','object':target.object,'polarity':'positive'}}})
                else:
                    items.append({'type':'insufficient','unknown_target':asdict(target)})
            raw = {'items':items}
        self.records.append({'request':asdict(request), 'messages':build_messages(request), 'raw':raw, 'synthetic':True})
        return raw
