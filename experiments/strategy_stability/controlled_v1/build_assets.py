"""Explicitly create v1 controlled assets. Never called automatically by a run."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random

from backend.teaching_strategies import get_teaching_prompt, get_teaching_prompt_version

ROOT = Path(__file__).resolve().parent
STRATEGIES = ('standard', 'a', 'b', 'ab')


def rel(a, b):
    return {'subject': a, 'object': b}


def context(*, required, original=('A', 'C'), missing=(), missing_rule=False,
            gaps=(), ask=False):
    return {
        'goal': {
            'original': rel(*original),
            'current_subgoals': [rel(*r) for r in required if r != original],
            'askable_gaps': [rel(*r) for r in gaps],
            'missing_relations': [rel(*r) for r in missing],
            'missing_rule': missing_rule,
            'gap_targets': [[rel(*r), rel(*original)] for r in gaps],
        },
        'turn_kind': 'question', 'allow_infer': True,
        'allow_ask': ask, 'allow_clarify': False,
        # Interactive asks defer their served target. Noninteractive targets
        # are all mandatory; interactive targets remain in original/subgoals.
        'required_targets': [] if ask else [rel(*r) for r in required],
    }


def evidence(*texts):
    return [{'message_id': i + 1, 'content': t} for i, t in enumerate(texts)]


def phase(name, question, ctx, gold):
    return {'id': name, 'question': question, 'context': ctx, 'gold': gold}


def gold(direct=(), inferred=(), unknown=(), optional_ask=False):
    return {
        'direct': [list(x) for x in direct],
        'inferred_for_a_ab': [list(x) for x in inferred],
        'unknown': [list(x) for x in unknown],
        'optional_content_ask': optional_ask,
    }


def assets():
    rule = 'Rule: if A links to B and B links to C then A links to C.'
    ab, bc = 'A links to B.', 'B links to C.'
    ac_question = 'Using only the current teaching, report whether A links to C can be confirmed. Do not ask a question.'
    tasks = [
        {'id': 'S1', 'kind': 'fixed', 'evidence': evidence(ab, bc, 'A links to D.', rule),
         'phases': [phase('Q1', 'Using only the current teaching, report A-D, A-C, and C-A separately. Distinguish direct teaching, permitted inference, and inability to confirm. Do not ask a question.',
                          context(required=(('A','D'),('A','C'),('C','A'))),
                          gold(direct=(('A','D','positive'),), inferred=(('A','C'),), unknown=(('C','A'),)))]},
        {'id': 'S2', 'kind': 'fixed', 'evidence': evidence(ab, rule),
         'phases': [phase('Q1', 'Using only the current teaching, report whether A links to C can be confirmed. If teaching is missing, identify the gap and stop without asking a question.',
                          context(required=(('A','C'),), missing=(('B','C'),)),
                          gold(unknown=(('A','C'),)))]},
        {'id': 'S3', 'kind': 'fixed', 'evidence': evidence(ab, bc),
         'phases': [phase('Q1', 'Using only the current teaching, report whether A links to C can be confirmed. Identify any missing rule and stop without asking a question.',
                          context(required=(('A','C'),), missing_rule=True),
                          gold(unknown=(('A','C'),)))]},
        {'id': 'S4', 'kind': 'fixed', 'evidence': evidence(ab, bc, rule + ' Condition: ready. Exception: flagged.', 'Condition ready is true.', 'Exception flagged is active.'),
         'phases': [phase('Q1', ac_question, context(required=(('A','C'),)), gold(unknown=(('A','C'),)))]},
        {'id': 'S5', 'kind': 'fixed', 'evidence': evidence(ab, bc, rule),
         'phases': [phase('Q1', ac_question, context(required=(('A','C'),)), gold(inferred=(('A','C'),))),
                    dict(phase('Q2', 'After the correction, report the current A-B and B-C teaching and whether A-C can be confirmed. Preserve negation. Do not ask a question.',
                               context(required=(('A','B'),('B','C'),('A','C'))),
                               gold(direct=(('A','B','positive'),('B','C','negative')), unknown=(('A','C'),))),
                         teacher_event={'message_id':4, 'content':'Correction: B does not link to C.'})]},
        {'id': 'S6', 'kind': 'fixed', 'evidence': evidence(ab, bc, rule),
         'phases': [phase('Q1', ac_question, context(required=(('A','C'),)), gold(inferred=(('A','C'),))),
                    dict(phase('Q2', 'After the rule revocation, report the current A-B and B-C teaching and whether A-C can be confirmed. Do not ask a question.',
                               context(required=(('A','B'),('B','C'),('A','C'))),
                               gold(direct=(('A','B','positive'),('B','C','positive')), unknown=(('A','C'),))),
                         teacher_event={'message_id':4, 'content':'Revoke rule.'})]},
        {'id': 'I2', 'kind': 'interactive', 'evidence': evidence(ab, rule),
         'phases': [phase('Q1', 'Using only the current teaching, report whether A links to C can be confirmed. If it cannot be confirmed, identify the missing teaching before answering.',
                          context(required=(('A','C'),), missing=(('B','C'),), gaps=(('B','C'),), ask=True),
                          gold(unknown=(('A','C'),), optional_ask=True))],
         'branches': {'missing_fact:B:C': {'message_id':50, 'content':bc},
                      'direct_target:A:C': {'message_id':50, 'content':'A links to C.'}},
         'continuation_question': 'Continue answering the original A-C question using only the current teaching. Do not ask another question.'},
        {'id': 'I3', 'kind': 'interactive', 'evidence': evidence(ab, bc),
         'phases': [phase('Q1', 'Using only the current teaching, report whether A links to C can be confirmed. If a rule is needed to combine the links, identify the missing teaching before answering.',
                          context(required=(('A','C'),), missing_rule=True, ask=True),
                          gold(unknown=(('A','C'),), optional_ask=True))],
         'branches': {'missing_rule:A:C': {'message_id':50, 'content':rule},
                      'direct_target:A:C': {'message_id':50, 'content':'A links to C.'}},
         'continuation_question': 'Continue answering the original A-C question using only the current teaching. Do not ask another question.'},
    ]
    return tasks


def write():
    (ROOT / 'prompts').mkdir(exist_ok=True)
    conditions = []
    for strategy in STRATEGIES:
        product = 'baseline' if strategy == 'standard' else strategy
        prompt = get_teaching_prompt(product)
        path = ROOT / 'prompts' / f'{strategy}.txt'
        path.write_text(prompt, encoding='utf-8')
        conditions.append({
            'id': strategy, 'product_value': product,
            'strategy_prompt': f'prompts/{strategy}.txt',
            'source_prompt_version': get_teaching_prompt_version(product),
            'source_prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(),
            'candidate_prompt_version': f'controlled-v1-{strategy}-json-en',
            'label': strategy.upper() if strategy != 'standard' else 'Standard',
        })
    def dump(name, data):
        (ROOT / name).write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    dump('tasks.json', {'version':'controlled-links-tasks-v1', 'language':'en',
                       'teaching_ingestion_generates_reply':False, 'tasks': assets()})
    dump('conditions.json', {'version':'controlled-conditions-v1', 'conditions':conditions,
                            'model':'deepseek-v4-flash', 'response_format':{'type':'json_object'},
                            'temperature':None, 'max_tokens':4096, 'timeout_seconds':90,
                            'reasoning_effort':'', 'automatic_retries':0})
    rng = random.Random(20261106)
    plan = []
    for task in assets():
        order = list(STRATEGIES)
        rng.shuffle(order)
        for strategy in order:
            plan.append({'ordinal':len(plan)+1, 'task_id':task['id'], 'strategy':strategy,
                         'repeat':1, 'anonymous_id':hashlib.sha256(f"20261106/{task['id']}/{strategy}".encode()).hexdigest()[:12]})
    dump('pilot_plan.json', {'version':'controlled-pilot-plan-v1', 'seed':20261106,
                            'repeats':1, 'sessions':plan,
                            'budget':{'sessions':32, 'base_calls':40, 'optional_calls':4, 'hard_call_cap':44}})


if __name__ == '__main__':
    write()
