"""Run frozen controlled scripts against the isolated app, never production."""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from dataclasses import asdict
from enum import Enum
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import time

import httpx
from fastapi.testclient import TestClient

from experiments.strategy_stability.integration_app import create_app
from experiments.strategy_stability.models import ExperimentSession
from experiments.strategy_stability.verifier import TeacherMessage, parse_teacher_evidence
from .core import ROOT, FixtureCandidateSource, build_messages, continuation_gold, load, relational_score
from .freeze import REPO, verify


class HardStop(RuntimeError):
    pass


def safe_json(value):
    if isinstance(value, Enum):
        return value.value
    raise TypeError(type(value).__name__)


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=safe_json) + '\n', encoding='utf-8')


class AuditTransport(httpx.AsyncBaseTransport):
    """Observe this one response's model/usage without any extra request."""
    def __init__(self):
        self.inner = httpx.AsyncHTTPTransport(retries=0)
        self.response_metadata = {}

    async def handle_async_request(self, request):
        response = await self.inner.handle_async_request(request)
        await response.aread()
        try:
            body = response.json()
            if isinstance(body, dict):
                self.response_metadata = {key:body.get(key) for key in ('id','model','usage')}
        except Exception:
            pass
        return response

    async def aclose(self):
        await self.inner.aclose()


class RealCandidateSource:
    def __init__(self, output, cap):
        from backend import config
        self.model = config.LLM_MODEL.strip()
        self.key = config.LLM_API_KEY.strip()
        self.url = config.LLM_API_BASE_URL.strip()
        if not self.key or not self.url or self.model != load('conditions.json')['model']:
            raise ValueError('Explicit configured credentials and the frozen model are required')
        self.calls = 0
        self.records = []
        self.output = output
        self.cap = cap

    def generate(self, request):
        from backend.llm_client import OpenAICompatibleLLMClient
        from backend.review_diagnostics import LLMCallDiagnostics
        verify()
        if self.calls >= self.cap:
            raise HardStop('call budget exhausted')
        if request.model != self.model:
            raise HardStop('requested model snapshot mismatch')
        self.calls += 1
        settings = load('conditions.json')
        messages = build_messages(request)
        record = {'call':self.calls, 'synthetic':False, 'request':asdict(request),
                  'messages':messages, 'raw':None, 'started_unix':time.time()}
        self.records.append(record)
        call_path = self.output / f'call-{self.calls:04d}.json'
        write_json(call_path, record)  # Durable intent before transport.
        transport = AuditTransport()
        diag = LLMCallDiagnostics()
        client = OpenAICompatibleLLMClient(api_key=self.key, base_url=self.url, model=self.model,
                                          timeout=settings['timeout_seconds'], transport=transport,
                                          reasoning_effort=settings['reasoning_effort'], trust_env=False)
        started = time.perf_counter()
        try:
            raw = asyncio.run(client.generate(messages, response_format=settings['response_format'],
                                             temperature=settings['temperature'], max_tokens=settings['max_tokens'],
                                             diagnostics=diag))
            record['raw'] = raw
            return raw
        except Exception as exc:
            record['error_class'] = type(exc).__name__
            raise
        finally:
            record['elapsed_seconds'] = time.perf_counter() - started
            record['diagnostics'] = asdict(diag)
            record['provider'] = transport.response_metadata
            write_json(call_path, record)


def state(client, sid):
    response = client.get(f'/sessions/{sid}')
    if response.status_code != 200:
        raise HardStop('cannot read isolated session')
    return response.json()


def apply_script_event(service, sid, phase):
    """Trusted, frozen teacher event; no model acknowledgment is generated."""
    with service._session_lock(sid):
        with service.SessionLocal.begin() as db:
            row = db.get(ExperimentSession, sid)
            evidence = json.loads(row.evidence_json)
            event = phase['teacher_event']
            if event['message_id'] <= max(m['message_id'] for m in evidence):
                raise HardStop('teacher event is not chronological')
            evidence.append(event)
            parsed = parse_teacher_evidence(
                [TeacherMessage(m['message_id'],sid,'user',m['content']) for m in evidence], session_id=sid)
            if parsed.unsupported_ids:
                raise HardStop('frozen teacher event not supported')
            row.evidence_json = json.dumps(evidence)
            row.turn_context_json = json.dumps(phase['context'])


def run(mode, output):
    manifest_hash = verify()
    if output.exists():
        raise ValueError('Evidence directory must be new; existing runs are never overwritten')
    if output.resolve().is_relative_to(REPO):
        raise ValueError('Run evidence must be outside the repository')
    output.mkdir(parents=True)
    write_json(output / 'frozen-manifest.json', load('manifest.json'))
    plan = load('pilot_plan.json')
    tasks = {t['id']:t for t in load('tasks.json')['tasks']}
    conditions = {c['id']:c for c in load('conditions.json')['conditions']}
    source = RealCandidateSource(output, plan['budget']['hard_call_cap']) if mode == 'real' else FixtureCandidateSource(mode == 'fake-seek')
    results = {'mode':mode, 'manifest_sha256':manifest_hash, 'planned_sessions':len(plan['sessions']),
               'budget':plan['budget'], 'sessions':[], 'status':'running', 'hard_stop':None}
    progress_path = output / 'results.json'
    failures, consecutive_transport_failures = set(), 0

    def persist():
        results['model_calls'] = source.calls
        results['started_sessions'] = len(results['sessions'])
        results['completed_sessions'] = sum(s.get('execution_status')=='completed' for s in results['sessions'])
        write_json(progress_path, results)

    with TemporaryDirectory(prefix='controlled-strategy-pilot-') as tmp:
        app = create_app(Path(tmp) / 'pilot.sqlite3', source)
        with TestClient(app) as client:
            service = app.state.service
            try:
                for planned in plan['sessions']:
                    task = tasks[planned['task_id']]
                    condition = conditions[planned['strategy']]
                    created = client.post('/sessions', json={
                        'strategy':planned['strategy'], 'language':'en',
                        'model_name':load('conditions.json')['model'] if mode == 'real' else 'controlled-fixture',
                        'evidence':task['evidence'], 'turn_context':task['phases'][0]['context']})
                    if created.status_code != 200:
                        raise HardStop('session creation failed')
                    sid = created.json()['id']
                    with service.SessionLocal.begin() as db:
                        row = db.get(ExperimentSession, sid)
                        row.contract_version = 'controlled-links-contract-v1'
                        row.behavior_version = condition['candidate_prompt_version']
                    session_result = {**planned, 'session_id':sid, 'turns':[], 'branch':'none', 'execution_status':'running'}
                    results['sessions'].append(session_result)

                    def send_phase(phase):
                        nonlocal consecutive_transport_failures
                        if source.calls >= plan['budget']['hard_call_cap']:
                            raise HardStop('call cap before request')
                        before = state(client, sid)
                        calls_before = source.calls
                        response = client.post(f'/sessions/{sid}/messages', json={'content':phase['question']})
                        after = state(client, sid)
                        if source.calls - calls_before != 1:
                            raise HardStop('unexpected generation count')
                        record = source.records[-1]
                        if record['request']['question'] != phase['question']:
                            raise HardStop('candidate question snapshot mismatch')
                        score = relational_score(record['raw'], phase['gold'], planned['strategy'])
                        accepted = response.status_code == 200
                        turn = {'phase':phase['id'], 'http':response.status_code, 'body':response.json(),
                                'call':source.calls, 'raw':record['raw'], 'score':score,
                                'before':before, 'after':after}
                        session_result['turns'].append(turn)
                        # Independent fixed relational oracle, not only the gate itself.
                        if accepted and not score['relational_match']:
                            raise HardStop('accepted relational oracle mismatch')
                        assistants_before = sum(m['role']=='assistant' for m in before['messages'])
                        assistants_after = sum(m['role']=='assistant' for m in after['messages'])
                        if not accepted:
                            if assistants_before != assistants_after:
                                raise HardStop('assistant saved after rejected candidate')
                            for key in ('asked','remediated','closed','pending_gap'):
                                if before[key] != after[key]:
                                    raise HardStop('state advanced after rejected candidate')
                        for message in after['messages']:
                            if (message['strategy'] != planned['strategy']
                                    or message['behavior_version'] != condition['candidate_prompt_version']
                                    or message['contract_version'] != record['request']['contract_version']
                                    or message['verifier_version'] != record['request']['verifier_version']
                                    or message['model'] != record['request']['model']
                                    or message['language'] != record['request']['language']):
                                raise HardStop('strategy/version snapshot mismatch')
                        transport_failed = response.json().get('stage') == 'candidate_generation_failed'
                        if transport_failed:
                            failures.add(sid)
                            consecutive_transport_failures += 1
                        else:
                            consecutive_transport_failures = 0
                        persist()
                        print(json.dumps({'mode':mode,'session':planned['ordinal'],'task':task['id'],
                                          'strategy':planned['strategy'],'phase':phase['id'],'http':response.status_code,
                                          'calls':source.calls,'appropriate':accepted and score['relational_match']}), flush=True)
                        if consecutive_transport_failures >= 3 or len(failures) > len(plan['sessions']) * .1:
                            raise HardStop('transport failure review threshold reached')
                        return after, accepted

                    for phase in task['phases']:
                        if 'teacher_event' in phase:
                            apply_script_event(service, sid, phase)
                            session_result.setdefault('teacher_events', []).append(phase['teacher_event'])
                        after, accepted = send_phase(phase)
                    if task['kind'] == 'interactive' and accepted and after['pending_gap'] is not None:
                        pending = after['pending_gap']
                        target = pending['target']
                        key = f"{pending['kind']}:{target['subject']}:{target['object']}"
                        if key not in task['branches']:
                            raise HardStop('unplanned teacher branch')
                        supplement = task['branches'][key]
                        response = client.post(f'/sessions/{sid}/remediation', json={
                            'gap':target,'teacher_message_id':supplement['message_id'],'content':supplement['content']})
                        session_result['branch'] = key
                        session_result['remediation'] = {'http':response.status_code,'body':response.json(),'teacher':supplement}
                        if response.status_code != 200:
                            raise HardStop('frozen matching remediation rejected')
                        # All relevant evidence is now complete; forbid a second ask
                        # by persisted state and the common continuation question.
                        with service.SessionLocal.begin() as db:
                            row = db.get(ExperimentSession, sid)
                            ctx = json.loads(row.turn_context_json)
                            ctx['allow_ask'] = False
                            ctx['required_targets'] = [ctx['goal']['original']]
                            row.turn_context_json = json.dumps(ctx)
                        send_phase({'id':'FINAL','question':task['continuation_question'],
                                    'gold':continuation_gold(task, pending)})
                    session_result['execution_status'] = 'completed'
                    persist()
                results['status'] = 'complete'
            except Exception as exc:
                results['status'] = 'stopped'
                results['hard_stop'] = str(exc) if isinstance(exc, HardStop) else type(exc).__name__
            finally:
                persist()
                write_json(output / 'candidate-records.json', source.records)
                summarize(results, output, source.records)
    print(json.dumps({'status':results['status'],'sessions':len(results['sessions']),
                      'calls':source.calls,'hard_stop':results['hard_stop']}), flush=True)
    return results


def summarize(results, output, records):
    groups = {}
    for strategy in ('standard','a','b','ab'):
        sessions = [s for s in results['sessions'] if s['strategy']==strategy]
        turns = [t for s in sessions for t in s['turns']]
        groups[strategy] = {
            'sessions':len(sessions), 'turns':len(turns),
            'parseable_candidates':sum(t['score']['parseable'] for t in turns),
            'canonical_candidates':sum(t['score']['canonical_items'] for t in turns),
            'raw_relational_matches':sum(t['score']['relational_match'] for t in turns),
            'accepted_turns':sum(t['http']==200 for t in turns),
            'appropriate_turns':sum(t['http']==200 and t['score']['relational_match'] for t in turns),
            'target_answered_turns':sum(t['http']==200 and t['score']['target_answered'] for t in turns),
            'questions':sum(t['http']==200 and t['score']['has_ask'] for t in turns),
            'branches':dict(Counter(s['branch'] for s in sessions)),
            'complete_gap_inference_chains':sum(s['branch'].startswith(('missing_fact:','missing_rule:'))
                and s['turns'][-1]['phase']=='FINAL' and s['turns'][-1]['http']==200
                and bool(s['turns'][-1]['score'].get('inferred')) for s in sessions),
        }
    usage = Counter()
    for record in records:
        for key,value in (record.get('provider', {}).get('usage') or {}).items():
            if isinstance(value,int):
                usage[key] += value
    summary = {'mode':results['mode'],'status':results['status'],'planned_sessions':results['planned_sessions'],
               'started_sessions':len(results['sessions']),
               'completed_script_sessions':results.get('completed_sessions',0),
               'actual_generations':results.get('model_calls',0),
               'requested_model':load('conditions.json')['model'] if results['mode']=='real' else None,
               'returned_models':dict(Counter(r.get('provider',{}).get('model') for r in records if r.get('provider',{}).get('model'))),
               'usage_totals':dict(usage),'by_strategy':groups,
               'limitations':['No human blind ratings or significance claims.',
                              'Relational oracle is not independent proof-source certification.',
                              'Preloaded atomic teaching and fixture events generate no teaching acknowledgment.',
                              'Question branching is deterministic from validated pending, distinct from old human branching.']}
    write_json(output / 'summary.json', summary)
    anonymous = []
    for session in results['sessions']:
        for turn in session['turns']:
            record = records[turn['call']-1]
            anonymous.append({'id':session['anonymous_id'],'phase':turn['phase'],
                              'question':record['request']['question'], 'evidence':record['request']['evidence'],
                              'candidate':turn['raw'],'rendered':turn['body'].get('assistant_text'),
                              'http':turn['http'], 'branch':session['branch']})
    anonymous.sort(key=lambda x:(x['id'],x['phase']))
    write_json(output / 'anonymous_cases.json', anonymous)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('fake-seek','fake-stop','real'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = run(args.mode,args.output)
    raise SystemExit(0 if result['status']=='complete' else 1)
