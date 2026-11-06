from dataclasses import replace
import json

import httpx
import pytest

from experiments.strategy_stability.integration_app import CandidateRequest, _turn_from_dict
from experiments.strategy_stability.verifier import TeacherMessage, parse_teacher_evidence
from experiments.strategy_stability.controlled_v1 import freeze
from experiments.strategy_stability.controlled_v1.core import load, public_packet, relational_score
from experiments.strategy_stability.controlled_v1.runner import HardStop, RealCandidateSource, AuditTransport, run


def request():
    return CandidateRequest(1,1,'Report the directly taught A-B relation.',
                            (TeacherMessage(1,1,'user','A links to B.'),),
                            _turn_from_dict(load('tasks.json')['tasks'][0]['phases'][0]['context']),
                            'standard','en','deepseek-v4-flash','controlled-links-contract-v1',
                            'strategy-stability-verifier-v5','controlled-v1-standard-json-en',0,0)


def test_atomic_teaching_and_all_events_are_supported():
    tasks = load('tasks.json')['tasks']
    assert {t['id'] for t in tasks} == {'S1','S2','S3','S4','S5','S6','I2','I3'}
    for task in tasks:
        rows = list(task['evidence'])
        for phase in task['phases']:
            if 'teacher_event' in phase:
                rows.append(phase['teacher_event'])
            parsed = parse_teacher_evidence([TeacherMessage(r['message_id'],1,'user',r['content']) for r in rows], session_id=1)
            assert not parsed.unsupported_ids


def test_budget_is_derived_from_executable_query_phases():
    tasks = load('tasks.json')['tasks']
    base = sum(len(t['phases']) for t in tasks) * 4
    optional = sum(t['kind']=='interactive' for t in tasks) * 2
    assert base == 40 and base + optional == 44
    plan = load('pilot_plan.json')
    assert plan['budget']['base_calls'] == base
    assert plan['budget']['hard_call_cap'] == base + optional
    assert len(plan['sessions']) == len({(s['task_id'],s['strategy'],s['repeat']) for s in plan['sessions']}) == 32
    for task in tasks:
        assert {s['strategy'] for s in plan['sessions'] if s['task_id']==task['id']} == {'standard','a','b','ab'}


def test_model_packet_excludes_server_gap_and_gold_oracles():
    packet = public_packet(request())
    assert set(packet) == {'teacher_evidence','public_constraints'}
    serialized = json.dumps(packet)
    for hidden in ('askable_gaps','missing_rule','missing_relations','gap_targets','gold'):
        assert hidden not in serialized
    assert packet['teacher_evidence'][0]['content'] == 'A links to B.'


@pytest.mark.parametrize('mode,expected_calls', [('fake-seek',44),('fake-stop',40)])
def test_full_offline_scripts_keep_actual_budget_and_branches(tmp_path, mode, expected_calls):
    output = tmp_path / mode
    result = run(mode, output)
    assert result['status'] == 'complete', result['hard_stop']
    assert result['model_calls'] == expected_calls
    assert len(result['sessions']) == 32
    assert all(t['http']==200 and t['score']['relational_match'] for s in result['sessions'] for t in s['turns'])
    summary = json.loads((output / 'summary.json').read_text())
    if mode == 'fake-seek':
        assert summary['by_strategy']['ab']['complete_gap_inference_chains'] == 2
        assert summary['by_strategy']['b']['complete_gap_inference_chains'] == 0
    else:
        assert all(s['branch']=='none' for s in result['sessions'])


def test_relational_oracle_detects_unrelated_answer_without_gate():
    wrong = {'items':[{'type':'insufficient','unknown_target':{'subject':'C','object':'A'}}]}
    gold = {'direct':[],'inferred_for_a_ab':[],'unknown':[['A','C']],'optional_content_ask':False}
    assert not relational_score(wrong,gold,'standard')['relational_match']


def test_freeze_detects_source_drift(monkeypatch):
    original = freeze.digest
    monkeypatch.setattr(freeze,'digest',lambda p:'changed' if p.name=='candidate_contract.txt' else original(p))
    with pytest.raises(ValueError,match='frozen asset drift'):
        freeze.verify()


def test_transport_failure_batch_stops_without_retry_or_backfill(tmp_path, monkeypatch):
    from experiments.strategy_stability.controlled_v1 import runner
    from experiments.strategy_stability.controlled_v1.core import FixtureCandidateSource
    class FailedSource(FixtureCandidateSource):
        def generate(self, request):
            super().generate(request)
            raise RuntimeError('synthetic generation failure')
    monkeypatch.setattr(runner,'FixtureCandidateSource',FailedSource)
    result = run('fake-seek',tmp_path/'partial')
    assert result['status']=='stopped'
    assert result['started_sessions']==result['model_calls']==3
    assert result['completed_sessions']==2
    assert 'threshold' in result['hard_stop']


def test_real_adapter_sends_once_records_alias_and_enforces_budget(tmp_path, monkeypatch):
    from backend import config
    from experiments.strategy_stability.controlled_v1 import runner
    monkeypatch.setattr(config,'LLM_API_KEY','fixture-key')
    monkeypatch.setattr(config,'LLM_API_BASE_URL','https://fixture.invalid/v1')
    monkeypatch.setattr(config,'LLM_MODEL','deepseek-v4-flash')
    requests = []
    def handler(req):
        payload = json.loads(req.content)
        requests.append(payload)
        assert payload['response_format'] == {'type':'json_object'}
        assert payload['max_tokens'] == 4096
        return httpx.Response(200,json={'id':'fixture','model':'provider-alias',
            'usage':{'prompt_tokens':12,'completion_tokens':8,'total_tokens':20},
            'choices':[{'message':{'content':'{"items":[]}'},'finish_reason':'stop'}]})
    class MockAudit(AuditTransport):
        def __init__(self):
            self.inner = httpx.MockTransport(handler)
            self.response_metadata = {}
    monkeypatch.setattr(runner,'AuditTransport',MockAudit)
    source = RealCandidateSource(tmp_path,1)
    assert source.generate(request()) == '{"items":[]}'
    assert source.records[0]['provider']['model'] == 'provider-alias'
    assert 'fixture-key' not in (tmp_path/'call-0001.json').read_text()
    with pytest.raises(HardStop,match='budget'):
        source.generate(request())
    assert source.calls == len(requests) == 1
