"""Isolated tests for the decision-budget A/B/C experiment."""

from __future__ import annotations

import asyncio
import csv
import json
import time
from pathlib import Path

import httpx
import pytest

from backend.general_history.decision import DecisionError as V2DecisionError
from backend.general_history.decision import parse_decision as parse_decision_v2
from backend.general_history.engine import GeneralSearchEngine
from experiments.general_tool_search.decision_protocol_v1 import (
    DECISION_SYSTEM as V1_DECISION_SYSTEM,
    DecisionError as V1DecisionError,
    parse_decision as parse_decision_v1,
)
from experiments.general_tool_search.decision_budget import (
    GROUPS,
    BudgetError,
    DecisionHTTPClient,
    _score_route,
    build_plan,
    build_decision_messages,
    load_cases,
    run_experiment,
    seed_cases,
    _canonical,
    _sha256,
)

ASSET_ROOT = Path(__file__).resolve().parents[1]
CASES = ASSET_ROOT / "fixtures" / "decision_budget_cases.json"


def _resp(body: dict | str | None, *, finish="stop", reasoning=None):
    if isinstance(body, dict):
        content = json.dumps(body)
    else:
        content = body
    usage = {"completion_tokens": 10}
    if reasoning is not None:
        usage["completion_tokens_details"] = {"reasoning_tokens": reasoning}
    return httpx.Response(
        200,
        json={
            "model": "mock-upstream",
            "choices": [{"finish_reason": finish, "message": {"content": content}}],
            "usage": usage,
        },
    )


def _cases():
    return load_cases(CASES, allow_synthetic_session51=True)


def _mock_transport(handler, counter=None):
    def inner(request: httpx.Request) -> httpx.Response:
        if counter is not None:
            counter.append(request)
        return handler(request)

    return httpx.MockTransport(inner)


def test_plan_is_frozen_24_and_has_no_retrieval_or_final(tmp_path):
    cases = _cases()
    requests = []

    def handler(request):
        requests.append(request)
        return _resp(
            {
                "need_history": False,
                "query": "",
                "visible_source_ids": [],
                "reason": "ordinary_recall",
            }
        )

    summary = asyncio.run(
        run_experiment(
            cases=cases,
            client=DecisionHTTPClient(
                base_url="https://mock.invalid/v1",
                model="mock-model",
                transport=_mock_transport(handler),
            ),
            output_dir=tmp_path / "run",
            model="mock-model",
            synthetic=True,
        )
    )
    assert summary["attempts"] == 24
    assert len(requests) == 24
    plan_ids = [item.item_id for item in build_plan(cases)]
    assert len(plan_ids) == 24
    assert len(set(plan_ids)) == 24
    # Only decision requests were sent; no retrieval/final endpoint exists.
    assert all(request.url.path.endswith("/chat/completions") for request in requests)
    meta = json.loads((tmp_path / "run" / "run_meta.json").read_text())
    assert meta["synthetic"] is True
    assert meta["plan_size"] == 24
    assert meta["decision_prompt_sha256"]
    assert meta["case_hashes"]
    rows = [
        json.loads(line)
        for line in (tmp_path / "run" / "attempts.jsonl").read_text().splitlines()
    ]
    assert len(rows) == 24
    assert all("messages" not in row for row in rows)
    assert all(row["synthetic"] is True for row in rows)


def test_a_b_c_request_bodies_differ_only_in_frozen_parameters(tmp_path):
    cases = _cases()
    requests = []

    def handler(request):
        requests.append(request)
        return _resp(
            {
                "need_history": False,
                "query": "",
                "visible_source_ids": [],
                "reason": "ordinary_recall",
            }
        )

    asyncio.run(
        run_experiment(
            cases=cases,
            client=DecisionHTTPClient(
                base_url="https://mock.invalid/v1",
                model="mock-model",
                transport=_mock_transport(handler),
            ),
            output_dir=tmp_path / "runs",
            model="mock-model",
            synthetic=True,
        )
    )
    payloads = [json.loads(request.content) for request in requests]
    # Cases are case-major; repeat-major; group order A/B/C.
    for case_index in range(4):
        for repeat in range(2):
            base = (case_index * 6) + (repeat * 3)
            a, b, c = payloads[base : base + 3]
            assert set(a) - {"max_tokens"} == set(b) - {"max_tokens", "thinking"}
            assert "thinking" not in a and "thinking" not in c
            assert b["thinking"] == {"type": "disabled"}
            assert a["max_tokens"] == 800
            assert b["max_tokens"] == 800
            assert c["max_tokens"] == 2048
            for key in ("model", "messages", "temperature", "response_format"):
                assert a[key] == b[key] == c[key]
            assert all("reasoning_effort" not in item for item in (a, b, c))
            assert a["model"] == "mock-model"


def test_stop_length_empty_and_missing_reasoning_metadata():
    cases = _cases()

    def make(handler):
        return DecisionHTTPClient(
            base_url="https://mock.invalid/v1",
            model="mock-model",
            transport=_mock_transport(handler),
        )

    async def call(handler):
        case = cases[1]
        # Build messages through the same public path by seeding once.
        from experiments.general_tool_search.decision_budget import (
            build_decision_messages,
            seed_cases,
        )

        contexts = seed_cases(cases, Path("/tmp/decision-budget-test.sqlite3"))
        messages = build_decision_messages(contexts[case.case_id])
        return await make(handler).call(
            messages,
            max_tokens=800,
            thinking=None,
            deadline=time.monotonic() + 1,
        )

    async def run():
        stop = await call(lambda request: _resp({"need_history": False, "query": "", "visible_source_ids": [], "reason": "ambiguous"}))
        length = await call(lambda request: _resp("", finish="length"))
        missing = await call(lambda request: _resp({"need_history": False, "query": "", "visible_source_ids": [], "reason": "ambiguous"}))
        return stop, length, missing

    stop, length, missing = asyncio.run(run())
    assert stop.status == "ok"
    assert stop.finish_reason == "stop"
    assert stop.reasoning_tokens is None
    assert length.status == "finish_reason"
    assert length.finish_reason == "length"
    assert length.content == ""
    assert missing.reasoning_tokens is None


def test_invalid_json_and_wrong_route_are_separate(tmp_path):
    cases = _cases()

    def handler(request):
        return _resp(
            {
                "need_history": False,
                "query": "",
                "visible_source_ids": [],
                "reason": "ordinary_recall",
            }
        )

    summary = asyncio.run(
        run_experiment(
            cases=cases,
            client=DecisionHTTPClient(
                base_url="https://mock.invalid/v1",
                model="mock-model",
                transport=_mock_transport(handler),
            ),
            output_dir=tmp_path / "route",
            model="mock-model",
            synthetic=True,
        )
    )
    assert summary["attempts"] == 24
    # Valid JSON but not the frozen route in cases 2-4; case 1 is manual.
    assert summary["groups"]["A"]["valid_json"] == 8
    assert summary["groups"]["A"]["route_correct"] == 0
    assert summary["groups"]["A"]["route_denominator"] == 6




def test_length_with_parseable_body_is_truncated_not_valid(tmp_path):
    cases = _cases()

    def handler(request):
        return _resp(
            {
                "need_history": False,
                "query": "",
                "visible_source_ids": [],
                "reason": "ambiguous",
            },
            finish="length",
        )

    summary = asyncio.run(
        run_experiment(
            cases=cases,
            client=DecisionHTTPClient(
                base_url="https://mock.invalid/v1",
                model="mock-model",
                transport=_mock_transport(handler),
            ),
            output_dir=tmp_path / "length-json",
            model="mock-model",
            synthetic=True,
        )
    )
    assert summary["attempts"] == 24
    assert all(group["valid_json"] == 0 for group in summary["groups"].values())
    assert all(group["truncated"] == 8 for group in summary["groups"].values())


def test_invalid_json_output_never_counts_as_route(tmp_path):
    cases = _cases()

    def handler(request):
        return _resp("{not-json")

    out = tmp_path / "invalid"
    summary = asyncio.run(
        run_experiment(
            cases=cases,
            client=DecisionHTTPClient(
                base_url="https://mock.invalid/v1",
                model="mock-model",
                transport=_mock_transport(handler),
            ),
            output_dir=out,
            model="mock-model",
            synthetic=True,
        )
    )
    assert summary["attempts"] == 24
    assert all(group["valid_json"] == 0 for group in summary["groups"].values())
    assert all(group["route_correct"] == 0 for group in summary["groups"].values())


def test_timeout_records_transport_entry_and_no_result(tmp_path):
    cases = _cases()
    from experiments.general_tool_search.decision_budget import (
        build_decision_messages,
        seed_cases,
    )

    class SlowTransport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):
            await asyncio.sleep(0.2)
            return _resp({"need_history": False, "query": "", "visible_source_ids": [], "reason": "ambiguous"})

    contexts = seed_cases(cases, tmp_path / "timeout.sqlite3")
    messages = build_decision_messages(contexts["ambiguous_reference"])

    async def run():
        client = DecisionHTTPClient(
            base_url="https://mock.invalid/v1",
            model="mock-model",
            transport=SlowTransport(),
        )
        return await client.call(
            messages,
            max_tokens=800,
            thinking=None,
            deadline=time.monotonic() + 0.01,
        )

    result = asyncio.run(run())
    assert result.status == "timeout"
    assert result.transport_started is True
    assert result.content == ""


def test_preflight_rejects_missing_external_and_existing_evidence_dir(tmp_path):
    with pytest.raises(BudgetError):
        load_cases(CASES, allow_synthetic_session51=False)
    out = tmp_path / "exists"
    out.mkdir()
    with pytest.raises(BudgetError, match="evidence_directory_exists"):
        asyncio.run(
            run_experiment(
                cases=_cases(),
                client=object(),
                output_dir=out,
                model="mock-model",
                synthetic=True,
            )
        )


def test_real_mode_missing_credential_fails_before_request(tmp_path, monkeypatch, capsys):
    from experiments.general_tool_search.decision_budget import main

    monkeypatch.delenv("DECISION_BUDGET_NO_KEY", raising=False)
    code = main(
        [
            "--real-api",
            "--evidence-dir",
            str(tmp_path / "real"),
            "--session51-input",
            "/does/not/exist.json",
            "--api-key-env",
            "DECISION_BUDGET_NO_KEY",
        ]
    )
    assert code == 2
    assert "missing credential" in capsys.readouterr().out
    assert not (tmp_path / "real").exists()


def test_score_route_rejects_wrong_visible_ids_and_query():
    decision = {
        "need_history": False,
        "query": "",
        "visible_source_ids": [2],
        "reason": "visible_complete",
    }
    assert _score_route(decision, {"reason": "visible_complete", "need_history": False, "visible_labels": ["x"]}, {"x": 3}) is False
    assert _score_route(
        {"need_history": True, "query": "other", "visible_source_ids": [], "reason": "ordinary_recall"},
        {"reason": "ordinary_recall", "need_history": True, "query_terms": ["onboarding"]},
        {},
    ) is False


def test_production_engine_probe_enters_decide_for_all_four_cases(tmp_path):
    from dataclasses import replace
    from experiments.general_tool_search.decision_budget import probe_cases

    cases = _cases()
    observed = asyncio.run(
        probe_cases(cases, database_path=tmp_path / "probe.sqlite3")
    )
    assert set(observed) == {
        "session51_original",
        "visible_complete",
        "hidden_correction",
        "ambiguous_reference",
    }
    # The probe stops at decision and must not reach final/retrieval.
    for messages in observed.values():
        assert messages[0]["role"] == "system"
        assert len(messages) == 3


def test_explicit_history_question_is_rejected_by_production_probe(tmp_path):
    from dataclasses import replace
    from experiments.general_tool_search.decision_budget import BudgetError, probe_cases

    cases = _cases()
    cases[0] = replace(
        cases[0],
        current="What was the original difference quotient I taught you?",
    )
    with pytest.raises(BudgetError, match="case_explicit_v1_history|final_before_decision|not_model_decision"):
        asyncio.run(probe_cases(cases, database_path=tmp_path / "probe.sqlite3"))


def test_evidence_dir_has_no_sqlite_copy(tmp_path):
    cases = _cases()

    def handler(request):
        return _resp(
            {
                "need_history": False,
                "query": "",
                "visible_source_ids": [],
                "reason": "ordinary_recall",
            }
        )

    asyncio.run(
        run_experiment(
            cases=cases,
            client=DecisionHTTPClient(
                base_url="https://mock.invalid/v1",
                model="mock-model",
                transport=_mock_transport(handler),
            ),
            output_dir=tmp_path / "no-db",
            model="mock-model",
            synthetic=True,
        )
    )
    assert not list((tmp_path / "no-db").rglob("*.sqlite3"))
    assert not list((tmp_path / "no-db").rglob("*.db"))


def test_no_response_is_counted_separately_from_blank_body(tmp_path):
    cases = _cases()

    def handler(request):
        raise httpx.ReadTimeout("synthetic timeout")

    result = asyncio.run(
        run_experiment(
            cases=cases,
            client=DecisionHTTPClient(
                base_url="https://mock.invalid/v1",
                model="mock-model",
                transport=httpx.MockTransport(handler),
                timeout_seconds=0.2,
            ),
            output_dir=tmp_path / "no-response",
            model="mock-model",
            synthetic=True,
        )
    )
    attempts = [
        json.loads(line)
        for line in (tmp_path / "no-response" / "attempts.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert len(attempts) == 24
    assert all(item["no_response"] is True for item in attempts)
    assert all(item["content_empty"] is False for item in attempts)
    summary = json.loads((tmp_path / "no-response" / "summary.json").read_text())
    assert summary["attempts"] == 24
    assert all(group["no_response"] == 8 for group in summary["groups"].values())
    assert all(group["empty"] == 0 for group in summary["groups"].values())
    assert result["attempts"] == 24


def _valid_external_payload():
    return {
        "case_id": "session51_original",
        "current_message_id": 750,
        "messages": [
            {
                "message_id": message_id,
                "role": "user",
                "content": f"Synthetic original lesson {message_id}",
            }
            for message_id in range(729, 750)
        ],
        "current": "Which difference quotient applies here?",
        "expected": {
            "manual": True,
            "route_criteria": "Human review required.",
        },
    }


def _write_external(tmp_path, payload):
    path = tmp_path / "external.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_external_input_wins_even_if_case_flag_is_disabled(tmp_path):
    data = json.loads(CASES.read_text(encoding="utf-8"))
    data["cases"][0]["requires_external_input"] = False
    case_file = tmp_path / "cases.json"
    case_file.write_text(json.dumps(data), encoding="utf-8")
    external = _write_external(tmp_path, _valid_external_payload())

    cases = load_cases(
        case_file,
        external_session51=external,
        allow_synthetic_session51=False,
    )
    fixed = cases[0]
    assert fixed.case_id == "session51_original"
    assert fixed.source == "external"
    assert fixed.synthetic is False
    assert fixed.current_message_id == 750
    assert [item["message_id"] for item in fixed.messages] == list(range(729, 750))


@pytest.mark.parametrize(
    "mutation, message",
    [
        (lambda p: p.update(current_message_id=749), "must_be_750"),
        (
            lambda p: p["messages"][0].pop("message_id"),
            "message_id_invalid",
        ),
        (
            lambda p: p.update(expected={"manual": True}),
            "missing_route_criteria",
        ),
        (
            lambda p: p["messages"].reverse(),
            "order_invalid",
        ),
    ],
)
def test_invalid_external_input_fails_before_any_request(tmp_path, mutation, message):
    payload = _valid_external_payload()
    mutation(payload)
    external = _write_external(tmp_path, payload)
    with pytest.raises(BudgetError, match=message):
        load_cases(
            CASES,
            external_session51=external,
            allow_synthetic_session51=False,
        )


def test_external_window_ids_are_preserved_in_actual_request(tmp_path):
    data = json.loads(CASES.read_text(encoding="utf-8"))
    case_file = tmp_path / "cases.json"
    case_file.write_text(json.dumps(data), encoding="utf-8")
    external = _write_external(tmp_path, _valid_external_payload())
    cases = load_cases(
        case_file,
        external_session51=external,
        allow_synthetic_session51=False,
    )
    requests = []

    def handler(request):
        requests.append(request)
        return _resp(
            {
                "need_history": False,
                "query": "",
                "visible_source_ids": [],
                "reason": "ordinary_recall",
            }
        )

    asyncio.run(
        run_experiment(
            cases=cases,
            client=DecisionHTTPClient(
                base_url="https://mock.invalid/v1",
                model="mock-model",
                transport=_mock_transport(handler),
            ),
            output_dir=tmp_path / "external-run",
            model="mock-model",
            synthetic=True,
        )
    )
    first_case_requests = requests[:6]
    assert len(first_case_requests) == 6
    payload = json.loads(first_case_requests[0].content)
    window = json.loads(payload["messages"][1]["content"])
    assert [item["message_id"] for item in window["visible_teacher"]] == list(
        range(730, 750)
    )
    assert payload["messages"][-1]["content"] == "Which difference quotient applies here?"


def test_http_error_is_not_counted_as_model_empty_body(tmp_path):
    cases = _cases()

    def handler(request):
        return httpx.Response(401, json={"error": "unauthorized"})

    result = asyncio.run(
        run_experiment(
            cases=cases,
            client=DecisionHTTPClient(
                base_url="https://mock.invalid/v1",
                model="mock-model",
                transport=_mock_transport(handler),
            ),
            output_dir=tmp_path / "http-error",
            model="mock-model",
            synthetic=True,
        )
    )
    rows = [
        json.loads(line)
        for line in (tmp_path / "http-error" / "attempts.jsonl").read_text().splitlines()
    ]
    assert all(row["http_error"] is True for row in rows)
    assert all(row["content_empty"] is False for row in rows)
    assert all(row["no_response"] is False for row in rows)
    assert all(group["http_error"] == 8 for group in result["groups"].values())
    assert all(group["empty"] == 0 for group in result["groups"].values())


def test_blank_200_and_length_blank_are_model_empty_body(tmp_path):
    cases = _cases()

    def handler(request):
        return _resp("", finish="stop")

    result = asyncio.run(
        run_experiment(
            cases=cases,
            client=DecisionHTTPClient(
                base_url="https://mock.invalid/v1",
                model="mock-model",
                transport=_mock_transport(handler),
            ),
            output_dir=tmp_path / "blank",
            model="mock-model",
            synthetic=True,
        )
    )
    rows = [
        json.loads(line)
        for line in (tmp_path / "blank" / "attempts.jsonl").read_text().splitlines()
    ]
    assert all(row["content_empty"] is True for row in rows)
    assert all(row["http_error"] is False for row in rows)
    assert all(row["no_response"] is False for row in rows)
    assert all(row["output_valid"] is False for row in rows)
    assert all(group["empty"] == 8 for group in result["groups"].values())


def test_custom_case_cannot_disable_external_session51_requirement(tmp_path):
    data = json.loads(CASES.read_text(encoding="utf-8"))
    data["cases"][0]["requires_external_input"] = False
    case_file = tmp_path / "cases.json"
    case_file.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(BudgetError, match="missing_session51_external_input"):
        load_cases(case_file, allow_synthetic_session51=False)


def test_length_with_blank_body_is_truncated_and_empty(tmp_path):
    cases = _cases()

    def handler(request):
        return _resp("", finish="length")

    summary = asyncio.run(
        run_experiment(
            cases=cases,
            client=DecisionHTTPClient(
                base_url="https://mock.invalid/v1",
                model="mock-model",
                transport=_mock_transport(handler),
            ),
            output_dir=tmp_path / "length-blank",
            model="mock-model",
            synthetic=True,
        )
    )
    rows = [
        json.loads(line)
        for line in (tmp_path / "length-blank" / "attempts.jsonl").read_text().splitlines()
    ]
    assert all(row["finish_reason"] == "length" for row in rows)
    assert all(row["truncated"] is True for row in rows)
    assert all(row["content_empty"] is True for row in rows)
    assert all(row["output_valid"] is False for row in rows)
    assert all(group["empty"] == 8 for group in summary["groups"].values())


def test_real_mode_requires_model_and_finite_positive_timeout(tmp_path, monkeypatch, capsys):
    from experiments.general_tool_search.decision_budget import main

    monkeypatch.setenv("DECISION_BUDGET_KEY", "synthetic")
    common = [
        "--real-api",
        "--api-key-env",
        "DECISION_BUDGET_KEY",
        "--evidence-dir",
        str(tmp_path / "real"),
        "--session51-input",
        str(tmp_path / "missing.json"),
    ]
    assert main(common) == 2
    assert "model is required" in capsys.readouterr().out
    assert main(common + ["--model", "deepseek-v4-flash"]) == 2
    assert "decision-timeout is required" in capsys.readouterr().out
    assert main(
        common
        + [
            "--model",
            "deepseek-v4-flash",
            "--decision-timeout",
            "0",
        ]
    ) == 2
    assert "finite positive" in capsys.readouterr().out


def test_200_invalid_structure_is_not_model_empty(tmp_path):
    cases = _cases()

    def handler(request):
        return httpx.Response(200, json={"model": "mock-upstream"})

    result = asyncio.run(
        run_experiment(
            cases=cases,
            client=DecisionHTTPClient(
                base_url="https://mock.invalid/v1",
                model="mock-model",
                transport=_mock_transport(handler),
            ),
            output_dir=tmp_path / "invalid-structure",
            model="mock-model",
            synthetic=True,
        )
    )
    rows = [
        json.loads(line)
        for line in (tmp_path / "invalid-structure" / "attempts.jsonl").read_text().splitlines()
    ]
    assert all(row["http_status"] == 200 for row in rows)
    assert all(row["error_code"] == "invalid_response" for row in rows)
    assert all(row["content_empty"] is False for row in rows)
    assert all(row["http_error"] is False for row in rows)
    assert all(row["no_response"] is False for row in rows)
    assert all(row["output_valid"] is False for row in rows)
    assert all(group["invalid_response"] == 8 for group in result["groups"].values())
    assert all(group["empty"] == 0 for group in result["groups"].values())


def test_200_blank_or_whitespace_is_empty_body_not_invalid_json(tmp_path):
    cases = _cases()

    def handler(request):
        return _resp("   \n\t", finish="stop")

    result = asyncio.run(
        run_experiment(
            cases=cases,
            client=DecisionHTTPClient(
                base_url="https://mock.invalid/v1",
                model="mock-model",
                transport=_mock_transport(handler),
            ),
            output_dir=tmp_path / "blank-whitespace",
            model="mock-model",
            synthetic=True,
        )
    )
    rows = [
        json.loads(line)
        for line in (tmp_path / "blank-whitespace" / "attempts.jsonl").read_text().splitlines()
    ]
    assert all(row["content_empty"] is True for row in rows)
    assert all(row["error_code"] == "empty_body" for row in rows)
    assert all(row["decision_error"] == "empty_body" for row in rows)
    assert all(row["http_error"] is False for row in rows)
    assert all(row["no_response"] is False for row in rows)
    assert all(group["empty"] == 8 for group in result["groups"].values())
    assert all(group["invalid_response"] == 0 for group in result["groups"].values())


def test_200_length_blank_is_truncated_and_empty_but_not_valid(tmp_path):
    cases = _cases()

    def handler(request):
        return _resp("", finish="length")

    result = asyncio.run(
        run_experiment(
            cases=cases,
            client=DecisionHTTPClient(
                base_url="https://mock.invalid/v1",
                model="mock-model",
                transport=_mock_transport(handler),
            ),
            output_dir=tmp_path / "length-empty",
            model="mock-model",
            synthetic=True,
        )
    )
    rows = [
        json.loads(line)
        for line in (tmp_path / "length-empty" / "attempts.jsonl").read_text().splitlines()
    ]
    assert all(row["http_status"] == 200 for row in rows)
    assert all(row["finish_reason"] == "length" for row in rows)
    assert all(row["truncated"] is True for row in rows)
    assert all(row["content_empty"] is True for row in rows)
    assert all(row["output_valid"] is False for row in rows)
    assert all(group["truncated"] == 8 for group in result["groups"].values())
    assert all(group["empty"] == 8 for group in result["groups"].values())
    assert all(group["valid_json"] == 0 for group in result["groups"].values())


def test_http_201_with_valid_structure_uses_2xx_success_semantics(tmp_path):
    cases = _cases()

    def handler(request):
        return httpx.Response(
            201,
            json={
                "model": "mock-upstream",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": json.dumps(
                                {
                                    "need_history": False,
                                    "query": "",
                                    "visible_source_ids": [],
                                    "reason": "ordinary_recall",
                                }
                            )
                        },
                    }
                ],
                "usage": {"completion_tokens": 10},
            },
        )

    result = asyncio.run(
        run_experiment(
            cases=cases,
            client=DecisionHTTPClient(
                base_url="https://mock.invalid/v1",
                model="mock-model",
                transport=_mock_transport(handler),
            ),
            output_dir=tmp_path / "http-201",
            model="mock-model",
            synthetic=True,
        )
    )
    rows = [
        json.loads(line)
        for line in (tmp_path / "http-201" / "attempts.jsonl").read_text().splitlines()
    ]
    assert all(row["http_status"] == 201 for row in rows)
    assert all(row["http_error"] is False for row in rows)
    assert all(row["content_empty"] is False for row in rows)
    assert all(row["output_valid"] is True for row in rows)
    assert all(group["http_error"] == 0 for group in result["groups"].values())
    assert all(group["valid_json"] == 8 for group in result["groups"].values())


def test_200_missing_message_is_invalid_not_empty(tmp_path):
    cases = _cases()

    def handler(request):
        return httpx.Response(
            200,
            json={"choices": [{"finish_reason": "stop"}]},
        )

    asyncio.run(
        run_experiment(
            cases=cases,
            client=DecisionHTTPClient(
                base_url="https://mock.invalid/v1",
                model="mock-model",
                transport=_mock_transport(handler),
            ),
            output_dir=tmp_path / "missing-message",
            model="mock-model",
            synthetic=True,
        )
    )
    rows = [
        json.loads(line)
        for line in (tmp_path / "missing-message" / "attempts.jsonl").read_text().splitlines()
    ]
    assert all(row["error_code"] == "invalid_response" for row in rows)
    assert all(row["content_empty"] is False for row in rows)
    assert all(row["http_error"] is False for row in rows)


def _invalid_metadata_response():
    return httpx.Response(
        200,
        json={
            "model": "mock-upstream",
            "choices": [
                {
                    "finish_reason": "length",
                    "message": {"content": None},
                }
            ],
            "usage": {
                "completion_tokens": 10,
                "completion_tokens_details": {"reasoning_tokens": 7},
            },
        },
    )


def test_null_content_preserves_finish_reason_and_usage(tmp_path):
    cases = _cases()

    def handler(request):
        return _invalid_metadata_response()

    result = asyncio.run(
        run_experiment(
            cases=cases,
            client=DecisionHTTPClient(
                base_url="https://mock.invalid/v1",
                model="mock-model",
                transport=_mock_transport(handler),
            ),
            output_dir=tmp_path / "null-content",
            model="mock-model",
            synthetic=True,
        )
    )
    rows = [
        json.loads(line)
        for line in (tmp_path / "null-content" / "attempts.jsonl").read_text().splitlines()
    ]
    assert all(row["http_status"] == 200 for row in rows)
    assert all(row["error_code"] == "invalid_response" for row in rows)
    assert all(row["finish_reason"] == "length" for row in rows)
    assert all(row["truncated"] is True for row in rows)
    assert all(row["content_empty"] is False for row in rows)
    assert all(row["no_response"] is False for row in rows)
    assert all(row["transport_started"] is True for row in rows)
    assert all(row["completion_tokens"] == 10 for row in rows)
    assert all(row["reasoning_tokens"] == 7 for row in rows)
    assert all(row["output_valid"] is False for row in rows)
    assert all(group["invalid_response"] == 8 for group in result["groups"].values())
    assert all(group["truncated"] == 8 for group in result["groups"].values())
    assert all(group["empty"] == 0 for group in result["groups"].values())


@pytest.mark.parametrize("bad_message", [["not", "a", "dict"], "not-a-dict"])
def test_non_object_message_is_invalid_response_with_http_metadata(tmp_path, bad_message):
    cases = _cases()

    def handler(request):
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": bad_message,
                    }
                ],
                "usage": {"completion_tokens": 10},
            },
        )

    asyncio.run(
        run_experiment(
            cases=cases,
            client=DecisionHTTPClient(
                base_url="https://mock.invalid/v1",
                model="mock-model",
                transport=_mock_transport(handler),
            ),
            output_dir=tmp_path / f"non-object-{type(bad_message).__name__}",
            model="mock-model",
            synthetic=True,
        )
    )
    rows = [
        json.loads(line)
        for line in (
            tmp_path / f"non-object-{type(bad_message).__name__}" / "attempts.jsonl"
        ).read_text().splitlines()
    ]
    assert all(row["http_status"] == 200 for row in rows)
    assert all(row["transport_started"] is True for row in rows)
    assert all(row["error_code"] == "invalid_response" for row in rows)
    assert all(row["no_response"] is False for row in rows)
    assert all(row["content_empty"] is False for row in rows)
    assert all(row["completion_tokens"] == 10 for row in rows)


@pytest.mark.parametrize(
    "bad_payload",
    [
        [],
        {"choices": ["not-a-choice"]},
        {"choices": []},
    ],
)
def test_invalid_root_or_first_choice_is_invalid_response(tmp_path, bad_payload):
    cases = _cases()
    name = f"bad-shape-{abs(hash(str(bad_payload))) % 100000}"

    def handler(request):
        return httpx.Response(200, json=bad_payload)

    asyncio.run(
        run_experiment(
            cases=cases,
            client=DecisionHTTPClient(
                base_url="https://mock.invalid/v1",
                model="mock-model",
                transport=_mock_transport(handler),
            ),
            output_dir=tmp_path / name,
            model="mock-model",
            synthetic=True,
        )
    )
    rows = [
        json.loads(line)
        for line in (tmp_path / name / "attempts.jsonl").read_text().splitlines()
    ]
    assert all(row["http_status"] == 200 for row in rows)
    assert all(row["transport_started"] is True for row in rows)
    assert all(row["error_code"] == "invalid_response" for row in rows)
    assert all(row["no_response"] is False for row in rows)
    assert all(row["output_valid"] is False for row in rows)


HISTORICAL_RUN = Path(
    "/Users/luo/Documents/teachable_agent_runs/decision_budget_real_20261104_2120"
)
HISTORICAL_INPUT = Path(
    "/Users/luo/Documents/teachable_agent_runs/session51_decision_input_20261104.json"
)


def test_v1_prompt_and_24_request_hashes_are_frozen(tmp_path):
    if not (HISTORICAL_RUN / "attempts.csv").exists() or not HISTORICAL_INPUT.exists():
        pytest.skip("historical decision-budget evidence is unavailable")
    meta = json.loads((HISTORICAL_RUN / "run_meta.json").read_text())
    assert _sha256(V1_DECISION_SYSTEM) == meta["decision_prompt_sha256"]
    cases = load_cases(
        CASES,
        external_session51=HISTORICAL_INPUT,
        allow_synthetic_session51=False,
    )
    database_path = tmp_path / "v1-hash.sqlite3"
    contexts = seed_cases(cases, database_path)
    rows = list(csv.DictReader((HISTORICAL_RUN / "attempts.csv").open()))
    assert len(rows) == 24
    for row in rows:
        context = contexts[row["case_id"]]
        messages = build_decision_messages(context)
        group = GROUPS[row["group"]]
        payload = {
            "model": row["request_model"],
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": group["max_tokens"],
            "response_format": {"type": "json_object"},
        }
        if group["thinking"] is not None:
            payload["thinking"] = group["thinking"]
        assert _sha256(_canonical(payload)) == row["request_sha256"]


class _ProbeReached(Exception):
    pass


class _DecisionProbe:
    synthetic = True

    def __init__(self):
        self.called = False
        self.messages = None

    async def generate_decision(self, messages, *, deadline):
        self.called = True
        self.messages = messages
        raise _ProbeReached()


class _FinalProbe:
    synthetic = True

    async def generate_final(self, messages, *, deadline=None):
        raise AssertionError("final must not be reached before decision")


async def _probe_v2_cases(cases, database_path):
    contexts = seed_cases(cases, database_path)
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    engine = create_engine(f"sqlite:///{database_path}")
    factory = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    try:
        db = factory()
        try:
            for case in cases:
                context = contexts[case.case_id]
                decision = _DecisionProbe()
                engine_instance = GeneralSearchEngine(
                    db,
                    session_id=context.session_id,
                    current_message_id=context.current_id,
                    decision_client=decision,
                    final_client=_FinalProbe(),
                    database_path=str(database_path),
                    decision_timeout=1.0,
                    overall_timeout=2.0,
                )
                try:
                    await engine_instance.run()
                except _ProbeReached:
                    pass
                assert decision.called, case.case_id
                assert decision.messages[-1]["content"] == case.current
                assert "visible_teacher" in decision.messages[1]["content"]
        finally:
            db.close()
    finally:
        engine.dispose()


def test_v2_production_probe_reaches_decision_for_all_four_cases(tmp_path):
    cases = _cases()
    asyncio.run(_probe_v2_cases(cases, tmp_path / "probe-v2.sqlite3"))


def test_v1_and_v2_reason_sets_cross_reject():
    raw_missing = json.dumps({
        "need_history": True,
        "query": "divisor nonzero",
        "visible_source_ids": [],
        "reason": "missing_fact",
    })
    with pytest.raises(V1DecisionError, match="invalid_reason"):
        parse_decision_v1(raw_missing, visible_ids=(1,))
    assert parse_decision_v2(raw_missing, visible_ids=(1,)).decision.reason == "missing_fact"

    raw_old = json.dumps({
        "need_history": True,
        "query": "divisor nonzero",
        "visible_source_ids": [],
        "reason": "ordinary_recall",
    })
    assert parse_decision_v1(raw_old, visible_ids=(1,)).decision.reason == "ordinary_recall"
    with pytest.raises(V2DecisionError, match="invalid_reason"):
        parse_decision_v2(raw_old, visible_ids=(1,))
