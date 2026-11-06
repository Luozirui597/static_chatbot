"""MockTransport tests for the opt-in real API adapter."""

from __future__ import annotations

import asyncio
import json
import time

import httpx
import pytest

from experiments.general_tool_search.api_client import (
    BudgetExhausted,
    ProviderError,
    ProviderTimeout,
    RealAPIClient,
    RealAPIConfig,
)
from experiments.general_tool_search.engine import GeneralSearchEngine

from .test_general_tool_search import Case, _decision


def _config(**overrides):
    values = {
        "api_key": "test-secret-key",
        "base_url": "https://provider.invalid/v1",
        "model": "test-request-model",
        "request_timeout_seconds": 2.0,
        "decision_json_mode": True,
    }
    values.update(overrides)
    return RealAPIConfig(**values)


def _response(
    content: str,
    *,
    model: str = "test-upstream-model",
    finish_reason: str | None = "stop",
    tool_calls=None,
):
    choice = {
        "message": {"content": content, **({"tool_calls": tool_calls} if tool_calls else {})},
        "finish_reason": finish_reason,
    }
    return httpx.Response(
        200,
        json={
            "model": model,
            "choices": [choice],
        },
    )


def _messages():
    return [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "question"},
    ]


def _run_case(case, *, handler):
    async def run():
        client = RealAPIClient(_config(), transport=httpx.MockTransport(handler))
        try:
            engine = GeneralSearchEngine(
                case.db,
                session_id=case.session_id,
                current_message_id=case.current_id,
                decision_client=client,
                final_client=client,
                database_path=case.engine_db.url.database,
                final_system_prompt="test system",
                decision_timeout=1.0,
                overall_timeout=5.0,
            )
            return await engine.run(), client
        finally:
            await client.aclose()

    result, client = asyncio.run(run())
    case.close()
    return result, client


def test_decision_and_final_requests_are_text_only_and_modeled(tmp_path):
    requests = []

    def handler(request: httpx.Request):
        body = json.loads(request.content)
        requests.append(body)
        if len(requests) == 1:
            return _response(
                _decision(True, "P7 screw", reason="ordinary_recall")
            )
        return _response("I found [message 3].")

    case = Case("process", tmp_path)
    result, client = _run_case(case, handler=handler)

    assert len(requests) == 2
    decision_body, final_body = requests
    assert decision_body["model"] == "test-request-model"
    assert decision_body["response_format"] == {"type": "json_object"}
    assert "decision_window" in decision_body["messages"][1]["content"]
    assert "visible_teacher" in decision_body["messages"][1]["content"]
    assert "tools" not in decision_body
    assert "tool_choice" not in decision_body
    assert final_body["model"] == "test-request-model"
    assert "response_format" not in final_body
    assert "tools" not in final_body
    assert "tool_choice" not in final_body
    assert final_body["messages"][-1]["content"] == case.question
    assert result.synthetic is False
    assert result.counts.decision_http_requests == 1
    assert result.counts.final_http_requests == 1
    assert result.counts.retrieval_attempts == 1
    events = [event.as_dict() for event in client.events]
    assert [event["stage"] for event in events] == ["decision", "final"]
    assert all(event["http_sent"] for event in events)
    assert all(event["request_model"] == "test-request-model" for event in events)
    assert all(event["upstream_model"] == "test-upstream-model" for event in events)
    assert "test-secret-key" not in json.dumps(events)


def test_direct_history_retrieval_sends_only_final_and_keeps_sources(tmp_path):
    requests = []

    def handler(request: httpx.Request):
        requests.append(json.loads(request.content))
        return _response("I found the original name.")

    case = Case("alpha_explicit", tmp_path)
    result, client = _run_case(case, handler=handler)

    assert len(requests) == 1
    assert "tools" not in requests[0]
    assert "tool_choice" not in requests[0]
    assert result.status == "completed_retrieved"
    assert result.counts.decision_attempts == 0
    assert result.counts.decision_http_requests == 0
    assert result.counts.final_http_requests == 1
    assert result.retrieved_source_ids
    assert all(
        event.stage == "final" for event in client.events
    )


def test_visible_complete_decision_skips_retrieval_but_calls_final(tmp_path):
    requests = []

    def handler(request: httpx.Request):
        body = json.loads(request.content)
        requests.append(body)
        if len(requests) == 1:
            return _response(
                json.dumps(
                    {
                        "need_history": False,
                        "query": "",
                        "visible_source_ids": case.visible_ids,
                        "reason": "visible_complete",
                    }
                )
            )
        return _response("The visible teaching is sufficient.")

    case = Case("visible_complete", tmp_path)
    result, _ = _run_case(case, handler=handler)

    assert len(requests) == 2
    assert result.status == "completed_visible_complete"
    assert result.counts.decision_http_requests == 1
    assert result.counts.retrieval_attempts == 0
    assert result.counts.retrieval_http_requests == 0
    assert result.counts.final_http_requests == 1


def test_invalid_decision_json_has_no_retrieval_or_final(tmp_path):
    requests = []

    def handler(request: httpx.Request):
        requests.append(json.loads(request.content))
        return _response("this is not JSON")

    case = Case("foobar", tmp_path)
    result, client = _run_case(case, handler=handler)

    assert len(requests) == 1
    assert result.status == "decision_invalid"
    assert result.counts.decision_http_requests == 1
    assert result.counts.retrieval_attempts == 0
    assert result.counts.final_attempts == 0
    assert result.counts.final_http_requests == 0
    assert [event.stage for event in client.events] == ["decision"]


def test_pre_send_budget_is_zero_http_and_recorded(tmp_path):
    called = []

    def handler(request: httpx.Request):
        called.append(request)
        return _response("{}")

    async def run():
        client = RealAPIClient(_config(), transport=httpx.MockTransport(handler))
        try:
            with pytest.raises(BudgetExhausted):
                await client.generate_decision(
                    _messages(), deadline=time.monotonic() - 0.1
                )
        finally:
            await client.aclose()
        return client

    client = asyncio.run(run())
    assert called == []
    assert len(client.events) == 1
    assert client.events[0].stage == "decision"
    assert client.events[0].http_sent is False
    assert client.events[0].status == "budget_exhausted"


def test_post_send_timeout_keeps_http_evidence_without_exception_text(tmp_path):
    def handler(request: httpx.Request):
        raise httpx.ReadTimeout("do-not-leak-this")

    async def run():
        client = RealAPIClient(_config(), transport=httpx.MockTransport(handler))
        try:
            with pytest.raises(ProviderTimeout):
                await client.generate_final(
                    _messages(), deadline=time.monotonic() + 5
                )
        finally:
            await client.aclose()
        return client

    client = asyncio.run(run())
    event = client.events[0].as_dict()
    assert event["http_sent"] is True
    assert event["status"] == "timeout"
    assert "do-not-leak-this" not in json.dumps(event)


def test_http_error_does_not_expose_body_and_no_retry(tmp_path):
    requests = []

    def handler(request: httpx.Request):
        requests.append(request)
        return httpx.Response(
            500,
            content=b"secret upstream body",
            request=request,
        )

    async def run():
        client = RealAPIClient(_config(), transport=httpx.MockTransport(handler))
        try:
            with pytest.raises(ProviderError):
                await client.generate_final(
                    _messages(), deadline=time.monotonic() + 5
                )
        finally:
            await client.aclose()
        return client

    client = asyncio.run(run())
    assert len(requests) == 1
    event = client.events[0].as_dict()
    assert event["http_sent"] is True
    assert event["status"] == "http_error"
    assert "secret upstream body" not in json.dumps(event)


def test_final_failure_stops_after_one_final_http(tmp_path):
    requests = []

    def handler(request: httpx.Request):
        body = json.loads(request.content)
        requests.append(body)
        if len(requests) == 1:
            return _response(
                _decision(True, "P7 screw", reason="ordinary_recall")
            )
        return httpx.Response(503, content=b"secret final body", request=request)

    case = Case("process", tmp_path)
    result, client = _run_case(case, handler=handler)

    assert len(requests) == 2
    assert result.status == "final_failed"
    assert result.counts.decision_http_requests == 1
    assert result.counts.final_http_requests == 1
    assert result.counts.final_attempts == 1
    assert [event.stage for event in client.events] == ["decision", "final"]
    assert all(
        "secret final body" not in json.dumps(event.as_dict())
        for event in client.events
    )


def test_pre_send_final_budget_is_zero_http():
    called = []

    def handler(request: httpx.Request):
        called.append(request)
        return _response("not called")

    async def run():
        client = RealAPIClient(_config(), transport=httpx.MockTransport(handler))
        try:
            with pytest.raises(BudgetExhausted):
                await client.generate_final(
                    _messages(), deadline=time.monotonic() - 0.1
                )
        finally:
            await client.aclose()
        return client

    client = asyncio.run(run())
    assert called == []
    assert client.events[0].stage == "final"
    assert client.events[0].http_sent is False


def test_missing_upstream_model_is_recorded_as_none():
    def handler(request: httpx.Request):
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"content": "{}"}, "finish_reason": "stop"}
                ]
            },
        )

    async def run():
        client = RealAPIClient(_config(), transport=httpx.MockTransport(handler))
        try:
            await client.generate_decision(
                _messages(), deadline=time.monotonic() + 5
            )
        finally:
            await client.aclose()
        return client

    client = asyncio.run(run())
    assert client.events[0].upstream_model is None


def test_decision_finish_length_is_not_parsed_or_followed(tmp_path):
    requests = []

    def handler(request: httpx.Request):
        requests.append(json.loads(request.content))
        return _response(
            _decision(True, "P7 screw"), finish_reason="length"
        )

    case = Case("process", tmp_path)
    result, client = _run_case(case, handler=handler)

    assert len(requests) == 1
    assert result.status == "decision_failed"
    assert result.error_code == "finish_reason_length"
    assert result.counts.decision_http_requests == 1
    assert result.counts.final_attempts == 0
    assert [event.stage for event in client.events] == ["decision"]
    assert client.events[0].status == "finish_reason_length"


def test_missing_finish_reason_and_tool_calls_are_explicit_failures():
    def missing(request: httpx.Request):
        return _response("{}", finish_reason=None)

    def tool_calls(request: httpx.Request):
        return _response(
            "{}",
            finish_reason="stop",
            tool_calls=[{"id": "x", "type": "function"}],
        )

    async def call(handler):
        client = RealAPIClient(_config(), transport=httpx.MockTransport(handler))
        try:
            with pytest.raises(ProviderError) as error:
                await client.generate_decision(
                    _messages(), deadline=time.monotonic() + 5
                )
        finally:
            await client.aclose()
        return error.value, client

    error, client = asyncio.run(call(missing))
    assert error.status == "finish_reason_missing"
    assert client.events[0].status == "finish_reason_missing"

    error, client = asyncio.run(call(tool_calls))
    assert error.status == "unexpected_tool_calls"
    assert client.events[0].status == "unexpected_tool_calls"


def test_late_response_after_parse_is_rejected():
    def handler(request: httpx.Request):
        time.sleep(0.06)
        return _response("{}")

    async def run():
        client = RealAPIClient(
            _config(request_timeout_seconds=2.0),
            transport=httpx.MockTransport(handler),
        )
        try:
            with pytest.raises(ProviderTimeout):
                await client.generate_decision(
                    _messages(), deadline=time.monotonic() + 0.01
                )
        finally:
            await client.aclose()
        return client

    client = asyncio.run(run())
    assert client.events[0].http_sent is True
    assert client.events[0].status in {"late_after_response", "late_after_parse"}


def test_final_finish_length_is_not_a_success(tmp_path):
    requests = []

    def handler(request: httpx.Request):
        requests.append(json.loads(request.content))
        if len(requests) == 1:
            return _response(_decision(True, "P7 screw"))
        return _response("partial", finish_reason="length")

    case = Case("process", tmp_path)
    result, client = _run_case(case, handler=handler)

    assert len(requests) == 2
    assert result.status == "final_failed"
    assert result.error_code == "finish_reason_length"
    assert result.counts.final_http_requests == 1
    assert client.events[-1].status == "finish_reason_length"


def test_synthetic_evidence_export_keeps_sources_and_flags(tmp_path):
    from experiments.general_tool_search.runner import write_evidence
    from experiments.general_tool_search.types import EngineResult, SearchOutcome

    result = EngineResult(
        status="completed_retrieved",
        answer="I found [message 3].",
        injected_source_ids=[3],
        cited_source_ids=[3],
        search=SearchOutcome(
            status="ok",
            results=[
                {
                    "message_id": 3,
                    "order": 1,
                    "content": "The alpha bridge is blue.",
                    "kind": "teaching_match",
                    "truncated": False,
                }
            ],
            truncated=True,
            scan_truncated=False,
            corrections_truncated=True,
            omitted_count=1,
            pair_missing=True,
        ),
    )
    path = write_evidence(
        tmp_path / "evidence", case_name="synthetic_case", result=result
    )
    data = json.loads(path.read_text())
    assert data["case"] == "synthetic_case"
    assert data["synthetic"] is True
    assert data["final_reply"] == "I found [message 3]."
    assert data["injected_source_ids"] == [3]
    assert data["retrieved_sources"][0]["message_id"] == 3
    assert data["retrieved_sources"][0]["content"] == "The alpha bridge is blue."
    assert data["search_limitations"]["truncated"] is True
    assert data["search_limitations"]["corrections_truncated"] is True
    assert data["search_limitations"]["pair_missing"] is True
    assert data["search_limitations"]["omitted_count"] == 1
    assert "messages" not in data
    assert "request" not in json.dumps(data).lower()
    assert "api_key" not in json.dumps(data).lower()


def test_evidence_export_rejects_non_synthetic_results(tmp_path):
    from experiments.general_tool_search.runner import write_evidence
    from experiments.general_tool_search.types import EngineResult

    result = EngineResult(status="completed_direct", synthetic=False)
    with pytest.raises(ValueError, match="synthetic-only"):
        write_evidence(tmp_path / "evidence", case_name="real", result=result)


def test_default_and_injected_transports_share_tracking_boundary():
    async def run():
        default = RealAPIClient(_config())
        injected = RealAPIClient(
            _config(),
            transport=httpx.MockTransport(lambda request: _response("{}")),
        )
        try:
            assert default._client._transport is default._tracked_transport
            assert injected._client._transport is injected._tracked_transport
        finally:
            await default.aclose()
            await injected.aclose()

    asyncio.run(run())


def test_request_preparation_failure_never_counts_as_transport(monkeypatch):
    called = []

    def handler(request: httpx.Request):
        called.append(request)
        return _response("{}")

    async def run():
        client = RealAPIClient(
            _config(), transport=httpx.MockTransport(handler)
        )
        monkeypatch.setattr(
            client._client,
            "build_request",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                httpx.InvalidURL("preparation failed")
            ),
        )
        try:
            with pytest.raises(ProviderError):
                await client.generate_final(
                    _messages(), deadline=time.monotonic() + 5
                )
        finally:
            await client.aclose()
        return client

    client = asyncio.run(run())
    assert called == []
    event = client.events[0].as_dict()
    assert event["http_sent"] is False
    assert event["status"] == "prepare_error"
    assert "preparation failed" not in json.dumps(event)


def test_cancellation_after_transport_entry_counts_one_request_and_is_not_retried():
    entered = asyncio.Event()

    class BlockingTransport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request: httpx.Request):
            entered.set()
            await asyncio.Event().wait()

    async def run():
        client = RealAPIClient(_config(), transport=BlockingTransport())
        task = asyncio.create_task(
            client.generate_final(
                _messages(), deadline=time.monotonic() + 10
            )
        )
        await asyncio.wait_for(entered.wait(), 1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        try:
            events = [event.as_dict() for event in client.events]
        finally:
            await client.aclose()
        return events

    events = asyncio.run(run())
    assert len(events) == 1
    assert events[0]["http_sent"] is True
    assert events[0]["status"] == "cancelled"
