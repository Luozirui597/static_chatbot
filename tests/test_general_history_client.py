"""Transport-contract tests for the production general-history adapter."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from backend.exceptions import LLMError, LLMInvalidResponseError
from backend.general_history.client import ProductionGeneralClient
from backend.llm_client import OpenAICompatibleLLMClient
from backend.review_diagnostics import LLMCallDiagnostics


def _run(coro):
    return asyncio.run(coro)


def _api_client(transport: httpx.AsyncBaseTransport) -> OpenAICompatibleLLMClient:
    return OpenAICompatibleLLMClient(
        api_key="test-key",
        base_url="https://example.test/v1",
        model="model-x",
        transport=transport,
    )


def _response(content: str, finish_reason: str = "stop") -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "choices": [
                {
                    "message": {"content": content},
                    "finish_reason": finish_reason,
                }
            ]
        },
    )



def test_build_request_failure_does_not_enter_transport():
    class CountingTransport(httpx.AsyncBaseTransport):
        def __init__(self):
            self.calls = 0

        async def handle_async_request(self, request):
            self.calls += 1
            return _response("unexpected")

        async def aclose(self):
            return None

    transport = CountingTransport()
    client = OpenAICompatibleLLMClient(
        api_key="test-key",
        base_url="https://example.test/v1",
        model="model-x",
        transport=transport,
    )
    diagnostics = LLMCallDiagnostics()
    with pytest.raises(Exception):
        _run(client.generate(
            [{"role": "user", "content": object()}],
            diagnostics=diagnostics,
        ))
    assert transport.calls == 0
    assert diagnostics.transport_started is False


def test_transport_entry_failure_is_counted_exactly_once():
    class FailingTransport(httpx.AsyncBaseTransport):
        def __init__(self):
            self.calls = 0

        async def handle_async_request(self, request):
            self.calls += 1
            raise httpx.ConnectError(
                "mock transport failure",
                request=request,
            )

        async def aclose(self):
            return None

    transport = FailingTransport()
    client = OpenAICompatibleLLMClient(
        api_key="test-key",
        base_url="https://example.test/v1",
        model="model-x",
        transport=transport,
    )
    diagnostics = LLMCallDiagnostics()
    with pytest.raises(LLMError):
        _run(client.generate(
            [{"role": "user", "content": "question"}],
            diagnostics=diagnostics,
        ))
    assert transport.calls == 1
    assert diagnostics.transport_started is True

def test_normal_decision_records_transport_entry_and_stop():
    def handler(request):
        return _response(json.dumps({
            "need_history": False,
            "query": "",
            "visible_source_ids": [],
            "reason": "not_needed",
        }))

    wrapped = ProductionGeneralClient(
        _api_client(httpx.MockTransport(handler)),
        model="model-x",
    )
    result = _run(
        wrapped.generate_decision(
            [{"role": "user", "content": "question"}],
            deadline=10.0,
        )
    )
    assert "need_history" in result
    assert wrapped.events[-1].stage == "decision"
    assert wrapped.events[-1].http_sent is True
    assert wrapped.events[-1].status == "completed"


def test_truncated_decision_is_rejected_even_when_json_parses():
    payload = json.dumps({
        "need_history": True,
        "query": "Foobar",
        "visible_source_ids": [],
        "reason": "ordinary_recall",
    })
    wrapped = ProductionGeneralClient(
        _api_client(httpx.MockTransport(lambda request: _response(payload, "length"))),
        model="model-x",
    )
    with pytest.raises(LLMInvalidResponseError) as raised:
        _run(wrapped.generate_decision(
            [{"role": "user", "content": "question"}],
            deadline=10.0,
        ))
    assert raised.value.status_code == 502
    assert wrapped.events[-1].http_sent is True


def test_pre_send_failure_is_not_counted_as_sent():
    class LocalFailure:
        async def generate(self, messages, *, diagnostics=None, **kwargs):
            raise ValueError("invalid local option")

    wrapped = ProductionGeneralClient(LocalFailure(), model="model-x")
    with pytest.raises(ValueError):
        _run(wrapped.generate_decision(
            [{"role": "user", "content": "question"}],
            deadline=10.0,
        ))
    assert wrapped.events[-1].http_sent is False
    assert wrapped.events[-1].status == "error"


def test_post_send_transport_failure_is_counted_as_sent():
    def handler(request):
        raise httpx.ConnectError(
            "connection refused",
            request=httpx.Request("POST", "https://example.test/v1/chat/completions"),
        )

    wrapped = ProductionGeneralClient(
        _api_client(httpx.MockTransport(handler)),
        model="model-x",
    )
    with pytest.raises(LLMError):
        _run(wrapped.generate_decision(
            [{"role": "user", "content": "question"}],
            deadline=10.0,
        ))
    assert wrapped.events[-1].http_sent is True
    assert wrapped.events[-1].status == "error"


def test_unobservable_transport_is_reported_as_unknown():
    class NoObservation:
        async def generate(self, messages, *, diagnostics=None, **kwargs):
            if diagnostics is not None:
                diagnostics.finish_reason = "stop"
            return "answer"

    wrapped = ProductionGeneralClient(NoObservation(), model="model-x")
    assert _run(wrapped.generate_final(
        [{"role": "user", "content": "question"}]
    )) == "answer"
    assert wrapped.events[-1].http_sent is None


def test_truncated_final_is_rejected():
    wrapped = ProductionGeneralClient(
        _api_client(httpx.MockTransport(
            lambda request: _response("partial answer", "length")
        )),
        model="model-x",
    )
    with pytest.raises(LLMInvalidResponseError):
        _run(wrapped.generate_final([{"role": "user", "content": "question"}]))
    assert wrapped.events[-1].http_sent is True
