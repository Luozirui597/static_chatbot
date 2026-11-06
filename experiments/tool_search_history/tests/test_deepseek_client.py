"""MockTransport tests for the experimental DeepSeek adapter."""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import httpx
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.database import create_tables
from experiments.tool_search_history.deepseek_client import (
    DEFAULT_MODEL,
    DeepSeekBudgetExhausted,
    DeepSeekHTTPError,
    DeepSeekProtocolError,
    DeepSeekTimeout,
    DeepSeekToolClient,
)
from experiments.tool_search_history.protocol import (
    TOOL_NAME,
    TOOL_SCHEMA,
    assistant_message,
    tool_message,
)
from experiments.tool_search_history.runner import Fixture, run_condition
from experiments.tool_search_history.tool import ToolExecutor
from experiments.tool_search_history.types import AssistantTurn, ToolCall, ToolResult

ASSET_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PATH = ASSET_ROOT / "fixtures" / "window_outsider.json"


def _tool_response(
    arguments: str = '{"query": "A C correction", "limit": 3}',
    *,
    call_id: str = "call_1",
    finish_reason: str = "tool_calls",
    content=None,
) -> dict:
    return {
        "id": "chatcmpl-first",
        "model": "deepseek-flash",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": TOOL_NAME,
                                "arguments": arguments,
                            },
                        }
                    ],
                },
                "finish_reason": finish_reason,
            }
        ],
    }


def _text_response(text: str, *, finish_reason: str = "stop") -> dict:
    return {
        "id": "chatcmpl-final",
        "model": "deepseek-flash",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": finish_reason,
            }
        ],
    }


def _client(handler, **kwargs):
    return DeepSeekToolClient(
        api_key="test-key-not-real",
        transport=httpx.MockTransport(handler),
        **kwargs,
    )


def _deadline(seconds: float = 60.0) -> float:
    return time.monotonic() + seconds


def test_first_request_body_and_direct_answer():
    seen = []

    def handler(request: httpx.Request):
        seen.append(json.loads(request.content))
        return httpx.Response(200, json=_text_response("Direct answer."))

    client = _client(handler)
    turn = asyncio.run(
        client.generate_turn(
            [{"role": "user", "content": "What is A?"}],
            tools=[TOOL_SCHEMA],
            phase="first",
            tool_choice="auto",
            deadline=_deadline(),
        )
    )
    assert turn.content == "Direct answer."
    assert turn.tool_calls == ()
    assert len(seen) == 1
    body = seen[0]
    assert body["model"] == DEFAULT_MODEL
    assert body["stream"] is False
    assert body["thinking"] == {"type": "disabled"}
    assert body["temperature"] == 0.0
    assert body["max_tokens"] == 800
    assert body["tool_choice"] == "auto"
    assert body["tools"] == [TOOL_SCHEMA]
    record = client.take_last_call()
    assert record["request_sent"] is True
    assert record["phase"] == "first"
    assert record["response_model"] == "deepseek-flash"
    assert record["finish_reason"] == "stop"


def test_first_request_parses_tool_call_and_preserves_raw_arguments():
    raw = '{ "limit": 3, "query": "A C correction" }'

    def handler(request: httpx.Request):
        return httpx.Response(
            200,
            json=_tool_response(raw),
        )

    client = _client(handler)
    turn = asyncio.run(
        client.generate_turn(
            [{"role": "user", "content": "What did I teach?"}],
            tools=[TOOL_SCHEMA],
            phase="first",
            tool_choice="auto",
            deadline=_deadline(),
        )
    )
    call = turn.tool_calls[0]
    assert call.id == "call_1"
    assert call.name == TOOL_NAME
    assert call.raw_arguments == raw
    assert call.arguments == {"limit": 3, "query": "A C correction"}
    record = client.take_last_call()
    assert record["tool_call_id"] == "call_1"
    assert record["raw_arguments"] == raw
    assert record["parsed_arguments"] == {"limit": 3, "query": "A C correction"}


def test_second_request_appends_exact_assistant_and_tool_pair():
    seen = []
    first_turn = AssistantTurn(
        content=None,
        tool_calls=(
            ToolCall(
                id="call_1",
                name=TOOL_NAME,
                arguments={"limit": 3, "query": "A C correction"},
                raw_arguments='{ "limit": 3, "query": "A C correction" }',
            ),
        ),
    )
    result = ToolResult(
        tool_call_id="call_1",
        name=TOOL_NAME,
        content={"status": "ok", "results": [{"message_id": 21}]},
    )

    def handler(request: httpx.Request):
        seen.append(json.loads(request.content))
        return httpx.Response(200, json=_text_response("Final [message 21]."))

    client = _client(handler)
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "question"},
        assistant_message(first_turn),
        tool_message(result),
    ]
    turn = asyncio.run(
        client.generate_turn(
            messages,
            tools=[TOOL_SCHEMA],
            phase="second",
            tool_choice="none",
            deadline=_deadline(),
        )
    )
    assert turn.content == "Final [message 21]."
    body = seen[0]
    assert body["tool_choice"] == "none"
    assert body["tools"] == [TOOL_SCHEMA]
    assert body["messages"] == messages
    assert body["messages"][2]["tool_calls"][0]["id"] == "call_1"
    assert body["messages"][3]["tool_call_id"] == "call_1"
    assert (
        body["messages"][2]["tool_calls"][0]["function"]["arguments"]
        == '{ "limit": 3, "query": "A C correction" }'
    )


def test_missing_call_id_is_one_request_and_no_tool():
    seen = []

    def handler(request: httpx.Request):
        seen.append(json.loads(request.content))
        body = _tool_response()
        body["choices"][0]["message"]["tool_calls"][0]["id"] = ""
        return httpx.Response(200, json=body)

    client = _client(handler)
    with pytest.raises(DeepSeekProtocolError):
        asyncio.run(
            client.generate_turn(
                [{"role": "user", "content": "question"}],
                tools=[TOOL_SCHEMA],
                phase="first",
                tool_choice="auto",
                deadline=_deadline(),
            )
        )
    assert len(seen) == 1
    record = client.take_last_call()
    assert record["request_sent"] is True
    assert record["tool_call_id"] is None
    assert record["error_stage"] == "parse"


def test_second_response_tool_call_is_rejected():
    def handler(request: httpx.Request):
        return httpx.Response(200, json=_tool_response())

    client = _client(handler)
    with pytest.raises(DeepSeekProtocolError, match="second_request_returned_tool"):
        asyncio.run(
            client.generate_turn(
                [{"role": "user", "content": "question"}],
                tools=[TOOL_SCHEMA],
                phase="second",
                tool_choice="none",
                deadline=_deadline(),
            )
        )
    record = client.take_last_call()
    assert record["request_sent"] is True
    assert record["http_status"] == 200


def test_http_error_records_request_without_secret():
    def handler(request: httpx.Request):
        return httpx.Response(500, text="do not leak this upstream body")

    client = _client(handler)
    with pytest.raises(DeepSeekHTTPError):
        asyncio.run(
            client.generate_turn(
                [{"role": "user", "content": "question"}],
                tools=[TOOL_SCHEMA],
                phase="first",
                tool_choice="auto",
                deadline=_deadline(),
            )
        )
    record = client.take_last_call()
    assert record["request_sent"] is True
    assert record["http_status"] == 500
    assert "Authorization" not in json.dumps(record)
    assert "test-key-not-real" not in json.dumps(record)
    assert "do not leak this upstream body" not in json.dumps(record)


def test_request_timeout_is_recorded_after_send():
    class SlowTransport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request: httpx.Request):
            await asyncio.sleep(0.05)
            return httpx.Response(200, json=_text_response("too late"))

    client = DeepSeekToolClient(
        api_key="test-key-not-real",
        transport=SlowTransport(),
        timeout=0.01,
    )
    with pytest.raises(DeepSeekTimeout):
        asyncio.run(
            client.generate_turn(
                [{"role": "user", "content": "question"}],
                tools=[TOOL_SCHEMA],
                phase="first",
                tool_choice="auto",
                deadline=_deadline(),
            )
        )
    record = client.take_last_call()
    assert record["request_sent"] is True
    assert record["error_stage"] == "timeout"


def test_budget_exhausted_before_request_sends_nothing():
    seen = []

    def handler(request: httpx.Request):
        seen.append(request)
        return httpx.Response(200, json=_text_response("unexpected"))

    client = _client(handler)
    with pytest.raises(DeepSeekBudgetExhausted):
        asyncio.run(
            client.generate_turn(
                [{"role": "user", "content": "question"}],
                tools=[TOOL_SCHEMA],
                phase="first",
                tool_choice="auto",
                deadline=time.monotonic() - 1,
            )
        )
    assert seen == []
    record = client.take_last_call()
    assert record["request_sent"] is False
    assert record["error_stage"] == "budget"


def test_effective_timeout_uses_remaining_budget():
    client = DeepSeekToolClient(
        api_key="test-key-not-real",
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json=_text_response("ok"))
        ),
        clock=lambda: 100.0,
    )
    # deadline 112 seconds away -> effective timeout 12 seconds
    turn = asyncio.run(
        client.generate_turn(
            [{"role": "user", "content": "question"}],
            tools=[TOOL_SCHEMA],
            phase="first",
            tool_choice="auto",
            deadline=112.0,
        )
    )
    assert turn.content == "ok"
    record = client.take_last_call()
    assert record["effective_timeout_s"] == pytest.approx(12.0)


def test_full_fixture_tool_loop_uses_real_sqlite_and_mock_http(tmp_path):
    fixture = Fixture.from_file(FIXTURE_PATH)
    engine = create_engine(f"sqlite:///{tmp_path / 'fixture.sqlite3'}")
    create_tables(engine)
    factory = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    seen = []
    responses = iter(
        [
            _tool_response(),
            _text_response(
                "You taught A-B [message 3] and B-C [message 5]. "
                "You corrected B-C [message 21], so I cannot confirm A-C."
            ),
        ]
    )

    def handler(request: httpx.Request):
        seen.append(json.loads(request.content))
        return httpx.Response(200, json=next(responses))

    client = _client(handler)
    result = asyncio.run(
        run_condition(
            session_factory=factory,
            database_path=tmp_path / "fixture.sqlite3",
            asset_root=ASSET_ROOT,
            condition="tool",
            fixture=fixture,
            client=client,
            tool_failure_policy="stop",
        )
    )
    assert result.status == "completed"
    assert result.protocol_loop is True
    assert result.model_calls == 2
    assert result.tool_calls == 1
    tool_payload = json.loads(result.tool_exchanges[0]["tool"]["content"])
    assert [item["message_id"] for item in tool_payload["results"]] == [3, 5, 21]
    assert result.answer is not None
    assert result.source_ids == [3, 5, 21]
    assert [item["order"] for item in tool_payload["results"]] == [2, 3, 11]
    correction = next(
        item for item in tool_payload["results"]
        if item["message_id"] == 21
    )
    assert correction["kind"] == "correction_candidate"
    assert correction["support_message_id"] == 5
    assert tool_payload["omitted_count"] == 1
    assert len(seen) == 2
    assert seen[1]["tool_choice"] == "none"
    assert seen[1]["messages"][-2]["role"] == "assistant"
    assert seen[1]["messages"][-1]["role"] == "tool"
    assert (
        seen[1]["messages"][-2]["tool_calls"][0]["id"]
        == seen[1]["messages"][-1]["tool_call_id"]
    )
    engine.dispose()


def test_runner_missing_call_id_counts_one_request_no_tool(tmp_path):
    fixture = Fixture.from_file(FIXTURE_PATH)
    engine = create_engine(f"sqlite:///{tmp_path / 'missing-id.sqlite3'}")
    create_tables(engine)
    factory = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    seen = []

    def handler(request: httpx.Request):
        seen.append(json.loads(request.content))
        body = _tool_response()
        body["choices"][0]["message"]["tool_calls"][0]["id"] = ""
        return httpx.Response(200, json=body)

    result = asyncio.run(
        run_condition(
            session_factory=factory,
            database_path=tmp_path / "missing-id.sqlite3",
            asset_root=ASSET_ROOT,
            condition="tool",
            fixture=fixture,
            client=_client(handler),
            tool_failure_policy="stop",
        )
    )
    assert result.status == "failed"
    assert result.model_calls == 1
    assert result.tool_calls == 0
    assert len(seen) == 1
    assert len(result.model_attempts) == 1
    assert result.model_attempts[0]["request_sent"] is True
    engine.dispose()


def test_real_adapter_tool_timeout_stops_before_second(tmp_path):
    fixture = Fixture.from_file(FIXTURE_PATH)
    engine = create_engine(f"sqlite:///{tmp_path / 'tool-timeout.sqlite3'}")
    create_tables(engine)
    factory = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    seen = []

    def handler(request: httpx.Request):
        seen.append(json.loads(request.content))
        return httpx.Response(200, json=_tool_response())

    class TimeoutExecutor:
        async def execute(self, call, context, **kwargs):
            return ToolResult(
                tool_call_id=call.id,
                name=call.name,
                content={
                    "status": "timeout",
                    "error_code": "tool_deadline_exceeded",
                    "results": [],
                },
                is_error=True,
            )

    result = asyncio.run(
        run_condition(
            session_factory=factory,
            database_path=tmp_path / "tool-timeout.sqlite3",
            asset_root=ASSET_ROOT,
            condition="tool",
            fixture=fixture,
            client=_client(handler),
            executor=TimeoutExecutor(),
            tool_failure_policy="stop",
        )
    )
    assert result.status == "failed"
    assert result.error_stage == "tool"
    assert result.model_calls == 1
    assert result.tool_calls == 1
    assert len(seen) == 1
    engine.dispose()


def test_second_sent_then_timeout_counts_two_requests(tmp_path):
    fixture = Fixture.from_file(FIXTURE_PATH)
    engine = create_engine(f"sqlite:///{tmp_path / 'second-timeout.sqlite3'}")
    create_tables(engine)
    factory = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    seen = []

    class SecondSlowTransport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):
            seen.append(json.loads(request.content))
            if len(seen) == 1:
                return httpx.Response(200, json=_tool_response())
            await asyncio.sleep(0.3)
            return httpx.Response(200, json=_text_response("too late"))

    result = asyncio.run(
        run_condition(
            session_factory=factory,
            database_path=tmp_path / "second-timeout.sqlite3",
            asset_root=ASSET_ROOT,
            condition="tool",
            fixture=fixture,
            client=DeepSeekToolClient(
                api_key="test-key-not-real",
                transport=SecondSlowTransport(),
                timeout=30.0,
            ),
            tool_failure_policy="stop",
            deadline_seconds=0.2,
        )
    )
    assert result.status == "failed"
    assert result.model_calls == 2
    assert result.tool_calls == 1
    assert len(seen) == 2
    assert result.model_attempts[-1]["error_stage"] == "timeout"
    engine.dispose()


def test_budget_exhausted_before_second_counts_one_request(tmp_path):
    fixture = Fixture.from_file(FIXTURE_PATH)
    engine = create_engine(f"sqlite:///{tmp_path / 'budget.sqlite3'}")
    create_tables(engine)
    factory = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    seen = []

    class SlowExecutor:
        async def execute(self, call, context, **kwargs):
            await asyncio.sleep(0.08)
            return ToolResult(
                tool_call_id=call.id,
                name=call.name,
                content={"status": "ok", "results": []},
            )

    def handler(request: httpx.Request):
        seen.append(json.loads(request.content))
        return httpx.Response(200, json=_tool_response())

    result = asyncio.run(
        run_condition(
            session_factory=factory,
            database_path=tmp_path / "budget.sqlite3",
            asset_root=ASSET_ROOT,
            condition="tool",
            fixture=fixture,
            client=_client(handler),
            executor=SlowExecutor(),
            tool_failure_policy="stop",
            deadline_seconds=0.05,
        )
    )
    assert result.status == "failed"
    assert result.model_calls == 1
    assert result.tool_calls == 1
    assert len(seen) == 1
    assert result.model_attempts[-1]["request_sent"] is False
    assert result.model_attempts[-1]["error_stage"] == "budget"
    engine.dispose()


def test_invalid_citation_is_recorded_separately(tmp_path):
    fixture = Fixture.from_file(FIXTURE_PATH)
    engine = create_engine(f"sqlite:///{tmp_path / 'citation.sqlite3'}")
    create_tables(engine)
    factory = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    responses = iter(
        [
            _tool_response(),
            _text_response("I rely on [message 999]."),
        ]
    )
    seen = []

    def handler(request: httpx.Request):
        seen.append(json.loads(request.content))
        return httpx.Response(200, json=next(responses))

    result = asyncio.run(
        run_condition(
            session_factory=factory,
            database_path=tmp_path / "citation.sqlite3",
            asset_root=ASSET_ROOT,
            condition="tool",
            fixture=fixture,
            client=_client(handler),
            tool_failure_policy="stop",
        )
    )
    assert result.status == "completed"
    assert result.source_ids == [999]
    assert result.visible_source_ids == [3, 5, 21]
    assert result.invalid_source_ids == [999]
    assert result.answer_quality == "invalid_citations"
    assert len(seen) == 2
    engine.dispose()


@pytest.mark.parametrize(
    "response_factory",
    [
        lambda: httpx.Response(200, text="not json"),
        lambda: httpx.Response(200, json={"choices": []}),
        lambda: httpx.Response(200, json=_text_response("", finish_reason="length")),
        lambda: httpx.Response(
            200,
            json=_text_response("partial", finish_reason="content_filter"),
        ),
    ],
)
def test_adapter_rejects_non_final_shapes(response_factory):
    client = _client(lambda request: response_factory())
    with pytest.raises(DeepSeekProtocolError):
        asyncio.run(
            client.generate_turn(
                [{"role": "user", "content": "question"}],
                tools=[TOOL_SCHEMA],
                phase="first",
                tool_choice="auto",
                deadline=_deadline(),
            )
        )
    record = client.take_last_call()
    assert record["request_sent"] is True
    assert record["http_status"] == 200
    assert record["error_stage"] == "parse"


def test_second_response_tool_call_after_runner_counts_two_requests(tmp_path):
    fixture = Fixture.from_file(FIXTURE_PATH)
    engine = create_engine(f"sqlite:///{tmp_path / 'second-tool.sqlite3'}")
    create_tables(engine)
    factory = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    seen = []
    responses = iter([_tool_response(), _tool_response(call_id="call_2")])

    def handler(request: httpx.Request):
        seen.append(json.loads(request.content))
        return httpx.Response(200, json=next(responses))

    result = asyncio.run(
        run_condition(
            session_factory=factory,
            database_path=tmp_path / "second-tool.sqlite3",
            asset_root=ASSET_ROOT,
            condition="tool",
            fixture=fixture,
            client=_client(handler),
            tool_failure_policy="stop",
        )
    )
    assert result.status == "failed"
    assert result.model_calls == 2
    assert result.tool_calls == 1
    assert len(seen) == 2
    assert result.error_stage == "parse"
    assert result.error_code == "second_request_returned_tool"
    engine.dispose()


def test_full_fixture_attempt_evidence_and_tool_schema_stability(tmp_path):
    fixture = Fixture.from_file(FIXTURE_PATH)
    engine = create_engine(f"sqlite:///{tmp_path / 'attempts.sqlite3'}")
    create_tables(engine)
    factory = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    seen = []
    responses = iter(
        [
            _tool_response(),
            _text_response("A [message 3], B [message 5], C [message 21]."),
        ]
    )

    def handler(request: httpx.Request):
        seen.append(json.loads(request.content))
        return httpx.Response(200, json=next(responses))

    result = asyncio.run(
        run_condition(
            session_factory=factory,
            database_path=tmp_path / "attempts.sqlite3",
            asset_root=ASSET_ROOT,
            condition="tool",
            fixture=fixture,
            client=_client(handler),
            tool_failure_policy="stop",
        )
    )
    assert result.status == "completed"
    assert [record["phase"] for record in result.model_attempts] == [
        "first",
        "second",
    ]
    assert all(record["request_sent"] for record in result.model_attempts)
    assert result.model_attempts[0]["tool_choice"] == "auto"
    assert result.model_attempts[1]["tool_choice"] == "none"
    assert seen[0]["tools"] == seen[1]["tools"]
    assert result.visible_source_ids == [3, 5, 21]
    assert result.invalid_source_ids == []
    assert result.answer_quality == "citations_valid"
    # Citation validity is deliberately separate from semantic review.
    assert result.answer_used_correction is None
    engine.dispose()


def test_valid_sources_without_final_citations_is_missing_citations(tmp_path):
    fixture = Fixture.from_file(FIXTURE_PATH)
    engine = create_engine(f"sqlite:///{tmp_path / 'missing-citations.sqlite3'}")
    create_tables(engine)
    factory = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    seen = []
    responses = iter(
        [
            _tool_response(),
            _text_response("The evidence is summarized above."),
        ]
    )

    def handler(request: httpx.Request):
        seen.append(json.loads(request.content))
        return httpx.Response(200, json=next(responses))

    result = asyncio.run(
        run_condition(
            session_factory=factory,
            database_path=tmp_path / "missing-citations.sqlite3",
            asset_root=ASSET_ROOT,
            condition="tool",
            fixture=fixture,
            client=_client(handler),
            tool_failure_policy="stop",
        )
    )
    assert result.status == "completed"
    assert result.protocol_loop is True
    assert result.model_calls == 2
    assert result.tool_calls == 1
    assert result.visible_source_ids == [3, 5, 21]
    assert result.source_ids == []
    assert result.invalid_source_ids == []
    assert result.answer_quality == "missing_citations"
    engine.dispose()


def test_empty_tool_evidence_does_not_require_citations(tmp_path):
    fixture = Fixture.from_file(FIXTURE_PATH)
    engine = create_engine(f"sqlite:///{tmp_path / 'empty-sources.sqlite3'}")
    create_tables(engine)
    factory = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    responses = iter(
        [
            _tool_response(
                '{"query": "zzzzqqq", "limit": 3}'
            ),
            _text_response(
                "The retrieved teaching evidence is empty; I cannot confirm it."
            ),
        ]
    )

    def handler(request: httpx.Request):
        return httpx.Response(200, json=next(responses))

    result = asyncio.run(
        run_condition(
            session_factory=factory,
            database_path=tmp_path / "empty-sources.sqlite3",
            asset_root=ASSET_ROOT,
            condition="tool",
            fixture=fixture,
            client=_client(handler),
            tool_failure_policy="stop",
        )
    )
    assert result.status == "completed"
    assert result.protocol_loop is True
    assert result.visible_source_ids == []
    assert result.source_ids == []
    assert result.answer_quality == "no_citations_available"
    engine.dispose()


class _MutableClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def test_json_parse_crossing_call_deadline_is_rejected(monkeypatch):
    clock = _MutableClock()
    original_json = httpx.Response.json

    def slow_json(self):
        value = original_json(self)
        clock.now = 31.0
        return value

    monkeypatch.setattr(httpx.Response, "json", slow_json)
    client = DeepSeekToolClient(
        api_key="test-key-not-real",
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json=_text_response("late"))
        ),
        clock=clock,
    )
    with pytest.raises(DeepSeekTimeout, match="deadline_after_json_parse"):
        asyncio.run(
            client.generate_turn(
                [{"role": "user", "content": "question"}],
                tools=[TOOL_SCHEMA],
                phase="first",
                tool_choice="auto",
                deadline=100.0,
            )
        )
    record = client.take_last_call()
    assert record["request_sent"] is True
    assert record["effective_timeout_s"] == pytest.approx(30.0)
    assert record["error_stage"] == "timeout"
    assert record["error_code"] == "deadline_after_json_parse"


def test_remaining_turn_budget_limits_parse_deadline(monkeypatch):
    clock = _MutableClock()
    original_json = httpx.Response.json

    def slow_json(self):
        value = original_json(self)
        clock.now = 11.0
        return value

    monkeypatch.setattr(httpx.Response, "json", slow_json)
    client = DeepSeekToolClient(
        api_key="test-key-not-real",
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json=_text_response("late"))
        ),
        clock=clock,
    )
    with pytest.raises(DeepSeekTimeout, match="deadline_after_json_parse"):
        asyncio.run(
            client.generate_turn(
                [{"role": "user", "content": "question"}],
                tools=[TOOL_SCHEMA],
                phase="first",
                tool_choice="auto",
                deadline=10.0,
            )
        )
    record = client.take_last_call()
    assert record["effective_timeout_s"] == pytest.approx(10.0)
    assert record["error_stage"] == "timeout"


def test_structure_parse_crossing_call_deadline_is_rejected(monkeypatch):
    clock = _MutableClock()
    client = DeepSeekToolClient(
        api_key="test-key-not-real",
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json=_text_response("late"))
        ),
        clock=clock,
    )
    original_parse = client._parse_body

    def slow_parse(body, attempt, *, phase):
        turn = original_parse(body, attempt, phase=phase)
        clock.now = 31.0
        return turn

    monkeypatch.setattr(client, "_parse_body", slow_parse)
    with pytest.raises(DeepSeekTimeout, match="deadline_during_structure_parse"):
        asyncio.run(
            client.generate_turn(
                [{"role": "user", "content": "question"}],
                tools=[TOOL_SCHEMA],
                phase="first",
                tool_choice="auto",
                deadline=100.0,
            )
        )
    record = client.take_last_call()
    assert record["request_sent"] is True
    assert record["error_stage"] == "timeout"
    assert record["error_code"] == "deadline_during_structure_parse"


def test_first_tool_parse_timeout_has_one_request_zero_tools(tmp_path):
    fixture = Fixture.from_file(FIXTURE_PATH)
    engine = create_engine(f"sqlite:///{tmp_path / 'first-parse-late.sqlite3'}")
    create_tables(engine)
    factory = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    clock = _MutableClock()
    seen = []

    def handler(request: httpx.Request):
        seen.append(json.loads(request.content))
        return httpx.Response(200, json=_tool_response())

    client = DeepSeekToolClient(
        api_key="test-key-not-real",
        transport=httpx.MockTransport(handler),
        clock=clock,
    )
    original_parse = client._parse_body

    def slow_parse(body, attempt, *, phase):
        turn = original_parse(body, attempt, phase=phase)
        clock.now = 31.0
        return turn

    client._parse_body = slow_parse  # type: ignore[method-assign]
    result = asyncio.run(
        run_condition(
            session_factory=factory,
            database_path=tmp_path / "first-parse-late.sqlite3",
            asset_root=ASSET_ROOT,
            condition="tool",
            fixture=fixture,
            client=client,
            tool_failure_policy="stop",
        )
    )
    assert result.status == "failed"
    assert result.model_calls == 1
    assert result.tool_calls == 0
    assert len(seen) == 1
    assert result.model_attempts[-1]["error_stage"] == "timeout"
    engine.dispose()


def test_second_parse_timeout_has_two_requests_one_tool(tmp_path):
    fixture = Fixture.from_file(FIXTURE_PATH)
    engine = create_engine(f"sqlite:///{tmp_path / 'second-parse-late.sqlite3'}")
    create_tables(engine)
    factory = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    clock = _MutableClock()
    seen = []
    responses = iter(
        [
            _tool_response(),
            _text_response("A [message 3], B [message 5], C [message 21]."),
        ]
    )

    def handler(request: httpx.Request):
        seen.append(json.loads(request.content))
        return httpx.Response(200, json=next(responses))

    client = DeepSeekToolClient(
        api_key="test-key-not-real",
        transport=httpx.MockTransport(handler),
        clock=clock,
    )
    original_parse = client._parse_body
    parse_count = {"value": 0}

    def slow_second_parse(body, attempt, *, phase):
        turn = original_parse(body, attempt, phase=phase)
        parse_count["value"] += 1
        if parse_count["value"] == 2:
            clock.now = 31.0
        return turn

    client._parse_body = slow_second_parse  # type: ignore[method-assign]
    result = asyncio.run(
        run_condition(
            session_factory=factory,
            database_path=tmp_path / "second-parse-late.sqlite3",
            asset_root=ASSET_ROOT,
            condition="tool",
            fixture=fixture,
            client=client,
            tool_failure_policy="stop",
        )
    )
    assert result.status == "failed"
    assert result.model_calls == 2
    assert result.tool_calls == 1
    assert result.protocol_loop is False
    assert len(seen) == 2
    assert result.model_attempts[-1]["error_stage"] == "timeout"
    engine.dispose()


class _ScriptedClock:
    def __init__(self, values):
        self._values = list(values)
        self.now = self._values[-1] if self._values else 0.0

    def __call__(self):
        if self._values:
            self.now = self._values.pop(0)
        return self.now


def test_late_response_is_rejected_at_same_absolute_deadline():
    clock = _ScriptedClock([0.0, 2.0])
    seen = []

    def handler(request: httpx.Request):
        seen.append(request)
        clock.now = 11.0
        return httpx.Response(200, json=_text_response("arrived after deadline"))

    client = DeepSeekToolClient(
        api_key="test-key-not-real",
        transport=httpx.MockTransport(handler),
        clock=clock,
    )
    with pytest.raises(DeepSeekTimeout):
        asyncio.run(
            client.generate_turn(
                [{"role": "user", "content": "question"}],
                tools=[TOOL_SCHEMA],
                phase="first",
                tool_choice="auto",
                deadline=10.0,
            )
        )
    assert len(seen) == 1
    record = client.take_last_call()
    assert record["request_sent"] is True
    assert record["effective_timeout_s"] == pytest.approx(8.0)
    assert record["error_stage"] == "timeout"


def test_budget_can_expire_during_request_preparation():
    clock = _ScriptedClock([0.0, 10.0])
    seen = []

    def handler(request: httpx.Request):
        seen.append(request)
        return httpx.Response(200, json=_text_response("unexpected"))

    client = DeepSeekToolClient(
        api_key="test-key-not-real",
        transport=httpx.MockTransport(handler),
        clock=clock,
    )
    with pytest.raises(DeepSeekBudgetExhausted):
        asyncio.run(
            client.generate_turn(
                [{"role": "user", "content": "question"}],
                tools=[TOOL_SCHEMA],
                phase="first",
                tool_choice="auto",
                deadline=10.0,
            )
        )
    assert seen == []
    record = client.take_last_call()
    assert record["request_sent"] is False
    assert record["effective_timeout_s"] == 0.0
    assert record["error_stage"] == "budget"
