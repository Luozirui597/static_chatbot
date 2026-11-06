"""Isolated tests for the first structured tool-search experiment."""

from __future__ import annotations

import asyncio
import json
import sqlite3
import threading
import time
from pathlib import Path

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from backend.database import create_tables
from backend.models import ChatSession, Message
from experiments.tool_search_history.protocol import (
    TOOL_NAME,
    parse_assistant_turn,
    tool_message,
    validate_result_for_call,
    validate_tool_call,
)
from experiments.tool_search_history.runner import (
    FakeToolClient,
    Fixture,
    extract_source_ids,
    run_condition,
)
from experiments.tool_search_history.tool import ToolExecutor, ToolLimits
from experiments.tool_search_history.types import (
    AssistantTurn,
    ToolCall,
    ToolContext,
    ToolProtocolError,
    ToolResult,
)

ASSET_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PATH = ASSET_ROOT / "fixtures" / "window_outsider.json"
FINAL = (
    "I found [message 3], [message 5], and [message 21]. "
    "Because [message 21] corrects B-C, I cannot confirm A-C."
)
WINDOW_FINAL = "I only see the recent window; I cannot confirm the early teaching."


def _engine(tmp_path: Path):
    path = tmp_path / "experiment.sqlite3"
    path.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(f"sqlite:///{path}")
    create_tables(engine)
    return engine, path


def _seed(engine, fixture: Fixture):
    from experiments.tool_search_history.runner import seed_fixture

    factory = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    db = factory()
    try:
        session_id, current_id = seed_fixture(
            db, fixture=fixture, title="tool experiment"
        )
    finally:
        db.close()
    return factory, session_id, current_id


def _call(**arguments) -> ToolCall:
    return ToolCall(
        id="call_1",
        name=TOOL_NAME,
        arguments=arguments,
        raw_arguments=json.dumps(arguments, ensure_ascii=False),
    )


def _run(coro):
    return asyncio.run(coro)


class TestStructuredProtocol:
    def test_valid_tool_call(self):
        call = _call(query="A B", limit=3)
        validate_tool_call(call)
        assert call.arguments["limit"] == 3

    @pytest.mark.parametrize(
        "arguments",
        [
            {"query": ""},
            {"query": "   "},
            {"query": "x" * 201},
            {"query": "A", "limit": 0},
            {"query": "A", "limit": 6},
            {"query": "A", "limit": True},
            {"query": "A", "session_id": 2},
            {"query": "A", "extra": 1},
        ],
    )
    def test_invalid_arguments_are_rejected(self, arguments):
        with pytest.raises(ToolProtocolError):
            validate_tool_call(_call(**arguments))

    def test_multiple_calls_are_rejected(self):
        with pytest.raises(ToolProtocolError, match="multiple_tool_calls"):
            parse_assistant_turn(
                content=None,
                raw_calls=[
                    {"id": "a", "function": {"name": TOOL_NAME, "arguments": {"query": "A"}}},
                    {"id": "b", "function": {"name": TOOL_NAME, "arguments": {"query": "B"}}},
                ],
            )

    def test_result_call_id_must_match(self):
        call = _call(query="A")
        wrong = ToolResult(
            tool_call_id="other",
            name=TOOL_NAME,
            content={"status": "ok", "results": []},
        )
        with pytest.raises(ToolProtocolError, match="tool_call_id_mismatch"):
            validate_result_for_call(wrong, call)


class TestToolExecutor:
    def test_session_bound_search_returns_teacher_sources(self, tmp_path):
        fixture = Fixture.from_file(FIXTURE_PATH)
        engine, path = _engine(tmp_path)
        factory, session_id, current_id = _seed(engine, fixture)
        result = _run(
            ToolExecutor(path).execute(
                _call(query=fixture.fixed_query, limit=3),
                ToolContext(session_id=session_id, current_message_id=current_id),
            )
        )
        assert result.content["status"] == "ok"
        ids = [item["message_id"] for item in result.content["results"]]
        assert ids == [3, 5, 21]
        assert [item["kind"] for item in result.content["results"]] == [
            "teaching_match",
            "teaching_match",
            "correction_candidate",
        ]
        assert [item["order"] for item in result.content["results"]] == sorted(
            item["order"] for item in result.content["results"]
        )
        assert all(item["content"] for item in result.content["results"])
        assert result.content["scan_truncated"] is False
        assert result.content["corrections_truncated"] is False
        factory().close()
        engine.dispose()

    def test_current_message_from_another_session_is_rejected(self, tmp_path):
        fixture = Fixture.from_file(FIXTURE_PATH)
        engine, path = _engine(tmp_path)
        factory, session_id, _ = _seed(engine, fixture)
        db = factory()
        other = ChatSession(title="other")
        db.add(other)
        db.flush()
        foreign = Message(
            session_id=other.id,
            role="user",
            content="A B C links",
            interaction_mode_snapshot="receive_teaching",
            prompt_version_snapshot="test",
            llm_profile_id_snapshot="fake",
            llm_profile_kind_snapshot="fake",
            llm_model_snapshot="fake",
        )
        db.add(foreign)
        db.commit()
        db.refresh(foreign)
        db.close()
        result = _run(
            ToolExecutor(path).execute(
                _call(query="A B C"),
                ToolContext(session_id=session_id, current_message_id=foreign.id),
            )
        )
        assert result.content["status"] == "invalid_context"
        assert result.is_error is True
        engine.dispose()

    def test_current_message_role_must_be_user(self, tmp_path):
        fixture = Fixture.from_file(FIXTURE_PATH)
        engine, path = _engine(tmp_path)
        factory, session_id, current_id = _seed(engine, fixture)
        db = factory()
        current = db.get(Message, current_id)
        current.role = "assistant"
        db.commit()
        db.close()
        result = _run(
            ToolExecutor(path).execute(
                _call(query="A B C"),
                ToolContext(session_id=session_id, current_message_id=current_id),
            )
        )
        assert result.content["status"] == "invalid_context"
        engine.dispose()

    def test_search_does_not_return_assistant_text(self, tmp_path):
        fixture = Fixture.from_file(FIXTURE_PATH)
        engine, path = _engine(tmp_path)
        factory, session_id, current_id = _seed(engine, fixture)
        result = _run(
            ToolExecutor(path).execute(
                _call(query="Understood", limit=5),
                ToolContext(session_id=session_id, current_message_id=current_id),
            )
        )
        assert result.content["status"] == "empty"
        assert result.content["results"] == []
        factory().close()
        engine.dispose()

    def test_pronoun_correction_without_shared_token_is_not_returned(self, tmp_path):
        engine, path = _engine(tmp_path)
        factory = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
        db = factory()
        session = ChatSession(title="pronoun")
        db.add(session)
        db.flush()
        for content in [
            "I teach you A links to B.",
            "Actually, it's false.",
        ]:
            db.add(
                Message(
                    session_id=session.id,
                    role="user",
                    content=content,
                    interaction_mode_snapshot="receive_teaching",
                    prompt_version_snapshot="test",
                    llm_profile_id_snapshot="fake",
                    llm_profile_kind_snapshot="fake",
                    llm_model_snapshot="fake",
                )
            )
        db.commit()
        current = Message(
            session_id=session.id,
            role="user",
            content="What about A?",
            interaction_mode_snapshot="receive_teaching",
            prompt_version_snapshot="test",
            llm_profile_id_snapshot="fake",
            llm_profile_kind_snapshot="fake",
            llm_model_snapshot="fake",
        )
        db.add(current)
        db.commit()
        db.refresh(current)
        result = _run(
            ToolExecutor(path).execute(
                _call(query="A links"),
                ToolContext(session_id=session.id, current_message_id=current.id),
            )
        )
        contents = [item["content"] for item in result.content["results"]]
        assert "I teach you A links to B." in contents
        assert "Actually, it's false." not in contents
        assert result.content["status"] == "ok"
        db.close()
        engine.dispose()

    def test_scan_limit_and_result_limit(self, tmp_path):
        engine, path = _engine(tmp_path)
        factory = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
        db = factory()
        session = ChatSession(title="many")
        db.add(session)
        db.flush()
        for index in range(220):
            db.add(
                Message(
                    session_id=session.id,
                    role="user",
                    content=f"target fact {index}",
                    interaction_mode_snapshot="receive_teaching",
                    prompt_version_snapshot="test",
                    llm_profile_id_snapshot="fake",
                    llm_profile_kind_snapshot="fake",
                    llm_model_snapshot="fake",
                )
            )
        db.commit()
        current = Message(
            session_id=session.id,
            role="user",
            content="target?",
            interaction_mode_snapshot="receive_teaching",
            prompt_version_snapshot="test",
            llm_profile_id_snapshot="fake",
            llm_profile_kind_snapshot="fake",
            llm_model_snapshot="fake",
        )
        db.add(current)
        db.commit()
        db.refresh(current)
        result = _run(
            ToolExecutor(path).execute(
                _call(query="target", limit=5),
                ToolContext(session_id=session.id, current_message_id=current.id),
            )
        )
        assert result.content["status"] == "ok"
        assert len(result.content["results"]) == 5
        assert result.content["scan_truncated"] is True
        assert result.content["omitted_count"] > 0
        db.close()
        engine.dispose()

    def test_timeout_returns_timeout_without_partial_results(self, tmp_path):
        fixture = Fixture.from_file(FIXTURE_PATH)
        engine, _ = _engine(tmp_path)
        factory, session_id, current_id = _seed(engine, fixture)
        opened = threading.Event()

        class SlowConnection:
            def __init__(self, path, **_kwargs):
                self._conn = sqlite3.connect(path)
                self.closed = threading.Event()

            def execute(self, sql, params=()):
                if "SELECT id, content" in sql:
                    opened.set()
                    time.sleep(0.3)
                return self._conn.execute(sql, params)

            def set_progress_handler(self, *args):
                return self._conn.set_progress_handler(*args)

            def close(self):
                self._conn.close()
                self.closed.set()

        executor = ToolExecutor(
            tmp_path / "experiment.sqlite3",
            limits=ToolLimits(timeout_seconds=0.03),
            connection_factory=SlowConnection,
        )
        result = _run(
            executor.execute(
                _call(query="A B C", limit=3),
                ToolContext(session_id=session_id, current_message_id=current_id),
            )
        )
        assert result.content["status"] == "timeout"
        assert result.content["results"] == []
        assert opened.wait(1.0)
        # The worker is allowed to finish; its late result is ignored.
        deadline = time.monotonic() + 2.0
        while not executor.last_connection_closed and time.monotonic() < deadline:
            time.sleep(0.01)
        assert executor.last_connection_closed is True
        factory().close()
        engine.dispose()


    def test_tool_timeout_is_returned_and_then_final_is_generated(self, tmp_path):
        fixture = Fixture.from_file(FIXTURE_PATH)
        engine, path = _engine(tmp_path)
        factory = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)

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

        client = FakeToolClient(
            [
                AssistantTurn(
                    content=None,
                    tool_calls=(ToolCall(
                        id="slow",
                        name=TOOL_NAME,
                        arguments={"query": fixture.fixed_query},
                    ),),
                ),
                AssistantTurn(content="The search timed out; evidence is insufficient."),
            ]
        )
        result = _run(
            run_condition(
                session_factory=factory,
                database_path=path,
                asset_root=ASSET_ROOT,
                condition="tool",
                fixture=fixture,
                client=client,
                executor=TimeoutExecutor(),
            )
        )
        assert result.status == "completed"
        assert result.model_calls == 2
        assert result.tool_calls == 1
        assert result.tool_status == "timeout"
        exchange = result.tool_exchanges[0]["tool"]
        assert exchange["role"] == "tool"
        assert exchange["tool_call_id"] == "slow"
        assert json.loads(exchange["content"])["status"] == "timeout"
        engine.dispose()

class TestThreeConditions:
    def _run_all(self, tmp_path):
        fixture = Fixture.from_file(FIXTURE_PATH)
        results = {}
        for condition in ("window", "fixed", "tool"):
            engine, path = _engine(tmp_path / condition)
            factory = sessionmaker(
                bind=engine, autoflush=False, expire_on_commit=False
            )
            if condition == "window":
                responses = [AssistantTurn(content=WINDOW_FINAL)]
            elif condition == "fixed":
                responses = [AssistantTurn(content=FINAL)]
            else:
                responses = [
                    AssistantTurn(
                        content=None,
                        tool_calls=(
                            ToolCall(
                                id="call_1",
                                name=TOOL_NAME,
                                arguments={
                                    "query": fixture.fixed_query,
                                    "limit": 3,
                                },
                            ),
                        ),
                    ),
                    AssistantTurn(content=FINAL),
                ]
            client = FakeToolClient(responses)
            result = _run(
                run_condition(
                    session_factory=factory,
                    database_path=path,
                    asset_root=ASSET_ROOT,
                    condition=condition,
                    fixture=fixture,
                    client=client,
                )
            )
            results[condition] = (result, engine, factory, client)
        return results

    def test_window_fixed_tool_compare(self, tmp_path):
        fixture = Fixture.from_file(FIXTURE_PATH)
        results = self._run_all(tmp_path)
        window, _, _, _ = results["window"]
        fixed, _, _, fixed_client = results["fixed"]
        tool, _, _, tool_client = results["tool"]
        assert window.status == "completed"
        assert window.model_calls == 1
        assert window.tool_calls == 0
        assert window.source_ids == []
        assert "Correction: B does not link" not in json.dumps(window.messages)
        assert fixed.status == "completed"
        assert fixed.program_retrievals == 1
        assert fixed.tool_calls == 0
        assert fixed.model_calls == 1
        assert fixed.tool_exchanges[0]["program_generated"] is True
        assert fixed.source_ids == [3, 5, 21]
        assert tool.status == "completed"
        assert tool.model_calls == 2
        assert tool.tool_calls == 1
        assert tool.source_ids == [3, 5, 21]

        def history_block(messages):
            return next(
                m["content"]
                for m in messages
                if m.get("role") == "user"
                and "receive_teaching_history_background" in m["content"]
            )

        window_history = history_block(window.messages)
        fixed_history = history_block(fixed_client.message_batches[0])
        tool_history = history_block(tool_client.message_batches[0])
        assert window_history == fixed_history == tool_history
        assert window.messages[-1]["content"] == fixture.current
        assert fixed_client.message_batches[0][-1]["content"] == fixture.current
        assert tool_client.message_batches[0][-1]["content"] == fixture.current

        fixed_tool_result = json.loads(
            fixed.tool_exchanges[0]["tool"]["content"]
        )
        assert [item["message_id"] for item in fixed_tool_result["results"]] == [
            3, 5, 21
        ]
        assert tool.source_ids == [
            item["message_id"]
            for item in json.loads(
                tool.tool_exchanges[0]["tool"]["content"]
            )["results"]
            if item["message_id"] in tool.source_ids
        ]
        assert window.tool_exchanges == []
        # The fake's second context contains the exact structured exchange.
        second_batch = tool_client.message_batches[1]
        assistant = [m for m in second_batch if m.get("tool_calls")][0]
        tool_message_row = [m for m in second_batch if m.get("role") == "tool"][0]
        assert assistant["tool_calls"][0]["id"] == "call_1"
        assert tool_message_row["tool_call_id"] == "call_1"
        assert json.loads(tool_message_row["content"])["results"][0]["message_id"] == 3
        # Only the final assistant is persisted; tool exchange stays contextual.
        factory = results["tool"][2]
        db = factory()
        roles = [
            row.role
            for row in db.execute(
                select(Message).where(Message.session_id == 1).order_by(Message.id)
            ).scalars()
        ]
        assert "tool" not in roles
        assert roles.count("assistant") >= 2
        db.close()
        for result, engine, factory, _ in results.values():
            engine.dispose()

    def test_fake_second_context_has_same_call_id_and_source(self, tmp_path):
        fixture = Fixture.from_file(FIXTURE_PATH)
        engine, path = _engine(tmp_path)
        factory = sessionmaker(
            bind=engine, autoflush=False, expire_on_commit=False
        )
        client = FakeToolClient(
            [
                AssistantTurn(
                    content=None,
                    tool_calls=(
                        ToolCall(
                            id="abc",
                            name=TOOL_NAME,
                            arguments={"query": fixture.fixed_query, "limit": 3},
                        ),
                    ),
                ),
                AssistantTurn(content=FINAL),
            ]
        )
        result = _run(
            run_condition(
                session_factory=factory,
                database_path=path,
                asset_root=ASSET_ROOT,
                condition="tool",
                fixture=fixture,
                client=client,
            )
        )
        assert result.status == "completed"
        second = client.message_batches[1]
        assert any(m.get("role") == "tool" for m in second)
        assert any(m.get("tool_calls") for m in second)
        engine.dispose()

    def test_source_ids_are_extracted(self):
        assert extract_source_ids("[message 3] and [message 21]") == [3, 21]
        assert extract_source_ids("no citation") == []

    def test_ok_result_without_correction_still_requires_uncertainty(self, tmp_path):
        fixture = Fixture(
            messages=(
                {"role": "user", "content": "I teach you A links to B."},
                {"role": "user", "content": "Actually, it's false."},
            ),
            current="What did I teach and correct about A?",
            fixed_query="A links",
            expected_source_ids=(1,),
        )
        engine, path = _engine(tmp_path)
        factory = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
        client = FakeToolClient(
            [
                AssistantTurn(
                    content=None,
                    tool_calls=(ToolCall(
                        id="pronoun",
                        name=TOOL_NAME,
                        arguments={"query": "A links"},
                    ),),
                ),
                AssistantTurn(
                    content="I found [message 1], but the retrieved evidence does not establish a correction."
                ),
            ]
        )
        result = _run(
            run_condition(
                session_factory=factory,
                database_path=path,
                asset_root=ASSET_ROOT,
                condition="tool",
                fixture=fixture,
                client=client,
            )
        )
        assert result.status == "completed"
        assert result.tool_status == "ok"
        assert result.source_ids == [1]
        tool_payload = json.loads(result.tool_exchanges[0]["tool"]["content"])
        assert [item["message_id"] for item in tool_payload["results"]] == [1]
        engine.dispose()

    def test_multiple_tool_calls_fail_before_execution(self, tmp_path):
        fixture = Fixture.from_file(FIXTURE_PATH)
        engine, path = _engine(tmp_path)
        factory = sessionmaker(
            bind=engine, autoflush=False, expire_on_commit=False
        )
        client = FakeToolClient(
            [
                AssistantTurn(
                    content=None,
                    tool_calls=(
                        ToolCall(id="a", name=TOOL_NAME, arguments={"query": "A"}),
                        ToolCall(id="b", name=TOOL_NAME, arguments={"query": "B"}),
                    ),
                )
            ]
        )
        result = _run(
            run_condition(
                session_factory=factory,
                database_path=path,
                asset_root=ASSET_ROOT,
                condition="tool",
                fixture=fixture,
                client=client,
            )
        )
        assert result.status == "failed"
        assert result.model_calls == 1
        assert result.tool_calls == 0
        assert client.message_batches
        engine.dispose()

    def test_invalid_arguments_return_tool_error_then_final(self, tmp_path):
        fixture = Fixture.from_file(FIXTURE_PATH)
        engine, path = _engine(tmp_path)
        factory = sessionmaker(
            bind=engine, autoflush=False, expire_on_commit=False
        )
        client = FakeToolClient(
            [
                AssistantTurn(
                    content=None,
                    tool_calls=(
                        ToolCall(
                            id="bad",
                            name=TOOL_NAME,
                            arguments={"query": "x" * 201},
                        ),
                    ),
                ),
                AssistantTurn(content="I cannot verify with the available evidence."),
            ]
        )
        result = _run(
            run_condition(
                session_factory=factory,
                database_path=path,
                asset_root=ASSET_ROOT,
                condition="tool",
                fixture=fixture,
                client=client,
            )
        )
        assert result.status == "completed"
        assert result.model_calls == 2
        assert result.tool_calls == 1
        assert result.tool_status == "invalid_arguments"
        engine.dispose()

    def test_second_tool_request_fails_without_third_generation(self, tmp_path):
        fixture = Fixture.from_file(FIXTURE_PATH)
        engine, path = _engine(tmp_path)
        factory = sessionmaker(
            bind=engine, autoflush=False, expire_on_commit=False
        )
        first = AssistantTurn(
            content=None,
            tool_calls=(
                ToolCall(id="a", name=TOOL_NAME, arguments={"query": "A"}),
            ),
        )
        second = AssistantTurn(
            content=None,
            tool_calls=(
                ToolCall(id="b", name=TOOL_NAME, arguments={"query": "B"}),
            ),
        )
        client = FakeToolClient([first, second])
        result = _run(
            run_condition(
                session_factory=factory,
                database_path=path,
                asset_root=ASSET_ROOT,
                condition="tool",
                fixture=fixture,
                client=client,
            )
        )
        assert result.status == "failed"
        assert result.model_calls == 2
        assert result.tool_calls == 1
        assert len(client.message_batches) == 2
        engine.dispose()

    def test_mismatched_tool_result_is_rejected(self, tmp_path):
        fixture = Fixture.from_file(FIXTURE_PATH)
        engine, path = _engine(tmp_path)
        factory = sessionmaker(
            bind=engine, autoflush=False, expire_on_commit=False
        )

        class WrongIdExecutor:
            async def execute(self, call, context, **kwargs):
                return ToolResult(
                    tool_call_id="wrong",
                    name=call.name,
                    content={"status": "ok", "results": []},
                )

        client = FakeToolClient(
            [
                AssistantTurn(
                    content=None,
                    tool_calls=(ToolCall(id="right", name=TOOL_NAME, arguments={"query": "A"}),),
                ),
                AssistantTurn(content="should not be called"),
            ]
        )
        result = _run(
            run_condition(
                session_factory=factory,
                database_path=path,
                asset_root=ASSET_ROOT,
                condition="tool",
                fixture=fixture,
                client=client,
                executor=WrongIdExecutor(),
            )
        )
        assert result.status == "failed"
        assert result.model_calls == 1
        assert result.tool_calls == 1
        assert len(client.message_batches) == 1
        engine.dispose()

    def test_unknown_tool_returns_error_then_final(self, tmp_path):
        fixture = Fixture.from_file(FIXTURE_PATH)
        engine, path = _engine(tmp_path)
        factory = sessionmaker(
            bind=engine, autoflush=False, expire_on_commit=False
        )
        client = FakeToolClient(
            [
                AssistantTurn(
                    content=None,
                    tool_calls=(ToolCall(id="unknown", name="not_a_tool", arguments={}),),
                ),
                AssistantTurn(content="I cannot verify with the available evidence."),
            ]
        )
        result = _run(
            run_condition(
                session_factory=factory,
                database_path=path,
                asset_root=ASSET_ROOT,
                condition="tool",
                fixture=fixture,
                client=client,
            )
        )
        assert result.status == "completed"
        assert result.model_calls == 2
        assert result.tool_status == "invalid_arguments"
        engine.dispose()


def test_newer_corrections_get_reserved_slots_and_flags(tmp_path):
    engine, path = _engine(tmp_path)
    factory = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    db = factory()
    session = ChatSession(title="corrections")
    db.add(session)
    db.flush()
    contents = [
        "I teach you A links to B.",
        "Correction: A links to C.",
        "Correction: A links to D.",
        "Correction: A links to E.",
    ]
    for content in contents:
        db.add(
            Message(
                session_id=session.id,
                role="user",
                content=content,
                interaction_mode_snapshot="receive_teaching",
                prompt_version_snapshot="test",
                llm_profile_id_snapshot="fake",
                llm_profile_kind_snapshot="fake",
                llm_model_snapshot="fake",
            )
        )
    db.commit()
    current = Message(
        session_id=session.id,
        role="user",
        content="What about A?",
        interaction_mode_snapshot="receive_teaching",
        prompt_version_snapshot="test",
        llm_profile_id_snapshot="fake",
        llm_profile_kind_snapshot="fake",
        llm_model_snapshot="fake",
    )
    db.add(current)
    db.commit()
    db.refresh(current)
    result = _run(
        ToolExecutor(path).execute(
            _call(query="A links", limit=5),
            ToolContext(session_id=session.id, current_message_id=current.id),
        )
    )
    assert result.content["status"] == "ok"
    ids = [item["message_id"] for item in result.content["results"]]
    # Newest two correction candidates are reserved; the older direct teaching
    # still fills the remaining slot, and final output is chronological.
    assert ids == [1, 3, 4]
    assert result.content["corrections_truncated"] is True
    assert result.content["omitted_count"] == 1
    db.close()
    engine.dispose()


def test_long_source_content_marks_truncated(tmp_path):
    engine, path = _engine(tmp_path)
    factory = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    db = factory()
    session = ChatSession(title="long")
    db.add(session)
    db.flush()
    db.add(
        Message(
            session_id=session.id,
            role="user",
            content="target " + ("x" * 1200),
            interaction_mode_snapshot="receive_teaching",
            prompt_version_snapshot="test",
            llm_profile_id_snapshot="fake",
            llm_profile_kind_snapshot="fake",
            llm_model_snapshot="fake",
        )
    )
    db.commit()
    current = Message(
        session_id=session.id,
        role="user",
        content="target?",
        interaction_mode_snapshot="receive_teaching",
        prompt_version_snapshot="test",
        llm_profile_id_snapshot="fake",
        llm_profile_kind_snapshot="fake",
        llm_model_snapshot="fake",
    )
    db.add(current)
    db.commit()
    db.refresh(current)
    result = _run(
        ToolExecutor(path).execute(
            _call(query="target"),
            ToolContext(session_id=session.id, current_message_id=current.id),
        )
    )
    item = result.content["results"][0]
    assert len(item["content"]) == 1000
    assert item["truncated"] is True
    assert result.content["truncated"] is True
    db.close()
    engine.dispose()


def _seed_contents(tmp_path, contents):
    engine, path = _engine(tmp_path)
    factory = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    db = factory()
    session = ChatSession(title="ad hoc")
    db.add(session)
    db.flush()
    for content in contents:
        db.add(
            Message(
                session_id=session.id,
                role="user",
                content=content,
                interaction_mode_snapshot="receive_teaching",
                prompt_version_snapshot="test",
                llm_profile_id_snapshot="fake",
                llm_profile_kind_snapshot="fake",
                llm_model_snapshot="fake",
            )
        )
    db.commit()
    current = Message(
        session_id=session.id,
        role="user",
        content="Which teaching applies?",
        interaction_mode_snapshot="receive_teaching",
        prompt_version_snapshot="test",
        llm_profile_id_snapshot="fake",
        llm_profile_kind_snapshot="fake",
        llm_model_snapshot="fake",
    )
    db.add(current)
    db.commit()
    db.refresh(current)
    db.close()
    return engine, path, session.id, current.id


def _execute(path, session_id, current_id, query, limits=None):
    return _run(
        ToolExecutor(path, limits=limits).execute(
            _call(query=query, limit=5),
            ToolContext(session_id=session_id, current_message_id=current_id),
        )
    )


def test_query_tokens_are_capped_at_six_distinct_terms(tmp_path):
    engine, path, session_id, current_id = _seed_contents(
        tmp_path,
        [
            "alpha beta gamma delta epsilon zeta",
            "eta only",
        ],
    )
    result = _execute(
        path,
        session_id,
        current_id,
        "alpha alpha beta beta gamma delta epsilon zeta eta",
    )
    assert [item["content"] for item in result.content["results"]] == [
        "alpha beta gamma delta epsilon zeta"
    ]
    assert "eta only" not in [
        item["content"] for item in result.content["results"]
    ]
    engine.dispose()


def test_isolated_negation_is_not_a_correction_candidate(tmp_path):
    engine, path, session_id, current_id = _seed_contents(
        tmp_path, ["Actually, A is false."]
    )
    result = _execute(path, session_id, current_id, "A")
    assert result.content["results"][0]["kind"] == "teaching_match"
    assert "correction_cue" not in result.content["results"][0]
    assert result.content["corrections_truncated"] is False
    assert result.content["omitted_count"] == 0
    engine.dispose()


def test_support_must_share_a_query_token(tmp_path):
    engine, path, session_id, current_id = _seed_contents(
        tmp_path,
        [
            "I teach B links to C.",
            "Correction: A links to D.",
        ],
    )
    result = _execute(path, session_id, current_id, "A B")
    by_content = {
        item["content"]: item for item in result.content["results"]
    }
    assert by_content["Correction: A links to D."]["kind"] == "teaching_match"
    assert "correction_cue" not in by_content["Correction: A links to D."]
    assert result.content["corrections_truncated"] is False
    engine.dispose()


def test_support_outside_scan_keeps_scan_truncated_and_no_reserved_correction(tmp_path):
    engine, path, session_id, current_id = _seed_contents(
        tmp_path,
        [
            "I teach A links to B.",
            "Correction: A links to C.",
        ],
    )
    result = _execute(
        path,
        session_id,
        current_id,
        "A",
        limits=ToolLimits(max_scan_messages=1),
    )
    assert result.content["scan_truncated"] is True
    assert result.content["results"][0]["kind"] == "teaching_match"
    assert "support_message_id" not in result.content["results"][0]
    assert result.content["corrections_truncated"] is False
    engine.dispose()


def test_supported_correction_is_marked_with_support(tmp_path):
    engine, path, session_id, current_id = _seed_contents(
        tmp_path,
        [
            "I teach A links to B.",
            "Correction: A links to C.",
        ],
    )
    result = _execute(path, session_id, current_id, "A")
    correction = next(
        item for item in result.content["results"]
        if item["kind"] == "correction_candidate"
    )
    assert correction["support_message_id"] == 1
    assert correction["correction_cue"] == "correction"
    engine.dispose()


def test_correction_priority_precedes_teaching_char_budget(tmp_path):
    old_teaching = "A " + ("x" * 68)
    correction = "Actually A " + ("y" * 59)
    small_teaching = "A small"
    engine, path, session_id, current_id = _seed_contents(
        tmp_path, [old_teaching, correction, small_teaching]
    )
    result = _execute(
        path,
        session_id,
        current_id,
        "A",
        limits=ToolLimits(
            max_results=5,
            max_content_chars=100,
            max_result_chars=100,
        ),
    )
    ids = [item["message_id"] for item in result.content["results"]]
    assert ids == [2, 3]
    assert result.content["results"][0]["kind"] == "correction_candidate"
    assert result.content["results"][0]["content"].startswith("Actually A")
    assert result.content["results"][1]["content"] == "A small"
    assert result.content["omitted_count"] == 1
    assert result.content["truncated"] is True
    assert result.content["corrections_truncated"] is False
    engine.dispose()


def test_truncated_correction_sets_corrections_truncated(tmp_path):
    engine, path, session_id, current_id = _seed_contents(
        tmp_path,
        [
            "I teach A links to B.",
            "Correction: A links to C. " + ("z" * 200),
        ],
    )
    result = _execute(
        path,
        session_id,
        current_id,
        "A",
        limits=ToolLimits(
            max_results=5,
            max_content_chars=80,
            max_result_chars=4_000,
        ),
    )
    correction = next(
        item for item in result.content["results"]
        if item["kind"] == "correction_candidate"
    )
    assert correction["truncated"] is True
    assert result.content["truncated"] is True
    assert result.content["corrections_truncated"] is True
    engine.dispose()
