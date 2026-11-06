"""Isolated tests for the programmatic retrieval condition."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.database import create_tables
from experiments.tool_search_history.deepseek_client import DeepSeekToolClient
from experiments.tool_search_history.runner import (
    FakeToolClient,
    Fixture,
    run_condition,
)
from experiments.tool_search_history.trigger import decide, load_window_boundary
from experiments.tool_search_history.types import AssistantTurn, ToolResult

ASSET_ROOT = Path(__file__).resolve().parents[1]


def _fixture(messages, current):
    normalized_list = []
    for item in messages:
        if isinstance(item, dict):
            normalized_list.append(dict(item))
        else:
            role, content = item
            normalized_list.append({"role": role, "content": content})
    normalized = tuple(normalized_list)
    return Fixture(
        messages=normalized,
        current=current,
        fixed_query="",
        expected_source_ids=(),
    )


def _seed(tmp_path, messages, current):
    engine = create_engine(f"sqlite:///{tmp_path / 'programmatic.sqlite3'}")
    create_tables(engine)
    factory = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    from experiments.tool_search_history.runner import seed_fixture

    db = factory()
    session_id, current_id = seed_fixture(
        db, fixture=_fixture(messages, current), title="programmatic"
    )
    db.close()
    return engine, tmp_path / "programmatic.sqlite3", factory, session_id, current_id


def _run(fixture, *, client=None, executor=None, tmp_path=None):
    assert tmp_path is not None
    engine = create_engine(f"sqlite:///{tmp_path / 'run.sqlite3'}")
    create_tables(engine)
    return asyncio.run(
        run_condition(
            session_factory=sessionmaker(
                bind=engine,
                autoflush=False,
                expire_on_commit=False,
            ),
            database_path=tmp_path / "run.sqlite3",
            asset_root=ASSET_ROOT,
            condition="programmatic",
            fixture=fixture,
            client=client or FakeToolClient([AssistantTurn(content="ok")]),
            executor=executor,
            deadline_seconds=65.0,
        )
    )


def _hidden_fixture(current="Does B link to C?"):
    messages = [
        ("user", "Fact: A links to B."),
        ("assistant", "Understood."),
        ("user", "Fact: B links to C."),
        ("assistant", "Understood."),
        ("user", "Correction: B does not link to C."),
        ("assistant", "Understood."),
    ]
    for i in range(10):
        messages.extend(
            [
                ("user", f"Fact: object_{i} links to object_{i + 1}."),
                ("assistant", "Understood."),
            ]
        )
    return _fixture(messages, current)


def _visible_complete_fixture():
    messages = [("user", "Fact: X links to Y.")]
    for i in range(9):
        messages.extend(
            [
                ("user", f"Fact: object_{i} links to object_{i + 1}."),
                ("assistant", "Understood."),
            ]
        )
    messages.extend(
        [
            ("user", "Fact: B links to C."),
            ("assistant", "Understood."),
            ("user", "Correction: B does not link to C."),
            ("assistant", "Understood."),
        ]
    )
    return _fixture(messages, "Does B link to C?")


def test_boundary_20_21_and_hidden_teacher(tmp_path):
    # 20 stored messages: 10 pairs, no overflow.
    twenty = []
    for i in range(10):
        twenty.extend(
            [
                ("user", f"Fact: object_{i} links to object_{i + 1}."),
                ("assistant", "Understood."),
            ]
        )
    engine, path, factory, sid, cid = _seed(tmp_path, twenty, "Does X link to Y?")
    db = factory()
    boundary = load_window_boundary(db, session_id=sid, current_message_id=cid)
    assert boundary.window_overflow is False
    assert boundary.hidden_teacher_exists is False
    assert len(boundary.visible_teacher) == 10
    db.close()
    engine.dispose()

    # 21 with oldest assistant: overflow, no hidden teacher.
    twenty_one_a = [("assistant", "Understood.")] + twenty
    engine, path, factory, sid, cid = _seed(tmp_path, twenty_one_a, "Does X link to Y?")
    db = factory()
    boundary = load_window_boundary(db, session_id=sid, current_message_id=cid)
    assert boundary.window_overflow is True
    assert boundary.hidden_teacher_exists is False
    db.close()
    engine.dispose()

    # 21 with oldest teacher: overflow and hidden teacher.
    twenty_one_t = [("user", "Fact: X links to Y.")] + twenty
    engine, path, factory, sid, cid = _seed(tmp_path, twenty_one_t, "Does X link to Y?")
    db = factory()
    boundary = load_window_boundary(db, session_id=sid, current_message_id=cid)
    assert boundary.window_overflow is True
    assert boundary.hidden_teacher_exists is True
    db.close()
    engine.dispose()


def test_trigger_hidden_visible_and_special_cases(tmp_path):
    hidden = _hidden_fixture()
    engine, path, factory, sid, cid = _seed(
        tmp_path, hidden.messages, hidden.current
    )
    db = factory()
    boundary = load_window_boundary(db, session_id=sid, current_message_id=cid)
    decision = decide(boundary, current_content=hidden.current)
    assert decision.action == "retrieve"
    assert decision.reason == "hidden_retrieval"
    assert decision.query == "b c"
    db.close()
    engine.dispose()

    visible = _visible_complete_fixture()
    engine, path, factory, sid, cid = _seed(
        tmp_path, visible.messages, visible.current
    )
    db = factory()
    boundary = load_window_boundary(db, session_id=sid, current_message_id=cid)
    decision = decide(boundary, current_content=visible.current)
    assert decision.action == "direct"
    assert decision.reason == "visible_complete"
    db.close()
    engine.dispose()


def test_trigger_reverse_mismatch_cooccurrence_and_mixed(tmp_path):
    cases = [
        ("Does C link to B?", "hidden_retrieval"),
        ("Does B link to D?", "hidden_retrieval"),
        ("I teach you this fact: D links to E. Does B link to C?", "hidden_retrieval"),
        ("What is the current B-C?", "unsupported_pattern"),
        ("Did that change?", "unsupported_pattern"),
    ]
    for current, reason in cases:
        fixture = _hidden_fixture(current)
        engine, path, factory, sid, cid = _seed(
            tmp_path, fixture.messages, fixture.current
        )
        db = factory()
        boundary = load_window_boundary(db, session_id=sid, current_message_id=cid)
        decision = decide(boundary, current_content=current)
        assert decision.reason == reason, (current, decision)
        if reason == "hidden_retrieval":
            assert decision.action == "retrieve"
        else:
            assert decision.action == "clarify"
        db.close()
        engine.dispose()


def test_cooccurrence_and_old_quote_are_not_complete(tmp_path):
    messages = [
        ("user", "B and C are blue."),
        ("assistant", "Understood."),
    ]
    for i in range(10):
        messages.extend(
            [
                ("user", f"Fact: filler_{i} links to filler_{i + 1}."),
                ("assistant", "Understood."),
            ]
        )
    fixture = _fixture(messages, "Does B link to C?")
    engine, path, factory, sid, cid = _seed(tmp_path, fixture.messages, fixture.current)
    db = factory()
    boundary = load_window_boundary(db, session_id=sid, current_message_id=cid)
    decision = decide(boundary, current_content=fixture.current)
    assert decision.action == "retrieve"
    assert decision.reason in {"hidden_retrieval", "history_recall"}
    db.close()
    engine.dispose()

    quote_fixture = _hidden_fixture()
    quote_fixture = _fixture(
        [
            ("user", "Earlier note: the old statement said B links to C; I am only quoting it."),
            ("assistant", "Understood."),
        ]
        + list(quote_fixture.messages[2:]),
        "Does B link to C?",
    )
    engine, path, factory, sid, cid = _seed(
        tmp_path, quote_fixture.messages, quote_fixture.current
    )
    db = factory()
    boundary = load_window_boundary(db, session_id=sid, current_message_id=cid)
    decision = decide(boundary, current_content=quote_fixture.current)
    assert decision.action == "retrieve"
    db.close()
    engine.dispose()


def test_main_hidden_programmatic_loop(tmp_path):
    fixture = _hidden_fixture()
    final = "B does not link to C [message 5]."
    result = _run(fixture, client=FakeToolClient([AssistantTurn(content=final)]), tmp_path=tmp_path)
    assert result.status == "completed"
    assert result.programmatic_status == "completed"
    assert result.trigger_reason == "hidden_retrieval"
    assert result.programmatic_retrieval_attempts == 1
    assert result.tool_calls == 1
    assert result.model_calls == 1
    assert result.http_requests == 1
    assert {3, 5}.issubset(set(result.retrieved_source_ids))
    assert {3, 5}.issubset(set(result.injected_source_ids))
    assert result.source_ids == [5]
    assert result.answer_quality == "citations_valid"
    assert result.protocol_loop is False


def test_visible_complete_programmatic_has_zero_retrieval(tmp_path):
    fixture = _visible_complete_fixture()
    result = _run(
        fixture,
        client=FakeToolClient(
            [AssistantTurn(content="B does not link to C [message 22].")]
        ),
        tmp_path=tmp_path,
    )
    assert result.status == "completed"
    assert result.trigger_reason == "visible_complete"
    assert result.programmatic_retrieval_attempts == 0
    assert result.tool_calls == 0
    assert result.model_calls == 1
    assert result.http_requests == 1


def test_tool_failure_stops_without_model_request(tmp_path):
    class TimeoutExecutor:
        async def execute(self, call, context, **kwargs):
            return ToolResult(
                tool_call_id=call.id,
                name=call.name,
                content={"status": "timeout", "results": []},
                is_error=True,
            )

    fixture = _hidden_fixture()
    result = _run(fixture, executor=TimeoutExecutor(), tmp_path=tmp_path)
    assert result.status == "failed"
    assert result.programmatic_status == "tool_failed"
    assert result.programmatic_retrieval_attempts == 1
    assert result.tool_calls == 1
    assert result.model_calls == 0
    assert result.http_requests == 0
    assert result.injected_source_ids == []
    assert result.retrieved_source_ids == []


def test_unexpected_tool_calls_in_direct_programmatic(tmp_path):
    fixture = _visible_complete_fixture()
    result = _run(
        fixture,
        client=FakeToolClient(
            [
                AssistantTurn(
                    content=None,
                    tool_calls=(
                        __import__(
                            "experiments.tool_search_history.types",
                            fromlist=["ToolCall"],
                        ).ToolCall(
                            id="unexpected",
                            name="search_teaching_history",
                            arguments={"query": "B C"},
                        ),
                    ),
                )
            ]
        ),
        tmp_path=tmp_path,
    )
    assert result.status == "failed"
    assert result.programmatic_status == "protocol_failed"
    assert result.programmatic_retrieval_attempts == 0
    assert result.model_calls == 1
    assert result.http_requests == 1
    assert result.tool_calls == 0


def test_invalid_citation_is_not_in_injected_set(tmp_path):
    fixture = _hidden_fixture()
    result = _run(
        fixture,
        client=FakeToolClient(
            [AssistantTurn(content="B does not link to C [message 5] [message 999].")]
        ),
        tmp_path=tmp_path,
    )
    assert result.status == "completed"
    assert result.source_ids == [5, 999]
    assert result.invalid_source_ids == [999]
    assert result.answer_quality == "invalid_citations"


def test_adapter_programmatic_body_has_no_tool_fields(tmp_path):
    fixture = _visible_complete_fixture()
    seen = []

    def handler(request: httpx.Request):
        seen.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-programmatic",
                "model": "deepseek-flash",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "B does not link to C."},
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    engine = create_engine(f"sqlite:///{tmp_path / 'adapter.sqlite3'}")
    create_tables(engine)
    factory = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    result = asyncio.run(
        run_condition(
            session_factory=factory,
            database_path=tmp_path / "adapter.sqlite3",
            asset_root=ASSET_ROOT,
            condition="programmatic",
            fixture=fixture,
            client=DeepSeekToolClient(
                api_key="test-key-not-real",
                transport=httpx.MockTransport(handler),
            ),
        )
    )
    assert result.status == "completed"
    assert len(seen) == 1
    assert "tools" not in seen[0]
    assert "tool_choice" not in seen[0]
    assert seen[0]["messages"][-1]["role"] == "user"
    assert seen[0]["messages"][-1]["content"] == fixture.current
    engine.dispose()


def test_programmatic_retrieval_then_unexpected_tool_calls(tmp_path):
    from experiments.tool_search_history.types import ToolCall

    fixture = _hidden_fixture()
    result = _run(
        fixture,
        client=FakeToolClient(
            [
                AssistantTurn(
                    content=None,
                    tool_calls=(
                        ToolCall(
                            id="unexpected",
                            name="search_teaching_history",
                            arguments={"query": "B C"},
                        ),
                    ),
                )
            ]
        ),
        tmp_path=tmp_path,
    )
    assert result.status == "failed"
    assert result.programmatic_status == "protocol_failed"
    assert result.programmatic_retrieval_attempts == 1
    assert result.tool_calls == 1
    assert result.model_calls == 1
    assert result.http_requests == 1


def test_empty_tool_result_does_not_require_citations(tmp_path):
    messages = [("user", "Fact: X links to Y.")]
    messages.extend([("assistant", "Understood.")] * 20)
    fixture = _fixture(messages, "Does B link to C?")
    result = _run(
        fixture,
        client=FakeToolClient(
            [AssistantTurn(content="The retrieved evidence is empty; I cannot confirm.")]
        ),
        tmp_path=tmp_path,
    )
    assert result.status == "completed"
    assert result.programmatic_retrieval_attempts == 1
    assert result.tool_calls == 1
    assert result.model_calls == 1
    assert result.http_requests == 1
    assert result.visible_source_ids == []
    assert result.answer_quality == "no_citations_available"


def test_truncated_tool_result_is_preserved(tmp_path):
    messages = [
        ("user", "Fact: B links to C."),
        ("assistant", "Understood."),
        ("user", "Fact: B links to C " + ("x" * 1_200)),
        ("assistant", "Understood."),
    ]
    for i in range(10):
        messages.extend(
            [
                ("user", f"Fact: filler_{i} links to filler_{i + 1}."),
                ("assistant", "Understood."),
            ]
        )
    fixture = _fixture(messages, "Does B link to C?")
    result = _run(
        fixture,
        client=FakeToolClient(
            [AssistantTurn(content="The retrieved evidence is incomplete.")]
        ),
        tmp_path=tmp_path,
    )
    payload = result.tool_exchanges[0]["tool"].content
    assert result.programmatic_retrieval_attempts == 1
    assert result.tool_calls == 1
    assert result.model_calls == 1
    assert payload["truncated"] is True
    assert result.answer_quality in {"missing_citations", "no_citations_available"}


def test_pronoun_correction_may_be_missed_without_fabrication(tmp_path):
    messages = [
        ("user", "Fact: B links to C."),
        ("assistant", "Understood."),
        ("user", "Correction: it is false."),
        ("assistant", "Understood."),
    ]
    for i in range(10):
        messages.extend(
            [
                ("user", f"Fact: filler_{i} links to filler_{i + 1}."),
                ("assistant", "Understood."),
            ]
        )
    fixture = _fixture(messages, "Does B link to C?")
    result = _run(
        fixture,
        client=FakeToolClient(
            [AssistantTurn(content="I found only the old B-C statement; I cannot establish a correction.")]
        ),
        tmp_path=tmp_path,
    )
    payload = result.tool_exchanges[0]["tool"].content
    assert result.programmatic_retrieval_attempts == 1
    assert all(
        "Correction: it is false." not in item["content"]
        for item in payload["results"]
    )
    assert result.status == "completed"


def test_cross_session_isolation(tmp_path):
    from backend.models import ChatSession, Message
    from experiments.tool_search_history.runner import seed_fixture

    engine = create_engine(f"sqlite:///{tmp_path / 'cross.sqlite3'}")
    create_tables(engine)
    factory = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    fixture = _hidden_fixture()
    db = factory()
    other = ChatSession(title="other")
    db.add(other)
    db.flush()
    db.add(
        Message(
            session_id=other.id,
            role="user",
            content="Fact: B links to C. This belongs to another session.",
            interaction_mode_snapshot="receive_teaching",
            prompt_version_snapshot="test",
            llm_profile_id_snapshot="fake",
            llm_profile_kind_snapshot="fake",
            llm_model_snapshot="fake",
        )
    )
    db.commit()
    db.close()

    result = asyncio.run(
        run_condition(
            session_factory=factory,
            database_path=tmp_path / "cross.sqlite3",
            asset_root=ASSET_ROOT,
            condition="programmatic",
            fixture=fixture,
            client=FakeToolClient(
                [AssistantTurn(content="B does not link to C [message 6].")]
            ),
        )
    )
    assert result.status == "completed"
    assert 6 in result.retrieved_source_ids
    assert {2, 4, 6}.issubset(set(result.retrieved_source_ids))
    assert result.tool_exchanges[0]["tool"].content["results"]
    assert any(
        "another session" in item["content"]
        for item in result.tool_exchanges[0]["tool"].content["results"]
    ) is False
    engine.dispose()


def test_cross_window_citation_legality(tmp_path):
    # Add an unrelated hidden teacher that must exist but is not injected.
    messages = [
        ("user", "Fact: X links to Y."),
        ("user", "Fact: A links to B."),
        ("assistant", "Understood."),
        ("user", "Fact: B links to C."),
        ("assistant", "Understood."),
        ("user", "Correction: B does not link to C."),
        ("assistant", "Understood."),
    ]
    for i in range(10):
        messages.extend(
            [
                ("user", f"Fact: object_{i} links to object_{i + 1}."),
                ("assistant", "Understood."),
            ]
        )
    fixture = _fixture(messages, "Does B link to C?")
    result = _run(
        fixture,
        client=FakeToolClient(
            [AssistantTurn(content="B does not link to C [message 4] [message 1].")]
        ),
        tmp_path=tmp_path,
    )
    # message 1 is X-Y; it exists but is not injected because query B C does not
    # retrieve it and it is not a visible teacher source.
    assert result.status == "completed"
    assert 4 in result.injected_source_ids
    assert 1 not in result.injected_source_ids
    assert result.invalid_source_ids == [1]


def test_ac_negative_is_not_turned_into_an_answer(tmp_path):
    fixture = _hidden_fixture("Does A link to C?")
    result = _run(
        fixture,
        client=FakeToolClient(
            [AssistantTurn(content="A-C is not established by the retrieved teaching.")]
        ),
        tmp_path=tmp_path,
    )
    assert result.status == "completed"
    assert result.programmatic_retrieval_attempts == 1
    assert result.model_calls == 1
    assert "A links to C" not in result.answer
    assert result.answer_quality == "missing_citations"


def test_parser_rejects_instruction_and_unsupported_shapes(tmp_path):
    fixture = _hidden_fixture()
    engine, path, factory, sid, cid = _seed(
        tmp_path, fixture.messages, fixture.current
    )
    db = factory()
    boundary = load_window_boundary(db, session_id=sid, current_message_id=cid)
    for current in [
        'Do not answer "Does B link to C?"',
        'Compare the original and current version of "red things".',
        "I teach you this fact: D links to E. What is the current B-C?",
    ]:
        decision = decide(boundary, current_content=current)
        assert decision.action == "clarify", (current, decision)
        assert decision.reason == "unsupported_pattern"
    db.close()
    engine.dispose()


def test_mixed_supported_question_keeps_full_current(tmp_path):
    fixture = _hidden_fixture(
        "I teach you this fact: D links to E. Does B link to C?"
    )
    engine, path, factory, sid, cid = _seed(
        tmp_path, fixture.messages, fixture.current
    )
    db = factory()
    boundary = load_window_boundary(db, session_id=sid, current_message_id=cid)
    decision = decide(boundary, current_content=fixture.current)
    assert decision.action == "retrieve"
    assert decision.reason == "hidden_retrieval"
    assert decision.query == "b c"
    db.close()
    engine.dispose()

    result = _run(
        fixture,
        client=FakeToolClient(
            [AssistantTurn(content="B does not link to C [message 5].")]
        ),
        tmp_path=tmp_path,
    )
    assert result.messages[-1]["content"] == fixture.current


def test_reverse_question_does_not_use_visible_bc_complete(tmp_path):
    messages = [
        ("user", "Fact: X links to Y."),
        ("assistant", "Understood."),
    ]
    for i in range(9):
        messages.extend(
            [
                ("user", f"Fact: object_{i} links to object_{i + 1}."),
                ("assistant", "Understood."),
            ]
        )
    messages.extend(
        [
            ("user", "Fact: B links to C."),
            ("assistant", "Understood."),
        ]
    )
    fixture = _fixture(messages, "Does C link to B?")
    engine, path, factory, sid, cid = _seed(
        tmp_path, fixture.messages, fixture.current
    )
    db = factory()
    boundary = load_window_boundary(db, session_id=sid, current_message_id=cid)
    decision = decide(boundary, current_content=fixture.current)
    assert decision.action == "retrieve"
    assert decision.reason == "hidden_retrieval"
    assert decision.query == "c b"
    db.close()
    engine.dispose()


def test_budget_exhausted_before_programmatic_send_has_no_injected_sources(tmp_path):
    from experiments.tool_search_history.deepseek_client import (
        DeepSeekBudgetExhausted,
    )

    class BudgetClient:
        def __init__(self):
            self.last_call = None
            self.message_batches = []

        async def generate_turn(self, messages, **kwargs):
            self.last_call = {
                "phase": kwargs.get("phase"),
                "tool_choice": kwargs.get("tool_choice"),
                "request_sent": False,
                "error_stage": "budget",
                "error_code": "budget_exhausted",
            }
            raise DeepSeekBudgetExhausted("budget_exhausted")

        def take_last_call(self):
            record = self.last_call
            self.last_call = None
            return record

    result = _run(_hidden_fixture(), client=BudgetClient(), tmp_path=tmp_path)
    assert result.status == "failed"
    assert result.programmatic_status == "protocol_failed"
    assert result.model_calls == 0
    assert result.http_requests == 0
    assert result.injected_source_ids == []
    assert result.retrieved_source_ids == [1, 3, 5]


def test_after_send_failure_keeps_injected_sources(tmp_path):
    fixture = _hidden_fixture()
    seen = []

    def handler(request: httpx.Request):
        seen.append(json.loads(request.content))
        return httpx.Response(500, text="upstream failure")

    engine = create_engine(f"sqlite:///{tmp_path / 'after-send.sqlite3'}")
    create_tables(engine)
    factory = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    result = asyncio.run(
        run_condition(
            session_factory=factory,
            database_path=tmp_path / "after-send.sqlite3",
            asset_root=ASSET_ROOT,
            condition="programmatic",
            fixture=fixture,
            client=DeepSeekToolClient(
                api_key="test-key-not-real",
                transport=httpx.MockTransport(handler),
            ),
        )
    )
    assert result.status == "failed"
    assert result.model_calls == 1
    assert result.http_requests == 1
    assert result.tool_calls == 1
    assert {3, 5}.issubset(set(result.injected_source_ids))
    assert len(seen) == 1
    engine.dispose()


def test_assistant_only_window_retrieves_and_missing_citation_is_scored(tmp_path):
    messages = [
        ("user", "Fact: B links to C."),
        ("assistant", "Understood."),
        ("user", "Correction: B does not link to C."),
        ("assistant", "Understood."),
    ]
    messages.extend([("assistant", "Understood.")] * 20)
    fixture = _fixture(messages, "Does B link to C?")
    seen = []

    def handler(request: httpx.Request):
        seen.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-missing-citation",
                "model": "deepseek-flash",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "I found the relevant teaching but will not cite it.",
                        },
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    engine = create_engine(f"sqlite:///{tmp_path / 'missing-citation.sqlite3'}")
    create_tables(engine)
    factory = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    result = asyncio.run(
        run_condition(
            session_factory=factory,
            database_path=tmp_path / "missing-citation.sqlite3",
            asset_root=ASSET_ROOT,
            condition="programmatic",
            fixture=fixture,
            client=DeepSeekToolClient(
                api_key="test-key-not-real",
                transport=httpx.MockTransport(handler),
            ),
        )
    )
    assert result.status == "completed"
    assert result.programmatic_retrieval_attempts == 1
    assert result.tool_calls == 1
    assert result.model_calls == 1
    assert result.http_requests == 1
    assert result.visible_source_ids == []
    assert {1, 3}.issubset(set(result.retrieved_source_ids))
    assert {1, 3}.issubset(set(result.injected_source_ids))
    assert result.source_ids == []
    assert result.answer_quality == "missing_citations"
    assert "tools" not in seen[0]
    assert "tool_choice" not in seen[0]
    engine.dispose()


def test_unsupported_shapes_do_not_retrieve_or_call_model(tmp_path):
    for current in [
        'Do not answer "Does B link to C?"',
        'Compare the original and current version of "red things".',
        "I teach you this fact: D links to E. What is the current B-C?",
    ]:
        fixture = _hidden_fixture(current)
        case_dir = tmp_path / current[:8].replace(" ", "_").replace('"', "")
        case_dir.mkdir()
        result = _run(fixture, tmp_path=case_dir)
        assert result.status == "completed"
        assert result.programmatic_status == "clarification_required"
        assert result.programmatic_retrieval_attempts == 0
        assert result.tool_calls == 0
        assert result.model_calls == 0
        assert result.http_requests == 0
        assert result.injected_source_ids == []


def test_supported_compare_generates_valid_query(tmp_path):
    fixture = _hidden_fixture(
        'Compare the original and current version of "B links to C".'
    )
    engine, path, factory, sid, cid = _seed(
        tmp_path, fixture.messages, fixture.current
    )
    db = factory()
    boundary = load_window_boundary(db, session_id=sid, current_message_id=cid)
    decision = decide(boundary, current_content=fixture.current)
    assert decision.action == "retrieve"
    assert decision.reason == "history_compare"
    assert decision.query == "b c"
    db.close()
    engine.dispose()
