"""Isolated tests for the general teaching-history search experiment."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.models import Base, ChatSession, Message
from backend.programmatic_history.trigger import load_window_boundary
from experiments.general_tool_search.anchors import Anchor
from experiments.general_tool_search.engine import (
    FakeDecisionClient,
    FakeFinalClient,
    GeneralSearchEngine,
)
from experiments.general_tool_search.search import SearchError, SearchTimeout

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "general_cases.json"


def _decision(need: bool, query: str = "", ids=None, reason: str = "ordinary_recall") -> str:
    return json.dumps(
        {
            "need_history": need,
            "query": query,
            "visible_source_ids": ids or [],
            "reason": reason,
        }
    )


class Case:
    def __init__(
        self, name: str, tmp_path: Path, *, hidden=None, visible=None, question=None
    ):
        data = json.loads(FIXTURE.read_text())
        case = data["cases"][name]
        self.hidden = list(hidden if hidden is not None else case["hidden"])
        self.visible = list(visible if visible is not None else data["visible"])
        self.question = question or case["question"]
        self.engine_db = create_engine(f"sqlite:///{tmp_path / 'case.sqlite3'}")
        Base.metadata.create_all(self.engine_db)
        factory = sessionmaker(
            bind=self.engine_db, autoflush=False, expire_on_commit=False
        )
        self.db = factory()
        session = ChatSession(title=name)
        self.db.add(session)
        self.db.flush()
        for content in self.hidden + self.visible:
            self.db.add(self._message(session.id, content))
        self.db.commit()
        self.current = self._message(session.id, self.question)
        self.db.add(self.current)
        self.db.commit()
        self.db.refresh(self.current)
        self.session_id = session.id
        self.current_id = self.current.id
        self.visible_ids = list(
            range(len(self.hidden) + 1, len(self.hidden) + len(self.visible) + 1)
        )
        boundary = load_window_boundary(
            self.db,
            session_id=self.session_id,
            current_message_id=self.current_id,
        )
        assert boundary.known
        assert boundary.window_overflow is bool(self.hidden)
        self.boundary_visible_ids = [message.id for message in boundary.visible_teacher]

    @staticmethod
    def _message(session_id: int, content: str) -> Message:
        return Message(
            session_id=session_id,
            role="user",
            content=content,
            interaction_mode_snapshot="receive_teaching",
            prompt_version_snapshot="test",
            llm_profile_id_snapshot="fake",
            llm_profile_kind_snapshot="fake",
            llm_model_snapshot="fake",
        )

    def run(
        self, decision=None, final=None, *, limits=None,
        overall_timeout=5.0, decision_timeout=0.7,
    ):
        self.decision_client = (
            decision if decision is not None else FakeDecisionClient([])
        )
        self.final_client = final or FakeFinalClient(["I cannot verify from the available evidence."])
        self.engine = GeneralSearchEngine(
            self.db,
            session_id=self.session_id,
            current_message_id=self.current_id,
            decision_client=self.decision_client,
            final_client=self.final_client,
            database_path=self.engine_db.url.database,
            search_limits=limits,
            overall_timeout=overall_timeout,
            decision_timeout=decision_timeout,
            final_system_prompt="test system",
        )
        return asyncio.run(self.engine.run())

    def close(self):
        self.db.close()
        self.engine_db.dispose()


def _run_case(tmp_path, name, decision=None, final=None, **kwargs):
    case = Case(name, tmp_path, **kwargs)
    try:
        return case.run(decision=decision, final=final)
    finally:
        case.close()


def test_plain_and_simple_confirmation_do_not_retrieve(tmp_path):
    result = _run_case(
        tmp_path,
        "no_hidden",
        final=FakeFinalClient(["The delta fold is [message 7]."]),
    )
    assert result.status == "completed_direct"
    assert result.counts.decision_attempts == 0
    assert result.counts.retrieval_attempts == 0
    assert result.counts.final_requests == 1
    assert result.retrieved_source_ids == []


def test_no_hidden_teacher_returns_directly(tmp_path):
    result = _run_case(
        tmp_path,
        "no_hidden",
        final=FakeFinalClient(["I only have visible evidence [message 7]."]),
    )
    assert result.status == "completed_direct"
    assert result.counts.decision_attempts == 0
    assert result.counts.retrieval_attempts == 0
    assert 7 in result.visible_source_ids


def test_process_correction_and_reason_are_returned(tmp_path):
    result = _run_case(
        tmp_path,
        "process",
        decision=FakeDecisionClient([_decision(True, "process P7 tighten screw")]),
        final=FakeFinalClient(
            ["The current order is [message 5]; the reason is [message 4]."]
        ),
    )
    assert result.status == "completed_retrieved"
    assert result.counts.as_dict() == {
        "decision_attempts": 1,
        "decision_requests": 1,
        "retrieval_attempts": 1,
        "retrieval_requests": 1,
        "final_attempts": 1,
        "final_requests": 1,
        "decision_http_requests": 0,
        "retrieval_http_requests": 0,
        "final_http_requests": 0,
    }
    assert {4, 5}.issubset(set(result.retrieved_source_ids))
    assert result.cited_source_ids == [4, 5]
    assert result.invalid_source_ids == []


def test_rule_and_exception_are_not_displaced_by_common_words(tmp_path):
    result = _run_case(
        tmp_path,
        "rule_exception",
        decision=FakeDecisionClient([_decision(True, "division zero")]),
        final=FakeFinalClient(
            ["The rule is [message 1]; the exception is [message 2]."]
        ),
    )
    assert result.status == "completed_retrieved"
    assert {1, 2}.issubset(set(result.retrieved_source_ids))
    assert result.cited_source_ids == [1, 2]
    assert result.invalid_source_ids == []


def test_formula_comparison_keeps_original_and_correction(tmp_path):
    result = _run_case(
        tmp_path,
        "formula_compare",
        final=FakeFinalClient(
            ["The original was [message 1]; the replacement is [message 2]."]
        ),
    )
    assert result.status == "completed_retrieved"
    assert result.counts.decision_attempts == 0
    assert result.counts.retrieval_attempts == 1
    assert {1, 2}.issubset(set(result.retrieved_source_ids))
    assert len(result.retrieved_source_ids) <= 3
    assert result.cited_source_ids == [1, 2]
    assert result.synthetic is True
    assert result.counts.final_http_requests == 0
    injected = json.dumps(result.messages, ensure_ascii=False)
    assert "1/n" in injected and "2/n" in injected


def test_formula_anchor_has_complete_token_boundary():
    anchor = Anchor("formula", "1/n", "1/n")
    assert anchor_matches_formula("Formula: 1/n.", anchor) is True
    assert anchor_matches_formula("Formula: 11/n.", anchor) is False
    assert anchor_matches_formula("Formula: 111/n.", anchor) is False


def anchor_matches_formula(content, anchor):
    from experiments.general_tool_search.anchors import anchor_matches

    return anchor_matches(content, anchor)


def test_explicit_alpha_term_finds_hidden_teacher(tmp_path):
    result = _run_case(
        tmp_path,
        "alpha_explicit",
        final=FakeFinalClient(["We called it the alpha bridge [message 1]."]),
    )
    assert result.status == "completed_retrieved"
    assert result.counts.decision_attempts == 0
    assert 1 in result.retrieved_source_ids


def test_explicit_visible_theme_still_searches_for_original(tmp_path):
    case = Case(
        "visible_explicit",
        tmp_path,
        hidden=[
            "We originally called the delta fold the old fold.",
            "Process P7 step one is in the old notes.",
        ],
    )
    try:
        result = case.run(
            final=FakeFinalClient(["The original was [message 1]."])
        )
    finally:
        case.close()
    assert result.status == "completed_retrieved"
    assert result.counts.decision_requests == 0
    assert result.counts.retrieval_requests == 1
    assert 1 in result.retrieved_source_ids
    assert result.cited_source_ids == [1]


def test_source_question_for_hidden_sigma_rule(tmp_path):
    result = _run_case(
        tmp_path,
        "sigma_source",
        final=FakeFinalClient(["The source is [message 1]."]),
    )
    assert result.status == "completed_retrieved"
    assert 1 in result.retrieved_source_ids
    assert result.cited_source_ids == [1]


def test_ambiguous_decision_ends_with_one_final_clarification(tmp_path):
    result = _run_case(
        tmp_path,
        "ambiguous",
        decision=FakeDecisionClient(
            [_decision(False, query="", ids=[], reason="ambiguous")]
        ),
        final=FakeFinalClient(["Which rule do you mean?"]),
    )
    assert result.status == "completed_clarify"
    assert result.counts.as_dict() == {
        "decision_attempts": 1,
        "decision_requests": 1,
        "retrieval_attempts": 0,
        "retrieval_requests": 0,
        "final_attempts": 1,
        "final_requests": 1,
        "decision_http_requests": 0,
        "retrieval_http_requests": 0,
        "final_http_requests": 0,
    }
    system = result.messages[0]["content"]
    assert "exactly one focused clarification" in system


def test_ordinary_foobar_recall_is_empty_and_does_not_fabricate(tmp_path):
    result = _run_case(
        tmp_path,
        "foobar",
        decision=FakeDecisionClient([_decision(True, "Foobar protocol")]),
        final=FakeFinalClient(["I found no teacher evidence for Foobar protocol."]),
    )
    assert result.status == "completed_empty"
    assert result.counts.decision_requests == 1
    assert result.counts.retrieval_requests == 1
    assert result.counts.final_requests == 1
    assert result.retrieved_source_ids == []


def test_cross_session_scope_does_not_leak(tmp_path):
    case = Case("foobar", tmp_path)
    other = ChatSession(title="other")
    case.db.add(other)
    case.db.flush()
    for content in [
        "The original source of the Foobar protocol is the other handbook.",
        "The Foobar protocol is in the other session.",
    ]:
        case.db.add(Case._message(other.id, content))
    case.db.commit()
    try:
        result = case.run(
            decision=FakeDecisionClient([_decision(True, "Foobar protocol")]),
            final=FakeFinalClient(["No Foobar evidence in this session."]),
        )
    finally:
        case.close()
    assert result.status == "completed_empty"
    assert result.retrieved_source_ids == []


def test_invalid_decision_stops_before_retrieval_or_final(tmp_path):
    result = _run_case(
        tmp_path,
        "invalid_decision",
        decision=FakeDecisionClient(["not json"]),
        final=FakeFinalClient(["must not be called"]),
    )
    assert result.status == "decision_invalid"
    assert result.counts.decision_attempts == 1
    assert result.counts.retrieval_attempts == 0
    assert result.counts.final_attempts == 0


def test_decision_timeout_with_anchor_does_one_search_and_one_final(tmp_path):
    case = Case("alpha", tmp_path)
    try:
        result = case.run(
            decision=FakeDecisionClient([None]),
            final=FakeFinalClient(["The alpha bridge is [message 1]."]),
        )
    finally:
        case.close()
    assert result.status == "completed_retrieved"
    assert result.counts.decision_attempts == 1
    assert result.counts.retrieval_attempts == 1
    assert result.counts.final_requests == 1


def test_decision_timeout_without_anchor_clarifies(tmp_path):
    result = _run_case(
        tmp_path,
        "rule_only",
        decision=FakeDecisionClient([None]),
        final=FakeFinalClient(["What rule do you mean?"]),
    )
    assert result.status == "completed_clarify"
    assert result.counts.decision_attempts == 1
    assert result.counts.retrieval_attempts == 0
    assert result.counts.final_requests == 1


def test_search_timeout_skips_final(tmp_path):
    case = Case("process", tmp_path)
    case.decision_client = FakeDecisionClient(
        [_decision(True, "process P7 tighten screw")]
    )
    case.final_client = FakeFinalClient(["must not be called"])
    case.engine = GeneralSearchEngine(
        case.db,
        session_id=case.session_id,
        current_message_id=case.current_id,
        decision_client=case.decision_client,
        final_client=case.final_client,
        database_path=case.engine_db.url.database,
        final_system_prompt="test system",
    )

    async def timeout(**kwargs):
        raise SearchTimeout("search_deadline_exceeded")

    case.engine.search.search = timeout
    try:
        result = asyncio.run(case.engine.run())
    finally:
        case.close()
    assert result.status == "retrieval_timeout"
    assert result.counts.final_requests == 0


def test_search_error_skips_final(tmp_path):
    case = Case("process", tmp_path)
    case.decision_client = FakeDecisionClient(
        [_decision(True, "process P7 tighten screw")]
    )
    case.final_client = FakeFinalClient(["must not be called"])
    case.engine = GeneralSearchEngine(
        case.db,
        session_id=case.session_id,
        current_message_id=case.current_id,
        decision_client=case.decision_client,
        final_client=case.final_client,
        database_path=case.engine_db.url.database,
        final_system_prompt="test system",
    )

    async def error(**kwargs):
        raise SearchError("search_failed")

    case.engine.search.search = error
    try:
        result = asyncio.run(case.engine.run())
    finally:
        case.close()
    assert result.status == "retrieval_error"
    assert result.counts.final_requests == 0


def test_final_failure_is_reported_without_retry(tmp_path):
    result = _run_case(
        tmp_path,
        "no_hidden",
        final=FakeFinalClient([]),
    )
    assert result.status == "final_failed"
    assert result.counts.final_attempts == 1
    assert result.counts.final_requests == 1


def test_decision_visible_ids_must_belong_to_window(tmp_path):
    result = _run_case(
        tmp_path,
        "visible_complete",
        decision=FakeDecisionClient(
            [_decision(False, ids=[99], reason="visible_complete")]
        ),
        final=FakeFinalClient(["must not be called"]),
    )
    assert result.status == "decision_invalid"
    assert result.counts.final_attempts == 0


def test_pure_teaching_and_simple_confirmation_do_not_retrieve(tmp_path):
    for question in (
        "I teach you this fact: the sky is blue.",
        "OK.",
    ):
        case = Case("process", tmp_path, question=question)
        try:
            result = case.run(final=FakeFinalClient(["Noted."]))
        finally:
            case.close()
        assert result.status == "completed_direct"
        assert result.counts.as_dict()["decision_attempts"] == 0
        assert result.counts.as_dict()["retrieval_attempts"] == 0
        assert result.counts.as_dict()["final_requests"] == 1


def test_mixed_teaching_and_question_still_uses_visible_answer(tmp_path):
    case = Case(
        "visible_complete",
        tmp_path,
        question=(
            "I teach you this fact: the delta fold is blue. "
            "What did I call it?"
        ),
    )
    try:
        result = case.run(
            decision=FakeDecisionClient(
                [_decision(False, ids=[7], reason="visible_complete")]
            ),
            final=FakeFinalClient(["The delta fold [message 7]."]),
        )
    finally:
        case.close()
    assert result.status == "completed_visible_complete"
    assert result.counts.decision_requests == 1
    assert result.counts.retrieval_attempts == 0


def test_anchor_search_truncates_large_content_and_keeps_flag(tmp_path):
    from experiments.general_tool_search.search import AnchorSearch, SearchLimits
    from experiments.general_tool_search.anchors import Anchor

    hidden = [
        "alpha distractor " + str(i) for i in range(1, 6)
    ] + ["alpha " + ("x" * 1200)]
    case = Case("alpha", tmp_path, hidden=hidden)
    try:
        outcome = asyncio.run(
            AnchorSearch(
                case.engine_db.url.database,
                limits=SearchLimits(max_content_chars=100),
            ).search(
                session_id=case.session_id,
                current_message_id=case.current_id,
                visible_start_id=case.visible_ids[0],
                anchors=[Anchor("term", "alpha", "alpha")],
                question="What did I call alpha?",
            )
        )
    finally:
        case.close()
    assert outcome.status == "ok"
    truncated = [item for item in outcome.results if item["truncated"]]
    assert truncated
    assert len(truncated[0]["content"]) == 100
    assert outcome.truncated is True


def test_final_citation_outside_injected_sources_is_invalid(tmp_path):
    result = _run_case(
        tmp_path,
        "no_hidden",
        final=FakeFinalClient(["I found [message 99]."]),
    )
    assert result.status == "completed_direct"
    assert result.invalid_source_ids == [99]


def test_generic_decision_client_error_has_no_final(tmp_path):
    class BrokenDecision:
        async def generate_decision(self, messages, *, deadline):
            raise RuntimeError("synthetic")

    result = _run_case(
        tmp_path,
        "foobar",
        decision=BrokenDecision(),
        final=FakeFinalClient(["must not be called"]),
    )
    assert result.status == "decision_failed"
    assert result.counts.decision_requests == 1
    assert result.counts.final_requests == 0


def test_boundary_rejects_non_user_current_message(tmp_path):
    case = Case("alpha", tmp_path)
    case.current.role = "assistant"
    case.db.commit()
    try:
        result = case.run(decision=FakeDecisionClient([_decision(True, "alpha")]))
    finally:
        case.close()
    assert result.status == "boundary_invalid"
    assert result.counts.decision_attempts == 0
    assert result.counts.final_attempts == 0


def test_visible_complete_ordinary_question_has_no_search(tmp_path):
    result = _run_case(
        tmp_path,
        "visible_complete",
        decision=FakeDecisionClient(
            [_decision(False, ids=[7], reason="visible_complete")]
        ),
        final=FakeFinalClient(["The delta fold [message 7]."]),
    )
    assert result.status == "completed_visible_complete"
    assert result.counts.as_dict() == {
        "decision_attempts": 1,
        "decision_requests": 1,
        "retrieval_attempts": 0,
        "retrieval_requests": 0,
        "final_attempts": 1,
        "final_requests": 1,
        "decision_http_requests": 0,
        "retrieval_http_requests": 0,
        "final_http_requests": 0,
    }
    assert result.cited_source_ids == [7]


def test_legacy_relation_branch_remains_direct(tmp_path):
    visible = ["A links to B."] + json.loads(FIXTURE.read_text())["visible"][1:]
    case = Case(
        "alpha",
        tmp_path,
        visible=visible,
        question="Does A link to B?",
    )
    try:
        result = case.run(final=FakeFinalClient(["That is a relationship question."]))
    finally:
        case.close()
    assert result.status == "completed_direct"
    assert result.counts.decision_attempts == 0
    assert result.counts.retrieval_attempts == 0
    assert result.counts.final_requests == 1


def test_decision_request_contains_visible_evidence_and_current_question(tmp_path):
    visible = ["Process P7: current order is step one."] + json.loads(
        FIXTURE.read_text()
    )["visible"][1:]
    case = Case(
        "process",
        tmp_path,
        visible=visible,
        hidden=["Process P7: original order is step zero."],
        question="What is the current order of the process?",
    )
    decision = FakeDecisionClient([_decision(True, "P7 order")])
    try:
        result = case.run(
            decision=decision,
            final=FakeFinalClient(["The current order is [message 1]."]),
        )
    finally:
        case.close()
    assert result.status == "completed_retrieved"
    assert result.counts.decision_requests == 1
    assert result.counts.retrieval_requests == 1
    assert result.retrieved_source_ids == [1]
    assert decision.last_messages is not None
    assert "Process P7" in decision.last_messages[1]["content"]
    assert decision.last_messages[-1]["content"] == "What is the current order of the process?"
    payload = json.loads(decision.last_messages[1]["content"])
    assert payload["type"] == "decision_window"
    assert [item["message_id"] for item in payload["visible_teacher"]] == list(
        range(2, 22)
    )
    assert [item["order"] for item in payload["visible_teacher"]] == list(
        range(1, 21)
    )


def test_unlocated_process_theme_is_not_grounded(tmp_path):
    case = Case(
        "process",
        tmp_path,
        hidden=["Process P7: original order is step zero."],
        question="What is the current order of the process?",
    )
    decision = FakeDecisionClient([_decision(True, "P7 order")])
    try:
        result = case.run(
            decision=decision,
            final=FakeFinalClient(["Which process do you mean?"]),
        )
    finally:
        case.close()
    assert result.status == "completed_clarify"
    assert result.counts.decision_requests == 1
    assert result.counts.retrieval_requests == 0
    assert result.counts.final_requests == 1


def test_null_decision_query_is_invalid(tmp_path):
    raw = json.dumps(
        {
            "need_history": True,
            "query": None,
            "visible_source_ids": [],
            "reason": "ordinary_recall",
        }
    )
    result = _run_case(
        tmp_path,
        "foobar",
        decision=FakeDecisionClient([raw]),
        final=FakeFinalClient(["must not be called"]),
    )
    assert result.status == "decision_invalid"
    assert result.counts.retrieval_requests == 0
    assert result.counts.final_requests == 0


def test_visible_relationship_is_direct_but_hidden_relationship_searches(tmp_path):
    question = "Does alpha link to beta?"
    hidden_case = Case(
        "alpha",
        tmp_path,
        hidden=["alpha links to beta."],
        question=question,
    )
    try:
        hidden_result = hidden_case.run(
            final=FakeFinalClient(["The relation is [message 1]."])
        )
    finally:
        hidden_case.close()
    assert hidden_result.status == "completed_retrieved"
    assert hidden_result.counts.decision_attempts == 0
    assert hidden_result.counts.retrieval_requests == 1
    assert hidden_result.retrieved_source_ids == [1]

    visible = ["alpha links to beta."] + json.loads(
        FIXTURE.read_text()
    )["visible"][1:]
    visible_case = Case(
        "alpha",
        tmp_path,
        visible=visible,
        hidden=["Process P7 step one is in the old notes."],
        question=question,
    )
    try:
        visible_result = visible_case.run(
            final=FakeFinalClient(["The relation is [message 1]."])
        )
    finally:
        visible_case.close()
    assert visible_result.status == "completed_direct"
    assert visible_result.counts.decision_attempts == 0
    assert visible_result.counts.retrieval_requests == 0
    assert visible_result.counts.final_requests == 1


def test_comparison_without_a_pair_sets_pair_missing(tmp_path):
    case = Case(
        "formula_compare",
        tmp_path,
        hidden=["Correction: For resistance R, Formula: R = 2/n."],
    )
    try:
        result = case.run(
            final=FakeFinalClient(["Only the replacement is available [message 1]."])
        )
    finally:
        case.close()
    assert result.status == "completed_retrieved"
    assert result.search is not None
    assert result.search.pair_missing is True
    assert result.search.corrections_truncated is False
    assert result.retrieved_source_ids == [1]
    assert '"pair_missing": true' in result.messages[1]["content"]


def test_blank_final_is_failure_without_retry(tmp_path):
    result = _run_case(
        tmp_path,
        "no_hidden",
        final=FakeFinalClient(["   "]),
    )
    assert result.status == "final_failed"
    assert result.counts.final_attempts == 1
    assert result.counts.final_requests == 1
    assert result.error == "empty_answer"


def test_final_generation_timeout_has_no_retry(tmp_path):
    class SlowFinal:
        def __init__(self):
            self.calls = 0

        async def generate_final(self, messages, *, deadline=None):
            self.calls += 1
            await asyncio.sleep(0.2)
            return "too late"

    case = Case("no_hidden", tmp_path)
    slow = SlowFinal()
    try:
        result = case.run(final=slow, overall_timeout=0.05)
    finally:
        case.close()
    assert result.status == "final_timeout"
    assert slow.calls == 1
    assert result.counts.final_attempts == 1
    assert result.counts.final_requests == 1


def test_fake_results_are_marked_synthetic_with_zero_http(tmp_path):
    result = _run_case(
        tmp_path,
        "no_hidden",
        final=FakeFinalClient(["Visible only [message 7]."]),
    )
    assert result.synthetic is True
    assert result.counts.decision_http_requests == 0
    assert result.counts.retrieval_http_requests == 0
    assert result.counts.final_http_requests == 0


def test_hidden_relationship_comparison_searches(tmp_path):
    case = Case(
        "alpha",
        tmp_path,
        hidden=["alpha links to beta."],
        question='Compare the original and current version of "alpha links to beta".',
    )
    try:
        result = case.run(
            final=FakeFinalClient(["The earlier evidence is [message 1]."])
        )
    finally:
        case.close()
    assert result.status == "completed_retrieved"
    assert result.counts.decision_attempts == 0
    assert result.counts.retrieval_requests == 1
    assert result.counts.final_requests == 1
    assert result.retrieved_source_ids == [1]


def test_explicit_without_anchor_asks_one_clarification_without_search(tmp_path):
    case = Case(
        "alpha",
        tmp_path,
        hidden=["We call the green shape the alpha bridge."],
        question="What did I originally call it?",
    )
    try:
        result = case.run(final=FakeFinalClient(["Which item do you mean?"]))
    finally:
        case.close()
    assert result.status == "completed_clarify"
    assert result.counts.decision_attempts == 0
    assert result.counts.retrieval_requests == 0
    assert result.counts.final_requests == 1
    assert "exactly one focused clarification" in result.messages[0]["content"]


def test_reversed_visible_relation_does_not_suppress_hidden_search(tmp_path):
    visible = ["B links to C."] + json.loads(FIXTURE.read_text())["visible"][1:]
    case = Case(
        "alpha",
        tmp_path,
        visible=visible,
        hidden=["C links to B."],
        question="Does C link to B?",
    )
    try:
        result = case.run(
            final=FakeFinalClient(["The hidden relation is [message 1]."])
        )
    finally:
        case.close()
    assert result.status == "completed_retrieved"
    assert result.counts.decision_attempts == 0
    assert result.counts.retrieval_requests == 1
    assert result.retrieved_source_ids == [1]


def test_visible_current_relation_with_hidden_original_still_compares(tmp_path):
    visible = ["alpha links to beta."] + json.loads(FIXTURE.read_text())["visible"][1:]
    case = Case(
        "alpha",
        tmp_path,
        visible=visible,
        hidden=["original alpha links to gamma."],
        question='Compare the original and current version of "alpha links to beta".',
    )
    try:
        result = case.run(
            final=FakeFinalClient(["The original is [message 1]."])
        )
    finally:
        case.close()
    assert result.status == "completed_retrieved"
    assert result.counts.decision_attempts == 0
    assert result.counts.retrieval_requests == 1
    assert result.counts.final_requests == 1


def test_non_relation_mention_of_links_uses_generic_decision(tmp_path):
    case = Case(
        "alpha",
        tmp_path,
        hidden=["The word links is used in grammar."],
        question="What does the word links mean?",
    )
    decision = FakeDecisionClient([_decision(True, "links")])
    try:
        result = case.run(
            decision=decision,
            final=FakeFinalClient(["The word links is [message 1]."]),
        )
    finally:
        case.close()
    assert result.status == "completed_retrieved"
    assert result.counts.decision_requests == 1
    assert result.counts.retrieval_requests == 1
    assert result.counts.final_requests == 1


def test_search_uses_phase_one_second_bound_when_turn_has_more(tmp_path):
    import time
    from experiments.general_tool_search.anchors import Anchor
    from experiments.general_tool_search.search import AnchorSearch, SearchLimits

    case = Case("alpha", tmp_path)
    opened = {"count": 0}

    class SlowConnection:
        def __init__(self, path, **kwargs):
            opened["count"] += 1
            self._conn = __import__("sqlite3").connect(path)

        def execute(self, sql, params=()):
            if "SELECT id, content" in sql:
                time.sleep(1.2)
            return self._conn.execute(sql, params)

        def set_progress_handler(self, *args):
            return self._conn.set_progress_handler(*args)

        def close(self):
            self._conn.close()

    search = AnchorSearch(
        case.engine_db.url.database,
        limits=SearchLimits(timeout_seconds=1.0),
        connection_factory=SlowConnection,
    )
    started = time.monotonic()
    try:
        with pytest.raises(SearchTimeout):
            asyncio.run(
                search.search(
                    session_id=case.session_id,
                    current_message_id=case.current_id,
                    visible_start_id=case.visible_ids[0],
                    anchors=[Anchor("term", "alpha", "alpha")],
                    question="What did I call alpha?",
                    deadline=started + 5.0,
                )
            )
    finally:
        case.close()
    assert opened["count"] == 1
    assert search.last_deadline is not None
    assert search.last_deadline - started <= 1.15


def test_search_uses_shorter_overall_deadline_when_turn_is_short(tmp_path):
    import time
    from experiments.general_tool_search.anchors import Anchor
    from experiments.general_tool_search.search import AnchorSearch, SearchLimits

    case = Case("alpha", tmp_path)
    opened = {"count": 0}

    class SlowConnection:
        def __init__(self, path, **kwargs):
            opened["count"] += 1
            self._conn = __import__("sqlite3").connect(path)

        def execute(self, sql, params=()):
            if "SELECT id, content" in sql:
                time.sleep(0.2)
            return self._conn.execute(sql, params)

        def set_progress_handler(self, *args):
            return self._conn.set_progress_handler(*args)

        def close(self):
            self._conn.close()

    search = AnchorSearch(
        case.engine_db.url.database,
        limits=SearchLimits(timeout_seconds=1.0),
        connection_factory=SlowConnection,
    )
    started = time.monotonic()
    try:
        with pytest.raises(SearchTimeout):
            asyncio.run(
                search.search(
                    session_id=case.session_id,
                    current_message_id=case.current_id,
                    visible_start_id=case.visible_ids[0],
                    anchors=[Anchor("term", "alpha", "alpha")],
                    question="What did I call alpha?",
                    deadline=started + 0.02,
                )
            )
    finally:
        case.close()
    assert opened["count"] == 1
    assert search.last_deadline is not None
    assert search.last_deadline - started <= 0.05


def test_search_does_not_open_connection_when_budget_is_exhausted(tmp_path):
    import time
    from experiments.general_tool_search.anchors import Anchor
    from experiments.general_tool_search.search import AnchorSearch, SearchLimits

    case = Case("alpha", tmp_path)
    opened = {"count": 0}

    class ShouldNotOpen:
        def __init__(self, *args, **kwargs):
            opened["count"] += 1
            raise AssertionError("connection must not open")

    search = AnchorSearch(
        case.engine_db.url.database,
        limits=SearchLimits(timeout_seconds=1.0),
        connection_factory=ShouldNotOpen,
    )
    try:
        with pytest.raises(SearchTimeout):
            asyncio.run(
                search.search(
                    session_id=case.session_id,
                    current_message_id=case.current_id,
                    visible_start_id=case.visible_ids[0],
                    anchors=[Anchor("term", "alpha", "alpha")],
                    question="What did I call alpha?",
                    deadline=time.monotonic() - 1.0,
                )
            )
    finally:
        case.close()
    assert opened["count"] == 0


def test_decision_query_over_200_chars_is_invalid(tmp_path):
    result = _run_case(
        tmp_path,
        "foobar",
        decision=FakeDecisionClient([_decision(True, "x" * 201)]),
        final=FakeFinalClient(["must not be called"]),
    )
    assert result.status == "decision_invalid"
    assert result.counts.retrieval_requests == 0
    assert result.counts.final_requests == 0


def test_engine_uses_at_most_six_grounded_anchors(tmp_path):
    from experiments.general_tool_search.search import SearchOutcome

    case = Case(
        "foobar",
        tmp_path,
        hidden=["alpha beta gamma delta epsilon zeta eta."],
        question="alpha beta gamma delta epsilon zeta eta?",
    )
    decision = FakeDecisionClient(
        [_decision(True, "alpha beta gamma delta epsilon zeta eta")]
    )

    class CapturingSearch:
        def __init__(self):
            self.anchors = []

        async def search(self, **kwargs):
            self.anchors = kwargs["anchors"]
            return SearchOutcome(status="empty")

    capture = CapturingSearch()
    try:
        case.decision_client = decision
        case.final_client = FakeFinalClient(["No complete evidence."])
        case.engine = GeneralSearchEngine(
            case.db,
            session_id=case.session_id,
            current_message_id=case.current_id,
            decision_client=case.decision_client,
            final_client=case.final_client,
            database_path=case.engine_db.url.database,
            final_system_prompt="test system",
        )
        case.engine.search = capture
        result = asyncio.run(case.engine.run())
    finally:
        case.close()
    assert result.status == "completed_empty"
    assert len(capture.anchors) == 6
    assert len({anchor.normalized for anchor in capture.anchors}) == 6


def test_actual_search_timeout_skips_final_generation(tmp_path):
    import time
    from experiments.general_tool_search.search import AnchorSearch, SearchLimits

    case = Case("process", tmp_path)
    final = FakeFinalClient(["must not be called"])

    class SlowConnection:
        def __init__(self, path, **kwargs):
            self._conn = __import__("sqlite3").connect(path)

        def execute(self, sql, params=()):
            if "SELECT id, content" in sql:
                time.sleep(0.2)
            return self._conn.execute(sql, params)

        def set_progress_handler(self, *args):
            return self._conn.set_progress_handler(*args)

        def close(self):
            self._conn.close()

    decision = FakeDecisionClient([_decision(True, "process P7 tighten screw")])
    case.decision_client = decision
    case.final_client = final
    case.engine = GeneralSearchEngine(
        case.db,
        session_id=case.session_id,
        current_message_id=case.current_id,
        decision_client=decision,
        final_client=final,
        database_path=case.engine_db.url.database,
        search_limits=SearchLimits(timeout_seconds=0.05),
        overall_timeout=5.0,
        final_system_prompt="test system",
    )
    case.engine.search = AnchorSearch(
        case.engine_db.url.database,
        limits=SearchLimits(timeout_seconds=0.05),
        connection_factory=SlowConnection,
    )
    try:
        result = asyncio.run(case.engine.run())
    finally:
        case.close()
    assert result.status == "retrieval_timeout"
    assert result.counts.final_attempts == 0
    assert final.calls == 0
