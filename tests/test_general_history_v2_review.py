"""Local-deadline regressions against the original decision-budget fixture."""

from __future__ import annotations

import asyncio

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.general_history.engine import GeneralSearchEngine
from experiments.general_tool_search.decision_budget import (
    DEFAULT_CASES,
    load_cases,
    seed_cases,
)


class SlowDecision:
    synthetic = True

    def __init__(self):
        self.calls = 0

    async def generate_decision(self, messages, *, deadline):
        self.calls += 1
        await asyncio.sleep(0.2)
        return "{}"


class ClarityFinal:
    synthetic = True

    def __init__(self):
        self.calls = 0
        self.messages = None

    async def generate_final(self, messages, *, deadline=None):
        self.calls += 1
        self.messages = messages
        return "Which process do you mean?"


def _fixture_case(case_id):
    cases = load_cases(DEFAULT_CASES, allow_synthetic_session51=True)
    return [case for case in cases if case.case_id == case_id]


def _engine(tmp_path, case, decision, final):
    path = tmp_path / f"{case.case_id}.sqlite3"
    contexts = seed_cases([case], path)
    context = contexts[case.case_id]
    sa_engine = create_engine(f"sqlite:///{path}")
    factory = sessionmaker(bind=sa_engine, autoflush=False, expire_on_commit=False)
    db = factory()
    engine = GeneralSearchEngine(
        db,
        session_id=context.session_id,
        current_message_id=context.current_id,
        decision_client=decision,
        final_client=final,
        database_path=str(path),
        decision_timeout=0.05,
        overall_timeout=2.0,
    )
    return engine, db, sa_engine


def test_original_ambiguous_fixture_local_deadline_clarifies(tmp_path):
    case = _fixture_case("ambiguous_reference")[0]
    decision = SlowDecision()
    final = ClarityFinal()
    engine, db, sa_engine = _engine(tmp_path, case, decision, final)
    try:
        result = asyncio.run(engine.run())
        assert result.status == "completed_clarify"
        assert result.counts.decision_attempts == 1
        assert result.counts.retrieval_attempts == 0
        assert result.counts.final_attempts == 1
        assert decision.calls == 1
        assert final.calls == 1
        assert "exactly one focused clarification" in final.messages[0]["content"]
    finally:
        db.close()
        sa_engine.dispose()


def test_unique_object_local_deadline_is_one_retrieval(tmp_path):
    from tests.test_general_history_endpoint import _db_factory, _seed

    engine_db, factory, path = _db_factory(tmp_path)
    session_id, current_id = _seed(
        factory,
        ["The alpha bridge is painted green."],
        [f"I teach you filler fact {index}." for index in range(20)],
        question="What color is the alpha bridge?",
    )
    decision = SlowDecision()
    final = ClarityFinal()
    db = factory()
    engine = GeneralSearchEngine(
        db,
        session_id=session_id,
        current_message_id=current_id,
        decision_client=decision,
        final_client=final,
        database_path=str(path),
        decision_timeout=0.05,
        overall_timeout=2.0,
    )
    try:
        result = asyncio.run(engine.run())
        assert result.counts.decision_attempts == 1
        assert result.counts.retrieval_attempts == 1
        assert result.counts.final_attempts == 1
        assert decision.calls == 1
        assert final.calls == 1
        assert result.retrieved_source_ids == [1]
    finally:
        db.close()
        engine_db.dispose()
