"""Formal endpoint coverage for opt-in general teaching-history retrieval."""

from __future__ import annotations

import asyncio
import copy
import json

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import event
from sqlalchemy.orm import Session, sessionmaker

import backend.main as main_module
from backend.chat_service import ChatService
from backend.database import create_database_engine, create_tables, get_db
from backend.general_history.service import (
    GeneralHistoryService,
    GeneralHistorySettings,
)
from backend.interaction_modes import PROGRAMMATIC_HISTORY_PROMPT_VERSION
from backend.reply_language import LANGUAGE_INSTRUCTIONS
from backend.llm_profiles import LLMProfile, LLMProfileRegistry
from backend.main import app
from backend.models import ChatSession, Message
from backend.programmatic_history.service import (
    ProgrammaticHistoryService,
    ProgrammaticSettings,
)


_previous_chat_service = None


class ScriptedModel:
    def __init__(
        self,
        responses: list[object],
        *,
        finish_reason: str = "stop",
        finish_reasons: list[str] | None = None,
    ):
        self.responses = list(responses)
        self.finish_reason = finish_reason
        self.finish_reasons = (
            list(finish_reasons) if finish_reasons is not None else None
        )
        self.calls: list[tuple[list[dict], dict]] = []

    async def generate(self, messages, **kwargs):
        self.calls.append((copy.deepcopy(messages), copy.deepcopy(kwargs)))
        if not self.responses:
            raise RuntimeError("scripted response exhausted")
        value = self.responses.pop(0)
        if isinstance(value, Exception):
            raise value
        diagnostics = kwargs.get("diagnostics")
        if diagnostics is not None:
            reason = self.finish_reason
            if self.finish_reasons is not None and self.finish_reasons:
                reason = self.finish_reasons.pop(0)
            diagnostics.finish_reason = reason
            diagnostics.response_content_chars = len(value)
        return value


class CapturingGeneralService(GeneralHistoryService):
    def __init__(self, settings):
        super().__init__(settings)
        self.last_result = None

    async def run(self, db, **kwargs):
        result = await super().run(db, **kwargs)
        self.last_result = result
        return result


def _db_factory(tmp_path, name="general.sqlite3"):
    path = tmp_path / name
    engine = create_database_engine(f"sqlite:///{path}")
    create_tables(engine)
    factory = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    return engine, factory, path


def _seed(factory, hidden, visible, *, question, mode="receive_teaching"):
    db = factory()
    session = ChatSession(
        title="general",
        interaction_mode=mode,
        llm_profile_id="default",
        llm_model_snapshot="model-x",
    )
    db.add(session)
    db.flush()
    for content in [*hidden, *visible]:
        db.add(
            Message(
                session_id=session.id,
                role="user",
                content=content,
                interaction_mode_snapshot=mode,
                prompt_version_snapshot="receive-teaching-v7+history-context-b1",
                llm_profile_id_snapshot="default",
                llm_profile_kind_snapshot="api",
                llm_model_snapshot="model-x",
            )
        )
    db.commit()
    current = Message(
        session_id=session.id,
        role="user",
        content=question,
        interaction_mode_snapshot=mode,
        prompt_version_snapshot="receive-teaching-v7+history-context-b1",
        llm_profile_id_snapshot="default",
        llm_profile_kind_snapshot="api",
        llm_model_snapshot="model-x",
    )
    db.add(current)
    db.commit()
    db.refresh(current)
    result = (session.id, current.id)
    db.close()
    return result


def _decision(need, query="", ids=(), reason="missing_fact"):
    return json.dumps(
        {
            "need_history": need,
            "query": query,
            "visible_source_ids": list(ids),
            "reason": reason,
        }
    )


def _install(factory, model, *, general=True, fixed=False,
             profile_id="default", profile_model="model-x",
             settings_profile_id="default", settings_model="model-x"):
    global _previous_chat_service
    if _previous_chat_service is None:
        _previous_chat_service = main_module.chat_service
    profile = LLMProfile(
        id=profile_id,
        label="API",
        kind="api",
        model=profile_model,
        client=model,
        is_default=True,
    )
    general_service = CapturingGeneralService(
        GeneralHistorySettings(
            enabled=general,
            profile_id=settings_profile_id,
            model=settings_model,
            decision_timeout=2.0,
            overall_timeout=8.0,
        )
    )
    fixed_service = ProgrammaticHistoryService(
        ProgrammaticSettings(fixed, "default", "model-x")
    )
    main_module.chat_service = ChatService(
        profiles=LLMProfileRegistry([profile]),
        programmatic_history=fixed_service,
        general_history=general_service,
    )

    def override_db():
        db = factory()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_db
    return general_service


def _clear():
    global _previous_chat_service
    app.dependency_overrides.pop(get_db, None)
    if _previous_chat_service is not None:
        main_module.chat_service = _previous_chat_service
        _previous_chat_service = None


def _visible_filler(count=20):
    return [f"I teach you filler fact {index}." for index in range(count)]


def test_window_outside_term_is_retrieved_and_injected(tmp_path):
    engine, factory, _ = _db_factory(tmp_path)
    hidden = ["The Foobar protocol uses a signed token."]
    visible = _visible_filler(20)
    session_id, _ = _seed(
        factory, hidden, visible, question="What is the Foobar protocol?"
    )
    model = ScriptedModel(
        [
            _decision(True, "Foobar token"),
            "It uses a signed token [message 1].",
        ]
    )
    service = _install(factory, model)
    try:
        with TestClient(app) as client:
            response = client.post(
                f"/api/sessions/{session_id}/messages",
                json={"message": "What is the Foobar protocol?"},
            )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["user_message"]["prompt_version_snapshot"] == (
            "receive-teaching-v7+programmatic-history-v1+l1-en"
        )
        assert body["assistant_message"]["prompt_version_snapshot"] == (
            "receive-teaching-v7+programmatic-history-v1+l1-en"
        )
        assert len(model.calls) == 2
        final_system = model.calls[1][0][0]["content"]
        assert final_system.endswith(LANGUAGE_INSTRUCTIONS["en"] + "\n")
        assert final_system.count("Reply in English.") == 1
        decision_messages = model.calls[0][0]
        assert decision_messages[-1]["content"] == "What is the Foobar protocol?"
        assert "visible_teacher" in decision_messages[1]["content"]
        assert "signed token" in model.calls[1][0][1]["content"]
        assert service.last_result is not None
        assert service.last_result.counts.retrieval_attempts == 1
        assert service.last_result.retrieved_source_ids == [1]
        assert "tools" not in model.calls[0][1]
        assert "tools" not in model.calls[1][1]
    finally:
        _clear()
        engine.dispose()


def test_visible_answer_skips_retrieval_and_uses_ordinary_generation(tmp_path):
    engine, factory, _ = _db_factory(tmp_path)
    hidden = _visible_filler(6)
    visible = [*_visible_filler(19), "The Foobar protocol uses a local flag."]
    session_id, _ = _seed(
        factory, hidden, visible, question="What is the Foobar protocol?"
    )
    visible_id = 26
    model = ScriptedModel(
        [
            _decision(False, ids=[visible_id], reason="visible_complete"),
            "It uses a local flag.",
        ]
    )
    service = _install(factory, model)
    try:
        with TestClient(app) as client:
            response = client.post(
                f"/api/sessions/{session_id}/messages",
                json={"message": "What is the Foobar protocol?"},
            )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["assistant_message"]["prompt_version_snapshot"] == (
            "receive-teaching-v7+history-context-b1+l1-en"
        )
        assert service.last_result is not None
        assert service.last_result.status == "completed_visible_complete"
        assert service.last_result.counts.retrieval_attempts == 0
        assert len(model.calls) == 2
    finally:
        _clear()
        engine.dispose()


def test_process_correction_uses_old_order_as_support_not_reason(tmp_path):
    engine, factory, _ = _db_factory(tmp_path)
    hidden = [
        "Process P7: Step one is to loosen the bolt.",
        "Process P7: Step two is to turn the wheel clockwise.",
        "Process P7: Step three is to tighten the screw.",
        "The reason for tightening the screw is to hold the bracket.",
        "Correction: Step three of process P7 must be done after step two; do not tighten the screw before the wheel is turned.",
        "The third step is a safety adjustment.",
    ]
    session_id, _ = _seed(
        factory,
        hidden,
        _visible_filler(20),
        question="What is the current order of process P7, and why do you tighten the screw?",
    )
    model = ScriptedModel(
        [
            _decision(True, "process P7 tighten screw"),
            "The current order is [message 5]; the reason is [message 4].",
        ]
    )
    service = _install(factory, model)
    try:
        with TestClient(app) as client:
            response = client.post(
                f"/api/sessions/{session_id}/messages",
                json={
                    "message": "What is the current order of process P7, and why do you tighten the screw?"
                },
            )
        assert response.status_code == 200, response.text
        result = service.last_result
        assert result is not None
        assert set(result.retrieved_source_ids) >= {3, 4, 5}
        evidence = json.loads(result.messages[1]["content"])
        by_id = {
            item["message_id"]: item
            for item in evidence["retrieved_sources"]
        }
        assert by_id[5]["support_message_id"] == 3
        assert by_id[5]["support_message_id"] != 4
        assert by_id[3]["kind"] == "teaching_match"
        assert by_id[4]["kind"] == "teaching_match"
        assert by_id[4].get("support_message_id") is None
    finally:
        _clear()
        engine.dispose()


def test_invalid_decision_returns_502_without_assistant(tmp_path, caplog):
    engine, factory, _ = _db_factory(tmp_path)
    session_id, question_id = _seed(
        factory,
        ["The Foobar protocol uses a signed token."],
        _visible_filler(20),
        question="What is the Foobar protocol?",
    )
    model = ScriptedModel(["private decision response, not JSON"])
    service = _install(factory, model)
    try:
        with TestClient(app) as client:
            response = client.post(
                f"/api/sessions/{session_id}/messages",
                json={"message": "What is the Foobar protocol?"},
            )
        assert response.status_code == 502
        assert len(model.calls) == 1
        db = factory()
        rows = db.query(Message).filter(Message.session_id == session_id).all()
        assert [row.role for row in rows if row.id == question_id] == ["user"]
        assert not any(row.role == "assistant" for row in rows)
        db.close()
        assert service.last_result is not None
        assert service.last_result.status == "decision_invalid"
        assert "status=decision_invalid error_code=invalid_json" in caplog.text
        assert "private decision response" not in caplog.text
    finally:
        _clear()
        engine.dispose()


def test_switch_off_preserves_ordinary_generation(tmp_path):
    engine, factory, _ = _db_factory(tmp_path)
    session_id, _ = _seed(
        factory,
        ["The Foobar protocol uses a signed token."],
        _visible_filler(20),
        question="What is the Foobar protocol?",
    )
    model = ScriptedModel(["ordinary answer"])
    service = _install(factory, model, general=False)
    try:
        with TestClient(app) as client:
            response = client.post(
                f"/api/sessions/{session_id}/messages",
                json={"message": "What is the Foobar protocol?"},
            )
        assert response.status_code == 200
        assert len(model.calls) == 1
        assert service.last_result is None
    finally:
        _clear()
        engine.dispose()


def test_general_is_not_used_for_corrective_mode(tmp_path):
    engine, factory, _ = _db_factory(tmp_path)
    session_id, _ = _seed(
        factory,
        ["The Foobar protocol uses a signed token."],
        _visible_filler(20),
        question="What is the Foobar protocol?",
        mode="corrective",
    )
    model = ScriptedModel(["corrective answer"])
    service = _install(factory, model)
    try:
        with TestClient(app) as client:
            response = client.post(
                f"/api/sessions/{session_id}/messages",
                json={"message": "What is the Foobar protocol?"},
            )
        assert response.status_code == 200
        assert len(model.calls) == 1
        assert service.last_result is None
    finally:
        _clear()
        engine.dispose()


def test_empty_result_still_returns_one_evidence_answer(tmp_path):
    engine, factory, _ = _db_factory(tmp_path)
    session_id, _ = _seed(
        factory,
        ["I teach you this fact: unrelated material."],
        _visible_filler(20),
        question="What is the Foobar protocol?",
    )
    model = ScriptedModel(
        [
            _decision(True, "Foobar token"),
            "The available teaching does not establish the Foobar protocol.",
        ]
    )
    service = _install(factory, model)
    try:
        with TestClient(app) as client:
            response = client.post(
                f"/api/sessions/{session_id}/messages",
                json={"message": "What is the Foobar protocol?"},
            )
        assert response.status_code == 200, response.text
        assert len(model.calls) == 2
        assert service.last_result is not None
        assert service.last_result.status == "completed_empty"
        assert service.last_result.retrieved_source_ids == []
        assert response.json()["assistant_message"]["prompt_version_snapshot"] == (
            "receive-teaching-v7+programmatic-history-v1+l1-en"
        )
    finally:
        _clear()
        engine.dispose()


def test_cross_session_teacher_is_never_injected(tmp_path):
    engine, factory, _ = _db_factory(tmp_path)
    session_id, _ = _seed(
        factory,
        ["I teach you this fact: unrelated material."],
        _visible_filler(20),
        question="What is the Foobar protocol?",
    )
    # Another session with the matching hidden teacher fact.
    db = factory()
    other = ChatSession(
        title="other", llm_profile_id="default", llm_model_snapshot="model-x"
    )
    db.add(other)
    db.flush()
    for content in [
        "The Foobar protocol uses a secret token.",
        *_visible_filler(20),
    ]:
        db.add(Message(
            session_id=other.id,
            role="user",
            content=content,
            interaction_mode_snapshot="receive_teaching",
            prompt_version_snapshot="receive-teaching-v7+history-context-b1",
            llm_profile_id_snapshot="default",
            llm_profile_kind_snapshot="api",
            llm_model_snapshot="model-x",
        ))
    db.commit()
    db.close()
    model = ScriptedModel(
        [
            _decision(True, "Foobar token"),
            "No relevant teaching was available.",
        ]
    )
    service = _install(factory, model)
    try:
        with TestClient(app) as client:
            response = client.post(
                f"/api/sessions/{session_id}/messages",
                json={"message": "What is the Foobar protocol?"},
            )
        assert response.status_code == 200, response.text
        result = service.last_result
        assert result is not None
        assert result.retrieved_source_ids == []
        assert "secret token" not in result.messages[1]["content"]
        assert "secret token" not in model.calls[1][0][1]["content"]
    finally:
        _clear()
        engine.dispose()


def test_decision_timeout_with_no_anchor_uses_clarifying_final(tmp_path):
    engine, factory, _ = _db_factory(tmp_path)
    session_id, _ = _seed(
        factory,
        ["The Foobar protocol uses a signed token."],
        _visible_filler(20),
        question="What about it?",
    )
    model = ScriptedModel([asyncio.TimeoutError(), "Which topic do you mean?"])
    service = _install(factory, model)
    try:
        with TestClient(app) as client:
            response = client.post(
                f"/api/sessions/{session_id}/messages",
                json={"message": "What about it?"},
            )
        assert response.status_code == 200, response.text
        result = service.last_result
        assert result is not None
        assert result.status == "completed_clarify"
        assert result.counts.decision_attempts == 1
        assert result.counts.retrieval_attempts == 0
        assert response.json()["user_message"]["prompt_version_snapshot"] == (
            "receive-teaching-v7+clarify-v1+l1-en"
        )
        assert response.json()["assistant_message"]["prompt_version_snapshot"] == (
            "receive-teaching-v7+clarify-v1+l1-en"
        )
        assert len(model.calls) == 2
        clarify_system = model.calls[1][0][0]["content"]
        assert clarify_system.endswith(LANGUAGE_INSTRUCTIONS["en"] + "\n")
        assert clarify_system.count("Reply in English.") == 1
    finally:
        _clear()
        engine.dispose()


def test_retrieval_timeout_returns_504_without_assistant(tmp_path, monkeypatch):
    from backend.general_history import engine as engine_module

    engine, factory, _ = _db_factory(tmp_path)
    session_id, question_id = _seed(
        factory,
        ["The Foobar protocol uses a signed token."],
        _visible_filler(20),
        question="What is the Foobar protocol?",
    )
    model = ScriptedModel([_decision(True, "Foobar token")])
    service = _install(factory, model)

    async def fail_search(*args, **kwargs):
        raise engine_module.SearchTimeout("forced")

    monkeypatch.setattr(engine_module.AnchorSearch, "search", fail_search)
    try:
        with TestClient(app) as client:
            response = client.post(
                f"/api/sessions/{session_id}/messages",
                json={"message": "What is the Foobar protocol?"},
            )
        assert response.status_code == 504, response.text
        assert len(model.calls) == 1
        db = factory()
        rows = db.query(Message).filter(Message.session_id == session_id).all()
        assert [row.role for row in rows if row.id == question_id] == ["user"]
        assert not any(row.role == "assistant" for row in rows)
        db.close()
    finally:
        _clear()
        engine.dispose()


def test_fixed_relation_branch_wins_when_general_is_enabled(tmp_path):
    engine, factory, _ = _db_factory(tmp_path)
    session_id, _ = _seed(
        factory,
        ["I teach you this fact: B links to C."],
        _visible_filler(20),
        question="Does B link to C?",
    )
    model = ScriptedModel(["Yes, B links to C [message 1]."])
    service = _install(factory, model, fixed=True)
    try:
        with TestClient(app) as client:
            response = client.post(
                f"/api/sessions/{session_id}/messages",
                json={"message": "Does B link to C?"},
            )
        assert response.status_code == 200, response.text
        assert len(model.calls) == 1
        assert service.last_result is None
        assert response.json()["user_message"]["prompt_version_snapshot"] == (
            "receive-teaching-v7+programmatic-history-v1+l1-en"
        )
        assert response.json()["assistant_message"]["prompt_version_snapshot"] == (
            "receive-teaching-v7+programmatic-history-v1+l1-en"
        )
        fixed_system = model.calls[0][0][0]["content"]
        assert fixed_system.endswith(LANGUAGE_INSTRUCTIONS["en"] + "\n")
        assert fixed_system.count("Reply in English.") == 1
    finally:
        _clear()
        engine.dispose()


def test_pure_teaching_uses_no_decision_or_retrieval(tmp_path):
    engine, factory, _ = _db_factory(tmp_path)
    question = "I teach you this fact: X links to Y."
    session_id, _ = _seed(
        factory,
        ["The Foobar protocol uses a signed token."],
        _visible_filler(20),
        question=question,
    )
    model = ScriptedModel(["Understood."])
    service = _install(factory, model)
    try:
        with TestClient(app) as client:
            response = client.post(
                f"/api/sessions/{session_id}/messages",
                json={"message": question},
            )
        assert response.status_code == 200, response.text
        result = service.last_result
        assert result is not None
        assert result.status == "completed_direct"
        assert result.counts.decision_attempts == 0
        assert result.counts.retrieval_attempts == 0
        assert len(model.calls) == 1
        assert response.json()["assistant_message"]["prompt_version_snapshot"] == (
            "receive-teaching-v7+history-context-b1+l1-en"
        )
    finally:
        _clear()
        engine.dispose()


def test_unconfirmed_correction_support_is_never_guessed(tmp_path):
    engine, factory, _ = _db_factory(tmp_path)
    session_id, _ = _seed(
        factory,
        [
            "Correction: Step three of process P7 must happen after step two.",
        ],
        _visible_filler(20),
        question="What is the current order of process P7?",
    )
    model = ScriptedModel(
        [
            _decision(True, "process P7 step three"),
            "The available correction has no confirmed earlier version.",
        ]
    )
    service = _install(factory, model)
    try:
        with TestClient(app) as client:
            response = client.post(
                f"/api/sessions/{session_id}/messages",
                json={"message": "What is the current order of process P7?"},
            )
        assert response.status_code == 200, response.text
        result = service.last_result
        assert result is not None
        evidence = json.loads(result.messages[1]["content"])
        correction = next(
            item
            for item in evidence["retrieved_sources"]
            if item["message_id"] == 1
        )
        assert correction.get("support_message_id") is None
        assert correction["support_unconfirmed"] is True
    finally:
        _clear()
        engine.dispose()


def test_general_logs_omit_question_and_teacher_content(tmp_path, caplog):
    import logging

    engine, factory, _ = _db_factory(tmp_path)
    question = "What is the secret Foobar protocol?"
    secret = "The Foobar protocol uses a hidden signature."
    session_id, _ = _seed(
        factory, [secret], _visible_filler(20), question=question
    )
    model = ScriptedModel(
        [
            _decision(True, "Foobar signature"),
            "It uses a hidden signature [message 1].",
        ]
    )
    service = _install(factory, model)
    caplog.set_level(logging.INFO)
    try:
        with TestClient(app) as client:
            response = client.post(
                f"/api/sessions/{session_id}/messages",
                json={"message": question},
            )
        assert response.status_code == 200
        assert "Foobar" not in caplog.text
        assert "hidden signature" not in caplog.text
        assert "secret" not in caplog.text.lower()
        assert "general_history completed" in caplog.text
    finally:
        _clear()
        engine.dispose()


def test_mixed_teaching_and_question_still_uses_general_path(tmp_path):
    engine, factory, _ = _db_factory(tmp_path)
    session_id, _ = _seed(
        factory,
        ["The Foobar protocol uses a hidden signature."],
        _visible_filler(20),
        question=(
            "I teach you a separate fact: the sky is blue. "
            "What is the Foobar protocol?"
        ),
    )
    model = ScriptedModel(
        [
            _decision(True, "Foobar protocol"),
            "It uses a hidden signature [message 1].",
        ]
    )
    service = _install(factory, model)
    try:
        with TestClient(app) as client:
            response = client.post(
                f"/api/sessions/{session_id}/messages",
                json={
                    "message": (
                        "I teach you a separate fact: the sky is blue. "
                        "What is the Foobar protocol?"
                    )
                },
            )
        assert response.status_code == 200, response.text
        result = service.last_result
        assert result is not None
        assert result.status == "completed_retrieved"
        assert result.counts.decision_attempts == 1
        assert result.counts.retrieval_attempts == 1
        assert result.counts.final_attempts == 1
    finally:
        _clear()
        engine.dispose()


def test_general_requires_matching_profile_and_model(tmp_path):
    for settings_profile, settings_model in [
        ("other-profile", "model-x"),
        ("default", "other-model"),
        ("", ""),
    ]:
        engine, factory, _ = _db_factory(tmp_path, f"mismatch-{settings_profile}-{settings_model}.sqlite3")
        session_id, _ = _seed(
            factory,
            ["The Foobar protocol uses a hidden signature."],
            _visible_filler(20),
            question="What is the Foobar protocol?",
        )
        model = ScriptedModel(["Ordinary answer from the selected profile."])
        service = _install(
            factory,
            model,
            general=True,
            settings_profile_id=settings_profile,
            settings_model=settings_model,
        )
        try:
            with TestClient(app) as client:
                response = client.post(
                    f"/api/sessions/{session_id}/messages",
                    json={"message": "What is the Foobar protocol?"},
                )
            assert response.status_code == 200
            assert service.last_result is None
            assert len(model.calls) == 1
        finally:
            _clear()
            engine.dispose()


def test_truncated_decision_is_rejected_even_if_json_parses(tmp_path, caplog):
    engine, factory, _ = _db_factory(tmp_path)
    session_id, _ = _seed(
        factory,
        ["The Foobar protocol uses a hidden signature."],
        _visible_filler(20),
        question="What is the Foobar protocol?",
    )
    model = ScriptedModel(
        [_decision(True, "Foobar protocol")],
        finish_reason="length",
    )
    service = _install(factory, model)
    try:
        with TestClient(app) as client:
            response = client.post(
                f"/api/sessions/{session_id}/messages",
                json={"message": "What is the Foobar protocol?"},
            )
        assert response.status_code == 502
        assert service.last_result is not None
        assert service.last_result.status == "decision_failed"
        assert "status=decision_failed" in caplog.text
        assert "finish_reason='length'" in caplog.text
        assert len(model.calls) == 1
        db = factory()
        session = db.get(ChatSession, session_id)
        messages = list(db.query(Message).filter(Message.session_id == session.id))
        assert messages
        assert all(m.role == "user" for m in messages)
        db.close()
    finally:
        _clear()
        engine.dispose()


def test_truncated_final_does_not_write_assistant(tmp_path):
    engine, factory, _ = _db_factory(tmp_path)
    session_id, _ = _seed(
        factory,
        ["The Foobar protocol uses a hidden signature."],
        _visible_filler(20),
        question="What is the Foobar protocol?",
    )
    model = ScriptedModel(
        [
            _decision(True, "Foobar protocol"),
            "It uses a hidden signature [message 1].",
        ],
        finish_reasons=["stop", "length"],
    )
    service = _install(factory, model)
    try:
        with TestClient(app) as client:
            response = client.post(
                f"/api/sessions/{session_id}/messages",
                json={"message": "What is the Foobar protocol?"},
            )
        assert response.status_code == 502
        assert service.last_result is not None
        assert service.last_result.status == "final_failed"
        assert len(model.calls) == 2
        db = factory()
        roles = [
            m.role
            for m in db.query(Message).filter(Message.session_id == session_id)
        ]
        assert roles
        assert "assistant" not in roles
        db.close()
    finally:
        _clear()
        engine.dispose()


def test_pre_send_model_failure_has_no_final_generation(tmp_path):
    engine, factory, _ = _db_factory(tmp_path)
    session_id, _ = _seed(
        factory,
        ["The Foobar protocol uses a hidden signature."],
        _visible_filler(20),
        question="What is the Foobar protocol?",
    )
    model = ScriptedModel([ValueError("local validation failed")])
    service = _install(factory, model)
    try:
        with TestClient(app) as client:
            response = client.post(
                f"/api/sessions/{session_id}/messages",
                json={"message": "What is the Foobar protocol?"},
            )
        assert response.status_code == 502
        assert service.last_result is not None
        assert service.last_result.status == "decision_failed"
        assert len(model.calls) == 1
        db = factory()
        roles = [
            m.role
            for m in db.query(Message).filter(Message.session_id == session_id)
        ]
        assert roles
        assert "assistant" not in roles
        db.close()
    finally:
        _clear()
        engine.dispose()


@pytest.mark.parametrize(
    ("fault", "expected_prompt", "expected_stage"),
    [
        ("flush", "receive-teaching-v7+history-context-b1+l1-en", "flush"),
        ("commit_before", "receive-teaching-v7+history-context-b1+l1-en", "commit_unknown"),
        ("commit_after", PROGRAMMATIC_HISTORY_PROMPT_VERSION + "+l1-en", "commit_unknown"),
    ],
)
def test_general_snapshot_faults_are_separated_and_stop_generation(
    tmp_path, caplog, fault, expected_prompt, expected_stage
):
    import logging

    engine, factory, _ = _db_factory(tmp_path, f"snapshot-{fault}.sqlite3")
    session_id, current_id = _seed(
        factory,
        ["The Foobar protocol uses a hidden signature."],
        _visible_filler(20),
        question="What is the Foobar protocol?",
    )
    model = ScriptedModel(
        [_decision(True, "Foobar protocol")],
        finish_reason="stop",
    )
    service = _install(factory, model)
    marker = {"user_committed": False}

    def dirty_snapshot(session):
        return any(
            isinstance(obj, Message)
            and obj in session.dirty
            and obj.prompt_version_snapshot.startswith(
                PROGRAMMATIC_HISTORY_PROMPT_VERSION
            )
            for obj in session.dirty
        )

    def after_commit(session):
        if session.info.get("snapshot_commit_after"):
            session.info.pop("snapshot_commit_after", None)
            raise RuntimeError("injected after-commit failure")
        if not marker["user_committed"]:
            marker["user_committed"] = True
            session.info["snapshot_user_committed"] = True

    def before_flush(session, flush_context, instances):
        if not session.info.get("snapshot_user_committed"):
            return
        if not dirty_snapshot(session):
            return
        if fault == "flush":
            raise RuntimeError("injected flush failure")
        # flush() clears session.dirty before commit(); remember the pending
        # snapshot update for the commit hooks.
        session.info["snapshot_pending_commit"] = True

    def before_commit(session):
        if not session.info.get("snapshot_user_committed"):
            return
        if not (
            session.info.get("snapshot_pending_commit")
            or dirty_snapshot(session)
        ):
            return
        if fault == "commit_before":
            raise RuntimeError("injected commit failure")
        if fault == "commit_after":
            session.info["snapshot_commit_after"] = True

    event.listen(Session, "after_commit", after_commit)
    event.listen(Session, "before_flush", before_flush)
    event.listen(Session, "before_commit", before_commit)
    caplog.set_level(logging.WARNING)
    try:
        with TestClient(app) as client:
            response = client.post(
                f"/api/sessions/{session_id}/messages",
                json={"message": "What is the Foobar protocol?"},
            )
        assert response.status_code == 500, (
            response.text + " || " + caplog.text + " || status=" +
            (service.last_result.status if service.last_result else "none")
        )
        assert service.last_result is not None
        assert service.last_result.status == "snapshot_failed"
        assert len(model.calls) == 1
        verify = factory()
        user = (
            verify.query(Message)
            .filter(Message.session_id == session_id, Message.role == "user")
            .order_by(Message.id.desc())
            .first()
        )
        assert user is not None
        assert user.prompt_version_snapshot == expected_prompt
        assert f"stage={expected_stage}" in caplog.text
        assert f"user_message_id={user.id}" in caplog.text
        roles = [
            m.role
            for m in verify.query(Message).filter(Message.session_id == session_id)
        ]
        assert "assistant" not in roles
        verify.close()
    finally:
        event.remove(Session, "after_commit", after_commit)
        event.remove(Session, "before_flush", before_flush)
        event.remove(Session, "before_commit", before_commit)
        _clear()
        engine.dispose()


# ---------------------------------------------------------------------------
# v2 routing and failure semantics
# ---------------------------------------------------------------------------

from backend.exceptions import LLMError
from backend.general_history.decision import DecisionError, parse_decision


def _post(client, session_id, question):
    return client.post(
        f"/api/sessions/{session_id}/messages",
        json={"message": question},
    )


def test_trace_with_unique_anchor_retrieves_without_decision(tmp_path):
    engine, factory, _ = _db_factory(tmp_path)
    hidden = ["The Foobar protocol was originally called the Quux protocol."]
    session_id, _ = _seed(
        factory,
        hidden,
        _visible_filler(20),
        question="What did you originally call the Foobar protocol?",
    )
    model = ScriptedModel(
        ["It was originally called the Quux protocol [message 1]."]
    )
    service = _install(factory, model)
    try:
        with TestClient(app) as client:
            response = _post(
                client,
                session_id,
                "What did you originally call the Foobar protocol?",
            )
        assert response.status_code == 200, response.text
        assert len(model.calls) == 1
        assert service.last_result.counts.decision_attempts == 0
        assert service.last_result.counts.retrieval_attempts == 1
        assert service.last_result.counts.final_attempts == 1
        assert "Quux protocol" in model.calls[0][0][1]["content"]
        assert response.json()["user_message"]["prompt_version_snapshot"] == (
            "receive-teaching-v7+programmatic-history-v1+l1-en"
        )
    finally:
        _clear()
        engine.dispose()


@pytest.mark.parametrize(
    "question",
    [
        "What did you originally call this rule?",
        "你最初把这个规则叫什么？",
    ],
)
def test_trace_without_unique_anchor_clarifies_without_decision(tmp_path, question):
    engine, factory, _ = _db_factory(tmp_path)
    session_id, _ = _seed(
        factory,
        ["An older rule exists."],
        _visible_filler(20),
        question=question,
    )
    model = ScriptedModel(
        ["Which rule do you mean?"]
        if question.startswith("What")
        else ["你指的是哪条规则？"]
    )
    service = _install(factory, model)
    try:
        with TestClient(app) as client:
            response = _post(client, session_id, question)
        assert response.status_code == 200, response.text
        assert len(model.calls) == 1
        assert service.last_result.counts.decision_attempts == 0
        assert service.last_result.counts.retrieval_attempts == 0
        assert service.last_result.counts.final_attempts == 1
        assert "exactly one focused clarification" in model.calls[0][0][0]["content"]
    finally:
        _clear()
        engine.dispose()


def test_source_word_alone_does_not_force_trace_route(tmp_path):
    engine, factory, _ = _db_factory(tmp_path)
    session_id, _ = _seed(
        factory,
        ["The Foobar protocol uses a signed token."],
        _visible_filler(20),
        question="What is the source of the Foobar protocol?",
    )
    model = ScriptedModel(
        [
            _decision(False, reason="not_needed"),
            "It uses a signed token.",
        ]
    )
    service = _install(factory, model)
    try:
        with TestClient(app) as client:
            response = _post(
                client,
                session_id,
                "What is the source of the Foobar protocol?",
            )
        assert response.status_code == 200, response.text
        assert len(model.calls) == 2
        assert service.last_result.counts.decision_attempts == 1
        assert service.last_result.counts.retrieval_attempts == 0
        assert service.last_result.counts.final_attempts == 0
    finally:
        _clear()
        engine.dispose()


def test_missing_explanation_uses_grounded_actual_retrieval(tmp_path):
    engine, factory, _ = _db_factory(tmp_path)
    hidden = [
        "The divisor rule says the divisor must be nonzero.",
        (
            "As h approaches zero, simplify the difference quotient and then "
            "substitute h=0; approaching is not the same as setting the divisor to zero."
        ),
    ]
    visible = [*_visible_filler(19), "The divisor must be nonzero."]
    question = (
        "For the difference quotient ((2+h)^3-8)/h, what does the divisor rule "
        "require, and why can we substitute h=0 after simplifying instead of "
        "treating approaching zero as setting the divisor to zero?"
    )
    session_id, _ = _seed(factory, hidden, visible, question=question)
    model = ScriptedModel(
        [
            _decision(True, "zero substitute", reason="missing_explanation"),
            "The evidence explains approaching zero and substituting h=0 [message 2].",
        ]
    )
    service = _install(factory, model)
    try:
        with TestClient(app) as client:
            response = _post(client, session_id, question)
        assert response.status_code == 200, response.text
        assert len(model.calls) == 2
        assert service.last_result.counts.decision_attempts == 1
        assert service.last_result.counts.retrieval_attempts == 1
        assert service.last_result.counts.final_attempts == 1
        assert service.last_result.retrieved_source_ids == [2]
        final_request = json.dumps(model.calls[1][0], ensure_ascii=False)
        assert "approaching is not the same as setting" in final_request
        assert "The divisor must be nonzero." in final_request
    finally:
        _clear()
        engine.dispose()


def test_decision_upstream_timeout_returns_504_without_retrieval_or_final(tmp_path):
    engine, factory, _ = _db_factory(tmp_path)
    session_id, _ = _seed(
        factory,
        ["The Foobar protocol uses a signed token."],
        _visible_filler(20),
        question="What is the Foobar protocol?",
    )
    model = ScriptedModel([LLMError("decision timeout", 504)])
    service = _install(factory, model)
    try:
        with TestClient(app) as client:
            response = _post(client, session_id, "What is the Foobar protocol?")
        assert response.status_code == 504, response.text
        assert len(model.calls) == 1
        assert service.last_result.counts.decision_attempts == 1
        assert service.last_result.counts.retrieval_attempts == 0
        assert service.last_result.counts.final_attempts == 0
        db = factory()
        assert db.query(Message).filter(Message.session_id == session_id).count() == 23
        db.close()
    finally:
        _clear()
        engine.dispose()


def test_final_upstream_timeout_returns_504_after_retrieval(tmp_path):
    engine, factory, _ = _db_factory(tmp_path)
    session_id, _ = _seed(
        factory,
        ["The Foobar protocol was originally called the Quux protocol."],
        _visible_filler(20),
        question="What did you originally call the Foobar protocol?",
    )
    model = ScriptedModel([LLMError("final timeout", 504)])
    service = _install(factory, model)
    try:
        with TestClient(app) as client:
            response = _post(
                client,
                session_id,
                "What did you originally call the Foobar protocol?",
            )
        assert response.status_code == 504, response.text
        assert len(model.calls) == 1
        assert service.last_result.counts.decision_attempts == 0
        assert service.last_result.counts.retrieval_attempts == 1
        assert service.last_result.counts.final_attempts == 1
        db = factory()
        assert db.query(Message).filter(Message.session_id == session_id).count() == 23
        db.close()
    finally:
        _clear()
        engine.dispose()


def test_v2_parser_separates_missing_fact_and_old_reason():
    check = parse_decision(
        _decision(True, "divisor nonzero", reason="missing_fact"),
        visible_ids=(1,),
    )
    assert check.decision.reason == "missing_fact"
    check = parse_decision(
        _decision(True, "divisor nonzero", reason="missing_explanation"),
        visible_ids=(1,),
    )
    assert check.decision.reason == "missing_explanation"
    with pytest.raises(DecisionError, match="invalid_reason"):
        parse_decision(
            _decision(True, "divisor nonzero", reason="ordinary_recall"),
            visible_ids=(1,),
        )


def test_decision_local_deadline_with_reliable_anchor_retries_once_as_search(tmp_path):
    engine, factory, _ = _db_factory(tmp_path)
    session_id, _ = _seed(
        factory,
        ["The Foobar protocol uses a signed token."],
        _visible_filler(20),
        question="What is the Foobar protocol?",
    )
    model = ScriptedModel(
        [asyncio.TimeoutError(), "It uses a signed token [message 1]."]
    )
    service = _install(factory, model)
    try:
        with TestClient(app) as client:
            response = _post(client, session_id, "What is the Foobar protocol?")
        assert response.status_code == 200, response.text
        result = service.last_result
        assert result.status == "completed_retrieved"
        assert result.counts.decision_attempts == 1
        assert result.counts.retrieval_attempts == 1
        assert result.counts.final_attempts == 1
        assert len(model.calls) == 2
        assert result.retrieved_source_ids == [1]
    finally:
        _clear()
        engine.dispose()


# ---------------------------------------------------------------------------
# Review regression: relation fallback, trace intent, query intent
# ---------------------------------------------------------------------------

def test_general_relation_retrieval_uses_defined_anchor_limit(tmp_path):
    engine, factory, _ = _db_factory(tmp_path)
    session_id, _ = _seed(
        factory,
        ["I teach you this fact: B links to C."],
        _visible_filler(20),
        question="Does B link to C?",
    )
    model = ScriptedModel(["Yes, B links to C [message 1]."])
    service = _install(factory, model, fixed=False)
    try:
        with TestClient(app) as client:
            response = _post(client, session_id, "Does B link to C?")
        assert response.status_code == 200, response.text
        assert service.last_result.counts.decision_attempts == 0
        assert service.last_result.counts.retrieval_attempts == 1
        assert service.last_result.counts.final_attempts == 1
        assert len(model.calls) == 1
        db = factory()
        assert db.query(Message).filter(
            Message.session_id == session_id,
            Message.role == "assistant",
        ).count() == 1
        db.close()
    finally:
        _clear()
        engine.dispose()


@pytest.mark.parametrize("question", ["Which source message?", "请引用原文。"])
def test_source_words_without_unique_object_clarify(tmp_path, question):
    engine, factory, _ = _db_factory(tmp_path)
    session_id, _ = _seed(
        factory,
        ["The Foobar protocol uses a signed token."],
        _visible_filler(20),
        question=question,
    )
    model = ScriptedModel(
        ["Which message do you mean?"]
        if question.startswith("Which")
        else ["你指的是哪条消息？"]
    )
    service = _install(factory, model)
    try:
        with TestClient(app) as client:
            response = _post(client, session_id, question)
        assert response.status_code == 200, response.text
        assert len(model.calls) == 1
        assert service.last_result.counts.decision_attempts == 0
        assert service.last_result.counts.retrieval_attempts == 0
        assert service.last_result.counts.final_attempts == 1
        assert "exactly one focused clarification" in model.calls[0][0][0]["content"]
    finally:
        _clear()
        engine.dispose()


def test_citation_format_request_enters_decision(tmp_path):
    engine, factory, _ = _db_factory(tmp_path)
    session_id, _ = _seed(
        factory,
        ["The Foobar protocol uses a signed token."],
        _visible_filler(20),
        question="Use the citation format I taught.",
    )
    model = ScriptedModel(
        [
            _decision(False, reason="not_needed"),
            "I will use the citation format.",
        ]
    )
    service = _install(factory, model)
    try:
        with TestClient(app) as client:
            response = _post(client, session_id, "Use the citation format I taught.")
        assert response.status_code == 200, response.text
        assert len(model.calls) == 2
        assert service.last_result.counts.decision_attempts == 1
        assert service.last_result.counts.retrieval_attempts == 0
        assert service.last_result.counts.final_attempts == 0
    finally:
        _clear()
        engine.dispose()


def test_chinese_unique_trace_object_still_retrieves(tmp_path):
    engine, factory, _ = _db_factory(tmp_path)
    session_id, _ = _seed(
        factory,
        ["The Foobar protocol was originally called the Quux protocol."],
        _visible_filler(20),
        question="你最初把 Foobar 规则叫什么？",
    )
    model = ScriptedModel(["那个协议叫 Quux。[message 1]"])
    service = _install(factory, model)
    try:
        with TestClient(app) as client:
            response = _post(client, session_id, "你最初把 Foobar 规则叫什么？")
        assert response.status_code == 200, response.text
        assert len(model.calls) == 1
        assert service.last_result.counts.decision_attempts == 0
        assert service.last_result.counts.retrieval_attempts == 1
        assert service.last_result.counts.final_attempts == 1
    finally:
        _clear()
        engine.dispose()


def test_two_possible_trace_objects_clarify(tmp_path):
    engine, factory, _ = _db_factory(tmp_path)
    session_id, _ = _seed(
        factory,
        ["The Foobar protocol uses a signed token."],
        _visible_filler(20),
        question="Which source message did you quote about Foobar or Baz?",
    )
    model = ScriptedModel(["Which topic do you mean?"])
    service = _install(factory, model)
    try:
        with TestClient(app) as client:
            response = _post(
                client,
                session_id,
                "Which source message did you quote about Foobar or Baz?",
            )
        assert response.status_code == 200, response.text
        assert len(model.calls) == 1
        assert service.last_result.counts.decision_attempts == 0
        assert service.last_result.counts.retrieval_attempts == 0
        assert service.last_result.counts.final_attempts == 1
    finally:
        _clear()
        engine.dispose()


def test_missing_explanation_source_query_is_invalid_and_not_retrieved(tmp_path):
    engine, factory, _ = _db_factory(tmp_path)
    session_id, _ = _seed(
        factory,
        ["The divisor rule says the divisor must be nonzero."],
        _visible_filler(20),
        question="What does the divisor rule require for this quotient?",
    )
    model = ScriptedModel(
        [
            _decision(
                True,
                "What did you originally call the divisor rule?",
                reason="missing_explanation",
            )
        ]
    )
    service = _install(factory, model)
    try:
        with TestClient(app) as client:
            response = _post(
                client,
                session_id,
                "What does the divisor rule require for this quotient?",
            )
        assert response.status_code == 502, response.text
        assert len(model.calls) == 1
        assert service.last_result.status == "decision_invalid"
        assert service.last_result.counts.decision_attempts == 1
        assert service.last_result.counts.retrieval_attempts == 0
        assert service.last_result.counts.final_attempts == 0
        db = factory()
        assert db.query(Message).filter(Message.session_id == session_id).count() == 23
        db.close()
    finally:
        _clear()
        engine.dispose()


def test_proper_name_containing_source_is_not_blocked(tmp_path):
    engine, factory, _ = _db_factory(tmp_path)
    session_id, _ = _seed(
        factory,
        ["Source Engine uses a signed token."],
        _visible_filler(20),
        question="What does Source Engine use?",
    )
    model = ScriptedModel(
        [
            _decision(True, "Source Engine token", reason="missing_fact"),
            "Source Engine uses a signed token [message 1].",
        ]
    )
    service = _install(factory, model)
    try:
        with TestClient(app) as client:
            response = _post(client, session_id, "What does Source Engine use?")
        assert response.status_code == 200, response.text
        assert len(model.calls) == 2
        assert service.last_result.counts.decision_attempts == 1
        assert service.last_result.counts.retrieval_attempts == 1
        assert service.last_result.counts.final_attempts == 1
    finally:
        _clear()
        engine.dispose()


def test_ambiguous_process_local_deadline_is_one_decision_zero_retrieval(tmp_path):
    engine, factory, _ = _db_factory(tmp_path)
    visible = [
        *_visible_filler(18),
        "Process P7 uses the blue lever.",
        "Process P8 uses the red lever.",
    ]
    session_id, _ = _seed(
        factory,
        ["Process P9 uses the green lever."],
        visible,
        question="What is the next process step?",
    )
    model = ScriptedModel([asyncio.TimeoutError(), "Which process do you mean?"])
    service = _install(factory, model)
    try:
        with TestClient(app) as client:
            response = _post(client, session_id, "What is the next process step?")
        assert response.status_code == 200, response.text
        result = service.last_result
        assert result.status == "completed_clarify"
        assert result.counts.decision_attempts == 1
        assert result.counts.retrieval_attempts == 0
        assert result.counts.final_attempts == 1
        assert len(model.calls) == 2
    finally:
        _clear()
        engine.dispose()


def test_unique_anchor_local_deadline_is_one_retrieval(tmp_path):
    engine, factory, _ = _db_factory(tmp_path)
    session_id, _ = _seed(
        factory,
        ["The alpha bridge is painted green."],
        _visible_filler(20),
        question="What color is the alpha bridge?",
    )
    model = ScriptedModel(
        [asyncio.TimeoutError(), "The alpha bridge is green [message 1]."]
    )
    service = _install(factory, model)
    try:
        with TestClient(app) as client:
            response = _post(client, session_id, "What color is the alpha bridge?")
        assert response.status_code == 200, response.text
        result = service.last_result
        assert result.counts.decision_attempts == 1
        assert result.counts.retrieval_attempts == 1
        assert result.counts.final_attempts == 1
        assert len(model.calls) == 2
        assert result.retrieved_source_ids == [1]
    finally:
        _clear()
        engine.dispose()


@pytest.mark.parametrize(
    "question",
    [
        "What is the “Source Message” format?",
        "按我教的引用格式回答 Foobar 的问题",
    ],
)
def test_source_message_name_or_citation_format_uses_decision(tmp_path, question):
    engine, factory, _ = _db_factory(tmp_path)
    session_id, _ = _seed(
        factory,
        ["The Foobar protocol uses a signed token."],
        _visible_filler(20),
        question=question,
    )
    model = ScriptedModel(
        [
            _decision(False, reason="not_needed"),
            "I will answer using the taught format."
            if question.startswith("What")
            else "我会按你教的引用格式回答。",
        ]
    )
    service = _install(factory, model)
    try:
        with TestClient(app) as client:
            response = _post(client, session_id, question)
        assert response.status_code == 200, response.text
        assert len(model.calls) == 2
        assert service.last_result.counts.decision_attempts == 1
        assert service.last_result.counts.retrieval_attempts == 0
    finally:
        _clear()
        engine.dispose()


def _seed_mixed_explanation(factory, *, add_assistant_id=False):
    hidden = [
        "The Juniper Relay checks are separate because the timer can be ready while the gate is still open."
    ]
    visible = [
        *_visible_filler(19),
        "The Juniper Relay rule requires testing the Cedar latch separately from the Amber lamp.",
    ]
    session_id, current_id = _seed(
        factory,
        hidden,
        visible,
        question="Why does the Juniper Relay rule require testing the Cedar latch separately from the Amber lamp?",
    )
    db = factory()
    rows = db.query(Message).filter(Message.session_id == session_id).order_by(Message.id).all()
    rule_id = next(m.id for m in rows if m.role == "user" and "separately" in m.content)
    assistant_id = None
    if add_assistant_id:
        extra = Message(
            session_id=session_id,
            role="assistant",
            content="An assistant note.",
            interaction_mode_snapshot="receive_teaching",
            prompt_version_snapshot="receive-teaching-v7+history-context-b1",
            llm_profile_id_snapshot="default",
            llm_profile_kind_snapshot="api",
            llm_model_snapshot="model-x",
        )
        db.add(extra)
        db.commit()
        db.refresh(extra)
        assistant_id = extra.id
    db.close()
    return session_id, rule_id, assistant_id


@pytest.mark.parametrize("include_partial_id", [True, False])
def test_missing_explanation_accepts_optional_partial_visible_id(tmp_path, include_partial_id):
    engine, factory, _ = _db_factory(tmp_path)
    session_id, rule_id, _ = _seed_mixed_explanation(factory)
    ids = [rule_id] if include_partial_id else []
    model = ScriptedModel(
        [
            _decision(True, "Juniper Relay", ids=ids, reason="missing_explanation"),
            "The evidence says the checks are separate [message 1].",
        ]
    )
    service = _install(factory, model)
    try:
        with TestClient(app) as client:
            response = _post(
                client,
                session_id,
                "Why does the Juniper Relay rule require testing the Cedar latch separately from the Amber lamp?",
            )
        assert response.status_code == 200, response.text
        result = service.last_result
        assert result.counts.decision_attempts == 1
        assert result.counts.retrieval_attempts == 1
        assert result.counts.final_attempts == 1
        assert len(model.calls) == 2
        final_request = json.dumps(model.calls[1][0], ensure_ascii=False)
        assert "Juniper Relay rule requires testing" in final_request
        assert "the timer can be ready" in final_request
        assert response.json()["user_message"]["prompt_version_snapshot"] == (
            "receive-teaching-v7+programmatic-history-v1+l1-en"
        )
        assert response.json()["assistant_message"]["prompt_version_snapshot"] == (
            "receive-teaching-v7+programmatic-history-v1+l1-en"
        )
    finally:
        _clear()
        engine.dispose()


@pytest.mark.parametrize("case", ["outside", "assistant", "other_session", "duplicate"])
def test_missing_explanation_invalid_partial_ids_stop_before_retrieval(tmp_path, case):
    engine, factory, _ = _db_factory(tmp_path)
    session_id, rule_id, assistant_id = _seed_mixed_explanation(
        factory, add_assistant_id=(case == "assistant")
    )
    if case == "outside":
        ids = [999999]
    elif case == "assistant":
        ids = [assistant_id]
    elif case == "other_session":
        db = factory()
        other = ChatSession(title="other", llm_profile_id="default", llm_model_snapshot="model-x")
        db.add(other)
        db.flush()
        message = Message(
            session_id=other.id,
            role="user",
            content="Other session fact.",
            interaction_mode_snapshot="receive_teaching",
            prompt_version_snapshot="receive-teaching-v7+history-context-b1",
            llm_profile_id_snapshot="default",
            llm_profile_kind_snapshot="api",
            llm_model_snapshot="model-x",
        )
        db.add(message)
        db.commit()
        db.refresh(message)
        ids = [message.id]
        db.close()
    else:
        ids = [rule_id, rule_id]
    model = ScriptedModel(
        [_decision(True, "Juniper Relay", ids=ids, reason="missing_explanation")]
    )
    service = _install(factory, model)
    try:
        with TestClient(app) as client:
            response = _post(
                client,
                session_id,
                "Why does the Juniper Relay rule require testing the Cedar latch separately from the Amber lamp?",
            )
        assert response.status_code == 502, response.text
        assert len(model.calls) == 1
        result = service.last_result
        assert result.status == "decision_invalid"
        assert result.counts.decision_attempts == 1
        assert result.counts.retrieval_attempts == 0
        assert result.counts.final_attempts == 0
        db = factory()
        rows = db.query(Message).filter(Message.session_id == session_id).all()
        assistant_rows = [row for row in rows if row.role == "assistant"]
        assert all(row.id in {assistant_id} for row in assistant_rows)
        db.close()
    finally:
        _clear()
        engine.dispose()


def test_missing_fact_with_visible_id_is_rejected(tmp_path):
    engine, factory, _ = _db_factory(tmp_path)
    session_id, rule_id, _ = _seed_mixed_explanation(factory)
    model = ScriptedModel(
        [_decision(True, "checks separate", ids=[rule_id], reason="missing_fact")]
    )
    service = _install(factory, model)
    try:
        with TestClient(app) as client:
            response = _post(
                client,
                session_id,
                "Why does the Juniper Relay rule require testing the Cedar latch separately from the Amber lamp?",
            )
        assert response.status_code == 502, response.text
        assert service.last_result.status == "decision_invalid"
        assert service.last_result.error_code == "missing_fact_with_visible_ids"
        assert service.last_result.counts.retrieval_attempts == 0
        assert service.last_result.counts.final_attempts == 0
    finally:
        _clear()
        engine.dispose()


def test_invalid_decision_log_exposes_safe_reason_metadata(tmp_path, caplog):
    engine, factory, _ = _db_factory(tmp_path)
    session_id, rule_id, _ = _seed_mixed_explanation(factory)
    model = ScriptedModel(
        [_decision(True, "checks separate", ids=[999999], reason="missing_explanation")]
    )
    _install(factory, model)
    try:
        with caplog.at_level("WARNING", logger="backend.chat_service"):
            with TestClient(app) as client:
                response = _post(
                    client,
                    session_id,
                    "Why does the Juniper Relay rule require testing the Cedar latch separately from the Amber lamp?",
                )
        assert response.status_code == 502
        record = next(
            record for record in caplog.records
            if "general_history_failure" in record.getMessage()
        )
        message = record.getMessage()
        assert "decision_reason=missing_explanation" in message
        assert "decision_has_query=True" in message
        assert "decision_visible_id_count=1" in message
        assert "error_code=visible_id_not_visible" in message
        assert "checks separate" not in message
        assert str(rule_id) not in message
    finally:
        _clear()
        engine.dispose()


def test_general_retrieval_502_when_final_reply_language_is_wrong(tmp_path):
    engine, factory, _ = _db_factory(tmp_path)
    session_id, _ = _seed(
        factory,
        ["The Foobar protocol uses a hidden signature."],
        _visible_filler(20),
        question="What is the Foobar protocol?",
    )
    model = ScriptedModel(
        [
            _decision(True, "Foobar protocol"),
            "这是中文回答，但目标语言是英文。",
        ]
    )
    service = _install(factory, model)
    try:
        with TestClient(app) as client:
            response = client.post(
                f"/api/sessions/{session_id}/messages",
                json={"message": "What is the Foobar protocol?"},
            )
        assert response.status_code == 502
        assert response.json()["detail"]["code"] == "receive_reply_language_mismatch"
        assert len(model.calls) == 2
        db = factory()
        user = (
            db.query(Message)
            .filter(Message.session_id == session_id, Message.role == "user")
            .order_by(Message.id.desc())
            .first()
        )
        assistant_count = (
            db.query(Message)
            .filter(Message.session_id == session_id, Message.role == "assistant")
            .count()
        )
        db.close()
        assert user.prompt_version_snapshot.endswith("+l1-en")
        assert assistant_count == 0
    finally:
        _clear()
        engine.dispose()


def test_fixed_retrieval_wrong_language_returns_502_and_keeps_user(tmp_path):
    engine, factory, _ = _db_factory(tmp_path)
    session_id, question_id = _seed(
        factory,
        ["I teach you this fact: B links to C."],
        _visible_filler(20),
        question="Does B link to C?",
    )
    model = ScriptedModel(["这是中文回复。"])
    service = _install(factory, model, fixed=True)
    try:
        with TestClient(app) as client:
            response = client.post(
                f"/api/sessions/{session_id}/messages",
                json={"message": "Does B link to C?"},
            )
        assert response.status_code == 502, response.text
        detail = response.json()["detail"]
        assert detail["code"] == "receive_reply_language_mismatch"
        assert "saved" in detail["message"].lower()
        assert "not saved" in detail["message"].lower()
        assert len(model.calls) == 1
        db = factory()
        rows = db.query(Message).filter(Message.session_id == session_id).all()
        db.close()
        assert [row.role for row in rows if row.id == question_id] == ["user"]
        assert not any(row.role == "assistant" for row in rows)
        user = max(
            (row for row in rows if row.role == "user"),
            key=lambda row: row.id,
        )
        assert user.prompt_version_snapshot.endswith("+l1-en")
    finally:
        _clear()
        engine.dispose()


def test_clarify_wrong_language_returns_502_and_keeps_user(tmp_path):
    engine, factory, _ = _db_factory(tmp_path)
    session_id, question_id = _seed(
        factory,
        ["The Foobar protocol uses a signed token."],
        _visible_filler(20),
        question="What about it?",
    )
    model = ScriptedModel([asyncio.TimeoutError(), "这是中文澄清。"])
    service = _install(factory, model)
    try:
        with TestClient(app) as client:
            response = client.post(
                f"/api/sessions/{session_id}/messages",
                json={"message": "What about it?"},
            )
        assert response.status_code == 502, response.text
        assert response.json()["detail"]["code"] == "receive_reply_language_mismatch"
        assert len(model.calls) == 2
        db = factory()
        rows = db.query(Message).filter(Message.session_id == session_id).all()
        db.close()
        assert not any(row.role == "assistant" for row in rows)
        user = max(
            (row for row in rows if row.role == "user"),
            key=lambda row: row.id,
        )
        assert user.prompt_version_snapshot == (
            "receive-teaching-v7+clarify-v1+l1-en"
        )
    finally:
        _clear()
        engine.dispose()


@pytest.mark.parametrize(
    "reply",
    [
        "Here is the code:\n```\n这是中文\n```",
        "Example: `if 这是中文`.",
        "Got it. $这是中文回答。$",
    ],
)
def test_english_target_blocks_chinese_inside_protected_fragments(
    tmp_path, reply
):
    engine, factory, _ = _db_factory(tmp_path)
    session_id, _ = _seed(
        factory,
        ["The Foobar protocol uses a hidden signature."],
        _visible_filler(20),
        question="What is the Foobar protocol?",
    )
    model = ScriptedModel([reply])
    _install(factory, model, general=False)
    try:
        with TestClient(app) as client:
            response = client.post(
                f"/api/sessions/{session_id}/messages",
                json={"message": "What is the Foobar protocol?"},
            )
        assert response.status_code == 502, response.text
        assert response.json()["detail"]["code"] == "receive_reply_language_mismatch"
        assert len(model.calls) == 1
        system = model.calls[0][0][0]["content"]
        assert system.endswith(LANGUAGE_INSTRUCTIONS["en"] + "\n")
        db = factory()
        user = (
            db.query(Message)
            .filter(Message.session_id == session_id, Message.role == "user")
            .order_by(Message.id.desc())
            .first()
        )
        assistant_count = (
            db.query(Message)
            .filter(Message.session_id == session_id, Message.role == "assistant")
            .count()
        )
        db.close()
        assert user.prompt_version_snapshot.endswith("+l1-en")
        assert assistant_count == 0
    finally:
        _clear()
        engine.dispose()


def test_zh_target_quote_does_not_hide_remaining_english(tmp_path):
    engine, factory, _ = _db_factory(tmp_path)
    session_id, _ = _seed(
        factory,
        ["I teach you the label 标签."],
        _visible_filler(20),
        question="请解释这个标签。",
    )
    model = ScriptedModel(["“标签” This is English."])
    _install(factory, model, general=False)
    try:
        with TestClient(app) as client:
            response = client.post(
                f"/api/sessions/{session_id}/messages",
                json={"message": "请解释这个标签。"},
            )
        assert response.status_code == 502, response.text
        assert response.json()["detail"]["code"] == "receive_reply_language_mismatch"
        assert len(model.calls) == 1
        db = factory()
        user = (
            db.query(Message)
            .filter(Message.session_id == session_id, Message.role == "user")
            .order_by(Message.id.desc())
            .first()
        )
        assistant_count = (
            db.query(Message)
            .filter(Message.session_id == session_id, Message.role == "assistant")
            .count()
        )
        db.close()
        assert user.prompt_version_snapshot.endswith("+l1-zh")
        assert assistant_count == 0
    finally:
        _clear()
        engine.dispose()


def test_zh_target_blocks_uppercase_english_prose(tmp_path):
    engine, factory, _ = _db_factory(tmp_path)
    session_id, _ = _seed(
        factory,
        ["I teach you the label 标签."],
        _visible_filler(20),
        question="请解释这个标签。",
    )
    model = ScriptedModel(["好，THIS IS NOT OK."])
    _install(factory, model, general=False)
    try:
        with TestClient(app) as client:
            response = client.post(
                f"/api/sessions/{session_id}/messages",
                json={"message": "请解释这个标签。"},
            )
        assert response.status_code == 502, response.text
        assert response.json()["detail"]["code"] == "receive_reply_language_mismatch"
        assert len(model.calls) == 1
        db = factory()
        user = (
            db.query(Message)
            .filter(Message.session_id == session_id, Message.role == "user")
            .order_by(Message.id.desc())
            .first()
        )
        assistant_count = (
            db.query(Message)
            .filter(Message.session_id == session_id, Message.role == "assistant")
            .count()
        )
        db.close()
        assert user.prompt_version_snapshot.endswith("+l1-zh")
        assert assistant_count == 0
    finally:
        _clear()
        engine.dispose()


def _persistent_english_seed(factory, question):
    return _seed(
        factory,
        ["The Foobar protocol uses a hidden signature."],
        ["For this conversation, answer in English."]
        + _visible_filler(18),
        question=question,
    )


def test_stop_step_does_not_revoke_persistent_english(tmp_path):
    engine, factory, _ = _db_factory(tmp_path)
    session_id, _ = _persistent_english_seed(factory, "请停止这个步骤。")
    model = ScriptedModel(["This step is stopped."])
    _install(factory, model, general=False)
    try:
        with TestClient(app) as client:
            response = client.post(
                f"/api/sessions/{session_id}/messages",
                json={"message": "请停止这个步骤。"},
            )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["user_message"]["prompt_version_snapshot"].endswith("+l1-en")
        assert body["assistant_message"]["prompt_version_snapshot"].endswith(
            "+l1-en"
        )
        assert len(model.calls) == 1
        assert model.calls[0][0][0]["content"].endswith(
            LANGUAGE_INSTRUCTIONS["en"] + "\n"
        )
    finally:
        _clear()
        engine.dispose()


def test_for_this_conversation_persists_into_next_chinese_input(tmp_path):
    engine, factory, _ = _db_factory(tmp_path)
    session_id, _ = _persistent_english_seed(factory, "请继续。")
    model = ScriptedModel(["好的，我继续。"])
    _install(factory, model, general=False)
    try:
        with TestClient(app) as client:
            response = client.post(
                f"/api/sessions/{session_id}/messages",
                json={"message": "请继续。"},
            )
        assert response.status_code == 502, response.text
        assert response.json()["detail"]["code"] == "receive_reply_language_mismatch"
        assert len(model.calls) == 1
        assert model.calls[0][0][0]["content"].endswith(
            LANGUAGE_INSTRUCTIONS["en"] + "\n"
        )
        db = factory()
        user = (
            db.query(Message)
            .filter(Message.session_id == session_id, Message.role == "user")
            .order_by(Message.id.desc())
            .first()
        )
        assistant_count = (
            db.query(Message)
            .filter(Message.session_id == session_id, Message.role == "assistant")
            .count()
        )
        db.close()
        assert user.prompt_version_snapshot.endswith("+l1-en")
        assert assistant_count == 0
    finally:
        _clear()
        engine.dispose()


@pytest.mark.parametrize(
    "reply",
    [
        "收到。\n```text\nThis is an English explanation.\n```",
        "收到。`if this is an English explanation`",
        "收到。$This is an English explanation.$",
    ],
)
def test_zh_target_blocks_english_inside_text_fence_inline_and_math(
    tmp_path, reply
):
    engine, factory, _ = _db_factory(tmp_path)
    session_id, _ = _seed(
        factory,
        ["I teach you the label 标签."],
        _visible_filler(20),
        question="请解释这个标签。",
    )
    model = ScriptedModel([reply])
    _install(factory, model, general=False)
    try:
        with TestClient(app) as client:
            response = client.post(
                f"/api/sessions/{session_id}/messages",
                json={"message": "请解释这个标签。"},
            )
        assert response.status_code == 502, response.text
        assert response.json()["detail"]["code"] == "receive_reply_language_mismatch"
        assert len(model.calls) == 1
        assert model.calls[0][0][0]["content"].endswith(
            LANGUAGE_INSTRUCTIONS["zh"] + "\n"
        )
        db = factory()
        user = (
            db.query(Message)
            .filter(Message.session_id == session_id, Message.role == "user")
            .order_by(Message.id.desc())
            .first()
        )
        assistant_count = (
            db.query(Message)
            .filter(Message.session_id == session_id, Message.role == "assistant")
            .count()
        )
        db.close()
        assert user.prompt_version_snapshot.endswith("+l1-zh")
        assert assistant_count == 0
    finally:
        _clear()
        engine.dispose()


def test_en_target_accepts_exact_source_code_with_chinese_string(tmp_path):
    engine, factory, _ = _db_factory(tmp_path)
    question = (
        "Please echo this code exactly:\n"
        "```python\n"
        'print("你好")\n'
        "```"
    )
    session_id, _ = _seed(
        factory,
        ["The Foobar protocol uses a hidden signature."],
        _visible_filler(20),
        question=question,
    )
    model = ScriptedModel(['print("你好")'])
    _install(factory, model, general=False)
    try:
        with TestClient(app) as client:
            response = client.post(
                f"/api/sessions/{session_id}/messages",
                json={"message": question},
            )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["assistant_message"]["content"] == 'print("你好")'
        assert body["user_message"]["prompt_version_snapshot"].endswith("+l1-en")
        assert body["assistant_message"]["prompt_version_snapshot"].endswith("+l1-en")
        assert len(model.calls) == 1
        assert model.calls[0][0][0]["content"].endswith(
            LANGUAGE_INSTRUCTIONS["en"] + "\n"
        )
    finally:
        _clear()
        engine.dispose()


@pytest.mark.parametrize("question", ["Understood.", "Thanks."])
def test_short_english_ack_overrides_chinese_history(tmp_path, question):
    engine, factory, _ = _db_factory(tmp_path)
    session_id, _ = _seed(
        factory,
        ["我教过你一条规则。"],
        _visible_filler(19),
        question=question,
    )
    model = ScriptedModel(["The instruction is to pour the water."])
    _install(factory, model, general=False)
    try:
        with TestClient(app) as client:
            response = client.post(
                f"/api/sessions/{session_id}/messages",
                json={"message": question},
            )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["user_message"]["prompt_version_snapshot"].endswith("+l1-en")
        assert body["assistant_message"]["prompt_version_snapshot"].endswith("+l1-en")
        assert len(model.calls) == 1
        assert model.calls[0][0][0]["content"].endswith(
            LANGUAGE_INSTRUCTIONS["en"] + "\n"
        )
    finally:
        _clear()
        engine.dispose()


def test_en_target_accepts_english_sentence_with_single_french_lookalike(tmp_path):
    engine, factory, _ = _db_factory(tmp_path)
    session_id, _ = _seed(
        factory,
        ["I teach you the pouring step."],
        _visible_filler(20),
        question="What is the instruction?",
    )
    model = ScriptedModel(["The instruction is to pour the water."])
    _install(factory, model, general=False)
    try:
        with TestClient(app) as client:
            response = client.post(
                f"/api/sessions/{session_id}/messages",
                json={"message": "What is the instruction?"},
            )
        assert response.status_code == 200, response.text
        assert len(model.calls) == 1
    finally:
        _clear()
        engine.dispose()


def test_en_target_still_rejects_french_sentence(tmp_path):
    engine, factory, _ = _db_factory(tmp_path)
    session_id, _ = _seed(
        factory,
        ["I teach you the Foobar protocol."],
        _visible_filler(20),
        question="What is the Foobar protocol?",
    )
    model = ScriptedModel(["Je ne comprends pas."])
    _install(factory, model, general=False)
    try:
        with TestClient(app) as client:
            response = client.post(
                f"/api/sessions/{session_id}/messages",
                json={"message": "What is the Foobar protocol?"},
            )
        assert response.status_code == 502, response.text
        assert response.json()["detail"]["code"] == "receive_reply_language_mismatch"
        assert len(model.calls) == 1
    finally:
        _clear()
        engine.dispose()


def _role_counts(factory, session_id):
    db = factory()
    try:
        users = (
            db.query(Message)
            .filter(Message.session_id == session_id, Message.role == "user")
            .count()
        )
        assistants = (
            db.query(Message)
            .filter(
                Message.session_id == session_id,
                Message.role == "assistant",
            )
            .count()
        )
        return users, assistants
    finally:
        db.close()


def test_chinese_target_rejects_unquoted_english_rule_ordinary_path(tmp_path):
    engine, factory, _ = _db_factory(tmp_path)
    rule = "The rule is to wait for the timer."
    visible = [*_visible_filler(19), rule]
    session_id, _ = _seed(
        factory, [], visible, question="请用中文复述这条规则"
    )
    users_before, assistants_before = _role_counts(factory, session_id)
    model = ScriptedModel([rule])
    _install(factory, model, general=False)
    try:
        with TestClient(app) as client:
            response = client.post(
                f"/api/sessions/{session_id}/messages",
                json={"message": "请用中文复述这条规则"},
            )
        assert response.status_code == 502, response.text
        assert response.json()["detail"]["code"] == (
            "receive_reply_language_mismatch"
        )
        assert len(model.calls) == 1
        users, assistants = _role_counts(factory, session_id)
        assert (users, assistants) == (users_before + 1, assistants_before)
        db = factory()
        try:
            current = (
                db.query(Message)
                .filter(Message.session_id == session_id, Message.role == "user")
                .order_by(Message.id.desc())
                .first()
            )
            assert current.prompt_version_snapshot.endswith("+l1-zh")
        finally:
            db.close()
    finally:
        _clear()
        engine.dispose()


def test_chinese_target_rejects_english_rule_after_general_retrieval(tmp_path):
    engine, factory, _ = _db_factory(tmp_path)
    hidden = ["请用中文复述这条规则。 The rule is to wait for the timer."]
    visible = _visible_filler(20)
    session_id, _ = _seed(
        factory, hidden, visible, question="请用中文复述这条规则"
    )
    users_before, assistants_before = _role_counts(factory, session_id)
    model = ScriptedModel(
        [
            _decision(True, "请用中文复述这条规则", reason="missing_fact"),
            "The rule is to wait for the timer.",
        ]
    )
    _install(factory, model)
    try:
        with TestClient(app) as client:
            response = client.post(
                f"/api/sessions/{session_id}/messages",
                json={"message": "请用中文复述这条规则"},
            )
        assert response.status_code == 502, response.text
        assert len(model.calls) == 2
        final_system = model.calls[1][0][0]["content"]
        assert final_system.endswith(LANGUAGE_INSTRUCTIONS["zh"] + "\n")
        assert final_system.count(LANGUAGE_INSTRUCTIONS["zh"]) == 1
        users, assistants = _role_counts(factory, session_id)
        assert (users, assistants) == (users_before + 1, assistants_before)
        db = factory()
        try:
            current = (
                db.query(Message)
                .filter(Message.session_id == session_id, Message.role == "user")
                .order_by(Message.id.desc())
                .first()
            )
            assert current.prompt_version_snapshot.endswith(
                "programmatic-history-v1+l1-zh"
            )
        finally:
            db.close()
    finally:
        _clear()
        engine.dispose()


def test_chinese_paraphrase_and_quoted_english_source_pass(tmp_path):
    engine, factory, _ = _db_factory(tmp_path)
    rule = "The rule is to wait for the timer."
    visible = [*_visible_filler(19), rule]
    session_id, _ = _seed(
        factory, [], visible, question="请用中文复述这条规则"
    )
    users_before, assistants_before = _role_counts(factory, session_id)
    model = ScriptedModel(
        ["规则是等待计时器。", "规则是“The rule is to wait for the timer.”"]
    )
    _install(factory, model, general=False)
    try:
        with TestClient(app) as client:
            first = client.post(
                f"/api/sessions/{session_id}/messages",
                json={"message": "请用中文复述这条规则"},
            )
            second = client.post(
                f"/api/sessions/{session_id}/messages",
                json={"message": "请再复述一次这条规则"},
            )
        assert first.status_code == 200, first.text
        assert second.status_code == 200, second.text
        assert len(model.calls) == 2
        users, assistants = _role_counts(factory, session_id)
        assert (users, assistants) == (users_before + 2, assistants_before + 2)
    finally:
        _clear()
        engine.dispose()


def test_chinese_target_allows_fixed_label(tmp_path):
    engine, factory, _ = _db_factory(tmp_path)
    visible = [*_visible_filler(19), "The Foobar label is stable."]
    session_id, _ = _seed(
        factory, [], visible, question="请用中文说明 Foobar 标签"
    )
    users_before, assistants_before = _role_counts(factory, session_id)
    model = ScriptedModel(["Foobar 标签保持不变。"])
    _install(factory, model, general=False)
    try:
        with TestClient(app) as client:
            response = client.post(
                f"/api/sessions/{session_id}/messages",
                json={"message": "请用中文说明 Foobar 标签"},
            )
        assert response.status_code == 200, response.text
        assert len(model.calls) == 1
        users, assistants = _role_counts(factory, session_id)
        assert (users, assistants) == (users_before + 1, assistants_before + 1)
    finally:
        _clear()
        engine.dispose()


def test_formula_syntax_and_required_prose_are_distinguished(tmp_path):
    engine, factory, _ = _db_factory(tmp_path)
    visible = _visible_filler(20)
    session_id, _ = _seed(
        factory, [], visible, question="请给出公式"
    )
    users_before, assistants_before = _role_counts(factory, session_id)
    model = ScriptedModel(
        [
            "$\\sin(x)+\\cos(x)$",
            "$\\sin(x)+\\cos(x)$ 表示正弦与余弦之和。",
        ]
    )
    _install(factory, model, general=False)
    try:
        with TestClient(app) as client:
            first = client.post(
                f"/api/sessions/{session_id}/messages",
                json={"message": "请给出公式"},
            )
            second = client.post(
                f"/api/sessions/{session_id}/messages",
                json={"message": "请解释这个公式"},
            )
        assert first.status_code == 200, first.text
        assert second.status_code == 200, second.text
        assert len(model.calls) == 2
        users, assistants = _role_counts(factory, session_id)
        assert (users, assistants) == (users_before + 2, assistants_before + 2)
    finally:
        _clear()
        engine.dispose()


def test_english_formula_explanation_without_prose_is_rejected(tmp_path):
    engine, factory, _ = _db_factory(tmp_path)
    session_id, _ = _seed(
        factory, [], _visible_filler(20), question="Please explain this formula."
    )
    users_before, assistants_before = _role_counts(factory, session_id)
    model = ScriptedModel(["$\\sin(x)+\\cos(x)$"])
    _install(factory, model, general=False)
    try:
        with TestClient(app) as client:
            response = client.post(
                f"/api/sessions/{session_id}/messages",
                json={"message": "Please explain this formula."},
            )
        assert response.status_code == 502, response.text
        assert len(model.calls) == 1
        users, assistants = _role_counts(factory, session_id)
        assert (users, assistants) == (users_before + 1, assistants_before)
    finally:
        _clear()
        engine.dispose()


def test_english_formula_with_english_prose_passes(tmp_path):
    engine, factory, _ = _db_factory(tmp_path)
    session_id, _ = _seed(
        factory, [], _visible_filler(20), question="Please explain this formula."
    )
    users_before, assistants_before = _role_counts(factory, session_id)
    model = ScriptedModel(
        ["$\\sin(x)+\\cos(x)$ is the sum of sine and cosine."]
    )
    _install(factory, model, general=False)
    try:
        with TestClient(app) as client:
            response = client.post(
                f"/api/sessions/{session_id}/messages",
                json={"message": "Please explain this formula."},
            )
        assert response.status_code == 200, response.text
        users, assistants = _role_counts(factory, session_id)
        assert (users, assistants) == (users_before + 1, assistants_before + 1)
    finally:
        _clear()
        engine.dispose()


@pytest.mark.parametrize(
    "reply",
    [
        "Received.",
        "Divisors Must Remain Nonzero.",
        "收到。Divisors Must Remain Nonzero.",
    ],
)
def test_capitalized_english_prose_is_not_a_label(tmp_path, reply):
    engine, factory, _ = _db_factory(tmp_path)
    session_id, _ = _seed(
        factory,
        [],
        _visible_filler(20),
        question="请用中文复述这条规则",
    )
    users_before, assistants_before = _role_counts(factory, session_id)
    model = ScriptedModel([reply])
    _install(factory, model, general=False)
    try:
        with TestClient(app) as client:
            response = client.post(
                f"/api/sessions/{session_id}/messages",
                json={"message": "请用中文复述这条规则"},
            )
        assert response.status_code == 502, response.text
        assert response.json()["detail"]["code"] == (
            "receive_reply_language_mismatch"
        )
        assert len(model.calls) == 1
        users, assistants = _role_counts(factory, session_id)
        assert (users, assistants) == (
            users_before + 1,
            assistants_before,
        )
        db = factory()
        try:
            current = (
                db.query(Message)
                .filter(Message.session_id == session_id, Message.role == "user")
                .order_by(Message.id.desc())
                .first()
            )
            assert current.prompt_version_snapshot.endswith("+l1-zh")
        finally:
            db.close()
    finally:
        _clear()
        engine.dispose()


def test_chinese_confirmation_and_source_confirmed_labels_pass(tmp_path):
    engine, factory, _ = _db_factory(tmp_path)
    visible = [
        *_visible_filler(17),
        "The label is Foobar.",
        "The label is Keep Upright.",
        "API",
    ]
    session_id, _ = _seed(
        factory,
        [],
        visible,
        question="请用中文说明 Foobar 标签",
    )
    users_before, assistants_before = _role_counts(factory, session_id)
    model = ScriptedModel(
        [
            "收到。",
            "Foobar 标签保持不变。",
            "请保持 Keep Upright。",
            "调用 API。",
        ]
    )
    _install(factory, model, general=False)
    try:
        with TestClient(app) as client:
            responses = [
                client.post(
                    f"/api/sessions/{session_id}/messages",
                    json={"message": "请用中文确认"},
                ),
                client.post(
                    f"/api/sessions/{session_id}/messages",
                    json={"message": "请用中文说明 Foobar 标签"},
                ),
                client.post(
                    f"/api/sessions/{session_id}/messages",
                    json={"message": "请用中文说明 Keep Upright 标签"},
                ),
                client.post(
                    f"/api/sessions/{session_id}/messages",
                    json={"message": "请用中文说明 API 标签"},
                ),
            ]
        assert all(item.status_code == 200 for item in responses), [
            item.text for item in responses
        ]
        assert len(model.calls) == 4
        users, assistants = _role_counts(factory, session_id)
        assert (users, assistants) == (
            users_before + 4,
            assistants_before + 4,
        )
    finally:
        _clear()
        engine.dispose()


def test_label_only_request_accepts_source_confirmed_label(tmp_path):
    engine, factory, _ = _db_factory(tmp_path)
    visible = [*_visible_filler(19), "The label is Keep Upright."]
    session_id, _ = _seed(
        factory,
        [],
        visible,
        question="请只返回标签 Keep Upright",
    )
    users_before, assistants_before = _role_counts(factory, session_id)
    model = ScriptedModel(["Keep Upright"])
    _install(factory, model, general=False)
    try:
        with TestClient(app) as client:
            response = client.post(
                f"/api/sessions/{session_id}/messages",
                json={"message": "请只返回标签 Keep Upright"},
            )
        assert response.status_code == 200, response.text
        assert len(model.calls) == 1
        users, assistants = _role_counts(factory, session_id)
        assert (users, assistants) == (
            users_before + 1,
            assistants_before + 1,
        )
    finally:
        _clear()
        engine.dispose()


def test_english_label_only_request_passes_with_source_label(tmp_path):
    engine, factory, _ = _db_factory(tmp_path)
    visible = [*_visible_filler(19), "The label is Keep Upright."]
    session_id, _ = _seed(
        factory,
        [],
        visible,
        question="Please only return the label Keep Upright.",
    )
    users_before, assistants_before = _role_counts(factory, session_id)
    model = ScriptedModel(["Keep Upright"])
    _install(factory, model, general=False)
    try:
        with TestClient(app) as client:
            response = client.post(
                f"/api/sessions/{session_id}/messages",
                json={
                    "message": "Please only return the label Keep Upright."
                },
            )
        assert response.status_code == 200, response.text
        assert len(model.calls) == 1
        assert model.calls[0][0][0]["content"].endswith(
            LANGUAGE_INSTRUCTIONS["en"] + "\n"
        )
        users, assistants = _role_counts(factory, session_id)
        assert (users, assistants) == (
            users_before + 1,
            assistants_before + 1,
        )
        db = factory()
        try:
            assistant = (
                db.query(Message)
                .filter(Message.session_id == session_id, Message.role == "assistant")
                .order_by(Message.id.desc())
                .first()
            )
            assert assistant.content == "Keep Upright"
        finally:
            db.close()
    finally:
        _clear()
        engine.dispose()


def test_label_definition_does_not_protect_following_sentence(tmp_path):
    engine, factory, _ = _db_factory(tmp_path)
    visible = [
        *_visible_filler(18),
        "The label is Foobar.\n\nDivisors Must Remain Nonzero.",
    ]
    session_id, _ = _seed(
        factory,
        [],
        visible,
        question="请用中文复述规则",
    )
    users_before, assistants_before = _role_counts(factory, session_id)
    model = ScriptedModel(["收到。Divisors Must Remain Nonzero."])
    _install(factory, model, general=False)
    try:
        with TestClient(app) as client:
            response = client.post(
                f"/api/sessions/{session_id}/messages",
                json={"message": "请用中文复述规则"},
            )
        assert response.status_code == 502, response.text
        assert len(model.calls) == 1
        users, assistants = _role_counts(factory, session_id)
        assert (users, assistants) == (
            users_before + 1,
            assistants_before,
        )
        db = factory()
        try:
            current = (
                db.query(Message)
                .filter(Message.session_id == session_id, Message.role == "user")
                .order_by(Message.id.desc())
                .first()
            )
            assert current.prompt_version_snapshot.endswith("+l1-zh")
        finally:
            db.close()
    finally:
        _clear()
        engine.dispose()


def test_nearby_and_actual_label_definition_still_pass(tmp_path):
    engine, factory, _ = _db_factory(tmp_path)
    visible = [
        *_visible_filler(18),
        "The label is Foobar. Keep this instruction nearby.",
    ]
    session_id, _ = _seed(
        factory,
        [],
        visible,
        question="请用中文确认标签",
    )
    users_before, assistants_before = _role_counts(factory, session_id)
    model = ScriptedModel(["收到。Foobar。"])
    _install(factory, model, general=False)
    try:
        with TestClient(app) as client:
            response = client.post(
                f"/api/sessions/{session_id}/messages",
                json={"message": "请用中文确认标签"},
            )
        assert response.status_code == 200, response.text
        users, assistants = _role_counts(factory, session_id)
        assert (users, assistants) == (
            users_before + 1,
            assistants_before + 1,
        )
        db = factory()
        try:
            assistant = (
                db.query(Message)
                .filter(Message.session_id == session_id, Message.role == "assistant")
                .order_by(Message.id.desc())
                .first()
            )
            assert assistant.content == "收到。Foobar。"
        finally:
            db.close()
    finally:
        _clear()
        engine.dispose()


def test_source_fence_body_is_not_protected_as_bare_prose(tmp_path):
    engine, factory, _ = _db_factory(tmp_path)
    source = "```text\n这是中文回答。\n```"
    session_id, _ = _seed(
        factory,
        [],
        [*_visible_filler(19), source],
        question="Please answer.",
    )
    users_before, assistants_before = _role_counts(factory, session_id)
    model = ScriptedModel(["Got it.这是中文回答。"])
    _install(factory, model, general=False)
    try:
        with TestClient(app) as client:
            response = client.post(
                f"/api/sessions/{session_id}/messages",
                json={"message": "Please answer."},
            )
        assert response.status_code == 502, response.text
        assert len(model.calls) == 1
        users, assistants = _role_counts(factory, session_id)
        assert (users, assistants) == (
            users_before + 1,
            assistants_before,
        )
        db = factory()
        try:
            current = (
                db.query(Message)
                .filter(Message.session_id == session_id, Message.role == "user")
                .order_by(Message.id.desc())
                .first()
            )
            assert current.prompt_version_snapshot.endswith("+l1-en")
        finally:
            db.close()
    finally:
        _clear()
        engine.dispose()


def test_source_fence_and_explicit_quote_still_pass(tmp_path):
    engine, factory, _ = _db_factory(tmp_path)
    source = "```text\n这是中文回答。\n```"
    session_id, _ = _seed(
        factory,
        [],
        [*_visible_filler(19), source],
        question="Please answer.",
    )
    users_before, assistants_before = _role_counts(factory, session_id)
    model = ScriptedModel(
        [
            "Got it.\n```text\n这是中文回答。\n```",
            'Got it. "这是中文回答。"',
        ]
    )
    _install(factory, model, general=False)
    try:
        with TestClient(app) as client:
            first = client.post(
                f"/api/sessions/{session_id}/messages",
                json={"message": "Please answer."},
            )
            second = client.post(
                f"/api/sessions/{session_id}/messages",
                json={"message": "Please answer again."},
            )
        assert first.status_code == 200, first.text
        assert second.status_code == 200, second.text
        users, assistants = _role_counts(factory, session_id)
        assert (users, assistants) == (
            users_before + 2,
            assistants_before + 2,
        )
        db = factory()
        try:
            assistant_contents = [
                item.content
                for item in (
                    db.query(Message)
                    .filter(
                        Message.session_id == session_id,
                        Message.role == "assistant",
                    )
                    .order_by(Message.id)
                    .all()
                )
            ]
            assert assistant_contents == [
                "Got it.\n```text\n这是中文回答。\n```",
                'Got it. "这是中文回答。"',
            ]
        finally:
            db.close()
    finally:
        _clear()
        engine.dispose()


def test_label_candidate_stops_at_lowercase_predicate(tmp_path):
    engine, factory, _ = _db_factory(tmp_path)
    visible = [
        *_visible_filler(19),
        "The label is Foobar and it requires testing.",
    ]
    session_id, _ = _seed(factory, [], visible, question="请用中文复述规则")
    users_before, assistants_before = _role_counts(factory, session_id)
    model = ScriptedModel(["收到。Foobar and it requires testing."])
    _install(factory, model, general=False)
    try:
        with TestClient(app) as client:
            response = client.post(
                f"/api/sessions/{session_id}/messages",
                json={"message": "请用中文复述规则"},
            )
        assert response.status_code == 502, response.text
        assert response.json()["detail"]["code"] == (
            "receive_reply_language_mismatch"
        )
        assert len(model.calls) == 1
        assert model.calls[0][0][0]["content"].endswith(
            LANGUAGE_INSTRUCTIONS["zh"] + "\n"
        )
        users, assistants = _role_counts(factory, session_id)
        assert (users, assistants) == (
            users_before + 1,
            assistants_before,
        )
        db = factory()
        try:
            current = (
                db.query(Message)
                .filter(Message.session_id == session_id, Message.role == "user")
                .order_by(Message.id.desc())
                .first()
            )
            assert current.prompt_version_snapshot.endswith("+l1-zh")
        finally:
            db.close()
    finally:
        _clear()
        engine.dispose()


def test_label_candidate_requires_uppercase_candidate_range(tmp_path):
    engine, factory, _ = _db_factory(tmp_path)
    visible = [*_visible_filler(19), "The label requires testing."]
    session_id, _ = _seed(factory, [], visible, question="请用中文复述规则")
    users_before, assistants_before = _role_counts(factory, session_id)
    model = ScriptedModel(["收到。requires testing."])
    _install(factory, model, general=False)
    try:
        with TestClient(app) as client:
            response = client.post(
                f"/api/sessions/{session_id}/messages",
                json={"message": "请用中文复述规则"},
            )
        assert response.status_code == 502, response.text
        assert response.json()["detail"]["code"] == (
            "receive_reply_language_mismatch"
        )
        assert len(model.calls) == 1
        assert model.calls[0][0][0]["content"].endswith(
            LANGUAGE_INSTRUCTIONS["zh"] + "\n"
        )
        users, assistants = _role_counts(factory, session_id)
        assert (users, assistants) == (
            users_before + 1,
            assistants_before,
        )
        db = factory()
        try:
            current = (
                db.query(Message)
                .filter(Message.session_id == session_id, Message.role == "user")
                .order_by(Message.id.desc())
                .first()
            )
            assert current.prompt_version_snapshot.endswith("+l1-zh")
        finally:
            db.close()
    finally:
        _clear()
        engine.dispose()


def test_language_name_in_topic_does_not_change_target(tmp_path):
    engine, factory, _ = _db_factory(tmp_path)
    session_id, _ = _seed(
        factory,
        [],
        _visible_filler(20),
        question="Explain the Chinese remainder theorem.",
    )
    users_before, assistants_before = _role_counts(factory, session_id)
    model = ScriptedModel(["这是中文回答。"])
    _install(factory, model, general=False)
    try:
        with TestClient(app) as client:
            response = client.post(
                f"/api/sessions/{session_id}/messages",
                json={"message": "Explain the Chinese remainder theorem."},
            )
        assert response.status_code == 502, response.text
        assert response.json()["detail"]["code"] == (
            "receive_reply_language_mismatch"
        )
        assert len(model.calls) == 1
        assert model.calls[0][0][0]["content"].endswith(
            LANGUAGE_INSTRUCTIONS["en"] + "\n"
        )
        users, assistants = _role_counts(factory, session_id)
        assert (users, assistants) == (
            users_before + 1,
            assistants_before,
        )
        db = factory()
        try:
            current = (
                db.query(Message)
                .filter(Message.session_id == session_id, Message.role == "user")
                .order_by(Message.id.desc())
                .first()
            )
            assert current.prompt_version_snapshot.endswith("+l1-en")
        finally:
            db.close()
    finally:
        _clear()
        engine.dispose()


def test_english_variable_name_request_keeps_chinese_target(tmp_path):
    engine, factory, _ = _db_factory(tmp_path)
    question = "请使用英文变量名表示这个公式。"
    session_id, _ = _seed(
        factory, [], _visible_filler(20), question=question
    )
    users_before, assistants_before = _role_counts(factory, session_id)
    model = ScriptedModel(["The variable is x."])
    _install(factory, model, general=False)
    try:
        with TestClient(app) as client:
            response = client.post(
                f"/api/sessions/{session_id}/messages",
                json={"message": question},
            )
        assert response.status_code == 502, response.text
        assert response.json()["detail"]["code"] == (
            "receive_reply_language_mismatch"
        )
        assert len(model.calls) == 1
        assert model.calls[0][0][0]["content"].endswith(
            LANGUAGE_INSTRUCTIONS["zh"] + "\n"
        )
        users, assistants = _role_counts(factory, session_id)
        assert (users, assistants) == (
            users_before + 1,
            assistants_before,
        )
        db = factory()
        try:
            current = (
                db.query(Message)
                .filter(Message.session_id == session_id, Message.role == "user")
                .order_by(Message.id.desc())
                .first()
            )
            assert current.prompt_version_snapshot.endswith("+l1-zh")
        finally:
            db.close()
    finally:
        _clear()
        engine.dispose()


def test_real_reply_language_directives_still_resolve(tmp_path):
    engine, factory, _ = _db_factory(tmp_path)
    session_id, _ = _seed(
        factory,
        [],
        _visible_filler(20),
        question="Please answer in Chinese.",
    )
    users_before, assistants_before = _role_counts(factory, session_id)
    model = ScriptedModel(
        ["收到。", "这是解释。", "This is the explanation."]
    )
    _install(factory, model, general=False)
    try:
        with TestClient(app) as client:
            responses = [
                client.post(
                    f"/api/sessions/{session_id}/messages",
                    json={"message": "Please answer in Chinese."},
                ),
                client.post(
                    f"/api/sessions/{session_id}/messages",
                    json={"message": "Please explain in Chinese."},
                ),
                client.post(
                    f"/api/sessions/{session_id}/messages",
                    json={"message": "请用英文解释这个公式"},
                ),
            ]
        assert all(item.status_code == 200 for item in responses), [
            item.text for item in responses
        ]
        users, assistants = _role_counts(factory, session_id)
        assert (users, assistants) == (
            users_before + 3,
            assistants_before + 3,
        )
    finally:
        _clear()
        engine.dispose()


@pytest.mark.parametrize(
    "source, reply",
    [
        (
            "The label is Foobar and it requires testing.",
            "收到。Foobar and it requires testing.",
        ),
        (
            "标签是 Foobar and it requires testing.",
            "收到。Foobar and it requires testing.",
        ),
        (
            "Foobar and it requires testing 标签。",
            "收到。Foobar and it requires testing.",
        ),
    ],
)
def test_label_cue_languages_share_strict_candidate_boundary(tmp_path, source, reply):
    engine, factory, _ = _db_factory(tmp_path)
    visible = [*_visible_filler(19), source]
    session_id, _ = _seed(factory, [], visible, question="请用中文复述规则")
    users_before, assistants_before = _role_counts(factory, session_id)
    model = ScriptedModel([reply])
    _install(factory, model, general=False)
    try:
        with TestClient(app) as client:
            response = client.post(
                f"/api/sessions/{session_id}/messages",
                json={"message": "请用中文复述规则"},
            )
        assert response.status_code == 502, response.text
        assert response.json()["detail"]["code"] == (
            "receive_reply_language_mismatch"
        )
        assert len(model.calls) == 1
        assert model.calls[0][0][0]["content"].endswith(
            LANGUAGE_INSTRUCTIONS["zh"] + "\n"
        )
        users, assistants = _role_counts(factory, session_id)
        assert (users, assistants) == (
            users_before + 1,
            assistants_before,
        )
        db = factory()
        try:
            current = (
                db.query(Message)
                .filter(Message.session_id == session_id, Message.role == "user")
                .order_by(Message.id.desc())
                .first()
            )
            assert current.prompt_version_snapshot.endswith("+l1-zh")
        finally:
            db.close()
    finally:
        _clear()
        engine.dispose()


@pytest.mark.parametrize(
    "source, label",
    [
        ("The label is Foobar.", "Foobar"),
        ("标签是 Foobar.", "Foobar"),
        ("Foobar 标签。", "Foobar"),
        ("The label is Keep Upright.", "Keep Upright"),
        ("标签是 Keep Upright.", "Keep Upright"),
        ("Keep Upright 标签。", "Keep Upright"),
    ],
)
def test_label_candidate_boundary_accepts_valid_symmetric_labels(
    tmp_path, source, label
):
    engine, factory, _ = _db_factory(tmp_path)
    visible = [*_visible_filler(19), source]
    session_id, _ = _seed(factory, [], visible, question="请用中文确认标签")
    users_before, assistants_before = _role_counts(factory, session_id)
    model = ScriptedModel([f"收到。{label}。"])
    _install(factory, model, general=False)
    try:
        with TestClient(app) as client:
            response = client.post(
                f"/api/sessions/{session_id}/messages",
                json={"message": "请用中文确认标签"},
            )
        assert response.status_code == 200, response.text
        users, assistants = _role_counts(factory, session_id)
        assert (users, assistants) == (
            users_before + 1,
            assistants_before + 1,
        )
        db = factory()
        try:
            assistant = (
                db.query(Message)
                .filter(Message.session_id == session_id, Message.role == "assistant")
                .order_by(Message.id.desc())
                .first()
            )
            assert assistant.content == f"收到。{label}。"
        finally:
            db.close()
    finally:
        _clear()
        engine.dispose()


def test_revoke_and_replace_persistent_language_applies_current_and_next_turn(tmp_path):
    engine, factory, _ = _db_factory(tmp_path)
    visible = [*_visible_filler(19), "以后都用中文。"]
    session_id, _ = _seed(
        factory,
        [],
        visible,
        question="取消中文要求，以后用英文回复",
    )
    users_before, assistants_before = _role_counts(factory, session_id)
    model = ScriptedModel(["中文错误一。", "中文错误二。"])
    _install(factory, model, general=False)
    try:
        with TestClient(app) as client:
            first = client.post(
                f"/api/sessions/{session_id}/messages",
                json={"message": "取消中文要求，以后用英文回复"},
            )
            second = client.post(
                f"/api/sessions/{session_id}/messages",
                json={"message": "请继续解释这个问题。"},
            )
        assert first.status_code == 502, first.text
        assert second.status_code == 502, second.text
        assert first.json()["detail"]["code"] == (
            "receive_reply_language_mismatch"
        )
        assert second.json()["detail"]["code"] == (
            "receive_reply_language_mismatch"
        )
        assert len(model.calls) == 2
        for batch in model.calls:
            assert batch[0][0]["content"].endswith(
                LANGUAGE_INSTRUCTIONS["en"] + "\n"
            )
        users, assistants = _role_counts(factory, session_id)
        assert (users, assistants) == (
            users_before + 2,
            assistants_before,
        )
        db = factory()
        try:
            recent_users = (
                db.query(Message)
                .filter(Message.session_id == session_id, Message.role == "user")
                .order_by(Message.id.desc())
                .limit(2)
                .all()
            )
            assert all(
                item.prompt_version_snapshot.endswith("+l1-en")
                for item in recent_users
            )
        finally:
            db.close()
    finally:
        _clear()
        engine.dispose()


def _trace_ambiguous_context():
    visible = _visible_filler(18) + [
        "The Maple Gate rule requires testing the Amber lamp separately.",
        "The division rule requires a nonzero divisor.",
    ]
    return ["An older teacher message mentions a Cedar latch."], visible


def test_trace_metadata_wording_without_named_topic_clarifies(tmp_path):
    engine, factory, _ = _db_factory(tmp_path)
    hidden, visible = _trace_ambiguous_context()
    question = "What was the original wording of that rule?"
    session_id, _ = _seed(factory, hidden, visible, question=question)
    model = ScriptedModel(["Which rule are you referring to?"])
    service = _install(factory, model)
    db = factory()
    before = db.query(Message).count()
    db.close()
    try:
        with TestClient(app) as client:
            response = _post(client, session_id, question)
        assert response.status_code == 200, response.text
        assert len(model.calls) == 1
        assert service.last_result.counts.decision_attempts == 0
        assert service.last_result.counts.retrieval_attempts == 0
        assert service.last_result.counts.final_attempts == 1
        db = factory()
        assert db.query(Message).count() == before + 2
        db.close()
        body = response.json()
        assert body["user_message"]["prompt_version_snapshot"] == (
            "receive-teaching-v7+clarify-v1+l1-en"
        )
        assert body["assistant_message"]["prompt_version_snapshot"] == (
            "receive-teaching-v7+clarify-v1+l1-en"
        )
        final_messages = model.calls[0][0]
        evidence = json.loads(final_messages[1]["content"])
        assert evidence["retrieved_sources"] == []
        assert len(evidence["window_sources"]) == 20
        contents = " ".join(item["content"] for item in evidence["window_sources"])
        assert "Maple Gate" in contents
        assert "division rule" in contents
    finally:
        _clear()
        engine.dispose()


def test_trace_metadata_originally_call_rule_clarifies(tmp_path):
    engine, factory, _ = _db_factory(tmp_path)
    hidden, visible = _trace_ambiguous_context()
    question = "What did you originally call this rule?"
    session_id, _ = _seed(factory, hidden, visible, question=question)
    model = ScriptedModel(["Which rule should I trace?"])
    service = _install(factory, model)
    try:
        with TestClient(app) as client:
            response = _post(client, session_id, question)
        assert response.status_code == 200, response.text
        assert len(model.calls) == 1
        assert service.last_result.counts.decision_attempts == 0
        assert service.last_result.counts.retrieval_attempts == 0
        assert service.last_result.counts.final_attempts == 1
        evidence = json.loads(model.calls[0][0][1]["content"])
        assert evidence["retrieved_sources"] == []
        assert len(evidence["window_sources"]) == 20
    finally:
        _clear()
        engine.dispose()


def test_trace_named_juniper_topic_still_retrieves(tmp_path):
    engine, factory, _ = _db_factory(tmp_path)
    session_id, _ = _seed(
        factory,
        ["The Juniper rule was originally called the Cedar rule."],
        _visible_filler(20),
        question="What was the original wording of the Juniper rule?",
    )
    model = ScriptedModel(["It was called the Cedar rule. [message 1]"])
    service = _install(factory, model)
    try:
        with TestClient(app) as client:
            response = _post(
                client,
                session_id,
                "What was the original wording of the Juniper rule?",
            )
        assert response.status_code == 200, response.text
        assert service.last_result.counts.decision_attempts == 0
        assert service.last_result.counts.retrieval_attempts == 1
        assert service.last_result.counts.final_attempts == 1
        evidence = json.loads(model.calls[0][0][1]["content"])
        assert evidence["retrieved_sources"]
        assert evidence["retrieved_sources"][0]["message_id"] == 1
    finally:
        _clear()
        engine.dispose()


def test_trace_explicitly_named_wording_topic_still_retrieves(tmp_path):
    engine, factory, _ = _db_factory(tmp_path)
    session_id, _ = _seed(
        factory,
        ["The Wording protocol was originally called the Quux protocol."],
        _visible_filler(20),
        question='What was the original wording of the "Wording" protocol?',
    )
    model = ScriptedModel(["It was called the Quux protocol. [message 1]"])
    service = _install(factory, model)
    try:
        with TestClient(app) as client:
            response = _post(
                client,
                session_id,
                'What was the original wording of the "Wording" protocol?',
            )
        assert response.status_code == 200, response.text
        assert service.last_result.counts.retrieval_attempts == 1
        assert service.last_result.counts.final_attempts == 1
        evidence = json.loads(model.calls[0][0][1]["content"])
        assert evidence["retrieved_sources"][0]["message_id"] == 1
    finally:
        _clear()
        engine.dispose()


@pytest.mark.parametrize("strategy", ["a", "b", "ab"])
@pytest.mark.parametrize("route", ["fixed", "general", "clarify"])
def test_production_strategy_survives_all_history_final_paths(tmp_path, strategy, route):
    from backend.teaching_strategies import get_teaching_prompt, get_teaching_prompt_version
    engine, factory, _ = _db_factory(tmp_path)
    question = {
        "fixed": "Does Kestrel link to Lumen?",
        "general": "What is the Foobar protocol?",
        "clarify": "What was the original wording of that rule?",
    }[route]
    session_id, _ = _seed(
        factory,
        ["I teach you this fact: Kestrel links to Lumen.", "The Foobar protocol uses a blue timer."],
        _visible_filler(20), question=question,
    )
    with factory() as db:
        db.get(ChatSession, session_id).teaching_strategy = strategy
        db.commit()
    responses = ["Which rule are you referring to?"] if route == "clarify" else ["The requested teaching is available."]
    if route == "general":
        responses.insert(0, _decision(True, "Foobar protocol"))
    model = ScriptedModel(responses)
    original = main_module.chat_service
    _install(factory, model, fixed=(route == "fixed"))
    try:
        with TestClient(app) as client:
            response = _post(client, session_id, question)
        assert response.status_code == 200, response.text
        version_route = "clarify" if route == "clarify" else "history"
        expected = get_teaching_prompt_version(strategy, version_route) + "+l1-en"
        for role in ["user", "assistant"]:
            row = response.json()[role + "_message"]
            assert row["prompt_version_snapshot"] == expected
            assert row["teaching_strategy_snapshot"] == strategy
        system = model.calls[-1][0][0]["content"]
        assert system.startswith(get_teaching_prompt(strategy, history=(route != "clarify")).rstrip())
        assert system.rstrip().endswith(LANGUAGE_INSTRUCTIONS["en"])
        assert system.count(LANGUAGE_INSTRUCTIONS["en"]) == 1
    finally:
        _clear()
        main_module.chat_service = original
        engine.dispose()
