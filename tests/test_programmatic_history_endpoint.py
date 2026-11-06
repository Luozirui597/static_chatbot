"""Formal endpoint coverage for programmatic Receive retrieval."""

from __future__ import annotations

import copy

from fastapi.testclient import TestClient
from sqlalchemy.orm import sessionmaker

from backend.chat_service import ChatService
from backend.database import create_database_engine, create_tables, get_db
from backend.llm_client import LLMMessage
from backend.llm_profiles import LLMProfile, LLMProfileRegistry
from backend.main import app
from backend.models import ChatSession, Message
from backend.programmatic_history.service import (
    ProgrammaticHistoryService,
    ProgrammaticSettings,
)


class Spy:
    def __init__(self):
        self.calls: list[list[LLMMessage]] = []

    async def generate(self, messages, **kwargs):
        self.calls.append(copy.deepcopy(messages))
        return "B does not link to C, per [message 2]."


def test_formal_endpoint_retrieves_only_hidden_current_session_sources(tmp_path):
    import backend.main as main_module

    db_path = tmp_path / "endpoint.sqlite3"
    engine = create_database_engine(f"sqlite:///{db_path}")
    create_tables(engine)
    factory = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    spy = Spy()
    profile = LLMProfile(
        id="default", label="API", kind="api", model="model-x",
        client=spy, is_default=True,
    )
    service = ProgrammaticHistoryService(
        ProgrammaticSettings(True, "default", "model-x")
    )
    main_module.chat_service = ChatService(
        profiles=LLMProfileRegistry([profile]),
        programmatic_history=service,
    )
    app.dependency_overrides[get_db] = lambda: (lambda db: (yield db))(factory())
    # dependency override needs a generator with cleanup
    def override_db():
        db = factory()
        try:
            yield db
        finally:
            db.close()
    app.dependency_overrides[get_db] = override_db

    db = factory()
    session = ChatSession(
        title="test", llm_profile_id="default", llm_model_snapshot="model-x"
    )
    db.add(session)
    db.flush()
    for content in [
        "I teach you this fact: B links to C.",
        "Correction: B does not link to C.",
        *[f"I teach you filler fact {i}." for i in range(20)],
    ]:
        db.add(Message(
            session_id=session.id, role="user", content=content,
            interaction_mode_snapshot="receive_teaching",
            prompt_version_snapshot="receive-teaching-v7+history-context-b1",
            llm_profile_id_snapshot="default",
            llm_profile_kind_snapshot="api",
            llm_model_snapshot="model-x",
        ))
    db.commit()
    session_id = session.id
    db.close()

    try:
        with TestClient(app) as client:
            response = client.post(
                f"/api/sessions/{session_id}/messages",
                json={"message": "Does B link to C?"},
            )
        assert response.status_code == 200
        body = response.json()
        assert body["user_message"]["prompt_version_snapshot"] == (
            "receive-teaching-v7+programmatic-history-v1+l1-en"
        )
        assert body["assistant_message"]["prompt_version_snapshot"] == (
            "receive-teaching-v7+programmatic-history-v1+l1-en"
        )
        assert len(spy.calls) == 1
        messages = spy.calls[0]
        assert messages[-1]["content"] == "Does B link to C?"
        data = messages[1]["content"]
        assert '"message_id": 1' in data
        assert '"message_id": 2' in data
        assert "B links to C" in data
        assert "does not link" in data
    finally:
        app.dependency_overrides.pop(get_db, None)
        engine.dispose()


class ErrorResultService(ProgrammaticHistoryService):
    async def retrieve_async(self, db, *, session_id, current_message_id, plan):
        from backend.programmatic_history.types import ToolResult

        return ToolResult(
            tool_call_id="programmatic_call_1",
            name="search_teaching_history",
            content={"status": "error", "error_code": "forced", "results": []},
            is_error=True,
        )


def test_programmatic_retrieval_error_returns_502_without_model(
    tmp_path, caplog
):
    import backend.main as main_module
    from backend.database import create_database_engine, create_tables, get_db
    from backend.llm_profiles import LLMProfile, LLMProfileRegistry
    from backend.models import ChatSession, Message

    db_path = tmp_path / "endpoint-error.sqlite3"
    engine = create_database_engine(f"sqlite:///{db_path}")
    create_tables(engine)
    factory = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    spy = Spy()
    profile = LLMProfile(
        id="default", label="API", kind="api", model="model-x",
        client=spy, is_default=True,
    )
    service = ErrorResultService(
        ProgrammaticSettings(True, "default", "model-x")
    )
    previous = main_module.chat_service
    main_module.chat_service = ChatService(
        profiles=LLMProfileRegistry([profile]), programmatic_history=service
    )
    def override_db():
        db = factory()
        try:
            yield db
        finally:
            db.close()
    app.dependency_overrides[get_db] = override_db
    db = factory()
    session = ChatSession(
        title="error", llm_profile_id="default", llm_model_snapshot="model-x"
    )
    db.add(session)
    db.flush()
    for content in [
        "I teach you this fact: B links to C.",
        *[f"I teach you filler fact {i}." for i in range(20)],
    ]:
        db.add(Message(
            session_id=session.id, role="user", content=content,
            interaction_mode_snapshot="receive_teaching",
            prompt_version_snapshot="receive-teaching-v7+history-context-b1",
            llm_profile_id_snapshot="default",
            llm_profile_kind_snapshot="api",
            llm_model_snapshot="model-x",
        ))
    db.commit()
    session_id = session.id
    db.close()
    try:
        with TestClient(app) as client:
            response = client.post(
                f"/api/sessions/{session_id}/messages",
                json={"message": "Does B link to C?"},
            )
        assert response.status_code == 502
        assert spy.calls == []
        db = factory()
        rows = db.query(Message).filter(Message.session_id == session_id).all()
        assert [row.role for row in rows if row.content == "Does B link to C?"] == [
            "user"
        ]
        assert "stage=retrieval" in caplog.text
        db.close()
    finally:
        app.dependency_overrides.pop(get_db, None)
        main_module.chat_service = previous
        engine.dispose()
