"""Tests for POST /api/sessions/{session_id}/messages.

Every test uses a temporary SQLite file and SpyLLMClient — no real
network requests are ever made.
"""

import copy
from datetime import datetime

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import sessionmaker

from backend.chat_service import ChatService
from backend.database import create_database_engine, create_tables, get_db
from backend.exceptions import LLMError
from backend.llm_client import LLMMessage
from backend.main import app
from backend.models import ChatSession, Message
from backend.system_prompt import SYSTEM_PROMPT

# ---------------------------------------------------------------------------
# Spy LLM Client
# ---------------------------------------------------------------------------


class SpyLLMClient:
    """Test double that records every ``generate()`` call.

    Parameters
    ----------
    response:
        The string to return from ``generate()`` (default ``"test reply"``).
    error:
        If set, ``generate()`` raises this instead of returning *response*.
    """

    def __init__(
        self,
        response: str = "test reply",
        error: Exception | None = None,
    ) -> None:
        self.calls: list[list[LLMMessage]] = []
        self.response = response
        self.error = error

    async def generate(self, messages: list[LLMMessage]) -> str:
        self.calls.append(copy.deepcopy(messages))
        if self.error is not None:
            raise self.error
        return self.response


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def test_engine(tmp_path):
    """Temporary SQLite engine with all tables created."""
    db_path = tmp_path / "test.db"
    url = f"sqlite:///{db_path}"
    eng = create_database_engine(url)
    create_tables(bind=eng)
    yield eng
    eng.dispose()


@pytest.fixture
def test_session_factory(test_engine):
    """Session factory bound to the temporary engine."""
    return sessionmaker(
        bind=test_engine, autoflush=False, expire_on_commit=False
    )


@pytest.fixture
def spy_llm():
    """Fresh SpyLLMClient for each test."""
    return SpyLLMClient()


@pytest.fixture
def client(test_engine, test_session_factory, spy_llm):
    """TestClient with ``get_db`` and ``chat_service`` overridden.

    ``get_db`` is overridden via ``dependency_overrides`` to use the
    temporary SQLite database.  ``chat_service`` is replaced with a
    ``ChatService`` that uses ``SpyLLMClient`` so no real network
    requests are made.
    """
    import backend.main as main_module

    def override_get_db():
        db = test_session_factory()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db

    original_chat_service = main_module.chat_service
    main_module.chat_service = ChatService(spy_llm)

    with TestClient(app) as c:
        yield c

    app.dependency_overrides.pop(get_db, None)
    main_module.chat_service = original_chat_service


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _create_session(client: TestClient) -> int:
    """Create a session via the API and return its id."""
    resp = client.post("/api/sessions")
    assert resp.status_code == 201
    return resp.json()["id"]


# ============================================================================
# Success path
# ============================================================================


class TestSendMessageSuccess:
    """Happy-path tests for POST /api/sessions/{session_id}/messages."""

    def test_returns_200(self, client, spy_llm):
        """A valid message returns HTTP 200."""
        session_id = _create_session(client)

        resp = client.post(
            f"/api/sessions/{session_id}/messages",
            json={"message": "Explain recursion"},
        )
        assert resp.status_code == 200

    def test_response_has_user_and_assistant_messages(self, client, spy_llm):
        """The response contains both messages with correct fields."""
        session_id = _create_session(client)

        resp = client.post(
            f"/api/sessions/{session_id}/messages",
            json={"message": "Explain recursion"},
        )
        body = resp.json()

        # user_message
        assert "user_message" in body
        um = body["user_message"]
        assert isinstance(um["id"], int)
        assert um["session_id"] == session_id
        assert um["role"] == "user"
        assert um["content"] == "Explain recursion"
        assert isinstance(um["created_at"], str)
        datetime.fromisoformat(um["created_at"])

        # assistant_message
        assert "assistant_message" in body
        am = body["assistant_message"]
        assert isinstance(am["id"], int)
        assert am["session_id"] == session_id
        assert am["role"] == "assistant"
        assert am["content"] == "test reply"
        assert isinstance(am["created_at"], str)
        datetime.fromisoformat(am["created_at"])

    def test_both_messages_persisted(self, client, spy_llm, test_session_factory):
        """Both user and assistant messages are written to the database."""
        session_id = _create_session(client)

        client.post(
            f"/api/sessions/{session_id}/messages",
            json={"message": "hello"},
        )

        db = test_session_factory()
        rows = (
            db.execute(
                select(Message)
                .where(Message.session_id == session_id)
                .order_by(Message.id.asc())
            )
            .scalars()
            .all()
        )
        db.close()

        assert len(rows) == 2

    def test_roles_and_content_correct(self, client, spy_llm, test_session_factory):
        """The persisted messages have the correct role and content."""
        session_id = _create_session(client)

        client.post(
            f"/api/sessions/{session_id}/messages",
            json={"message": "hello"},
        )

        db = test_session_factory()
        rows = (
            db.execute(
                select(Message)
                .where(Message.session_id == session_id)
                .order_by(Message.id.asc())
            )
            .scalars()
            .all()
        )
        db.close()

        assert rows[0].role == "user"
        assert rows[0].content == "hello"
        assert rows[1].role == "assistant"
        assert rows[1].content == "test reply"

    def test_second_round_includes_history(self, client, spy_llm):
        """Round 2's LLM call includes round 1's user + assistant messages."""
        session_id = _create_session(client)

        # Round 1
        client.post(
            f"/api/sessions/{session_id}/messages",
            json={"message": "first question"},
        )
        # Round 2
        client.post(
            f"/api/sessions/{session_id}/messages",
            json={"message": "second question"},
        )

        msgs = spy_llm.calls[1]  # second call
        roles = [m["role"] for m in msgs]
        contents = [m["content"] for m in msgs]
        assert roles == ["system", "user", "assistant", "user"]
        assert contents == [
            SYSTEM_PROMPT,
            "first question",
            "test reply",
            "second question",
        ]

    def test_system_prompt_first_llm_message(self, client, spy_llm):
        """The system prompt is the first message sent to the LLM."""
        session_id = _create_session(client)

        client.post(
            f"/api/sessions/{session_id}/messages",
            json={"message": "hello"},
        )

        assert spy_llm.calls[0][0] == {
            "role": "system",
            "content": SYSTEM_PROMPT,
        }

    def test_current_user_is_last_llm_message(self, client, spy_llm):
        """The current user message is the last one sent to the LLM."""
        session_id = _create_session(client)

        client.post(
            f"/api/sessions/{session_id}/messages",
            json={"message": "hello"},
        )

        last = spy_llm.calls[0][-1]
        assert last == {"role": "user", "content": "hello"}

    def test_system_prompt_not_in_database(self, client, spy_llm, test_session_factory):
        """No Message row has role='system'."""
        session_id = _create_session(client)

        client.post(
            f"/api/sessions/{session_id}/messages",
            json={"message": "hello"},
        )

        db = test_session_factory()
        system_rows = (
            db.execute(
                select(Message).where(Message.role == "system")
            )
            .scalars()
            .all()
        )
        db.close()
        assert len(system_rows) == 0

    def test_updated_at_updates(self, client, spy_llm, test_session_factory):
        """ChatSession.updated_at is bumped after a successful round."""
        session_id = _create_session(client)

        # Set updated_at to a known old timestamp
        old_time = datetime(2020, 1, 1)
        db = test_session_factory()
        session = db.get(ChatSession, session_id)
        session.updated_at = old_time
        db.commit()
        db.close()

        client.post(
            f"/api/sessions/{session_id}/messages",
            json={"message": "hello"},
        )

        db = test_session_factory()
        session = db.get(ChatSession, session_id)
        assert session.updated_at is not None
        assert session.updated_at > old_time
        db.close()


# ============================================================================
# Input validation
# ============================================================================


class TestSendMessageValidation:
    """Pydantic validation rejects invalid inputs with HTTP 422."""

    def test_empty_message_422(self, client):
        """An empty message string is rejected."""
        session_id = _create_session(client)
        resp = client.post(
            f"/api/sessions/{session_id}/messages",
            json={"message": ""},
        )
        assert resp.status_code == 422

    def test_blank_message_422(self, client):
        """A whitespace-only message is rejected."""
        session_id = _create_session(client)
        resp = client.post(
            f"/api/sessions/{session_id}/messages",
            json={"message": "   "},
        )
        assert resp.status_code == 422

    def test_overlong_message_422(self, client):
        """A message exceeding 4000 characters is rejected."""
        session_id = _create_session(client)
        resp = client.post(
            f"/api/sessions/{session_id}/messages",
            json={"message": "x" * 4001},
        )
        assert resp.status_code == 422


# ============================================================================
# Error handling
# ============================================================================


class TestSendMessageErrors:
    """Error-path tests — missing sessions, LLM failures, empty replies."""

    def test_session_not_found_404(self, client):
        """A non-existent session returns 404."""
        resp = client.post(
            "/api/sessions/9999/messages",
            json={"message": "hello"},
        )
        assert resp.status_code == 404
        assert resp.json() == {"detail": "Session not found"}

    def test_llm_error_502_preserves_user(
        self, client, spy_llm, test_session_factory
    ):
        """LLMError with 502 → user saved, no assistant, HTTP 502."""
        session_id = _create_session(client)
        spy_llm.error = LLMError("Upstream failure", status_code=502)

        resp = client.post(
            f"/api/sessions/{session_id}/messages",
            json={"message": "hello"},
        )
        assert resp.status_code == 502
        assert resp.json() == {"detail": "Upstream failure"}

        db = test_session_factory()
        rows = (
            db.execute(
                select(Message)
                .where(Message.session_id == session_id)
                .order_by(Message.id.asc())
            )
            .scalars()
            .all()
        )
        assert len(rows) == 1
        assert rows[0].role == "user"
        assert rows[0].content == "hello"

        # Explicit: no assistant message
        assistant_rows = (
            db.execute(
                select(Message).where(
                    Message.session_id == session_id,
                    Message.role == "assistant",
                )
            )
            .scalars()
            .all()
        )
        db.close()
        assert len(assistant_rows) == 0

    def test_llm_error_504_mapped(
        self, client, spy_llm, test_session_factory
    ):
        """LLMError with 504 → HTTP 504, detail preserved, user saved."""
        session_id = _create_session(client)
        spy_llm.error = LLMError("Upstream API timed out", status_code=504)

        resp = client.post(
            f"/api/sessions/{session_id}/messages",
            json={"message": "hello"},
        )
        assert resp.status_code == 504
        assert resp.json() == {"detail": "Upstream API timed out"}

        # User message preserved, no assistant
        db = test_session_factory()
        rows = (
            db.execute(
                select(Message)
                .where(Message.session_id == session_id)
            )
            .scalars()
            .all()
        )
        assert len(rows) == 1
        assert rows[0].role == "user"

        # Explicit: no assistant message
        assistant_rows = (
            db.execute(
                select(Message).where(
                    Message.session_id == session_id,
                    Message.role == "assistant",
                )
            )
            .scalars()
            .all()
        )
        db.close()
        assert len(assistant_rows) == 0

    def test_empty_reply_502(self, client, spy_llm, test_session_factory):
        """Empty LLM reply → HTTP 502, user saved, no assistant."""
        session_id = _create_session(client)
        spy_llm.response = ""

        resp = client.post(
            f"/api/sessions/{session_id}/messages",
            json={"message": "hello"},
        )
        assert resp.status_code == 502
        assert "detail" in resp.json()

        # DB: only user, no assistant
        db = test_session_factory()
        rows = (
            db.execute(
                select(Message)
                .where(Message.session_id == session_id)
            )
            .scalars()
            .all()
        )
        assert len(rows) == 1
        assert rows[0].role == "user"

        # Explicit: no assistant message
        assistant_rows = (
            db.execute(
                select(Message).where(
                    Message.session_id == session_id,
                    Message.role == "assistant",
                )
            )
            .scalars()
            .all()
        )
        db.close()
        assert len(assistant_rows) == 0

    def test_blank_reply_502(self, client, spy_llm, test_session_factory):
        """Whitespace-only LLM reply → HTTP 502, user saved, no assistant."""
        session_id = _create_session(client)
        spy_llm.response = "   \n\t  "

        resp = client.post(
            f"/api/sessions/{session_id}/messages",
            json={"message": "hello"},
        )
        assert resp.status_code == 502
        assert "detail" in resp.json()

        db = test_session_factory()
        rows = (
            db.execute(
                select(Message)
                .where(Message.session_id == session_id)
            )
            .scalars()
            .all()
        )
        assert len(rows) == 1
        assert rows[0].role == "user"

        # Explicit: no assistant message
        assistant_rows = (
            db.execute(
                select(Message).where(
                    Message.session_id == session_id,
                    Message.role == "assistant",
                )
            )
            .scalars()
            .all()
        )
        db.close()
        assert len(assistant_rows) == 0


# ============================================================================
# Session isolation
# ============================================================================


class TestSessionIsolation:
    """Messages and LLM context are isolated between sessions."""

    def test_sessions_isolated_in_db(
        self, client, spy_llm, test_session_factory
    ):
        """Messages for session A do not appear in session B's data."""
        id_a = _create_session(client)
        id_b = _create_session(client)

        client.post(f"/api/sessions/{id_a}/messages", json={"message": "msg A"})
        client.post(f"/api/sessions/{id_b}/messages", json={"message": "msg B"})

        db = test_session_factory()

        msgs_a = (
            db.execute(
                select(Message)
                .where(Message.session_id == id_a)
                .order_by(Message.id.asc())
            )
            .scalars()
            .all()
        )
        msgs_b = (
            db.execute(
                select(Message)
                .where(Message.session_id == id_b)
                .order_by(Message.id.asc())
            )
            .scalars()
            .all()
        )
        db.close()

        assert len(msgs_a) == 2
        user_a = [m for m in msgs_a if m.role == "user"][0]
        assert user_a.content == "msg A"

        assert len(msgs_b) == 2
        user_b = [m for m in msgs_b if m.role == "user"][0]
        assert user_b.content == "msg B"

    def test_sessions_isolated_in_llm_context(self, client, spy_llm):
        """Session B's LLM call does not include session A's messages."""
        id_a = _create_session(client)
        id_b = _create_session(client)

        # Post to session A first
        client.post(f"/api/sessions/{id_a}/messages", json={"message": "msg A"})
        # Post to session B
        client.post(f"/api/sessions/{id_b}/messages", json={"message": "msg B"})

        # spy_llm.calls[1] is session B's call
        msgs = spy_llm.calls[1]
        contents = [m["content"] for m in msgs]
        assert "msg A" not in contents
        assert contents == [SYSTEM_PROMPT, "msg B"]


# ============================================================================
# Regression — old /api/chat
# ============================================================================


class TestRegression:
    """Ensure existing endpoints still work."""

    def test_old_chat_endpoint_unaffected(self, client):
        """POST /api/chat still returns a normal reply."""
        resp = client.post("/api/chat", json={"message": "hello"})
        assert resp.status_code == 200
        assert resp.json() == {"reply": "test reply"}
