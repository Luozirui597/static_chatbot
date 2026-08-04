"""Tests for the session management API.

Every test uses a temporary SQLite file — the real ``data/chatbot.db``
is never touched.
"""

from datetime import datetime

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text
from sqlalchemy.orm import sessionmaker

from backend.database import create_database_engine, create_tables, get_db
from backend.main import app
from backend.models import ChatSession, Message


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def test_engine(tmp_path):
    """Return a SQLAlchemy engine backed by a temporary SQLite file.

    Uses the official ``create_database_engine`` factory so the
    ``PRAGMA foreign_keys = ON`` listener is always registered.
    """
    db_path = tmp_path / "test.db"
    url = f"sqlite:///{db_path}"
    eng = create_database_engine(url)
    create_tables(bind=eng)
    yield eng
    eng.dispose()


@pytest.fixture
def test_session_factory(test_engine):
    """Return a sessionmaker bound to *test_engine*."""
    return sessionmaker(
        bind=test_engine, autoflush=False, expire_on_commit=False
    )


@pytest.fixture
def client(test_engine, test_session_factory):
    """Return a TestClient whose ``get_db`` dependency uses the temp DB."""

    def override_get_db():
        db = test_session_factory()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.pop(get_db, None)


# ---------------------------------------------------------------------------
# POST /api/sessions
# ---------------------------------------------------------------------------


class TestCreateSession:
    """Creating a new session."""

    def test_create_session_returns_201_and_valid_response(self, client):
        """POST /api/sessions returns 201 with correct fields."""
        response = client.post("/api/sessions")
        assert response.status_code == 201

        body = response.json()
        assert isinstance(body["id"], int)
        assert body["title"] == "New Chat"
        # created_at / updated_at should be valid ISO 8601 strings.
        for key in ("created_at", "updated_at"):
            assert isinstance(body[key], str)
            # Round-trip through datetime: must not raise.
            datetime.fromisoformat(body[key])

    def test_create_session_default_title(self, client):
        """A session created without explicit title defaults to 'New Chat'."""
        response = client.post("/api/sessions")
        assert response.status_code == 201
        assert response.json()["title"] == "New Chat"


# ---------------------------------------------------------------------------
# GET /api/sessions
# ---------------------------------------------------------------------------


class TestListSessions:
    """Listing all sessions."""

    def test_list_sessions_empty(self, client):
        """When no sessions exist the list is empty."""
        response = client.get("/api/sessions")
        assert response.status_code == 200
        assert response.json() == []

    def test_list_sessions_returns_all(self, client):
        """The list contains every created session."""
        for _ in range(3):
            client.post("/api/sessions")

        response = client.get("/api/sessions")
        assert response.status_code == 200
        assert len(response.json()) == 3

    def test_list_sessions_ordered_by_updated_at_desc(
        self, client, test_session_factory
    ):
        """Sessions are returned newest-first.

        Uses fixed timestamps so the test is deterministic — no sleep.
        """
        db = test_session_factory()

        # Create three sessions with known updated_at values.
        times = [
            datetime(2026, 1, 1, 10, 0, 0),
            datetime(2026, 1, 1, 12, 0, 0),  # newest
            datetime(2026, 1, 1, 11, 0, 0),
        ]
        for t in times:
            s = ChatSession(updated_at=t, created_at=t)
            db.add(s)
        db.commit()
        db.close()

        response = client.get("/api/sessions")
        data = response.json()
        assert len(data) == 3

        # Should be ordered by updated_at DESC → most recent first.
        updated_ats = [r["updated_at"] for r in data]
        assert updated_ats == sorted(updated_ats, reverse=True)


# ---------------------------------------------------------------------------
# GET /api/sessions/{session_id}
# ---------------------------------------------------------------------------


class TestGetSession:
    """Fetching a single session."""

    def test_get_session_found(self, client):
        """GET returns the requested session."""
        created = client.post("/api/sessions")
        session_id = created.json()["id"]

        response = client.get(f"/api/sessions/{session_id}")
        assert response.status_code == 200
        assert response.json()["id"] == session_id

    def test_get_session_not_found_404(self, client):
        """A non-existent session returns 404."""
        response = client.get("/api/sessions/9999")
        assert response.status_code == 404
        assert response.json() == {"detail": "Session not found"}


# ---------------------------------------------------------------------------
# GET /api/sessions/{session_id}/messages
# ---------------------------------------------------------------------------


class TestGetMessages:
    """Fetching messages for a session."""

    def test_get_messages_returns_all(self, client, test_session_factory):
        """GET messages returns every message for the session."""
        created = client.post("/api/sessions")
        session_id = created.json()["id"]

        # Insert messages directly so the test doesn't need a POST
        # messages endpoint.
        db = test_session_factory()
        db.add_all([
            Message(session_id=session_id, role="user", content="Hello"),
            Message(session_id=session_id, role="assistant", content="Hi"),
        ])
        db.commit()
        db.close()

        response = client.get(f"/api/sessions/{session_id}/messages")
        assert response.status_code == 200
        assert len(response.json()) == 2

    def test_get_messages_empty_session(self, client):
        """A session with no messages returns an empty list."""
        created = client.post("/api/sessions")
        session_id = created.json()["id"]

        response = client.get(f"/api/sessions/{session_id}/messages")
        assert response.status_code == 200
        assert response.json() == []

    def test_get_messages_session_not_found_404(self, client):
        """Messages for a non-existent session return 404."""
        response = client.get("/api/sessions/9999/messages")
        assert response.status_code == 404
        assert response.json() == {"detail": "Session not found"}

    def test_get_messages_ordered_by_id(
        self, client, test_session_factory
    ):
        """Messages are returned in id-ascending order."""
        created = client.post("/api/sessions")
        session_id = created.json()["id"]

        # Insert with non-sequential ids to prove ordering is explicit.
        db = test_session_factory()
        db.add_all([
            Message(
                id=10, session_id=session_id, role="user", content="third"
            ),
            Message(
                id=5, session_id=session_id, role="user", content="first"
            ),
            Message(
                id=8, session_id=session_id, role="user", content="second"
            ),
        ])
        db.commit()
        db.close()

        response = client.get(f"/api/sessions/{session_id}/messages")
        data = response.json()
        assert [m["id"] for m in data] == [5, 8, 10]


# ---------------------------------------------------------------------------
# DELETE /api/sessions/{session_id}
# ---------------------------------------------------------------------------


class TestDeleteSession:
    """Deleting a session."""

    def test_delete_session_returns_ok(self, client):
        """DELETE returns {"ok": true}."""
        created = client.post("/api/sessions")
        session_id = created.json()["id"]

        response = client.delete(f"/api/sessions/{session_id}")
        assert response.status_code == 200
        assert response.json() == {"ok": True}

    def test_delete_session_not_found_404(self, client):
        """Deleting a non-existent session returns 404."""
        response = client.delete("/api/sessions/9999")
        assert response.status_code == 404
        assert response.json() == {"detail": "Session not found"}

    def test_delete_session_cascades(
        self, client, test_session_factory
    ):
        """After DELETE, the session's messages are gone from the DB."""
        created = client.post("/api/sessions")
        session_id = created.json()["id"]

        # Add messages through the factory.
        db = test_session_factory()
        db.add_all([
            Message(session_id=session_id, role="user", content="msg 1"),
            Message(session_id=session_id, role="assistant", content="msg 2"),
        ])
        db.commit()
        db.close()

        # Delete via the API.
        delete_resp = client.delete(f"/api/sessions/{session_id}")
        assert delete_resp.status_code == 200

        # Verify directly against the database — no messages remain.
        db = test_session_factory()
        remaining = db.execute(
            text("SELECT COUNT(*) FROM messages WHERE session_id = :sid"),
            {"sid": session_id},
        ).scalar()
        db.close()
        assert remaining == 0


# ---------------------------------------------------------------------------
# Session isolation
# ---------------------------------------------------------------------------


class TestSessionIsolation:
    """Sessions are fully independent of each other."""

    def test_sessions_are_independent(self, client, test_session_factory):
        """Messages for session A do not leak into session B."""
        resp_a = client.post("/api/sessions")
        resp_b = client.post("/api/sessions")
        id_a = resp_a.json()["id"]
        id_b = resp_b.json()["id"]

        # Insert different messages for each session.
        db = test_session_factory()
        db.add(Message(session_id=id_a, role="user", content="for A"))
        db.add(Message(session_id=id_b, role="user", content="for B"))
        db.commit()
        db.close()

        msgs_a = client.get(f"/api/sessions/{id_a}/messages").json()
        msgs_b = client.get(f"/api/sessions/{id_b}/messages").json()

        assert len(msgs_a) == 1
        assert msgs_a[0]["content"] == "for A"
        assert msgs_a[0]["session_id"] == id_a

        assert len(msgs_b) == 1
        assert msgs_b[0]["content"] == "for B"
        assert msgs_b[0]["session_id"] == id_b
