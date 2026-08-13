"""Tests for the session management API.

Every test uses a temporary SQLite file — the real ``data/chatbot.db``
is never touched.
"""

import sqlite3
from datetime import datetime

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text as sa_text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker

from backend.database import (
    create_database_engine,
    create_tables,
    get_db,
    run_migrations,
)
from backend.main import app
from backend.models import ChatSession, Message
from backend.session_titles import derive_auto_title

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
            sa_text("SELECT COUNT(*) FROM messages WHERE session_id = :sid"),
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


# ============================================================================
# PATCH /api/sessions/{session_id} — rename
# ============================================================================


class TestRenameSession:
    """Renaming a session via PATCH."""

    def test_rename_returns_updated_session(self, client):
        """PATCH returns the session with the new title."""
        created = client.post("/api/sessions")
        sid = created.json()["id"]

        resp = client.patch(
            f"/api/sessions/{sid}",
            json={"title": "My Chat"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["id"] == sid
        assert body["title"] == "My Chat"

    def test_rename_persisted(self, client):
        """After PATCH, GET returns the updated title."""
        created = client.post("/api/sessions")
        sid = created.json()["id"]

        client.patch(f"/api/sessions/{sid}", json={"title": "Persisted"})

        resp = client.get(f"/api/sessions/{sid}")
        assert resp.status_code == 200
        assert resp.json()["title"] == "Persisted"

    def test_rename_blank_title_rejected(self, client):
        """Whitespace-only title → 422."""
        created = client.post("/api/sessions")
        sid = created.json()["id"]

        resp = client.patch(
            f"/api/sessions/{sid}",
            json={"title": "   "},
        )
        assert resp.status_code == 422

    def test_rename_empty_title_rejected(self, client):
        """Empty string title → 422."""
        created = client.post("/api/sessions")
        sid = created.json()["id"]

        resp = client.patch(
            f"/api/sessions/{sid}",
            json={"title": ""},
        )
        assert resp.status_code == 422

    def test_rename_overlong_title_rejected(self, client):
        """Title > 255 characters after normalisation → 422."""
        created = client.post("/api/sessions")
        sid = created.json()["id"]

        resp = client.patch(
            f"/api/sessions/{sid}",
            json={"title": "x" * 256},
        )
        assert resp.status_code == 422

    def test_rename_not_found_404(self, client):
        """PATCH on non-existent session → 404."""
        resp = client.patch(
            "/api/sessions/9999",
            json={"title": "Nope"},
        )
        assert resp.status_code == 404
        assert resp.json() == {"detail": "Session not found"}

    def test_rename_normalises_whitespace(self, client):
        """Title with newlines, tabs, and multiple spaces is normalised."""
        created = client.post("/api/sessions")
        sid = created.json()["id"]

        resp = client.patch(
            f"/api/sessions/{sid}",
            json={"title": "  hello\n\tworld   chat  "},
        )
        assert resp.status_code == 200
        assert resp.json()["title"] == "hello world chat"

    def test_rename_preserves_existing_messages(self, client, test_session_factory):
        """Renaming does not affect messages."""
        created = client.post("/api/sessions")
        sid = created.json()["id"]

        db = test_session_factory()
        db.add(Message(session_id=sid, role="user", content="hello"))
        db.commit()
        db.close()

        client.patch(f"/api/sessions/{sid}", json={"title": "Renamed"})

        msgs = client.get(f"/api/sessions/{sid}/messages").json()
        assert len(msgs) == 1
        assert msgs[0]["content"] == "hello"


# ============================================================================
# derive_auto_title — unit tests
# ============================================================================


class TestNormalizeTitle:
    """Direct unit tests for the pure title-normalisation function."""

    def test_short_text_unchanged(self):
        """Short text without extra whitespace is returned as-is."""
        assert derive_auto_title("Hello") == "Hello"

    def test_collapses_newlines_and_tabs(self):
        """Newlines and tabs become single spaces."""
        result = derive_auto_title("Hello\nworld\tchat")
        assert result == "Hello world chat"

    def test_collapses_consecutive_spaces(self):
        """Multiple consecutive spaces collapse to one."""
        result = derive_auto_title("Hello   world")
        assert result == "Hello world"

    def test_strips_leading_trailing_whitespace(self):
        """Leading and trailing whitespace is removed."""
        result = derive_auto_title("   hello   ")
        assert result == "hello"

    def test_truncates_with_ellipsis(self):
        """Text longer than 40 chars is truncated with …."""
        long_text = "a" * 50
        result = derive_auto_title(long_text, max_chars=40)
        assert len(result) == 41  # 40 chars + ellipsis
        assert result.endswith("…")

    def test_exactly_40_chars_no_ellipsis(self):
        """Text exactly 40 chars is not truncated."""
        text = "a" * 40
        result = derive_auto_title(text, max_chars=40)
        assert result == text
        assert "…" not in result

    def test_clamped_to_255(self):
        """Result is capped at 255 characters (DB column limit)."""
        long_text = "a" * 300
        result = derive_auto_title(long_text, max_chars=40)
        assert len(result) <= 255


# ============================================================================
# Auto-title on first message
# ============================================================================


class TestAutoTitle:
    """The first user message in a session automatically sets the title."""

    def test_new_session_default_title(self, client):
        """A freshly created session has title 'New Chat'."""
        created = client.post("/api/sessions")
        assert created.json()["title"] == "New Chat"

    def test_first_message_generates_title(self, client):
        """After the first user message, the title is auto-generated."""
        created = client.post("/api/sessions")
        sid = created.json()["id"]

        client.post(
            f"/api/sessions/{sid}/messages",
            json={"message": "Explain quantum computing"},
        )

        session = client.get(f"/api/sessions/{sid}").json()
        assert session["title"] == "Explain quantum computing"

    def test_first_message_normalises_whitespace_in_title(self, client):
        """Whitespace in the first message is normalised for the title."""
        created = client.post("/api/sessions")
        sid = created.json()["id"]

        client.post(
            f"/api/sessions/{sid}/messages",
            json={"message": "  hello\n\tworld   "},
        )

        session = client.get(f"/api/sessions/{sid}").json()
        assert session["title"] == "hello world"

    def test_long_first_message_truncated_with_ellipsis(self, client):
        """A long first message is truncated at 40 chars with …."""
        created = client.post("/api/sessions")
        sid = created.json()["id"]

        long_msg = "A" * 60
        client.post(
            f"/api/sessions/{sid}/messages",
            json={"message": long_msg},
        )

        session = client.get(f"/api/sessions/{sid}").json()
        assert len(session["title"]) == 41  # 40 + …
        assert session["title"].endswith("…")
        assert session["title"][:40] == "A" * 40

    def test_second_message_does_not_change_title(self, client):
        """Only the first message triggers auto-title; second does not."""
        created = client.post("/api/sessions")
        sid = created.json()["id"]

        client.post(
            f"/api/sessions/{sid}/messages",
            json={"message": "First message"},
        )
        client.post(
            f"/api/sessions/{sid}/messages",
            json={"message": "Second message"},
        )

        session = client.get(f"/api/sessions/{sid}").json()
        assert session["title"] == "First message"

    def test_manual_rename_not_overwritten_by_auto_title(self, client):
        """After manual rename, sending a message does not overwrite."""
        created = client.post("/api/sessions")
        sid = created.json()["id"]

        # Rename manually first (before any message)
        client.patch(
            f"/api/sessions/{sid}",
            json={"title": "Custom Title"},
        )

        # Send first message
        client.post(
            f"/api/sessions/{sid}/messages",
            json={"message": "First message"},
        )

        session = client.get(f"/api/sessions/{sid}").json()
        assert session["title"] == "Custom Title"

    def test_manual_rename_after_auto_title_persists(self, client):
        """Manual rename after auto-title persists on re-fetch."""
        created = client.post("/api/sessions")
        sid = created.json()["id"]

        client.post(
            f"/api/sessions/{sid}/messages",
            json={"message": "Auto title"},
        )

        client.patch(
            f"/api/sessions/{sid}",
            json={"title": "Manual Override"},
        )

        session = client.get(f"/api/sessions/{sid}").json()
        assert session["title"] == "Manual Override"

        # Subsequent message must not overwrite
        client.post(
            f"/api/sessions/{sid}/messages",
            json={"message": "Another message"},
        )
        session2 = client.get(f"/api/sessions/{sid}").json()
        assert session2["title"] == "Manual Override"


# ============================================================================
# Manual rename to "New Chat" must not be overwritten
# ============================================================================


class TestRenameNewChat:
    """Renaming to exactly 'New Chat' must still block auto-title."""

    def test_manual_new_chat_not_overwritten(self, client):
        """PATCH title='New Chat' → send → title stays 'New Chat'."""
        created = client.post("/api/sessions")
        sid = created.json()["id"]

        client.patch(f"/api/sessions/{sid}", json={"title": "New Chat"})

        client.post(
            f"/api/sessions/{sid}/messages",
            json={"message": "Should not become title"},
        )

        session = client.get(f"/api/sessions/{sid}").json()
        assert session["title"] == "New Chat"


# ============================================================================
# Title validation edge cases
# ============================================================================


class TestRenameValidation:
    """Pydantic validation for PATCH /api/sessions/{id}."""

    def test_normalised_exactly_255_chars_accepted(self, client):
        """Title that normalises to exactly 255 chars → 200."""
        created = client.post("/api/sessions")
        sid = created.json()["id"]

        resp = client.patch(
            f"/api/sessions/{sid}",
            json={"title": "a" * 255},
        )
        assert resp.status_code == 200
        assert len(resp.json()["title"]) == 255

    def test_normalised_256_chars_rejected(self, client):
        """Title that normalises to 256 chars → 422 (no silent truncation)."""
        created = client.post("/api/sessions")
        sid = created.json()["id"]

        resp = client.patch(
            f"/api/sessions/{sid}",
            json={"title": "a" * 256},
        )
        assert resp.status_code == 422

    def test_raw_long_but_normalised_short_accepted(self, client):
        """'a' + 300 spaces normalises to 'a' → 200 OK."""
        created = client.post("/api/sessions")
        sid = created.json()["id"]

        resp = client.patch(
            f"/api/sessions/{sid}",
            json={"title": "a" + " " * 300},
        )
        assert resp.status_code == 200
        assert resp.json()["title"] == "a"

    def test_non_string_integer_rejected(self, client):
        """Integer title → 422."""
        created = client.post("/api/sessions")
        sid = created.json()["id"]

        resp = client.patch(
            f"/api/sessions/{sid}",
            json={"title": 123},
        )
        assert resp.status_code == 422

    def test_non_string_boolean_rejected(self, client):
        """Boolean title → 422."""
        created = client.post("/api/sessions")
        sid = created.json()["id"]

        resp = client.patch(
            f"/api/sessions/{sid}",
            json={"title": True},
        )
        assert resp.status_code == 422

    def test_non_string_null_rejected(self, client):
        """Null title → 422."""
        created = client.post("/api/sessions")
        sid = created.json()["id"]

        resp = client.patch(
            f"/api/sessions/{sid}",
            json={"title": None},
        )
        assert resp.status_code == 422

    def test_non_string_array_rejected(self, client):
        """Array title → 422."""
        created = client.post("/api/sessions")
        sid = created.json()["id"]

        resp = client.patch(
            f"/api/sessions/{sid}",
            json={"title": []},
        )
        assert resp.status_code == 422

    def test_non_string_object_rejected(self, client):
        """Object title → 422."""
        created = client.post("/api/sessions")
        sid = created.json()["id"]

        resp = client.patch(
            f"/api/sessions/{sid}",
            json={"title": {}},
        )
        assert resp.status_code == 422


# ============================================================================
# Schema migration tests
# ============================================================================


class TestMigration:
    """Verify run_migrations behaves correctly."""

    def test_fresh_database_runs_migration(self, tmp_path):
        """New database: create_tables + run_migrations succeeds."""
        eng = create_database_engine(f"sqlite:///{tmp_path}/fresh.db")
        try:
            create_tables(bind=eng)
            run_migrations(eng)

            with eng.begin() as conn:
                row = conn.execute(
                    sa_text(
                        "SELECT version, applied_at FROM schema_migrations "
                        "WHERE version = 'title_is_manual_v1'"
                    )
                ).fetchone()
                assert row is not None
                assert row[1] is not None  # applied_at must be set
        finally:
            eng.dispose()

    def test_old_database_without_column_gets_migrated(self, tmp_path):
        """Simulate old DB lacking title_is_manual column."""
        db_path = tmp_path / "old.db"
        # Manually create old-format tables via raw sqlite3
        raw = sqlite3.connect(str(db_path))
        raw.execute("PRAGMA foreign_keys = ON")
        raw.execute(
            "CREATE TABLE chat_sessions ("
            "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
            "  title VARCHAR(255) NOT NULL DEFAULT 'New Chat',"
            "  created_at DATETIME NOT NULL,"
            "  updated_at DATETIME NOT NULL"
            ")"
        )
        raw.execute(
            "CREATE TABLE messages ("
            "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
            "  session_id INTEGER NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,"
            "  role VARCHAR(20) NOT NULL,"
            "  content TEXT NOT NULL,"
            "  created_at DATETIME NOT NULL,"
            "  CHECK (role IN ('user', 'assistant')),"
            "  CHECK (length(trim(content, "
            "    char(9) || char(10) || char(13) || char(32))) > 0)"
            ")"
        )
        raw.execute(
            "CREATE TABLE schema_migrations ("
            "  version VARCHAR(255) PRIMARY KEY,"
            "  applied_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP"
            ")"
        )
        # Insert old sessions
        raw.execute(
            "INSERT INTO chat_sessions (title, created_at, updated_at) "
            "VALUES ('New Chat', '2026-01-01', '2026-01-01')"
        )
        raw.execute(
            "INSERT INTO chat_sessions (title, created_at, updated_at) "
            "VALUES ('Custom Title', '2026-01-02', '2026-01-02')"
        )
        raw.commit()
        raw.close()

        eng = create_database_engine(f"sqlite:///{db_path}")
        try:
            run_migrations(eng)

            with eng.begin() as conn:
                # Column exists
                cols = [
                    r[1] for r in conn.execute(
                        sa_text("PRAGMA table_info('chat_sessions')")
                    ).fetchall()
                ]
                assert "title_is_manual" in cols

                # Row with 'New Chat' → title_is_manual = 0
                row1 = conn.execute(
                    sa_text(
                        "SELECT title_is_manual FROM chat_sessions "
                        "WHERE title = 'New Chat'"
                    )
                ).fetchone()
                assert row1[0] == 0

                # Row with 'Custom Title' → title_is_manual = 1
                row2 = conn.execute(
                    sa_text(
                        "SELECT title_is_manual FROM chat_sessions "
                        "WHERE title = 'Custom Title'"
                    )
                ).fetchone()
                assert row2[0] == 1

                # Migration record exists
                row3 = conn.execute(
                    sa_text(
                        "SELECT version FROM schema_migrations "
                        "WHERE version = 'title_is_manual_v1'"
                    )
                ).fetchone()
                assert row3 is not None
        finally:
            eng.dispose()

    def test_migration_idempotent_on_second_run(self, tmp_path):
        """Running migration twice is safe — no errors, no duplicates."""
        db_path = tmp_path / "idem.db"
        eng = create_database_engine(f"sqlite:///{db_path}")
        try:
            create_tables(bind=eng)
            run_migrations(eng)  # first
            run_migrations(eng)  # second — must not raise

            with eng.begin() as conn:
                count = conn.execute(
                    sa_text(
                        "SELECT COUNT(*) FROM schema_migrations "
                        "WHERE version = 'title_is_manual_v1'"
                    )
                ).scalar()
                assert count == 1
        finally:
            eng.dispose()

    def test_migration_recovery_when_column_exists_but_record_missing(
        self, tmp_path,
    ):
        """If ALTER TABLE succeeded but the migration record is missing,
        the next run must backfill and write the record."""
        db_path = tmp_path / "partial.db"
        raw = sqlite3.connect(str(db_path))
        raw.execute("PRAGMA foreign_keys = ON")
        raw.execute(
            "CREATE TABLE chat_sessions ("
            "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
            "  title VARCHAR(255) NOT NULL DEFAULT 'New Chat',"
            "  created_at DATETIME NOT NULL,"
            "  updated_at DATETIME NOT NULL,"
            "  title_is_manual INTEGER NOT NULL DEFAULT 0"
            ")"
        )
        raw.execute(
            "CREATE TABLE messages ("
            "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
            "  session_id INTEGER NOT NULL "
            "REFERENCES chat_sessions(id) ON DELETE CASCADE,"
            "  role VARCHAR(20) NOT NULL,"
            "  content TEXT NOT NULL,"
            "  created_at DATETIME NOT NULL,"
            "  CHECK (role IN ('user', 'assistant')),"
            "  CHECK (length(trim(content, "
            "    char(9) || char(10) || char(13) || char(32))) > 0)"
            ")"
        )
        raw.execute(
            "CREATE TABLE schema_migrations ("
            "  version VARCHAR(255) PRIMARY KEY,"
            "  applied_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP"
            ")"
        )
        raw.execute(
            "INSERT INTO chat_sessions (title, created_at, updated_at) "
            "VALUES ('Custom', '2026-01-01', '2026-01-01')"
        )
        raw.commit()
        raw.close()

        eng = create_database_engine(f"sqlite:///{db_path}")
        try:
            run_migrations(eng)  # should backfill + write record

            with eng.begin() as conn:
                # title_is_manual should now be 1 for 'Custom'
                row = conn.execute(
                    sa_text(
                        "SELECT title_is_manual FROM chat_sessions "
                        "WHERE title = 'Custom'"
                    )
                ).fetchone()
                assert row[0] == 1

                # Record must exist
                row2 = conn.execute(
                    sa_text(
                        "SELECT version FROM schema_migrations "
                        "WHERE version = 'title_is_manual_v1'"
                    )
                ).fetchone()
                assert row2 is not None
        finally:
            eng.dispose()

    def test_new_session_default_title_is_manual_false(
        self, client, test_session_factory,
    ):
        """New session after migration has title_is_manual = False."""
        created = client.post("/api/sessions")
        sid = created.json()["id"]

        db = test_session_factory()
        row = db.execute(
            sa_text(
                "SELECT title_is_manual FROM chat_sessions WHERE id = :sid"
            ),
            {"sid": sid},
        ).fetchone()
        db.close()
        assert row is not None
        assert row[0] == 0

    def test_migration_skips_when_record_already_exists(self, tmp_path):
        """If migration record exists, UPDATE is not re-executed."""
        db_path = tmp_path / "skip.db"
        raw = sqlite3.connect(str(db_path))
        raw.execute("PRAGMA foreign_keys = ON")
        raw.execute(
            "CREATE TABLE chat_sessions ("
            "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
            "  title VARCHAR(255) NOT NULL DEFAULT 'New Chat',"
            "  created_at DATETIME NOT NULL,"
            "  updated_at DATETIME NOT NULL,"
            "  title_is_manual INTEGER NOT NULL DEFAULT 0"
            ")"
        )
        raw.execute(
            "CREATE TABLE messages ("
            "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
            "  session_id INTEGER NOT NULL "
            "REFERENCES chat_sessions(id) ON DELETE CASCADE,"
            "  role VARCHAR(20) NOT NULL,"
            "  content TEXT NOT NULL,"
            "  created_at DATETIME NOT NULL,"
            "  CHECK (role IN ('user', 'assistant')),"
            "  CHECK (length(trim(content, "
            "    char(9) || char(10) || char(13) || char(32))) > 0)"
            ")"
        )
        raw.execute(
            "CREATE TABLE schema_migrations ("
            "  version VARCHAR(255) PRIMARY KEY,"
            "  applied_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP"
            ")"
        )
        raw.execute(
            "INSERT INTO schema_migrations (version) "
            "VALUES ('title_is_manual_v1')"
        )
        raw.execute(
            "INSERT INTO chat_sessions "
            "(title, created_at, updated_at, title_is_manual) "
            "VALUES ('Custom', '2026-01-01', '2026-01-01', 0)"
        )
        raw.commit()
        raw.close()

        eng = create_database_engine(f"sqlite:///{db_path}")
        try:
            run_migrations(eng)

            with eng.begin() as conn:
                flag = conn.execute(
                    sa_text(
                        "SELECT title_is_manual FROM chat_sessions "
                        "WHERE title = 'Custom'"
                    )
                ).fetchone()[0]
                assert flag == 0
        finally:
            eng.dispose()

    def test_backfill_failure_no_record_then_retry_succeeds(
        self, tmp_path,
    ):
        """UPDATE fails → no record → retry succeeds after fix."""
        db_path = tmp_path / "fail.db"
        raw = sqlite3.connect(str(db_path))
        raw.execute("PRAGMA foreign_keys = ON")
        raw.execute(
            "CREATE TABLE chat_sessions ("
            "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
            "  title VARCHAR(255) NOT NULL DEFAULT 'New Chat',"
            "  created_at DATETIME NOT NULL,"
            "  updated_at DATETIME NOT NULL,"
            "  title_is_manual INTEGER NOT NULL DEFAULT 0"
            ")"
        )
        raw.execute(
            "CREATE TABLE messages ("
            "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
            "  session_id INTEGER NOT NULL "
            "REFERENCES chat_sessions(id) ON DELETE CASCADE,"
            "  role VARCHAR(20) NOT NULL,"
            "  content TEXT NOT NULL,"
            "  created_at DATETIME NOT NULL,"
            "  CHECK (role IN ('user', 'assistant')),"
            "  CHECK (length(trim(content, "
            "    char(9) || char(10) || char(13) || char(32))) > 0)"
            ")"
        )
        raw.execute(
            "CREATE TABLE schema_migrations ("
            "  version VARCHAR(255) PRIMARY KEY,"
            "  applied_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP"
            ")"
        )
        raw.execute(
            "INSERT INTO chat_sessions "
            "(title, created_at, updated_at, title_is_manual) "
            "VALUES ('Custom', '2026-01-01', '2026-01-01', 0)"
        )
        raw.execute(
            "CREATE TRIGGER fail_backfill "
            "BEFORE UPDATE OF title_is_manual ON chat_sessions "
            "BEGIN "
            "  SELECT RAISE(FAIL, 'simulated failure'); "
            "END"
        )
        raw.commit()
        raw.close()

        eng = create_database_engine(f"sqlite:///{db_path}")
        try:
            with pytest.raises(IntegrityError, match="simulated failure"):
                run_migrations(eng)

            with eng.begin() as conn:
                rec_count = conn.execute(
                    sa_text(
                        "SELECT COUNT(*) FROM schema_migrations "
                        "WHERE version = 'title_is_manual_v1'"
                    )
                ).scalar()
                assert rec_count == 0

                flag = conn.execute(
                    sa_text(
                        "SELECT title_is_manual FROM chat_sessions "
                        "WHERE title = 'Custom'"
                    )
                ).fetchone()[0]
                assert flag == 0

            raw2 = sqlite3.connect(str(db_path))
            raw2.execute("DROP TRIGGER IF EXISTS fail_backfill")
            raw2.commit()
            raw2.close()

            run_migrations(eng)

            with eng.begin() as conn:
                flag = conn.execute(
                    sa_text(
                        "SELECT title_is_manual FROM chat_sessions "
                        "WHERE title = 'Custom'"
                    )
                ).fetchone()[0]
                assert flag == 1

                rec_count = conn.execute(
                    sa_text(
                        "SELECT COUNT(*) FROM schema_migrations "
                        "WHERE version = 'title_is_manual_v1'"
                    )
                ).scalar()
                assert rec_count == 1
        finally:
            eng.dispose()


# ============================================================================
# Migration: message_llm_snapshot_v1
# ============================================================================


# Each snapshot column's real target type.  Pre-existing columns in
# test fixtures are created with these exact types — the migration only
# adds missing columns and never repairs mistyped ones.
_MESSAGE_SNAPSHOT_COL_TYPES = {
    "llm_profile_id_snapshot": "VARCHAR(50)",
    "llm_profile_kind_snapshot": "VARCHAR(20)",
    "llm_model_snapshot": "VARCHAR(255)",
}


class TestMessageSnapshotMigration:
    """The message provenance columns are added idempotently."""

    SNAPSHOT_COLS = (
        "llm_profile_id_snapshot",
        "llm_profile_kind_snapshot",
        "llm_model_snapshot",
    )

    SNAPSHOT_COL_TYPES = _MESSAGE_SNAPSHOT_COL_TYPES

    def _make_old_db(self, db_path, present_cols):
        """Build an old-format messages table with *present_cols* of
        the snapshot columns already added (simulating partial ALTER
        survival after a failed migration)."""
        raw = sqlite3.connect(str(db_path))
        raw.execute("PRAGMA foreign_keys = ON")
        raw.execute(
            "CREATE TABLE chat_sessions ("
            "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
            "  title VARCHAR(255) NOT NULL DEFAULT 'New Chat',"
            "  created_at DATETIME NOT NULL,"
            "  updated_at DATETIME NOT NULL,"
            "  title_is_manual INTEGER NOT NULL DEFAULT 0"
            ")"
        )
        cols_sql = (
            "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
            "  session_id INTEGER NOT NULL "
            "REFERENCES chat_sessions(id) ON DELETE CASCADE,"
            "  role VARCHAR(20) NOT NULL,"
            "  content TEXT NOT NULL,"
            "  created_at DATETIME NOT NULL"
        )
        for col in present_cols:
            cols_sql += f",\n  {col} {self.SNAPSHOT_COL_TYPES[col]}"
        raw.execute(f"CREATE TABLE messages ({cols_sql})")
        raw.execute(
            "CREATE TABLE schema_migrations ("
            "  version VARCHAR(255) PRIMARY KEY,"
            "  applied_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP"
            ")"
        )
        raw.execute(
            "INSERT INTO schema_migrations (version) "
            "VALUES ('title_is_manual_v1')"
        )
        raw.execute(
            "INSERT INTO schema_migrations (version) "
            "VALUES ('llm_profile_v1')"
        )
        raw.execute(
            "INSERT INTO chat_sessions (title, created_at, updated_at) "
            "VALUES ('Old Chat', '2026-01-01', '2026-01-01')"
        )
        raw.execute(
            "INSERT INTO messages (session_id, role, content, created_at) "
            "VALUES (1, 'user', 'old message', '2026-01-02')"
        )
        raw.commit()
        raw.close()

    def _cols(self, conn, table):
        return [
            r[1]
            for r in conn.execute(
                sa_text(f"PRAGMA table_info('{table}')")
            ).fetchall()
        ]

    def _record_count(self, conn, version):
        return conn.execute(
            sa_text(
                "SELECT COUNT(*) FROM schema_migrations "
                "WHERE version = :v"
            ),
            {"v": version},
        ).scalar()

    def _assert_snapshot_schema(self, conn):
        """Unified schema assertion for the three snapshot columns.

        Each column must appear exactly once, carry its real target
        type, be nullable (notnull == 0) and have no default value
        (dflt_value is NULL).  PRAGMA rows are
        (cid, name, type, notnull, dflt_value, pk).
        """
        by_name = {}
        for row in conn.execute(sa_text("PRAGMA table_info('messages')")):
            by_name.setdefault(row[1], []).append(row)

        for col, expected_type in self.SNAPSHOT_COL_TYPES.items():
            entries = by_name.get(col, [])
            assert len(entries) == 1, (
                f"{col} appears {len(entries)} times"
            )
            entry = entries[0]
            assert entry[2].upper() == expected_type, (
                f"{col} type {entry[2]!r} != {expected_type}"
            )
            assert entry[3] == 0, f"{col} notnull == {entry[3]}, expected 0"
            assert entry[4] is None, (
                f"{col} default {entry[4]!r} is not NULL"
            )

    def test_old_db_missing_all_columns(self, tmp_path):
        """All three columns missing → added, one record, old row NULL."""
        db_path = tmp_path / "none.db"
        self._make_old_db(db_path, present_cols=[])

        eng = create_database_engine(f"sqlite:///{db_path}")
        try:
            run_migrations(eng)

            with eng.begin() as conn:
                cols = self._cols(conn, "messages")
                for col in self.SNAPSHOT_COLS:
                    assert col in cols, f"{col} missing"
                assert self._record_count(conn, "message_llm_snapshot_v1") == 1
                self._assert_snapshot_schema(conn)

                row = conn.execute(sa_text(
                    "SELECT llm_profile_id_snapshot, "
                    "llm_profile_kind_snapshot, llm_model_snapshot "
                    "FROM messages WHERE content = 'old message'"
                )).fetchone()
                assert row == (None, None, None)
        finally:
            eng.dispose()

    def test_old_db_missing_one_column(self, tmp_path):
        """Two present, one missing → only the missing one added."""
        db_path = tmp_path / "one.db"
        self._make_old_db(
            db_path,
            present_cols=["llm_profile_id_snapshot", "llm_profile_kind_snapshot"],
        )

        eng = create_database_engine(f"sqlite:///{db_path}")
        try:
            run_migrations(eng)

            with eng.begin() as conn:
                cols = self._cols(conn, "messages")
                for col in self.SNAPSHOT_COLS:
                    assert cols.count(col) == 1, f"{col} count != 1"
                assert self._record_count(conn, "message_llm_snapshot_v1") == 1
                self._assert_snapshot_schema(conn)
        finally:
            eng.dispose()

    def test_old_db_missing_two_columns(self, tmp_path):
        """One present, two missing → only the missing two added."""
        db_path = tmp_path / "two.db"
        self._make_old_db(
            db_path, present_cols=["llm_model_snapshot"],
        )

        eng = create_database_engine(f"sqlite:///{db_path}")
        try:
            run_migrations(eng)

            with eng.begin() as conn:
                cols = self._cols(conn, "messages")
                for col in self.SNAPSHOT_COLS:
                    assert cols.count(col) == 1, f"{col} count != 1"
                assert self._record_count(conn, "message_llm_snapshot_v1") == 1
                self._assert_snapshot_schema(conn)
        finally:
            eng.dispose()

    def test_all_columns_exist_record_missing(self, tmp_path):
        """Columns already present, record missing → converge, no dup."""
        db_path = tmp_path / "cols.db"
        self._make_old_db(db_path, present_cols=list(self.SNAPSHOT_COLS))

        eng = create_database_engine(f"sqlite:///{db_path}")
        try:
            run_migrations(eng)

            with eng.begin() as conn:
                cols = self._cols(conn, "messages")
                for col in self.SNAPSHOT_COLS:
                    assert cols.count(col) == 1, f"{col} duplicated"
                assert self._record_count(conn, "message_llm_snapshot_v1") == 1
                self._assert_snapshot_schema(conn)
        finally:
            eng.dispose()

    def test_record_exists_skips_alter(self, tmp_path):
        """Record present → migration fully skipped, no ALTER."""
        db_path = tmp_path / "done.db"
        self._make_old_db(db_path, present_cols=list(self.SNAPSHOT_COLS))

        raw = sqlite3.connect(str(db_path))
        raw.execute(
            "INSERT INTO schema_migrations (version) "
            "VALUES ('message_llm_snapshot_v1')"
        )
        raw.commit()
        raw.close()

        eng = create_database_engine(f"sqlite:///{db_path}")
        try:
            run_migrations(eng)  # must not raise

            with eng.begin() as conn:
                cols = self._cols(conn, "messages")
                for col in self.SNAPSHOT_COLS:
                    assert cols.count(col) == 1
                assert self._record_count(conn, "message_llm_snapshot_v1") == 1
        finally:
            eng.dispose()

    def test_idempotent_on_second_run(self, tmp_path):
        """Running the migration twice leaves one record, no errors."""
        db_path = tmp_path / "idem.db"
        self._make_old_db(db_path, present_cols=[])

        eng = create_database_engine(f"sqlite:///{db_path}")
        try:
            run_migrations(eng)
            run_migrations(eng)

            with eng.begin() as conn:
                assert self._record_count(conn, "message_llm_snapshot_v1") == 1
                for col in self.SNAPSHOT_COLS:
                    assert self._cols(conn, "messages").count(col) == 1
        finally:
            eng.dispose()

    def test_fresh_db_create_tables_then_migrate(self, tmp_path):
        """create_all() provides the columns; migration only records."""
        eng = create_database_engine(f"sqlite:///{tmp_path}/fresh.db")
        try:
            create_tables(bind=eng)
            run_migrations(eng)

            with eng.begin() as conn:
                for col in self.SNAPSHOT_COLS:
                    assert col in self._cols(conn, "messages")
                assert self._record_count(conn, "message_llm_snapshot_v1") == 1
                self._assert_snapshot_schema(conn)
        finally:
            eng.dispose()

    def test_record_insert_failure_then_retry_converges(self, tmp_path):
        """Record insert fails → no record; retry converges even if the
        ALTER results survived (no duplicate columns)."""
        db_path = tmp_path / "fail.db"
        self._make_old_db(db_path, present_cols=[])

        raw = sqlite3.connect(str(db_path))
        raw.execute(
            "CREATE TRIGGER fail_message_snapshot_migration "
            "BEFORE INSERT ON schema_migrations "
            "WHEN NEW.version = 'message_llm_snapshot_v1' "
            "BEGIN "
            "  SELECT RAISE(FAIL, 'simulated failure'); "
            "END"
        )
        raw.commit()
        raw.close()

        eng = create_database_engine(f"sqlite:///{db_path}")
        try:
            with pytest.raises(IntegrityError, match="simulated failure"):
                run_migrations(eng)

            with eng.begin() as conn:
                assert self._record_count(conn, "message_llm_snapshot_v1") == 0

            # Drop the trigger and retry — the per-column checks make
            # this converge whether or not SQLite kept the ALTERs.
            raw2 = sqlite3.connect(str(db_path))
            raw2.execute(
                "DROP TRIGGER IF EXISTS fail_message_snapshot_migration"
            )
            raw2.commit()
            raw2.close()

            run_migrations(eng)

            with eng.begin() as conn:
                assert self._record_count(conn, "message_llm_snapshot_v1") == 1
                cols = self._cols(conn, "messages")
                for col in self.SNAPSHOT_COLS:
                    assert cols.count(col) == 1, f"{col} count != 1"
                self._assert_snapshot_schema(conn)
        finally:
            eng.dispose()

    def test_messages_table_missing_fails_closed(self, tmp_path):
        """A database without the messages table fails loudly and never
        records the migration; creating the table afterwards lets a
        retry converge."""
        db_path = tmp_path / "notable.db"
        raw = sqlite3.connect(str(db_path))
        raw.execute("PRAGMA foreign_keys = ON")
        raw.execute(
            "CREATE TABLE chat_sessions ("
            "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
            "  title VARCHAR(255) NOT NULL DEFAULT 'New Chat',"
            "  created_at DATETIME NOT NULL,"
            "  updated_at DATETIME NOT NULL,"
            "  title_is_manual INTEGER NOT NULL DEFAULT 0"
            ")"
        )
        # NOTE: no messages table on purpose — this fixture tests the
        # fail-closed path.
        raw.execute(
            "CREATE TABLE schema_migrations ("
            "  version VARCHAR(255) PRIMARY KEY,"
            "  applied_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP"
            ")"
        )
        raw.execute(
            "INSERT INTO schema_migrations (version) "
            "VALUES ('title_is_manual_v1')"
        )
        raw.execute(
            "INSERT INTO schema_migrations (version) "
            "VALUES ('llm_profile_v1')"
        )
        raw.commit()
        raw.close()

        eng = create_database_engine(f"sqlite:///{db_path}")
        try:
            # No messages table → fail closed, no success record.
            with pytest.raises(RuntimeError, match="messages.*missing"):
                run_migrations(eng)

            with eng.begin() as conn:
                assert self._record_count(conn, "message_llm_snapshot_v1") == 0

            # Creating the table (with all modern columns) lets the
            # retry converge to one record.
            create_tables(bind=eng)
            run_migrations(eng)

            with eng.begin() as conn:
                assert self._record_count(conn, "message_llm_snapshot_v1") == 1
                self._assert_snapshot_schema(conn)
        finally:
            eng.dispose()

    def test_does_not_break_existing_migrations(self, tmp_path):
        """title_is_manual_v1 and llm_profile_v1 records survive."""
        db_path = tmp_path / "coexist.db"
        self._make_old_db(db_path, present_cols=[])

        eng = create_database_engine(f"sqlite:///{db_path}")
        try:
            run_migrations(eng)

            with eng.begin() as conn:
                for version in (
                    "title_is_manual_v1",
                    "llm_profile_v1",
                    "message_llm_snapshot_v1",
                ):
                    assert self._record_count(conn, version) == 1, version
        finally:
            eng.dispose()
