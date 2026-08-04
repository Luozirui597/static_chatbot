"""Tests for SQLAlchemy models, constraints, and cascade behaviour.

Every test uses a temporary SQLite file created via ``tmp_path`` — the
real ``data/chatbot.db`` is never touched.
"""

from datetime import datetime

import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker

from backend.database import create_database_engine, create_tables
from backend.models import Message, ChatSession


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def test_session(tmp_path):
    """Yield a SQLAlchemy session backed by a temporary SQLite file.

    The engine is created through the official factory so the
    ``PRAGMA foreign_keys = ON`` listener is always registered.
    """
    db_path = tmp_path / "test.db"
    url = f"sqlite:///{db_path}"
    eng = create_database_engine(url)
    create_tables(bind=eng)
    Session = sessionmaker(bind=eng, autoflush=False, expire_on_commit=False)
    session = Session()
    yield session
    session.close()
    eng.dispose()


# ---------------------------------------------------------------------------
# Basic creation
# ---------------------------------------------------------------------------


class TestCreateSession:
    """ChatSession creation and defaults."""

    def test_create_session_default_title(self, test_session):
        """A ChatSession gets the default title 'New Chat'."""
        s = ChatSession()
        test_session.add(s)
        test_session.commit()
        assert s.id is not None
        assert s.title == "New Chat"


class TestCreateMessage:
    """Message creation and basic field validation."""

    def test_create_message(self, test_session):
        """A Message can be linked to a ChatSession."""
        s = ChatSession()
        test_session.add(s)
        test_session.commit()

        m = Message(session_id=s.id, role="user", content="Hello")
        test_session.add(m)
        test_session.commit()

        assert m.id is not None
        assert m.role == "user"
        assert m.content == "Hello"
        assert m.session_id == s.id


# ---------------------------------------------------------------------------
# Relationship & ordering
# ---------------------------------------------------------------------------


class TestSessionMessagesRelationship:
    """Relationship between ChatSession and Message."""

    def test_session_messages_relationship(self, test_session):
        """A session's messages are accessible and ordered by id."""
        s = ChatSession()
        test_session.add(s)
        test_session.commit()

        m1 = Message(session_id=s.id, role="user", content="first")
        m2 = Message(session_id=s.id, role="assistant", content="second")
        m3 = Message(session_id=s.id, role="user", content="third")
        test_session.add_all([m1, m2, m3])
        test_session.commit()

        assert len(s.messages) == 3
        assert s.messages[0].content == "first"
        assert s.messages[1].content == "second"
        assert s.messages[2].content == "third"


class TestMessagesOrderedById:
    """Message ordering is explicitly set to ``Message.id``."""

    def test_messages_ordered_by_id_from_database(self, test_session):
        """After a fresh load from the database messages are sorted by id."""
        s = ChatSession()
        test_session.add(s)
        test_session.commit()

        # Create messages with intentionally non-sequential ids so we can
        # prove the ordering comes from the relationship, not coincidence.
        test_session.add_all([
            Message(id=10, session_id=s.id, role="user", content="third"),
            Message(id=5, session_id=s.id, role="user", content="first"),
            Message(id=8, session_id=s.id, role="user", content="second"),
        ])
        test_session.commit()

        # Force a fresh load from the database.
        test_session.expire_all()
        reloaded = test_session.get(ChatSession, s.id)
        assert reloaded is not None

        ids = [m.id for m in reloaded.messages]
        assert ids == [5, 8, 10]


# ---------------------------------------------------------------------------
# Cascading deletes
# ---------------------------------------------------------------------------


class TestCascadeDeleteSQL:
    """Database-level ON DELETE CASCADE via raw SQL."""

    def test_cascade_delete_via_sql_delete(self, test_session):
        """SQL DELETE on chat_sessions removes related messages."""
        s = ChatSession()
        test_session.add(s)
        test_session.commit()

        test_session.add_all([
            Message(session_id=s.id, role="user", content="msg 1"),
            Message(session_id=s.id, role="assistant", content="msg 2"),
        ])
        test_session.commit()

        # Raw SQL DELETE — this is what tests the database-level
        # ondelete="CASCADE" + PRAGMA foreign_keys = ON.
        test_session.execute(text("DELETE FROM chat_sessions"))
        test_session.commit()

        # Re-query the messages table directly to confirm they are gone.
        remaining = test_session.execute(
            text("SELECT COUNT(*) FROM messages")
        ).scalar()
        assert remaining == 0


class TestCascadeDeleteORM:
    """ORM-level cascade via ``session.delete()``."""

    def test_cascade_delete_via_orm_delete(self, test_session):
        """ORM session.delete() cascades through the relationship."""
        s = ChatSession()
        test_session.add(s)
        test_session.commit()

        test_session.add_all([
            Message(session_id=s.id, role="user", content="msg 1"),
            Message(session_id=s.id, role="assistant", content="msg 2"),
        ])
        test_session.commit()

        test_session.delete(s)
        test_session.commit()

        remaining = test_session.execute(
            text("SELECT COUNT(*) FROM messages")
        ).scalar()
        assert remaining == 0


# ---------------------------------------------------------------------------
# Constraints — role
# ---------------------------------------------------------------------------


class TestRoleConstraint:
    """``ck_messages_role`` rejects anything except 'user' / 'assistant'."""

    def test_role_constraint_rejects_invalid(self, test_session):
        """A role outside the allow-list raises IntegrityError."""
        s = ChatSession()
        test_session.add(s)
        test_session.commit()

        m = Message(session_id=s.id, role="admin", content="nope")
        test_session.add(m)
        with pytest.raises(IntegrityError):
            test_session.commit()
        test_session.rollback()


# ---------------------------------------------------------------------------
# Constraints — content
# ---------------------------------------------------------------------------


class TestContentConstraints:
    """``ck_messages_content_not_blank`` and NOT NULL reject blank content."""

    def test_content_null_rejected(self, test_session):
        """content=None is rejected by the NOT NULL constraint."""
        s = ChatSession()
        test_session.add(s)
        test_session.commit()

        m = Message(session_id=s.id, role="user", content=None)
        test_session.add(m)
        with pytest.raises(IntegrityError):
            test_session.commit()
        test_session.rollback()

    def test_content_empty_rejected(self, test_session):
        """content='' is rejected by the blank-check constraint."""
        s = ChatSession()
        test_session.add(s)
        test_session.commit()

        m = Message(session_id=s.id, role="user", content="")
        test_session.add(m)
        with pytest.raises(IntegrityError):
            test_session.commit()
        test_session.rollback()

    def test_content_spaces_rejected(self, test_session):
        r"""content='   ' (spaces only) is rejected."""
        s = ChatSession()
        test_session.add(s)
        test_session.commit()

        m = Message(session_id=s.id, role="user", content="   ")
        test_session.add(m)
        with pytest.raises(IntegrityError):
            test_session.commit()
        test_session.rollback()

    def test_content_tab_newline_rejected(self, test_session):
        r"""content='\t\n ' (mixed whitespace) is rejected."""
        s = ChatSession()
        test_session.add(s)
        test_session.commit()

        m = Message(session_id=s.id, role="user", content="\t\n ")
        test_session.add(m)
        with pytest.raises(IntegrityError):
            test_session.commit()
        test_session.rollback()

    def test_content_multiline_allowed(self, test_session):
        r"""content='line one\nline two' (non-blank with internal newline)
        is allowed."""
        s = ChatSession()
        test_session.add(s)
        test_session.commit()

        m = Message(session_id=s.id, role="user", content="line one\nline two")
        test_session.add(m)
        test_session.commit()  # must not raise

        assert m.id is not None
        assert m.content == "line one\nline two"


# ---------------------------------------------------------------------------
# Foreign key enforcement
# ---------------------------------------------------------------------------


class TestForeignKeyEnforced:
    """Foreign key constraints are actually enforced by SQLite."""

    def test_foreign_key_enforced(self, test_session):
        """A message with a non-existent session_id raises IntegrityError."""
        m = Message(session_id=999, role="user", content="orphan")
        test_session.add(m)
        with pytest.raises(IntegrityError):
            test_session.commit()
        test_session.rollback()


# ---------------------------------------------------------------------------
# Timestamps
# ---------------------------------------------------------------------------


class TestTimestamps:
    """created_at and updated_at are managed automatically."""

    def test_created_at_auto_set(self, test_session):
        """created_at is set to a non-None datetime on commit."""
        s = ChatSession()
        test_session.add(s)
        test_session.commit()

        assert s.created_at is not None
        assert isinstance(s.created_at, datetime)

    def test_updated_at_on_update(self, test_session):
        """onupdate refreshes updated_at when the row is modified."""
        s = ChatSession(title="Original")
        test_session.add(s)
        test_session.commit()

        # Pin updated_at to a known old timestamp so the test is
        # deterministic — no sleep, no reliance on clock granularity.
        far_past = datetime(2020, 1, 1, 0, 0, 0)
        s.updated_at = far_past
        test_session.commit()

        # Modify the title — onupdate should refresh updated_at.
        s.title = "Modified"
        test_session.commit()

        assert s.updated_at > far_past
