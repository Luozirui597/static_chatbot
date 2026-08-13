"""Tests for message-level model provenance snapshots.

Covers the ORM columns (all nullable, default None) and the
``MessageResponse`` triple invariant: the three snapshot fields are
either all NULL (pre-tracking messages) or all non-blank, correctly
typed values.  Nothing in between is accepted.

Every test uses a temporary SQLite file — the real ``data/chatbot.db``
is never touched.
"""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError
from sqlalchemy.orm import sessionmaker

from backend.database import create_database_engine, create_tables
from backend.models import ChatSession, Message
from backend.schemas import MessageResponse, SendMessageResponse

# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def db_session(tmp_path):
    """Yield a SQLAlchemy session backed by a temporary SQLite file."""
    db_path = tmp_path / "test.db"
    url = f"sqlite:///{db_path}"
    eng = create_database_engine(url)
    create_tables(bind=eng)
    SessionLocal = sessionmaker(
        bind=eng, autoflush=False, expire_on_commit=False,
    )
    session = SessionLocal()
    yield session
    session.close()
    eng.dispose()


def _make_session(db_session) -> int:
    s = ChatSession()
    db_session.add(s)
    db_session.commit()
    return s.id


def _base_message_kwargs(session_id: int) -> dict:
    return {
        "id": 1,
        "session_id": session_id,
        "role": "assistant",
        "content": "hello",
        "created_at": datetime(2026, 8, 13, 10, 0, 0, tzinfo=UTC),
    }


# ---------------------------------------------------------------------------
# ORM defaults
# ---------------------------------------------------------------------------


class TestMessageSnapshotORMDefaults:
    """New Message rows default to NULL snapshots."""

    def test_new_message_snapshots_default_to_none(self, db_session):
        sid = _make_session(db_session)

        m = Message(**_base_message_kwargs(sid))
        db_session.add(m)
        db_session.commit()

        assert m.llm_profile_id_snapshot is None
        assert m.llm_profile_kind_snapshot is None
        assert m.llm_model_snapshot is None

    def test_snapshot_columns_persist_null_for_explicit_values(self, db_session):
        """Explicit NULL values persist as NULL (no fabricated source)."""
        sid = _make_session(db_session)

        m = Message(
            **_base_message_kwargs(sid),
            llm_profile_id_snapshot=None,
            llm_profile_kind_snapshot=None,
            llm_model_snapshot=None,
        )
        db_session.add(m)
        db_session.commit()

        # Re-read from the database to prove the round-trip.
        db_session.expire_all()
        reloaded = db_session.get(Message, m.id)
        assert reloaded.llm_profile_id_snapshot is None
        assert reloaded.llm_profile_kind_snapshot is None
        assert reloaded.llm_model_snapshot is None

    def test_snapshot_values_round_trip(self, db_session):
        sid = _make_session(db_session)

        m = Message(
            **_base_message_kwargs(sid),
            llm_profile_id_snapshot="default",
            llm_profile_kind_snapshot="api",
            llm_model_snapshot="deepseek-v4-flash",
        )
        db_session.add(m)
        db_session.commit()

        db_session.expire_all()
        reloaded = db_session.get(Message, m.id)
        assert reloaded.llm_profile_id_snapshot == "default"
        assert reloaded.llm_profile_kind_snapshot == "api"
        assert reloaded.llm_model_snapshot == "deepseek-v4-flash"


# ---------------------------------------------------------------------------
# MessageResponse triple invariant
# ---------------------------------------------------------------------------


class TestMessageResponseSnapshotTriple:
    """The three snapshot fields are all-NULL or all-valid."""

    def test_all_null_is_valid(self):
        resp = MessageResponse(**_base_message_kwargs(1))
        assert resp.llm_profile_id_snapshot is None
        assert resp.llm_profile_kind_snapshot is None
        assert resp.llm_model_snapshot is None

    @pytest.mark.parametrize("kind", ["fake", "api", "local"])
    def test_all_valid_kinds_accepted(self, kind):
        resp = MessageResponse(
            **_base_message_kwargs(1),
            llm_profile_id_snapshot="default",
            llm_profile_kind_snapshot=kind,
            llm_model_snapshot="some-model",
        )
        assert resp.llm_profile_id_snapshot == "default"
        assert resp.llm_profile_kind_snapshot == kind
        assert resp.llm_model_snapshot == "some-model"

    def test_partial_null_rejected(self):
        with pytest.raises(ValidationError):
            MessageResponse(
                **_base_message_kwargs(1),
                llm_profile_id_snapshot="default",
                llm_profile_kind_snapshot=None,
                llm_model_snapshot=None,
            )
        with pytest.raises(ValidationError):
            MessageResponse(
                **_base_message_kwargs(1),
                llm_profile_id_snapshot=None,
                llm_profile_kind_snapshot="api",
                llm_model_snapshot="some-model",
            )
        with pytest.raises(ValidationError):
            MessageResponse(
                **_base_message_kwargs(1),
                llm_profile_id_snapshot="default",
                llm_profile_kind_snapshot="api",
                llm_model_snapshot=None,
            )

    def test_empty_string_rejected(self):
        with pytest.raises(ValidationError):
            MessageResponse(
                **_base_message_kwargs(1),
                llm_profile_id_snapshot="",
                llm_profile_kind_snapshot="api",
                llm_model_snapshot="some-model",
            )

    def test_whitespace_only_rejected(self):
        with pytest.raises(ValidationError):
            MessageResponse(
                **_base_message_kwargs(1),
                llm_profile_id_snapshot="   ",
                llm_profile_kind_snapshot="api",
                llm_model_snapshot="some-model",
            )
        with pytest.raises(ValidationError):
            MessageResponse(
                **_base_message_kwargs(1),
                llm_profile_id_snapshot="default",
                llm_profile_kind_snapshot="api",
                llm_model_snapshot="\t\n",
            )

    def test_invalid_kind_rejected(self):
        with pytest.raises(ValidationError):
            MessageResponse(
                **_base_message_kwargs(1),
                llm_profile_id_snapshot="default",
                llm_profile_kind_snapshot="quantum",
                llm_model_snapshot="some-model",
            )

    def test_overlong_values_rejected(self):
        with pytest.raises(ValidationError):
            MessageResponse(
                **_base_message_kwargs(1),
                llm_profile_id_snapshot="a" * 51,
                llm_profile_kind_snapshot="api",
                llm_model_snapshot="m",
            )
        with pytest.raises(ValidationError):
            MessageResponse(
                **_base_message_kwargs(1),
                llm_profile_id_snapshot="a",
                llm_profile_kind_snapshot="api",
                llm_model_snapshot="m" * 256,
            )

    def test_non_string_types_rejected(self):
        with pytest.raises(ValidationError):
            MessageResponse(
                **_base_message_kwargs(1),
                llm_profile_id_snapshot=5,
                llm_profile_kind_snapshot="api",
                llm_model_snapshot="m",
            )
        with pytest.raises(ValidationError):
            MessageResponse(
                **_base_message_kwargs(1),
                llm_profile_id_snapshot="default",
                llm_profile_kind_snapshot="api",
                llm_model_snapshot=42,
            )

    def test_send_message_response_includes_snapshots(self):
        user = MessageResponse(
            id=1, session_id=7, role="user", content="q",
            created_at=datetime(2026, 8, 13, 10, 0, 0, tzinfo=UTC),
            llm_profile_id_snapshot="default",
            llm_profile_kind_snapshot="api",
            llm_model_snapshot="deepseek-v4-flash",
        )
        assistant = MessageResponse(
            id=2, session_id=7, role="assistant", content="a",
            created_at=datetime(2026, 8, 13, 10, 0, 1, tzinfo=UTC),
            llm_profile_id_snapshot="default",
            llm_profile_kind_snapshot="api",
            llm_model_snapshot="deepseek-v4-flash",
        )
        send_resp = SendMessageResponse(
            user_message=user, assistant_message=assistant,
        )
        assert send_resp.user_message.llm_profile_id_snapshot == "default"
        assert send_resp.user_message.llm_profile_kind_snapshot == "api"
        assert send_resp.user_message.llm_model_snapshot == "deepseek-v4-flash"
        assert send_resp.assistant_message.llm_model_snapshot == "deepseek-v4-flash"
