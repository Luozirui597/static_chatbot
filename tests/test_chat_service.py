"""Tests for ChatService — unit and integration tests with SpyLLMClient.

Every test uses a temporary SQLite database and a SpyLLMClient — no
real network requests are ever made.
"""

import asyncio
import copy
from contextlib import asynccontextmanager

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import sessionmaker

from backend.chat_service import (
    MAX_HISTORY_MESSAGES,
    ChatService,
    SessionLockRegistry,
    SessionNotFoundError,
)
from backend.database import create_database_engine, create_tables
from backend.exceptions import (
    LLMError,
    LLMInvalidResponseError,
    UnknownLLMProfileError,
)
from backend.llm_client import LLMMessage
from backend.llm_profiles import SessionProfileStatus
from backend.models import ChatSession, Message
from backend.system_prompt import SYSTEM_PROMPT


# ============================================================================
# Spy LLM Client
# ============================================================================


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


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def engine(tmp_path):
    """Per-test SQLite engine with tables created."""
    db_path = tmp_path / "test.db"
    db_url = f"sqlite:///{db_path}"
    eng = create_database_engine(db_url)
    create_tables(bind=eng)
    yield eng
    eng.dispose()


@pytest.fixture
def db_session(engine):
    """Per-test SQLAlchemy session bound to the temporary engine."""
    SessionLocal = sessionmaker(
        bind=engine,
        autoflush=False,
        expire_on_commit=False,
    )
    db = SessionLocal()
    yield db
    db.close()


@pytest.fixture
def spy_llm():
    """Fresh SpyLLMClient for each test."""
    return SpyLLMClient()


# ============================================================================
# Helpers
# ============================================================================


def _create_session(db_session_ref) -> ChatSession:
    """Insert and return a new ChatSession.

    Sets ``llm_model_snapshot`` to ``"injected-test-model"`` so the
    session is compatible with the registry created by
    ``ChatService(spy)`` / ``LLMProfileRegistry.from_single_client()``.
    """
    s = ChatSession(llm_model_snapshot="injected-test-model")
    db_session_ref.add(s)
    db_session_ref.commit()
    db_session_ref.refresh(s)
    return s


def _add_messages(
    db_session_ref,
    session_id: int,
    pairs: list[tuple[str, str]],
) -> None:
    """Insert (user_content, assistant_content) pairs for *session_id*."""
    for user_text, assistant_text in pairs:
        db_session_ref.add(
            Message(session_id=session_id, role="user", content=user_text)
        )
        db_session_ref.add(
            Message(
                session_id=session_id,
                role="assistant",
                content=assistant_text,
            )
        )
    db_session_ref.commit()


# ============================================================================
# Tests — handle_message (stateless)
# ============================================================================


class TestHandleMessage:
    @pytest.mark.anyio
    async def test_sends_system_and_current_user(self, spy_llm):
        """handle_message sends [system, user] to the LLM."""
        service = ChatService(spy_llm)
        result = await service.handle_message("hello")

        assert result == "test reply"
        assert len(spy_llm.calls) == 1
        msgs = spy_llm.calls[0]
        assert len(msgs) == 2
        assert msgs[0] == {"role": "system", "content": SYSTEM_PROMPT}
        assert msgs[1] == {"role": "user", "content": "hello"}

    @pytest.mark.anyio
    async def test_empty_reply_raises(self):
        """Empty LLM reply raises LLMInvalidResponseError."""
        spy = SpyLLMClient(response="")
        service = ChatService(spy)

        with pytest.raises(LLMInvalidResponseError):
            await service.handle_message("hello")

    @pytest.mark.anyio
    async def test_blank_reply_raises(self):
        """Whitespace-only LLM reply raises LLMInvalidResponseError."""
        spy = SpyLLMClient(response="   \n\t  ")
        service = ChatService(spy)

        with pytest.raises(LLMInvalidResponseError):
            await service.handle_message("hello")


# ============================================================================
# Tests — handle_session_message (persistent)
# ============================================================================


class TestHandleSessionMessageBasic:
    """Core success-path behaviour."""

    @pytest.mark.anyio
    async def test_saves_user_and_assistant(self, db_session, spy_llm):
        """Both messages are persisted after a successful round."""
        chat_session = _create_session(db_session)
        service = ChatService(spy_llm)

        user_msg, asst_msg = await service.handle_session_message(
            chat_session.id, "hello", db_session
        )

        # Returned objects
        assert user_msg.role == "user"
        assert user_msg.content == "hello"
        assert asst_msg.role == "assistant"
        assert asst_msg.content == "test reply"

        # Database
        rows = (
            db_session.execute(
                select(Message).where(
                    Message.session_id == chat_session.id
                ).order_by(Message.id.asc())
            )
            .scalars()
            .all()
        )
        assert len(rows) == 2
        assert rows[0].role == "user"
        assert rows[1].role == "assistant"

    @pytest.mark.anyio
    async def test_returns_correct_roles(self, db_session, spy_llm):
        """Return tuple roles are user then assistant."""
        chat_session = _create_session(db_session)
        service = ChatService(spy_llm)

        user_msg, asst_msg = await service.handle_session_message(
            chat_session.id, "hi", db_session
        )

        assert user_msg.role == "user"
        assert asst_msg.role == "assistant"

    @pytest.mark.anyio
    async def test_system_prompt_is_first_llm_message(self, db_session, spy_llm):
        """The system prompt is the first message sent to the LLM."""
        chat_session = _create_session(db_session)
        service = ChatService(spy_llm)

        await service.handle_session_message(chat_session.id, "hi", db_session)

        assert spy_llm.calls[0][0] == {
            "role": "system",
            "content": SYSTEM_PROMPT,
        }

    @pytest.mark.anyio
    async def test_system_prompt_not_in_database(self, db_session, spy_llm):
        """No Message row has role='system'."""
        chat_session = _create_session(db_session)
        service = ChatService(spy_llm)

        await service.handle_session_message(chat_session.id, "hi", db_session)

        system_rows = (
            db_session.execute(
                select(Message).where(Message.role == "system")
            )
            .scalars()
            .all()
        )
        assert len(system_rows) == 0

    @pytest.mark.anyio
    async def test_current_user_is_last_llm_message(self, db_session, spy_llm):
        """The current user message is the last one sent to the LLM."""
        chat_session = _create_session(db_session)
        service = ChatService(spy_llm)

        await service.handle_session_message(chat_session.id, "hello", db_session)

        last = spy_llm.calls[0][-1]
        assert last == {"role": "user", "content": "hello"}


# ============================================================================
# Tests — multi-turn history
# ============================================================================


class TestMultiTurnHistory:
    @pytest.mark.anyio
    async def test_second_round_includes_first_round(self, db_session, spy_llm):
        """Round 2 sends first round's user + assistant as context."""
        chat_session = _create_session(db_session)
        service = ChatService(spy_llm)

        # Round 1
        await service.handle_session_message(chat_session.id, "q1", db_session)
        # Round 2
        await service.handle_session_message(chat_session.id, "q2", db_session)

        msgs = spy_llm.calls[1]
        roles = [m["role"] for m in msgs]
        contents = [m["content"] for m in msgs]
        assert roles == ["system", "user", "assistant", "user"]
        assert contents == [SYSTEM_PROMPT, "q1", "test reply", "q2"]

    @pytest.mark.anyio
    async def test_history_ordered_by_id_asc(self, db_session, spy_llm):
        """History messages are sent in chronological (id ASC) order."""
        chat_session = _create_session(db_session)
        _add_messages(
            db_session,
            chat_session.id,
            [("q1", "a1"), ("q2", "a2"), ("q3", "a3")],
        )
        service = ChatService(spy_llm)

        await service.handle_session_message(chat_session.id, "q4", db_session)

        # Non-system messages in order
        msgs = spy_llm.calls[0][1:]  # skip system
        contents = [m["content"] for m in msgs]
        assert contents == ["q1", "a1", "q2", "a2", "q3", "a3", "q4"]

    @pytest.mark.anyio
    async def test_max_20_history_messages(self, db_session, spy_llm):
        """At most 20 previous messages are sent."""
        chat_session = _create_session(db_session)
        # Insert 25 user/assistant pairs = 50 history messages
        pairs = [(f"q{i}", f"a{i}") for i in range(25)]
        _add_messages(db_session, chat_session.id, pairs)
        service = ChatService(spy_llm)

        await service.handle_session_message(chat_session.id, "current", db_session)

        non_system = spy_llm.calls[0][1:]  # skip system prompt
        # 20 history + 1 current user = 21 non-system messages
        assert len(non_system) == MAX_HISTORY_MESSAGES + 1

    @pytest.mark.anyio
    async def test_current_user_not_counted_in_history_limit(self, db_session, spy_llm):
        """The current user message is extra, not part of the 20 limit."""
        chat_session = _create_session(db_session)
        # Insert exactly 20 history messages
        pairs = [(f"q{i}", f"a{i}") for i in range(10)]
        _add_messages(db_session, chat_session.id, pairs)
        service = ChatService(spy_llm)

        await service.handle_session_message(chat_session.id, "current", db_session)

        non_system = spy_llm.calls[0][1:]
        # 20 history + 1 current = 21
        assert len(non_system) == 21
        # Last must be the current user
        assert non_system[-1] == {"role": "user", "content": "current"}

    @pytest.mark.anyio
    async def test_oldest_history_excluded_when_over_limit(self, db_session, spy_llm):
        """When history exceeds 20, the oldest messages are dropped."""
        chat_session = _create_session(db_session)
        # Insert 25 pairs = 50 history messages
        pairs = [(f"q{i}", f"a{i}") for i in range(25)]
        _add_messages(db_session, chat_session.id, pairs)
        service = ChatService(spy_llm)

        await service.handle_session_message(chat_session.id, "current", db_session)

        non_system = spy_llm.calls[0][1:]
        # First non-system message should be q15 (oldest 5 pairs = 10 msgs dropped)
        assert non_system[0]["content"] == "q15"
        assert non_system[0]["role"] == "user"

    @pytest.mark.anyio
    async def test_database_saves_all_messages(self, db_session, spy_llm):
        """Database stores *all* messages regardless of the LLM window."""
        chat_session = _create_session(db_session)
        pairs = [(f"q{i}", f"a{i}") for i in range(25)]
        _add_messages(db_session, chat_session.id, pairs)
        service = ChatService(spy_llm)

        await service.handle_session_message(chat_session.id, "current", db_session)

        count = db_session.execute(
            select(func.count()).select_from(Message).where(
                Message.session_id == chat_session.id
            )
        ).scalar()
        # 25*2 history + 1 current user + 1 assistant = 52
        assert count == 52

    @pytest.mark.anyio
    async def test_different_sessions_isolated(self, db_session, spy_llm):
        """Messages from session A never leak into session B."""
        session_a = _create_session(db_session)
        session_b = _create_session(db_session)
        service = ChatService(spy_llm)

        # Add history in session A
        _add_messages(db_session, session_a.id, [("qa1", "aa1")])
        # One round in session A
        await service.handle_session_message(session_a.id, "qa2", db_session)

        # First round in session B — should see NO history
        await service.handle_session_message(session_b.id, "qb1", db_session)

        msgs = spy_llm.calls[1]  # session B's call
        roles = [m["role"] for m in msgs]
        contents = [m["content"] for m in msgs]
        assert roles == ["system", "user"]
        assert contents == [SYSTEM_PROMPT, "qb1"]


# ============================================================================
# Tests — error handling
# ============================================================================


class TestLLMErrorHandling:
    @pytest.mark.anyio
    async def test_llm_error_preserves_user_message(self, db_session):
        """When the LLM raises, the user message stays in the database."""
        chat_session = _create_session(db_session)
        llm_error = LLMError("upstream failure", status_code=502)
        spy = SpyLLMClient(error=llm_error)
        service = ChatService(spy)

        with pytest.raises(LLMError):
            await service.handle_session_message(chat_session.id, "hello", db_session)

        # Transaction must be closed after rollback — check BEFORE any new
        # query that would start a fresh transaction.
        assert db_session.in_transaction() is False

        rows = (
            db_session.execute(
                select(Message).where(
                    Message.session_id == chat_session.id
                )
            )
            .scalars()
            .all()
        )
        assert len(rows) == 1
        assert rows[0].role == "user"
        assert rows[0].content == "hello"

    @pytest.mark.anyio
    async def test_llm_error_no_assistant_message(self, db_session):
        """When the LLM raises, no assistant message is persisted."""
        chat_session = _create_session(db_session)
        spy = SpyLLMClient(error=LLMError("fail", status_code=502))
        service = ChatService(spy)

        with pytest.raises(LLMError):
            await service.handle_session_message(chat_session.id, "hello", db_session)

        assistant_rows = (
            db_session.execute(
                select(Message).where(
                    Message.session_id == chat_session.id,
                    Message.role == "assistant",
                )
            )
            .scalars()
            .all()
        )
        assert len(assistant_rows) == 0

    @pytest.mark.anyio
    async def test_llm_error_updates_session_timestamp(self, db_session):
        """ChatSession.updated_at is updated even when the LLM fails."""
        chat_session = _create_session(db_session)
        original_updated_at = chat_session.updated_at
        spy = SpyLLMClient(error=LLMError("fail", status_code=502))
        service = ChatService(spy)

        with pytest.raises(LLMError):
            await service.handle_session_message(chat_session.id, "hello", db_session)

        # Re-read from DB
        db_session.refresh(chat_session)
        assert chat_session.updated_at is not None
        # Phase 1 commit updated the timestamp
        assert chat_session.updated_at > original_updated_at

    @pytest.mark.anyio
    async def test_empty_reply_preserves_user_message(self, db_session):
        """Empty LLM reply → user saved, no assistant, LLMInvalidResponseError."""
        chat_session = _create_session(db_session)
        spy = SpyLLMClient(response="")
        service = ChatService(spy)

        with pytest.raises(LLMInvalidResponseError):
            await service.handle_session_message(chat_session.id, "hello", db_session)

        assert db_session.in_transaction() is False

        rows = (
            db_session.execute(
                select(Message).where(
                    Message.session_id == chat_session.id
                )
            )
            .scalars()
            .all()
        )
        assert len(rows) == 1
        assert rows[0].role == "user"

    @pytest.mark.anyio
    async def test_blank_reply_preserves_user_message(self, db_session):
        """Whitespace-only LLM reply → user saved, no assistant, error."""
        chat_session = _create_session(db_session)
        spy = SpyLLMClient(response="   \n\t  ")
        service = ChatService(spy)

        with pytest.raises(LLMInvalidResponseError):
            await service.handle_session_message(chat_session.id, "hello", db_session)

        assert db_session.in_transaction() is False

        rows = (
            db_session.execute(
                select(Message).where(
                    Message.session_id == chat_session.id
                )
            )
            .scalars()
            .all()
        )
        assert len(rows) == 1
        assert rows[0].role == "user"


# ============================================================================
# Tests — transaction failure
# ============================================================================


class TestTransactionFailure:
    @pytest.mark.anyio
    async def test_phase1_commit_failure_does_not_call_llm(
        self, db_session, spy_llm, monkeypatch
    ):
        """If Phase 1 commit fails, the LLM is never called."""
        chat_session = _create_session(db_session)
        service = ChatService(spy_llm)

        def _failing_commit():
            raise RuntimeError("simulated DB failure")

        monkeypatch.setattr(db_session, "commit", _failing_commit)

        with pytest.raises(RuntimeError, match="simulated DB failure"):
            await service.handle_session_message(
                chat_session.id, "hello", db_session
            )

        assert len(spy_llm.calls) == 0

    @pytest.mark.anyio
    async def test_phase2_failure_db_session_usable(
        self, db_session, spy_llm, monkeypatch
    ):
        """After Phase 2 commit fails, the db session is still usable."""
        chat_session = _create_session(db_session)
        service = ChatService(spy_llm)

        call_count = 0
        original_commit = db_session.commit

        def _selective_failing_commit():
            nonlocal call_count
            call_count += 1
            # Phase 1 (user + optional auto-title) commit (=1)
            # succeeds; Phase 2 (assistant) commit (=2) fails.
            if call_count == 2:
                raise RuntimeError("simulated DB failure on second commit")
            return original_commit()

        monkeypatch.setattr(db_session, "commit", _selective_failing_commit)

        with pytest.raises(RuntimeError, match="simulated DB failure"):
            await service.handle_session_message(
                chat_session.id, "hello", db_session
            )

        assert db_session.in_transaction() is False

        # User message should be preserved (first commit succeeded)
        rows = (
            db_session.execute(
                select(Message).where(
                    Message.session_id == chat_session.id
                ).order_by(Message.id.asc())
            )
            .scalars()
            .all()
        )
        assert len(rows) == 1
        assert rows[0].role == "user"
        assert rows[0].content == "hello"

        # DB session is still usable — can query
        count = db_session.execute(
            select(func.count()).select_from(ChatSession)
        ).scalar()
        assert count >= 1


# ============================================================================
# Message provenance snapshots
# ============================================================================


class TestMessageProvenanceSnapshots:
    """handle_session_message writes the captured profile's snapshot
    triple onto both the user and the assistant message."""

    def _read_messages(self, db_session, session_id):
        """Re-read all messages from the database, ordered by id."""
        db_session.expire_all()
        return (
            db_session.execute(
                select(Message)
                .where(Message.session_id == session_id)
                .order_by(Message.id.asc())
            )
            .scalars()
            .all()
        )

    @pytest.mark.anyio
    async def test_default_profile_success_writes_full_triple(
        self, db_session, spy_llm,
    ):
        """Both messages carry (default, api, injected-test-model)."""
        chat_session = _create_session(db_session)
        service = ChatService(spy_llm)

        await service.handle_session_message(
            chat_session.id, "hello", db_session,
        )

        rows = self._read_messages(db_session, chat_session.id)
        assert len(rows) == 2
        for msg in rows:
            assert msg.llm_profile_id_snapshot == "default"
            assert msg.llm_profile_kind_snapshot == "api"
            assert msg.llm_model_snapshot == "injected-test-model"

    @pytest.mark.anyio
    async def test_local_profile_session_writes_local_triple(self, db_session):
        """A session bound to a local profile stores that exact triple."""
        from backend.llm_profiles import LLMProfile, LLMProfileRegistry

        default_spy = SpyLLMClient(response="default reply")
        local_spy = SpyLLMClient(response="local reply")
        registry = LLMProfileRegistry([
            LLMProfile(
                id="default", label="Default", kind="fake",
                model="fake", client=default_spy, is_default=True,
            ),
            LLMProfile(
                id="local", label="Local", kind="local",
                model="qwen3.5:4b", client=local_spy, is_default=False,
            ),
        ])
        service = ChatService(profiles=registry)

        session = ChatSession(
            llm_profile_id="local",
            llm_model_snapshot="qwen3.5:4b",
        )
        db_session.add(session)
        db_session.commit()
        db_session.refresh(session)

        await service.handle_session_message(session.id, "hello", db_session)

        rows = self._read_messages(db_session, session.id)
        assert len(rows) == 2
        for msg in rows:
            assert msg.llm_profile_id_snapshot == "local"
            assert msg.llm_profile_kind_snapshot == "local"
            assert msg.llm_model_snapshot == "qwen3.5:4b"

    @pytest.mark.anyio
    async def test_user_and_assistant_triples_identical(
        self, db_session, spy_llm,
    ):
        """Both messages carry exactly the same triple values."""
        chat_session = _create_session(db_session)
        service = ChatService(spy_llm)

        await service.handle_session_message(
            chat_session.id, "hello", db_session,
        )

        rows = self._read_messages(db_session, chat_session.id)
        user = next(m for m in rows if m.role == "user")
        assistant = next(m for m in rows if m.role == "assistant")
        assert (
            user.llm_profile_id_snapshot,
            user.llm_profile_kind_snapshot,
            user.llm_model_snapshot,
        ) == (
            assistant.llm_profile_id_snapshot,
            assistant.llm_profile_kind_snapshot,
            assistant.llm_model_snapshot,
        )

    @pytest.mark.anyio
    async def test_llm_error_keeps_user_message_with_triple(self, db_session):
        """When the LLM raises, only the user message persists — with a
        complete, accurate snapshot."""
        chat_session = _create_session(db_session)
        spy = SpyLLMClient(error=LLMError("upstream failure", status_code=502))
        service = ChatService(spy)

        with pytest.raises(LLMError):
            await service.handle_session_message(
                chat_session.id, "hello", db_session,
            )

        rows = self._read_messages(db_session, chat_session.id)
        assert len(rows) == 1
        assert rows[0].role == "user"
        assert rows[0].llm_profile_id_snapshot == "default"
        assert rows[0].llm_profile_kind_snapshot == "api"
        assert rows[0].llm_model_snapshot == "injected-test-model"

    @pytest.mark.anyio
    async def test_blank_reply_keeps_user_message_with_triple(self, db_session):
        """A blank reply keeps only the user message with its triple."""
        chat_session = _create_session(db_session)
        spy = SpyLLMClient(response="   \n\t  ")
        service = ChatService(spy)

        with pytest.raises(LLMInvalidResponseError):
            await service.handle_session_message(
                chat_session.id, "hello", db_session,
            )

        rows = self._read_messages(db_session, chat_session.id)
        assert len(rows) == 1
        assert rows[0].role == "user"
        assert rows[0].llm_profile_id_snapshot == "default"
        assert rows[0].llm_profile_kind_snapshot == "api"
        assert rows[0].llm_model_snapshot == "injected-test-model"

    @pytest.mark.anyio
    async def test_phase1_commit_failure_no_llm_no_messages(
        self, db_session, spy_llm, monkeypatch,
    ):
        """A Phase 1 commit failure calls no LLM and persists nothing."""
        chat_session = _create_session(db_session)
        service = ChatService(spy_llm)

        def _failing_commit():
            raise RuntimeError("simulated DB failure")

        monkeypatch.setattr(db_session, "commit", _failing_commit)

        with pytest.raises(RuntimeError, match="simulated DB failure"):
            await service.handle_session_message(
                chat_session.id, "hello", db_session,
            )

        assert len(spy_llm.calls) == 0

        rows = self._read_messages(db_session, chat_session.id)
        assert len(rows) == 0

    @pytest.mark.anyio
    async def test_phase2_commit_failure_keeps_user_with_triple(
        self, db_session, spy_llm, monkeypatch,
    ):
        """A Phase 2 commit failure keeps the user message and its
        snapshot; no assistant message survives."""
        chat_session = _create_session(db_session)
        service = ChatService(spy_llm)

        call_count = 0
        original_commit = db_session.commit

        def _selective_failing_commit():
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise RuntimeError("simulated DB failure on second commit")
            return original_commit()

        monkeypatch.setattr(db_session, "commit", _selective_failing_commit)

        with pytest.raises(RuntimeError, match="simulated DB failure"):
            await service.handle_session_message(
                chat_session.id, "hello", db_session,
            )

        rows = self._read_messages(db_session, chat_session.id)
        assert len(rows) == 1
        assert rows[0].role == "user"
        assert rows[0].llm_profile_id_snapshot == "default"
        assert rows[0].llm_profile_kind_snapshot == "api"
        assert rows[0].llm_model_snapshot == "injected-test-model"

    @pytest.mark.anyio
    async def test_null_snapshot_history_not_backfilled(
        self, db_session, spy_llm,
    ):
        """Pre-existing NULL-snapshot history enters the LLM context in
        order and is NOT rewritten or backfilled in the database."""
        chat_session = _create_session(db_session)
        _add_messages(db_session, chat_session.id, [("q1", "a1")])
        service = ChatService(spy_llm)

        await service.handle_session_message(
            chat_session.id, "q2", db_session,
        )

        # History was sent as plain role/content
        contents = [m["content"] for m in spy_llm.calls[0][1:]]
        assert contents == ["q1", "a1", "q2"]

        # The old rows keep NULL snapshots; only the new round has them
        rows = self._read_messages(db_session, chat_session.id)
        assert len(rows) == 4
        old_user, old_assistant = rows[0], rows[1]
        assert old_user.llm_profile_id_snapshot is None
        assert old_user.llm_profile_kind_snapshot is None
        assert old_user.llm_model_snapshot is None
        assert old_assistant.llm_profile_id_snapshot is None
        assert old_assistant.llm_profile_kind_snapshot is None
        assert old_assistant.llm_model_snapshot is None

        new_user, new_assistant = rows[2], rows[3]
        for msg in (new_user, new_assistant):
            assert msg.llm_profile_id_snapshot == "default"
            assert msg.llm_profile_kind_snapshot == "api"
            assert msg.llm_model_snapshot == "injected-test-model"


# ============================================================================
# Switch session profile (service level)
# ============================================================================


def _switch_registry(default_client, default_kind="fake",
                     default_model="fake", extras=()):
    """Build a registry: one default profile plus *extras*
    (dicts with id/label/kind/model/client)."""
    from backend.llm_profiles import LLMProfile, LLMProfileRegistry

    profiles = [
        LLMProfile(
            id="default", label="Default", kind=default_kind,
            model=default_model, client=default_client, is_default=True,
        ),
    ]
    for e in extras:
        profiles.append(LLMProfile(
            id=e["id"], label=e["label"], kind=e["kind"],
            model=e["model"], client=e["client"], is_default=False,
        ))
    return LLMProfileRegistry(profiles)


def _make_bound_session(db, profile_id, snapshot):
    """Insert a ChatSession bound to *profile_id* with *snapshot*."""
    s = ChatSession(
        llm_profile_id=profile_id,
        llm_model_snapshot=snapshot,
    )
    db.add(s)
    db.commit()
    db.refresh(s)
    return s


def _add_history(db, session_id, snapshot_triple):
    """Insert one user + one assistant message with *snapshot_triple*
    (a 3-tuple) as the provenance snapshot."""
    db.add(Message(
        session_id=session_id, role="user", content="old q",
        llm_profile_id_snapshot=snapshot_triple[0],
        llm_profile_kind_snapshot=snapshot_triple[1],
        llm_model_snapshot=snapshot_triple[2],
    ))
    db.add(Message(
        session_id=session_id, role="assistant", content="old a",
        llm_profile_id_snapshot=snapshot_triple[0],
        llm_profile_kind_snapshot=snapshot_triple[1],
        llm_model_snapshot=snapshot_triple[2],
    ))
    db.commit()


def _read_session_from_db(db, session_id):
    """Re-read the session through a fresh identity-map-free query."""
    from sqlalchemy.orm import sessionmaker

    engine = db.get_bind()
    SessionLocal = sessionmaker(
        bind=engine, autoflush=False, expire_on_commit=False,
    )
    fresh = SessionLocal()
    try:
        return fresh.get(ChatSession, session_id)
    finally:
        fresh.close()


class TestSwitchSessionProfile:
    """ChatService.switch_session_profile behaviour."""

    def _api_registry(self):
        """default=api(remote-m1), local=local(qwen3.5:4b)."""
        self.default_spy = SpyLLMClient(response="default reply")
        self.local_spy = SpyLLMClient(response="local reply")
        return _switch_registry(
            default_client=self.default_spy,
            default_kind="api",
            default_model="remote-m1",
            extras=[{
                "id": "local", "label": "Local", "kind": "local",
                "model": "qwen3.5:4b", "client": self.local_spy,
            }],
        )

    # -- A. basic ---------------------------------------------------------

    @pytest.mark.anyio
    async def test_api_to_local_switch_succeeds(self, db_session):
        registry = self._api_registry()
        service = ChatService(profiles=registry)
        session = _make_bound_session(
            db_session, "default", "remote-m1",
        )

        updated = await service.switch_session_profile(
            session.id, "local", False, db_session,
        )

        assert updated.llm_profile_id == "local"
        assert updated.llm_model_snapshot == "qwen3.5:4b"

        fresh = _read_session_from_db(db_session, session.id)
        assert fresh.llm_profile_id == "local"
        assert fresh.llm_model_snapshot == "qwen3.5:4b"
        # Re-resolving returns READY
        profile = service.resolve_session_profile(fresh)
        assert profile.id == "local"

    @pytest.mark.anyio
    async def test_local_to_api_with_history_ack_true(self, db_session):
        registry = self._api_registry()
        service = ChatService(profiles=registry)
        session = _make_bound_session(db_session, "local", "qwen3.5:4b")
        _add_history(db_session, session.id, ("local", "local", "qwen3.5:4b"))

        updated = await service.switch_session_profile(
            session.id, "default", True, db_session,
        )

        assert updated.llm_profile_id == "default"
        assert updated.llm_model_snapshot == "remote-m1"

    @pytest.mark.anyio
    async def test_local_to_api_no_history_needs_no_ack(self, db_session):
        registry = self._api_registry()
        service = ChatService(profiles=registry)
        session = _make_bound_session(db_session, "local", "qwen3.5:4b")

        updated = await service.switch_session_profile(
            session.id, "default", False, db_session,
        )

        assert updated.llm_profile_id == "default"

    @pytest.mark.anyio
    async def test_switch_returns_ready_status(self, db_session):
        registry = self._api_registry()
        service = ChatService(profiles=registry)
        session = _make_bound_session(db_session, "local", "qwen3.5:4b")

        updated = await service.switch_session_profile(
            session.id, "local", False, db_session,
        )

        response = service.build_session_response(updated)
        assert response["llm_profile_status"] == SessionProfileStatus.READY

    @pytest.mark.anyio
    async def test_next_message_uses_new_client(self, db_session):
        registry = self._api_registry()
        service = ChatService(profiles=registry)
        session = _make_bound_session(db_session, "default", "remote-m1")

        await service.switch_session_profile(
            session.id, "local", False, db_session,
        )
        await service.handle_session_message(session.id, "hello", db_session)

        assert len(self.local_spy.calls) == 1
        assert len(self.default_spy.calls) == 0

    @pytest.mark.anyio
    async def test_history_order_preserved_in_new_client(self, db_session):
        registry = self._api_registry()
        service = ChatService(profiles=registry)
        session = _make_bound_session(db_session, "default", "remote-m1")
        _add_history(db_session, session.id, ("default", "api", "remote-m1"))

        await service.switch_session_profile(
            session.id, "local", False, db_session,
        )
        await service.handle_session_message(session.id, "new q", db_session)

        contents = [m["content"] for m in self.local_spy.calls[0][1:]]
        assert contents == ["old q", "old a", "new q"]

    @pytest.mark.anyio
    async def test_existing_messages_not_rewritten(self, db_session):
        registry = self._api_registry()
        service = ChatService(profiles=registry)
        session = _make_bound_session(db_session, "default", "remote-m1")
        _add_history(db_session, session.id, ("default", "api", "remote-m1"))

        await service.switch_session_profile(
            session.id, "local", False, db_session,
        )
        await service.handle_session_message(session.id, "new q", db_session)

        rows = (
            db_session.execute(
                select(Message)
                .where(Message.session_id == session.id)
                .order_by(Message.id.asc())
            )
            .scalars()
            .all()
        )
        assert len(rows) == 4
        # Old rows keep their original triples
        assert rows[0].llm_profile_id_snapshot == "default"
        assert rows[0].llm_profile_kind_snapshot == "api"
        assert rows[0].llm_model_snapshot == "remote-m1"
        assert rows[1].llm_profile_id_snapshot == "default"
        # New rows carry the new profile
        for msg in (rows[2], rows[3]):
            assert msg.llm_profile_id_snapshot == "local"
            assert msg.llm_profile_kind_snapshot == "local"
            assert msg.llm_model_snapshot == "qwen3.5:4b"

    # -- B. idempotence and repair ---------------------------------------

    @pytest.mark.anyio
    async def test_same_profile_idempotent_no_update(self, db_session, monkeypatch):
        from datetime import datetime as dt

        registry = self._api_registry()
        service = ChatService(profiles=registry)
        session = _make_bound_session(db_session, "default", "remote-m1")
        pinned = dt(2020, 1, 1, 0, 0, 0)  # noqa: DTZ001
        session.updated_at = pinned
        db_session.commit()

        # Count every persistence entry point: the idempotent path
        # must call NONE of them.
        calls = {"flush": 0, "refresh": 0, "commit": 0}

        def _counted_flush(*_a, **_k):
            calls["flush"] += 1
            return original_flush()

        def _counted_refresh(*_a, **_k):
            calls["refresh"] += 1
            return original_refresh()

        def _counted_commit(*_a, **_k):
            calls["commit"] += 1
            return original_commit()

        original_flush = db_session.flush
        original_refresh = db_session.refresh
        original_commit = db_session.commit
        monkeypatch.setattr(db_session, "flush", _counted_flush)
        monkeypatch.setattr(db_session, "refresh", _counted_refresh)
        monkeypatch.setattr(db_session, "commit", _counted_commit)

        statements = []
        from sqlalchemy import event

        def _record(_conn, _cursor, statement, _params, _ctx, _emany):
            statements.append(statement)

        engine = db_session.get_bind()
        event.listen(engine, "before_cursor_execute", _record)
        try:
            updated = await service.switch_session_profile(
                session.id, "default", False, db_session,
            )
        finally:
            event.remove(engine, "before_cursor_execute", _record)

        assert calls == {"flush": 0, "refresh": 0, "commit": 0}
        assert updated.llm_profile_id == "default"
        assert updated.llm_model_snapshot == "remote-m1"
        assert updated.updated_at == pinned

        fresh = _read_session_from_db(db_session, session.id)
        assert fresh.updated_at == pinned

        updates = [
            s for s in statements
            if "UPDATE chat_sessions" in s.upper()
        ]
        assert updates == []

    @pytest.mark.anyio
    async def test_same_profile_null_snapshot_repaired(self, db_session):
        """A legacy NULL snapshot on a local binding is repaired when
        re-applying the same local profile (no ack needed)."""
        registry = self._api_registry()
        service = ChatService(profiles=registry)
        session = _make_bound_session(db_session, "local", None)

        updated = await service.switch_session_profile(
            session.id, "local", False, db_session,
        )

        assert updated.llm_model_snapshot == "qwen3.5:4b"
        fresh = _read_session_from_db(db_session, session.id)
        assert fresh.llm_model_snapshot == "qwen3.5:4b"
        assert fresh.updated_at is not None

    @pytest.mark.anyio
    async def test_same_profile_stale_snapshot_repaired(self, db_session):
        registry = self._api_registry()
        service = ChatService(profiles=registry)
        session = _make_bound_session(db_session, "local", "old-model")

        updated = await service.switch_session_profile(
            session.id, "local", False, db_session,
        )

        assert updated.llm_model_snapshot == "qwen3.5:4b"
        assert (
            service.resolve_session_profile(
                _read_session_from_db(db_session, session.id)
            ).id
            == "local"
        )

    @pytest.mark.anyio
    async def test_legacy_rebind_restores_ready(self, db_session):
        registry = self._api_registry()
        service = ChatService(profiles=registry)
        session = _make_bound_session(db_session, "local", None)

        await service.switch_session_profile(
            session.id, "local", False, db_session,
        )

        fresh = _read_session_from_db(db_session, session.id)
        profile = service.resolve_session_profile(fresh)
        assert profile.id == "local"

    @pytest.mark.anyio
    async def test_model_changed_rebind_restores_ready(self, db_session):
        registry = self._api_registry()
        service = ChatService(profiles=registry)
        session = _make_bound_session(db_session, "local", "old-model")

        await service.switch_session_profile(
            session.id, "local", False, db_session,
        )

        fresh = _read_session_from_db(db_session, session.id)
        profile = service.resolve_session_profile(fresh)
        assert profile.id == "local"

    @pytest.mark.anyio
    async def test_profile_unavailable_rebind_restores_ready(self, db_session):
        registry = self._api_registry()
        service = ChatService(profiles=registry)
        session = _make_bound_session(db_session, "gone", "old-model")

        await service.switch_session_profile(
            session.id, "local", False, db_session,
        )

        fresh = _read_session_from_db(db_session, session.id)
        profile = service.resolve_session_profile(fresh)
        assert profile.id == "local"

    @pytest.mark.anyio
    async def test_real_change_updates_updated_at(self, db_session):
        from datetime import datetime as dt

        registry = self._api_registry()
        service = ChatService(profiles=registry)
        session = _make_bound_session(db_session, "default", "remote-m1")
        pinned = dt(2020, 1, 1, 0, 0, 0)  # noqa: DTZ001
        session.updated_at = pinned
        db_session.commit()

        await service.switch_session_profile(
            session.id, "local", False, db_session,
        )

        fresh = _read_session_from_db(db_session, session.id)
        assert fresh.updated_at > pinned

    @pytest.mark.anyio
    async def test_title_and_manual_flag_unchanged(self, db_session):
        registry = self._api_registry()
        service = ChatService(profiles=registry)
        session = _make_bound_session(db_session, "default", "remote-m1")
        session.title = "My Chat"
        session.title_is_manual = True
        db_session.commit()

        updated = await service.switch_session_profile(
            session.id, "local", False, db_session,
        )

        assert updated.title == "My Chat"
        assert updated.title_is_manual is True

    # -- C. privacy matrix -----------------------------------------------

    @pytest.mark.anyio
    async def test_local_to_api_with_history_ack_false_raises(self, db_session):
        from backend.exceptions import SessionProfileSwitchAckRequiredError

        registry = self._api_registry()
        service = ChatService(profiles=registry)
        session = _make_bound_session(db_session, "local", "qwen3.5:4b")
        _add_history(db_session, session.id, ("local", "local", "qwen3.5:4b"))
        before = _read_session_from_db(db_session, session.id)

        with pytest.raises(SessionProfileSwitchAckRequiredError) as exc:
            await service.switch_session_profile(
                session.id, "default", False, db_session,
            )
        assert exc.value.code == "remote_history_ack_required"

        after = _read_session_from_db(db_session, session.id)
        assert after.llm_profile_id == "local"
        assert after.llm_model_snapshot == "qwen3.5:4b"
        assert after.updated_at == before.updated_at
        assert len(self.default_spy.calls) == 0
        assert len(self.local_spy.calls) == 0

    @pytest.mark.anyio
    async def test_api_a_to_api_b_with_history_requires_ack(self, db_session):
        from backend.exceptions import SessionProfileSwitchAckRequiredError

        spy_b = SpyLLMClient(response="b")
        registry = _switch_registry(
            default_client=SpyLLMClient(response="a"),
            default_kind="api",
            default_model="remote-m1",
            extras=[{
                "id": "api-b", "label": "API B", "kind": "api",
                "model": "remote-m2", "client": spy_b,
            }],
        )
        service = ChatService(profiles=registry)
        session = _make_bound_session(db_session, "default", "remote-m1")
        _add_history(db_session, session.id, ("default", "api", "remote-m1"))

        with pytest.raises(SessionProfileSwitchAckRequiredError):
            await service.switch_session_profile(
                session.id, "api-b", False, db_session,
            )

        fresh = _read_session_from_db(db_session, session.id)
        assert fresh.llm_profile_id == "default"
        assert fresh.llm_model_snapshot == "remote-m1"

    @pytest.mark.anyio
    async def test_same_reliable_api_binding_needs_no_ack(self, db_session):
        registry = self._api_registry()
        service = ChatService(profiles=registry)
        session = _make_bound_session(db_session, "default", "remote-m1")
        _add_history(db_session, session.id, ("default", "api", "remote-m1"))

        updated = await service.switch_session_profile(
            session.id, "default", False, db_session,
        )

        assert updated.llm_profile_id == "default"

    @pytest.mark.anyio
    async def test_legacy_to_api_requires_ack(self, db_session):
        from backend.exceptions import SessionProfileSwitchAckRequiredError

        registry = self._api_registry()
        service = ChatService(profiles=registry)
        session = _make_bound_session(db_session, "default", None)
        _add_history(db_session, session.id, (None, None, None))

        with pytest.raises(SessionProfileSwitchAckRequiredError):
            await service.switch_session_profile(
                session.id, "default", False, db_session,
            )

    @pytest.mark.anyio
    async def test_model_changed_to_api_requires_ack(self, db_session):
        from backend.exceptions import SessionProfileSwitchAckRequiredError

        registry = self._api_registry()
        service = ChatService(profiles=registry)
        session = _make_bound_session(db_session, "default", "old-model")
        _add_history(db_session, session.id, ("default", "api", "old-model"))

        with pytest.raises(SessionProfileSwitchAckRequiredError):
            await service.switch_session_profile(
                session.id, "default", False, db_session,
            )

    @pytest.mark.anyio
    async def test_profile_unavailable_to_api_requires_ack(self, db_session):
        from backend.exceptions import SessionProfileSwitchAckRequiredError

        registry = self._api_registry()
        service = ChatService(profiles=registry)
        session = _make_bound_session(db_session, "gone", "old-model")
        _add_history(db_session, session.id, (None, None, None))

        with pytest.raises(SessionProfileSwitchAckRequiredError):
            await service.switch_session_profile(
                session.id, "default", False, db_session,
            )

    @pytest.mark.anyio
    async def test_api_to_local_with_history_needs_no_ack(self, db_session):
        registry = self._api_registry()
        service = ChatService(profiles=registry)
        session = _make_bound_session(db_session, "default", "remote-m1")
        _add_history(db_session, session.id, ("default", "api", "remote-m1"))

        updated = await service.switch_session_profile(
            session.id, "local", False, db_session,
        )

        assert updated.llm_profile_id == "local"

    # -- D. errors and transactions --------------------------------------

    @pytest.mark.anyio
    async def test_session_not_found(self, db_session):
        registry = self._api_registry()
        service = ChatService(profiles=registry)

        with pytest.raises(SessionNotFoundError):
            await service.switch_session_profile(
                9999, "local", False, db_session,
            )
        assert len(service._lock_registry._entries) == 0

    @pytest.mark.anyio
    async def test_unknown_profile_binding_unchanged(self, db_session):
        registry = self._api_registry()
        service = ChatService(profiles=registry)
        session = _make_bound_session(db_session, "default", "remote-m1")

        with pytest.raises(UnknownLLMProfileError):
            await service.switch_session_profile(
                session.id, "nope", False, db_session,
            )

        fresh = _read_session_from_db(db_session, session.id)
        assert fresh.llm_profile_id == "default"
        assert fresh.llm_model_snapshot == "remote-m1"
        assert len(service._lock_registry._entries) == 0

    @pytest.mark.anyio
    async def test_commit_failure_rolls_back_and_preserves_state(
        self, db_session, monkeypatch,
    ):
        registry = self._api_registry()
        service = ChatService(profiles=registry)
        session = _make_bound_session(db_session, "default", "remote-m1")
        before = _read_session_from_db(db_session, session.id)

        def _failing_commit():
            raise RuntimeError("simulated commit failure")

        monkeypatch.setattr(db_session, "commit", _failing_commit)

        with pytest.raises(RuntimeError, match="simulated commit failure"):
            await service.switch_session_profile(
                session.id, "local", False, db_session,
            )

        monkeypatch.undo()

        # The rollback inside the service must have closed the
        # transaction.  Check BEFORE opening any fresh session: the
        # shared StaticPool connection makes a later query re-open a
        # transaction on this session (a test-environment artefact).
        assert db_session.in_transaction() is False

        fresh = _read_session_from_db(db_session, session.id)
        assert fresh.llm_profile_id == "default"
        assert fresh.llm_model_snapshot == "remote-m1"
        assert fresh.updated_at == before.updated_at
        assert len(service._lock_registry._entries) == 0

    @pytest.mark.anyio
    async def test_switch_never_calls_any_llm(self, db_session):
        registry = self._api_registry()
        service = ChatService(profiles=registry)
        session = _make_bound_session(db_session, "default", "remote-m1")
        _add_history(db_session, session.id, ("default", "api", "remote-m1"))

        await service.switch_session_profile(
            session.id, "local", False, db_session,
        )

        assert len(self.default_spy.calls) == 0
        assert len(self.local_spy.calls) == 0
        assert len(service._lock_registry._entries) == 0


class _GatedLockRegistry:
    """Test-only wrapper around the real SessionLockRegistry.

    The FIRST coroutine to enter the per-session lock signals
    ``first_acquired`` and then waits on ``allow_first`` while still
    HOLDING the real lock.  Any later entrant signals
    ``second_attempted`` before waiting on the real lock.  This proves
    that the second operation genuinely waits on the same lock —
    entirely event-driven, no wall-clock sleeps.

    Only used by tests; production code keeps its own real registry.
    """

    def __init__(self, real):
        self._real = real
        self.first_acquired = asyncio.Event()
        self.allow_first = asyncio.Event()
        self.second_attempted = asyncio.Event()
        self._entrants = 0

    @asynccontextmanager
    async def session_lock(self, session_id):
        self._entrants += 1
        is_first = self._entrants == 1
        if not is_first:
            self.second_attempted.set()
        try:
            async with self._real.session_lock(session_id):
                if is_first:
                    self.first_acquired.set()
                    await self.allow_first.wait()
                yield
        finally:
            self._entrants -= 1

    @property
    def entries(self):
        return self._real._entries


async def _wait_event(event, name):
    """Fail-safe wait for a gate event.

    The timeout only prevents the TEST from hanging forever — it never
    decides concurrency ordering (the events and task-done checks do).
    """
    try:
        await asyncio.wait_for(event.wait(), timeout=5)
    except TimeoutError as exc:
        raise AssertionError(f"Timed out waiting for {name}") from exc


async def _wait_task(task, name):
    """Fail-safe await of a spawned test task."""
    try:
        return await asyncio.wait_for(task, timeout=5)
    except TimeoutError as exc:
        raise AssertionError(f"Timed out waiting for task {name}") from exc


async def _cancel_and_reap(tasks):
    """Cancel unfinished tasks and reap every task (including raised
    exceptions) so no background coroutine leaks out of a test."""
    for t in tasks:
        if t is not None and not t.done():
            t.cancel()
    await asyncio.gather(*[t for t in tasks if t is not None],
                         return_exceptions=True)


class TestSwitchSessionProfileConcurrency:
    """Lock ordering between switch and send / delete / rename.

    Every test wraps the real SessionLockRegistry with
    ``_GatedLockRegistry`` and drives the interleaving with
    asyncio.Events — no sleeps, no wall-clock timing.  Event waits and
    task awaits carry 5-second fail-safe timeouts so a production lock
    regression fails loudly instead of hanging."""

    def _registry(self):
        self.default_spy = SpyLLMClient(response="default reply")
        self.local_spy = SpyLLMClient(response="local reply")
        return _switch_registry(
            default_client=self.default_spy,
            default_kind="api",
            default_model="remote-m1",
            extras=[{
                "id": "local", "label": "Local", "kind": "local",
                "model": "qwen3.5:4b", "client": self.local_spy,
            }],
        )

    def _gated_service(self):
        registry = self._registry()
        service = ChatService(profiles=registry)
        self.gate = _GatedLockRegistry(SessionLockRegistry())
        service._lock_registry = self.gate
        return service

    async def _run_gated(self, first_fn, second_fn):
        """Run *first_fn* (gated, holds the lock) and *second_fn* (waits
        on the lock); assert the wait, release, and collect both.

        All waits carry 5-second fail-safe timeouts.  On ANY failure
        (event timeout, assertion, task exception) the gate is
        released, unfinished tasks are cancelled and every task is
        reaped so nothing leaks into the background."""
        first_task = asyncio.create_task(first_fn())
        second_task = None
        try:
            await _wait_event(self.gate.first_acquired, "first_acquired")
            second_task = asyncio.create_task(second_fn())
            await _wait_event(self.gate.second_attempted, "second_attempted")
            # second is now genuinely waiting on the real lock.
            assert not second_task.done()
        except BaseException:
            self.gate.allow_first.set()
            await _cancel_and_reap([second_task, first_task])
            raise

        self.gate.allow_first.set()
        try:
            first_result = await _wait_task(first_task, "first")
            second_result = await _wait_task(second_task, "second")
        except BaseException:
            await _cancel_and_reap([second_task, first_task])
            raise
        return first_result, second_result

    @pytest.mark.anyio
    async def test_send_holds_lock_switch_waits(self, db_session):
        """send acquires first; switch genuinely waits, then runs after
        the send round completes with the OLD profile."""
        service = self._gated_service()
        session = _make_bound_session(db_session, "default", "remote-m1")

        async def send():
            return await service.handle_session_message(
                session.id, "q1", db_session,
            )

        async def switch():
            return await service.switch_session_profile(
                session.id, "local", False, db_session,
            )

        _, _ = await self._run_gated(send, switch)

        rows = (
            db_session.execute(
                select(Message)
                .where(Message.session_id == session.id)
                .order_by(Message.id.asc())
            )
            .scalars()
            .all()
        )
        assert len(rows) == 2
        for msg in rows:
            assert msg.llm_profile_id_snapshot == "default"
            assert msg.llm_profile_kind_snapshot == "api"
            assert msg.llm_model_snapshot == "remote-m1"

        fresh = _read_session_from_db(db_session, session.id)
        assert fresh.llm_profile_id == "local"
        assert len(self.gate.entries) == 0

    @pytest.mark.anyio
    async def test_switch_holds_lock_send_waits_uses_new_client(
        self, db_session,
    ):
        """switch acquires first; send genuinely waits and afterwards
        uses the NEW profile's client."""
        service = self._gated_service()
        session = _make_bound_session(db_session, "default", "remote-m1")

        async def switch():
            return await service.switch_session_profile(
                session.id, "local", False, db_session,
            )

        async def send():
            return await service.handle_session_message(
                session.id, "q1", db_session,
            )

        _, _ = await self._run_gated(switch, send)

        assert len(self.local_spy.calls) == 1
        assert len(self.default_spy.calls) == 0
        rows = (
            db_session.execute(
                select(Message).where(Message.session_id == session.id)
            )
            .scalars()
            .all()
        )
        for msg in rows:
            assert msg.llm_profile_id_snapshot == "local"
        assert len(self.gate.entries) == 0

    @pytest.mark.anyio
    async def test_delete_holds_lock_switch_waits_then_404(self, db_session):
        """delete acquires first; switch waits and then gets 404."""
        service = self._gated_service()
        session = _make_bound_session(db_session, "default", "remote-m1")

        first_task = asyncio.create_task(
            service.delete_session(session.id, db_session)
        )
        switch_task = None
        try:
            await _wait_event(self.gate.first_acquired, "first_acquired")
            switch_task = asyncio.create_task(
                service.switch_session_profile(
                    session.id, "local", False, db_session,
                )
            )
            await _wait_event(self.gate.second_attempted, "second_attempted")
            assert not switch_task.done()
        except BaseException:
            self.gate.allow_first.set()
            await _cancel_and_reap([switch_task, first_task])
            raise

        self.gate.allow_first.set()
        try:
            await _wait_task(first_task, "delete")
        except BaseException:
            await _cancel_and_reap([switch_task, first_task])
            raise
        with pytest.raises(SessionNotFoundError):
            await _wait_task(switch_task, "switch")
        assert len(self.gate.entries) == 0

    @pytest.mark.anyio
    async def test_switch_holds_lock_delete_waits_then_removes(
        self, db_session,
    ):
        """switch acquires first; delete waits and then removes the
        session."""
        service = self._gated_service()
        session = _make_bound_session(db_session, "default", "remote-m1")

        async def switch():
            await service.switch_session_profile(
                session.id, "local", False, db_session,
            )

        async def delete():
            await service.delete_session(session.id, db_session)

        await self._run_gated(switch, delete)

        fresh = _read_session_from_db(db_session, session.id)
        assert fresh is None
        assert len(self.gate.entries) == 0

    @pytest.mark.anyio
    async def test_rename_holds_lock_switch_waits_both_preserved(
        self, db_session,
    ):
        service = self._gated_service()
        session = _make_bound_session(db_session, "default", "remote-m1")

        async def rename():
            await service.rename_session(
                session.id, "Manual Title", db_session,
            )

        async def switch():
            await service.switch_session_profile(
                session.id, "local", False, db_session,
            )

        await self._run_gated(rename, switch)

        fresh = _read_session_from_db(db_session, session.id)
        assert fresh.title == "Manual Title"
        assert fresh.title_is_manual is True
        assert fresh.llm_profile_id == "local"
        assert len(self.gate.entries) == 0

    @pytest.mark.anyio
    async def test_switch_holds_lock_rename_waits_both_preserved(
        self, db_session,
    ):
        service = self._gated_service()
        session = _make_bound_session(db_session, "default", "remote-m1")

        async def switch():
            await service.switch_session_profile(
                session.id, "local", False, db_session,
            )

        async def rename():
            await service.rename_session(
                session.id, "Manual Title", db_session,
            )

        await self._run_gated(switch, rename)

        fresh = _read_session_from_db(db_session, session.id)
        assert fresh.title == "Manual Title"
        assert fresh.title_is_manual is True
        assert fresh.llm_profile_id == "local"
        assert len(self.gate.entries) == 0

    @pytest.mark.anyio
    async def test_switch_cancelled_while_waiting_for_lock(self, db_session):
        """Cancelling the waiting switch leaves the binding unchanged
        and the registry clean."""
        service = self._gated_service()
        session = _make_bound_session(db_session, "default", "remote-m1")

        async def send():
            await service.handle_session_message(session.id, "q1", db_session)

        first_task = asyncio.create_task(send())
        switch_task = None
        try:
            await _wait_event(self.gate.first_acquired, "first_acquired")
            switch_task = asyncio.create_task(
                service.switch_session_profile(
                    session.id, "local", False, db_session,
                )
            )
            await _wait_event(self.gate.second_attempted, "second_attempted")
            switch_task.cancel()
            try:
                await _wait_task(switch_task, "cancelled switch")
            except asyncio.CancelledError:
                pass
        except BaseException:
            self.gate.allow_first.set()
            await _cancel_and_reap([switch_task, first_task])
            raise
        finally:
            self.gate.allow_first.set()
        await _wait_task(first_task, "send")
        await _cancel_and_reap([switch_task])

        fresh = _read_session_from_db(db_session, session.id)
        assert fresh.llm_profile_id == "default"
        assert fresh.llm_model_snapshot == "remote-m1"
        assert len(self.gate.entries) == 0

    @pytest.mark.anyio
    async def test_registry_clean_after_409_path(self, db_session):
        from backend.exceptions import SessionProfileSwitchAckRequiredError

        service = self._gated_service()
        # Single-operation test: let the first entrant pass the gate.
        self.gate.allow_first.set()
        session = _make_bound_session(db_session, "local", "qwen3.5:4b")
        _add_history(db_session, session.id, ("local", "local", "qwen3.5:4b"))

        with pytest.raises(SessionProfileSwitchAckRequiredError):
            await service.switch_session_profile(
                session.id, "default", False, db_session,
            )
        assert len(self.gate.entries) == 0


# ============================================================================
# 20-message history boundary after a switch
# ============================================================================


class TestSwitchHistoryBoundary:
    """After switching, the new client receives at most the 20 most
    recent prior messages, in order."""

    @pytest.mark.anyio
    async def test_24_history_switch_sends_only_last_20(self, db_session):
        old_spy = SpyLLMClient(response="old reply")
        new_spy = SpyLLMClient(response="new reply")
        registry = _switch_registry(
            default_client=old_spy,
            default_kind="fake",
            default_model="fake",
            extras=[{
                "id": "local", "label": "Local", "kind": "local",
                "model": "qwen3.5:4b", "client": new_spy,
            }],
        )
        service = ChatService(profiles=registry)
        session = _make_bound_session(db_session, "default", "fake")

        # 24 prior messages (12 pairs) with the old-profile snapshot.
        for i in range(12):
            for role, text in (("user", f"q{i}"), ("assistant", f"a{i}")):
                db_session.add(Message(
                    session_id=session.id, role=role, content=text,
                    llm_profile_id_snapshot="default",
                    llm_profile_kind_snapshot="fake",
                    llm_model_snapshot="fake",
                ))
        db_session.commit()

        await service.switch_session_profile(
            session.id, "local", False, db_session,
        )
        await service.handle_session_message(session.id, "new-q", db_session)

        # The new client saw: system + exactly 20 history + new user.
        msgs = new_spy.calls[0]
        assert msgs[0] == {"role": "system", "content": SYSTEM_PROMPT}
        history = msgs[1:-1]
        assert len(history) == MAX_HISTORY_MESSAGES

        expected = []
        # 12 pairs = 24 messages; the oldest 4 (q0,a0,q1,a1) are out of
        # the window.  The newest 20 start at q2.
        for i in range(2, 12):
            expected.append({"role": "user", "content": f"q{i}"})
            expected.append({"role": "assistant", "content": f"a{i}"})
        assert history == expected
        assert msgs[-1] == {"role": "user", "content": "new-q"}

        # Old messages untouched; new round carries the new profile.
        rows = (
            db_session.execute(
                select(Message)
                .where(Message.session_id == session.id)
                .order_by(Message.id.asc())
            )
            .scalars()
            .all()
        )
        assert len(rows) == 26
        for msg in rows[:24]:
            assert msg.llm_profile_id_snapshot == "default"
            assert msg.llm_profile_kind_snapshot == "fake"
            assert msg.llm_model_snapshot == "fake"
        for msg in rows[24:]:
            assert msg.llm_profile_id_snapshot == "local"
            assert msg.llm_profile_kind_snapshot == "local"
            assert msg.llm_model_snapshot == "qwen3.5:4b"
