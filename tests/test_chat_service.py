"""Tests for ChatService — unit and integration tests with SpyLLMClient.

Every test uses a temporary SQLite database and a SpyLLMClient — no
real network requests are ever made.
"""

import copy

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import sessionmaker

from backend.chat_service import MAX_HISTORY_MESSAGES, ChatService
from backend.database import create_database_engine, create_tables
from backend.exceptions import LLMError, LLMInvalidResponseError
from backend.llm_client import LLMMessage
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
    """Insert and return a new ChatSession."""
    s = ChatSession()
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
