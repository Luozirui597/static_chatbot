"""Tests for ChatService — unit and integration tests with SpyLLMClient.

Every test uses a temporary SQLite database and a SpyLLMClient — no
real network requests are ever made.
"""

import asyncio
import copy
import json
import logging
from contextlib import asynccontextmanager

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import sessionmaker

import backend.chat_service as chat_service_module
from backend.chat_service import (
    MAX_HISTORY_MESSAGES,
    ChatService,
    SessionLockRegistry,
    SessionNotFoundError,
    _project_history_for_mode,
    _serialize_receive_teaching_history,
)
from backend.interaction_modes import (
    CORRECTIVE_MODE,
    CORRECTIVE_PROMPT,
    CORRECTIVE_PROMPT_VERSION,
    RECEIVE_TEACHING_HISTORY_CONTEXT_VERSION,
    RECEIVE_TEACHING_MODE,
    RECEIVE_TEACHING_PROMPT,
    RECEIVE_TEACHING_PROMPT_VERSION,
)
from backend.database import create_database_engine, create_tables
from backend.exceptions import (
    LLMError,
    LLMInvalidResponseError,
    UnknownLLMProfileError,
)
from backend.review_diagnostics import ErrorCategory, LLMCallDiagnostics
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

    async def generate(
        self,
        messages: list[LLMMessage],
        *,
        response_format=None,
        temperature=None,
        max_tokens=None,
        diagnostics=None,
    ) -> str:
        self.calls.append(copy.deepcopy(messages))
        if self.error is not None:
            raise self.error
        if self.response == "test reply" and any(
            msg.get("role") == "system"
            and "请用中文回复。" in str(msg.get("content", ""))
            for msg in messages
        ):
            return "测试回复。"
        return self.response


class KwargSpyLLMClient:
    """Spy that records exactly which generate options were supplied."""

    _UNSET = object()

    def __init__(self, response: str = "test reply") -> None:
        self.calls: list[list[LLMMessage]] = []
        self.generate_kwargs: list[dict[str, object]] = []
        self.response = response

    async def generate(
        self,
        messages: list[LLMMessage],
        *,
        response_format=_UNSET,
        temperature=_UNSET,
        max_tokens=_UNSET,
        diagnostics=_UNSET,
    ) -> str:
        self.calls.append(copy.deepcopy(messages))
        if self.response == "test reply" and any(
            msg.get("role") == "system"
            and "请用中文回复。" in str(msg.get("content", ""))
            for msg in messages
        ):
            self.response_override = "测试回复。"
        else:
            self.response_override = self.response
        supplied = {}
        for name, value in (
            ("response_format", response_format),
            ("temperature", temperature),
            ("max_tokens", max_tokens),
            ("diagnostics", diagnostics),
        ):
            if value is not self._UNSET:
                supplied[name] = value
        self.generate_kwargs.append(supplied)
        return self.response_override


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


def _chat_failure_payloads(caplog) -> list[dict]:
    """Return decoded chat failure payloads from caplog records."""
    payloads = []
    prefix = "chat_llm_failure "
    for record in caplog.records:
        if record.name != "backend.chat_service":
            continue
        message = record.getMessage()
        if message.startswith(prefix):
            payloads.append(json.loads(message[len(prefix):]))
    return payloads


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


def _decode_receive_history_background(messages):
    """Decode the one background block from a Receive history call."""
    assert [m["role"] for m in messages] == ["system", "user", "user"]
    background = messages[1]
    assert background["role"] == "user"
    payload = json.loads(background["content"])
    assert payload["type"] == "receive_teaching_history_background"
    assert payload["version"] == RECEIVE_TEACHING_HISTORY_CONTEXT_VERSION
    assert isinstance(payload["note"], str)
    assert payload["note"] != ""
    items = payload["messages"]
    assert [item["index"] for item in items] == list(
        range(1, len(items) + 1)
    )
    return [item["content"] for item in items]


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

        assert spy_llm.calls[0][0]["role"] == "system"
        assert spy_llm.calls[0][0]["content"].startswith(RECEIVE_TEACHING_PROMPT)

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
    async def test_second_round_includes_first_round_user_only(self, db_session, spy_llm):
        """Receive Round 2 sends Round 1 user history but no assistant."""
        chat_session = _create_session(db_session)
        service = ChatService(spy_llm)

        # Round 1
        await service.handle_session_message(chat_session.id, "q1", db_session)
        # Round 2
        await service.handle_session_message(chat_session.id, "q2", db_session)

        assert len(spy_llm.calls) == 2
        msgs = spy_llm.calls[1]
        assert [m["role"] for m in msgs] == ["system", "user", "user"]
        assert msgs[0]["role"] == "system"
        assert msgs[0]["content"].startswith(RECEIVE_TEACHING_PROMPT)
        assert msgs[-1] == {"role": "user", "content": "q2"}
        assert _decode_receive_history_background(msgs) == ["q1"]

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

        msgs = spy_llm.calls[0]
        assert [m["role"] for m in msgs] == ["system", "user", "user"]
        assert msgs[-1] == {"role": "user", "content": "q4"}
        assert _decode_receive_history_background(msgs) == [
            "q1",
            "q2",
            "q3",
        ]

    @pytest.mark.anyio
    async def test_receive_background_round_trips_tricky_content(
        self, db_session, spy_llm,
    ):
        raw_history = "第一行\n第二行\r\n引号\" 反斜杠\\ 花括号{}"
        chat_session = _create_session(db_session)
        _add_messages(
            db_session,
            chat_session.id,
            [(raw_history, "a1")],
        )
        service = ChatService(spy_llm)

        await service.handle_session_message(
            chat_session.id, "current request", db_session,
        )

        assert len(spy_llm.calls) == 1
        msgs = spy_llm.calls[0]
        assert msgs[-1] == {"role": "user", "content": "current request"}
        assert _decode_receive_history_background(msgs) == [raw_history]

    @pytest.mark.anyio
    async def test_receive_background_is_session_scoped(
        self, db_session, spy_llm,
    ):
        session_a = _create_session(db_session)
        session_b = _create_session(db_session)
        _add_messages(db_session, session_a.id, [("qa1", "aa1")])
        _add_messages(db_session, session_b.id, [("qb1", "ab1")])
        service = ChatService(spy_llm)

        await service.handle_session_message(session_a.id, "qa2", db_session)
        await service.handle_session_message(session_b.id, "qb2", db_session)

        assert len(spy_llm.calls) == 2
        session_b_call = spy_llm.calls[1]
        assert session_b_call[-1] == {"role": "user", "content": "qb2"}
        assert _decode_receive_history_background(session_b_call) == ["qb1"]
        assert "qa1" not in session_b_call[1]["content"]



    @pytest.mark.anyio
    async def test_max_20_history_messages(self, db_session, spy_llm):
        """At most 20 previous messages are sent."""
        chat_session = _create_session(db_session)
        # Insert 25 user/assistant pairs = 50 history messages
        pairs = [(f"q{i}", f"a{i}") for i in range(25)]
        _add_messages(db_session, chat_session.id, pairs)
        service = ChatService(spy_llm)

        await service.handle_session_message(chat_session.id, "current", db_session)

        assert len(spy_llm.calls) == 1
        msgs = spy_llm.calls[0]
        # Query keeps the last 20 stored rows, then Receive projection
        # removes all assistant rows.  With 25 pairs, the window starts
        # at q15 and yields 10 historical user rows plus the current message.
        assert [m["role"] for m in msgs] == ["system", "user", "user"]
        assert msgs[-1] == {"role": "user", "content": "current"}
        assert _decode_receive_history_background(msgs) == [
            f"q{i}" for i in range(15, 25)
        ]

    @pytest.mark.anyio
    async def test_current_user_not_counted_in_history_limit(self, db_session, spy_llm):
        """The current user message is extra, not part of the 20 limit."""
        chat_session = _create_session(db_session)
        # Insert exactly 20 history messages
        pairs = [(f"q{i}", f"a{i}") for i in range(10)]
        _add_messages(db_session, chat_session.id, pairs)
        service = ChatService(spy_llm)

        await service.handle_session_message(chat_session.id, "current", db_session)

        msgs = spy_llm.calls[0]
        # 20 stored history rows -> 10 historical user rows after
        # projection, plus the current user message.
        assert [m["role"] for m in msgs] == ["system", "user", "user"]
        assert _decode_receive_history_background(msgs) == [
            f"q{i}" for i in range(10)
        ]
        # Last must be the current user
        assert msgs[-1] == {"role": "user", "content": "current"}

    @pytest.mark.anyio
    async def test_oldest_history_excluded_when_over_limit(self, db_session, spy_llm):
        """When history exceeds 20, the oldest messages are dropped."""
        chat_session = _create_session(db_session)
        # Insert 25 pairs = 50 history messages
        pairs = [(f"q{i}", f"a{i}") for i in range(25)]
        _add_messages(db_session, chat_session.id, pairs)
        service = ChatService(spy_llm)

        await service.handle_session_message(chat_session.id, "current", db_session)

        msgs = spy_llm.calls[0]
        # First background item should be q15 (oldest 5 pairs = 10 msgs dropped)
        historical = _decode_receive_history_background(msgs)
        assert historical[0] == "q15"

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
        assert contents[0].startswith(RECEIVE_TEACHING_PROMPT)
        assert contents[1:] == ["qb1"]


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
# Tests — chat LLM failure logging
# ============================================================================


class TestChatLLMFailureLogging:
    @pytest.mark.anyio
    async def test_timeout_failure_logs_session_and_saved_message_once(
        self, db_session, caplog
    ):
        chat_session = _create_session(db_session)
        diagnostics = LLMCallDiagnostics(
            error_category=ErrorCategory.UPSTREAM_READ_TIMEOUT,
            exception_type="ReadTimeout",
            timeout_configured_seconds=30.0,
            call_elapsed_ms=30012,
            http_status=None,
            upstream_request_id="synthetic-request-123",
        )
        error = LLMError(
            "synthetic detail placeholder",
            status_code=504,
            diagnostics=diagnostics,
        )
        spy = SpyLLMClient(error=error)
        service = ChatService(spy)

        with caplog.at_level(logging.WARNING, logger="backend.chat_service"):
            with pytest.raises(LLMError) as excinfo:
                await service.handle_session_message(
                    chat_session.id, "hello", db_session
                )

        assert excinfo.value is error
        assert excinfo.value.status_code == 504
        assert len(spy.calls) == 1
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

        payloads = _chat_failure_payloads(caplog)
        assert len(payloads) == 1
        payload = payloads[0]
        assert payload["event"] == "chat_llm_failure"
        assert payload["timestamp_utc"].endswith("+00:00")
        assert payload["session_id"] == chat_session.id
        assert payload["user_message_id"] == rows[0].id
        assert payload["local_http_status"] == 504
        assert payload["error_category"] == "upstream_read_timeout"
        assert payload["exception_type"] == "ReadTimeout"
        assert payload["timeout_configured_seconds"] == 30.0
        assert payload["call_elapsed_ms"] == 30012
        assert payload["upstream_http_status"] is None
        assert payload["upstream_request_id"] == "synthetic-request-123"
        assert "synthetic detail placeholder" not in caplog.text

    @pytest.mark.anyio
    async def test_failure_without_diagnostics_keeps_diagnostic_fields_null(
        self, db_session, caplog
    ):
        chat_session = _create_session(db_session)
        error = LLMError("synthetic detail placeholder", status_code=502)
        spy = SpyLLMClient(error=error)
        service = ChatService(spy)

        with caplog.at_level(logging.WARNING, logger="backend.chat_service"):
            with pytest.raises(LLMError):
                await service.handle_session_message(
                    chat_session.id, "hello", db_session
                )

        payloads = _chat_failure_payloads(caplog)
        assert len(payloads) == 1
        payload = payloads[0]
        assert payload["local_http_status"] == 502
        assert payload["upstream_http_status"] is None
        for key in (
            "error_category",
            "exception_type",
            "timeout_configured_seconds",
            "call_elapsed_ms",
            "upstream_request_id",
        ):
            assert payload[key] is None

    @pytest.mark.anyio
    async def test_success_path_emits_no_failure_event(
        self, db_session, caplog
    ):
        chat_session = _create_session(db_session)
        spy = SpyLLMClient(response="test reply")
        service = ChatService(spy)

        with caplog.at_level(logging.WARNING, logger="backend.chat_service"):
            await service.handle_session_message(
                chat_session.id, "hello", db_session
            )

        assert _chat_failure_payloads(caplog) == []
        assert len(spy.calls) == 1

    @pytest.mark.anyio
    async def test_failure_log_omits_synthetic_sensitive_placeholders(
        self, db_session, caplog
    ):
        chat_session = _create_session(db_session)
        marker = "SYNTHETIC_CHAT_SECRET_7f2a"
        error = LLMError(marker, status_code=504)
        spy = SpyLLMClient(error=error)
        service = ChatService(spy)

        with caplog.at_level(logging.WARNING, logger="backend.chat_service"):
            with pytest.raises(LLMError):
                await service.handle_session_message(
                    chat_session.id, marker, db_session
                )

        assert marker not in caplog.text
        assert len(_chat_failure_payloads(caplog)) == 1

    @pytest.mark.anyio
    async def test_logging_failure_does_not_replace_original_error(
        self, db_session, monkeypatch
    ):
        chat_session = _create_session(db_session)
        error = LLMError("synthetic detail placeholder", status_code=504)
        spy = SpyLLMClient(error=error)
        service = ChatService(spy)

        def _raising_warning(*args, **kwargs):
            raise RuntimeError("synthetic logger failure")

        monkeypatch.setattr(
            chat_service_module.logger, "warning", _raising_warning
        )

        with pytest.raises(LLMError) as excinfo:
            await service.handle_session_message(
                chat_session.id, "hello", db_session
            )

        assert excinfo.value is error
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
        """Pre-existing NULL-snapshot user history enters the LLM
        context in order, assistant history is projected out, and old
        rows are NOT rewritten or backfilled in the database."""
        chat_session = _create_session(db_session)
        _add_messages(db_session, chat_session.id, [("q1", "a1")])
        service = ChatService(spy_llm)

        await service.handle_session_message(
            chat_session.id, "q2", db_session,
        )

        # Receive projection keeps the user row and drops the assistant.
        msgs = spy_llm.calls[0]
        assert msgs[-1] == {"role": "user", "content": "q2"}
        assert _decode_receive_history_background(msgs) == ["q1"]

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

        # Receive projection keeps user history only.
        msgs = self.local_spy.calls[0]
        assert msgs[-1] == {"role": "user", "content": "new q"}
        assert _decode_receive_history_background(msgs) == ["old q"]

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

        # The new client saw: system + one history background + new user.
        # The query still keeps the last 20 stored rows, starting at q2.
        # Receive projection then removes all assistant rows, leaving
        # the historical user messages q2..q11.
        msgs = new_spy.calls[0]
        assert msgs[0]["role"] == "system"
        assert msgs[0]["content"].startswith(RECEIVE_TEACHING_PROMPT)
        assert [m["role"] for m in msgs] == ["system", "user", "user"]
        history_contents = _decode_receive_history_background(msgs)
        assert len(history_contents) == MAX_HISTORY_MESSAGES // 2
        assert history_contents == [f"q{i}" for i in range(2, 12)]
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

# ============================================================================
# Receive teaching history projection (pure)
# ============================================================================


class TestHistoryProjectionForMode:
    @staticmethod
    def _message(role, content, *, mode=None, version=None):
        return Message(
            session_id=1,
            role=role,
            content=content,
            interaction_mode_snapshot=mode,
            prompt_version_snapshot=version,
        )

    def test_receive_projection_keeps_only_users(self):
        user_v3 = self._message(
            "user", "u-v3",
            mode=RECEIVE_TEACHING_MODE, version="receive-teaching-v3",
        )
        user_legacy = self._message("user", "u-legacy")
        user_corrective = self._message(
            "user", "u-corrective",
            mode=CORRECTIVE_MODE, version=CORRECTIVE_PROMPT_VERSION,
        )
        assistant_v3 = self._message(
            "assistant", "a-v3",
            mode=RECEIVE_TEACHING_MODE, version="receive-teaching-v3",
        )
        assistant_v4 = self._message(
            "assistant", "a-v4",
            mode=RECEIVE_TEACHING_MODE,
            version=RECEIVE_TEACHING_PROMPT_VERSION,
        )
        assistant_corrective = self._message(
            "assistant", "a-corrective",
            mode=CORRECTIVE_MODE, version=CORRECTIVE_PROMPT_VERSION,
        )
        assistant_legacy = self._message("assistant", "a-legacy")

        history = [
            user_v3,
            assistant_v3,
            assistant_v4,
            assistant_corrective,
            user_legacy,
            assistant_legacy,
            user_corrective,
        ]
        history_before = list(history)
        attributes_before = [
            (
                message.role,
                message.content,
                message.interaction_mode_snapshot,
                message.prompt_version_snapshot,
            )
            for message in history
        ]

        projected = _project_history_for_mode(
            history,
            interaction_mode=RECEIVE_TEACHING_MODE,
        )

        assert projected == [user_v3, user_legacy, user_corrective]
        assert projected[0] is user_v3
        assert projected[1] is user_legacy
        assert projected[2] is user_corrective
        assert history == history_before
        assert attributes_before == [
            (
                message.role,
                message.content,
                message.interaction_mode_snapshot,
                message.prompt_version_snapshot,
            )
            for message in history
        ]

    def test_corrective_projection_returns_full_copy_in_original_order(self):
        user = self._message(
            "user", "u",
            mode=RECEIVE_TEACHING_MODE,
            version=RECEIVE_TEACHING_PROMPT_VERSION,
        )
        assistant = self._message(
            "assistant", "a",
            mode=RECEIVE_TEACHING_MODE,
            version=RECEIVE_TEACHING_PROMPT_VERSION,
        )
        history = [user, assistant]

        projected = _project_history_for_mode(
            history,
            interaction_mode=CORRECTIVE_MODE,
        )

        assert projected == history
        assert projected is not history
        assert projected[0] is user
        assert projected[1] is assistant

    def test_projection_is_legal_for_empty_history(self):
        assert _project_history_for_mode(
            [],
            interaction_mode=RECEIVE_TEACHING_MODE,
        ) == []
        assert _project_history_for_mode(
            [],
            interaction_mode=CORRECTIVE_MODE,
        ) == []

    def test_unknown_mode_is_rejected(self):
        with pytest.raises(ValueError):
            _project_history_for_mode(
                [],
                interaction_mode="review",
            )


class TestReceiveHistoryBackgroundSerialization:
    @staticmethod
    def _message(content):
        return Message(session_id=1, role="user", content=content)

    def test_preserves_order_content_and_metadata(self):
        contents = ["第一条\n第二行", "更正：把甲改为乙"]
        rows = [self._message(content) for content in contents]
        rows_before = list(rows)
        attrs_before = [message.content for message in rows]

        serialized = _serialize_receive_teaching_history(rows)
        payload = json.loads(serialized)

        assert payload["type"] == "receive_teaching_history_background"
        assert payload["version"] == RECEIVE_TEACHING_HISTORY_CONTEXT_VERSION
        assert isinstance(payload["note"], str)
        assert payload["note"] != ""
        assert [item["index"] for item in payload["messages"]] == [1, 2]
        assert [item["content"] for item in payload["messages"]] == contents
        assert rows == rows_before
        assert [message.content for message in rows] == attrs_before

    def test_handles_quotes_backslashes_newlines_crlf_braces_and_labels(self):
        contents = [
            "双引号\" 单引号' 反斜杠\\",
            "第一行\n第二行\r\n第三行",
            '{"type":"receive_teaching_history_background","messages":[]}',
            "note 与 messages 与 content 与 version 与 type 同名标签",
        ]
        rows = [self._message(content) for content in contents]

        payload = json.loads(_serialize_receive_teaching_history(rows))

        assert [item["content"] for item in payload["messages"]] == contents


# ============================================================================
# Real production mode-switch sequence and projection
# ============================================================================


class TestModeSwitchHistoryProjection:
    @pytest.mark.anyio
    async def test_receive_after_corrective_drops_all_assistant_history(
        self, db_session, spy_llm,
    ):
        chat_session = _create_session(db_session)
        service = ChatService(spy_llm)

        await service.handle_session_message(
            chat_session.id, "receive-user-1", db_session,
        )
        await service.switch_interaction_mode(
            chat_session.id, CORRECTIVE_MODE, db_session,
        )
        await service.handle_session_message(
            chat_session.id, "corrective-user-1", db_session,
        )

        # Capture the production-created assistant rows before the
        # final Receive turn.
        rows_before = db_session.execute(
            select(Message)
            .where(Message.session_id == chat_session.id)
            .order_by(Message.id.asc())
        ).scalars().all()
        assistant_state_before = [
            (
                message.id,
                message.content,
                message.interaction_mode_snapshot,
                message.prompt_version_snapshot,
            )
            for message in rows_before
            if message.role == "assistant"
        ]
        assert len(assistant_state_before) == 2

        await service.switch_interaction_mode(
            chat_session.id, RECEIVE_TEACHING_MODE, db_session,
        )
        await service.handle_session_message(
            chat_session.id, "receive-user-2", db_session,
        )

        corrective_call = spy_llm.calls[1]
        assert corrective_call[0]["content"] == CORRECTIVE_PROMPT
        assert [message["role"] for message in corrective_call] == [
            "system", "user", "assistant", "user",
        ]

        last_receive_call = spy_llm.calls[-1]
        assert last_receive_call[0]["role"] == "system"
        assert last_receive_call[0]["content"].startswith(RECEIVE_TEACHING_PROMPT)
        assert [message["role"] for message in last_receive_call] == [
            "system", "user", "user",
        ]
        assert last_receive_call[-1] == {
            "role": "user",
            "content": "receive-user-2",
        }
        assert _decode_receive_history_background(last_receive_call) == [
            "receive-user-1",
            "corrective-user-1",
        ]
        assert all(
            "test reply" not in message["content"]
            for message in last_receive_call
            if message["role"] == "assistant"
        )

        rows_after = db_session.execute(
            select(Message)
            .where(Message.session_id == chat_session.id)
            .order_by(Message.id.asc())
        ).scalars().all()
        assistant_state_after = [
            (
                message.id,
                message.content,
                message.interaction_mode_snapshot,
                message.prompt_version_snapshot,
            )
            for message in rows_after
            if message.role == "assistant"
        ]
        for old_state in assistant_state_before:
            assert old_state in assistant_state_after

        assistant_rows_after = [
            message for message in rows_after
            if message.role == "assistant"
        ]
        assert assistant_rows_after[0].interaction_mode_snapshot == (
            RECEIVE_TEACHING_MODE
        )
        assert assistant_rows_after[0].prompt_version_snapshot == (
            RECEIVE_TEACHING_PROMPT_VERSION + "+l1-en"
        )
        assert assistant_rows_after[1].interaction_mode_snapshot == (
            CORRECTIVE_MODE
        )
        assert assistant_rows_after[1].prompt_version_snapshot == (
            CORRECTIVE_PROMPT_VERSION
        )
        assert assistant_rows_after[2].interaction_mode_snapshot == (
            RECEIVE_TEACHING_MODE
        )
        assert assistant_rows_after[2].prompt_version_snapshot == (
            RECEIVE_TEACHING_PROMPT_VERSION + "+l1-en"
        )


# ============================================================================
# Receive teaching reply-language assembly
# ============================================================================


class TestReceiveTeachingLanguageFollowingAssembly:
    """Production message assembly for reply-language following.

    These tests prove the *inputs* sent to the model: the current
    request is last and byte-for-byte unchanged, history is ordered and
    unchanged, and the reply-language rule is in the actual system
    prompt.  They cannot prove that a real model obeys the rule.
    """

    ENGLISH_MATH = (
        "What is the difference between y_i and y_hat_i?\n"
        "Use only the meanings I taught you."
    )
    CHINESE_WITH_ACTION = (
        "请说明 action A 在流程里的作用，变量 n 保持不变。"
    )
    CODE_AND_QUOTES = (
        'Keep `dL/dw`, `y_i`, and "y_hat_i = w * x_i + b" '
        "exactly as written."
    )

    @pytest.mark.anyio
    @pytest.mark.parametrize(
        "current",
        (
            ENGLISH_MATH,
            CHINESE_WITH_ACTION,
            "y_hat_i = w * x_i + b",
            CODE_AND_QUOTES,
        ),
    )
    async def test_no_history_request_is_last_and_unchanged(
        self, db_session, spy_llm, current,
    ):
        chat_session = _create_session(db_session)
        service = ChatService(spy_llm)

        await service.handle_session_message(
            chat_session.id, current, db_session,
        )

        assert len(spy_llm.calls) == 1
        messages = spy_llm.calls[0]
        assert [message["role"] for message in messages] == [
            "system", "user",
        ]
        assert messages[0]["role"] == "system"
        assert messages[0]["content"].startswith(RECEIVE_TEACHING_PROMPT)
        assert messages[-1] == {"role": "user", "content": current}

        stored_content = db_session.execute(
            select(Message.content).where(
                Message.session_id == chat_session.id,
                Message.role == "user",
            )
        ).scalars().all()
        assert stored_content == [current]

    @pytest.mark.anyio
    @pytest.mark.parametrize(
        ("history_users", "current"),
        (
            (
                (
                    "我教你：动作甲之后是动作乙。",
                    "更正：动作乙之后是动作丙。",
                ),
                ENGLISH_MATH,
            ),
            (
                (
                    "I am teaching you: action A comes before action B.",
                    "Correction: action B comes before action C.",
                ),
                CHINESE_WITH_ACTION,
            ),
        ),
    )
    async def test_history_language_does_not_change_current_request(
        self, db_session, spy_llm, history_users, current,
    ):
        chat_session = _create_session(db_session)
        _add_messages(
            db_session,
            chat_session.id,
            [(text, "assistant reply") for text in history_users],
        )
        service = ChatService(spy_llm)

        await service.handle_session_message(
            chat_session.id, current, db_session,
        )

        assert len(spy_llm.calls) == 1
        messages = spy_llm.calls[0]
        assert [message["role"] for message in messages] == [
            "system", "user", "user",
        ]
        assert messages[0]["role"] == "system"
        assert messages[0]["content"].startswith(RECEIVE_TEACHING_PROMPT)
        assert messages[-1] == {"role": "user", "content": current}
        assert _decode_receive_history_background(messages) == list(
            history_users
        )

        stored_user_content = db_session.execute(
            select(Message.content)
            .where(
                Message.session_id == chat_session.id,
                Message.role == "user",
            )
            .order_by(Message.id.asc())
        ).scalars().all()
        assert stored_user_content == [*history_users, current]

    @pytest.mark.anyio
    @pytest.mark.parametrize(
        ("history_users", "current"),
        (
            (
                (
                    "For this conversation, please answer in English.",
                    "我教你：动作 A 之后是动作 B；只有 C 出现时才执行 D。",
                ),
                "请告诉我：D 在什么条件下执行？",
            ),
            (
                (
                    "For this conversation, please answer in English.",
                    "我教你：动作 A 之后是动作 B。",
                ),
                "这一次请用中文解释：A 和 B 的顺序是什么？",
            ),
            (
                (
                    "For this conversation, please answer in English.",
                    "I am teaching you: D is performed only when C is present.",
                    "从这条消息开始，后续请都用中文回答。",
                ),
                "What is the condition for performing D?",
            ),
        ),
    )
    async def test_persistent_and_one_turn_language_context_is_verbatim(
        self, db_session, spy_llm, history_users, current,
    ):
        chat_session = _create_session(db_session)
        _add_messages(
            db_session,
            chat_session.id,
            [(text, "assistant reply") for text in history_users],
        )
        service = ChatService(spy_llm)

        await service.handle_session_message(
            chat_session.id, current, db_session,
        )

        assert len(spy_llm.calls) == 1
        messages = spy_llm.calls[0]
        assert [message["role"] for message in messages] == [
            "system", "user", "user",
        ]
        assert messages[0]["role"] == "system"
        assert messages[0]["content"].startswith(RECEIVE_TEACHING_PROMPT)
        assert messages[-1] == {"role": "user", "content": current}
        assert _decode_receive_history_background(messages) == list(
            history_users
        )

        stored_user_content = db_session.execute(
            select(Message.content)
            .where(
                Message.session_id == chat_session.id,
                Message.role == "user",
            )
            .order_by(Message.id.asc())
        ).scalars().all()
        assert stored_user_content == [*history_users, current]

    @pytest.mark.anyio
    @pytest.mark.parametrize(
        ("history_user", "formula"),
        (
            (
                "I am teaching you: y_i is the target value; "
                "y_hat_i is the predicted value; "
                "y_hat_i = w * x_i + b.",
                "y_hat_i = w * x_i + b",
            ),
            (
                "我教你：y_i 是目标值；y_hat_i 是预测值；"
                "y_hat_i = w * x_i + b。",
                "y_hat_i = w * x_i + b",
            ),
        ),
    )
    async def test_formula_only_current_sends_raw_history_verbatim(
        self, db_session, spy_llm, history_user, formula,
    ):
        chat_session = _create_session(db_session)
        _add_messages(
            db_session, chat_session.id, [(history_user, "assistant reply")],
        )
        service = ChatService(spy_llm)

        await service.handle_session_message(
            chat_session.id, formula, db_session,
        )

        assert len(spy_llm.calls) == 1
        messages = spy_llm.calls[0]
        assert [message["role"] for message in messages] == [
            "system", "user", "user",
        ]
        assert messages[-1] == {"role": "user", "content": formula}
        assert _decode_receive_history_background(messages) == [history_user]

    @pytest.mark.anyio
    @pytest.mark.parametrize(
        ("current", "history_user"),
        (
            (
                "请用英文总结我刚才教的内容。",
                "我教你：y_i 是目标值；y_hat_i 是预测值。",
            ),
            (
                "Please explain this in Chinese.",
                "I am teaching you: y_i is the target value.",
            ),
        ),
    )
    async def test_explicit_language_request_is_not_rewritten(
        self, db_session, spy_llm, current, history_user,
    ):
        chat_session = _create_session(db_session)
        _add_messages(
            db_session, chat_session.id, [(history_user, "assistant reply")],
        )
        service = ChatService(spy_llm)

        await service.handle_session_message(
            chat_session.id, current, db_session,
        )

        assert len(spy_llm.calls) == 1
        messages = spy_llm.calls[0]
        assert messages[-1] == {"role": "user", "content": current}
        assert _decode_receive_history_background(messages) == [history_user]

    @pytest.mark.anyio
    async def test_api_and_local_receive_get_identical_messages(
        self, db_session,
    ):
        api_spy = KwargSpyLLMClient(response="api reply")
        local_spy = KwargSpyLLMClient(response="local reply")
        registry = _switch_registry(
            default_client=api_spy,
            default_kind="api",
            default_model="remote-m1",
            extras=[{
                "id": "local",
                "label": "Local",
                "kind": "local",
                "model": "qwen3.5:9b",
                "client": local_spy,
            }],
        )
        service = ChatService(profiles=registry)
        api_session = _make_bound_session(db_session, "default", "remote-m1")
        local_session = _make_bound_session(
            db_session, "local", "qwen3.5:9b",
        )
        for chat_session in (api_session, local_session):
            _add_messages(
                db_session,
                chat_session.id,
                [("我教你：动作甲之后是动作乙。", "assistant reply")],
            )

        await service.handle_session_message(
            api_session.id, self.ENGLISH_MATH, db_session,
        )
        await service.handle_session_message(
            local_session.id, self.ENGLISH_MATH, db_session,
        )

        assert api_spy.calls == local_spy.calls
        assert api_spy.calls[0][0]["role"] == "system"
        assert api_spy.calls[0][0]["content"].startswith(RECEIVE_TEACHING_PROMPT)
        assert api_spy.calls[0][-1] == {
            "role": "user", "content": self.ENGLISH_MATH,
        }
        assert api_spy.generate_kwargs == [{}]
        assert local_spy.generate_kwargs == [{"temperature": 0.0}]


# ============================================================================
# Local Receive temperature routing
# ============================================================================


class TestReceiveTeachingTemperatureRouting:
    def _registry(self):
        api_spy = KwargSpyLLMClient(response="api reply")
        local_spy = KwargSpyLLMClient(response="local reply")
        registry = _switch_registry(
            default_client=api_spy,
            default_kind="api",
            default_model="remote-m1",
            extras=[{
                "id": "local",
                "label": "Local",
                "kind": "local",
                "model": "qwen3.5:4b",
                "client": local_spy,
            }],
        )
        return registry, api_spy, local_spy

    @pytest.mark.anyio
    async def test_local_receive_passes_temperature_zero(self, db_session):
        registry, api_spy, local_spy = self._registry()
        service = ChatService(profiles=registry)
        session = _make_bound_session(db_session, "local", "qwen3.5:4b")

        await service.handle_session_message(session.id, "hello", db_session)

        assert local_spy.calls[0][0]["role"] == "system"
        assert local_spy.calls[0][0]["content"].startswith(RECEIVE_TEACHING_PROMPT)
        assert local_spy.generate_kwargs == [{"temperature": 0.0}]
        assert api_spy.generate_kwargs == []

    @pytest.mark.anyio
    async def test_api_receive_omits_temperature(self, db_session):
        registry, api_spy, local_spy = self._registry()
        service = ChatService(profiles=registry)
        session = _make_bound_session(db_session, "default", "remote-m1")

        await service.handle_session_message(session.id, "hello", db_session)

        assert api_spy.calls[0][0]["role"] == "system"
        assert api_spy.calls[0][0]["content"].startswith(RECEIVE_TEACHING_PROMPT)
        assert api_spy.generate_kwargs == [{}]
        assert local_spy.generate_kwargs == []

    @pytest.mark.anyio
    async def test_local_corrective_omits_temperature(self, db_session):
        registry, api_spy, local_spy = self._registry()
        service = ChatService(profiles=registry)
        session = _make_bound_session(db_session, "local", "qwen3.5:4b")

        await service.switch_interaction_mode(
            session.id, CORRECTIVE_MODE, db_session,
        )
        await service.handle_session_message(session.id, "hello", db_session)

        assert local_spy.generate_kwargs == [{}]
        assert api_spy.generate_kwargs == []

    @pytest.mark.anyio
    async def test_api_corrective_omits_temperature(self, db_session):
        registry, api_spy, local_spy = self._registry()
        service = ChatService(profiles=registry)
        session = _make_bound_session(db_session, "default", "remote-m1")

        await service.switch_interaction_mode(
            session.id, CORRECTIVE_MODE, db_session,
        )
        await service.handle_session_message(session.id, "hello", db_session)

        assert api_spy.generate_kwargs == [{}]
        assert local_spy.generate_kwargs == []

    @pytest.mark.anyio
    async def test_temperature_is_not_persisted_between_calls(self, db_session):
        registry, api_spy, local_spy = self._registry()
        service = ChatService(profiles=registry)
        session = _make_bound_session(db_session, "local", "qwen3.5:4b")

        await service.handle_session_message(session.id, "receive-1", db_session)
        await service.switch_interaction_mode(
            session.id, CORRECTIVE_MODE, db_session,
        )
        await service.handle_session_message(
            session.id, "corrective-1", db_session,
        )
        await service.switch_interaction_mode(
            session.id, RECEIVE_TEACHING_MODE, db_session,
        )
        await service.handle_session_message(
            session.id, "receive-2", db_session,
        )

        assert local_spy.generate_kwargs == [
            {"temperature": 0.0},
            {},
            {"temperature": 0.0},
        ]
        assert api_spy.generate_kwargs == []
