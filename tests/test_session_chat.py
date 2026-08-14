"""Tests for POST /api/sessions/{session_id}/messages.

Every test uses a temporary SQLite file and SpyLLMClient — no real
network requests are ever made.
"""

import asyncio
import copy
import time
from datetime import datetime

import httpx
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import sessionmaker

from backend.chat_service import ChatService, SessionLockRegistry
from backend.database import (
    _is_memory_database,
    create_database_engine,
    create_tables,
    get_db,
)
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
# Controlled Spy (Event-based concurrency tests)
# ---------------------------------------------------------------------------


class ControlledSpy:
    """Spy that blocks inside ``generate()`` until :meth:`unblock` is called.

    Tracks ``active`` (currently in generate), ``max_active`` (peak
    concurrency), and ``entered_count`` (total entries) so tests can
    use structural assertions instead of wall-clock thresholds.
    """

    def __init__(self, response: str = "reply") -> None:
        self._block = asyncio.Event()
        self.active = 0
        self.max_active = 0
        self.entered_count = 0
        self.calls: list[list[LLMMessage]] = []
        self.response = response

    async def generate(self, messages: list[LLMMessage]) -> str:
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        self.entered_count += 1
        self.calls.append(copy.deepcopy(messages))
        await self._block.wait()
        self.active -= 1
        return self.response

    def unblock(self) -> None:
        """Release all currently blocked ``generate()`` calls."""
        self._block.set()

    def reset_block(self) -> None:
        """Reset the block for the next round."""
        self._block.clear()

    async def wait_entered(self, target: int, timeout: float = 5.0) -> None:
        """Spin until at least *target* calls have entered ``generate()``."""
        deadline = time.monotonic() + timeout
        while self.entered_count < target:
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"Expected {target} entered, got {self.entered_count}"
                )
            await asyncio.sleep(0.01)


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

    try:
        with TestClient(app) as c:
            yield c
    finally:
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

    def test_system_prompt_not_in_database(
        self, client, spy_llm, test_session_factory
    ):
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
        user_a = next(m for m in msgs_a if m.role == "user")
        assert user_a.content == "msg A"

        assert len(msgs_b) == 2
        user_b = next(m for m in msgs_b if m.role == "user")
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


# ============================================================================
# Concurrency — Event-based (no time thresholds)
# ============================================================================


class TestConcurrency:
    """Concurrent sends to the same session are serialised; different
    sessions remain parallel.  Uses ControlledSpy with explicit Events
    so assertions are structural, not time-based."""

    @pytest.mark.anyio
    async def test_same_session_max_active_is_one(
        self, test_engine, test_session_factory,
    ):
        """Three concurrent sends to the same session → max_active == 1.

        The per-session lock serialises access so at most one task is
        inside ``generate()`` at any moment.  The ControlledSpy blocks
        all calls, so the first task enters and blocks while the other
        two queue up on the lock.  When we release, they execute
        one-by-one under the lock — *max_active* never exceeds 1.
        """
        import backend.main as main_module

        spy = ControlledSpy()
        chat_svc = ChatService(spy)

        def override_get_db():
            db = test_session_factory()
            try:
                yield db
            finally:
                db.close()

        app.dependency_overrides[get_db] = override_get_db
        original_svc = main_module.chat_service
        main_module.chat_service = chat_svc

        try:
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://test"
            ) as ac:
                create_resp = await ac.post("/api/sessions")
                assert create_resp.status_code == 201
                sid = create_resp.json()["id"]

                async def send(text: str):
                    resp = await ac.post(
                        f"/api/sessions/{sid}/messages",
                        json={"message": text},
                    )
                    return resp

                # Launch all three concurrently
                task1 = asyncio.create_task(send("msg-1"))
                task2 = asyncio.create_task(send("msg-2"))
                task3 = asyncio.create_task(send("msg-3"))

                # The first task enters generate and blocks; the other
                # two queue on the per-session lock (not yet in generate).
                await spy.wait_entered(1)
                # Give tasks 2 & 3 time to reach the lock
                await asyncio.sleep(0.1)
                assert spy.max_active == 1
                assert spy.entered_count == 1

                # Release all — they serialize under the session lock
                spy.unblock()
                results = await asyncio.gather(task1, task2, task3)

                for r in results:
                    assert r.status_code == 200, r.text

                # max_active never exceeded 1 (session lock serialises)
                assert spy.max_active == 1
                assert spy.entered_count == 3

                # Third call must see history from the first two
                last_call = spy.calls[2]
                contents = [m["content"] for m in last_call]
                assert "msg-1" in contents
                assert "msg-2" in contents

                # Registry must be clean
                assert len(chat_svc._lock_registry._entries) == 0
        finally:
            app.dependency_overrides.pop(get_db, None)
            main_module.chat_service = original_svc

    @pytest.mark.anyio
    async def test_different_sessions_max_active_is_two(
        self, test_engine, test_session_factory,
    ):
        """Two sends to different sessions → max_active == 2."""
        import backend.main as main_module

        spy = ControlledSpy()
        chat_svc = ChatService(spy)

        def override_get_db():
            db = test_session_factory()
            try:
                yield db
            finally:
                db.close()

        app.dependency_overrides[get_db] = override_get_db
        original_svc = main_module.chat_service
        main_module.chat_service = chat_svc

        try:
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://test"
            ) as ac:
                r1 = await ac.post("/api/sessions")
                r2 = await ac.post("/api/sessions")
                sid_a = r1.json()["id"]
                sid_b = r2.json()["id"]

                async def send(sid: int, text: str):
                    resp = await ac.post(
                        f"/api/sessions/{sid}/messages",
                        json={"message": text},
                    )
                    return resp

                # Launch both concurrently — different session locks
                task_a = asyncio.create_task(send(sid_a, "msg-a"))
                task_b = asyncio.create_task(send(sid_b, "msg-b"))

                # Both should enter generate in parallel
                await spy.wait_entered(2)
                assert spy.max_active == 2
                assert spy.entered_count == 2

                # Release both
                spy.unblock()
                results = await asyncio.gather(task_a, task_b)

                for r in results:
                    assert r.status_code == 200, r.text

                assert len(chat_svc._lock_registry._entries) == 0
        finally:
            app.dependency_overrides.pop(get_db, None)
            main_module.chat_service = original_svc


# ============================================================================
# Lock cancellation safety
# ============================================================================


class TestLockCancellation:
    """Cancelling a task while it waits for a session lock must not
    leak reference counts."""

    @pytest.mark.anyio
    async def test_waiting_task_cancelled_ref_count_returns_to_zero(
        self, test_engine, test_session_factory,
    ):
        """Cancel a waiter — registry must be clean afterwards."""
        import backend.main as main_module

        spy = ControlledSpy()
        chat_svc = ChatService(spy)

        def override_get_db():
            db = test_session_factory()
            try:
                yield db
            finally:
                db.close()

        app.dependency_overrides[get_db] = override_get_db
        original_svc = main_module.chat_service
        main_module.chat_service = chat_svc

        try:
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://test"
            ) as ac:
                create_resp = await ac.post("/api/sessions")
                assert create_resp.status_code == 201
                sid = create_resp.json()["id"]

                async def send(text: str):
                    resp = await ac.post(
                        f"/api/sessions/{sid}/messages",
                        json={"message": text},
                    )
                    return resp

                # Task 1 — enters generate, blocks
                task1 = asyncio.create_task(send("msg-1"))
                await spy.wait_entered(1)

                # Task 2 — waits for the per-session lock
                task2 = asyncio.create_task(send("msg-2"))
                await asyncio.sleep(0.1)  # let it hit lock.acquire()

                # Cancel task 2 while it's waiting
                task2.cancel()
                try:
                    await task2
                except asyncio.CancelledError:
                    pass

                # Release task 1
                spy.unblock()
                await task1

                # Registry must be clean — no leaked entry
                assert len(chat_svc._lock_registry._entries) == 0
        finally:
            app.dependency_overrides.pop(get_db, None)
            main_module.chat_service = original_svc


# ============================================================================
# Delete during generation (no race)
# ============================================================================


class TestDeleteDuringGeneration:
    """DELETE /api/sessions/{id} uses the same per-session lock, so a
    delete cannot race with an in-progress generation."""

    @pytest.mark.anyio
    async def test_delete_waits_for_generation_then_deletes(
        self, test_engine, test_session_factory,
    ):
        """Send holds lock first → delete waits → both succeed in order."""
        import backend.main as main_module

        spy = ControlledSpy()
        chat_svc = ChatService(spy)

        def override_get_db():
            db = test_session_factory()
            try:
                yield db
            finally:
                db.close()

        app.dependency_overrides[get_db] = override_get_db
        original_svc = main_module.chat_service
        main_module.chat_service = chat_svc

        try:
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://test"
            ) as ac:
                create_resp = await ac.post("/api/sessions")
                assert create_resp.status_code == 201
                sid = create_resp.json()["id"]

                async def send(text: str):
                    return await ac.post(
                        f"/api/sessions/{sid}/messages",
                        json={"message": text},
                    )

                async def delete():
                    return await ac.delete(f"/api/sessions/{sid}")

                # Start send — enters generate, blocks
                task_send = asyncio.create_task(send("hello"))
                await spy.wait_entered(1)

                # Start delete — blocked by per-session lock
                task_del = asyncio.create_task(delete())

                # Release the send
                spy.unblock()
                send_resp = await task_send
                assert send_resp.status_code == 200

                # Delete should now complete
                del_resp = await task_del
                assert del_resp.status_code == 200
                assert del_resp.json() == {"ok": True}

                # Session is gone
                get_resp = await ac.get(f"/api/sessions/{sid}")
                assert get_resp.status_code == 404

                # No orphan messages — query the DB directly
                db = test_session_factory()
                orphan_msgs = db.execute(
                    select(Message).where(Message.session_id == sid)
                ).scalars().all()
                db.close()
                assert len(orphan_msgs) == 0

                assert len(chat_svc._lock_registry._entries) == 0
        finally:
            app.dependency_overrides.pop(get_db, None)
            main_module.chat_service = original_svc

    @pytest.mark.anyio
    async def test_delete_holds_lock_first_send_gets_404(
        self, test_engine, test_session_factory,
    ):
        """Delete holds lock first → send sees 404."""
        import backend.main as main_module

        spy = ControlledSpy()
        chat_svc = ChatService(spy)

        def override_get_db():
            db = test_session_factory()
            try:
                yield db
            finally:
                db.close()

        app.dependency_overrides[get_db] = override_get_db
        original_svc = main_module.chat_service
        main_module.chat_service = chat_svc

        try:
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://test"
            ) as ac:
                create_resp = await ac.post("/api/sessions")
                assert create_resp.status_code == 201
                sid = create_resp.json()["id"]

                async def send(text: str):
                    return await ac.post(
                        f"/api/sessions/{sid}/messages",
                        json={"message": text},
                    )

                async def delete():
                    return await ac.delete(f"/api/sessions/{sid}")

                # Start delete first — uses the lock
                task_del = asyncio.create_task(delete())
                # The delete operation acquires the lock, checks session
                # exists, deletes it, and commits.  Since delete is fast
                # (no LLM call), it completes before we even start send.
                del_resp = await task_del
                assert del_resp.status_code == 200
                assert del_resp.json() == {"ok": True}

                # Send should get 404
                send_resp = await send("hello")
                assert send_resp.status_code == 404

                assert len(chat_svc._lock_registry._entries) == 0
        finally:
            app.dependency_overrides.pop(get_db, None)
            main_module.chat_service = original_svc


# ============================================================================
# In-memory SQLite
# ============================================================================


class TestInMemorySQLite:
    """Verify that :memory: databases work correctly."""

    def test_is_memory_detects_sqlite_memory(self):
        """_is_memory_database returns True for sqlite:///:memory:."""
        assert _is_memory_database("sqlite:///:memory:") is True

    def test_is_memory_detects_sqlite_empty(self):
        """_is_memory_database returns True for sqlite:// (no path)."""
        assert _is_memory_database("sqlite://") is True

    def test_is_memory_detects_pysqlite_memory(self):
        """_is_memory_database returns True for sqlite+pysqlite:///:memory:."""
        assert _is_memory_database("sqlite+pysqlite:///:memory:") is True

    def test_is_memory_rejects_file(self):
        """_is_memory_database returns False for file URLs."""
        assert _is_memory_database("sqlite:///foo.db") is False

    def test_memory_engine_uses_static_pool(self):
        """create_database_engine with :memory: uses StaticPool."""
        from sqlalchemy.pool import StaticPool

        eng = create_database_engine("sqlite:///:memory:")
        pool = eng.pool
        assert isinstance(pool, StaticPool)
        eng.dispose()

    def test_file_engine_does_not_use_static_pool(self):
        """create_database_engine with a file URL uses default pool."""
        from sqlalchemy.pool import StaticPool

        eng = create_database_engine("sqlite:///test_file.db")
        assert not isinstance(eng.pool, StaticPool)
        eng.dispose()

    def test_memory_sqlite_real_app_post_sessions(self):
        """Real FastAPI app with in-memory SQLite can POST /api/sessions.

        This test uses the actual ``app`` and the real ``get_db``
        dependency (via dependency_overrides pointing to an in-memory
        engine), not a mock.  It proves that ``StaticPool`` +
        ``check_same_thread=False`` works across threads.
        """
        import backend.main as main_module

        # Build an in-memory engine
        mem_engine = create_database_engine("sqlite:///:memory:")
        create_tables(bind=mem_engine)
        MemSessionLocal = sessionmaker(
            bind=mem_engine, autoflush=False, expire_on_commit=False
        )

        def override_get_db():
            db = MemSessionLocal()
            try:
                yield db
            finally:
                db.close()

        app.dependency_overrides[get_db] = override_get_db
        original_svc = main_module.chat_service
        main_module.chat_service = ChatService(SpyLLMClient(response="test reply"))

        try:
            with TestClient(app) as client:
                # POST /api/sessions
                resp = client.post("/api/sessions")
                assert resp.status_code == 201
                data = resp.json()
                assert "id" in data
                sid = data["id"]

                # POST /api/sessions/{id}/messages
                resp2 = client.post(
                    f"/api/sessions/{sid}/messages",
                    json={"message": "hello memory"},
                )
                assert resp2.status_code == 200
                body = resp2.json()
                assert body["user_message"]["content"] == "hello memory"
                assert body["assistant_message"]["content"] == "test reply"

                # GET /api/sessions
                resp3 = client.get("/api/sessions")
                assert resp3.status_code == 200
                sessions_list = resp3.json()
                assert len(sessions_list) == 1
                assert sessions_list[0]["id"] == sid
        finally:
            app.dependency_overrides.pop(get_db, None)
            main_module.chat_service = original_svc
            mem_engine.dispose()


# ============================================================================
# Lock registry unit tests
# ============================================================================


class TestSessionLockRegistry:
    """Direct unit tests for SessionLockRegistry."""

    @pytest.mark.anyio
    async def test_entry_cleaned_up_after_use(self):
        """Registry has no entries after a normal acquire/release cycle."""
        registry = SessionLockRegistry()

        async with registry.session_lock(1):
            assert len(registry._entries) == 1
            assert registry._entries[1].ref_count == 1

        assert len(registry._entries) == 0

    @pytest.mark.anyio
    async def test_ref_count_with_multiple_waiters(self):
        """Three waiters → ref_count peaks at 3, then 0 after all done."""
        registry = SessionLockRegistry()

        async def worker():
            async with registry.session_lock(1):
                pass

        # First worker grabs the lock
        async with registry.session_lock(1):
            assert registry._entries[1].ref_count == 1

            # Two more workers queue up
            task2 = asyncio.create_task(worker())
            task3 = asyncio.create_task(worker())
            await asyncio.sleep(0.1)

            # Both are waiting — ref_count should be 3
            assert registry._entries[1].ref_count == 3

        # Lock released — waiters proceed
        await asyncio.gather(task2, task3)
        assert len(registry._entries) == 0

    @pytest.mark.anyio
    async def test_cancelled_waiter_decrements_ref_count(self):
        """Cancelling a waiter still decrements ref_count."""
        registry = SessionLockRegistry()

        async def worker():
            async with registry.session_lock(1):
                pass

        # Hold the lock
        async with registry.session_lock(1):
            assert registry._entries[1].ref_count == 1

            # Start a waiter
            task = asyncio.create_task(worker())
            await asyncio.sleep(0.1)
            assert registry._entries[1].ref_count == 2

            # Cancel the waiter
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

            # ref_count should be back to 1 (just us)
            assert registry._entries[1].ref_count == 1

        # After we release, everything is clean
        assert len(registry._entries) == 0


# ============================================================================
# Auto-title behaviour
# ============================================================================


class TestAutoTitle:
    """Auto-title generation from the first user message."""

    def test_auto_title_persists_when_llm_fails(
        self, test_engine, test_session_factory, spy_llm,
    ):
        """Even when the LLM raises, the auto-title is already saved."""
        import backend.main as main_module

        # Use a spy that raises after Phase 1
        spy_llm.error = LLMError("upstream failure", status_code=502)

        def override_get_db():
            db = test_session_factory()
            try:
                yield db
            finally:
                db.close()

        app.dependency_overrides[get_db] = override_get_db
        original_svc = main_module.chat_service
        main_module.chat_service = ChatService(spy_llm)

        try:
            with TestClient(app) as client:
                resp = client.post("/api/sessions")
                sid = resp.json()["id"]

                client.post(
                    f"/api/sessions/{sid}/messages",
                    json={"message": "My cool topic"},
                )

                session = client.get(f"/api/sessions/{sid}").json()
                assert session["title"] == "My cool topic"
        finally:
            app.dependency_overrides.pop(get_db, None)
            main_module.chat_service = original_svc


# ============================================================================
# Rename concurrency safety
# ============================================================================


class TestRenameConcurrency:
    """Rename uses the same per-session lock — no races with send/delete."""

    @pytest.mark.anyio
    async def test_rename_during_send_completes_after(
        self, test_engine, test_session_factory,
    ):
        """Send holds lock first → rename waits → both succeed."""
        import backend.main as main_module

        spy = ControlledSpy()
        chat_svc = ChatService(spy)

        def override_get_db():
            db = test_session_factory()
            try:
                yield db
            finally:
                db.close()

        app.dependency_overrides[get_db] = override_get_db
        original_svc = main_module.chat_service
        main_module.chat_service = chat_svc

        try:
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://test"
            ) as ac:
                create_resp = await ac.post("/api/sessions")
                sid = create_resp.json()["id"]

                # Start send — enters generate, blocks
                task_send = asyncio.create_task(
                    ac.post(
                        f"/api/sessions/{sid}/messages",
                        json={"message": "hello"},
                    )
                )
                await spy.wait_entered(1)

                # Start rename — blocked by lock
                task_rename = asyncio.create_task(
                    ac.patch(
                        f"/api/sessions/{sid}",
                        json={"title": "Renamed During Send"},
                    )
                )

                # Release send
                spy.unblock()
                send_resp = await task_send
                assert send_resp.status_code == 200

                # Rename should now complete
                rename_resp = await task_rename
                assert rename_resp.status_code == 200
                assert rename_resp.json()["title"] == "Renamed During Send"

                # Verify title persisted
                get_resp = await ac.get(f"/api/sessions/{sid}")
                assert get_resp.json()["title"] == "Renamed During Send"

                assert len(chat_svc._lock_registry._entries) == 0
        finally:
            app.dependency_overrides.pop(get_db, None)
            main_module.chat_service = original_svc

    @pytest.mark.anyio
    async def test_rename_blocks_auto_title_then_send(
        self, test_engine, test_session_factory,
    ):
        """Rename first → send → auto-title does NOT overwrite."""
        import backend.main as main_module

        spy = ControlledSpy()
        chat_svc = ChatService(spy)

        def override_get_db():
            db = test_session_factory()
            try:
                yield db
            finally:
                db.close()

        app.dependency_overrides[get_db] = override_get_db
        original_svc = main_module.chat_service
        main_module.chat_service = chat_svc

        try:
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://test"
            ) as ac:
                create_resp = await ac.post("/api/sessions")
                sid = create_resp.json()["id"]

                # Rename first (sets title_is_manual = True)
                await ac.patch(
                    f"/api/sessions/{sid}",
                    json={"title": "My Manual Title"},
                )

                # Now send
                task_send = asyncio.create_task(
                    ac.post(
                        f"/api/sessions/{sid}/messages",
                        json={"message": "auto should not apply"},
                    )
                )
                await spy.wait_entered(1)
                spy.unblock()
                send_resp = await task_send
                assert send_resp.status_code == 200

                get_resp = await ac.get(f"/api/sessions/{sid}")
                assert get_resp.json()["title"] == "My Manual Title"

                assert len(chat_svc._lock_registry._entries) == 0
        finally:
            app.dependency_overrides.pop(get_db, None)
            main_module.chat_service = original_svc

    @pytest.mark.anyio
    async def test_delete_first_then_rename_404(
        self, test_engine, test_session_factory,
    ):
        """Delete holds lock first → rename gets 404."""
        import backend.main as main_module

        chat_svc = ChatService(SpyLLMClient())

        def override_get_db():
            db = test_session_factory()
            try:
                yield db
            finally:
                db.close()

        app.dependency_overrides[get_db] = override_get_db
        original_svc = main_module.chat_service
        main_module.chat_service = chat_svc

        try:
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://test"
            ) as ac:
                create_resp = await ac.post("/api/sessions")
                sid = create_resp.json()["id"]

                del_resp = await ac.delete(f"/api/sessions/{sid}")
                assert del_resp.status_code == 200

                rename_resp = await ac.patch(
                    f"/api/sessions/{sid}",
                    json={"title": "After Delete"},
                )
                assert rename_resp.status_code == 404

                assert len(chat_svc._lock_registry._entries) == 0
        finally:
            app.dependency_overrides.pop(get_db, None)
            main_module.chat_service = original_svc


# ============================================================================
# Phase 1 atomicity — user message + auto-title in one commit
# ============================================================================


class TestPhase1Atomicity:
    """Auto-title and user message share the same Phase 1 commit."""

    def test_llm_failure_both_user_msg_and_title_persist(
        self, test_engine, test_session_factory, spy_llm,
    ):
        """LLM error → user msg + auto-title both persisted."""
        import backend.main as main_module

        spy_llm.error = LLMError("upstream failure", status_code=502)

        def override_get_db():
            db = test_session_factory()
            try:
                yield db
            finally:
                db.close()

        app.dependency_overrides[get_db] = override_get_db
        original_svc = main_module.chat_service
        main_module.chat_service = ChatService(spy_llm)

        try:
            with TestClient(app) as client:
                resp = client.post("/api/sessions")
                sid = resp.json()["id"]

                client.post(
                    f"/api/sessions/{sid}/messages",
                    json={"message": "My cool topic"},
                )

                db = test_session_factory()
                user_msgs = db.execute(
                    select(Message).where(
                        Message.session_id == sid,
                        Message.role == "user",
                    )
                ).scalars().all()
                assert len(user_msgs) == 1
                assert user_msgs[0].content == "My cool topic"

                session = db.get(ChatSession, sid)
                assert session.title == "My cool topic"
                db.close()
        finally:
            app.dependency_overrides.pop(get_db, None)
            main_module.chat_service = original_svc

    def test_phase1_commit_failure_nothing_persisted(self, monkeypatch):
        """Phase 1 commit fails → no user msg, no title change."""
        import asyncio as aio

        from backend.database import create_database_engine, create_tables

        eng = create_database_engine("sqlite:///:memory:")
        create_tables(bind=eng)
        SessionLocal = sessionmaker(
            bind=eng, autoflush=False, expire_on_commit=False
        )
        db = SessionLocal()

        chat_session = ChatSession(
            llm_model_snapshot="injected-test-model",
        )
        db.add(chat_session)
        db.commit()
        db.refresh(chat_session)
        sid = chat_session.id  # save before session detaches

        spy = SpyLLMClient(response="reply")

        original_commit = db.commit
        commit_count = 0

        def _failing_commit():
            nonlocal commit_count
            commit_count += 1
            if commit_count == 1:  # Phase 1
                raise RuntimeError("simulated DB failure")
            return original_commit()

        monkeypatch.setattr(db, "commit", _failing_commit)

        service = ChatService(spy)
        with pytest.raises(RuntimeError, match="simulated DB failure"):
            aio.run(
                service.handle_session_message(
                    sid, "should not persist", db,
                )
            )

        db.close()
        db2 = SessionLocal()
        rows = db2.execute(
            select(Message).where(Message.session_id == sid)
        ).scalars().all()
        assert len(rows) == 0
        s = db2.get(ChatSession, sid)
        assert s.title == "New Chat"
        db2.close()
        eng.dispose()

    def test_auto_title_then_manual_rename_then_send(
        self, test_engine, test_session_factory, spy_llm,
    ):
        """Auto-title → manual rename → send does not overwrite."""
        import backend.main as main_module

        def override_get_db():
            db = test_session_factory()
            try:
                yield db
            finally:
                db.close()

        app.dependency_overrides[get_db] = override_get_db
        original_svc = main_module.chat_service
        main_module.chat_service = ChatService(spy_llm)

        try:
            with TestClient(app) as client:
                resp = client.post("/api/sessions")
                sid = resp.json()["id"]

                client.post(
                    f"/api/sessions/{sid}/messages",
                    json={"message": "Auto Title Text"},
                )
                s1 = client.get(f"/api/sessions/{sid}").json()
                assert s1["title"] == "Auto Title Text"

                client.patch(
                    f"/api/sessions/{sid}",
                    json={"title": "Manual Override"},
                )

                client.post(
                    f"/api/sessions/{sid}/messages",
                    json={"message": "Second message"},
                )
                s2 = client.get(f"/api/sessions/{sid}").json()
                assert s2["title"] == "Manual Override"
        finally:
            app.dependency_overrides.pop(get_db, None)
            main_module.chat_service = original_svc


# ============================================================================
# Message provenance snapshots via the API
# ============================================================================


class TestMessageProvenanceSnapshotsAPI:
    """The send API returns and persists accurate provenance triples."""

    def test_success_response_contains_snapshot_triple(
        self, client, spy_llm,
    ):
        """Both response messages carry
        (default, api, injected-test-model)."""
        sid = _create_session(client)

        resp = client.post(
            f"/api/sessions/{sid}/messages",
            json={"message": "hello"},
        )
        assert resp.status_code == 200
        body = resp.json()

        for key in ("user_message", "assistant_message"):
            msg = body[key]
            assert msg["llm_profile_id_snapshot"] == "default"
            assert msg["llm_profile_kind_snapshot"] == "api"
            assert msg["llm_model_snapshot"] == "injected-test-model"

    def test_database_rows_match_response_triple(
        self, client, spy_llm, test_session_factory,
    ):
        """The persisted rows carry the same triple as the response."""
        sid = _create_session(client)

        resp = client.post(
            f"/api/sessions/{sid}/messages",
            json={"message": "hello"},
        )
        assert resp.status_code == 200
        body = resp.json()

        db = test_session_factory()
        rows = (
            db.execute(
                select(Message)
                .where(Message.session_id == sid)
                .order_by(Message.id.asc())
            )
            .scalars()
            .all()
        )
        db.close()

        assert len(rows) == 2
        for row in rows:
            assert row.llm_profile_id_snapshot == "default"
            assert row.llm_profile_kind_snapshot == "api"
            assert row.llm_model_snapshot == "injected-test-model"

        assert (
            rows[0].llm_profile_id_snapshot,
            rows[0].llm_profile_kind_snapshot,
            rows[0].llm_model_snapshot,
        ) == (
            body["user_message"]["llm_profile_id_snapshot"],
            body["user_message"]["llm_profile_kind_snapshot"],
            body["user_message"]["llm_model_snapshot"],
        )
        assert (
            rows[1].llm_profile_id_snapshot,
            rows[1].llm_profile_kind_snapshot,
            rows[1].llm_model_snapshot,
        ) == (
            body["assistant_message"]["llm_profile_id_snapshot"],
            body["assistant_message"]["llm_profile_kind_snapshot"],
            body["assistant_message"]["llm_model_snapshot"],
        )

    def test_llm_error_502_keeps_user_with_triple(
        self, client, spy_llm, test_session_factory,
    ):
        """A 502 leaves only the user message, with an accurate triple."""
        sid = _create_session(client)
        spy_llm.error = LLMError("Upstream failure", status_code=502)

        resp = client.post(
            f"/api/sessions/{sid}/messages",
            json={"message": "hello"},
        )
        assert resp.status_code == 502

        db = test_session_factory()
        rows = (
            db.execute(
                select(Message)
                .where(Message.session_id == sid)
                .order_by(Message.id.asc())
            )
            .scalars()
            .all()
        )
        db.close()

        assert len(rows) == 1
        assert rows[0].role == "user"
        assert rows[0].llm_profile_id_snapshot == "default"
        assert rows[0].llm_profile_kind_snapshot == "api"
        assert rows[0].llm_model_snapshot == "injected-test-model"

    def test_llm_error_504_keeps_user_with_triple(
        self, client, spy_llm, test_session_factory,
    ):
        """A 504 leaves only the user message, with an accurate triple."""
        sid = _create_session(client)
        spy_llm.error = LLMError("Upstream API timed out", status_code=504)

        resp = client.post(
            f"/api/sessions/{sid}/messages",
            json={"message": "hello"},
        )
        assert resp.status_code == 504

        db = test_session_factory()
        rows = (
            db.execute(
                select(Message)
                .where(Message.session_id == sid)
            )
            .scalars()
            .all()
        )
        db.close()

        assert len(rows) == 1
        assert rows[0].role == "user"
        assert rows[0].llm_profile_id_snapshot == "default"
        assert rows[0].llm_profile_kind_snapshot == "api"
        assert rows[0].llm_model_snapshot == "injected-test-model"

    def test_blank_reply_keeps_user_with_triple(
        self, client, spy_llm, test_session_factory,
    ):
        """A blank reply leaves only the user message, with its triple."""
        sid = _create_session(client)
        spy_llm.response = "   \n\t  "

        resp = client.post(
            f"/api/sessions/{sid}/messages",
            json={"message": "hello"},
        )
        assert resp.status_code == 502

        db = test_session_factory()
        rows = (
            db.execute(
                select(Message)
                .where(Message.session_id == sid)
            )
            .scalars()
            .all()
        )
        db.close()

        assert len(rows) == 1
        assert rows[0].role == "user"
        assert rows[0].llm_profile_id_snapshot == "default"
        assert rows[0].llm_profile_kind_snapshot == "api"
        assert rows[0].llm_model_snapshot == "injected-test-model"


# ============================================================================
# Switch session profile via the API
# ============================================================================


class TestSwitchProfileAPI:
    """PATCH /api/sessions/{id}/llm-profile contract."""

    @pytest.fixture
    def switch_client(self, test_engine, test_session_factory):
        """TestClient wired to a two-profile registry (api default +
        local)."""
        import backend.main as main_module
        from backend.llm_profiles import LLMProfile, LLMProfileRegistry

        self.local_spy = SpyLLMClient(response="local reply")
        registry = LLMProfileRegistry([
            LLMProfile(
                id="default", label="API", kind="api",
                model="remote-m1",
                client=SpyLLMClient(response="api reply"),
                is_default=True,
            ),
            LLMProfile(
                id="local", label="Local", kind="local",
                model="qwen3.5:4b", client=self.local_spy,
                is_default=False,
            ),
        ])
        svc = ChatService(profiles=registry)

        def override_get_db():
            db = test_session_factory()
            try:
                yield db
            finally:
                db.close()

        app.dependency_overrides[get_db] = override_get_db
        original = main_module.chat_service
        main_module.chat_service = svc
        try:
            with TestClient(app) as c:
                yield c
        finally:
            app.dependency_overrides.pop(get_db, None)
            main_module.chat_service = original

    def _create_local_session(self, switch_client) -> int:
        resp = switch_client.post(
            "/api/sessions", json={"llm_profile_id": "local"},
        )
        assert resp.status_code == 201
        return resp.json()["id"]

    def test_switch_200_returns_full_session_response(self, switch_client):
        sid = self._create_local_session(switch_client)
        # No messages → no acknowledgement needed even for API target.
        resp = switch_client.patch(
            f"/api/sessions/{sid}/llm-profile",
            json={"llm_profile_id": "default",
                  "acknowledge_remote_history": False},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["id"] == sid
        assert body["llm_profile_id"] == "default"
        assert body["llm_profile_label"] == "API"
        assert body["llm_profile_status"] == "ready"
        assert body["llm_model_snapshot"] == "remote-m1"
        assert "title" in body

    def test_switch_404_session_not_found(self, switch_client):
        resp = switch_client.patch(
            "/api/sessions/9999/llm-profile",
            json={"llm_profile_id": "local",
                  "acknowledge_remote_history": False},
        )
        assert resp.status_code == 404
        assert resp.json() == {"detail": "Session not found"}

    def test_switch_422_unknown_profile(self, switch_client):
        sid = self._create_local_session(switch_client)
        resp = switch_client.patch(
            f"/api/sessions/{sid}/llm-profile",
            json={"llm_profile_id": "nope",
                  "acknowledge_remote_history": False},
        )
        assert resp.status_code == 422

    @pytest.mark.parametrize("bad_id", ["bad/id", "Bad", "", "   "])
    def test_switch_422_invalid_profile_id(self, switch_client, bad_id):
        sid = self._create_local_session(switch_client)
        resp = switch_client.patch(
            f"/api/sessions/{sid}/llm-profile",
            json={"llm_profile_id": bad_id,
                  "acknowledge_remote_history": False},
        )
        assert resp.status_code == 422

    def test_switch_422_non_string_profile_id(self, switch_client):
        sid = self._create_local_session(switch_client)
        resp = switch_client.patch(
            f"/api/sessions/{sid}/llm-profile",
            json={"llm_profile_id": 123,
                  "acknowledge_remote_history": False},
        )
        assert resp.status_code == 422

    @pytest.mark.parametrize("bad_ack", ["true", 1, 0, None])
    def test_switch_422_non_strict_ack(self, switch_client, bad_ack):
        sid = self._create_local_session(switch_client)
        resp = switch_client.patch(
            f"/api/sessions/{sid}/llm-profile",
            json={"llm_profile_id": "default",
                  "acknowledge_remote_history": bad_ack},
        )
        assert resp.status_code == 422

    def test_switch_409_structured_and_binding_unchanged(
        self, switch_client,
    ):
        sid = self._create_local_session(switch_client)
        # One message on the local profile → switching to the API
        # profile requires acknowledgement.
        send = switch_client.post(
            f"/api/sessions/{sid}/messages", json={"message": "hello"},
        )
        assert send.status_code == 200

        before = switch_client.get(f"/api/sessions/{sid}").json()

        resp = switch_client.patch(
            f"/api/sessions/{sid}/llm-profile",
            json={"llm_profile_id": "default",
                  "acknowledge_remote_history": False},
        )
        assert resp.status_code == 409
        body = resp.json()
        assert isinstance(body["detail"], dict)
        assert body["detail"]["code"] == "remote_history_ack_required"
        assert isinstance(body["detail"]["message"], str)
        assert "remote API" in body["detail"]["message"]

        after = switch_client.get(f"/api/sessions/{sid}").json()
        assert after["llm_profile_id"] == "local"
        assert after["llm_model_snapshot"] == "qwen3.5:4b"
        assert after["updated_at"] == before["updated_at"]

    def test_switch_409_with_ack_true_succeeds(self, switch_client):
        sid = self._create_local_session(switch_client)
        switch_client.post(
            f"/api/sessions/{sid}/messages", json={"message": "hello"},
        )

        resp = switch_client.patch(
            f"/api/sessions/{sid}/llm-profile",
            json={"llm_profile_id": "default",
                  "acknowledge_remote_history": True},
        )
        assert resp.status_code == 200
        assert resp.json()["llm_profile_id"] == "default"
        assert resp.json()["llm_profile_status"] == "ready"

    def test_switch_moves_session_to_top(self, switch_client,
                                         test_session_factory):
        from datetime import datetime as dt

        id_a = switch_client.post(
            "/api/sessions", json={"llm_profile_id": "local"},
        ).json()["id"]
        id_b = switch_client.post(
            "/api/sessions", json={"llm_profile_id": "local"},
        ).json()["id"]

        # Pin deterministic updated_at values: A older than B.
        db = test_session_factory()
        db.get(ChatSession, id_a).updated_at = dt(2020, 1, 1, 0, 0, 0)  # noqa: DTZ001
        db.get(ChatSession, id_b).updated_at = dt(2020, 1, 2, 0, 0, 0)  # noqa: DTZ001
        db.commit()
        db.close()

        # A real binding change (local → default; no messages, so no
        # acknowledgement needed) must bump updated_at and move the
        # session to the top of the list.
        resp = switch_client.patch(
            f"/api/sessions/{id_a}/llm-profile",
            json={"llm_profile_id": "default",
                  "acknowledge_remote_history": False},
        )
        assert resp.status_code == 200

        listed = switch_client.get("/api/sessions").json()
        assert listed[0]["id"] == id_a

    def test_switch_preserves_title(self, switch_client):
        sid = self._create_local_session(switch_client)
        rename = switch_client.patch(
            f"/api/sessions/{sid}", json={"title": "My Chat"},
        )
        assert rename.status_code == 200

        resp = switch_client.patch(
            f"/api/sessions/{sid}/llm-profile",
            json={"llm_profile_id": "default",
                  "acknowledge_remote_history": False},
        )
        assert resp.status_code == 200
        assert resp.json()["title"] == "My Chat"


class TestSwitchProfilePydantic422SkipsService:
    """Pydantic request validation rejects the body BEFORE the service
    layer is reached — switch_session_profile is never called."""

    @pytest.fixture
    def counting_client(self, test_engine, test_session_factory):
        import backend.main as main_module
        from backend.llm_profiles import LLMProfile, LLMProfileRegistry

        registry = LLMProfileRegistry([
            LLMProfile(
                id="default", label="API", kind="api",
                model="remote-m1", client=SpyLLMClient(),
                is_default=True,
            ),
        ])
        svc = ChatService(profiles=registry)
        self.service_calls = {"n": 0}
        original_switch = svc.switch_session_profile

        async def _counting_switch(*args, **kwargs):
            self.service_calls["n"] += 1
            return await original_switch(*args, **kwargs)

        svc.switch_session_profile = _counting_switch

        def override_get_db():
            db = test_session_factory()
            try:
                yield db
            finally:
                db.close()

        app.dependency_overrides[get_db] = override_get_db
        original = main_module.chat_service
        main_module.chat_service = svc
        try:
            with TestClient(app) as c:
                yield c
        finally:
            app.dependency_overrides.pop(get_db, None)
            main_module.chat_service = original

    def test_invalid_profile_id_does_not_call_service(self, counting_client):
        sid = counting_client.post("/api/sessions").json()["id"]

        resp = counting_client.patch(
            f"/api/sessions/{sid}/llm-profile",
            json={"llm_profile_id": "bad/id",
                  "acknowledge_remote_history": False},
        )
        assert resp.status_code == 422
        assert self.service_calls["n"] == 0

    def test_non_strict_ack_does_not_call_service(self, counting_client):
        sid = counting_client.post("/api/sessions").json()["id"]

        resp = counting_client.patch(
            f"/api/sessions/{sid}/llm-profile",
            json={"llm_profile_id": "default",
                  "acknowledge_remote_history": "true"},
        )
        assert resp.status_code == 422
        assert self.service_calls["n"] == 0
