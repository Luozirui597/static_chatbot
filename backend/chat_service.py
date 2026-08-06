"""Chat service — orchestrates validation, LLM calls, and replies."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.exceptions import LLMInvalidResponseError
from backend.llm_client import LLMClient, LLMMessage
from backend.models import ChatSession, Message, utc_now
from backend.system_prompt import SYSTEM_PROMPT

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_HISTORY_MESSAGES = 20
"""Maximum number of *previous* messages sent to the LLM as context.

The current user message and the system prompt are NOT counted toward
this limit.  The database always stores every message regardless of
this window.
"""


# ---------------------------------------------------------------------------
# Per-session lock registry
# ---------------------------------------------------------------------------


@dataclass
class _LockEntry:
    """A per-session asyncio lock with a reference count.

    *ref_count* tracks how many requests are currently holding or waiting
    for this lock.  The entry is only safe to delete when *ref_count*
    reaches zero.
    """

    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    ref_count: int = 0


class SessionLockRegistry:
    """Per-session ``asyncio.Lock`` registry with reference counting.

    Provides an async context manager :meth:`session_lock` that:

    1. Atomically increments *ref_count* (creating the entry if needed).
    2. Awaits the per-session ``asyncio.Lock`` inside a ``try/finally``
       so cancellation during the wait still decrements *ref_count*.
    3. Releases the lock on exit (only if it was actually acquired).
    4. Decrements *ref_count* and removes the entry when it reaches zero.

    The internal guard is an ``asyncio.Lock`` to avoid blocking the
    event loop.
    """

    def __init__(self) -> None:
        self._entries: dict[int, _LockEntry] = {}
        self._guard = asyncio.Lock()

    @asynccontextmanager
    async def session_lock(self, session_id: int):
        """Acquire the per-session lock and manage the reference count.

        Yields control inside the critical section.  The lock is released
        and the reference count is decremented on exit — regardless of
        whether the block completes normally, raises an exception, or is
        cancelled.

        Usage::

            async with registry.session_lock(session_id):
                ...  # at most one coroutine per session at a time
        """
        # -- atomically bump ref_count under the internal guard -----------
        async with self._guard:
            entry = self._entries.get(session_id)
            if entry is None:
                entry = _LockEntry()
                self._entries[session_id] = entry
            entry.ref_count += 1
            lock = entry.lock

        # -- wait for the per-session lock (cancellation-safe) ------------
        acquired = False
        try:
            await lock.acquire()
            acquired = True
            try:
                yield
            finally:
                if acquired:
                    lock.release()
        finally:
            # -- decrement ref_count and maybe delete the entry -----------
            async with self._guard:
                cur = self._entries.get(session_id)
                if cur is not None:
                    cur.ref_count -= 1
                    if cur.ref_count <= 0 and self._entries.get(session_id) is cur:
                        del self._entries[session_id]


# ---------------------------------------------------------------------------
# Service-level exceptions
# ---------------------------------------------------------------------------


class SessionNotFoundError(Exception):
    """Raised when a session id does not exist in the database."""

    def __init__(self, session_id: int) -> None:
        self.session_id = session_id
        super().__init__(f"Session {session_id} not found")


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class ChatService:
    """Receives a validated user message, calls the LLM client, and
    returns the reply.

    The LLM client is injected so it can be swapped without touching
    the service or route code.
    """

    def __init__(self, llm_client: LLMClient) -> None:
        self._llm = llm_client
        self._lock_registry = SessionLockRegistry()

    # -- stateless ---------------------------------------------------------

    async def handle_message(self, message: str) -> str:
        """Return the assistant reply for *message*.

        Raises :exc:`LLMInvalidResponseError` when the LLM returns an
        empty or whitespace-only reply (matching the validation in the
        session-based handler).
        """
        messages: list[LLMMessage] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": message},
        ]
        reply = await self._llm.generate(messages)

        if not reply or not reply.strip():
            raise LLMInvalidResponseError(
                detail="LLM returned an empty or blank response",
                status_code=502,
            )
        return reply

    # -- persistent --------------------------------------------------------

    async def handle_session_message(
        self,
        session_id: int,
        content: str,
        db: Session,
    ) -> tuple[Message, Message]:
        """Process a user message within a persistent chat session.

        Acquires a per-session lock and queries the ``ChatSession``
        inside it, so a concurrent delete cannot race with the lookup.

        Parameters
        ----------
        session_id:
            The id of the target chat session.  The ORM object is
            looked up inside the lock to avoid TOCTOU races.
        content:
            The user's message text.
        db:
            An active SQLAlchemy ``Session``.

        Returns
        -------
        tuple[Message, Message]
            ``(user_message, assistant_message)`` — both are persisted
            and ``db.refresh()`` has been called on each.

        Raises
        ------
        SessionNotFoundError
            When *session_id* does not exist in the database.
        LLMInvalidResponseError
            When the LLM replies with an empty or whitespace-only string.
        """
        async with self._lock_registry.session_lock(session_id):
            chat_session = db.get(ChatSession, session_id)
            if chat_session is None:
                raise SessionNotFoundError(session_id)

            # -- Phase 1: save user message --------------------------------
            user_message = Message(
                session_id=chat_session.id,
                role="user",
                content=content,
            )
            db.add(user_message)
            chat_session.updated_at = utc_now()

            try:
                db.commit()
                db.refresh(user_message)
            except Exception:
                db.rollback()
                raise

            # -- Phase 2: history + LLM + save assistant -------------------
            try:
                # Query recent history
                stmt = (
                    select(Message)
                    .where(
                        Message.session_id == chat_session.id,
                        Message.id < user_message.id,
                    )
                    .order_by(Message.id.desc())
                    .limit(MAX_HISTORY_MESSAGES)
                )
                history_rows = db.execute(stmt).scalars().all()
                history_rows.reverse()  # deliver in chronological order

                # Build LLM messages
                llm_messages: list[LLMMessage] = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                ]
                for msg in history_rows:
                    llm_messages.append({
                        "role": msg.role,  # type: ignore[typeddict-item]
                        "content": msg.content,
                    })
                llm_messages.append({
                    "role": "user",
                    "content": user_message.content,
                })

                # Call LLM
                reply = await self._llm.generate(llm_messages)

                # Validate reply — empty or blank is an error
                if not reply or not reply.strip():
                    raise LLMInvalidResponseError(
                        detail="LLM returned an empty or blank response",
                        status_code=502,
                    )

                # Save assistant message
                assistant_message = Message(
                    session_id=chat_session.id,
                    role="assistant",
                    content=reply,
                )
                db.add(assistant_message)
                chat_session.updated_at = utc_now()
                db.commit()
                db.refresh(assistant_message)

                return user_message, assistant_message
            except Exception:
                db.rollback()
                raise

    async def delete_session(self, session_id: int, db: Session) -> None:
        """Delete a session and all of its messages.

        Acquires the same per-session lock as
        :meth:`handle_session_message`, so a delete cannot race with an
        in-progress generation:

        * If the send holds the lock first → the send completes, then
          the delete runs (session is deleted).
        * If the delete holds the lock first → the send sees a 404.

        Parameters
        ----------
        session_id:
            The id of the chat session to delete.
        db:
            An active SQLAlchemy ``Session``.

        Raises
        ------
        SessionNotFoundError
            When *session_id* does not exist in the database.
        """
        async with self._lock_registry.session_lock(session_id):
            chat_session = db.get(ChatSession, session_id)
            if chat_session is None:
                raise SessionNotFoundError(session_id)
            db.delete(chat_session)
            db.commit()
