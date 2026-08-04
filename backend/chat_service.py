"""Chat service — orchestrates validation, LLM calls, and replies."""

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

    # -- stateless ---------------------------------------------------------

    async def handle_message(self, message: str) -> str:
        """Return the assistant reply for *message*."""
        messages: list[LLMMessage] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": message},
        ]
        return await self._llm.generate(messages)

    # -- persistent --------------------------------------------------------

    async def handle_session_message(
        self,
        chat_session: ChatSession,
        content: str,
        db: Session,
    ) -> tuple[Message, Message]:
        """Process a user message within a persistent chat session.

        Parameters
        ----------
        chat_session:
            An already-persisted ``ChatSession`` ORM object.  The caller
            is responsible for looking it up.
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
        LLMInvalidResponseError
            When the LLM replies with an empty or whitespace-only string.
        """
        # -- Phase 1: save user message ------------------------------------
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

        # -- Phase 2: history + LLM + save assistant -----------------------
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
            history_rows.reverse()  # deliver in chronological (ASC) order

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
