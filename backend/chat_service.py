"""Chat service — orchestrates validation, LLM calls, and replies."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Sequence
from datetime import UTC, datetime
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from backend.exceptions import (
    LLMError,
    LLMInvalidResponseError,
    SessionProfileConflictError,
    SessionProfileSwitchAckRequiredError,
    SessionProfileUnavailableError,
    UnknownLLMProfileError,
)
from backend.llm_client import LLMClient, LLMMessage
from backend.llm_profiles import (
    LLMProfile,
    LLMProfileRegistry,
    SessionProfileStatus,
)
from backend.history_boundary import capture_history_boundary
from backend.interaction_modes import (
    CORRECTIVE_MODE,
    RECEIVE_TEACHING_HISTORY_CONTEXT_VERSION,
    RECEIVE_TEACHING_MODE,
    get_prompt_version_for_mode,
    get_programmatic_system_prompt,
    get_system_prompt_for_mode,
    is_valid_interaction_mode,
    PROGRAMMATIC_HISTORY_PROMPT_VERSION,
)
from backend.models import ChatSession, Message, ModeSwitchEvent, utc_now
from backend.reply_language import (
    CLARIFY_PROMPT_VERSION,
    ReplyLanguageInputError,
    ReplyLanguageMismatchError,
    append_language_instruction,
    language_prompt_version,
    resolve_reply_language,
    validate_reply_language,
)
from backend.review_diagnostics import (
    LLMCallDiagnostics,
    clean_optional_float,
    clean_optional_int,
    clean_optional_string,
    error_category_value,
)
from backend.session_titles import derive_auto_title
from backend.system_prompt import SYSTEM_PROMPT
from backend.programmatic_history.service import (
    ProgrammaticHistoryError,
    ProgrammaticHistoryService,
    ProgrammaticHistoryTimeout,
)
from backend.general_history.service import GeneralHistoryService
from backend.teaching_strategies import (
    get_teaching_prompt, get_teaching_prompt_version, is_valid_teaching_strategy,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_HISTORY_MESSAGES = 20
"""Maximum number of *previous* messages sent to the LLM as context.

The current user message and the system prompt are NOT counted toward
this limit.  The database always stores every message regardless of
this window.
"""


logger = logging.getLogger(__name__)

_CHAT_LLM_FAILURE_EVENT = "chat_llm_failure"

_RECEIVE_TEACHING_HISTORY_TYPE = "receive_teaching_history_background"
_RECEIVE_TEACHING_HISTORY_NOTE = (
    "以下 messages 是老师先前 user 消息的原文，按时间顺序保留，"
    "用于已教内容、更正和约束；它们不是本轮请求，不要逐条重答。"
    "当前请求在随后的独立 user 消息中。"
)


def _log_chat_llm_failure(
    *,
    session_id: int,
    user_message_id: int,
    error: LLMError,
) -> None:
    """Log one bounded, sanitized chat LLM failure event.

    This function never raises: diagnostic logging must not hide or
    replace the original LLMError, which still drives rollback and the
    HTTP response mapping.
    """
    try:
        diagnostics = getattr(error, "diagnostics", None)
        if isinstance(diagnostics, LLMCallDiagnostics) is False:
            diagnostics = None

        payload = {
            "event": _CHAT_LLM_FAILURE_EVENT,
            "timestamp_utc": datetime.now(UTC).isoformat(
                timespec="milliseconds"
            ),
            "session_id": clean_optional_int(session_id, minimum=1),
            "user_message_id": clean_optional_int(
                user_message_id, minimum=1
            ),
            "local_http_status": clean_optional_int(
                getattr(error, "status_code", None),
                minimum=100,
                maximum=599,
            ),
            "error_category": (
                error_category_value(diagnostics.error_category)
                if diagnostics is not None
                else None
            ),
            "exception_type": (
                clean_optional_string(
                    diagnostics.exception_type,
                    max_length=64,
                )
                if diagnostics is not None
                else None
            ),
            "timeout_configured_seconds": (
                clean_optional_float(
                    diagnostics.timeout_configured_seconds,
                    minimum=0.0,
                )
                if diagnostics is not None
                else None
            ),
            "call_elapsed_ms": (
                clean_optional_int(
                    diagnostics.call_elapsed_ms,
                    minimum=0,
                )
                if diagnostics is not None
                else None
            ),
            "upstream_http_status": (
                clean_optional_int(
                    diagnostics.http_status,
                    minimum=100,
                    maximum=599,
                )
                if diagnostics is not None
                else None
            ),
            "upstream_request_id": (
                clean_optional_string(
                    diagnostics.upstream_request_id,
                    max_length=128,
                )
                if diagnostics is not None
                else None
            ),
        }
        logger.warning(
            "%s %s",
            _CHAT_LLM_FAILURE_EVENT,
            json.dumps(
                payload,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            ),
        )
    except Exception:
        return


def _project_history_for_mode(
    history_rows: Sequence[Message],
    *,
    interaction_mode: str,
) -> list[Message]:
    """Project stored history onto the context allowed for a mode.

    Receive teaching intentionally excludes every historical assistant
    message, regardless of its mode or prompt-version snapshot.  Only
    historical user messages are sent to the model.  Corrective mode
    receives the original history unchanged, in the same order.

    The returned list is new; the input sequence and the ORM objects
    are never mutated.
    """
    if interaction_mode == RECEIVE_TEACHING_MODE:
        return [
            message for message in history_rows if message.role == "user"
        ]
    if interaction_mode == CORRECTIVE_MODE:
        return list(history_rows)
    raise ValueError("Unknown interaction mode")


def _serialize_receive_teaching_history(
    history_rows: Sequence[Message],
) -> str:
    """Serialize projected Receive history as one JSON background block.

    The rows are read only.  Their order and content are preserved exactly
    through JSON encoding; no teaching content is extracted, filtered,
    summarized, merged, corrected, or marked as completed.
    """
    payload = {
        "type": _RECEIVE_TEACHING_HISTORY_TYPE,
        "version": RECEIVE_TEACHING_HISTORY_CONTEXT_VERSION,
        "note": _RECEIVE_TEACHING_HISTORY_NOTE,
        "messages": [
            {
                "index": index,
                "content": message.content,
            }
            for index, message in enumerate(history_rows, start=1)
        ],
    }
    return json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
    )


async def _generate_reply_for_mode(
    *,
    llm_client: LLMClient,
    messages: list[LLMMessage],
    interaction_mode: str,
    profile_kind: str,
) -> str:
    """Call the model, adding low-randomness only for local Receive.

    The temperature option is never persisted on the client and never
    affects non-target calls.
    """
    if (
        interaction_mode == RECEIVE_TEACHING_MODE
        and profile_kind == "local"
    ):
        return await llm_client.generate(
            messages,
            temperature=0.0,
        )
    return await llm_client.generate(messages)


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


def _log_programmatic_failure(
    *,
    session_id: int,
    user_message_id: int,
    stage: str,
    error: BaseException | None = None,
    retrieval_count: int = 0,
    latency_ms: int | None = None,
    source_ids: list[int] | None = None,
    trigger_reason: str | None = None,
    visible_complete: bool | None = None,
) -> None:
    try:
        logger.warning(
            "programmatic_history_failure session_id=%s user_message_id=%s "
            "stage=%s error_class=%s retrieval_count=%s latency_ms=%s "
            "source_ids=%s trigger_reason=%s visible_complete=%s",
            session_id,
            user_message_id,
            stage,
            type(error).__name__ if error is not None else "none",
            retrieval_count,
            latency_ms,
            source_ids or [],
            trigger_reason or "unknown",
            visible_complete,
        )
    except Exception:
        # Observability must never replace the original failure.
        pass


@dataclass(frozen=True)
class FrozenReceiveTeachingPolicy:
    """Exact prompt contract for an explicitly isolated teaching experiment.

    Language behavior belongs to the frozen prompt and is scored by the
    experiment. Production services omit this policy and keep the backend
    language instructions, versions, and pre-write validation.
    """

    system_prompt: str
    prompt_version: str

    def __post_init__(self) -> None:
        if not isinstance(self.system_prompt, str) or not self.system_prompt.strip():
            raise ValueError("Frozen teaching policy requires a system prompt")
        if (
            not isinstance(self.prompt_version, str)
            or not self.prompt_version.strip()
            or len(self.prompt_version) > 50
        ):
            raise ValueError("Frozen teaching policy requires a valid prompt version")


class ChatService:
    """Receives a validated user message, calls the LLM client, and
    returns the reply.

    Construct with **either** a single ``llm_client`` (for stateless
    usage or tests) **or** a ``profiles`` registry (for session-aware
    production usage).  Passing both or neither is an error.
    """

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        *,
        profiles: LLMProfileRegistry | None = None,
        lock_registry: SessionLockRegistry | None = None,
        programmatic_history: ProgrammaticHistoryService | None = None,
        general_history: GeneralHistoryService | None = None,
        frozen_receive_teaching_policy: FrozenReceiveTeachingPolicy | None = None,
    ) -> None:
        if llm_client is not None and profiles is not None:
            raise ValueError(
                "Pass either llm_client or profiles, not both"
            )
        if llm_client is None and profiles is None:
            raise ValueError(
                "Either llm_client or profiles must be provided"
            )
        if frozen_receive_teaching_policy is not None:
            if not isinstance(frozen_receive_teaching_policy, FrozenReceiveTeachingPolicy):
                raise TypeError("Expected FrozenReceiveTeachingPolicy")
            if programmatic_history is not None or general_history is not None:
                raise ValueError("Frozen teaching experiments cannot enable history tools")

        if llm_client is not None:
            self._profiles = LLMProfileRegistry.from_single_client(llm_client)
        else:
            # profiles is not None (guaranteed by the checks above)
            assert profiles is not None  # mypy narrowing
            self._profiles = profiles

        self._lock_registry = lock_registry or SessionLockRegistry()
        self._programmatic_history = programmatic_history
        self._general_history = general_history
        self._frozen_receive_teaching_policy = frozen_receive_teaching_policy

    # -- helpers ------------------------------------------------------------

    def build_session_response(self, session: ChatSession) -> dict:
        """Return a dict with all public session fields plus profile info.

        Used by every route that returns session data — no ORM attribute
        monkey-patching and no access to ``_profiles`` from outside.
        """
        profile = self._profiles.get(session.llm_profile_id)
        resolution = self._profiles.resolve(
            session.llm_profile_id, session.llm_model_snapshot,
        )
        return {
            "id": session.id,
            "title": session.title,
            "interaction_mode": session.interaction_mode,
            "teaching_strategy": session.teaching_strategy,
            "teaching_strategy_switchable": (
                self._frozen_receive_teaching_policy is None
                and session.interaction_mode == RECEIVE_TEACHING_MODE
            ),
            "created_at": session.created_at,
            "updated_at": session.updated_at,
            "llm_profile_id": session.llm_profile_id,
            "llm_profile_label": (
                profile.label if profile else session.llm_profile_id
            ),
            "llm_profile_status": resolution.status,
            "llm_model_snapshot": session.llm_model_snapshot,
        }

    def list_profiles_public(self) -> list[dict]:
        """Return public-safe profile summaries for every profile.

        Never exposes API keys, base URLs, reasoning effort, or clients.
        """
        return [
            {
                "id": p.id,
                "label": p.label,
                "kind": p.kind,
                "model": p.model,
                "is_default": p.is_default,
            }
            for p in self._profiles.list_all()
        ]

    # -- stateless ---------------------------------------------------------

    async def handle_message(self, message: str) -> str:
        """Return the assistant reply for *message*.

        Uses the registry's default profile client.

        Raises :exc:`LLMInvalidResponseError` when the LLM returns an
        empty or whitespace-only reply (matching the validation in the
        session-based handler).
        """
        messages: list[LLMMessage] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": message},
        ]
        client = self._profiles.default.client
        reply = await client.generate(messages)

        if not reply or not reply.strip():
            raise LLMInvalidResponseError(
                detail="LLM returned an empty or blank response",
                status_code=502,
            )
        return reply

    # -- persistent --------------------------------------------------------

    def create_session(
        self, profile_id: str, db: Session,
    ) -> ChatSession:
        """Create a new chat session with a validated profile.

        Looks up *profile_id* in the registry, writes the id and the
        current model as a snapshot, and returns the persisted ORM object.

        Raises :exc:`UnknownLLMProfileError` when *profile_id* is not
        in the registry.
        """
        profile = self._profiles.get(profile_id)
        if profile is None:
            raise UnknownLLMProfileError(profile_id)

        session = ChatSession(
            llm_profile_id=profile.id,
            llm_model_snapshot=profile.model,
        )
        db.add(session)
        try:
            db.commit()
            db.refresh(session)
        except Exception:
            db.rollback()
            raise
        return session

    def resolve_session_profile(
        self, chat_session: ChatSession,
    ) -> LLMProfile:
        """Resolve the session's profile compatibility and return the
        validated, immutable profile.

        This method receives a persisted ``ChatSession`` ORM object (not a
        profile id string).  It checks the stored ``llm_profile_id`` and
        ``llm_model_snapshot`` against the registry.

        Returns the full ``LLMProfile`` when the status is *ready*.  The
        profile is a frozen dataclass carrying ``id``, ``kind``,
        ``model`` and ``client`` — callers use ``profile.client`` to
        send and the profile fields as the message provenance
        snapshot.

        Raises
        ------
        SessionProfileUnavailableError
            The session's profile no longer exists in the registry.
        SessionProfileConflictError
            The profile's model has changed (``model_changed``) or the
            session predates model tracking (``legacy_unknown``).
        """
        resolution = self._profiles.resolve(
            chat_session.llm_profile_id,
            chat_session.llm_model_snapshot,
        )

        if resolution.status == SessionProfileStatus.PROFILE_UNAVAILABLE:
            raise SessionProfileUnavailableError(chat_session.llm_profile_id)

        if resolution.status in (
            SessionProfileStatus.MODEL_CHANGED,
            SessionProfileStatus.LEGACY_UNKNOWN,
        ):
            raise SessionProfileConflictError(
                status=resolution.status.value,
                snapshot=chat_session.llm_model_snapshot,
            )

        # status is READY — profile is guaranteed non-None
        assert resolution.profile is not None
        return resolution.profile

    async def handle_session_message(
        self,
        session_id: int,
        content: str,
        db: Session,
    ) -> tuple[Message, Message]:
        """Process a user message within a persistent chat session.

        Acquires a per-session lock and queries the ``ChatSession``
        inside it, so a concurrent delete cannot race with the lookup.

        **Compatibility check** runs before any data is saved:

        * ``profile_unavailable`` → raises, user message NOT saved.
        * ``model_changed`` → raises, user message NOT saved.
        * ``legacy_unknown`` → raises, user message NOT saved.

        Only ``ready`` sessions proceed to save the user message and
        call the LLM.

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
        SessionProfileUnavailableError
            When the session's profile is no longer in the registry.
        SessionProfileConflictError
            When the model has changed or the session predates tracking.
        LLMInvalidResponseError
            When the LLM replies with an empty or whitespace-only string.
        """
        async with self._lock_registry.session_lock(session_id):
            chat_session = db.get(ChatSession, session_id)
            if chat_session is None:
                raise SessionNotFoundError(session_id)

            # -- Resolve profile compatibility ---------------------------
            # Raises before any data is saved if not ready.  The frozen
            # profile is captured ONCE — both message snapshots below
            # come from this same object, never re-read from the
            # registry or the session after the LLM call completes.
            profile = self.resolve_session_profile(chat_session)
            llm_client = profile.client

            # -- Capture this turn's interaction policy -------------------
            # Both messages below share this mode and prompt version, and
            # both also share the already-resolved model snapshot triple.
            interaction_mode = chat_session.interaction_mode
            frozen_policy = (
                self._frozen_receive_teaching_policy
                if interaction_mode == RECEIVE_TEACHING_MODE
                else None
            )
            effective_teaching_strategy = (
                chat_session.teaching_strategy
                if interaction_mode == RECEIVE_TEACHING_MODE and frozen_policy is None
                else "baseline"
            )
            teaching_strategy_snapshot = (
                effective_teaching_strategy
                if interaction_mode == RECEIVE_TEACHING_MODE and frozen_policy is None
                else None
            )
            if frozen_policy is not None:
                system_prompt = frozen_policy.system_prompt
                prompt_version = frozen_policy.prompt_version
            elif interaction_mode == RECEIVE_TEACHING_MODE and effective_teaching_strategy != "baseline":
                system_prompt = get_teaching_prompt(effective_teaching_strategy)
                prompt_version = get_teaching_prompt_version(effective_teaching_strategy)
            else:
                system_prompt = get_system_prompt_for_mode(interaction_mode)
                prompt_version = get_prompt_version_for_mode(interaction_mode)
            receive_system_prompt = system_prompt
            history_prompt_version = get_teaching_prompt_version(effective_teaching_strategy, "history")
            clarify_prompt_version = get_teaching_prompt_version(effective_teaching_strategy, "clarify")

            def receive_history_prompt():
                return (
                    get_programmatic_system_prompt()
                    if effective_teaching_strategy == "baseline"
                    else get_teaching_prompt(effective_teaching_strategy, history=True)
                )

            # -- Resolve the reply language before saving anything --------
            # The language window is the same recent stored-message window
            # used by ordinary Receive teaching.  The current message is not
            # in the DB yet, so it is excluded naturally.
            language_plan = None
            language_target = None
            language_history_rows: list[Message] = []
            if interaction_mode == RECEIVE_TEACHING_MODE and frozen_policy is None:
                language_history_rows = list(
                    db.execute(
                        select(Message)
                        .where(Message.session_id == chat_session.id)
                        .order_by(Message.id.desc())
                        .limit(MAX_HISTORY_MESSAGES)
                    ).scalars().all()
                )
                language_plan = resolve_reply_language(
                    content,
                    tuple(
                        m.content
                        for m in reversed(language_history_rows)
                        if m.role == "user"
                    ),
                )
                language_target = language_plan.target
                system_prompt = append_language_instruction(
                    system_prompt, language_target
                )
                prompt_version = language_prompt_version(
                    prompt_version, language_target
                )

            # -- Determine whether this is the first user message --------
            existing_user_msg_count = db.execute(
                select(func.count()).select_from(Message).where(
                    Message.session_id == chat_session.id,
                    Message.role == "user",
                )
            ).scalar()
            is_first_user_message = (existing_user_msg_count == 0)

            # -- Phase 1: save user message + optional auto-title --------
            # Both are committed in a single transaction.
            user_message = Message(
                session_id=chat_session.id,
                role="user",
                content=content,
                llm_profile_id_snapshot=profile.id,
                llm_profile_kind_snapshot=profile.kind,
                llm_model_snapshot=profile.model,
                interaction_mode_snapshot=interaction_mode,
                teaching_strategy_snapshot=teaching_strategy_snapshot,
                prompt_version_snapshot=prompt_version,
            )
            db.add(user_message)
            chat_session.updated_at = utc_now()

            if is_first_user_message and not chat_session.title_is_manual:
                chat_session.title = derive_auto_title(content, max_chars=40)

            try:
                db.commit()
                db.refresh(user_message)
                saved_user_message_id = user_message.id
            except Exception:
                db.rollback()
                raise

            # -- Phase 2: optional programmatic history + generation -----
            programmatic_messages: list[LLMMessage] | None = None
            programmatic_retrieval_count = 0
            try:
                # Optional production-only programmatic retrieval.  The user
                # message above is already committed; failures in this phase
                # must never claim that the user message was rolled back.
                programmatic_result = None
                applicability = None
                if self._programmatic_history is not None:
                    applicability = self._programmatic_history.applicability_reason(
                        db, profile=profile, interaction_mode=interaction_mode
                    )
                fixed_plan = None
                fixed_relation = False
                general_applicability = None
                if self._general_history is not None:
                    general_applicability = (
                        self._general_history.applicability_reason(
                            db,
                            profile=profile,
                            interaction_mode=interaction_mode,
                        )
                    )
                if applicability == "enabled":
                    try:
                        fixed_plan = self._programmatic_history.prepare(
                            db,
                            session_id=session_id,
                            current_message_id=saved_user_message_id,
                            content=user_message.content,
                        )
                    except ProgrammaticHistoryError as exc:
                        _log_programmatic_failure(
                            session_id=session_id,
                            user_message_id=saved_user_message_id,
                            stage="decision",
                            error=exc,
                        )
                        raise LLMError(
                            detail="History lookup failed before generation",
                            status_code=500,
                        ) from exc
                    fixed_relation = fixed_plan.decision.question is not None

                general_takes_over = (
                    general_applicability == "enabled" and not fixed_relation
                )
                general_result = None
                general_answer = None
                if general_takes_over:
                    def attach_general_snapshot(*, hidden, clarify):
                        if hidden is None and not clarify:
                            return
                        saved_user = db.get(Message, saved_user_message_id)
                        if saved_user is None:
                            raise RuntimeError("saved user message missing")
                        snapshot_base = (
                            history_prompt_version
                            if hidden is not None
                            else clarify_prompt_version
                        )
                        saved_user.prompt_version_snapshot = (
                            language_prompt_version(snapshot_base, language_target)
                            if language_target is not None
                            else snapshot_base
                        )
                        try:
                            db.flush()
                        except Exception as exc:
                            rollback_succeeded = False
                            try:
                                db.rollback()
                                rollback_succeeded = True
                            except Exception:
                                pass
                            try:
                                logger.warning(
                                    "general_snapshot_failure session_id=%s "
                                    "user_message_id=%s stage=flush "
                                    "rollback_succeeded=%s error_class=%s",
                                    session_id,
                                    saved_user_message_id,
                                    rollback_succeeded,
                                    type(exc).__name__,
                                )
                            except Exception:
                                pass
                            raise
                        try:
                            db.commit()
                        except Exception as exc:
                            # COMMIT may have reached SQLite before the
                            # exception.  Do not claim a rollback and do not
                            # compensate; the caller must stop generation.
                            try:
                                logger.warning(
                                    "general_snapshot_failure session_id=%s "
                                    "user_message_id=%s stage=commit_unknown "
                                    "error_class=%s",
                                    session_id,
                                    saved_user_message_id,
                                    type(exc).__name__,
                                )
                            except Exception:
                                pass
                            raise

                    try:
                        general_result = await self._general_history.run(
                            db,
                            session_id=session_id,
                            current_message_id=saved_user_message_id,
                            profile=profile,
                            before_final=attach_general_snapshot,
                            reply_language=language_target,
                            **({"teaching_strategy": effective_teaching_strategy}
                               if effective_teaching_strategy != "baseline" else {}),
                        )
                    except Exception as exc:
                        try:
                            logger.warning(
                                "general_history failed session_id=%s user_message_id=%s stage=run error=%s",
                                session_id,
                                saved_user_message_id,
                                type(exc).__name__,
                            )
                        except Exception:
                            pass
                        raise LLMError(
                            detail="History lookup failed",
                            status_code=502,
                        ) from exc
                    general_status = general_result.status
                    try:
                        logger.info(
                            "general_history completed session_id=%s user_message_id=%s status=%s "
                            "counts=%s source_ids=%s scan_truncated=%s truncated=%s "
                            "corrections_truncated=%s pair_missing=%s",
                            session_id,
                            saved_user_message_id,
                            general_status,
                            general_result.counts.as_dict(),
                            general_result.retrieved_source_ids,
                            bool(getattr(general_result.search, "scan_truncated", False)),
                            bool(getattr(general_result.search, "truncated", False)),
                            bool(getattr(general_result.search, "corrections_truncated", False)),
                            bool(getattr(general_result.search, "pair_missing", False)),
                        )
                    except Exception:
                        pass
                    if general_result.answer is not None:
                        programmatic_messages = general_result.messages
                        general_answer = general_result.answer
                        programmatic_retrieval_count = general_result.counts.retrieval_attempts
                        if general_status == "completed_clarify":
                            system_prompt = receive_system_prompt
                            if language_target is not None:
                                system_prompt = append_language_instruction(
                                    system_prompt, language_target
                                )
                                prompt_version = language_prompt_version(
                                    clarify_prompt_version, language_target
                                )
                            else:
                                prompt_version = clarify_prompt_version
                        else:
                            system_prompt = receive_history_prompt()
                            if language_target is not None:
                                system_prompt = append_language_instruction(
                                    system_prompt, language_target
                                )
                                prompt_version = language_prompt_version(
                                    history_prompt_version,
                                    language_target,
                                )
                            else:
                                prompt_version = history_prompt_version
                    elif general_status not in {
                        "completed_direct",
                        "completed_visible_complete",
                    }:
                        if general_status in {
                            "retrieval_timeout",
                            "final_timeout",
                            "deadline_exceeded",
                            "decision_upstream_timeout",
                        }:
                            status_code = 504
                        elif general_status in {
                            "decision_invalid",
                            "decision_failed",
                            "retrieval_error",
                            "final_failed",
                        }:
                            status_code = 502
                        else:
                            status_code = 500
                        try:
                            logger.warning(
                                "general_history_failure session_id=%s user_message_id=%s "
                                "status=%s error_code=%s decision_reason=%s "
                                "decision_has_query=%s decision_visible_id_count=%s "
                                "decision_prompt_sha256=%s local_http_status=%s "
                                "counts=%s transport_events=%s",
                                session_id,
                                saved_user_message_id,
                                general_status,
                                general_result.error_code,
                                general_result.decision_reason,
                                general_result.decision_has_query,
                                general_result.decision_visible_id_count,
                                general_result.decision_prompt_sha256,
                                status_code,
                                general_result.counts.as_dict(),
                                general_result.transport_events,
                            )
                        except Exception:
                            pass
                        raise LLMError(
                            detail="History lookup failed",
                            status_code=status_code,
                        )

                if applicability == "enabled" and not general_takes_over:
                    try:
                        plan = fixed_plan if fixed_plan is not None else self._programmatic_history.prepare(
                            db,
                            session_id=session_id,
                            current_message_id=saved_user_message_id,
                            content=user_message.content,
                        )
                    except ProgrammaticHistoryError as exc:
                        _log_programmatic_failure(
                            session_id=session_id,
                            user_message_id=saved_user_message_id,
                            stage="decision",
                            error=exc,
                        )
                        raise LLMError(
                            detail="History lookup failed before generation",
                            status_code=500,
                        ) from exc
                    try:
                        _log_programmatic_failure(
                            session_id=session_id,
                            user_message_id=saved_user_message_id,
                            stage="decision",
                            retrieval_count=0,
                            trigger_reason=plan.decision.reason,
                            visible_complete=(plan.decision.reason == "visible_complete"),
                        )
                    except Exception:
                        pass
                    if plan.decision.action == "retrieve":
                        programmatic_retrieval_count = 1
                        try:
                            programmatic_result = await self._programmatic_history.retrieve_async(
                                db,
                                session_id=session_id,
                                current_message_id=saved_user_message_id,
                                plan=plan,
                            )
                        except ProgrammaticHistoryTimeout as exc:
                            _log_programmatic_failure(
                                session_id=session_id,
                                user_message_id=saved_user_message_id,
                                stage="retrieval_timeout",
                                error=exc,
                                retrieval_count=1,
                            )
                            raise LLMError(
                                detail="History lookup timed out",
                                status_code=504,
                            ) from exc
                        except ProgrammaticHistoryError as exc:
                            _log_programmatic_failure(
                                session_id=session_id,
                                user_message_id=saved_user_message_id,
                                stage="retrieval",
                                error=exc,
                                retrieval_count=1,
                            )
                            raise LLMError(
                                detail="History lookup failed",
                                status_code=502,
                            ) from exc
                        if programmatic_result is None or programmatic_result.is_error:
                            status = getattr(programmatic_result, "content", {})
                            stage = str(status.get("status", "retrieval_error")) if isinstance(status, dict) else "retrieval_error"
                            _log_programmatic_failure(
                                session_id=session_id,
                                user_message_id=saved_user_message_id,
                                stage="retrieval",
                                retrieval_count=1,
                            )
                            raise LLMError(
                                detail="History lookup failed",
                                status_code=(
                                    504 if stage == "timeout" else 502
                                ),
                            )
                        try:
                            result_content = getattr(programmatic_result, "content", {})
                            source_ids = [
                                int(item["message_id"])
                                for item in result_content.get("results", [])
                                if isinstance(item, dict) and isinstance(item.get("message_id"), int)
                            ]
                            _log_programmatic_failure(
                                session_id=session_id,
                                user_message_id=saved_user_message_id,
                                stage="retrieval_completed",
                                retrieval_count=1,
                                source_ids=source_ids,
                                trigger_reason=plan.decision.reason,
                                visible_complete=False,
                            )
                        except Exception:
                            pass
                        # The user message is already saved.  Commit the
                        # effective Prompt version in a separate transaction;
                        # only a confirmed commit permits model generation.
                        try:
                            saved_user = db.get(Message, saved_user_message_id)
                            if saved_user is None:
                                raise RuntimeError("saved user message missing")
                            saved_user.prompt_version_snapshot = (
                                language_prompt_version(
                                    history_prompt_version,
                                    language_target,
                                )
                                if language_target is not None
                                else history_prompt_version
                            )
                            db.flush()
                        except Exception as exc:
                            rollback_failed = False
                            try:
                                db.rollback()
                            except Exception:
                                rollback_failed = True
                            stage = (
                                "snapshot_update_rollback_failed"
                                if rollback_failed
                                else "snapshot_update_rolled_back"
                            )
                            try:
                                _log_programmatic_failure(
                                    session_id=session_id,
                                    user_message_id=saved_user_message_id,
                                    stage=stage,
                                    error=exc,
                                    retrieval_count=1,
                                )
                            except Exception:
                                pass
                            raise LLMError(
                                detail="History evidence could not be attached",
                                status_code=500,
                            ) from exc
                        try:
                            db.commit()
                        except Exception as exc:
                            try:
                                db.rollback()
                            except Exception:
                                pass
                            _log_programmatic_failure(
                                session_id=session_id,
                                user_message_id=saved_user_message_id,
                                stage="snapshot_commit_unknown",
                                error=exc,
                                retrieval_count=1,
                            )
                            raise LLMError(
                                detail="History lookup result is uncertain",
                                status_code=500,
                            ) from exc
                        system_prompt = receive_history_prompt()
                        if language_target is not None:
                            system_prompt = append_language_instruction(
                                system_prompt, language_target
                            )
                            prompt_version = language_prompt_version(
                                history_prompt_version,
                                language_target,
                            )
                        else:
                            prompt_version = history_prompt_version
                        programmatic_messages = self._programmatic_history.messages(
                            db,
                            session_id=session_id,
                            current_message_id=saved_user_message_id,
                            plan=plan,
                            result=programmatic_result,
                        )
                        if language_target is not None and programmatic_messages:
                            programmatic_messages[0] = {
                                "role": "system",
                                "content": system_prompt,
                            }

                if applicability != "enabled" and applicability is not None:
                    try:
                        _log_programmatic_failure(
                            session_id=session_id,
                            user_message_id=saved_user_message_id,
                            stage="not_applicable",
                            retrieval_count=0,
                            trigger_reason=applicability,
                            visible_complete=False,
                        )
                    except Exception:
                        pass
                if programmatic_messages is None:
                    # Ordinary path: reuse the pre-save window when available.
                    if language_target is not None:
                        history_rows = list(language_history_rows)
                    else:
                        stmt = (
                            select(Message)
                            .where(
                                Message.session_id == chat_session.id,
                                Message.id < user_message.id,
                            )
                            .order_by(Message.id.desc())
                            .limit(MAX_HISTORY_MESSAGES)
                        )
                        history_rows = list(db.execute(stmt).scalars().all())
                    history_rows.reverse()
                    projected_history_rows = _project_history_for_mode(
                        history_rows,
                        interaction_mode=interaction_mode,
                    )
                    llm_messages = [
                        {"role": "system", "content": system_prompt},
                    ]
                    if (
                        interaction_mode == RECEIVE_TEACHING_MODE
                        and projected_history_rows
                    ):
                        llm_messages.append({
                            "role": "user",
                            "content": _serialize_receive_teaching_history(
                                projected_history_rows
                            ),
                        })
                    else:
                        for msg in projected_history_rows:
                            llm_messages.append({
                                "role": msg.role,
                                "content": msg.content,
                            })
                    llm_messages.append({
                        "role": "user",
                        "content": user_message.content,
                    })
                else:
                    # Programmatic path already built its bounded context.
                    llm_messages = programmatic_messages

                # Call LLM via the resolved profile's client.  Programmatic
                # calls are ordinary text-only requests; no tools are sent.
                if general_answer is not None:
                    reply = general_answer
                elif programmatic_messages is not None:
                    diagnostics = LLMCallDiagnostics()
                    try:
                        reply = await llm_client.generate(
                            llm_messages,
                            diagnostics=diagnostics,
                        )
                    except Exception as exc:
                        try:
                            _log_programmatic_failure(
                                session_id=session_id,
                                user_message_id=saved_user_message_id,
                                stage="model_failed_after_snapshot",
                                error=exc,
                                retrieval_count=programmatic_retrieval_count,
                            )
                        except Exception:
                            pass
                        raise
                else:
                    reply = await _generate_reply_for_mode(
                        llm_client=llm_client,
                        messages=llm_messages,
                        interaction_mode=interaction_mode,
                        profile_kind=profile.kind,
                    )

                # Validate reply — empty or blank is an error
                if not reply or not reply.strip():
                    raise LLMInvalidResponseError(
                        detail="LLM returned an empty or blank response",
                        status_code=502,
                    )

                if language_target is not None:
                    try:
                        validate_reply_language(
                            reply,
                            target=language_target,
                            current_text=user_message.content,
                            evidence_texts=_language_evidence_texts(
                                llm_messages,
                                current_content=user_message.content,
                            ),
                        )
                    except ReplyLanguageMismatchError as exc:
                        raise LLMError(
                            detail={"code": exc.code, "message": exc.message},
                            status_code=502,
                        ) from exc

                # Save assistant message with the SAME captured snapshot
                # triple as the user message — never re-resolved.
                assistant_message = Message(
                    session_id=chat_session.id,
                    role="assistant",
                    content=reply,
                    llm_profile_id_snapshot=profile.id,
                    llm_profile_kind_snapshot=profile.kind,
                    llm_model_snapshot=profile.model,
                    interaction_mode_snapshot=interaction_mode,
                    teaching_strategy_snapshot=teaching_strategy_snapshot,
                    prompt_version_snapshot=prompt_version,
                )
                db.add(assistant_message)
                chat_session.updated_at = utc_now()
                db.commit()
                db.refresh(assistant_message)

                return user_message, assistant_message
            except LLMError as exc:
                try:
                    _log_chat_llm_failure(
                        session_id=session_id,
                        user_message_id=saved_user_message_id,
                        error=exc,
                    )
                except Exception:
                    pass
                try:
                    db.rollback()
                except Exception:
                    pass
                raise
            except Exception:
                try:
                    db.rollback()
                except Exception:
                    pass
                raise


    async def switch_teaching_strategy(self, session_id: int, strategy: str, db: Session) -> ChatSession:
        if not is_valid_teaching_strategy(strategy):
            raise ValueError("Unknown teaching strategy")
        if self._frozen_receive_teaching_policy is not None:
            raise ValueError("Frozen experiments do not switch teaching strategies")
        async with self._lock_registry.session_lock(session_id):
            try:
                session = db.scalars(
                    select(ChatSession).where(ChatSession.id == session_id)
                    .execution_options(populate_existing=True)
                ).first()
                if session is None:
                    raise SessionNotFoundError(session_id)
                if session.interaction_mode != RECEIVE_TEACHING_MODE:
                    raise ValueError("Switch to Receive teaching before changing teaching strategy")
                if session.teaching_strategy != strategy:
                    session.teaching_strategy = strategy
                    session.updated_at = utc_now()
                    db.commit()
                    db.refresh(session)
                return session
            except Exception:
                db.rollback()
                raise

    async def switch_interaction_mode(
        self,
        session_id: int,
        interaction_mode: str,
        db: Session,
    ) -> tuple[ChatSession, ModeSwitchEvent | None]:
        """Switch a session's interaction mode.

        Uses the same per-session lock as message sending, so a mode
        switch can never interleave with a user/assistant round and tear
        the mode snapshot.  Switching to the current mode is an
        idempotent no-op: no event row and no updated_at change.
        """
        if is_valid_interaction_mode(interaction_mode) is False:
            raise ValueError("Unknown interaction mode")

        async with self._lock_registry.session_lock(session_id):
            try:
                session_stmt = (
                    select(ChatSession)
                    .where(ChatSession.id == session_id)
                    .execution_options(populate_existing=True)
                )
                chat_session = db.execute(session_stmt).scalars().first()
                if chat_session is None:
                    raise SessionNotFoundError(session_id)

                if chat_session.interaction_mode == interaction_mode:
                    return chat_session, None

                boundary = capture_history_boundary(
                    db=db,
                    session_id=chat_session.id,
                    from_mode=chat_session.interaction_mode,
                    to_mode=interaction_mode,
                )
                event = ModeSwitchEvent(
                    session_id=chat_session.id,
                    from_mode=chat_session.interaction_mode,
                    to_mode=interaction_mode,
                    history_through_message_id=(
                        boundary.history_through_message_id
                    ),
                    history_boundary_version=(
                        boundary.history_boundary_version
                    ),
                    reviewable_user_message_count=(
                        boundary.reviewable_user_message_count
                    ),
                )
                chat_session.interaction_mode = interaction_mode
                chat_session.updated_at = utc_now()
                db.add(event)
                db.flush()
                db.refresh(chat_session)
                db.refresh(event)
                db.commit()

                return chat_session, event
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

    async def rename_session(
        self, session_id: int, title: str, db: Session,
    ) -> ChatSession:
        """Rename a chat session.

        Acquires the same per-session lock as
        :meth:`handle_session_message` and :meth:`delete_session`, so a
        rename cannot race with a send or delete.

        The caller is responsible for normalising and validating
        *title* before calling this method.  The stored title is used
        as-is — no further transformation is applied.

        Parameters
        ----------
        session_id:
            The id of the target chat session.
        title:
            The new title (already normalised by the Pydantic schema).
        db:
            An active SQLAlchemy ``Session``.

        Returns
        -------
        ChatSession
            The updated session (``db.refresh()`` has been called).

        Raises
        ------
        SessionNotFoundError
            When *session_id* does not exist in the database.
        """
        async with self._lock_registry.session_lock(session_id):
            chat_session = db.get(ChatSession, session_id)
            if chat_session is None:
                raise SessionNotFoundError(session_id)
            chat_session.title = title
            chat_session.title_is_manual = True
            chat_session.updated_at = utc_now()
            db.commit()
            db.refresh(chat_session)
            return chat_session

    def _requires_remote_history_ack(
        self,
        chat_session: ChatSession,
        target_profile: LLMProfile,
        db: Session,
    ) -> bool:
        """Decide whether switching to *target_profile* requires the
        client to acknowledge that chat history will be sent to a
        remote API service.

        Must run while holding the session lock, and counts the
        messages table directly — it never trusts a client-supplied
        message count.

        Returns ``True`` only when all of the following hold:

        * the target profile is an API profile;
        * the session has at least one message;
        * the session is NOT already the same reliable API binding
          (same profile id, same model snapshot, resolution status
          ready).
        """
        if target_profile.kind != "api":
            return False

        message_count = db.execute(
            select(func.count()).select_from(Message).where(
                Message.session_id == chat_session.id,
            )
        ).scalar()
        if not message_count:
            return False

        same_reliable_api_binding = (
            target_profile.kind == "api"
            and chat_session.llm_profile_id == target_profile.id
            and chat_session.llm_model_snapshot == target_profile.model
            and self._profiles.resolve(
                chat_session.llm_profile_id,
                chat_session.llm_model_snapshot,
            ).status
            == SessionProfileStatus.READY
        )
        return not same_reliable_api_binding

    async def switch_session_profile(
        self,
        session_id: int,
        profile_id: str,
        acknowledge_remote_history: bool,
        db: Session,
    ) -> ChatSession:
        """Re-bind a session to another LLM profile.

        Acquires the same per-session lock as send / delete / rename.
        Holding the lock, it re-reads the session, looks the target
        profile up in the registry, and runs the remote-history
        privacy decision against the messages table.  The binding is
        modified only after the decision passes.

        Does **not** call any LLM, probe or start Ollama, touch the
        messages, the title or ``title_is_manual``.

        Idempotence: when the stored profile id AND model snapshot
        already equal the target, no attribute is changed — no UPDATE
        is generated and ``updated_at`` is untouched.  A real change
        (different id, or a NULL / stale snapshot being repaired)
        writes the target values and bumps ``updated_at``.

        Parameters
        ----------
        session_id:
            The id of the target chat session.
        profile_id:
            The registry id of the profile to bind.
        acknowledge_remote_history:
            The client's acknowledgement for remote-history transfer.
        db:
            An active SQLAlchemy ``Session``.

        Returns
        -------
        ChatSession
            The updated session.

        Raises
        ------
        SessionNotFoundError
            When *session_id* does not exist in the database.
        UnknownLLMProfileError
            When *profile_id* is not in the registry.
        SessionProfileSwitchAckRequiredError
            When the remote-history acknowledgement is required but
            not given — the binding is left untouched.
        """
        async with self._lock_registry.session_lock(session_id):
            chat_session = db.get(ChatSession, session_id)
            if chat_session is None:
                raise SessionNotFoundError(session_id)

            target_profile = self._profiles.get(profile_id)
            if target_profile is None:
                raise UnknownLLMProfileError(profile_id)

            if self._requires_remote_history_ack(
                chat_session, target_profile, db,
            ) and not acknowledge_remote_history:
                raise SessionProfileSwitchAckRequiredError()

            # True idempotence: when the stored binding already equals
            # the target, return immediately — no attribute writes, no
            # flush, no refresh, no commit, no UPDATE, and updated_at
            # stays untouched.  (The acknowledgement decision above
            # still runs first, so an unreliable API binding cannot
            # bypass the confirmation through this path.)
            if (
                chat_session.llm_profile_id == target_profile.id
                and chat_session.llm_model_snapshot == target_profile.model
            ):
                return chat_session

            chat_session.llm_profile_id = target_profile.id
            chat_session.llm_model_snapshot = target_profile.model
            # updated_at is written explicitly here (not via an
            # onupdate hook) so the change is deterministic.
            chat_session.updated_at = utc_now()

            try:
                # Flush applies the writes, refresh re-reads the row,
                # commit finalises.  Any failure rolls the whole
                # transaction back.
                db.flush()
                db.refresh(chat_session)
                db.commit()
            except Exception:
                db.rollback()
                raise

            return chat_session

def _language_evidence_texts(
    messages: list[LLMMessage], *, current_content: str
) -> tuple[str, ...]:
    """Collect teacher originals actually present in the final context."""
    texts: list[str] = []
    for message in messages:
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if not isinstance(content, str) or content == current_content:
            continue
        try:
            data = json.loads(content)
        except (TypeError, ValueError):
            texts.append(content)
            continue
        if not isinstance(data, dict):
            continue
        for key in ("window_sources", "retrieved_sources"):
            items = data.get(key)
            if not isinstance(items, list):
                continue
            for item in items:
                if isinstance(item, dict) and isinstance(item.get("content"), str):
                    texts.append(item["content"])
        items = data.get("messages")
        if isinstance(items, list):
            for item in items:
                if isinstance(item, dict) and isinstance(item.get("content"), str):
                    texts.append(item["content"])
    return tuple(dict.fromkeys(texts))
