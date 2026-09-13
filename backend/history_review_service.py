"""Internal preparation service for history reviews.

This module deliberately stops before any LLM call, prompt building,
JSON parsing, finding persistence, retry execution, or HTTP mapping.
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from backend.chat_service import SessionLockRegistry
from backend.exceptions import (
    HistoryReviewBoundaryUnavailable,
    HistoryReviewEventNotFound,
    HistoryReviewEventNotReviewable,
    HistoryReviewNoReviewableHistory,
    HistoryReviewRemoteAckRequired,
    HistoryReviewSessionNotFound,
    SessionProfileConflictError,
    SessionProfileUnavailableError,
)
from backend.history_boundary import HISTORY_BOUNDARY_VERSION
from backend.history_review_prompt import HISTORY_REVIEW_PROMPT_VERSION
from backend.history_review_selection import (
    HISTORY_REVIEW_BUDGET_VERSION,
    HISTORY_REVIEW_SELECTION_POLICY_VERSION,
    HistoryReviewCandidate,
    select_history_review_sources,
)
from backend.interaction_modes import CORRECTIVE_MODE, RECEIVE_TEACHING_MODE
from backend.llm_profiles import LLMProfile, LLMProfileRegistry, SessionProfileStatus
from backend.models import (
    ChatSession,
    HistoryReview,
    HistoryReviewSource,
    Message,
    ModeSwitchEvent,
    utc_now,
)

@dataclass(frozen=True)
class HistoryReviewPreparationResult:
    """Immutable result of preparing (or reusing) a history review."""

    review: HistoryReview
    sources: tuple[HistoryReviewSource, ...]
    created: bool


class HistoryReviewService:
    """Prepare pending history reviews and fixed source rows.

    The service never calls an LLM and never writes findings.
    """

    def __init__(
        self,
        *,
        profiles: LLMProfileRegistry,
        lock_registry: SessionLockRegistry,
    ) -> None:
        self._profiles = profiles
        self._lock_registry = lock_registry

    @staticmethod
    def _require_positive_int(name: str, value: object) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} must be a positive integer")
        if value < 1:
            raise ValueError(f"{name} must be a positive integer")
        return value

    @staticmethod
    def _require_bool(name: str, value: object) -> bool:
        if type(value) is not bool:
            raise ValueError(f"{name} must be a boolean")
        return value

    # -- database helpers -------------------------------------------------

    def _load_session(self, db: Session, session_id: int) -> ChatSession:
        stmt = (
            select(ChatSession)
            .where(ChatSession.id == session_id)
            .execution_options(populate_existing=True)
        )
        chat_session = db.execute(stmt).scalars().first()
        if chat_session is None:
            raise HistoryReviewSessionNotFound()
        return chat_session

    def _load_event(
        self, db: Session, session_id: int, event_id: int,
    ) -> ModeSwitchEvent:
        stmt = select(ModeSwitchEvent).where(
            ModeSwitchEvent.id == event_id,
            ModeSwitchEvent.session_id == session_id,
        )
        event = db.execute(stmt).scalars().first()
        if event is None:
            raise HistoryReviewEventNotFound()
        return event

    def _load_existing_review(
        self, db: Session, session_id: int, event_id: int,
    ) -> HistoryReview | None:
        stmt = select(HistoryReview).where(
            HistoryReview.session_id == session_id,
            HistoryReview.mode_switch_event_id == event_id,
        )
        return db.execute(stmt).scalars().first()

    def _load_sources(
        self, db: Session, review_id: int,
    ) -> tuple[HistoryReviewSource, ...]:
        stmt = (
            select(HistoryReviewSource)
            .where(HistoryReviewSource.review_id == review_id)
            .order_by(HistoryReviewSource.seq.asc())
        )
        return tuple(db.execute(stmt).scalars().all())

    def _resolve_lower_bound(
        self, db: Session, session_id: int, event: ModeSwitchEvent,
    ) -> tuple[str, int | None]:
        previous = db.execute(
            select(ModeSwitchEvent)
            .where(
                ModeSwitchEvent.session_id == session_id,
                ModeSwitchEvent.id < event.id,
                ModeSwitchEvent.from_mode == CORRECTIVE_MODE,
                ModeSwitchEvent.to_mode == RECEIVE_TEACHING_MODE,
            )
            .order_by(ModeSwitchEvent.id.desc())
            .limit(1)
        ).scalars().first()

        if previous is None:
            return "session_start", None

        if previous.history_boundary_version != HISTORY_BOUNDARY_VERSION:
            raise HistoryReviewBoundaryUnavailable()
        return "corrective_switch", previous.history_through_message_id

    def _load_eligible_candidates(
        self,
        db: Session,
        session_id: int,
        lower_bound: int | None,
        upper_bound: int,
    ) -> tuple[
        tuple[HistoryReviewCandidate, ...],
        dict[int, Message],
    ]:
        stmt = select(Message).where(
            Message.session_id == session_id,
            Message.role == "user",
            Message.interaction_mode_snapshot == RECEIVE_TEACHING_MODE,
            Message.id <= upper_bound,
        )
        if lower_bound is not None:
            stmt = stmt.where(Message.id > lower_bound)
        stmt = stmt.order_by(Message.id.asc())

        messages = db.execute(stmt).scalars().all()
        candidates: list[HistoryReviewCandidate] = []
        messages_by_id: dict[int, Message] = {}
        previous_id = 0

        for message in messages:
            if message.session_id != session_id:
                raise HistoryReviewBoundaryUnavailable()
            if message.role != "user":
                raise HistoryReviewBoundaryUnavailable()
            if message.interaction_mode_snapshot != RECEIVE_TEACHING_MODE:
                raise HistoryReviewBoundaryUnavailable()
            if message.id <= previous_id:
                raise HistoryReviewBoundaryUnavailable()
            if message.id > upper_bound:
                raise HistoryReviewBoundaryUnavailable()
            if lower_bound is not None and message.id <= lower_bound:
                raise HistoryReviewBoundaryUnavailable()

            candidate = HistoryReviewCandidate(
                message_id=message.id,
                content=message.content,
            )
            candidates.append(candidate)
            messages_by_id[message.id] = message
            previous_id = message.id

        return tuple(candidates), messages_by_id

    def _resolve_reviewer_profile(
        self, chat_session: ChatSession,
    ) -> LLMProfile:
        resolution = self._profiles.resolve(
            chat_session.llm_profile_id,
            chat_session.llm_model_snapshot,
        )
        if resolution.status == SessionProfileStatus.PROFILE_UNAVAILABLE:
            raise SessionProfileUnavailableError(
                chat_session.llm_profile_id
            )
        if resolution.status in (
            SessionProfileStatus.MODEL_CHANGED,
            SessionProfileStatus.LEGACY_UNKNOWN,
        ):
            raise SessionProfileConflictError(
                status=resolution.status.value,
                snapshot=chat_session.llm_model_snapshot,
            )
        assert resolution.profile is not None
        return resolution.profile

    @staticmethod
    def _requires_remote_ack(
        reviewer: LLMProfile,
        selected_messages: tuple[Message, ...],
    ) -> bool:
        if reviewer.kind != "api":
            return False
        for message in selected_messages:
            if message.llm_profile_id_snapshot != reviewer.id:
                return True
            if message.llm_profile_kind_snapshot != "api":
                return True
            if message.llm_model_snapshot != reviewer.model:
                return True
        return False

    @staticmethod
    def _validate_event_direction(event: ModeSwitchEvent) -> None:
        if (
            event.from_mode != RECEIVE_TEACHING_MODE
            or event.to_mode != CORRECTIVE_MODE
        ):
            raise HistoryReviewEventNotReviewable()

    async def prepare_history_review(
        self,
        *,
        session_id: int,
        mode_switch_event_id: int,
        acknowledge_remote_history: bool,
        db: Session,
    ) -> HistoryReviewPreparationResult:
        """Prepare a pending review and fixed source rows.

        No LLM call, prompt text, JSON parsing, findings, retry, or
        HTTP mapping is performed here.
        """
        session_id = self._require_positive_int("session_id", session_id)
        mode_switch_event_id = self._require_positive_int(
            "mode_switch_event_id", mode_switch_event_id
        )
        acknowledge_remote_history = self._require_bool(
            "acknowledge_remote_history", acknowledge_remote_history
        )

        async with self._lock_registry.session_lock(session_id):
            try:
                chat_session = self._load_session(db, session_id)
                event = self._load_event(db, session_id, mode_switch_event_id)
                self._validate_event_direction(event)

                existing = self._load_existing_review(
                    db, session_id, mode_switch_event_id
                )
                if existing is not None:
                    return HistoryReviewPreparationResult(
                        review=existing,
                        sources=self._load_sources(db, existing.id),
                        created=False,
                    )

                if event.history_boundary_version != HISTORY_BOUNDARY_VERSION:
                    raise HistoryReviewBoundaryUnavailable()
                reviewable_count = event.reviewable_user_message_count
                if reviewable_count is None or reviewable_count < 0:
                    raise HistoryReviewBoundaryUnavailable()

                upper_bound = event.history_through_message_id
                if upper_bound is None:
                    raise HistoryReviewNoReviewableHistory()

                lower_kind, lower_bound = self._resolve_lower_bound(
                    db, session_id, event
                )
                if (
                    lower_bound is not None
                    and lower_bound > upper_bound
                ):
                    raise HistoryReviewBoundaryUnavailable()

                candidates, messages_by_id = self._load_eligible_candidates(
                    db,
                    session_id=session_id,
                    lower_bound=lower_bound,
                    upper_bound=upper_bound,
                )

                if len(candidates) != reviewable_count:
                    raise HistoryReviewBoundaryUnavailable()
                if len(candidates) == 0:
                    raise HistoryReviewNoReviewableHistory()

                selection = select_history_review_sources(candidates)
                if selection.source_message_count == 0:
                    raise HistoryReviewNoReviewableHistory()

                selected_messages = tuple(
                    messages_by_id[candidate.message_id]
                    for candidate in selection.selected_candidates
                )

                reviewer = self._resolve_reviewer_profile(chat_session)
                requires_ack = self._requires_remote_ack(
                    reviewer, selected_messages
                )
                if requires_ack and not acknowledge_remote_history:
                    raise HistoryReviewRemoteAckRequired(
                        source_message_count=selection.source_message_count,
                        reviewer_profile_label=reviewer.label,
                        reviewer_model=reviewer.model,
                        truncated=selection.truncated,
                    )

                review = HistoryReview(
                    session_id=session_id,
                    mode_switch_event_id=event.id,
                    status="pending",
                    lower_bound_kind=lower_kind,
                    lower_bound_message_id=lower_bound,
                    upper_bound_message_id=upper_bound,
                    eligible_from_message_id=(
                        selection.eligible_from_message_id
                    ),
                    eligible_through_message_id=(
                        selection.eligible_through_message_id
                    ),
                    eligible_message_count=(
                        selection.eligible_message_count
                    ),
                    eligible_char_count=selection.eligible_char_count,
                    source_from_message_id=(
                        selection.source_from_message_id
                    ),
                    source_through_message_id=(
                        selection.source_through_message_id
                    ),
                    source_message_count=selection.source_message_count,
                    source_char_count=selection.source_char_count,
                    truncated=selection.truncated,
                    reviewer_llm_profile_id_snapshot=reviewer.id,
                    reviewer_llm_profile_kind_snapshot=reviewer.kind,
                    reviewer_llm_model_snapshot=reviewer.model,
                    prompt_version_snapshot=HISTORY_REVIEW_PROMPT_VERSION,
                    budget_version_snapshot=(
                        selection.budget_version
                    ),
                    selection_policy_version=(
                        selection.selection_policy_version
                    ),
                    attempt_count=1,
                    remote_history_acknowledged=requires_ack,
                    remote_history_acknowledged_at=(
                        utc_now() if requires_ack else None
                    ),
                    remote_history_ack_message_count=(
                        selection.source_message_count
                        if requires_ack
                        else None
                    ),
                )
                db.add(review)
                db.flush()

                for seq, message in enumerate(selected_messages, start=1):
                    db.add(HistoryReviewSource(
                        review_id=review.id,
                        session_id=session_id,
                        message_id=message.id,
                        seq=seq,
                        content_char_count=len(message.content),
                    ))
                db.flush()

                sources = self._load_sources(db, review.id)
                db.commit()
                return HistoryReviewPreparationResult(
                    review=review,
                    sources=sources,
                    created=True,
                )
            except IntegrityError:
                db.rollback()
                existing = self._load_existing_review(
                    db, session_id, mode_switch_event_id
                )
                if existing is not None:
                    return HistoryReviewPreparationResult(
                        review=existing,
                        sources=self._load_sources(db, existing.id),
                        created=False,
                    )
                raise
            except Exception:
                db.rollback()
                raise
