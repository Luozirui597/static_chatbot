"""Internal preparation and execution service for history reviews.

Preparation creates pending review rows. Execution claims the review
once via CAS, then runs the frozen two-stage History Review pipeline
(extraction followed by optional verification) against the frozen
sources and exposes the final parsed output.  Bounded repair retries
are owned by the pipeline; timeout recovery, HTTP mapping, background
tasks, and source selection reruns remain out of scope.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Literal

from sqlalchemy import delete, select, update
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session

from backend.chat_service import SessionLockRegistry
from backend.exceptions import (
    HistoryReviewBoundaryUnavailable,
    HistoryReviewEventNotFound,
    HistoryReviewEventNotReviewable,
    HistoryReviewError,
    HistoryReviewNoReviewableHistory,
    HistoryReviewRemoteAckRequired,
    HistoryReviewSessionNotFound,
    LLMError,
    SessionProfileConflictError,
    SessionProfileUnavailableError,
)
from backend.history_boundary import HISTORY_BOUNDARY_VERSION
from backend.history_review_parser import truncate_raw_output
from backend.history_review_prompt import (
    HistoryReviewExecutionSnapshot,
    HistoryReviewSourceSnapshot,
)
from backend.history_review_pipeline import (
    HistoryReviewPipelineError,
    run_history_review_pipeline,
)
from backend.history_review_selection import (
    HISTORY_REVIEW_BUDGET_VERSION,
    HISTORY_REVIEW_SELECTION_POLICY_VERSION,
    HistoryReviewCandidate,
    select_history_review_sources,
)
from backend.history_review_stages import HISTORY_REVIEW_PIPELINE_VERSION
from backend.interaction_modes import CORRECTIVE_MODE, RECEIVE_TEACHING_MODE
from backend.llm_profiles import LLMProfile, LLMProfileRegistry, SessionProfileStatus
from backend.models import (
    ChatSession,
    HistoryReview,
    HistoryReviewFinding,
    HistoryReviewSource,
    Message,
    ModeSwitchEvent,
    utc_now,
)

logger = logging.getLogger(__name__)

# Execution error codes -----------------------------------------------------

HISTORY_REVIEW_NOT_FOUND = "history_review_not_found"
HISTORY_REVIEW_EXECUTION_CONFLICT = "history_review_execution_conflict"
REVIEWER_UNAVAILABLE = "reviewer_unavailable"
REVIEWER_MODEL_CHANGED = "reviewer_model_changed"
REVIEWER_KIND_CHANGED = "reviewer_kind_changed"
PROMPT_VERSION_UNSUPPORTED = "unsupported_prompt_version"
SOURCE_CORRUPTED = "history_review_source_corrupted"
PRIVACY_ACK_MISSING = "privacy_ack_missing"
PRIVACY_CONFLICT = "privacy_conflict"
LLM_UPSTREAM_ERROR = "history_review_llm_upstream_error"
LLM_EMPTY_RESPONSE = "history_review_llm_empty_response"
LLM_NON_STRING_RESPONSE = "history_review_llm_non_string_response"
FINALIZE_FAILED = "history_review_finalize_failed"
INTERNAL_ERROR = "history_review_internal_error"
HISTORY_REVIEW_EXECUTION_INTERRUPTED = (
    "history_review_execution_interrupted"
)
HISTORY_REVIEW_RETRY_NOT_ALLOWED = "history_review_retry_not_allowed"

_SAFE_ERROR_MESSAGES = {
    HISTORY_REVIEW_NOT_FOUND: "History review not found.",
    HISTORY_REVIEW_EXECUTION_CONFLICT: "History review execution conflict.",
    REVIEWER_UNAVAILABLE: "Reviewer profile is unavailable.",
    REVIEWER_MODEL_CHANGED: "Reviewer model changed.",
    REVIEWER_KIND_CHANGED: "Reviewer profile kind changed.",
    PROMPT_VERSION_UNSUPPORTED: "Prompt version is unsupported.",
    SOURCE_CORRUPTED: "Frozen source data is corrupted.",
    PRIVACY_ACK_MISSING: "Remote history acknowledgement is missing.",
    PRIVACY_CONFLICT: "Remote history acknowledgement is inconsistent.",
    LLM_UPSTREAM_ERROR: "LLM request failed.",
    LLM_EMPTY_RESPONSE: "LLM returned an empty response.",
    LLM_NON_STRING_RESPONSE: "LLM returned a non-string response.",
    FINALIZE_FAILED: "History review finalization failed.",
    INTERNAL_ERROR: "History review execution failed.",
    HISTORY_REVIEW_EXECUTION_INTERRUPTED: (
        "History review execution was interrupted."
    ),
    HISTORY_REVIEW_RETRY_NOT_ALLOWED: (
        "This history review cannot be retried."
    ),
    "history_review_output_too_large": (
        "History review output exceeds configured limits."
    ),
    "history_review_json_syntax_error": (
        "History review output is not valid JSON."
    ),
    "history_review_json_structure_error": (
        "History review output has an invalid JSON structure."
    ),
    "history_review_json_semantic_error": (
        "History review output violates the semantic contract."
    ),
    "history_review_source_reference_invalid": (
        "History review output references an invalid source."
    ),
    "history_review_source_uncovered": (
        "History review output does not cover every source."
    ),
    "history_review_extraction_json_syntax_error": (
        "History review extraction output is not valid JSON."
    ),
    "history_review_extraction_json_structure_error": (
        "History review extraction output has an invalid JSON structure."
    ),
    "history_review_extraction_json_semantic_error": (
        "History review extraction output violates the semantic contract."
    ),
    "history_review_extraction_source_invalid": (
        "History review extraction references an invalid source."
    ),
    "history_review_extraction_source_uncovered": (
        "History review extraction does not cover every source."
    ),
    "history_review_extraction_quote_invalid": (
        "History review extraction contains an invalid source quote."
    ),
    "history_review_extraction_too_large": (
        "History review extraction output exceeds configured limits."
    ),
    "history_review_extraction_input_too_large": (
        "History review extraction input exceeds configured limits."
    ),
    "history_review_extraction_capacity_exceeded": (
        "History review extraction cannot fit all factual claims."
    ),
    "history_review_verification_json_syntax_error": (
        "History review verification output is not valid JSON."
    ),
    "history_review_verification_json_structure_error": (
        "History review verification output has an invalid JSON structure."
    ),
    "history_review_verification_json_semantic_error": (
        "History review verification output violates the semantic contract."
    ),
    "history_review_verification_claim_invalid": (
        "History review verification references an invalid claim."
    ),
    "history_review_verification_claim_uncovered": (
        "History review verification does not cover every claim."
    ),
    "history_review_verification_too_large": (
        "History review verification output exceeds configured limits."
    ),
    "history_review_verification_input_too_large": (
        "History review verification input exceeds configured limits."
    ),
}


def _safe_error_message(code: str) -> str:
    return _SAFE_ERROR_MESSAGES.get(
        code,
        "History review execution failed.",
    )


class HistoryReviewExecutionError(HistoryReviewError):
    """Base class for safe history-review execution domain errors."""

    code = "history_review_execution_error"
    default_message = "History review execution failed."

    def __init__(self, message: str | None = None) -> None:
        super().__init__(message or self.default_message)


class HistoryReviewExecutionNotFound(HistoryReviewExecutionError):
    code = HISTORY_REVIEW_NOT_FOUND
    default_message = "History review not found."


class HistoryReviewExecutionConflict(HistoryReviewExecutionError):
    code = HISTORY_REVIEW_EXECUTION_CONFLICT
    default_message = "History review execution conflict."


class HistoryReviewRetryNotAllowed(HistoryReviewExecutionError):
    code = HISTORY_REVIEW_RETRY_NOT_ALLOWED
    default_message = "This history review cannot be retried."


class _PreflightFailure(Exception):
    """Internal marker for deterministic pending-review failures."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(_safe_error_message(code))


@dataclass(frozen=True)
class HistoryReviewExecutionFinding:
    seq: int
    source_message_id: int
    verdict: Literal[
        "correct",
        "incorrect",
        "uncertain",
        "not_a_claim",
    ]
    claim_text: str
    correction_text: str | None
    explanation_text: str | None


@dataclass(frozen=True)
class HistoryReviewExecutionResult:
    review_id: int
    session_id: int
    status: Literal["running", "completed", "failed"]
    called_llm: bool
    summary: str | None
    coverage_note: str | None
    error_code: str | None
    error_message: str | None
    raw_output_truncated: bool
    findings: tuple[HistoryReviewExecutionFinding, ...]


@dataclass(frozen=True)
class _ClaimedExecution:
    snapshot: HistoryReviewExecutionSnapshot
    profile: LLMProfile


@dataclass(frozen=True)
class HistoryReviewPreparationResult:
    """Immutable result of preparing (or reusing) a history review."""

    review: HistoryReview
    sources: tuple[HistoryReviewSource, ...]
    created: bool


class HistoryReviewService:
    """Prepare and execute history reviews.

    Preparation never calls an LLM.  Execution freezes reviewer/source
    values, claims the review once, delegates LLM calls and bounded
    repair retries to the frozen pipeline, and persists either completed
    output or a safe failed state.
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

    # -- execution helpers ------------------------------------------------

    def _load_execution_review(
        self,
        db: Session,
        session_id: int,
        review_id: int,
    ) -> HistoryReview | None:
        stmt = (
            select(HistoryReview)
            .where(
                HistoryReview.id == review_id,
                HistoryReview.session_id == session_id,
            )
            .execution_options(populate_existing=True)
        )
        return db.execute(stmt).scalars().first()

    def _load_result_findings(
        self,
        db: Session,
        review_id: int,
    ) -> tuple[HistoryReviewExecutionFinding, ...]:
        stmt = (
            select(HistoryReviewFinding)
            .where(HistoryReviewFinding.review_id == review_id)
            .order_by(HistoryReviewFinding.seq.asc())
        )
        rows = db.execute(stmt).scalars().all()
        return tuple(
            HistoryReviewExecutionFinding(
                seq=row.seq,
                source_message_id=row.source_message_id,
                verdict=row.verdict,
                claim_text=row.claim_text,
                correction_text=row.correction_text,
                explanation_text=row.explanation_text,
            )
            for row in rows
        )

    def _build_execution_result(
        self,
        db: Session,
        review: HistoryReview,
        *,
        called_llm: bool,
    ) -> HistoryReviewExecutionResult:
        if review.status not in ("running", "completed", "failed"):
            raise HistoryReviewExecutionConflict()

        findings: tuple[HistoryReviewExecutionFinding, ...] = ()
        if review.status == "completed":
            findings = self._load_result_findings(db, review.id)

        return HistoryReviewExecutionResult(
            review_id=review.id,
            session_id=review.session_id,
            status=review.status,
            called_llm=called_llm,
            summary=review.summary,
            coverage_note=review.coverage_note,
            error_code=review.error_code,
            error_message=review.error_message,
            raw_output_truncated=bool(review.raw_output_truncated),
            findings=findings,
        )

    def _reload_authoritative_result(
        self,
        db: Session,
        session_id: int,
        review_id: int,
        *,
        called_llm: bool,
    ) -> HistoryReviewExecutionResult:
        db.rollback()
        review = self._load_execution_review(db, session_id, review_id)
        if review is None:
            raise HistoryReviewExecutionNotFound()
        if review.status == "pending":
            raise HistoryReviewExecutionConflict()
        result = self._build_execution_result(
            db, review, called_llm=called_llm
        )
        db.rollback()
        return result

    @staticmethod
    def _failed_execution_result(
        *,
        review_id: int,
        session_id: int,
        error_code: str,
        called_llm: bool,
    ) -> HistoryReviewExecutionResult:
        return HistoryReviewExecutionResult(
            review_id=review_id,
            session_id=session_id,
            status="failed",
            called_llm=called_llm,
            summary=None,
            coverage_note=None,
            error_code=error_code,
            error_message=_safe_error_message(error_code),
            raw_output_truncated=False,
            findings=(),
        )

    def _persist_pending_failure(
        self,
        db: Session,
        *,
        session_id: int,
        review_id: int,
        error_code: str,
    ) -> HistoryReviewExecutionResult:
        now = utc_now()
        try:
            update_result = db.execute(
                update(HistoryReview)
                .where(
                    HistoryReview.id == review_id,
                    HistoryReview.session_id == session_id,
                    HistoryReview.status == "pending",
                )
                .values(
                    status="failed",
                    error_code=error_code,
                    error_message=_safe_error_message(error_code),
                    started_at=None,
                    completed_at=now,
                    updated_at=now,
                    summary=None,
                    coverage_note=None,
                    raw_output=None,
                    raw_output_truncated=False,
                )
                .execution_options(synchronize_session=False)
            )
            if update_result.rowcount != 1:
                db.rollback()
                return self._reload_authoritative_result(
                    db,
                    session_id,
                    review_id,
                    called_llm=False,
                )
            db.commit()
        except Exception:
            db.rollback()
            raise

        return self._failed_execution_result(
            review_id=review_id,
            session_id=session_id,
            error_code=error_code,
            called_llm=False,
        )

    def _validate_sources_for_execution(
        self,
        db: Session,
        review: HistoryReview,
    ) -> tuple[
        list[HistoryReviewSource],
        tuple[Message, ...],
    ]:
        source_message_count = review.source_message_count
        if (
            type(source_message_count) is not int
            or source_message_count < 1
        ):
            raise _PreflightFailure(SOURCE_CORRUPTED)

        source_rows = list(self._load_sources(db, review.id))
        if len(source_rows) != source_message_count:
            raise _PreflightFailure(SOURCE_CORRUPTED)

        previous_message_id = 0
        message_ids: list[int] = []
        for index, source in enumerate(source_rows, start=1):
            if type(source.seq) is not int or source.seq != index:
                raise _PreflightFailure(SOURCE_CORRUPTED)
            if source.review_id != review.id:
                raise _PreflightFailure(SOURCE_CORRUPTED)
            if source.session_id != review.session_id:
                raise _PreflightFailure(SOURCE_CORRUPTED)
            if (
                type(source.message_id) is not int
                or source.message_id < 1
            ):
                raise _PreflightFailure(SOURCE_CORRUPTED)
            if source.message_id <= previous_message_id:
                raise _PreflightFailure(SOURCE_CORRUPTED)
            if (
                type(source.content_char_count) is not int
                or source.content_char_count < 0
            ):
                raise _PreflightFailure(SOURCE_CORRUPTED)
            previous_message_id = source.message_id
            message_ids.append(source.message_id)

        message_stmt = select(Message).where(
            Message.id.in_(message_ids),
        )
        messages_by_id = {
            message.id: message
            for message in db.execute(message_stmt).scalars().all()
        }
        if len(messages_by_id) != len(message_ids):
            raise _PreflightFailure(SOURCE_CORRUPTED)

        ordered_messages: list[Message] = []
        total_source_chars = 0
        for source in source_rows:
            message = messages_by_id.get(source.message_id)
            if message is None:
                raise _PreflightFailure(SOURCE_CORRUPTED)
            if message.session_id != review.session_id:
                raise _PreflightFailure(SOURCE_CORRUPTED)
            if message.role != "user":
                raise _PreflightFailure(SOURCE_CORRUPTED)
            if (
                message.interaction_mode_snapshot
                != RECEIVE_TEACHING_MODE
            ):
                raise _PreflightFailure(SOURCE_CORRUPTED)
            if not isinstance(message.content, str):
                raise _PreflightFailure(SOURCE_CORRUPTED)
            if source.content_char_count != len(message.content):
                raise _PreflightFailure(SOURCE_CORRUPTED)
            total_source_chars += len(message.content)
            ordered_messages.append(message)

        if (
            type(review.source_from_message_id) is not int
            or review.source_from_message_id
            != ordered_messages[0].id
        ):
            raise _PreflightFailure(SOURCE_CORRUPTED)
        if (
            type(review.source_through_message_id) is not int
            or review.source_through_message_id
            != ordered_messages[-1].id
        ):
            raise _PreflightFailure(SOURCE_CORRUPTED)
        if (
            type(review.upper_bound_message_id) is not int
            or review.upper_bound_message_id < 1
            or ordered_messages[-1].id > review.upper_bound_message_id
        ):
            raise _PreflightFailure(SOURCE_CORRUPTED)

        if review.lower_bound_kind == "session_start":
            if review.lower_bound_message_id is not None:
                raise _PreflightFailure(SOURCE_CORRUPTED)
        elif review.lower_bound_kind == "corrective_switch":
            lower_bound = review.lower_bound_message_id
            if (
                type(lower_bound) is not int
                or lower_bound < 1
                or ordered_messages[0].id <= lower_bound
            ):
                raise _PreflightFailure(SOURCE_CORRUPTED)
        else:
            raise _PreflightFailure(SOURCE_CORRUPTED)

        if (
            type(review.source_char_count) is not int
            or review.source_char_count != total_source_chars
        ):
            raise _PreflightFailure(SOURCE_CORRUPTED)

        return source_rows, tuple(ordered_messages)

    def _validate_privacy_for_execution(
        self,
        review: HistoryReview,
        reviewer: LLMProfile,
        source_messages: tuple[Message, ...],
    ) -> None:
        requires_ack = self._requires_remote_ack(
            reviewer, source_messages
        )
        ack = review.remote_history_acknowledged
        acked_at = review.remote_history_acknowledged_at
        ack_count = review.remote_history_ack_message_count

        if reviewer.kind != "api":
            if (
                ack is not False
                or acked_at is not None
                or ack_count is not None
            ):
                raise _PreflightFailure(PRIVACY_CONFLICT)
            return

        if not requires_ack:
            if (
                ack is not False
                or acked_at is not None
                or ack_count is not None
            ):
                raise _PreflightFailure(PRIVACY_CONFLICT)
            return

        if ack is not True:
            raise _PreflightFailure(PRIVACY_ACK_MISSING)
        if (
            acked_at is None
            or ack_count != review.source_message_count
        ):
            raise _PreflightFailure(PRIVACY_CONFLICT)

    def _validate_preflight(
        self,
        db: Session,
        review: HistoryReview,
    ) -> _ClaimedExecution:
        if review.prompt_version_snapshot != HISTORY_REVIEW_PIPELINE_VERSION:
            raise _PreflightFailure(PROMPT_VERSION_UNSUPPORTED)

        reviewer = self._profiles.get(
            review.reviewer_llm_profile_id_snapshot
        )
        if reviewer is None:
            raise _PreflightFailure(REVIEWER_UNAVAILABLE)
        if (
            reviewer.kind
            != review.reviewer_llm_profile_kind_snapshot
        ):
            raise _PreflightFailure(REVIEWER_KIND_CHANGED)
        if reviewer.model != review.reviewer_llm_model_snapshot:
            raise _PreflightFailure(REVIEWER_MODEL_CHANGED)

        source_rows, source_messages = (
            self._validate_sources_for_execution(db, review)
        )
        self._validate_privacy_for_execution(
            review, reviewer, source_messages
        )

        snapshot = HistoryReviewExecutionSnapshot(
            review_id=review.id,
            session_id=review.session_id,
            source_message_count=review.source_message_count,
            reviewer_profile_id=reviewer.id,
            reviewer_kind=reviewer.kind,
            reviewer_model=reviewer.model,
            prompt_version=review.prompt_version_snapshot,
            sources=tuple(
                HistoryReviewSourceSnapshot(
                    seq=source.seq,
                    message_id=source.message_id,
                    content=message.content,
                )
                for source, message in zip(
                    source_rows, source_messages, strict=True
                )
            ),
        )
        return _ClaimedExecution(
            snapshot=snapshot,
            profile=reviewer,
        )

    def _prepare_existing_review(
        self,
        db: Session,
        existing: HistoryReview,
        *,
        retry_failed: bool,
    ) -> HistoryReviewPreparationResult:
        if retry_failed:
            if existing.status == "failed":
                if existing.error_code != HISTORY_REVIEW_EXECUTION_INTERRUPTED:
                    raise HistoryReviewRetryNotAllowed()
            elif existing.status not in (
                "pending", "running", "completed",
            ):
                raise HistoryReviewRetryNotAllowed()
        return HistoryReviewPreparationResult(
            review=existing,
            sources=self._load_sources(db, existing.id),
            created=False,
        )

    def recover_abandoned_running_reviews(self, db: Session) -> int:
        """Mark all running reviews from a previous process as interrupted."""
        review_ids = list(db.execute(
            select(HistoryReview.id)
            .where(HistoryReview.status == "running")
        ).scalars().all())
        if not review_ids:
            db.rollback()
            return 0

        now = utc_now()
        try:
            update_result = db.execute(
                update(HistoryReview)
                .where(
                    HistoryReview.id.in_(review_ids),
                    HistoryReview.status == "running",
                )
                .values(
                    status="failed",
                    error_code=HISTORY_REVIEW_EXECUTION_INTERRUPTED,
                    error_message=_safe_error_message(
                        HISTORY_REVIEW_EXECUTION_INTERRUPTED
                    ),
                    summary=None,
                    coverage_note=None,
                    raw_output=None,
                    raw_output_truncated=False,
                    completed_at=now,
                    updated_at=now,
                )
                .execution_options(synchronize_session=False)
            )
            if update_result.rowcount:
                db.execute(
                    delete(HistoryReviewFinding).where(
                        HistoryReviewFinding.review_id.in_(review_ids)
                    )
                )
            db.commit()
        except Exception:
            db.rollback()
            raise
        return int(update_result.rowcount or 0)

    def _reset_interrupted_review_for_retry(
        self,
        db: Session,
        session_id: int,
        review_id: int,
    ) -> bool:
        now = utc_now()
        try:
            update_result = db.execute(
                update(HistoryReview)
                .where(
                    HistoryReview.id == review_id,
                    HistoryReview.session_id == session_id,
                    HistoryReview.status == "failed",
                    HistoryReview.error_code
                    == HISTORY_REVIEW_EXECUTION_INTERRUPTED,
                )
                .values(
                    status="pending",
                    attempt_count=HistoryReview.attempt_count + 1,
                    summary=None,
                    coverage_note=None,
                    raw_output=None,
                    raw_output_truncated=False,
                    error_code=None,
                    error_message=None,
                    started_at=None,
                    completed_at=None,
                    updated_at=now,
                )
                .execution_options(synchronize_session=False)
            )
            if update_result.rowcount != 1:
                db.rollback()
                return False
            db.execute(
                delete(HistoryReviewFinding).where(
                    HistoryReviewFinding.review_id == review_id
                )
            )
            db.commit()
        except Exception:
            db.rollback()
            raise
        return True

    def _claim_retry_or_authoritative(
        self,
        db: Session,
        session_id: int,
        review_id: int,
    ) -> _ClaimedExecution | HistoryReviewExecutionResult:
        review = self._load_execution_review(db, session_id, review_id)
        if review is None:
            db.rollback()
            raise HistoryReviewExecutionNotFound()

        if review.status != "failed":
            if review.status in ("running", "completed"):
                result = self._build_execution_result(
                    db, review, called_llm=False
                )
                db.rollback()
                return result
            db.rollback()
            raise HistoryReviewExecutionConflict()

        if review.error_code != HISTORY_REVIEW_EXECUTION_INTERRUPTED:
            db.rollback()
            raise HistoryReviewRetryNotAllowed()

        if not self._reset_interrupted_review_for_retry(
            db, session_id, review_id
        ):
            current = self._load_execution_review(db, session_id, review_id)
            if current is None:
                db.rollback()
                raise HistoryReviewExecutionNotFound()
            if current.status in ("running", "completed"):
                result = self._build_execution_result(
                    db, current, called_llm=False
                )
                db.rollback()
                return result
            if current.status == "pending":
                return self._claim_pending_review(
                    db, session_id, review_id
                )
            if (
                current.status == "failed"
                and current.error_code
                == HISTORY_REVIEW_EXECUTION_INTERRUPTED
            ):
                db.rollback()
                raise HistoryReviewExecutionConflict()
            db.rollback()
            raise HistoryReviewRetryNotAllowed()

        return self._claim_pending_review(db, session_id, review_id)

    def _claim_pending_review(
        self,
        db: Session,
        session_id: int,
        review_id: int,
    ) -> _ClaimedExecution | HistoryReviewExecutionResult:
        review = self._load_execution_review(db, session_id, review_id)
        if review is None:
            db.rollback()
            raise HistoryReviewExecutionNotFound()

        if review.status != "pending":
            result = self._build_execution_result(
                db, review, called_llm=False
            )
            db.rollback()
            return result

        try:
            claim = self._validate_preflight(db, review)
        except _PreflightFailure as exc:
            return self._persist_pending_failure(
                db,
                session_id=session_id,
                review_id=review_id,
                error_code=exc.code,
            )
        except Exception:
            db.rollback()
            raise

        now = utc_now()
        try:
            update_result = db.execute(
                update(HistoryReview)
                .where(
                    HistoryReview.id == review_id,
                    HistoryReview.session_id == session_id,
                    HistoryReview.status == "pending",
                )
                .values(
                    status="running",
                    started_at=now,
                    updated_at=now,
                )
                .execution_options(synchronize_session=False)
            )
            if update_result.rowcount != 1:
                db.rollback()
                return self._reload_authoritative_result(
                    db,
                    session_id,
                    review_id,
                    called_llm=False,
                )
            db.commit()
        except Exception:
            db.rollback()
            raise
        return claim

    @staticmethod
    def _raw_for_storage(
        raw_output: object,
    ) -> tuple[str | None, bool]:
        if not isinstance(raw_output, str) or not raw_output.strip():
            return None, False
        return truncate_raw_output(raw_output=raw_output)

    def _record_running_failure(
        self,
        db: Session,
        claim: _ClaimedExecution,
        *,
        error_code: str,
        raw_output: object,
        called_llm: bool,
    ) -> HistoryReviewExecutionResult | None:
        stored_raw, raw_truncated = self._raw_for_storage(raw_output)
        now = utc_now()
        try:
            update_result = db.execute(
                update(HistoryReview)
                .where(
                    HistoryReview.id == claim.snapshot.review_id,
                    HistoryReview.session_id == claim.snapshot.session_id,
                    HistoryReview.status == "running",
                )
                .values(
                    status="failed",
                    error_code=error_code,
                    error_message=_safe_error_message(error_code),
                    summary=None,
                    coverage_note=None,
                    raw_output=stored_raw,
                    raw_output_truncated=raw_truncated,
                    completed_at=now,
                    updated_at=now,
                )
                .execution_options(synchronize_session=False)
            )
            if update_result.rowcount != 1:
                db.rollback()
                return None
            db.execute(
                delete(HistoryReviewFinding).where(
                    HistoryReviewFinding.review_id
                    == claim.snapshot.review_id
                )
            )
            db.commit()
        except Exception:
            db.rollback()
            raise

        return HistoryReviewExecutionResult(
            review_id=claim.snapshot.review_id,
            session_id=claim.snapshot.session_id,
            status="failed",
            called_llm=called_llm,
            summary=None,
            coverage_note=None,
            error_code=error_code,
            error_message=_safe_error_message(error_code),
            raw_output_truncated=raw_truncated,
            findings=(),
        )

    def _handle_running_failure(
        self,
        db: Session,
        claim: _ClaimedExecution,
        *,
        session_id: int,
        review_id: int,
        error_code: str,
        raw_output: object,
        called_llm: bool,
    ) -> HistoryReviewExecutionResult:
        result = self._record_running_failure(
            db,
            claim,
            error_code=error_code,
            raw_output=raw_output,
            called_llm=called_llm,
        )
        if result is not None:
            return result
        return self._reload_authoritative_result(
            db,
            session_id,
            review_id,
            called_llm=called_llm,
        )

    def _handle_cancelled_execution(
        self,
        db: Session,
        claim: _ClaimedExecution,
    ) -> None:
        try:
            db.rollback()
            self._record_running_failure(
                db,
                claim,
                error_code=HISTORY_REVIEW_EXECUTION_INTERRUPTED,
                raw_output=None,
                called_llm=True,
            )
        except Exception:
            try:
                db.rollback()
            except Exception:
                pass
            logger.exception(
                "Failed to mark cancelled history review as interrupted"
            )

    def _finalize_success(
        self,
        db: Session,
        claim: _ClaimedExecution,
        parsed: object,
        raw_output: str,
    ) -> HistoryReviewExecutionResult | None:
        stored_raw, raw_truncated = truncate_raw_output(
            raw_output=raw_output
        )
        now = utc_now()
        try:
            update_result = db.execute(
                update(HistoryReview)
                .where(
                    HistoryReview.id == claim.snapshot.review_id,
                    HistoryReview.session_id == claim.snapshot.session_id,
                    HistoryReview.status == "running",
                )
                .values(
                    status="completed",
                    summary=parsed.summary,
                    coverage_note=parsed.coverage_note,
                    raw_output=stored_raw,
                    raw_output_truncated=raw_truncated,
                    error_code=None,
                    error_message=None,
                    completed_at=now,
                    updated_at=now,
                )
                .execution_options(synchronize_session=False)
            )
            if update_result.rowcount != 1:
                db.rollback()
                return None

            for seq, finding in enumerate(parsed.findings, start=1):
                db.add(HistoryReviewFinding(
                    review_id=claim.snapshot.review_id,
                    seq=seq,
                    source_message_id=finding.source_message_id,
                    verdict=finding.verdict,
                    claim_text=finding.claim_text,
                    correction_text=finding.correction_text,
                    explanation_text=finding.explanation_text,
                ))
            db.commit()
        except Exception:
            db.rollback()
            raise

        result_findings = tuple(
            HistoryReviewExecutionFinding(
                seq=seq,
                source_message_id=finding.source_message_id,
                verdict=finding.verdict,
                claim_text=finding.claim_text,
                correction_text=finding.correction_text,
                explanation_text=finding.explanation_text,
            )
            for seq, finding in enumerate(parsed.findings, start=1)
        )
        return HistoryReviewExecutionResult(
            review_id=claim.snapshot.review_id,
            session_id=claim.snapshot.session_id,
            status="completed",
            called_llm=True,
            summary=parsed.summary,
            coverage_note=parsed.coverage_note,
            error_code=None,
            error_message=None,
            raw_output_truncated=raw_truncated,
            findings=result_findings,
        )

    async def execute_history_review(
        self,
        *,
        session_id: int,
        review_id: int,
        db: Session,
        retry_failed: bool = False,
    ) -> HistoryReviewExecutionResult:
        """Execute a pending history review exactly once per claim."""
        session_id = self._require_positive_int("session_id", session_id)
        review_id = self._require_positive_int("review_id", review_id)
        retry_failed = self._require_bool("retry_failed", retry_failed)

        async with self._lock_registry.session_lock(session_id):
            if retry_failed:
                claim_or_result = self._claim_retry_or_authoritative(
                    db, session_id, review_id
                )
            else:
                claim_or_result = self._claim_pending_review(
                    db, session_id, review_id
                )
            if isinstance(claim_or_result, HistoryReviewExecutionResult):
                return claim_or_result
            claim = claim_or_result

            try:
                pipeline_result = await run_history_review_pipeline(
                    client=claim.profile.client,
                    sources=claim.snapshot.sources,
                )
            except asyncio.CancelledError:
                self._handle_cancelled_execution(db, claim)
                raise
            except LLMError:
                return self._handle_running_failure(
                    db,
                    claim,
                    session_id=session_id,
                    review_id=review_id,
                    error_code=LLM_UPSTREAM_ERROR,
                    raw_output=None,
                    called_llm=True,
                )
            except HistoryReviewPipelineError as exc:
                return self._handle_running_failure(
                    db,
                    claim,
                    session_id=session_id,
                    review_id=review_id,
                    error_code=exc.code,
                    raw_output=exc.audit_raw_output,
                    called_llm=exc.called_llm,
                )
            except Exception:
                return self._handle_running_failure(
                    db,
                    claim,
                    session_id=session_id,
                    review_id=review_id,
                    error_code=INTERNAL_ERROR,
                    raw_output=None,
                    called_llm=True,
                )

            try:
                finalized = self._finalize_success(
                    db,
                    claim,
                    pipeline_result.parsed,
                    pipeline_result.audit_raw_output,
                )
            except SQLAlchemyError as original_exc:
                try:
                    failed = self._record_running_failure(
                        db,
                        claim,
                        error_code=FINALIZE_FAILED,
                        raw_output=pipeline_result.audit_raw_output,
                        called_llm=True,
                    )
                except Exception:
                    db.rollback()
                    raise original_exc from None

                if failed is not None:
                    return failed

                return self._reload_authoritative_result(
                    db,
                    session_id,
                    review_id,
                    called_llm=True,
                )

            if finalized is not None:
                return finalized
            return self._reload_authoritative_result(
                db,
                session_id,
                review_id,
                called_llm=True,
            )

    async def prepare_history_review(
        self,
        *,
        session_id: int,
        mode_switch_event_id: int,
        acknowledge_remote_history: bool,
        db: Session,
        retry_failed: bool = False,
    ) -> HistoryReviewPreparationResult:
        """Prepare a pending review and fixed source rows.

        No LLM call, prompt text, JSON parsing, findings, model retry,
        or HTTP mapping is performed here.
        """
        session_id = self._require_positive_int("session_id", session_id)
        mode_switch_event_id = self._require_positive_int(
            "mode_switch_event_id", mode_switch_event_id
        )
        acknowledge_remote_history = self._require_bool(
            "acknowledge_remote_history", acknowledge_remote_history
        )
        retry_failed = self._require_bool("retry_failed", retry_failed)

        async with self._lock_registry.session_lock(session_id):
            try:
                chat_session = self._load_session(db, session_id)
                event = self._load_event(db, session_id, mode_switch_event_id)
                self._validate_event_direction(event)

                existing = self._load_existing_review(
                    db, session_id, mode_switch_event_id
                )
                if existing is not None:
                    return self._prepare_existing_review(
                        db,
                        existing,
                        retry_failed=retry_failed,
                    )
                if retry_failed:
                    raise HistoryReviewRetryNotAllowed()

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
                    prompt_version_snapshot=HISTORY_REVIEW_PIPELINE_VERSION,
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
                    return self._prepare_existing_review(
                        db,
                        existing,
                        retry_failed=retry_failed,
                    )
                raise
            except Exception:
                db.rollback()
                raise
