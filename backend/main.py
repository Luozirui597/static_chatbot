"""FastAPI application entry point."""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Response
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import func, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from backend import config
from backend.chat_service import (
    ChatService,
    SessionLockRegistry,
    SessionNotFoundError,
)
from backend.database import create_tables, engine, get_db, run_migrations
from backend.exceptions import (
    HistoryReviewBoundaryUnavailable,
    HistoryReviewEventNotFound,
    HistoryReviewEventNotReviewable,
    HistoryReviewNoReviewableHistory,
    HistoryReviewRemoteAckRequired,
    HistoryReviewSessionNotFound,
    LLMError,
    SessionProfileConflictError,
    SessionProfileSwitchAckRequiredError,
    SessionProfileUnavailableError,
    UnknownLLMProfileError,
)
from backend.llm_client import create_llm_client
from backend.history_boundary import HISTORY_BOUNDARY_VERSION
from backend.history_review_selection import HistoryReviewSourceMessageTooLarge
from backend.history_review_service import (
    HistoryReviewExecutionConflict,
    HistoryReviewExecutionNotFound,
    HistoryReviewService,
)
from backend.interaction_modes import CORRECTIVE_MODE, RECEIVE_TEACHING_MODE
from backend.llm_profiles import LLMProfile, LLMProfileRegistry
from backend.models import (
    ChatSession,
    HistoryReview,
    HistoryReviewFinding,
    Message,
    ModeSwitchEvent,
)
from backend.schemas import (
    ChatRequest,
    ChatResponse,
    CreateSessionRequest,
    DeleteResponse,
    HistoryReviewAckRequiredDetail,
    HistoryReviewAckRequiredResponse,
    HistoryReviewCreateRequest,
    HistoryReviewDetailResponse,
    HistoryReviewErrorDetail,
    HistoryReviewErrorResponse,
    HistoryReviewFindingResponse,
    HistoryReviewSummaryResponse,
    LLMProfilePublic,
    MessageResponse,
    ModeSwitchEventResponse,
    RemoteHistoryAckRequiredDetail,
    RenameSessionRequest,
    ReviewIdPath,
    SendMessageResponse,
    SessionIdPath,
    SessionResponse,
    SwitchInteractionModeRequest,
    SwitchInteractionModeResponse,
    SwitchSessionProfileRequest,
)

# ---------------------------------------------------------------------------
# Paths — resolved relative to this file so the app works regardless of CWD
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"

# ---------------------------------------------------------------------------
# Profile registry factory
# ---------------------------------------------------------------------------


def _build_production_registry() -> LLMProfileRegistry:
    """Build the production profile registry from environment config.

    Always includes a ``"default"`` profile.  When ``LOCAL_LLM_ENABLED``
    is true, also includes a ``"local"`` profile backed by Ollama.

    The default profile reflects ``LLM_MODE``:

    * ``fake`` → kind ``"fake"``, model ``"fake"``, FakeLLMClient.
    * ``real`` → kind ``"api"``, model from ``LLM_MODEL``,
      label from ``LLM_PROFILE_LABEL`` or ``"API Model"``.
    """
    profiles: list[LLMProfile] = []

    # -- Default profile --------------------------------------------------
    if config.LLM_MODE.strip().lower() == "fake":
        default_client = create_llm_client()
        default_label = config.LLM_PROFILE_LABEL.strip() or "Fake Model"
        profiles.append(LLMProfile(
            id="default",
            label=default_label,
            kind="fake",
            model="fake",
            client=default_client,
            is_default=True,
        ))
    else:
        # real / API mode
        default_client = create_llm_client()
        default_label = config.LLM_PROFILE_LABEL.strip() or "API Model"
        default_model = config.LLM_MODEL.strip()
        if not default_model:
            raise ValueError("LLM_MODEL is required when LLM_MODE is not 'fake'")
        profiles.append(LLMProfile(
            id="default",
            label=default_label,
            kind="api",
            model=default_model,
            client=default_client,
            is_default=True,
        ))

    # -- Optional local profile -------------------------------------------
    if config.LOCAL_LLM_ENABLED:
        local_model = config.LOCAL_LLM_MODEL.strip()
        if not local_model:
            raise ValueError(
                "LOCAL_LLM_MODEL is required when LOCAL_LLM_ENABLED=true"
            )
        local_api_key = config.LOCAL_LLM_API_KEY.strip() or "ollama"
        local_base_url = (
            config.LOCAL_LLM_API_BASE_URL.strip()
            or "http://127.0.0.1:11435/v1"
        )
        local_label = config.LOCAL_LLM_PROFILE_LABEL.strip() or "Local Model"
        local_reasoning = config.LOCAL_LLM_REASONING_EFFORT.strip()

        local_client = create_llm_client(
            mode="real",
            api_key=local_api_key,
            base_url=local_base_url,
            model=local_model,
            reasoning_effort=local_reasoning,
        )
        profiles.append(LLMProfile(
            id="local",
            label=local_label,
            kind="local",
            model=local_model,
            client=local_client,
            is_default=False,
        ))

    return LLMProfileRegistry(profiles)


def _is_strict_positive_int(value: object) -> bool:
    """Whether *value* is a Python int strictly greater than zero."""
    return type(value) is int and value > 0


def _build_mode_switch_event_response(
    event: ModeSwitchEvent,
) -> ModeSwitchEventResponse:
    """Build one public event response with its derived support flag.

    ``review_supported`` describes only this event's immutable boundary
    metadata.  It does not guarantee that a review can be executed.
    """
    review_supported = (
        event.from_mode == RECEIVE_TEACHING_MODE
        and event.to_mode == CORRECTIVE_MODE
        and event.history_boundary_version == HISTORY_BOUNDARY_VERSION
        and _is_strict_positive_int(event.history_through_message_id)
        and _is_strict_positive_int(event.reviewable_user_message_count)
    )
    return ModeSwitchEventResponse(
        id=event.id,
        session_id=event.session_id,
        from_mode=event.from_mode,
        to_mode=event.to_mode,
        created_at=event.created_at,
        history_through_message_id=event.history_through_message_id,
        history_boundary_version=event.history_boundary_version,
        reviewable_user_message_count=event.reviewable_user_message_count,
        review_supported=review_supported,
    )


# ---------------------------------------------------------------------------
# Application & dependencies
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Create database tables and run migrations on startup."""
    create_tables()
    run_migrations(engine)
    yield


app = FastAPI(title="Static Chatbot", lifespan=lifespan)

logger = logging.getLogger(__name__)

_profiles = _build_production_registry()
_session_locks = SessionLockRegistry()

chat_service = ChatService(
    profiles=_profiles,
    lock_registry=_session_locks,
)
history_review_service = HistoryReviewService(
    profiles=_profiles,
    lock_registry=_session_locks,
)

# ---------------------------------------------------------------------------
# API routes
# ---------------------------------------------------------------------------


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Accept a user message and return a chat reply."""
    try:
        reply = await chat_service.handle_message(request.message)
    except LLMError as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail=exc.detail,
        ) from exc
    return ChatResponse(reply=reply)


# ---------------------------------------------------------------------------
# Profile routes
# ---------------------------------------------------------------------------


@app.get(
    "/api/llm/profiles",
    response_model=list[LLMProfilePublic],
)
def list_profiles():
    """Return every available LLM profile (public fields only)."""
    return chat_service.list_profiles_public()


# ---------------------------------------------------------------------------
# Session routes
# ---------------------------------------------------------------------------


@app.post(
    "/api/sessions",
    response_model=SessionResponse,
    status_code=201,
)
def create_session(
    request: CreateSessionRequest | None = None,
    db: Session = Depends(get_db),
):
    """Create a new chat session.

    When no body, ``{}``, or JSON ``null`` is sent the default profile
    is used.
    """
    profile_id = request.llm_profile_id if request is not None else "default"

    try:
        session = chat_service.create_session(profile_id, db)
    except UnknownLLMProfileError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    return chat_service.build_session_response(session)


@app.get(
    "/api/sessions",
    response_model=list[SessionResponse],
)
def list_sessions(db: Session = Depends(get_db)):
    """Return all sessions, newest first.

    Orders by ``updated_at DESC, id DESC`` so the result is stable even
    when multiple sessions share the same ``updated_at``.
    """
    stmt = select(ChatSession).order_by(
        ChatSession.updated_at.desc(),
        ChatSession.id.desc(),
    )
    sessions = db.execute(stmt).scalars().all()
    return [chat_service.build_session_response(s) for s in sessions]


@app.get(
    "/api/sessions/{session_id}",
    response_model=SessionResponse,
)
def get_session(session_id: int, db: Session = Depends(get_db)):
    """Return a single session by id."""
    session = db.get(ChatSession, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return chat_service.build_session_response(session)


@app.get(
    "/api/sessions/{session_id}/messages",
    response_model=list[MessageResponse],
)
def get_messages(session_id: int, db: Session = Depends(get_db)):
    """Return all messages for a session, ordered by id ascending."""
    session = db.get(ChatSession, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    stmt = (
        select(Message)
        .where(Message.session_id == session_id)
        .order_by(Message.id.asc())
    )
    return db.execute(stmt).scalars().all()


@app.post(
    "/api/sessions/{session_id}/messages",
    response_model=SendMessageResponse,
    status_code=200,
)
async def send_message(
    session_id: int,
    request: ChatRequest,
    db: Session = Depends(get_db),
):
    """Send a message within an existing chat session.

    Returns both the saved user message and the assistant reply.
    """
    try:
        user_msg, asst_msg = await chat_service.handle_session_message(
            session_id=session_id,
            content=request.message,
            db=db,
        )
    except SessionNotFoundError:
        raise HTTPException(
            status_code=404, detail="Session not found"
        ) from None
    except SessionProfileUnavailableError as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc),
        ) from exc
    except SessionProfileConflictError as exc:
        raise HTTPException(
            status_code=409,
            detail=str(exc),
        ) from exc
    except LLMError as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail=exc.detail,
        ) from exc

    return SendMessageResponse(
        user_message=user_msg,
        assistant_message=asst_msg,
    )


@app.delete(
    "/api/sessions/{session_id}",
    response_model=DeleteResponse,
)
async def delete_session(session_id: int, db: Session = Depends(get_db)):
    """Delete a session and all of its messages.

    Uses the same per-session lock as ``send_message`` so a delete
    cannot race with an in-progress generation.
    """
    try:
        await chat_service.delete_session(session_id, db)
    except SessionNotFoundError:
        raise HTTPException(
            status_code=404, detail="Session not found"
        ) from None
    return DeleteResponse(ok=True)


@app.patch(
    "/api/sessions/{session_id}",
    response_model=SessionResponse,
)
async def rename_session(
    session_id: int,
    request: RenameSessionRequest,
    db: Session = Depends(get_db),
):
    """Rename a chat session.

    Uses the same per-session lock as ``send_message`` and
    ``delete_session`` so a rename cannot race with a send or delete.
    The title is normalised by Pydantic before reaching the service
    layer.
    """
    try:
        session = await chat_service.rename_session(
            session_id, request.title, db,
        )
    except SessionNotFoundError:
        raise HTTPException(
            status_code=404, detail="Session not found"
        ) from None
    return chat_service.build_session_response(session)


@app.patch(
    "/api/sessions/{session_id}/interaction-mode",
    response_model=SwitchInteractionModeResponse,
)
async def switch_interaction_mode(
    session_id: int,
    request: SwitchInteractionModeRequest,
    db: Session = Depends(get_db),
):
    """Switch a session between receive_teaching and corrective."""
    try:
        session, event = await chat_service.switch_interaction_mode(
            session_id=session_id,
            interaction_mode=request.interaction_mode,
            db=db,
        )
    except SessionNotFoundError:
        raise HTTPException(
            status_code=404, detail="Session not found"
        ) from None

    return SwitchInteractionModeResponse(
        session=chat_service.build_session_response(session),
        switch_event=(
            _build_mode_switch_event_response(event)
            if event is not None
            else None
        ),
    )


@app.patch(
    "/api/sessions/{session_id}/llm-profile",
    response_model=SessionResponse,
)
async def switch_session_profile(
    session_id: int,
    request: SwitchSessionProfileRequest,
    db: Session = Depends(get_db),  # noqa: B008 — FastAPI dependency pattern
):
    """Re-bind a session to another LLM profile.

    Waits on the same per-session lock as send / delete / rename —
    there is no "busy" response.  Switching never calls any LLM.
    """
    try:
        session = await chat_service.switch_session_profile(
            session_id=session_id,
            profile_id=request.llm_profile_id,
            acknowledge_remote_history=request.acknowledge_remote_history,
            db=db,
        )
    except SessionNotFoundError:
        raise HTTPException(
            status_code=404, detail="Session not found"
        ) from None
    except UnknownLLMProfileError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc
    except SessionProfileSwitchAckRequiredError as exc:
        # Structured, machine-readable 409 — callers match on the
        # stable code, never on the message text.
        raise HTTPException(
            status_code=409,
            detail=RemoteHistoryAckRequiredDetail(
                code=exc.code,
                message=str(exc),
            ).model_dump(),
        ) from exc
    return chat_service.build_session_response(session)


# ---------------------------------------------------------------------------
# History review API helpers
# ---------------------------------------------------------------------------

_HISTORY_REVIEW_SAFE_MESSAGES = {
    "history_review_session_not_found": "Session not found.",
    "history_review_event_not_found": "Mode switch event not found.",
    "history_review_event_not_reviewable": (
        "Mode switch event is not reviewable."
    ),
    "history_review_boundary_unavailable": (
        "History boundary is unavailable."
    ),
    "history_review_no_reviewable_history": (
        "No reviewable history is available."
    ),
    "history_review_source_message_too_large": (
        "The reviewable message is too large to process."
    ),
    "history_review_reviewer_unavailable": (
        "Reviewer profile is unavailable."
    ),
    "history_review_reviewer_conflict": (
        "Reviewer profile configuration changed."
    ),
    "history_review_not_found": "History review not found.",
    "history_review_execution_conflict": (
        "History review execution conflict."
    ),
    "history_review_internal_error": (
        "History review processing failed."
    ),
    "history_review_remote_ack_required": (
        "Remote API review requires explicit acknowledgement that the "
        "selected source messages will be sent to the API model."
    ),
}


def _history_review_error(
    status_code: int,
    code: str,
    *,
    message: str | None = None,
) -> HTTPException:
    return HTTPException(
        status_code=status_code,
        detail=HistoryReviewErrorDetail(
            code=code,
            message=message or _HISTORY_REVIEW_SAFE_MESSAGES[code],
        ).model_dump(),
    )


def _history_review_internal_error(db: Session) -> HTTPException:
    db.rollback()
    logger.exception("Unexpected history review internal error")
    return _history_review_error(
        500,
        "history_review_internal_error",
    )


def _build_history_review_summary(
    review: HistoryReview,
    *,
    findings_count: int,
) -> HistoryReviewSummaryResponse:
    return HistoryReviewSummaryResponse(
        id=review.id,
        session_id=review.session_id,
        mode_switch_event_id=review.mode_switch_event_id,
        status=review.status,
        eligible_message_count=review.eligible_message_count,
        source_message_count=review.source_message_count,
        source_from_message_id=review.source_from_message_id,
        source_through_message_id=review.source_through_message_id,
        truncated=bool(review.truncated),
        summary=review.summary,
        coverage_note=review.coverage_note,
        error_code=review.error_code,
        error_message=review.error_message,
        findings_count=findings_count,
        created_at=review.created_at,
        started_at=review.started_at,
        completed_at=review.completed_at,
        updated_at=review.updated_at,
    )


def _load_history_review_summaries(
    db: Session,
    session_id: int,
) -> list[HistoryReviewSummaryResponse]:
    db.expire_all()
    reviews = db.execute(
        select(HistoryReview)
        .where(HistoryReview.session_id == session_id)
        .order_by(
            HistoryReview.created_at.desc(),
            HistoryReview.id.desc(),
        )
    ).scalars().all()
    if not reviews:
        return []

    review_ids = [review.id for review in reviews]
    counts = dict(
        db.execute(
            select(
                HistoryReviewFinding.review_id,
                func.count(HistoryReviewFinding.id),
            )
            .where(HistoryReviewFinding.review_id.in_(review_ids))
            .group_by(HistoryReviewFinding.review_id)
        ).all()
    )
    return [
        _build_history_review_summary(
            review,
            findings_count=(
                int(counts.get(review.id, 0))
                if review.status == "completed"
                else 0
            ),
        )
        for review in reviews
    ]


def _load_history_review_detail(
    db: Session,
    session_id: int,
    review_id: int,
) -> HistoryReviewDetailResponse:
    db.expire_all()
    review = db.execute(
        select(HistoryReview)
        .where(
            HistoryReview.id == review_id,
            HistoryReview.session_id == session_id,
        )
    ).scalars().first()
    if review is None:
        raise _history_review_error(404, "history_review_not_found")

    finding_rows = []
    if review.status == "completed":
        finding_rows = db.execute(
            select(HistoryReviewFinding)
            .where(HistoryReviewFinding.review_id == review.id)
            .order_by(HistoryReviewFinding.seq.asc())
        ).scalars().all()
    findings = [
        HistoryReviewFindingResponse(
            seq=row.seq,
            source_message_id=row.source_message_id,
            verdict=row.verdict,
            claim_text=row.claim_text,
            correction_text=row.correction_text,
            explanation_text=row.explanation_text,
        )
        for row in finding_rows
    ]

    summary = _build_history_review_summary(
        review,
        findings_count=len(findings),
    )
    return HistoryReviewDetailResponse(
        **summary.model_dump(),
        findings=findings,
    )


# ---------------------------------------------------------------------------
# History review routes
# ---------------------------------------------------------------------------


@app.post(
    "/api/sessions/{session_id}/history-reviews",
    response_model=HistoryReviewDetailResponse,
    status_code=201,
    responses={
        200: {
            "model": HistoryReviewDetailResponse,
            "description": "Existing history review reused.",
        },
        404: {
            "model": HistoryReviewErrorResponse,
            "description": "Session, event, or review not found.",
        },
        409: {
            "model": (
                HistoryReviewErrorResponse
                | HistoryReviewAckRequiredResponse
            ),
            "description": (
                "Remote acknowledgement or execution conflict."
            ),
        },
        422: {
            "description": (
                "Request validation error or non-reviewable input."
            ),
            "content": {
                "application/json": {
                    "schema": {
                        "anyOf": [
                            {
                                "$ref": (
                                    "#/components/schemas/"
                                    "HistoryReviewErrorResponse"
                                )
                            },
                            {
                                "$ref": (
                                    "#/components/schemas/"
                                    "HTTPValidationError"
                                )
                            },
                        ]
                    }
                }
            },
        },
        500: {
            "model": HistoryReviewErrorResponse,
            "description": "Internal history review error.",
        },
        503: {
            "model": HistoryReviewErrorResponse,
            "description": "Reviewer profile unavailable.",
        },
    },
)
async def create_history_review(
    session_id: SessionIdPath,
    request: HistoryReviewCreateRequest,
    response: Response,
    db: Session = Depends(get_db),
):
    """Create or reuse a review and execute it synchronously."""
    try:
        preparation = await history_review_service.prepare_history_review(
            session_id=session_id,
            mode_switch_event_id=request.mode_switch_event_id,
            acknowledge_remote_history=request.acknowledge_remote_history,
            db=db,
        )
    except HistoryReviewSessionNotFound:
        raise _history_review_error(
            404, "history_review_session_not_found"
        ) from None
    except HistoryReviewEventNotFound:
        raise _history_review_error(
            404, "history_review_event_not_found"
        ) from None
    except HistoryReviewEventNotReviewable:
        raise _history_review_error(
            422, "history_review_event_not_reviewable"
        ) from None
    except HistoryReviewBoundaryUnavailable:
        raise _history_review_error(
            422, "history_review_boundary_unavailable"
        ) from None
    except HistoryReviewNoReviewableHistory:
        raise _history_review_error(
            422, "history_review_no_reviewable_history"
        ) from None
    except HistoryReviewSourceMessageTooLarge:
        raise _history_review_error(
            422, "history_review_source_message_too_large"
        ) from None
    except SessionProfileUnavailableError:
        raise _history_review_error(
            503, "history_review_reviewer_unavailable"
        ) from None
    except SessionProfileConflictError:
        raise _history_review_error(
            409, "history_review_reviewer_conflict"
        ) from None
    except HTTPException:
        raise
    except HistoryReviewRemoteAckRequired as exc:
        raise HTTPException(
            status_code=409,
            detail=HistoryReviewAckRequiredDetail(
                code="history_review_remote_ack_required",
                message=_HISTORY_REVIEW_SAFE_MESSAGES[
                    "history_review_remote_ack_required"
                ],
                source_message_count=exc.source_message_count,
                reviewer_profile_label=exc.reviewer_profile_label,
                reviewer_model=exc.reviewer_model,
                truncated=exc.truncated,
            ).model_dump(),
        ) from None
    except SQLAlchemyError:
        raise _history_review_internal_error(db) from None
    except Exception:
        raise _history_review_internal_error(db) from None

    review_id = preparation.review.id

    try:
        await history_review_service.execute_history_review(
            session_id=session_id,
            review_id=review_id,
            db=db,
        )
    except HistoryReviewExecutionNotFound:
        raise _history_review_error(
            404, "history_review_not_found"
        ) from None
    except HistoryReviewExecutionConflict:
        raise _history_review_error(
            409, "history_review_execution_conflict"
        ) from None
    except HTTPException:
        raise
    except SQLAlchemyError:
        raise _history_review_internal_error(db) from None
    except Exception:
        raise _history_review_internal_error(db) from None

    response.status_code = 201 if preparation.created else 200
    try:
        return _load_history_review_detail(db, session_id, review_id)
    except HTTPException:
        raise
    except SQLAlchemyError:
        raise _history_review_internal_error(db) from None
    except Exception:
        raise _history_review_internal_error(db) from None


@app.get(
    "/api/sessions/{session_id}/history-reviews",
    response_model=list[HistoryReviewSummaryResponse],
    responses={
        404: {
            "model": HistoryReviewErrorResponse,
            "description": "Session not found.",
        },
        500: {
            "model": HistoryReviewErrorResponse,
            "description": "Internal history review error.",
        },
    },
)
def list_history_reviews(
    session_id: SessionIdPath,
    db: Session = Depends(get_db),
):
    """Return history-review summaries for a session."""
    try:
        if db.get(ChatSession, session_id) is None:
            raise _history_review_error(
                404, "history_review_session_not_found"
            )
        return _load_history_review_summaries(db, session_id)
    except HTTPException:
        raise
    except SQLAlchemyError:
        raise _history_review_internal_error(db) from None
    except Exception:
        raise _history_review_internal_error(db) from None


@app.get(
    "/api/sessions/{session_id}/history-reviews/{review_id}",
    response_model=HistoryReviewDetailResponse,
    responses={
        404: {
            "model": HistoryReviewErrorResponse,
            "description": "History review not found for this session.",
        },
        500: {
            "model": HistoryReviewErrorResponse,
            "description": "Internal history review error.",
        },
    },
)
def get_history_review(
    session_id: SessionIdPath,
    review_id: ReviewIdPath,
    db: Session = Depends(get_db),
):
    """Return one history review scoped to its session."""
    try:
        return _load_history_review_detail(db, session_id, review_id)
    except HTTPException:
        raise
    except SQLAlchemyError:
        raise _history_review_internal_error(db) from None
    except Exception:
        raise _history_review_internal_error(db) from None


@app.get(
    "/api/sessions/{session_id}/mode-switch-events",
    response_model=list[ModeSwitchEventResponse],
    responses={
        404: {
            "model": HistoryReviewErrorResponse,
            "description": "Session not found.",
        },
        500: {
            "model": HistoryReviewErrorResponse,
            "description": "Internal history review error.",
        },
    },
)
def list_mode_switch_events(
    session_id: SessionIdPath,
    db: Session = Depends(get_db),
):
    """Return all mode-switch events for one session, newest first."""
    try:
        if db.get(ChatSession, session_id) is None:
            raise _history_review_error(
                404, "history_review_session_not_found"
            )
        events = db.execute(
            select(ModeSwitchEvent)
            .where(ModeSwitchEvent.session_id == session_id)
            .order_by(
                ModeSwitchEvent.created_at.desc(),
                ModeSwitchEvent.id.desc(),
            )
        ).scalars().all()
        return [
            _build_mode_switch_event_response(event)
            for event in events
        ]
    except HTTPException:
        raise
    except SQLAlchemyError:
        raise _history_review_internal_error(db) from None
    except Exception:
        raise _history_review_internal_error(db) from None


# ---------------------------------------------------------------------------
# Frontend
# ---------------------------------------------------------------------------


@app.get("/", include_in_schema=False)
def index():
    """Serve the chat page."""
    return FileResponse(
        FRONTEND_DIR / "index.html",
        headers={"Cache-Control": "no-store"},
    )


app.mount(
    "/static",
    StaticFiles(directory=str(FRONTEND_DIR)),
    name="static",
)
