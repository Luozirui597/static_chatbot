"""FastAPI application entry point."""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend import config
from backend.chat_service import ChatService, SessionNotFoundError
from backend.database import create_tables, engine, get_db, run_migrations
from backend.exceptions import (
    LLMError,
    SessionProfileConflictError,
    SessionProfileSwitchAckRequiredError,
    SessionProfileUnavailableError,
    UnknownLLMProfileError,
)
from backend.llm_client import create_llm_client
from backend.llm_profiles import LLMProfile, LLMProfileRegistry
from backend.models import ChatSession, Message
from backend.schemas import (
    ChatRequest,
    ChatResponse,
    CreateSessionRequest,
    DeleteResponse,
    LLMProfilePublic,
    MessageResponse,
    RemoteHistoryAckRequiredDetail,
    RenameSessionRequest,
    SendMessageResponse,
    SessionResponse,
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

chat_service = ChatService(profiles=_build_production_registry())

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
