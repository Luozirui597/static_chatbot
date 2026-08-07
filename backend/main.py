"""FastAPI application entry point."""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.chat_service import ChatService, SessionNotFoundError
from backend.database import create_tables, engine, get_db, run_migrations
from backend.exceptions import LLMError
from backend.llm_client import create_llm_client
from backend.models import ChatSession, Message
from backend.schemas import (
    ChatRequest,
    ChatResponse,
    DeleteResponse,
    MessageResponse,
    RenameSessionRequest,
    SendMessageResponse,
    SessionResponse,
)

# ---------------------------------------------------------------------------
# Paths — resolved relative to this file so the app works regardless of CWD
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"

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

chat_service = ChatService(llm_client=create_llm_client())

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
# Session routes
# ---------------------------------------------------------------------------


@app.post(
    "/api/sessions",
    response_model=SessionResponse,
    status_code=201,
)
def create_session(db: Session = Depends(get_db)):
    """Create a new chat session with the default title."""
    session = ChatSession()
    db.add(session)
    db.commit()
    db.refresh(session)
    return session


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
    return db.execute(stmt).scalars().all()


@app.get(
    "/api/sessions/{session_id}",
    response_model=SessionResponse,
)
def get_session(session_id: int, db: Session = Depends(get_db)):
    """Return a single session by id."""
    session = db.get(ChatSession, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


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
        return await chat_service.rename_session(
            session_id, request.title, db,
        )
    except SessionNotFoundError:
        raise HTTPException(
            status_code=404, detail="Session not found"
        ) from None


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
