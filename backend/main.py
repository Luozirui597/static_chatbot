"""FastAPI application entry point."""

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.chat_service import ChatService
from backend.exceptions import LLMError
from backend.llm_client import create_llm_client
from backend.schemas import ChatRequest, ChatResponse

# ---------------------------------------------------------------------------
# Paths — resolved relative to this file so the app works regardless of CWD
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"

# ---------------------------------------------------------------------------
# Application & dependencies
# ---------------------------------------------------------------------------

app = FastAPI(title="Static Chatbot")

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
# Frontend
# ---------------------------------------------------------------------------


@app.get("/", include_in_schema=False)
def index():
    """Serve the chat page."""
    return FileResponse(FRONTEND_DIR / "index.html")


app.mount(
    "/static",
    StaticFiles(directory=str(FRONTEND_DIR)),
    name="static",
)
