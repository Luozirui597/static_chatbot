"""Tests for the chat endpoint.

These tests use TestClient which does not touch the network.  The
application is wired with FakeLLMClient so no external API calls are
made.
"""

import pytest
from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)


# ---------------------------------------------------------------------------
# Normal chat flow
# ---------------------------------------------------------------------------

def test_chat_normal_message():
    """A normal message returns the expected fake reply."""
    response = client.post("/api/chat", json={"message": "你好"})
    assert response.status_code == 200
    assert response.json() == {"reply": "测试回复：你好"}


# ---------------------------------------------------------------------------
# Input validation — empty / blank / too-long
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "payload",
    [
        {"message": ""},
        {"message": "   "},
        {"message": "x" * 4001},
    ],
)
def test_chat_rejects_invalid(payload):
    """Empty, whitespace-only and over-length messages are rejected."""
    response = client.post("/api/chat", json=payload)
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# Whitespace trimming
# ---------------------------------------------------------------------------

def test_chat_trims_whitespace():
    """Leading / trailing whitespace is stripped before the LLM call."""
    response = client.post("/api/chat", json={"message": "  你好  "})
    assert response.status_code == 200
    assert response.json() == {"reply": "测试回复：你好"}


# ---------------------------------------------------------------------------
# Static frontend
# ---------------------------------------------------------------------------

def test_get_root_returns_html():
    """GET / serves the chat page."""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert response.headers["cache-control"] == "no-store"


def test_get_static_app_js():
    """GET /static/app.js serves the frontend JavaScript."""
    response = client.get("/static/app.js")
    assert response.status_code == 200


# ---------------------------------------------------------------------------
# Route error conversion — LLMError → HTTPException
# ---------------------------------------------------------------------------

from backend.exceptions import LLMError  # noqa: E402


class _FakeRaisingService:
    """A ChatService stub that unconditionally raises the given error."""

    def __init__(self, error: LLMError) -> None:
        self._error = error

    async def handle_message(self, message: str) -> str:
        raise self._error


def test_chat_llm_error_502(monkeypatch):
    """LLMError with status_code=502 → HTTP 502."""
    import backend.main as main_module

    monkeypatch.setattr(
        main_module,
        "chat_service",
        _FakeRaisingService(
            LLMError("Upstream API returned 500", status_code=502)
        ),
    )

    client = TestClient(main_module.app)
    response = client.post("/api/chat", json={"message": "hi"})
    assert response.status_code == 502
    assert response.json() == {"detail": "Upstream API returned 500"}


def test_chat_llm_error_504(monkeypatch):
    """LLMError with status_code=504 → HTTP 504."""
    import backend.main as main_module

    monkeypatch.setattr(
        main_module,
        "chat_service",
        _FakeRaisingService(
            LLMError("Upstream API timed out", status_code=504)
        ),
    )

    client = TestClient(main_module.app)
    response = client.post("/api/chat", json={"message": "hi"})
    assert response.status_code == 504
    assert response.json() == {"detail": "Upstream API timed out"}


def test_chat_blank_reply_502(monkeypatch):
    """Blank LLM reply via legacy endpoint → HTTP 502."""
    from backend.exceptions import LLMInvalidResponseError

    import backend.main as main_module

    monkeypatch.setattr(
        main_module,
        "chat_service",
        _FakeRaisingService(
            LLMInvalidResponseError(
                "LLM returned an empty or blank response",
                status_code=502,
            )
        ),
    )

    client = TestClient(main_module.app)
    response = client.post("/api/chat", json={"message": "hi"})
    assert response.status_code == 502
    assert "detail" in response.json()
