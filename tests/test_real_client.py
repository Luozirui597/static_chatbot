"""Unit tests for OpenAICompatibleLLMClient.

Every test uses ``httpx.MockTransport`` — no real network requests are
ever made.
"""

import json

import httpx
import pytest

from backend.exceptions import LLMError
from backend.llm_client import OpenAICompatibleLLMClient, SYSTEM_PROMPT

# Fixed test parameters
TEST_KEY = "test-key"
TEST_URL = "https://api.example.com/v1"
TEST_MODEL = "test-model"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_client(**kwargs) -> OpenAICompatibleLLMClient:
    defaults = {
        "api_key": TEST_KEY,
        "base_url": TEST_URL,
        "model": TEST_MODEL,
    }
    defaults.update(kwargs)
    return OpenAICompatibleLLMClient(**defaults)


# ---------------------------------------------------------------------------
# Normal response — also validates the outgoing request
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_normal_response():
    """A valid 200 response returns the assistant's reply."""

    async def handler(request: httpx.Request) -> httpx.Response:
        # --- request integrity checks ---
        assert request.method == "POST"
        assert request.url.path.endswith("/chat/completions")
        assert request.headers["authorization"] == f"Bearer {TEST_KEY}"
        assert "application/json" in request.headers["content-type"]

        body = json.loads(request.content)
        assert body["model"] == TEST_MODEL
        assert "api_key" not in body
        assert "apiKey" not in body
        assert "apikey" not in body
        msgs = body["messages"]
        assert msgs[0]["role"] == "system"
        assert msgs[0]["content"] == SYSTEM_PROMPT
        assert msgs[1]["role"] == "user"
        assert msgs[1]["content"] == "你好"

        return httpx.Response(200, json={
            "choices": [{"message": {"content": "你好！"}}],
        })

    client = _make_client(transport=httpx.MockTransport(handler))
    result = await client.generate("你好")
    assert result == "你好！"


# ---------------------------------------------------------------------------
# Timeout
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_timeout():
    """A timeout raises LLMError with status_code 504."""

    async def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)

    client = _make_client(transport=httpx.MockTransport(handler))

    with pytest.raises(LLMError) as exc_info:
        await client.generate("hi")
    assert exc_info.value.status_code == 504
    assert exc_info.value.detail == "Upstream API timed out"


# ---------------------------------------------------------------------------
# RequestError (non-timeout)
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_request_error():
    """A connection error raises LLMError with status_code 502."""

    async def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    client = _make_client(transport=httpx.MockTransport(handler))

    with pytest.raises(LLMError) as exc_info:
        await client.generate("hi")
    assert exc_info.value.status_code == 502
    assert exc_info.value.detail == "Unable to reach upstream API"


# ---------------------------------------------------------------------------
# HTTP 4xx / 5xx
# ---------------------------------------------------------------------------


@pytest.mark.anyio
@pytest.mark.parametrize(
    "http_status",
    [401, 403, 429, 500, 502, 503],
)
async def test_http_error(http_status):
    """Non-2xx upstream responses raise LLMError with status_code 502."""

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(http_status, json={"error": "some error"})

    client = _make_client(transport=httpx.MockTransport(handler))

    with pytest.raises(LLMError) as exc_info:
        await client.generate("hi")
    assert exc_info.value.status_code == 502
    assert exc_info.value.detail == f"Upstream API returned {http_status}"


# ---------------------------------------------------------------------------
# Invalid response bodies (parametrised)
# ---------------------------------------------------------------------------

INVALID_BODY_CASES = [
    pytest.param(
        "not json",  # body — raw string, not JSON
        id="invalid_json",
    ),
    pytest.param(
        {"choices": []},
        id="empty_choices",
    ),
    pytest.param(
        {"choices": [{}]},
        id="missing_message",
    ),
    pytest.param(
        {"choices": [{"message": {}}]},
        id="missing_content",
    ),
    pytest.param(
        {"choices": [{"message": {"content": None}}]},
        id="content_null",
    ),
    pytest.param(
        {"choices": [{"message": {"content": ""}}]},
        id="content_empty",
    ),
    pytest.param(
        {"choices": [{"message": {"content": "   "}}]},
        id="content_blank",
    ),
    pytest.param(
        {"choices": [{"message": {"content": 123}}]},
        id="content_not_string",
    ),
]


@pytest.mark.anyio
@pytest.mark.parametrize("body", INVALID_BODY_CASES)
async def test_invalid_response_body(body):
    """Malformed upstream responses raise LLMError 502."""

    async def handler(request: httpx.Request) -> httpx.Response:
        if isinstance(body, str):
            return httpx.Response(200, content=body.encode())
        return httpx.Response(200, json=body)

    client = _make_client(transport=httpx.MockTransport(handler))

    with pytest.raises(LLMError) as exc_info:
        await client.generate("hi")
    assert exc_info.value.status_code == 502
    assert exc_info.value.detail == "Invalid response from upstream API"
