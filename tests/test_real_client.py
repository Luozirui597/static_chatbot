"""Unit tests for OpenAICompatibleLLMClient.

Every test uses ``httpx.MockTransport`` — no real network requests are
ever made.
"""

import json

import httpx
import pytest

from backend.exceptions import LLMError
from backend.llm_client import LLMMessage, OpenAICompatibleLLMClient

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


def _user_msg(content: str) -> LLMMessage:
    return {"role": "user", "content": content}


# ---------------------------------------------------------------------------
# Normal response — also validates the outgoing request
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_normal_response():
    """A valid 200 response returns the assistant's reply."""

    input_messages: list[LLMMessage] = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "你好"},
    ]

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

        # Payload messages must exactly match what was passed — no injected
        # system prompt.
        assert body["messages"] == input_messages

        # reasoning_effort must NOT be present when not explicitly set
        assert "reasoning_effort" not in body

        return httpx.Response(200, json={
            "choices": [{"message": {"content": "你好！"}}],
        })

    client = _make_client(transport=httpx.MockTransport(handler))
    result = await client.generate(input_messages)
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
        await client.generate([_user_msg("hi")])
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
        await client.generate([_user_msg("hi")])
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
        await client.generate([_user_msg("hi")])
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
        await client.generate([_user_msg("hi")])
    assert exc_info.value.status_code == 502
    assert exc_info.value.detail == "Invalid response from upstream API"


# ---------------------------------------------------------------------------
# reasoning_effort
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_reasoning_effort_absent_when_whitespace():
    """Payload does not contain reasoning_effort when the value is whitespace-only."""
    async def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert "reasoning_effort" not in body
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "ok"}}],
        })

    client = _make_client(
        reasoning_effort="   ",
        transport=httpx.MockTransport(handler),
    )
    result = await client.generate([_user_msg("hi")])
    assert result == "ok"


@pytest.mark.anyio
async def test_reasoning_effort_none_normalized():
    """Payload contains reasoning_effort: none when value is ' none '."""
    async def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["reasoning_effort"] == "none"
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "ok"}}],
        })

    client = _make_client(
        reasoning_effort=" none ",
        transport=httpx.MockTransport(handler),
    )
    result = await client.generate([_user_msg("hi")])
    assert result == "ok"


@pytest.mark.anyio
async def test_reasoning_effort_low_normalized():
    """Payload contains reasoning_effort: low when value is ' LOW '."""
    async def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["reasoning_effort"] == "low"
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "ok"}}],
        })

    client = _make_client(
        reasoning_effort=" LOW ",
        transport=httpx.MockTransport(handler),
    )
    result = await client.generate([_user_msg("hi")])
    assert result == "ok"


def test_reasoning_effort_invalid_raises():
    """Invalid reasoning_effort raises ValueError at construction time."""
    with pytest.raises(ValueError, match="LLM_REASONING_EFFORT"):
        _make_client(reasoning_effort="invalid")


@pytest.mark.anyio
async def test_reasoning_effort_extra_fields_ignored():
    """Only content is returned when API includes extra reasoning fields."""
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "the answer",
                    "reasoning": "internal reasoning",
                },
            }],
        })

    client = _make_client(
        reasoning_effort="none",
        transport=httpx.MockTransport(handler),
    )
    result = await client.generate([_user_msg("hi")])
    assert result == "the answer"
