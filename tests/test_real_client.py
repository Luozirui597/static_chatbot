"""Unit tests for OpenAICompatibleLLMClient.

Every test uses ``httpx.MockTransport`` — no real network requests are
ever made.
"""

import json

import httpx
import pytest

from backend.exceptions import LLMError
from backend.llm_client import (
    FakeLLMClient,
    LLMMessage,
    OpenAICompatibleLLMClient,
)

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

        # Per-call options and reasoning_effort must NOT be present when
        # not explicitly set.
        assert "reasoning_effort" not in body
        assert "response_format" not in body
        assert "temperature" not in body
        assert "max_tokens" not in body

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


# ---------------------------------------------------------------------------
# Per-call generation options
# ---------------------------------------------------------------------------


def _payload_capture_client(**kwargs):
    captured: list[dict] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "ok"}}],
        })

    client = _make_client(
        transport=httpx.MockTransport(handler),
        **kwargs,
    )
    return client, captured


@pytest.mark.anyio
async def test_generate_options_are_sent_for_this_call():
    client, captured = _payload_capture_client()

    result = await client.generate(
        [_user_msg("hi")],
        response_format={"type": "json_object"},
        temperature=0,
        max_tokens=1024,
    )

    assert result == "ok"
    assert captured == [{
        "model": TEST_MODEL,
        "messages": [_user_msg("hi")],
        "response_format": {"type": "json_object"},
        "temperature": 0,
        "max_tokens": 1024,
    }]


@pytest.mark.anyio
async def test_generate_options_coexist_with_reasoning_effort():
    client, captured = _payload_capture_client(reasoning_effort="low")

    await client.generate(
        [_user_msg("hi")],
        response_format={"type": "json_object"},
        temperature=0,
        max_tokens=1,
    )

    assert captured[0]["reasoning_effort"] == "low"
    assert captured[0]["response_format"] == {"type": "json_object"}
    assert captured[0]["temperature"] == 0
    assert captured[0]["max_tokens"] == 1


@pytest.mark.anyio
async def test_generate_options_do_not_leak_to_next_call():
    client, captured = _payload_capture_client()

    await client.generate(
        [_user_msg("first")],
        response_format={"type": "json_object"},
        temperature=0,
        max_tokens=1024,
    )
    await client.generate([_user_msg("second")])

    first, second = captured
    assert first["response_format"] == {"type": "json_object"}
    assert first["temperature"] == 0
    assert first["max_tokens"] == 1024
    assert "response_format" not in second
    assert "temperature" not in second
    assert "max_tokens" not in second


@pytest.mark.anyio
async def test_generate_option_boundary_values_are_accepted():
    client, captured = _payload_capture_client()

    await client.generate(
        [_user_msg("hi")],
        temperature=2,
        max_tokens=1,
    )

    assert captured[0]["temperature"] == 2
    assert captured[0]["max_tokens"] == 1


@pytest.mark.anyio
async def test_fake_client_accepts_generate_options():
    client = FakeLLMClient()

    result = await client.generate(
        [_user_msg("hello")],
        response_format={"type": "json_object"},
        temperature=0,
        max_tokens=1024,
    )

    assert result == "测试回复：hello"


# ---------------------------------------------------------------------------
# Per-call option validation (must happen before network access)
# ---------------------------------------------------------------------------


def _client_with_forbidden_network() -> OpenAICompatibleLLMClient:
    async def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("network request must not be made")

    return _make_client(transport=httpx.MockTransport(handler))


@pytest.mark.anyio
@pytest.mark.parametrize(
    "response_format",
    ["json_object", ["json_object"], 123, object()],
)
async def test_response_format_must_be_dict_or_none(response_format):
    client = _client_with_forbidden_network()
    with pytest.raises(ValueError):
        await client.generate(
            [_user_msg("hi")],
            response_format=response_format,
        )


@pytest.mark.anyio
async def test_none_options_are_accepted_as_default():
    client, captured = _payload_capture_client()

    await client.generate(
        [_user_msg("hi")],
        response_format=None,
        temperature=None,
        max_tokens=None,
    )

    body = captured[0]
    assert "response_format" not in body
    assert "temperature" not in body
    assert "max_tokens" not in body


@pytest.mark.anyio
@pytest.mark.parametrize(
    "temperature",
    [
        True,
        False,
        "0",
        [],
        float("nan"),
        float("inf"),
        float("-inf"),
        -0.1,
        2.1,
        10**1000,
        -(10**1000),
    ],
)
async def test_temperature_validation_rejects_invalid_values(temperature):
    client = _client_with_forbidden_network()
    with pytest.raises(ValueError):
        await client.generate(
            [_user_msg("hi")],
            temperature=temperature,
        )


@pytest.mark.anyio
@pytest.mark.parametrize(
    "max_tokens",
    [
        True,
        1.0,
        1.5,
        0,
        -1,
        "1",
        [],
    ],
)
async def test_max_tokens_validation_rejects_invalid_values(max_tokens):
    client = _client_with_forbidden_network()
    with pytest.raises(ValueError):
        await client.generate(
            [_user_msg("hi")],
            max_tokens=max_tokens,
        )
