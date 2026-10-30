"""LLM client protocol, fake implementation, and OpenAI-compatible client."""

from __future__ import annotations

import math
import time
from typing import Literal, Protocol, TypedDict

import httpx

from backend import config
from backend.exceptions import LLMError
from backend.review_diagnostics import (
    ErrorCategory,
    LLMCallDiagnostics,
    TokenUsage,
    clean_optional_float,
    clean_optional_int,
    clean_optional_string,
)


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


class LLMMessage(TypedDict):
    """A single message in a chat conversation."""

    role: Literal["system", "user", "assistant"]
    content: str


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


class LLMClient(Protocol):
    """Protocol for replaceable LLM backends.

    Every concrete client must implement ``generate``.  The method may
    be synchronous or asynchronous — callers should ``await`` it.
    """

    async def generate(
        self,
        messages: list[LLMMessage],
        *,
        response_format: dict[str, object] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        diagnostics: LLMCallDiagnostics | None = None,
    ) -> str: ...


# ---------------------------------------------------------------------------
# Fake client
# ---------------------------------------------------------------------------


class FakeLLMClient:
    """Returns a fixed test reply without touching the network.

    >>> client = FakeLLMClient()
    >>> await client.generate([{"role": "user", "content": "你好"}])
    '测试回复：你好'
    """

    async def generate(
        self,
        messages: list[LLMMessage],
        *,
        response_format: dict[str, object] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        diagnostics: LLMCallDiagnostics | None = None,
    ) -> str:
        """Return a test reply based on the last user message."""
        _validate_generate_options(
            response_format=response_format,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        for msg in reversed(messages):
            if msg["role"] == "user":
                reply = f"测试回复：{msg['content']}"
                if diagnostics is not None:
                    diagnostics.response_content_chars = len(reply)
                return reply
        if diagnostics is not None:
            diagnostics.response_content_chars = len("测试回复：")
        return "测试回复："


# ---------------------------------------------------------------------------
# Per-call option validation
# ---------------------------------------------------------------------------


def _validate_generate_options(
    *,
    response_format: dict[str, object] | None,
    temperature: float | None,
    max_tokens: int | None,
) -> None:
    """Validate optional per-call generation settings.

    Invalid values are rejected before any network request is made and
    are deliberately not wrapped in ``LLMError``.
    """

    if response_format is not None and not isinstance(response_format, dict):
        raise ValueError("response_format must be a dict or None")

    if temperature is not None:
        if isinstance(temperature, bool) or not isinstance(
            temperature, (int, float)
        ):
            raise ValueError("temperature must be a number or None")
        if isinstance(temperature, int):
            # Avoid math.isfinite() overflow for arbitrarily large ints.
            if temperature < 0 or temperature > 2:
                raise ValueError("temperature must be between 0 and 2")
        else:
            if not math.isfinite(temperature):
                raise ValueError("temperature must be finite")
            if temperature < 0 or temperature > 2:
                raise ValueError("temperature must be between 0 and 2")

    if max_tokens is not None:
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int):
            raise ValueError("max_tokens must be an integer or None")
        if max_tokens <= 0:
            raise ValueError("max_tokens must be greater than 0")


def _elapsed_ms(start: float) -> int:
    return max(0, int((time.perf_counter() - start) * 1000))


def _timeout_category(exc: httpx.TimeoutException) -> ErrorCategory:
    if isinstance(exc, httpx.ConnectTimeout):
        return ErrorCategory.UPSTREAM_CONNECT_TIMEOUT
    if isinstance(exc, httpx.ReadTimeout):
        return ErrorCategory.UPSTREAM_READ_TIMEOUT
    if isinstance(exc, httpx.WriteTimeout):
        return ErrorCategory.UPSTREAM_WRITE_TIMEOUT
    if isinstance(exc, httpx.PoolTimeout):
        return ErrorCategory.UPSTREAM_POOL_TIMEOUT
    return ErrorCategory.UPSTREAM_TIMEOUT_UNKNOWN


def _request_error_category(exc: httpx.RequestError) -> ErrorCategory:
    if isinstance(exc, httpx.ConnectError):
        return ErrorCategory.UPSTREAM_CONNECT_ERROR
    if isinstance(exc, httpx.ReadError):
        return ErrorCategory.UPSTREAM_READ_ERROR
    if isinstance(exc, httpx.WriteError):
        return ErrorCategory.UPSTREAM_WRITE_ERROR
    if isinstance(exc, httpx.RemoteProtocolError):
        return ErrorCategory.UPSTREAM_REMOTE_PROTOCOL_ERROR
    if isinstance(exc, httpx.ProxyError):
        return ErrorCategory.UPSTREAM_PROXY_ERROR
    if isinstance(exc, httpx.UnsupportedProtocol):
        return ErrorCategory.UPSTREAM_UNSUPPORTED_PROTOCOL
    if isinstance(exc, httpx.DecodingError):
        return ErrorCategory.UPSTREAM_TRANSPORT_DECODE_ERROR
    return ErrorCategory.UPSTREAM_TRANSPORT_ERROR_UNKNOWN


def _set_transport_error(
    diagnostics: LLMCallDiagnostics,
    category: ErrorCategory,
    exc: BaseException,
    start: float,
) -> None:
    diagnostics.error_category = category
    diagnostics.exception_type = clean_optional_string(
        type(exc).__name__,
        max_length=64,
    )
    diagnostics.call_elapsed_ms = _elapsed_ms(start)


def _http_error_category(status_code: int) -> ErrorCategory:
    if status_code == 400:
        return ErrorCategory.UPSTREAM_HTTP_400
    if status_code == 401:
        return ErrorCategory.UPSTREAM_HTTP_401
    if status_code == 402:
        return ErrorCategory.UPSTREAM_HTTP_402
    if status_code == 429:
        return ErrorCategory.UPSTREAM_HTTP_429
    if 500 <= status_code <= 599:
        return ErrorCategory.UPSTREAM_HTTP_5XX
    return ErrorCategory.UPSTREAM_HTTP_OTHER


def _header_value(response: httpx.Response, name: str) -> str | None:
    return clean_optional_string(response.headers.get(name), max_length=128)


def _upstream_request_id(response: httpx.Response) -> str | None:
    for name in ("x-request-id", "x-ds-request-id", "request-id"):
        value = _header_value(response, name)
        if value is not None:
            return value
    return None


def _retry_after_seconds(response: httpx.Response) -> float | None:
    value = response.headers.get("retry-after")
    if value is None:
        return None
    candidate = clean_optional_string(value, max_length=32)
    if candidate is None:
        return None
    try:
        parsed = float(candidate)
    except ValueError:
        return None
    return clean_optional_float(parsed, minimum=0.0)


def _token_usage(body: object) -> TokenUsage | None:
    if not isinstance(body, dict):
        return None
    usage = body.get("usage")
    if not isinstance(usage, dict):
        return None
    prompt = clean_optional_int(usage.get("prompt_tokens"), minimum=0)
    completion = clean_optional_int(
        usage.get("completion_tokens"),
        minimum=0,
    )
    total = clean_optional_int(usage.get("total_tokens"), minimum=0)
    if prompt is None and completion is None and total is None:
        return None
    return TokenUsage(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=total,
    )


def _upstream_error_metadata(body: object) -> tuple[str | None, str | None]:
    if not isinstance(body, dict):
        return None, None
    error = body.get("error")
    if not isinstance(error, dict):
        return None, None
    return (
        clean_optional_string(error.get("code"), max_length=64),
        clean_optional_string(error.get("type"), max_length=64),
    )


def _set_upstream_error_metadata(
    diagnostics: LLMCallDiagnostics,
    response: httpx.Response,
) -> None:
    try:
        body = response.json()
    except ValueError:
        return
    code, error_type = _upstream_error_metadata(body)
    diagnostics.upstream_error_code = code
    diagnostics.upstream_error_type = error_type


# ---------------------------------------------------------------------------
# OpenAI-compatible client
# ---------------------------------------------------------------------------


class OpenAICompatibleLLMClient:
    """Talks to any OpenAI-compatible ``/chat/completions`` endpoint.

    Parameters
    ----------
    api_key:
        Bearer token sent in the ``Authorization`` header.
    base_url:
        API base URL, e.g. ``https://api.openai.com/v1``.  Trailing
        slashes are stripped before ``/chat/completions`` is appended.
    model:
        Model name sent in the request body.
    timeout:
        Per-request timeout in seconds (default 30).
    transport:
        Optional custom ``httpx`` transport.  In production this is
        ``None`` (the default transport is used).  In tests an
        ``httpx.MockTransport`` can be injected.
    reasoning_effort:
        Optional reasoning-effort hint sent as ``reasoning_effort`` in
        the request payload.  An empty string (the default) omits the
        field entirely.  Accepted values after stripping and lowercasing
        are ``none``, ``low``, ``medium``, ``high``.  Any other non-empty
        value raises ``ValueError`` at construction time.
    """

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        timeout: float = 30.0,
        transport: httpx.AsyncBaseTransport | None = None,
        reasoning_effort: str = "",
    ) -> None:
        self._api_key = api_key.strip()
        self._base_url = base_url.strip().rstrip("/")
        self._model = model.strip()
        self._timeout = timeout
        self._transport = transport

        self._reasoning_effort = reasoning_effort.strip()
        if self._reasoning_effort:
            allowed = {"none", "low", "medium", "high"}
            if self._reasoning_effort.lower() not in allowed:
                raise ValueError(
                    f"LLM_REASONING_EFFORT must be one of "
                    f"{sorted(allowed)} or empty, "
                    f"got '{reasoning_effort}'"
                )
            self._reasoning_effort = self._reasoning_effort.lower()

    async def generate(
        self,
        messages: list[LLMMessage],
        *,
        response_format: dict[str, object] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        diagnostics: LLMCallDiagnostics | None = None,
    ) -> str:
        """Send *messages* to the upstream API and return the reply."""
        _validate_generate_options(
            response_format=response_format,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        diag = diagnostics if diagnostics is not None else LLMCallDiagnostics()
        diag.timeout_configured_seconds = clean_optional_float(
            self._timeout,
            minimum=0.0,
        )
        start = time.perf_counter()

        url = f"{self._base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
        }
        payload = {
            "model": self._model,
            "messages": messages,
        }
        if self._reasoning_effort:
            payload["reasoning_effort"] = self._reasoning_effort
        if response_format is not None:
            payload["response_format"] = response_format
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        client_kwargs: dict = {"timeout": self._timeout}
        if self._transport is not None:
            client_kwargs["transport"] = self._transport

        try:
            async with httpx.AsyncClient(**client_kwargs) as client:
                response = await client.post(url, headers=headers, json=payload)
        except httpx.TimeoutException as exc:
            _set_transport_error(
                diag,
                _timeout_category(exc),
                exc,
                start,
            )
            raise LLMError(
                detail="Upstream API timed out",
                status_code=504,
                diagnostics=diag,
            ) from None
        except httpx.RequestError as exc:
            _set_transport_error(
                diag,
                _request_error_category(exc),
                exc,
                start,
            )
            raise LLMError(
                detail="Unable to reach upstream API",
                status_code=502,
                diagnostics=diag,
            ) from None

        diag.http_status = clean_optional_int(
            response.status_code,
            minimum=100,
            maximum=599,
        )
        diag.upstream_request_id = _upstream_request_id(response)
        diag.retry_after_seconds = _retry_after_seconds(response)
        diag.call_elapsed_ms = _elapsed_ms(start)

        # -- non-2xx -------------------------------------------------------
        if not (200 <= response.status_code < 300):
            diag.error_category = _http_error_category(response.status_code)
            _set_upstream_error_metadata(diag, response)
            raise LLMError(
                detail=f"Upstream API returned {response.status_code}",
                status_code=502,
                diagnostics=diag,
            )

        # -- parse body ----------------------------------------------------
        try:
            body = response.json()
        except ValueError:
            diag.error_category = ErrorCategory.UPSTREAM_RESPONSE_INVALID_JSON
            raise LLMError(
                detail="Invalid response from upstream API",
                status_code=502,
                diagnostics=diag,
            ) from None

        diag.token_usage = _token_usage(body)
        diag.upstream_error_code, diag.upstream_error_type = (
            _upstream_error_metadata(body)
        )

        if not isinstance(body, dict):
            diag.error_category = ErrorCategory.UPSTREAM_RESPONSE_SHAPE_ERROR
            raise LLMError(
                detail="Invalid response from upstream API",
                status_code=502,
                diagnostics=diag,
            ) from None

        choices = body.get("choices")
        if not isinstance(choices, list) or not choices:
            diag.error_category = ErrorCategory.UPSTREAM_RESPONSE_SHAPE_ERROR
            raise LLMError(
                detail="Invalid response from upstream API",
                status_code=502,
                diagnostics=diag,
            ) from None

        first_choice = choices[0]
        if not isinstance(first_choice, dict):
            diag.error_category = ErrorCategory.UPSTREAM_RESPONSE_SHAPE_ERROR
            raise LLMError(
                detail="Invalid response from upstream API",
                status_code=502,
                diagnostics=diag,
            ) from None

        diag.finish_reason = clean_optional_string(
            first_choice.get("finish_reason"),
            max_length=32,
        )
        message = first_choice.get("message")
        if not isinstance(message, dict):
            diag.error_category = ErrorCategory.UPSTREAM_RESPONSE_SHAPE_ERROR
            raise LLMError(
                detail="Invalid response from upstream API",
                status_code=502,
                diagnostics=diag,
            ) from None

        if "content" not in message:
            diag.error_category = (
                ErrorCategory.UPSTREAM_RESPONSE_MISSING_CONTENT
            )
            raise LLMError(
                detail="Invalid response from upstream API",
                status_code=502,
                diagnostics=diag,
            ) from None

        content = message.get("content")
        if content is None:
            diag.error_category = ErrorCategory.UPSTREAM_RESPONSE_NULL_CONTENT
            raise LLMError(
                detail="Invalid response from upstream API",
                status_code=502,
                diagnostics=diag,
            ) from None

        if not isinstance(content, str):
            diag.error_category = (
                ErrorCategory.UPSTREAM_RESPONSE_NON_STRING_CONTENT
            )
            raise LLMError(
                detail="Invalid response from upstream API",
                status_code=502,
                diagnostics=diag,
            ) from None

        diag.response_content_chars = len(content)

        if not content.strip():
            if diag.finish_reason == "content_filter":
                diag.error_category = ErrorCategory.UPSTREAM_RESPONSE_FILTERED
            else:
                diag.error_category = (
                    ErrorCategory.UPSTREAM_RESPONSE_EMPTY_CONTENT
                )
            raise LLMError(
                detail="Invalid response from upstream API",
                status_code=502,
                diagnostics=diag,
            ) from None

        return content


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_llm_client(
    *,
    mode: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
    reasoning_effort: str | None = None,
) -> LLMClient:
    """Return the correct LLM client based on *mode*.

    Parameters with a value of ``None`` are read from
    ``backend.config``.  An explicit empty or whitespace-only string is
    treated as *missing* — it never falls back to the local config.
    """

    # ---- resolve: None → config, otherwise use explicit value ----------
    resolved_mode = config.LLM_MODE if mode is None else mode
    resolved_api_key = config.LLM_API_KEY if api_key is None else api_key
    resolved_base_url = config.LLM_API_BASE_URL if base_url is None else base_url
    resolved_model = config.LLM_MODEL if model is None else model
    resolved_reasoning_effort = (
        config.LLM_REASONING_EFFORT if reasoning_effort is None
        else reasoning_effort
    )

    # ---- normalize ------------------------------------------------------
    resolved_mode = resolved_mode.strip().lower()
    resolved_api_key = resolved_api_key.strip()
    resolved_base_url = resolved_base_url.strip()
    resolved_model = resolved_model.strip()

    # ---- route ----------------------------------------------------------

    if resolved_mode == "fake":
        return FakeLLMClient()

    if resolved_mode == "real":
        if not resolved_api_key:
            raise ValueError("LLM_API_KEY is required when LLM_MODE=real")
        if not resolved_base_url:
            raise ValueError("LLM_API_BASE_URL is required when LLM_MODE=real")
        if not resolved_model:
            raise ValueError("LLM_MODEL is required when LLM_MODE=real")

        return OpenAICompatibleLLMClient(
            api_key=resolved_api_key,
            base_url=resolved_base_url,
            model=resolved_model,
            reasoning_effort=resolved_reasoning_effort,
        )

    raise ValueError(
        f"Unknown LLM_MODE '{resolved_mode}'. Must be 'fake' or 'real'."
    )
