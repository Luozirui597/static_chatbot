"""LLM client protocol, fake implementation, and OpenAI-compatible client."""

from __future__ import annotations

from typing import Literal, Protocol, TypedDict

import httpx

from backend import config
from backend.exceptions import LLMError


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

    async def generate(self, messages: list[LLMMessage]) -> str: ...


# ---------------------------------------------------------------------------
# Fake client
# ---------------------------------------------------------------------------


class FakeLLMClient:
    """Returns a fixed test reply without touching the network.

    >>> client = FakeLLMClient()
    >>> await client.generate([{"role": "user", "content": "你好"}])
    '测试回复：你好'
    """

    async def generate(self, messages: list[LLMMessage]) -> str:
        """Return a test reply based on the last user message."""
        for msg in reversed(messages):
            if msg["role"] == "user":
                return f"测试回复：{msg['content']}"
        return "测试回复："


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

    async def generate(self, messages: list[LLMMessage]) -> str:
        """Send *messages* to the upstream API and return the reply."""
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

        client_kwargs: dict = {"timeout": self._timeout}
        if self._transport is not None:
            client_kwargs["transport"] = self._transport

        try:
            async with httpx.AsyncClient(**client_kwargs) as client:
                response = await client.post(url, headers=headers, json=payload)
        except httpx.TimeoutException:
            raise LLMError(
                detail="Upstream API timed out",
                status_code=504,
            ) from None
        except httpx.RequestError:
            raise LLMError(
                detail="Unable to reach upstream API",
                status_code=502,
            ) from None

        # -- non-2xx -------------------------------------------------------
        if not (200 <= response.status_code < 300):
            raise LLMError(
                detail=f"Upstream API returned {response.status_code}",
                status_code=502,
            )

        # -- parse body ----------------------------------------------------
        try:
            body = response.json()
        except ValueError:
            raise LLMError(
                detail="Invalid response from upstream API",
                status_code=502,
            ) from None

        try:
            content = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            raise LLMError(
                detail="Invalid response from upstream API",
                status_code=502,
            ) from None

        # -- validate content ----------------------------------------------
        if not isinstance(content, str):
            raise LLMError(
                detail="Invalid response from upstream API",
                status_code=502,
            )
        if not content.strip():
            raise LLMError(
                detail="Invalid response from upstream API",
                status_code=502,
            )

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
