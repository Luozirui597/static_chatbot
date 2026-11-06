"""Opt-in real HTTP transport for the isolated general-search experiment.

This module never reads dotenv or the production application configuration.
Credentials must be supplied explicitly by the runner or by an explicitly
named process environment variable.  It is intentionally outside production
code and is suitable for MockTransport tests without a provider.
"""

from __future__ import annotations

import asyncio
import contextvars
import time
from dataclasses import dataclass
from typing import Any

import httpx

from experiments.general_tool_search.engine import DecisionClient, FinalClient


class ProviderError(RuntimeError):
    """Base provider failure; messages remain deliberately generic."""

    def __init__(self, status: str, *, http_status: int | None = None) -> None:
        super().__init__(status)
        self.status = status
        self.http_status = http_status


class ProviderTimeout(ProviderError, asyncio.TimeoutError):
    pass


class BudgetExhausted(ProviderError, asyncio.TimeoutError):
    pass


@dataclass(frozen=True)
class RealAPIConfig:
    api_key: str
    base_url: str
    model: str
    request_timeout_seconds: float = 30.0
    temperature: float = 0.0
    max_tokens: int = 800
    decision_json_mode: bool = True

    def __post_init__(self) -> None:
        if not self.api_key or not self.base_url or not self.model:
            raise ValueError("real API requires key, base URL, and model")
        if self.request_timeout_seconds <= 0:
            raise ValueError("request_timeout_seconds must be positive")

    @property
    def endpoint(self) -> str:
        return self.base_url.rstrip("/") + "/chat/completions"


@dataclass
class TransportEvent:
    """Safe evidence only; never stores prompts, headers, or response bodies."""

    stage: str
    attempt: int
    http_sent: bool
    status: str
    elapsed_ms: float
    request_model: str
    upstream_model: str | None
    http_status: int | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "attempt": self.attempt,
            "http_sent": self.http_sent,
            "status": self.status,
            "elapsed_ms": round(self.elapsed_ms, 3),
            "request_model": self.request_model,
            "upstream_model": self.upstream_model,
            "http_status": self.http_status,
        }


class _TrackingTransport(httpx.AsyncBaseTransport):
    """Mark entry at the transport boundary, independent of client prep."""

    def __init__(
        self,
        owner: "RealAPIClient",
        inner: httpx.AsyncBaseTransport,
    ) -> None:
        self._owner = owner
        self._inner = inner

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        state = self._owner._transport_entry.get()
        if state is not None:
            state["entered"] = True
        return await self._inner.handle_async_request(request)

    async def aclose(self) -> None:
        await self._inner.aclose()


class RealAPIClient(DecisionClient, FinalClient):
    """One HTTP attempt per phase; no automatic retries."""

    synthetic = False

    def __init__(
        self,
        config: RealAPIConfig,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.config = config
        self.events: list[TransportEvent] = []
        self._attempts = {"decision": 0, "final": 0}
        self._transport_entry: contextvars.ContextVar[
            dict[str, bool] | None
        ] = contextvars.ContextVar("general_tool_transport_entry", default=None)
        inner_transport = transport or httpx.AsyncHTTPTransport()
        self._tracked_transport = _TrackingTransport(self, inner_transport)
        self._client = httpx.AsyncClient(
            timeout=config.request_timeout_seconds,
            transport=self._tracked_transport,
        )

    async def __aenter__(self) -> "RealAPIClient":
        return self

    async def __aexit__(self, _exc_type, _exc, _tb) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._client.aclose()

    async def generate_decision(
        self, messages: list[dict[str, Any]], *, deadline: float
    ) -> str:
        return await self._post(
            "decision",
            messages,
            deadline,
            json_mode=self.config.decision_json_mode,
        )

    async def generate_final(
        self, messages: list[dict[str, Any]], *, deadline: float | None = None
    ) -> str:
        return await self._post("final", messages, deadline, json_mode=False)

    async def _post(
        self,
        stage: str,
        messages: list[dict[str, Any]],
        deadline: float | None,
        *,
        json_mode: bool,
    ) -> str:
        self._attempts[stage] += 1
        attempt = self._attempts[stage]
        started = time.monotonic()

        def event(
            *,
            http_sent: bool,
            status: str,
            upstream_model: str | None = None,
            http_status: int | None = None,
        ) -> None:
            self.events.append(
                TransportEvent(
                    stage=stage,
                    attempt=attempt,
                    http_sent=http_sent,
                    status=status,
                    elapsed_ms=(time.monotonic() - started) * 1000.0,
                    request_model=self.config.model,
                    upstream_model=upstream_model,
                    http_status=http_status,
                )
            )

        if deadline is not None and time.monotonic() >= deadline:
            event(http_sent=False, status="budget_exhausted")
            raise BudgetExhausted("budget_exhausted")

        remaining = self.config.request_timeout_seconds
        if deadline is not None:
            remaining = min(remaining, max(0.001, deadline - time.monotonic()))
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        if json_mode and stage == "decision":
            payload["response_format"] = {"type": "json_object"}

        entry_state: dict[str, bool] = {"entered": False}
        token = self._transport_entry.set(entry_state)
        try:
            try:
                response = await self._client.post(
                    self.config.endpoint,
                    headers={"Authorization": f"Bearer {self.config.api_key}"},
                    json=payload,
                    timeout=remaining,
                )
            except (httpx.TimeoutException, asyncio.TimeoutError):
                entered = entry_state["entered"]
                event(
                    http_sent=entered,
                    status="timeout" if entered else "prepare_error",
                )
                if entered:
                    raise ProviderTimeout("timeout") from None
                raise ProviderError("prepare_error") from None
            except asyncio.CancelledError:
                event(
                    http_sent=entry_state["entered"],
                    status="cancelled",
                )
                raise
            except httpx.HTTPError:
                entered = entry_state["entered"]
                event(
                    http_sent=entered,
                    status="transport_error" if entered else "prepare_error",
                )
                if entered:
                    raise ProviderError("transport_error") from None
                raise ProviderError("prepare_error") from None
            except Exception:
                entered = entry_state["entered"]
                event(
                    http_sent=entered,
                    status="transport_error" if entered else "prepare_error",
                )
                if entered:
                    raise ProviderError("transport_error") from None
                raise ProviderError("prepare_error") from None
        finally:
            self._transport_entry.reset(token)

        if response.status_code < 200 or response.status_code >= 300:
            event(
                http_sent=True,
                status="http_error",
                http_status=response.status_code,
            )
            raise ProviderError("http_error", http_status=response.status_code)

        if deadline is not None and time.monotonic() > deadline:
            event(
                http_sent=True,
                status="late_after_response",
                http_status=response.status_code,
            )
            raise ProviderTimeout("late_after_response")

        try:
            data = response.json()
            choice = data["choices"][0]
            message = choice["message"]
            finish_reason = choice.get("finish_reason")
            if message.get("tool_calls"):
                event(
                    http_sent=True,
                    status="unexpected_tool_calls",
                    http_status=response.status_code,
                )
                raise ProviderError("unexpected_tool_calls")
            if finish_reason is None:
                event(
                    http_sent=True,
                    status="finish_reason_missing",
                    http_status=response.status_code,
                )
                raise ProviderError("finish_reason_missing")
            if finish_reason == "length":
                event(
                    http_sent=True,
                    status="finish_reason_length",
                    http_status=response.status_code,
                )
                raise ProviderError("finish_reason_length")
            if finish_reason == "tool_calls":
                event(
                    http_sent=True,
                    status="finish_reason_tool_calls",
                    http_status=response.status_code,
                )
                raise ProviderError("finish_reason_tool_calls")
            if finish_reason != "stop":
                event(
                    http_sent=True,
                    status="finish_reason_unsupported",
                    http_status=response.status_code,
                )
                raise ProviderError("finish_reason_unsupported")
            content = message["content"]
            if not isinstance(content, str):
                raise TypeError("content")
            upstream_model = data.get("model")
            if upstream_model is not None and not isinstance(upstream_model, str):
                upstream_model = None
        except ProviderError:
            raise
        except (KeyError, IndexError, TypeError, ValueError):
            event(
                http_sent=True,
                status="invalid_response",
                http_status=response.status_code,
            )
            raise ProviderError("invalid_response") from None

        if deadline is not None and time.monotonic() > deadline:
            event(
                http_sent=True,
                status="late_after_parse",
                upstream_model=upstream_model,
                http_status=response.status_code,
            )
            raise ProviderTimeout("late_after_parse")

        event(
            http_sent=True,
            status="ok",
            upstream_model=upstream_model,
            http_status=response.status_code,
        )
        return content
