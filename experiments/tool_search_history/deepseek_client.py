"""DeepSeek Chat Completions adapter for the isolated tool experiment.

This module deliberately lives outside ``backend``.  The production LLM
client returns text only and does not expose tool calls or provider metadata.
The adapter used here keeps the experiment's structured ``AssistantTurn``
contract while recording one sanitized attempt per HTTP call.
"""

from __future__ import annotations

import asyncio
import copy
import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

import httpx

from experiments.tool_search_history.protocol import parse_assistant_turn
from experiments.tool_search_history.types import (
    AssistantTurn,
    ExperimentError,
    ToolProtocolError,
)

DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-flash"
MODEL_REQUEST_TIMEOUT = 30.0
DEFAULT_MAX_TOKENS = 800
DEFAULT_TEMPERATURE = 0.0

Phase = Literal["first", "second", "programmatic"]
ToolChoice = Literal["auto", "none"]


class DeepSeekAdapterError(ExperimentError):
    """Base class for experiment adapter failures."""


class DeepSeekBudgetExhausted(DeepSeekAdapterError):
    """The turn deadline expired before a request could be issued."""


class DeepSeekTimeout(DeepSeekAdapterError):
    """An HTTP request was issued and then exceeded its effective deadline."""


class DeepSeekHTTPError(DeepSeekAdapterError):
    """The upstream returned a non-2xx response."""


class DeepSeekProtocolError(DeepSeekAdapterError):
    """The response could not be turned into a valid structured turn."""


@dataclass
class ModelAttempt:
    """Sanitized evidence for one adapter call.

    No API key, Authorization header, raw exception body, or reasoning text is
    stored here.  ``request_messages`` and ``tools`` are the synthetic
    experiment inputs needed to audit the exchange.
    """

    phase: str
    tool_choice: str
    request_sent: bool = False
    effective_timeout_s: float | None = None
    remaining_budget_ms: int | None = None
    elapsed_ms: int = 0
    request_model: str = ""
    response_model: str | None = None
    response_id: str | None = None
    http_status: int | None = None
    finish_reason: str | None = None
    tool_call_id: str | None = None
    raw_arguments: str | None = None
    parsed_arguments: dict[str, Any] | None = None
    usage: dict[str, Any] | None = None
    reasoning_present: bool = False
    error_stage: str | None = None
    error_code: str | None = None
    request_messages: list[dict[str, Any]] = field(default_factory=list)
    tools: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "tool_choice": self.tool_choice,
            "request_sent": self.request_sent,
            "effective_timeout_s": self.effective_timeout_s,
            "remaining_budget_ms": self.remaining_budget_ms,
            "elapsed_ms": self.elapsed_ms,
            "request_model": self.request_model,
            "response_model": self.response_model,
            "response_id": self.response_id,
            "http_status": self.http_status,
            "finish_reason": self.finish_reason,
            "tool_call_id": self.tool_call_id,
            "raw_arguments": self.raw_arguments,
            "parsed_arguments": self.parsed_arguments,
            "usage": self.usage,
            "reasoning_present": self.reasoning_present,
            "error_stage": self.error_stage,
            "error_code": self.error_code,
            "request_messages": self.request_messages,
            "tools": self.tools,
        }


def _remaining(deadline: float, clock: Callable[[], float]) -> float:
    return deadline - clock()


class DeepSeekToolClient:
    """Structured DeepSeek client with one tool per turn.

    The runner passes ``phase`` and ``tool_choice`` explicitly.  Phase one
    must use ``auto``; phase two must use ``none``.  Both calls receive the
    same tools definition.  The adapter never retries and never executes a
    tool itself.
    """

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_MODEL,
        transport: httpx.AsyncBaseTransport | None = None,
        timeout: float = MODEL_REQUEST_TIMEOUT,
        clock: Callable[[], float] = time.monotonic,
        thinking: str = "disabled",
    ) -> None:
        if not api_key.strip():
            raise ValueError("api_key is required")
        if thinking != "disabled":
            raise ValueError("the first adapter supports thinking='disabled' only")
        self._api_key = api_key.strip()
        self._base_url = base_url.strip().rstrip("/")
        self._model = model.strip()
        self._transport = transport
        self._timeout = float(timeout)
        self._clock = clock
        self._thinking = thinking
        self.last_call: ModelAttempt | None = None
        self.message_batches: list[list[dict[str, Any]]] = []
        self.tools_seen: list[list[dict[str, Any]]] = []

    def take_last_call(self) -> dict[str, Any] | None:
        record = self.last_call
        self.last_call = None
        return record.as_dict() if record is not None else None

    def _new_attempt(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        phase: str,
        tool_choice: str,
    ) -> ModelAttempt:
        attempt = ModelAttempt(
            phase=phase,
            tool_choice=tool_choice,
            request_model=self._model,
            request_messages=copy.deepcopy(messages),
            tools=copy.deepcopy(tools),
        )
        self.last_call = attempt
        if phase not in {"first", "second", "programmatic"}:
            attempt.error_stage = "validation"
            attempt.error_code = "invalid_phase"
            raise DeepSeekProtocolError("invalid_phase")
        if phase == "programmatic" and tools:
            attempt.error_stage = "validation"
            attempt.error_code = "programmatic_tools_not_allowed"
            raise DeepSeekProtocolError("programmatic_tools_not_allowed")
        expected_choice = "auto" if phase == "first" else "none"
        if tool_choice != expected_choice:
            attempt.error_stage = "validation"
            attempt.error_code = "invalid_tool_choice"
            raise DeepSeekProtocolError("invalid_tool_choice")
        return attempt

    def _payload(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        tool_choice: str,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "thinking": {"type": "disabled"},
            "temperature": DEFAULT_TEMPERATURE,
            "max_tokens": DEFAULT_MAX_TOKENS,
            "stream": False,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice
        return payload

    async def generate_turn(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]],
        phase: Phase,
        tool_choice: ToolChoice,
        deadline: float,
    ) -> AssistantTurn:
        # Read the same clock exactly once for this call.  The effective
        # deadline is absolute, not derived from a later clock read.
        started = self._clock()
        call_deadline = min(deadline, started + self._timeout)
        attempt = self._new_attempt(
            messages=messages,
            tools=tools,
            phase=phase,
            tool_choice=tool_choice,
        )
        payload = self._payload(
            messages=messages, tools=tools, tool_choice=tool_choice
        )
        url = f"{self._base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {self._api_key}"}
        self.message_batches.append(copy.deepcopy(messages))
        self.tools_seen.append(copy.deepcopy(tools))

        # Recompute the remaining budget immediately before dispatch, using
        # the one absolute deadline established above.
        remaining_at_send = call_deadline - self._clock()
        attempt.remaining_budget_ms = max(0, int(remaining_at_send * 1000))
        attempt.effective_timeout_s = max(0.0, remaining_at_send)
        if remaining_at_send <= 0:
            attempt.error_stage = "budget"
            attempt.error_code = "budget_exhausted"
            raise DeepSeekBudgetExhausted("budget_exhausted")

        try:
            async with asyncio.timeout(remaining_at_send):
                async with httpx.AsyncClient(
                    timeout=remaining_at_send,
                    transport=self._transport,
                ) as client:
                    attempt.request_sent = True
                    response = await client.post(url, headers=headers, json=payload)
                    status = response.status_code
                    attempt.http_status = status
                    if not (200 <= status < 300):
                        attempt.error_stage = "http"
                        attempt.error_code = f"http_{status}"
                        raise DeepSeekHTTPError(f"http_{status}")

                    def check_parse_deadline(stage: str) -> None:
                        if self._clock() >= call_deadline:
                            attempt.error_stage = "timeout"
                            attempt.error_code = stage
                            raise DeepSeekTimeout(stage)

                    # The HTTP response is available.  Check the same absolute
                    # deadline before JSON parsing, after JSON parsing, and
                    # before accepting the structured turn.
                    check_parse_deadline("deadline_before_json_parse")
                    try:
                        body = response.json()
                    except ValueError as exc:
                        attempt.error_stage = "parse"
                        attempt.error_code = "invalid_json"
                        raise DeepSeekProtocolError("invalid_json") from exc
                    attempt.elapsed_ms = max(
                        0, int((self._clock() - started) * 1000)
                    )
                    check_parse_deadline("deadline_after_json_parse")
                    try:
                        turn = self._parse_body(body, attempt, phase=phase)
                    except Exception:
                        if self._clock() >= call_deadline:
                            attempt.error_stage = "timeout"
                            attempt.error_code = "deadline_during_structure_parse"
                            raise DeepSeekTimeout(
                                "deadline_during_structure_parse"
                            ) from None
                        raise
                    check_parse_deadline("deadline_during_structure_parse")
                    return turn
        except (asyncio.TimeoutError, httpx.TimeoutException) as exc:
            attempt.error_stage = "timeout"
            attempt.error_code = "request_timeout"
            attempt.elapsed_ms = max(
                0, int((self._clock() - started) * 1000)
            )
            raise DeepSeekTimeout("request_timeout") from exc
        except (DeepSeekAdapterError, ToolProtocolError):
            attempt.elapsed_ms = max(
                0, int((self._clock() - started) * 1000)
            )
            raise
        except httpx.HTTPError as exc:
            attempt.error_stage = "http"
            attempt.error_code = "transport_error"
            attempt.elapsed_ms = max(
                0, int((self._clock() - started) * 1000)
            )
            raise DeepSeekHTTPError("transport_error") from exc

    def _parse_body(
        self,
        body: Any,
        attempt: ModelAttempt,
        *,
        phase: str,
    ) -> AssistantTurn:
        attempt.response_model = body.get("model") if isinstance(body, dict) else None
        attached = self.last_call is attempt
        try:
            if not isinstance(body, dict):
                raise DeepSeekProtocolError("body_must_be_object")
            attempt.response_id = (
                body.get("id") if isinstance(body.get("id"), str) else None
            )
            choices = body.get("choices")
            if not isinstance(choices, list) or not choices:
                raise DeepSeekProtocolError("choices_missing")
            choice = choices[0]
            if not isinstance(choice, dict):
                raise DeepSeekProtocolError("choice_must_be_object")
            attempt.finish_reason = (
                choice.get("finish_reason")
                if isinstance(choice.get("finish_reason"), str)
                else None
            )
            message = choice.get("message")
            if not isinstance(message, dict):
                raise DeepSeekProtocolError("message_missing")
            if "content" not in message:
                raise DeepSeekProtocolError("content_missing")
            content = message.get("content")
            if content is not None and not isinstance(content, str):
                raise DeepSeekProtocolError("content_must_be_text_or_null")
            attempt.reasoning_present = isinstance(
                message.get("reasoning_content"), str
            ) and bool(message.get("reasoning_content"))
            raw_calls = message.get("tool_calls")
            if raw_calls is None:
                raw_calls = []

            usage = body.get("usage")
            attempt.usage = usage if isinstance(usage, dict) else None

            if raw_calls:
                if phase in {"second", "programmatic"}:
                    raise DeepSeekProtocolError(
                        "second_request_returned_tool"
                        if phase == "second"
                        else "programmatic_returned_tool"
                    )
                if attempt.finish_reason != "tool_calls":
                    raise DeepSeekProtocolError("tool_finish_reason_mismatch")
                if not isinstance(raw_calls, list) or len(raw_calls) != 1:
                    raise DeepSeekProtocolError("multiple_tool_calls")
                turn = parse_assistant_turn(
                    content=content,
                    raw_calls=raw_calls,
                )
                call = turn.tool_calls[0]
                attempt.tool_call_id = call.id
                attempt.raw_arguments = call.raw_arguments
                attempt.parsed_arguments = dict(call.arguments)
                return turn

            if attempt.finish_reason != "stop":
                raise DeepSeekProtocolError("finish_reason_not_stop")
            if not isinstance(content, str) or not content.strip():
                raise DeepSeekProtocolError("empty_final_content")
            return AssistantTurn(content=content, tool_calls=())
        except ToolProtocolError as exc:
            attempt.error_stage = "parse"
            attempt.error_code = str(exc)
            raise DeepSeekProtocolError(str(exc)) from exc
        except DeepSeekProtocolError as exc:
            attempt.error_stage = "parse"
            attempt.error_code = str(exc)
            raise
        finally:
            if attached and self.last_call is attempt:
                pass
