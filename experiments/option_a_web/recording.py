"""Request-local recording around the real ChatService LLM client."""

from __future__ import annotations

import asyncio
import hashlib
import json
import threading
import time
from contextvars import ContextVar, Token
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any

from experiments.option_a_web.gate import ArmedTurn, RunState


class RecordingError(RuntimeError):
    """A safe test-harness error."""


@dataclass
class RequestContext:
    request_id: str
    session_id: int
    arm: ArmedTurn
    run_state: RunState
    calls: list[str] = field(default_factory=list)
    failure: str | None = None
    claimed: bool = False
    _claim_lock: threading.Lock = field(
        default_factory=threading.Lock, repr=False, compare=False
    )

    def claim_model_call(self) -> None:
        """Claim the single model call allowed for this HTTP request."""

        with self._claim_lock:
            if self.claimed:
                reason = "duplicate model call in one request; batch stopped"
                self.failure = reason
                self.run_state.fail(reason)
                raise RecordingError(reason)
            self.claimed = True


_REQUEST_CONTEXT: ContextVar[RequestContext | None] = ContextVar(
    "option_a_web_request_context", default=None
)


def set_request_context(context: RequestContext) -> Token:
    return _REQUEST_CONTEXT.set(context)


def reset_request_context(token: Token) -> None:
    _REQUEST_CONTEXT.reset(token)


def current_request_context() -> RequestContext:
    context = _REQUEST_CONTEXT.get()
    if context is None:
        raise RecordingError("model call has no request-local context")
    return context


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if hasattr(value, "value"):
        return _jsonable(value.value)
    return None


def _safe_diagnostics(diagnostics: Any) -> dict[str, Any] | None:
    if diagnostics is None:
        return None
    return _jsonable(diagnostics)


def _safe_messages(messages: Any) -> list[dict[str, str]]:
    if not isinstance(messages, list):
        raise RecordingError("messages must be a list")
    safe: list[dict[str, str]] = []
    for item in messages:
        if not isinstance(item, dict):
            raise RecordingError("message is not a dictionary")
        role = item.get("role")
        content = item.get("content")
        if not isinstance(role, str) or not isinstance(content, str):
            raise RecordingError("message role/content is not text")
        safe.append({"role": role, "content": content})
    return safe


class RunRecorder:
    """Append-only evidence sink.  It never writes to the repository."""

    def __init__(self, *, run_dir: Path, evidence_dir: Path) -> None:
        self.run_dir = run_dir
        self.evidence_dir = evidence_dir
        self.evidence_dir.mkdir(parents=True, exist_ok=True)
        self.calls: list[dict[str, Any]] = []
        self.http: list[dict[str, Any]] = []
        self.snapshots: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._handles = {
            "calls": (self.evidence_dir / "calls.jsonl").open("x", encoding="utf-8"),
            "http": (self.evidence_dir / "http.jsonl").open("x", encoding="utf-8"),
            "snapshots": (self.evidence_dir / "snapshots.jsonl").open(
                "x", encoding="utf-8"
            ),
        }

    def _append(self, name: str, record: dict[str, Any]) -> None:
        payload = _jsonable(record)
        line = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
        handle = self._handles[name]
        handle.write(line + "\n")
        handle.flush()

    def record_call(self, record: dict[str, Any]) -> str:
        with self._lock:
            context = current_request_context()
            call_id = f"{context.request_id}:{len(context.calls) + 1}"
            record = dict(record)
            record["call_id"] = call_id
            record["request_id"] = context.request_id
            record["session_id"] = context.session_id
            record["script_id"] = context.arm.script_id
            record["turn_id"] = context.arm.turn_id
            self.calls.append(record)
            context.calls.append(call_id)
            self._append("calls", record)
            return call_id

    def record_http(self, record: dict[str, Any]) -> None:
        with self._lock:
            self.http.append(record)
            self._append("http", record)

    def record_snapshot(self, record: dict[str, Any]) -> None:
        with self._lock:
            self.snapshots.append(record)
            self._append("snapshots", record)

    def close(self) -> None:
        for handle in self._handles.values():
            handle.close()


class RecordingLLMClient:
    """Delegate to the real client while recording only safe metadata.

    The wrapper deliberately does not expose upstream response bodies, headers,
    or error messages.  It also enforces the frozen system prompt before any
    model call is made.
    """

    def __init__(
        self,
        inner: Any,
        *,
        recorder: RunRecorder,
        system_prompt: str,
        system_sha256: str,
        profile_id: str,
        request_model: str,
        synthetic: bool,
        timeout_seconds: float = 30.0,
        temperature: float = 0.0,
        max_tokens: int = 800,
    ) -> None:
        self._inner = inner
        self._recorder = recorder
        self._system_prompt = system_prompt
        self._system_sha256 = system_sha256
        self._profile_id = profile_id
        self._request_model = request_model
        self._synthetic = synthetic
        self._timeout_seconds = timeout_seconds
        self._temperature = temperature
        self._max_tokens = max_tokens

    def _record_call(self, record: dict[str, Any], *, suppress_error: bool = False) -> None:
        try:
            self._recorder.record_call(record)
        except Exception as exc:
            context = current_request_context()
            context.run_state.fail("evidence write failed; batch stopped")
            if suppress_error:
                return
            raise RecordingError("evidence write failed; batch stopped") from exc

    async def generate(
        self,
        messages: list[dict[str, str]],
        *,
        response_format: Any = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        diagnostics: Any = None,
    ) -> str:
        context = current_request_context()
        context.claim_model_call()
        from backend.exceptions import LLMError
        from backend.review_diagnostics import ErrorCategory, LLMCallDiagnostics

        safe_messages = _safe_messages(messages)
        call_diagnostics = LLMCallDiagnostics()
        original_parameters = {
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": _jsonable(response_format),
            "timeout_seconds": None,
            "reasoning_effort": None,
        }
        effective_parameters = {
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
            "response_format": _jsonable(response_format),
            "timeout_seconds": self._timeout_seconds,
            "reasoning_effort": None,
            "reasoning_effort_sent": False,
        }

        system_message = safe_messages[0] if safe_messages else {}
        actual_system = (
            system_message.get("content")
            if system_message.get("role") == "system"
            else None
        )
        actual_system_sha = (
            hashlib.sha256(actual_system.encode("utf-8")).hexdigest()
            if isinstance(actual_system, str)
            else None
        )
        base_record: dict[str, Any] = {
            "status": "started",
            "instance_id": context.arm.instance_id,
            "client": {
                "kind": "fake" if self._synthetic else "api",
                "synthetic": self._synthetic,
            },
            "model": {
                "profile_id": self._profile_id,
                "request_model": self._request_model,
                "upstream_response_model": None,
                "upstream_response_model_reason": (
                    "LLMClient.generate() returns only content; the OpenAI-compatible "
                    "response body is not exposed by the production client."
                ),
            },
            "parameters": {
                "original_from_chat_service": original_parameters,
                "effective_sent_by_wrapper": effective_parameters,
            },
            "request": {
                "messages": safe_messages,
                "system_sha256": actual_system_sha,
            },
            "response": None,
            "response_sha256": None,
            "diagnostics": None,
            "error_class": None,
            "error_reason": None,
        }

        if actual_system != self._system_prompt or actual_system_sha != self._system_sha256:
            reason = "system prompt does not match frozen system.txt"
            context.run_state.fail(reason)
            base_record["status"] = "error"
            base_record["error_reason"] = reason
            self._record_call(base_record, suppress_error=True)
            raise LLMError(reason, 500, diagnostics=call_diagnostics)

        base_record["status"] = "calling"
        started = time.monotonic()
        try:
            result = await asyncio.wait_for(
                self._inner.generate(
                    messages,
                    response_format=response_format,
                    temperature=self._temperature,
                    max_tokens=self._max_tokens,
                    diagnostics=call_diagnostics,
                ),
                timeout=self._timeout_seconds,
            )
        except asyncio.TimeoutError:
            call_diagnostics.error_category = ErrorCategory.UPSTREAM_TIMEOUT_UNKNOWN
            call_diagnostics.timeout_configured_seconds = self._timeout_seconds
            call_diagnostics.call_elapsed_ms = int(
                (time.monotonic() - started) * 1000
            )
            base_record["status"] = "error"
            base_record["error_class"] = "TimeoutError"
            base_record["error_reason"] = "test client timeout"
            base_record["diagnostics"] = _safe_diagnostics(call_diagnostics)
            base_record["call_elapsed_ms"] = call_diagnostics.call_elapsed_ms
            context.run_state.fail("model call timed out; batch stopped")
            self._record_call(base_record, suppress_error=True)
            raise LLMError(
                "Model call timed out in test client",
                504,
                diagnostics=call_diagnostics,
            ) from None
        except asyncio.CancelledError:
            base_record["status"] = "cancelled"
            base_record["error_class"] = "CancelledError"
            base_record["error_reason"] = "model call cancelled; batch stopped"
            base_record["diagnostics"] = _safe_diagnostics(call_diagnostics)
            base_record["call_elapsed_ms"] = int(
                (time.monotonic() - started) * 1000
            )
            context.run_state.fail("model call cancelled; batch stopped")
            self._record_call(base_record, suppress_error=True)
            raise
        except Exception as exc:
            error_diagnostics = getattr(exc, "diagnostics", None)
            payload = _safe_diagnostics(error_diagnostics or call_diagnostics)
            base_record["status"] = "error"
            base_record["error_class"] = type(exc).__name__
            base_record["diagnostics"] = payload
            base_record["call_elapsed_ms"] = int(
                (time.monotonic() - started) * 1000
            )
            context.run_state.fail("model call failed; batch stopped")
            self._record_call(base_record, suppress_error=True)
            raise

        elapsed_ms = int((time.monotonic() - started) * 1000)
        if not isinstance(result, str):
            base_record["status"] = "error"
            base_record["error_class"] = "NonStringResponse"
            base_record["error_reason"] = "client returned a non-string response"
            base_record["diagnostics"] = _safe_diagnostics(call_diagnostics)
            base_record["call_elapsed_ms"] = elapsed_ms
            context.run_state.fail("non-string model response; batch stopped")
            self._record_call(base_record, suppress_error=True)
            from backend.exceptions import LLMInvalidResponseError

            raise LLMInvalidResponseError(
                "Model returned a non-string response",
                502,
                diagnostics=call_diagnostics,
            )

        finish_reason = str(call_diagnostics.finish_reason or "").strip().lower()
        finish_classification = {
            "stop": "stop",
            "length": "truncated",
            "": "missing",
            "content_filter": "content_filter",
            "tool_calls": "tool_calls",
        }.get(finish_reason, "unknown")
        base_record["finish_reason"] = finish_reason or None
        base_record["finish_classification"] = finish_classification

        if not result.strip():
            base_record["status"] = "empty"
            base_record["response"] = result
            base_record["response_sha256"] = hashlib.sha256(
                result.encode("utf-8")
            ).hexdigest()
            base_record["response_chars"] = len(result)
            base_record["diagnostics"] = _safe_diagnostics(call_diagnostics)
            base_record["call_elapsed_ms"] = elapsed_ms
            context.run_state.fail("empty model response; batch stopped")
            self._record_call(base_record, suppress_error=True)
            # Let ChatService raise its normal empty-response error.  It will
            # not retry and no assistant row is persisted.
            return result

        if finish_reason != "stop":
            base_record["status"] = (
                "truncated" if finish_reason == "length" else "unconfirmed"
            )
            base_record["response"] = result
            base_record["response_sha256"] = hashlib.sha256(
                result.encode("utf-8")
            ).hexdigest()
            base_record["response_chars"] = len(result)
            base_record["diagnostics"] = _safe_diagnostics(call_diagnostics)
            base_record["call_elapsed_ms"] = elapsed_ms
            context.run_state.fail(
                "model completion was not confirmed; batch stopped"
            )
            self._record_call(base_record, suppress_error=True)
            raise LLMError(
                "Model completion was not confirmed",
                502,
                diagnostics=call_diagnostics,
            )

        upstream_model = getattr(self._inner, "upstream_response_model", None)
        if not isinstance(upstream_model, str) or not upstream_model:
            upstream_model = None
            upstream_reason = base_record["model"]["upstream_response_model_reason"]
        else:
            upstream_reason = None

        base_record["status"] = "completed"
        base_record["model"]["upstream_response_model"] = upstream_model
        base_record["model"]["upstream_response_model_reason"] = upstream_reason
        base_record["response"] = result
        base_record["response_sha256"] = hashlib.sha256(
            result.encode("utf-8")
        ).hexdigest()
        base_record["response_chars"] = len(result)
        base_record["diagnostics"] = _safe_diagnostics(call_diagnostics)
        base_record["call_elapsed_ms"] = elapsed_ms
        self._record_call(base_record)
        return result
