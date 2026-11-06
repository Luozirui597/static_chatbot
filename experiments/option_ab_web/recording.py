"""Request-local evidence recording for the Option A+B web harness."""

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

from experiments.option_ab_web.state import ArmedTurn, BatchState


class RecordingError(RuntimeError):
    """A safe test-harness recording error."""


@dataclass
class RequestContext:
    request_id: str
    session_id: int
    armed: ArmedTurn
    state: BatchState
    calls: list[str] = field(default_factory=list)
    call_results: dict[str, str] = field(default_factory=dict)
    claimed: bool = False
    _claim_lock: threading.Lock = field(
        default_factory=threading.Lock, repr=False, compare=False
    )

    def model_status(self) -> str:
        if not self.calls:
            return "missing"
        if len(self.calls) > 1:
            return "duplicate"
        status = self.call_results.get(self.calls[0])
        if status in {
            "completed",
            "truncated",
            "empty",
            "cancelled",
            "error",
            "unconfirmed",
        }:
            return status
        return "unknown"

    def claim_model_call(self) -> None:
        with self._claim_lock:
            if self.claimed:
                reason = "duplicate model call in one request; batch stopped"
                self.state.fail(reason)
                raise RecordingError(reason)
            self.claimed = True


_REQUEST_CONTEXT: ContextVar[RequestContext | None] = ContextVar(
    "option_ab_web_request_context", default=None
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


def _safe_diagnostics(diagnostics: Any) -> dict[str, Any] | None:
    if diagnostics is None:
        return None
    safe = _jsonable(diagnostics)
    return safe if isinstance(safe, dict) else None


class RunRecorder:
    """Append-only evidence sink; never writes into the repository."""

    def __init__(self, *, run_dir: Path, evidence_dir: Path) -> None:
        self.run_dir = run_dir
        self.evidence_dir = evidence_dir
        self.evidence_dir.mkdir(parents=True, exist_ok=True)
        self.calls: list[dict[str, Any]] = []
        self.http: list[dict[str, Any]] = []
        self.snapshots: list[dict[str, Any]] = []
        self.branches: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._handles = {
            "calls": (self.evidence_dir / "calls.jsonl").open("x", encoding="utf-8"),
            "http": (self.evidence_dir / "http.jsonl").open("x", encoding="utf-8"),
            "snapshots": (self.evidence_dir / "snapshots.jsonl").open(
                "x", encoding="utf-8"
            ),
            "branches": (self.evidence_dir / "branches.jsonl").open(
                "x", encoding="utf-8"
            ),
        }

    def _append(self, name: str, record: dict[str, Any]) -> None:
        payload = _jsonable(record)
        line = json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False)
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
            record["script_id"] = context.armed.script_id
            record["turn_id"] = context.armed.turn_id
            if context.armed.branch_id:
                record["branch_id"] = context.armed.branch_id
            self.calls.append(record)
            context.calls.append(call_id)
            self._append("calls", record)
            status = record.get("status")
            if isinstance(status, str):
                context.call_results[call_id] = status
            return call_id

    def record_http(self, record: dict[str, Any]) -> None:
        with self._lock:
            self.http.append(record)
            self._append("http", record)

    def record_snapshot(self, record: dict[str, Any]) -> None:
        with self._lock:
            self.snapshots.append(record)
            self._append("snapshots", record)

    def record_branch(self, record: dict[str, Any]) -> None:
        with self._lock:
            self.branches.append(record)
            self._append("branches", record)

    def close(self) -> None:
        for handle in self._handles.values():
            handle.close()


class RecordingLLMClient:
    """Instrument the real client without exposing secrets or wire bodies."""

    def __init__(
        self,
        inner: Any,
        *,
        recorder: RunRecorder,
        state: BatchState,
        system_prompt: str,
        system_sha256: str,
        prompt_version: str,
        profile_id: str,
        request_model: str,
        client_kind: str,
        synthetic: bool,
        timeout_seconds: float = 30.0,
        temperature: float = 0.0,
        max_tokens: int = 800,
    ) -> None:
        self._inner = inner
        self._recorder = recorder
        self._state = state
        self._system_prompt = system_prompt
        self._system_sha256 = system_sha256
        self._prompt_version = prompt_version
        self._profile_id = profile_id
        self._request_model = request_model
        self._client_kind = client_kind
        self._synthetic = synthetic
        self._timeout_seconds = timeout_seconds
        self._temperature = temperature
        self._max_tokens = max_tokens
        self.kind = client_kind

    def _record(self, record: dict[str, Any], *, suppress_error: bool = False) -> None:
        try:
            self._recorder.record_call(record)
        except Exception:
            self._state.fail("evidence write failed; batch stopped")
            if suppress_error:
                return
            raise RecordingError("evidence write failed; batch stopped") from None

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
        from backend.review_diagnostics import LLMCallDiagnostics

        safe_messages = _safe_messages(messages)
        if not safe_messages or safe_messages[0].get("role") != "system":
            self._state.fail("system prompt missing; batch stopped")
            raise RecordingError("system prompt missing")
        if safe_messages[0].get("content") != self._system_prompt:
            self._state.fail("system prompt hash mismatch; batch stopped")
            raise RecordingError("system prompt mismatch")
        if hashlib.sha256(
            safe_messages[0]["content"].encode("utf-8")
        ).hexdigest() != self._system_sha256:
            self._state.fail("system prompt hash mismatch; batch stopped")
            raise RecordingError("system prompt hash mismatch")

        started = time.monotonic()
        diag = diagnostics if diagnostics is not None else LLMCallDiagnostics()
        record: dict[str, Any] = {
            "kind": "model_call",
            "status": "started",
            "client": {"kind": self._client_kind, "synthetic": self._synthetic},
            "profile_id": self._profile_id,
            "prompt_version": self._prompt_version,
            "system_sha256": self._system_sha256,
            "request_model": self._request_model,
            "upstream_response_model": None,
            "upstream_model_unavailable_reason": (
                "the production OpenAI-compatible client does not expose "
                "the upstream response model"
            ),
            "request_messages": safe_messages,
            "request_messages_sha256": hashlib.sha256(
                json.dumps(safe_messages, ensure_ascii=False, sort_keys=True).encode()
            ).hexdigest(),
            "original_chat_service_parameters": {
                "temperature": temperature,
                "max_tokens": max_tokens,
                "response_format": _jsonable(response_format),
                "timeout": None,
                "reasoning_effort": None,
            },
            "effective_parameters": {
                "temperature": self._temperature,
                "max_tokens": self._max_tokens,
                "timeout_seconds": self._timeout_seconds,
                "timeout_source": "wrapper deadline; inner transport timeout is set by client constructor",
                "reasoning_effort": None,
                "reasoning_effort_sent": False,
            },
            "synthetic": self._synthetic,
            "turn": {
                "instance_id": context.armed.instance_id,
                "session_id": context.session_id,
                "script_id": context.armed.script_id,
                "turn_id": context.armed.turn_id,
                "branch_id": context.armed.branch_id,
                "content_sha256": context.armed.content_sha256,
            },
        }
        try:
            raw = await asyncio.wait_for(
                self._inner.generate(
                    messages,
                    response_format=response_format,
                    temperature=self._temperature,
                    max_tokens=self._max_tokens,
                    diagnostics=diag,
                ),
                timeout=self._timeout_seconds,
            )
        except asyncio.CancelledError:
            self._state.cancel()
            self._record(
                {
                    **record,
                    "status": "cancelled",
                    "elapsed_ms": int((time.monotonic() - started) * 1000),
                    "error_type": "CancelledError",
                    "diagnostics": _safe_diagnostics(diag),
                },
                suppress_error=True,
            )
            raise
        except Exception as exc:
            self._state.fail(f"model call failed: {type(exc).__name__}")
            self._record(
                {
                    **record,
                    "status": "error",
                    "elapsed_ms": int((time.monotonic() - started) * 1000),
                    "error_type": type(exc).__name__,
                    "diagnostics": _safe_diagnostics(diag),
                },
                suppress_error=True,
            )
            raise
        elapsed_ms = int((time.monotonic() - started) * 1000)
        if not isinstance(raw, str):
            self._state.fail("model returned non-text; batch stopped")
            self._record(
                {
                    **record,
                    "status": "error",
                    "response": None,
                    "response_sha256": None,
                    "elapsed_ms": elapsed_ms,
                    "error_type": "NonTextResponse",
                    "diagnostics": _safe_diagnostics(diag),
                },
                suppress_error=True,
            )
            raise LLMError("model returned non-text")
        finish = getattr(diag, "finish_reason", None)
        response_hash = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        if not raw.strip():
            status = "empty"
            self._state.fail("empty response; batch stopped")
        elif finish == "length":
            status = "truncated"
            self._state.fail("finish_reason=length; batch stopped")
        elif finish == "stop":
            status = "completed"
        elif finish in (None, ""):
            status = "unconfirmed"
            self._state.fail("finish_reason missing; batch stopped")
        else:
            status = "unconfirmed"
            self._state.fail("non-normal finish_reason; batch stopped")
        self._record(
            {
                **record,
                "status": status,
                "response": raw,
                "response_sha256": response_hash,
                "elapsed_ms": elapsed_ms,
                "diagnostics": _safe_diagnostics(diag),
                "finish_reason": finish,
            },
            suppress_error=status != "completed",
        )
        if status == "empty":
            raise LLMError("empty model response")
        if status != "completed":
            raise LLMError(f"model completion not confirmed: {status}")
        return raw
