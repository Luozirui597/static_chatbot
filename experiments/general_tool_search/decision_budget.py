"""Isolated A/B/C decision-budget experiment.

This module only builds and executes the decision stage.  It never runs the
retrieval or final-answer stage, and it never changes production code.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import math
import os
import re
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import httpx
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from experiments.general_tool_search.decision_protocol_v1 import (
    DECISION_DATA_NOTE,
    DECISION_SYSTEM,
    DecisionError,
    parse_decision,
)
from backend.models import Base, ChatSession, Message
from backend.programmatic_history.trigger import decide, load_window_boundary

ASSET_ROOT = Path(__file__).resolve().parent
REPO_ROOT = ASSET_ROOT.parents[1]
DEFAULT_CASES = ASSET_ROOT / "fixtures" / "decision_budget_cases.json"
SYNTHETIC_DECISION_TIMEOUT_SECONDS = 3.0
REAL_DEFAULT_DECISION_TIMEOUT_SECONDS = 12.0
REQUEST_TEMPERATURE = 0.0
RESPONSE_FORMAT = {"type": "json_object"}
GROUPS = {
    "A": {"max_tokens": 800, "thinking": None},
    "B": {"max_tokens": 800, "thinking": {"type": "disabled"}},
    "C": {"max_tokens": 2048, "thinking": None},
}
GROUP_ORDER = ("A", "B", "C")
REPEATS = (1, 2)
REQUIRED_CASE_IDS = (
    "session51_original",
    "visible_complete",
    "hidden_correction",
    "ambiguous_reference",
)
SIMPLE_RE = re.compile(r"^\s*(?:ok|okay|understood|got it|thanks)[.!]?\s*$", re.I)
TEACHING_RE = re.compile(r"^\s*i teach you\b", re.I)
_V1_HISTORY_CUE = re.compile(
    r"\b(original|originally|initially|before|previous|earlier|corrected|"
    r"correction|revoked|revoke|source|used to|did i teach you|have i taught)\b|"
    r"(原来|最初|之前|后来改|更正|撤销|原始出处|你教过)",
    re.I,
)
_V1_AMBIGUOUS_REF = re.compile(
    r"那条|这条|那个|这个|它|它们|the\s+(?:one|former|latter)\b",
    re.I,
)


class BudgetError(RuntimeError):
    """Experiment precondition or orchestration failure."""


class BudgetTimeout(BudgetError):
    pass


@dataclass(frozen=True)
class CaseSpec:
    case_id: str
    messages: tuple[dict[str, Any], ...]
    current: str
    expected: dict[str, Any]
    source: str
    requires_external_input: bool = False
    synthetic: bool = True
    current_message_id: int | None = None


@dataclass(frozen=True)
class PlanItem:
    case_id: str
    group: str
    repeat: int
    item_id: str


@dataclass
class BudgetResponse:
    status: str
    http_status: int | None = None
    finish_reason: str | None = None
    content: str = ""
    content_available: bool = False
    upstream_model: str | None = None
    completion_tokens: int | None = None
    reasoning_tokens: int | None = None
    transport_started: bool = False
    elapsed_ms: float = 0.0
    error_code: str | None = None
    request_sha256: str = ""
    synthetic: bool = False


@dataclass
class Attempt:
    item_id: str
    case_id: str
    group: str
    repeat: int
    request_model: str
    upstream_model: str | None
    max_tokens: int
    thinking: dict[str, Any] | None
    temperature: float
    timeout_seconds: float
    response_format: dict[str, Any]
    request_sha256: str
    transport_started: bool
    http_status: int | None
    finish_reason: str | None
    truncated: bool
    content_empty: bool
    no_response: bool
    http_error: bool
    output_valid: bool
    decision: dict[str, Any] | None
    decision_error: str | None
    route_correct: bool | None
    completion_tokens: int | None
    reasoning_tokens: int | None
    elapsed_ms: float
    error_code: str | None
    synthetic: bool
    case_source: str
    visible_source_ids: list[int] = field(default_factory=list)


@dataclass
class CaseContext:
    spec: CaseSpec
    session_id: int
    current_id: int
    visible_ids: tuple[int, ...]
    label_ids: dict[str, int]
    boundary: Any


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _validate_external_session51(payload: dict[str, Any]) -> None:
    current_id = payload.get("current_message_id")
    if type(current_id) is not int or current_id != 750:
        raise BudgetError("external_current_id_must_be_750")
    current = payload.get("current")
    if not isinstance(current, str) or not current.strip():
        raise BudgetError("external_current_text_empty")
    expected = payload.get("expected")
    if not isinstance(expected, dict) or expected.get("manual") is not True:
        raise BudgetError("missing_manual_scoring_criteria")
    if not isinstance(expected.get("route_criteria"), str) or not expected["route_criteria"].strip():
        raise BudgetError("missing_route_criteria")
    messages = payload.get("messages")
    if not isinstance(messages, list):
        raise BudgetError("external_messages_missing")
    previous_id = 0
    seen: set[int] = set()
    for item in messages:
        if not isinstance(item, dict):
            raise BudgetError("external_message_invalid")
        if item.get("role") not in {"user", "assistant"} or not isinstance(item.get("content"), str):
            raise BudgetError("external_message_invalid")
        message_id = item.get("message_id")
        if type(message_id) is not int or message_id <= 0:
            raise BudgetError("external_message_id_invalid")
        if message_id in seen:
            raise BudgetError("external_message_id_duplicate")
        if message_id <= previous_id:
            raise BudgetError("external_message_order_invalid")
        if message_id >= current_id:
            raise BudgetError("external_message_not_before_current")
        seen.add(message_id)
        previous_id = message_id


def load_cases(
    path: str | Path = DEFAULT_CASES,
    *,
    external_session51: str | Path | None = None,
    allow_synthetic_session51: bool,
) -> list[CaseSpec]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    raw_cases = data.get("cases")
    if not isinstance(raw_cases, list):
        raise BudgetError("invalid_case_file")
    external: dict[str, Any] | None = None
    if external_session51 is not None:
        external = json.loads(Path(external_session51).read_text(encoding="utf-8"))
        if not isinstance(external, dict) or external.get("case_id") != "session51_original":
            raise BudgetError("invalid_external_session51")
    result: list[CaseSpec] = []
    for raw in raw_cases:
        if not isinstance(raw, dict) or not isinstance(raw.get("id"), str):
            raise BudgetError("invalid_case")
        case_id = raw["id"]
        is_fixed_external = case_id == "session51_original"
        if is_fixed_external:
            if external is not None:
                payload = {
                    **raw,
                    **external,
                    "id": case_id,
                    "requires_external_input": False,
                    "source": "external",
                    "synthetic": False,
                }
                source = "external"
                synthetic = False
                _validate_external_session51(payload)
            elif not allow_synthetic_session51:
                raise BudgetError("missing_session51_external_input")
            else:
                payload = {**raw, **raw.get("synthetic_fallback", {})}
                source = "synthetic"
                synthetic = True
        else:
            payload = raw
            source = str(raw.get("source", "synthetic"))
            synthetic = bool(raw.get("synthetic", True))
        messages = payload.get("messages")
        current = payload.get("current")
        expected = payload.get("expected", {})
        current_message_id = payload.get("current_message_id")
        if current_message_id is not None and (
            not isinstance(current_message_id, int) or isinstance(current_message_id, bool)
        ):
            raise BudgetError(f"invalid_current_message_id:{case_id}")
        if not isinstance(messages, list) or not isinstance(current, str):
            raise BudgetError(f"invalid_case:{case_id}")
        if not all(
            isinstance(item, dict)
            and item.get("role") in {"user", "assistant"}
            and isinstance(item.get("content"), str)
            for item in messages
        ):
            raise BudgetError(f"invalid_case_messages:{case_id}")
        result.append(
            CaseSpec(
                case_id=case_id,
                messages=tuple(dict(item) for item in messages),
                current=current,
                expected=dict(expected) if isinstance(expected, dict) else {},
                source=source,
                requires_external_input=is_fixed_external and external is None,
                synthetic=synthetic,
                current_message_id=current_message_id,
            )
        )
    ids = [case.case_id for case in result]
    if len(ids) != len(set(ids)) or set(ids) != set(REQUIRED_CASE_IDS):
        raise BudgetError("required_cases_mismatch")
    return result


def build_plan(cases: list[CaseSpec]) -> list[PlanItem]:
    plan = []
    for case in cases:
        for repeat in REPEATS:
            for group in GROUP_ORDER:
                plan.append(
                    PlanItem(
                        case_id=case.case_id,
                        group=group,
                        repeat=repeat,
                        item_id=f"{case.case_id}:{group}:{repeat}",
                    )
                )
    if len(plan) != 24:
        raise BudgetError("unexpected_plan_size")
    return plan


def seed_cases(cases: list[CaseSpec], database_path: str | Path) -> dict[str, CaseContext]:
    path = Path(database_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(f"sqlite:///{path}")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    db: Session = factory()
    contexts: dict[str, CaseContext] = {}
    try:
        for case in cases:
            session = ChatSession(title=f"decision-budget-{case.case_id}")
            db.add(session)
            db.flush()
            label_ids: dict[str, int] = {}
            for item in case.messages:
                message = Message(
                    id=item.get("message_id") if isinstance(item.get("message_id"), int) else None,
                    session_id=session.id,
                    role=item["role"],
                    content=item["content"],
                    interaction_mode_snapshot="receive_teaching",
                    prompt_version_snapshot="test",
                    llm_profile_id_snapshot="fake",
                    llm_profile_kind_snapshot="fake",
                    llm_model_snapshot="fake",
                )
                db.add(message)
                db.flush()
                if item.get("label"):
                    label_ids[str(item["label"])] = message.id
            current = Message(
                id=case.current_message_id,
                session_id=session.id,
                role="user",
                content=case.current,
                interaction_mode_snapshot="receive_teaching",
                prompt_version_snapshot="test",
                llm_profile_id_snapshot="fake",
                llm_profile_kind_snapshot="fake",
                llm_model_snapshot="fake",
            )
            db.add(current)
            db.commit()
            db.refresh(current)
            boundary = load_window_boundary(
                db, session_id=session.id, current_message_id=current.id
            )
            if not boundary.known or not boundary.window_overflow or not boundary.hidden_teacher_exists:
                raise BudgetError(f"case_not_model_decision:{case.case_id}")
            text = " ".join(case.current.split())
            if SIMPLE_RE.fullmatch(text) or TEACHING_RE.match(text):
                raise BudgetError(f"case_not_model_decision:{case.case_id}")
            fixed = decide(boundary, current_content=text)
            if fixed.action in {"direct", "retrieve"} and "links" in text.lower():
                raise BudgetError(f"case_fixed_relation:{case.case_id}")
            visible_ids = tuple(m.id for m in boundary.visible_teacher)
            contexts[case.case_id] = CaseContext(
                spec=case,
                session_id=session.id,
                current_id=current.id,
                visible_ids=visible_ids,
                label_ids=label_ids,
                boundary=boundary,
            )
    finally:
        db.close()
        engine.dispose()
    return contexts


def build_decision_messages(context: CaseContext) -> list[dict[str, str]]:
    visible = [
        {
            "message_id": message.id,
            "order": index + 1,
            "content": message.content,
        }
        for index, message in enumerate(context.boundary.visible_teacher)
    ]
    payload = {
        "type": "decision_window",
        "note": DECISION_DATA_NOTE,
        "visible_teacher": visible,
    }
    return [
        {"role": "system", "content": DECISION_SYSTEM},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        {"role": "user", "content": context.spec.current},
    ]


class _ObservedTransport(httpx.AsyncBaseTransport):
    def __init__(self, inner: httpx.AsyncBaseTransport, state: dict[str, Any]):
        self._inner = inner
        self._state = state

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self._state["started"] = True
        return await self._inner.handle_async_request(request)

    async def aclose(self) -> None:
        await self._inner.aclose()


class DecisionHTTPClient:
    """Minimal OpenAI-compatible client for the decision stage only."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        timeout_seconds: float = SYNTHETIC_DECISION_TIMEOUT_SECONDS,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key or ""
        self.transport = transport
        self.timeout_seconds = timeout_seconds

    async def call(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int,
        thinking: dict[str, Any] | None,
        deadline: float,
        case_id: str = "",
    ) -> BudgetResponse:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": REQUEST_TEMPERATURE,
            "max_tokens": max_tokens,
            "response_format": RESPONSE_FORMAT,
        }
        if thinking is not None:
            payload["thinking"] = thinking
        request_sha = _sha256(_canonical(payload))
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return BudgetResponse(
                status="timeout",
                error_code="pre_send_deadline",
                request_sha256=request_sha,
            )
        state: dict[str, Any] = {"started": False}
        inner = self.transport or httpx.AsyncHTTPTransport()
        observed = _ObservedTransport(inner, state)
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        started = time.perf_counter()
        try:
            async with httpx.AsyncClient(
                transport=observed,
                timeout=httpx.Timeout(max(0.001, min(remaining, self.timeout_seconds))),
            ) as client:
                try:
                    response = await asyncio.wait_for(
                        client.post(
                            f"{self.base_url}/chat/completions",
                            headers=headers,
                            json=payload,
                        ),
                        timeout=max(0.001, min(remaining, self.timeout_seconds)),
                    )
                except (asyncio.TimeoutError, httpx.TimeoutException):
                    return BudgetResponse(
                        status="timeout",
                        transport_started=bool(state["started"]),
                        elapsed_ms=(time.perf_counter() - started) * 1000,
                        error_code="decision_timeout",
                        request_sha256=request_sha,
                    )
                except httpx.HTTPError as exc:
                    return BudgetResponse(
                        status="transport_error",
                        transport_started=bool(state["started"]),
                        elapsed_ms=(time.perf_counter() - started) * 1000,
                        error_code=type(exc).__name__,
                        request_sha256=request_sha,
                    )
        except Exception:
            return BudgetResponse(
                status="transport_error",
                transport_started=bool(state.get("started")),
                elapsed_ms=(time.perf_counter() - started) * 1000,
                error_code="transport_exception",
                request_sha256=request_sha,
            )

        elapsed = (time.perf_counter() - started) * 1000
        if not 200 <= response.status_code < 300:
            return BudgetResponse(
                status="http_error",
                http_status=response.status_code,
                transport_started=True,
                elapsed_ms=elapsed,
                error_code="http_status",
                request_sha256=request_sha,
            )
        finish_reason: str | None = None
        completion: int | None = None
        reasoning: int | None = None
        upstream: str | None = None

        def invalid_response() -> BudgetResponse:
            return BudgetResponse(
                status="invalid_response",
                http_status=response.status_code,
                finish_reason=finish_reason,
                completion_tokens=completion,
                reasoning_tokens=reasoning,
                upstream_model=upstream,
                transport_started=True,
                elapsed_ms=elapsed,
                error_code="invalid_response",
                request_sha256=request_sha,
            )

        try:
            data = response.json()
        except (TypeError, ValueError):
            return invalid_response()
        if not isinstance(data, dict):
            return invalid_response()
        if isinstance(data.get("model"), str):
            upstream = data["model"]
        usage = data.get("usage")
        if isinstance(usage, dict):
            raw_completion = usage.get("completion_tokens")
            if isinstance(raw_completion, int) and not isinstance(raw_completion, bool):
                completion = raw_completion
            details = usage.get("completion_tokens_details")
            if isinstance(details, dict):
                raw_reasoning = details.get("reasoning_tokens")
                if isinstance(raw_reasoning, int) and not isinstance(raw_reasoning, bool):
                    reasoning = raw_reasoning

        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            return invalid_response()
        choice = choices[0]
        if not isinstance(choice, dict):
            return invalid_response()
        raw_finish = choice.get("finish_reason")
        if isinstance(raw_finish, str):
            finish_reason = raw_finish
        message = choice.get("message")
        if not isinstance(message, dict):
            return invalid_response()
        content = message.get("content")
        if not isinstance(content, str):
            return invalid_response()

        status = "ok" if finish_reason == "stop" else "finish_reason"
        if finish_reason is None:
            error = "finish_reason_missing"
        elif finish_reason == "length":
            error = "finish_reason_length"
        elif finish_reason == "stop":
            error = None
        else:
            error = "finish_reason_unsupported"
        if not content.strip():
            error = error or "empty_body"
        return BudgetResponse(
            status=status,
            http_status=response.status_code,
            finish_reason=finish_reason,
            content=content,
            content_available=True,
            upstream_model=upstream,
            completion_tokens=completion,
            reasoning_tokens=reasoning,
            transport_started=True,
            elapsed_ms=elapsed,
            error_code=error,
            request_sha256=request_sha,
        )


class FakeDecisionClient:
    """Default dry-run client: no network, synthetic responses."""

    def __init__(self, cases: list[CaseSpec]) -> None:
        self.cases = {case.case_id: case for case in cases}

    async def call(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int,
        thinking: dict[str, Any] | None,
        deadline: float,
        case_id: str = "",
    ) -> BudgetResponse:
        case = self.cases.get(case_id)
        if case is None:
            raise BudgetError("unknown_case")
        visible = _extract_visible_ids(messages)
        if case.case_id == "visible_complete":
            label = (case.expected.get("visible_labels") or [""])[0]
            source = next(
                (int(item["message_id"]) for item in visible if item.get("content") == "The return policy I teach is 30 days from delivery."),
                _first_visible_id(visible),
            )
            body = json.dumps(
                {
                    "need_history": False,
                    "query": "",
                    "visible_source_ids": [source] if source is not None else [],
                    "reason": "visible_complete",
                }
            )
        elif case.case_id == "hidden_correction":
            body = json.dumps(
                {
                    "need_history": True,
                    "query": "onboarding order",
                    "visible_source_ids": [],
                    "reason": "ordinary_recall",
                }
            )
        elif case.case_id == "ambiguous_reference":
            body = json.dumps(
                {
                    "need_history": False,
                    "query": "",
                    "visible_source_ids": [],
                    "reason": "ambiguous",
                }
            )
        else:
            body = json.dumps(
                {
                    "need_history": True,
                    "query": "difference quotient",
                    "visible_source_ids": [],
                    "reason": "ordinary_recall",
                }
            )
        return BudgetResponse(
            status="ok",
            http_status=200,
            finish_reason="stop",
            content=body,
            upstream_model="synthetic-fake",
            completion_tokens=20,
            reasoning_tokens=None,
            transport_started=False,
            elapsed_ms=0.0,
            synthetic=True,
        )


def _first_visible_id(visible: list[dict[str, Any]]) -> int | None:
    for item in visible:
        if isinstance(item.get("message_id"), int):
            return item["message_id"]
    return None


def _extract_visible_ids(messages: list[dict[str, str]]) -> list[dict[str, Any]]:
    for message in messages:
        if message.get("role") != "user":
            continue
        try:
            payload = json.loads(message.get("content", ""))
        except (TypeError, ValueError):
            continue
        if isinstance(payload, dict) and isinstance(payload.get("visible_teacher"), list):
            return payload["visible_teacher"]
    return []


def _score_route(
    decision: dict[str, Any],
    expected: dict[str, Any],
    label_ids: dict[str, int],
) -> bool | None:
    if not expected or expected.get("manual"):
        return None
    if expected.get("reason") is not None and decision.get("reason") != expected["reason"]:
        return False
    if expected.get("need_history") is not None and decision.get("need_history") is not expected["need_history"]:
        return False
    labels = expected.get("visible_labels") or []
    expected_ids = {label_ids[label] for label in labels if label in label_ids}
    if expected_ids and set(decision.get("visible_source_ids") or []) != expected_ids:
        return False
    terms = expected.get("query_terms") or []
    query = str(decision.get("query") or "").lower()
    if terms and not all(str(term).lower() in query for term in terms):
        return False
    return True


class _DecisionProbeReached(Exception):
    pass


class _FinalProbeReached(Exception):
    pass


class _DecisionProbe:
    synthetic = True

    def __init__(self) -> None:
        self.called = False
        self.messages: list[dict[str, Any]] | None = None

    async def generate_decision(self, messages, *, deadline):
        self.called = True
        self.messages = messages
        raise _DecisionProbeReached()


class _FinalProbe:
    synthetic = True

    def __init__(self) -> None:
        self.called = False

    async def generate_final(self, messages, *, deadline=None):
        self.called = True
        raise _FinalProbeReached()


async def probe_cases(
    cases: list[CaseSpec],
    *,
    database_path: str | Path,
) -> dict[str, list[dict[str, Any]]]:
    """Offline v1 preflight: prove the frozen cases reach a decision request.

    This intentionally does not instantiate the current production engine or
    parser.  The v1 request construction is frozen independently.
    """
    contexts = seed_cases(cases, database_path)
    observed: dict[str, list[dict[str, Any]]] = {}
    for case in cases:
        context = contexts[case.case_id]
        boundary = context.boundary
        if not boundary.known or not boundary.hidden_teacher_exists:
            raise BudgetError(f"case_not_model_decision:{case.case_id}")
        current = case.current
        if SIMPLE_RE.match(current) or TEACHING_RE.match(current):
            raise BudgetError(f"case_not_model_decision:{case.case_id}")
        fixed = decide(boundary, current_content=current)
        if fixed.action in {"direct", "retrieve"} and "links" in current.lower():
            raise BudgetError(f"case_fixed_relation:{case.case_id}")
        if _V1_HISTORY_CUE.search(current) and not _V1_AMBIGUOUS_REF.search(current):
            raise BudgetError(f"case_explicit_v1_history:{case.case_id}")
        expected_messages = build_decision_messages(context)
        if not expected_messages or expected_messages[0]["role"] != "system":
            raise BudgetError(f"decision_request_mismatch:{case.case_id}")
        observed[case.case_id] = expected_messages
    return observed

async def run_experiment(
    *,
    cases: list[CaseSpec],
    client: Any,
    output_dir: str | Path,
    model: str,
    synthetic: bool,
    case_source_path: str | Path = DEFAULT_CASES,
    decision_timeout: float = SYNTHETIC_DECISION_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    out = Path(output_dir).resolve()
    if not synthetic and out.is_relative_to(REPO_ROOT.resolve()):
        raise BudgetError("real_evidence_must_be_outside_repository")
    if out.exists():
        raise BudgetError("evidence_directory_exists")
    out.mkdir(parents=True, exist_ok=False)
    plan = build_plan(cases)
    attempts: list[Attempt] = []
    temp_database = tempfile.TemporaryDirectory(prefix="decision-budget-db-")
    db_path = Path(temp_database.name) / "decision_budget.sqlite3"
    try:
        await probe_cases(
            cases,
            database_path=Path(temp_database.name) / "probe.sqlite3",
        )
        contexts = seed_cases(cases, db_path)
    except Exception:
        temp_database.cleanup()
        raise
    decision_prompt_sha = _sha256(DECISION_SYSTEM)
    case_sha = _sha256(Path(case_source_path).read_text(encoding="utf-8"))
    meta = {
        "version": "decision-budget-v1",
        "synthetic": synthetic,
        "model": model,
        "decision_prompt_sha256": decision_prompt_sha,
        "cases_sha256": case_sha,
        "plan_size": len(plan),
        "groups": {key: dict(value) for key, value in GROUPS.items()},
        "timeout_seconds": decision_timeout,
        "case_sources": {case.case_id: case.source for case in cases},
        "case_hashes": {
            case.case_id: _sha256(
                _canonical(
                    {
                        "messages": case.messages,
                        "current": case.current,
                        "current_message_id": case.current_message_id,
                        "expected": case.expected,
                    }
                )
            )
            for case in cases
        },
    }
    (out / "run_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n")
    try:
        with (out / "attempts.jsonl").open("x", encoding="utf-8") as stream:
            for item in plan:
                context = contexts[item.case_id]
                messages = build_decision_messages(context)
                group = GROUPS[item.group]
                deadline = time.monotonic() + decision_timeout
                try:
                    response = await client.call(
                        messages,
                        max_tokens=group["max_tokens"],
                        thinking=group["thinking"],
                        deadline=deadline,
                        case_id=item.case_id,
                    )
                except Exception as exc:  # keep the run going, record only a safe class
                    response = BudgetResponse(
                        status="client_error",
                        transport_started=False,
                        error_code=type(exc).__name__,
                        synthetic=synthetic,
                    )
                decision: dict[str, Any] | None = None
                decision_error: str | None = None
                output_valid = False
                if (
                    response.status == "ok"
                    and response.finish_reason == "stop"
                    and response.content.strip()
                ):
                    try:
                        parsed = parse_decision(
                            response.content,
                            visible_ids=context.visible_ids,
                        )
                        decision = asdict(parsed.decision)
                        output_valid = True
                    except DecisionError as exc:
                        decision_error = str(exc)
                elif response.status == "ok":
                    decision_error = (
                        "empty_body"
                        if response.content_available and not response.content.strip()
                        else "empty_or_non_stop"
                    )
                else:
                    decision_error = response.error_code or response.status
                route_correct = (
                    _score_route(decision, context.spec.expected, context.label_ids)
                    if output_valid and decision is not None
                    else None
                )
                attempt = Attempt(
                    item_id=item.item_id,
                    case_id=item.case_id,
                    group=item.group,
                    repeat=item.repeat,
                    request_model=model,
                    upstream_model=response.upstream_model,
                    max_tokens=group["max_tokens"],
                    thinking=group["thinking"],
                    temperature=REQUEST_TEMPERATURE,
                    timeout_seconds=decision_timeout,
                    response_format=RESPONSE_FORMAT,
                    request_sha256=response.request_sha256,
                    transport_started=response.transport_started,
                    http_status=response.http_status,
                    finish_reason=response.finish_reason,
                    truncated=response.finish_reason == "length",
                    content_empty=(
                        response.content_available
                        and not (response.content or "").strip()
                    ),
                    no_response=response.http_status is None,
                    http_error=(
                        response.http_status is not None
                        and not 200 <= response.http_status < 300
                    ),
                    output_valid=output_valid,
                    decision=decision,
                    decision_error=decision_error,
                    route_correct=route_correct,
                    completion_tokens=response.completion_tokens,
                    reasoning_tokens=response.reasoning_tokens,
                    elapsed_ms=response.elapsed_ms,
                    error_code=response.error_code,
                    synthetic=synthetic,
                    case_source=context.spec.source,
                    visible_source_ids=list(context.visible_ids),
                )
                attempts.append(attempt)
                stream.write(json.dumps(asdict(attempt), ensure_ascii=False) + "\n")
                stream.flush()
    finally:
        temp_database.cleanup()
    summary = summarize(attempts)
    (out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    _write_csv(out / "attempts.csv", attempts)
    return summary


def summarize(attempts: list[Attempt]) -> dict[str, Any]:
    groups: dict[str, Any] = {}
    for group in GROUP_ORDER:
        rows = [item for item in attempts if item.group == group]
        valid = [item for item in rows if item.output_valid]
        route = [item for item in valid if item.route_correct is True]
        groups[group] = {
            "attempts": len(rows),
            "valid_json": len(valid),
            "route_correct": len(route),
            "route_denominator": len([item for item in valid if item.route_correct is not None]),
            "truncated": len([item for item in rows if item.finish_reason == "length"]),
            "empty": len([item for item in rows if item.content_empty]),
            "no_response": len([item for item in rows if item.no_response]),
            "http_error": len([item for item in rows if item.http_error]),
            "invalid_response": len(
                [item for item in rows if item.error_code == "invalid_response"]
            ),
            "timeout": len([item for item in rows if item.error_code == "decision_timeout" or (item.no_response and item.error_code is not None and "timeout" in item.error_code)]),
            "avg_elapsed_ms": round(sum(item.elapsed_ms for item in rows) / len(rows), 2) if rows else 0,
        }
    return {"attempts": len(attempts), "groups": groups}


def _write_csv(path: Path, attempts: list[Attempt]) -> None:
    fields = list(Attempt.__dataclass_fields__)
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for attempt in attempts:
            row = asdict(attempt)
            row["decision"] = json.dumps(row["decision"], ensure_ascii=False) if row["decision"] is not None else ""
            writer.writerow(row)


def _dry_transport(cases: list[CaseSpec]) -> httpx.MockTransport:
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        messages = payload["messages"]
        visible = _extract_visible_ids(messages)
        case_id = cases[min(calls["count"] // 6, len(cases) - 1)].case_id
        calls["count"] += 1
        if case_id == "visible_complete":
            source = _first_visible_id(visible)
            body = {"need_history": False, "query": "", "visible_source_ids": [source] if source else [], "reason": "visible_complete"}
        elif case_id == "hidden_correction":
            body = {"need_history": True, "query": "onboarding order", "visible_source_ids": [], "reason": "ordinary_recall"}
        elif case_id == "ambiguous_reference":
            body = {"need_history": False, "query": "", "visible_source_ids": [], "reason": "ambiguous"}
        else:
            body = {"need_history": True, "query": "difference quotient", "visible_source_ids": [], "reason": "ordinary_recall"}
        return httpx.Response(200, json={"model": "mock-upstream", "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(body)}}], "usage": {"completion_tokens": 10}})

    return httpx.MockTransport(handler)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Decision-budget A/B/C dry-run by default")
    parser.add_argument("--real-api", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--case-file", default=str(DEFAULT_CASES))
    parser.add_argument("--session51-input")
    parser.add_argument("--evidence-dir")
    parser.add_argument("--api-key-env", default="DECISION_BUDGET_API_KEY")
    parser.add_argument("--base-url", default="https://api.deepseek.com/v1")
    parser.add_argument("--model", default=None)
    parser.add_argument("--decision-timeout", type=float)
    args = parser.parse_args(argv)
    real = bool(args.real_api)
    if real and args.dry_run:
        parser.error("--real-api and --dry-run are mutually exclusive")
    try:
        if real:
            if not args.evidence_dir:
                parser.error("--evidence-dir is required for real runs")
            if not args.session51_input:
                parser.error("--session51-input is required for real runs")
            api_key = os.environ.get(args.api_key_env)
            if not api_key:
                print(f"missing credential environment variable: {args.api_key_env}", flush=True)
                return 2
            if not args.model or not args.model.strip():
                print("--model is required and must be non-empty for real runs", flush=True)
                return 2
            if args.decision_timeout is None:
                print("--decision-timeout is required for real runs", flush=True)
                return 2
            if not math.isfinite(args.decision_timeout) or args.decision_timeout <= 0:
                print("--decision-timeout must be a finite positive number", flush=True)
                return 2
            cases = load_cases(
                args.case_file,
                external_session51=args.session51_input,
                allow_synthetic_session51=False,
            )
            model = args.model
            timeout = args.decision_timeout
            client = DecisionHTTPClient(
                base_url=args.base_url,
                model=args.model,
                api_key=api_key,
                timeout_seconds=timeout,
            )
            out = Path(args.evidence_dir).resolve()
        else:
            cases = load_cases(args.case_file, allow_synthetic_session51=True)
            model = args.model or "deepseek-chat"
            timeout = (
                args.decision_timeout
                if args.decision_timeout is not None
                else SYNTHETIC_DECISION_TIMEOUT_SECONDS
            )
            client = DecisionHTTPClient(
                base_url="https://mock.invalid/v1",
                model=model,
                api_key="",
                transport=_dry_transport(cases),
                timeout_seconds=timeout,
            )
            out = Path(args.evidence_dir).resolve() if args.evidence_dir else None
    except BudgetError as exc:
        print(f"preflight failed: {exc}", flush=True)
        return 2
    if out is None:
        # Default dry-run executes the fake 24-item plan in a temporary
        # directory, then removes the synthetic evidence.
        with tempfile.TemporaryDirectory(prefix="decision-budget-") as temp:
            summary = asyncio.run(
                run_experiment(
                    cases=cases,
                    client=FakeDecisionClient(cases),
                    output_dir=Path(temp) / "run",
                    model=model,
                    synthetic=True,
                    case_source_path=args.case_file,
                    decision_timeout=SYNTHETIC_DECISION_TIMEOUT_SECONDS,
                )
            )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    summary = asyncio.run(
        run_experiment(
            cases=cases,
            client=client,
            output_dir=out,
            model=model,
            synthetic=not real,
            case_source_path=args.case_file,
            decision_timeout=(
                args.decision_timeout
                if args.decision_timeout is not None
                else (REAL_DEFAULT_DECISION_TIMEOUT_SECONDS if real else SYNTHETIC_DECISION_TIMEOUT_SECONDS)
            ),
        )
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
