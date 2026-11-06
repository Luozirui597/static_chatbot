"""Isolated three-condition runner for search_teaching_history."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from sqlalchemy.orm import Session, sessionmaker

from backend.chat_service import get_system_prompt_for_mode
from backend.interaction_modes import RECEIVE_TEACHING_MODE
from backend.models import Message
from experiments.tool_search_history.history import (
    build_programmatic_messages,
    build_receive_messages,
)
from experiments.tool_search_history.trigger import decide, load_window_boundary
from experiments.tool_search_history.protocol import (
    TOOL_NAME,
    TOOL_SCHEMA,
    assistant_message,
    parse_assistant_turn,
    tool_message,
    validate_result_for_call,
)
from experiments.tool_search_history.tool import ToolExecutor
from experiments.tool_search_history.types import (
    AssistantTurn,
    ExperimentError,
    ToolCall,
    ToolContext,
    ToolProtocolError,
    ToolResult,
)

PROMPT_VERSION = "tool-search-history-experiment-v0.1"
CONDITIONS = ("window", "fixed", "tool", "programmatic")
MAX_MODEL_GENERATIONS = 2
MAX_TOOL_CALLS = 1


@dataclass(frozen=True)
class Fixture:
    messages: tuple[dict[str, str], ...]
    current: str
    fixed_query: str
    expected_source_ids: tuple[int, ...]
    description: str = ""

    @classmethod
    def from_file(cls, path: str | Path) -> "Fixture":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("fixture must be an object")
        messages = payload.get("messages")
        if not isinstance(messages, list) or not all(
            isinstance(item, dict)
            and item.get("role") in {"user", "assistant"}
            and isinstance(item.get("content"), str)
            for item in messages
        ):
            raise ValueError("fixture.messages is invalid")
        expected = payload.get("expected_source_ids", [])
        if not isinstance(expected, list) or not all(
            isinstance(value, int) for value in expected
        ):
            raise ValueError("fixture.expected_source_ids is invalid")
        return cls(
            messages=tuple(
                {"role": item["role"], "content": item["content"]}
                for item in messages
            ),
            current=str(payload.get("current", "")),
            fixed_query=str(payload.get("fixed_query", "")),
            expected_source_ids=tuple(expected),
            description=str(payload.get("description", "")),
        )


@dataclass
class ConditionResult:
    condition: str
    status: str = "failed"
    answer: str | None = None
    error: str | None = None
    model_calls: int = 0
    tool_calls: int = 0
    program_retrievals: int = 0
    prompt_sha256: str = ""
    system_prompt: str = ""
    model_latency_ms: float = 0.0
    tool_latency_ms: float = 0.0
    messages: list[dict[str, Any]] = field(default_factory=list)
    tool_exchanges: list[dict[str, Any]] = field(default_factory=list)
    source_ids: list[int] = field(default_factory=list)
    visible_source_ids: list[int] = field(default_factory=list)
    injected_source_ids: list[int] = field(default_factory=list)
    retrieved_source_ids: list[int] = field(default_factory=list)
    invalid_source_ids: list[int] = field(default_factory=list)
    answer_quality: str | None = None
    answer_used_correction: bool | None = None
    latency_ms: float = 0.0
    assistant_message_id: int | None = None
    tool_status: str | None = None
    model_batches: list[list[dict[str, Any]]] = field(default_factory=list)
    model_attempts: list[dict[str, Any]] = field(default_factory=list)
    protocol_loop: bool = False
    error_stage: str | None = None
    error_code: str | None = None
    trigger_reason: str | None = None
    programmatic_status: str | None = None
    programmatic_retrieval_attempts: int = 0
    model_tool_calls: int = 0
    http_requests: int = 0
    window_overflow: bool | None = None
    hidden_teacher_exists: bool | None = None
    visible_start_id: int | None = None


class FakeToolClient:
    """Deterministic structured-response client for the first round."""

    def __init__(self, responses: Sequence[AssistantTurn]) -> None:
        self._responses = list(responses)
        self.model_calls = 0
        self.message_batches: list[list[dict[str, Any]]] = []
        self.tools_seen: list[list[dict[str, Any]]] = []
        self.last_call: dict[str, Any] | None = None

    async def generate_turn(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]],
        phase: str = "first",
        tool_choice: str = "auto",
        deadline: float | None = None,
    ) -> AssistantTurn:
        self.model_calls += 1
        self.message_batches.append(copy.deepcopy(messages))
        self.tools_seen.append(copy.deepcopy(tools))
        self.last_call = {
            "phase": phase,
            "tool_choice": tool_choice,
            "request_sent": True,
            "request_model": "fake",
            "response_model": "fake",
            "finish_reason": "stop",
            "effective_timeout_s": None,
        }
        if not self._responses:
            raise ExperimentError("fake response exhausted")
        return self._responses.pop(0)

    def take_last_call(self) -> dict[str, Any] | None:
        record = self.last_call
        self.last_call = None
        return record


def _load_instruction(asset_root: Path, name: str) -> str:
    return (asset_root / "prompts" / name).read_text(encoding="utf-8").strip()


def build_system_prompt(condition: str, asset_root: Path) -> str:
    base = get_system_prompt_for_mode(RECEIVE_TEACHING_MODE).strip()
    if condition == "window":
        return base
    if condition == "fixed":
        return base + "\n\n" + _load_instruction(asset_root, "fixed.txt")
    if condition == "tool":
        return base + "\n\n" + _load_instruction(asset_root, "tool.txt")
    if condition == "programmatic":
        return base + "\n\n" + _load_instruction(
            asset_root, "programmatic.txt"
        )
    raise ValueError(f"unknown condition: {condition}")


def seed_fixture(
    db: Session,
    *,
    fixture: Fixture,
    title: str = "tool experiment",
) -> tuple[int, int]:
    from backend.models import ChatSession

    chat = ChatSession(title=title)
    db.add(chat)
    db.flush()
    for item in fixture.messages:
        db.add(
            Message(
                session_id=chat.id,
                role=item["role"],
                content=item["content"],
                interaction_mode_snapshot=RECEIVE_TEACHING_MODE,
                prompt_version_snapshot=PROMPT_VERSION,
                llm_profile_id_snapshot="fake",
                llm_profile_kind_snapshot="fake",
                llm_model_snapshot="fake",
            )
        )
    db.flush()
    current = Message(
        session_id=chat.id,
        role="user",
        content=fixture.current,
        interaction_mode_snapshot=RECEIVE_TEACHING_MODE,
        prompt_version_snapshot=PROMPT_VERSION,
        llm_profile_id_snapshot="fake",
        llm_profile_kind_snapshot="fake",
        llm_model_snapshot="fake",
    )
    db.add(current)
    db.commit()
    db.refresh(current)
    return chat.id, current.id


_SOURCE_RE = re.compile(r"\[message\s+(\d+)\]", re.IGNORECASE)


def _tool_visible_source_ids(tool_result: ToolResult | None) -> list[int]:
    if tool_result is None or not isinstance(tool_result.content, dict):
        return []
    results = tool_result.content.get("results")
    if not isinstance(results, list):
        return []
    return [
        item["message_id"]
        for item in results
        if isinstance(item, dict) and isinstance(item.get("message_id"), int)
    ]


def extract_source_ids(answer: str) -> list[int]:
    seen: set[int] = set()
    ordered: list[int] = []
    for value in _SOURCE_RE.findall(answer):
        message_id = int(value)
        if message_id not in seen:
            seen.add(message_id)
            ordered.append(message_id)
    return ordered


def _apply_answer_quality(result: ConditionResult) -> None:
    """Classify citations separately from protocol and semantic quality."""
    valid_ids = set(result.injected_source_ids or result.visible_source_ids)
    result.invalid_source_ids = [
        source_id
        for source_id in result.source_ids
        if source_id not in valid_ids
    ]
    if result.invalid_source_ids:
        result.answer_quality = "invalid_citations"
    elif result.source_ids:
        result.answer_quality = "citations_valid"
    elif result.injected_source_ids or result.visible_source_ids:
        result.answer_quality = "missing_citations"
    else:
        # No tool evidence was returned; do not demand a citation for a
        # non-existent source.  The answer still needs the separate
        # evidence-insufficiency check.
        result.answer_quality = "no_citations_available"


async def _call_client(
    client: Any,
    messages: list[dict[str, Any]],
    *,
    tools: list[dict[str, Any]],
    phase: str,
    tool_choice: str,
    deadline: float,
    result: ConditionResult,
) -> AssistantTurn:
    """Call one client turn and always collect its attempt record."""
    try:
        return await client.generate_turn(
            messages,
            tools=tools,
            phase=phase,
            tool_choice=tool_choice,
            deadline=deadline,
        )
    finally:
        record = client.take_last_call() if hasattr(client, "take_last_call") else None
        if record is None:
            record = {
                "phase": phase,
                "tool_choice": tool_choice,
                "request_sent": False,
                "error_stage": "client",
                "error_code": "missing_attempt_record",
            }
        result.model_attempts.append(record)
        if record.get("request_sent"):
            result.model_calls += 1
            result.http_requests += 1


async def _run_tool_turn(
    *,
    db: Session,
    session_id: int,
    current_id: int,
    condition: str,
    system_prompt: str,
    client: Any,
    executor: ToolExecutor,
    result: ConditionResult,
    deadline: float,
    tool_failure_policy: str,
) -> None:
    messages = build_receive_messages(
        db,
        session_id=session_id,
        current_message_id=current_id,
        system_prompt=system_prompt,
    )
    result.messages = copy.deepcopy(messages)
    tools = [TOOL_SCHEMA] if condition == "tool" else []
    model_started = time.perf_counter()
    first = await _call_client(
        client,
        messages,
        tools=tools,
        phase="first",
        tool_choice="auto" if tools else "none",
        deadline=deadline,
        result=result,
    )
    result.model_latency_ms += (time.perf_counter() - model_started) * 1000.0
    result.model_batches = client.message_batches
    result.messages = copy.deepcopy(messages)
    if len(first.tool_calls) > 1:
        raise ToolProtocolError("multiple_tool_calls")
    if not first.tool_calls:
        content = first.content or ""
        if not content.strip():
            raise ExperimentError("empty final answer")
        result.answer = content
        result.source_ids = extract_source_ids(content)
        _apply_answer_quality(result)
        return

    if condition != "tool":
        raise ToolProtocolError("tool_not_available")
    if result.tool_calls >= MAX_TOOL_CALLS:
        raise ToolProtocolError("tool_limit")
    call = first.tool_calls[0]
    result.messages = copy.deepcopy(messages)
    messages.append(assistant_message(first))
    tool_started = time.perf_counter()
    tool_result = await executor.execute(
        call,
        ToolContext(session_id=session_id, current_message_id=current_id),
        deadline=deadline,
    )
    result.tool_latency_ms += (time.perf_counter() - tool_started) * 1000.0
    result.tool_calls += 1
    result.visible_source_ids = _tool_visible_source_ids(tool_result)
    result.injected_source_ids = list(result.visible_source_ids)
    result.retrieved_source_ids = list(result.visible_source_ids)
    result.tool_status = str(tool_result.content.get("status", "error"))
    result.tool_exchanges.append(
        {
            "source": "model",
            "assistant": assistant_message(first),
            "tool": tool_message(tool_result),
        }
    )
    try:
        validate_result_for_call(tool_result, call)
    except ToolProtocolError:
        raise
    if tool_failure_policy == "stop" and tool_result.is_error:
        result.error_stage = "tool"
        result.error_code = result.tool_status
        raise ExperimentError("tool_failed")
    messages.append(tool_message(tool_result))

    model_started = time.perf_counter()
    second = await _call_client(
        client,
        messages,
        tools=tools,
        phase="second",
        tool_choice="none",
        deadline=deadline,
        result=result,
    )
    result.model_latency_ms += (time.perf_counter() - model_started) * 1000.0
    result.model_batches = client.message_batches
    result.messages = copy.deepcopy(messages)
    if second.tool_calls:
        raise ToolProtocolError("tool_round_limit")
    content = second.content or ""
    if not content.strip():
        raise ExperimentError("empty final answer")
    result.answer = content
    result.source_ids = extract_source_ids(content)
    _apply_answer_quality(result)
    result.protocol_loop = True


async def _run_programmatic_turn(
    *,
    db: Session,
    session_id: int,
    current_id: int,
    system_prompt: str,
    client: Any,
    executor: ToolExecutor,
    result: ConditionResult,
    deadline: float,
) -> None:
    boundary = load_window_boundary(
        db, session_id=session_id, current_message_id=current_id
    )
    result.window_overflow = boundary.window_overflow
    result.hidden_teacher_exists = boundary.hidden_teacher_exists
    result.visible_start_id = boundary.visible_start_id
    result.visible_source_ids = [m.id for m in boundary.visible_teacher]

    if not boundary.known:
        result.programmatic_status = "boundary_unknown"
        result.trigger_reason = "boundary_unknown"
        result.error = "ExperimentError"
        result.error_stage = "boundary"
        result.error_code = "boundary_unknown"
        return

    decision = decide(boundary, current_content=db.get(Message, current_id).content)
    result.trigger_reason = decision.reason

    if decision.action == "clarify":
        result.programmatic_status = "clarification_required"
        result.answer = (
            "The reference is not specific enough for a safe search; "
            "please identify the teaching topic."
        )
        result.status = "completed"
        return
    if decision.action == "boundary_unknown":
        result.programmatic_status = "boundary_unknown"
        result.error = "ExperimentError"
        result.error_stage = "boundary"
        result.error_code = "boundary_unknown"
        return

    hidden_result = None
    if decision.action == "retrieve":
        result.programmatic_retrieval_attempts = 1
        call = ToolCall(
            id="programmatic_call_1",
            name=TOOL_NAME,
            arguments={"query": decision.query or "", "limit": 3},
            raw_arguments="",
        )
        tool_started = time.perf_counter()
        tool_result = await executor.execute(
            call,
            ToolContext(session_id=session_id, current_message_id=current_id),
            deadline=deadline,
            before_id=boundary.visible_start_id,
        )
        result.tool_latency_ms += (time.perf_counter() - tool_started) * 1000.0
        result.tool_calls += 1
        result.tool_status = str(tool_result.content.get("status", "error"))
        result.tool_exchanges.append(
            {
                "source": "programmatic",
                "program_generated": True,
                "call": call,
                "tool": tool_result,
            }
        )
        result.retrieved_source_ids = _tool_visible_source_ids(tool_result)
        if tool_result.is_error:
            result.programmatic_status = "tool_failed"
            result.error = "ToolExecutionError"
            result.error_stage = "tool"
            result.error_code = result.tool_status
            return
        hidden_result = tool_result.content

    messages = build_programmatic_messages(
        db,
        session_id=session_id,
        current_message_id=current_id,
        system_prompt=system_prompt,
        boundary=boundary,
        hidden_result=hidden_result,
        query=decision.query or "",
    )
    result.messages = copy.deepcopy(messages)
    result.programmatic_status = "requesting_model"
    try:
        turn = await _call_client(
            client,
            messages,
            tools=[],
            phase="programmatic",
            tool_choice="none",
            deadline=deadline,
            result=result,
        )
    except Exception:
        if result.model_attempts and result.model_attempts[-1].get(
            "request_sent"
        ):
            result.injected_source_ids = list(
                dict.fromkeys(
                    list(result.visible_source_ids)
                    + list(result.retrieved_source_ids)
                )
            )
        result.programmatic_status = "protocol_failed"
        raise
    if result.model_attempts and result.model_attempts[-1].get("request_sent"):
        result.injected_source_ids = list(
            dict.fromkeys(
                list(result.visible_source_ids)
                + list(result.retrieved_source_ids)
            )
        )
    result.model_batches = client.message_batches
    if turn.tool_calls:
        result.programmatic_status = "protocol_failed"
        result.error = "ToolProtocolError"
        result.error_stage = "protocol"
        result.error_code = "programmatic_returned_tool"
        raise ToolProtocolError("programmatic_returned_tool")
    content = turn.content or ""
    if not content.strip():
        result.programmatic_status = "protocol_failed"
        result.error = "ExperimentError"
        result.error_stage = "final"
        result.error_code = "empty_final"
        return
    result.answer = content
    result.source_ids = extract_source_ids(content)
    _apply_answer_quality(result)
    result.programmatic_status = "completed"


async def run_condition(
    *,
    session_factory: sessionmaker,
    database_path: str | Path,
    asset_root: str | Path,
    condition: str,
    fixture: Fixture,
    client: Any,
    executor: ToolExecutor | None = None,
    tool_failure_policy: str = "continue",
    deadline_seconds: float = 65.0,
) -> ConditionResult:
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition: {condition}")
    result = ConditionResult(condition=condition)
    started = time.perf_counter()
    deadline = time.monotonic() + deadline_seconds
    asset_root = Path(asset_root)
    system_prompt = build_system_prompt(condition, asset_root)
    result.prompt_sha256 = hashlib.sha256(
        system_prompt.encode("utf-8")
    ).hexdigest()
    result.system_prompt = system_prompt
    if condition == "window" and fixture.fixed_query:
        pass
    db = session_factory()
    try:
        session_id, current_id = seed_fixture(db, fixture=fixture)
        if executor is None:
            executor = ToolExecutor(database_path)
        if condition == "programmatic":
            await _run_programmatic_turn(
                db=db,
                session_id=session_id,
                current_id=current_id,
                system_prompt=system_prompt,
                client=client,
                executor=executor,
                result=result,
                deadline=deadline,
            )
        elif condition != "fixed":
            await _run_tool_turn(
                db=db,
                session_id=session_id,
                current_id=current_id,
                condition=condition,
                system_prompt=system_prompt,
                client=client,
                executor=executor,
                result=result,
                deadline=deadline,
                tool_failure_policy=tool_failure_policy,
            )
        else:
            fixed_call = ToolCall(
                id="fixed_call_1",
                name=TOOL_NAME,
                arguments={"query": fixture.fixed_query, "limit": 3},
                raw_arguments="",
            )
            tool_started = time.perf_counter()
            tool_result = await executor.execute(
                fixed_call,
                ToolContext(
                    session_id=session_id,
                    current_message_id=current_id,
                ),
            )
            result.tool_latency_ms += (time.perf_counter() - tool_started) * 1000.0
            result.program_retrievals = 1
            result.visible_source_ids = _tool_visible_source_ids(tool_result)
            result.injected_source_ids = list(result.visible_source_ids)
            result.retrieved_source_ids = list(result.visible_source_ids)
            result.tool_status = str(tool_result.content.get("status", "error"))
            validate_result_for_call(tool_result, fixed_call)
            result.tool_exchanges.append(
                {
                    "source": "fixture_preload",
                    "program_generated": True,
                    "assistant": assistant_message(
                        AssistantTurn(content=None, tool_calls=(fixed_call,))
                    ),
                    "tool": tool_message(tool_result),
                }
            )
            messages = build_receive_messages(
                db,
                session_id=session_id,
                current_message_id=current_id,
                system_prompt=system_prompt,
            )
            messages = (
                messages[:-1]
                + [result.tool_exchanges[0]["assistant"]]
                + [result.tool_exchanges[0]["tool"]]
                + messages[-1:]
            )
            model_started = time.perf_counter()
            turn = await _call_client(
                client,
                messages,
                tools=[],
                phase="first",
                tool_choice="none",
                deadline=deadline,
                result=result,
            )
            result.model_latency_ms += (time.perf_counter() - model_started) * 1000.0
            result.model_batches = client.message_batches
            if turn.tool_calls:
                raise ToolProtocolError("tool_not_available")
            content = turn.content or ""
            if not content.strip():
                raise ExperimentError("empty final answer")
            result.answer = content
            result.source_ids = extract_source_ids(content)
            _apply_answer_quality(result)
            result.messages = messages

        if result.answer is None:
            raise ExperimentError("no final answer")
        assistant = Message(
            session_id=session_id,
            role="assistant",
            content=result.answer,
            interaction_mode_snapshot=RECEIVE_TEACHING_MODE,
            prompt_version_snapshot=PROMPT_VERSION,
            llm_profile_id_snapshot="fake",
            llm_profile_kind_snapshot="fake",
            llm_model_snapshot="fake",
        )
        db.add(assistant)
        db.commit()
        db.refresh(assistant)
        result.assistant_message_id = assistant.id
        result.status = "completed"
    except Exception as exc:
        result.status = "failed"
        result.error = type(exc).__name__
        if result.error_stage is None and result.model_attempts:
            last = result.model_attempts[-1]
            result.error_stage = last.get("error_stage")
            result.error_code = last.get("error_code")
        try:
            db.rollback()
        except Exception:
            pass
    finally:
        db.close()
        result.latency_ms = (time.perf_counter() - started) * 1000.0
    return result
