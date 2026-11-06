"""Isolated fake-client runner for the frozen Option B experiment.

The default and only built-in client is deterministic and local.  This module
does not import a production LLM client, read environment files, or perform
network I/O.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
import sys
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence

try:
    from .freeze_manifest import FrozenAssetError, load_and_verify_manifest
    from .history import (
        MAX_HISTORY_MESSAGES,
        StoredMessage,
        build_request_messages,
        project_receive,
        recent_history,
        serialize_receive_history,
    )
except ImportError:  # direct script execution
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from freeze_manifest import (  # type: ignore[no-redef]
        FrozenAssetError,
        load_and_verify_manifest,
    )
    from history import (  # type: ignore[no-redef]
        MAX_HISTORY_MESSAGES,
        StoredMessage,
        build_request_messages,
        project_receive,
        recent_history,
        serialize_receive_history,
    )

ASSET_VERSION = "option-b-isolated-v0.1"
RUN_SCHEMA_VERSION = "option-b-run-v1"
TURN_SCHEMA_VERSION = "option-b-turn-v1"
FIXED_IDS = ("S1", "S2", "S3", "S4", "S5", "S6")
INTERACTIVE_IDS = ("I1", "I2", "I3")
FIXED_CALLS = {"S1": 2, "S2": 2, "S3": 2, "S4": 2, "S5": 4, "S6": 4}
BRANCH_OPTIONS = {
    "I2": ("none", "bc", "direct_ac", "no_extra"),
    "I3": ("none", "rule_request", "direct_ac", "no_extra"),
}
INTERACTIVE_MAX_CALLS = {"I1": 2, "I2": 3, "I3": 3}


class EmptyResponseError(RuntimeError):
    """The fake client returned an empty/blank response."""


class FakeLLMClient:
    """Deterministic local fake; it never opens a socket."""

    synthetic = True

    def __init__(self, replies: Sequence[str] | None = None) -> None:
        self._replies = None if replies is None else list(replies)
        self.calls: list[list[dict[str, str]]] = []

    async def generate(
        self,
        messages: Sequence[dict[str, str]],
        **_: Any,
    ) -> str:
        self.calls.append([dict(message) for message in messages])
        if self._replies is not None:
            if not self._replies:
                raise RuntimeError("fake reply queue exhausted")
            return self._replies.pop(0)
        return f"SYNTHETIC_OPTION_B_FAKE_REPLY_{len(self.calls)}"


class StaticReplyClient:
    synthetic = True

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.calls: list[list[dict[str, str]]] = []

    async def generate(
        self,
        messages: Sequence[dict[str, str]],
        **_: Any,
    ) -> str:
        self.calls.append([dict(message) for message in messages])
        return self.reply


class RaisingClient:
    synthetic = True

    def __init__(self, error: BaseException, fail_on_call: int = 1) -> None:
        self.error = error
        self.fail_on_call = fail_on_call
        self.calls: list[list[dict[str, str]]] = []

    async def generate(
        self,
        messages: Sequence[dict[str, str]],
        **_: Any,
    ) -> str:
        self.calls.append([dict(message) for message in messages])
        if len(self.calls) == self.fail_on_call:
            raise self.error
        return f"SYNTHETIC_OPTION_B_FAKE_REPLY_{len(self.calls)}"


class LLMClient(Protocol):
    async def generate(
        self,
        messages: Sequence[dict[str, str]],
        **kwargs: Any,
    ) -> str: ...


class TurnOrderError(RuntimeError):
    """A run tried to skip, repeat, or exceed its bounded turn sequence."""


@dataclass(frozen=True)
class TurnSpec:
    id: str
    phase: str
    content: str


@dataclass(frozen=True)
class BranchSpec:
    id: str
    label: str
    content: str
    adds: tuple[str, ...]
    direct_ac: bool
    rule_status: str


@dataclass(frozen=True)
class ScriptSpec:
    id: str
    mode: str
    title: str
    focus: str
    calls_expected: int | dict[str, int]
    turns: tuple[TurnSpec, ...]
    branches: tuple[BranchSpec, ...]

    @property
    def calls_min(self) -> int:
        if isinstance(self.calls_expected, int):
            return self.calls_expected
        return self.calls_expected["minimum"]

    @property
    def calls_max(self) -> int:
        if isinstance(self.calls_expected, int):
            return self.calls_expected
        return self.calls_expected["maximum"]


@dataclass(frozen=True)
class BranchDecision:
    id: str
    label: str
    selected_by: str = "human"
    note: str = ""


@dataclass
class RunResult:
    run_id: str
    script_id: str
    mode: str
    status: str
    calls: int
    records: list[dict[str, Any]] = field(default_factory=list)
    error: dict[str, Any] | None = None
    run_dir: str = ""
    branch_choice: str | None = None


@dataclass
class PausedInteractive:
    script: ScriptSpec
    system_prompt: str
    system_sha256: str
    prompt_version_value: str
    client: LLMClient
    recorder: RunRecorder
    storage: list[StoredMessage]
    records: list[dict[str, Any]]
    ledger: TurnLedger
    run_id: str
    started_at: str
    calls: int = 2
    state: str = "paused"
    selection_note: str | None = None
    cleanup_error: str | None = None


class RunRecorder:
    """Exclusive run directory and append-only JSONL records."""

    def __init__(self, output_root: Path, run_id: str) -> None:
        self.run_id = safe_run_id(run_id)
        self.run_dir = output_root / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self.turn_path = self.run_dir / "turns.jsonl"
        self.meta_path = self.run_dir / "run_meta.json"
        self._handle = self.turn_path.open("x", encoding="utf-8", newline="\n")
        self.closed = False

    def append(self, record: dict[str, Any]) -> None:
        if self.closed:
            raise RuntimeError("recorder is closed")
        self._handle.write(
            json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n"
        )
        self._handle.flush()

    def write_meta(self, meta: dict[str, Any]) -> None:
        with self.meta_path.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(
                json.dumps(meta, ensure_ascii=False, sort_keys=True, indent=2)
                + "\n"
            )

    def update_meta(self, meta: dict[str, Any]) -> None:
        """Update a paused run's meta without silently losing its prior state."""
        if not self.meta_path.is_file():
            raise RuntimeError("cannot update missing run_meta.json")
        paused_copy = self.run_dir / "run_meta.paused.json"
        if not paused_copy.exists():
            paused_copy.write_bytes(self.meta_path.read_bytes())
        temporary = self.meta_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(meta, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        temporary.replace(self.meta_path)

    def close(self) -> None:
        if not self.closed:
            self._handle.close()
            self.closed = True

    def __enter__(self) -> "RunRecorder":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


class TurnLedger:
    """Per-run sequence, content hash, and budget enforcement.

    A new ledger is required for every logical run.  The ledger cannot skip a
    turn, repeat a turn, change its frozen content, or exceed its max call
    count.
    """

    def __init__(
        self,
        run_id: str,
        script_id: str,
        expected_turns: Sequence[TurnSpec],
        max_calls: int,
    ) -> None:
        self.run_id = run_id
        self.script_id = script_id
        self._turns = tuple(expected_turns)
        self._by_id = {turn.id: turn for turn in self._turns}
        self._index = 0
        self._attempts = 0
        self._completed: list[str] = []
        self._in_flight: str | None = None
        self._failed = False
        self._failure: str | None = None
        self.max_calls = max_calls

    @property
    def index(self) -> int:
        return self._index

    @property
    def attempts(self) -> int:
        return self._attempts

    @property
    def failed(self) -> bool:
        return self._failed

    @property
    def failure(self) -> str | None:
        return self._failure

    @property
    def completed(self) -> tuple[str, ...]:
        return tuple(self._completed)

    @property
    def in_flight(self) -> str | None:
        return self._in_flight

    def begin(
        self,
        turn_id: str,
        content: str,
        *,
        allowed_contents: Sequence[str] | None = None,
    ) -> TurnSpec:
        if self._failed:
            raise TurnOrderError(f"run is stopped: {self._failure}")
        if self._in_flight is not None:
            raise TurnOrderError(
                f"turn {self._in_flight} is already in flight; one at a time"
            )
        if self._index >= len(self._turns):
            raise TurnOrderError("all planned turns are already complete")
        expected = self._turns[self._index]
        if turn_id != expected.id:
            raise TurnOrderError(
                f"expected {expected.id}, got {turn_id}; skipping/reordering is forbidden"
            )
        if expected.content:
            if sha256_text(content) != sha256_text(expected.content):
                raise TurnOrderError(
                    f"{turn_id}: content does not match the frozen input"
                )
        else:
            # A manual branch's T2 content is not known until after the Q1
            # pause.  It is still checked against the frozen branch text that
            # the human selected.
            if allowed_contents is None:
                raise TurnOrderError(
                    f"{turn_id}: manual branch content requires allowed_contents"
                )
            allowed_hashes = {sha256_text(item) for item in allowed_contents}
            if sha256_text(content) not in allowed_hashes:
                raise TurnOrderError(
                    f"{turn_id}: content does not match any allowed branch"
                )
        if self._attempts >= self.max_calls:
            raise TurnOrderError("model call budget exceeded")
        self._attempts += 1
        self._in_flight = expected.id
        return expected

    def commit(self) -> None:
        if self._failed:
            raise TurnOrderError("cannot commit a failed run")
        if self._in_flight is None:
            raise TurnOrderError("commit requires a turn begun but not committed")
        if self._index >= len(self._turns):
            raise TurnOrderError("no turn is in flight")
        expected_id = self._turns[self._index].id
        if self._in_flight != expected_id:
            raise TurnOrderError(
                f"in-flight turn {self._in_flight} is not the planned turn {expected_id}"
            )
        self._completed.append(expected_id)
        self._in_flight = None
        self._index += 1

    def fail(self, reason: str) -> None:
        self._failed = True
        self._failure = reason
        self._in_flight = None

    def ensure_complete(self) -> None:
        if self._failed:
            raise TurnOrderError(f"run failed: {self._failure}")
        if self._in_flight is not None:
            raise TurnOrderError(
                f"turn {self._in_flight} is still in flight"
            )
        if self._index != len(self._turns):
            raise TurnOrderError("run is not complete")


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_json(value: Any) -> str:
    canonical = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return sha256_text(canonical)


def safe_run_id(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    if not cleaned or cleaned in {".", ".."} or ".." in cleaned.split("."):
        raise ValueError("run_id must contain a safe non-empty component")
    return cleaned


def prompt_version(system_sha256: str) -> str:
    version = f"{ASSET_VERSION}+sys-{system_sha256[:12]}"
    if len(version) > 50:
        raise ValueError("prompt version exceeds snapshot limit")
    return version


def load_system_prompt(asset_root: Path) -> str:
    return (asset_root / "prompts" / "system.txt").read_text(encoding="utf-8")


def load_scripts(asset_root: Path) -> list[ScriptSpec]:
    payload = json.loads(
        (asset_root / "scripts" / "scripts.json").read_text(encoding="utf-8")
    )
    if payload.get("condition") != "B":
        raise ValueError("scripts condition must be B")
    if payload.get("asset_version") != ASSET_VERSION:
        raise ValueError("scripts asset_version mismatch")
    items = payload.get("scripts")
    if not isinstance(items, list):
        raise ValueError("scripts must be a list")

    result: list[ScriptSpec] = []
    seen: set[str] = set()
    for item in items:
        script_id = item.get("id")
        if not isinstance(script_id, str) or script_id in seen:
            raise ValueError(f"invalid/duplicate script id: {script_id}")
        seen.add(script_id)
        turns_raw = item.get("turns")
        branches_raw = item.get("branches", [])
        if not isinstance(turns_raw, list) or not turns_raw:
            raise ValueError(f"{script_id}: non-empty turns required")
        if not isinstance(branches_raw, list):
            raise ValueError(f"{script_id}: branches list required")
        turns: list[TurnSpec] = []
        for turn in turns_raw:
            if not isinstance(turn, dict) or set(turn) != {"id", "phase", "content"}:
                raise ValueError(f"{script_id}: invalid turn fields")
            if not isinstance(turn["content"], str) or not turn["content"].strip():
                raise ValueError(f"{script_id}: empty turn content")
            turns.append(TurnSpec(turn["id"], turn["phase"], turn["content"]))
        branches: list[BranchSpec] = []
        for branch in branches_raw:
            if not isinstance(branch, dict):
                raise ValueError(f"{script_id}: invalid branch")
            branches.append(
                BranchSpec(
                    id=branch["id"],
                    label=branch.get("label", ""),
                    content=branch["content"],
                    adds=tuple(branch.get("adds", [])),
                    direct_ac=bool(branch.get("direct_ac", False)),
                    rule_status=branch.get("rule_status", ""),
                )
            )
        if script_id in FIXED_IDS:
            expected = FIXED_CALLS[script_id]
            if item.get("mode") != "fixed" or item.get("calls_expected") != expected:
                raise ValueError(f"{script_id}: fixed call count mismatch")
            if len(turns) != expected or branches:
                raise ValueError(f"{script_id}: fixed turn/branch mismatch")
        elif script_id in INTERACTIVE_IDS:
            if item.get("mode") != "interactive":
                raise ValueError(f"{script_id}: interactive mode required")
            if len(turns) != 2:
                raise ValueError(f"{script_id}: interactive requires T1/Q1")
            calls_expected = item.get("calls_expected")
            if script_id == "I1":
                if calls_expected != 2 or branches:
                    raise ValueError("I1 must have 2 calls and no branches")
            elif calls_expected != {"minimum": 2, "maximum": 3}:
                raise ValueError(f"{script_id}: interactive calls must be 2-3")
            elif script_id in BRANCH_OPTIONS:
                allowed = set(BRANCH_OPTIONS[script_id]) - {"none"}
                if {branch.id for branch in branches} != allowed:
                    raise ValueError(f"{script_id}: unexpected branch set")
        else:
            raise ValueError(f"unknown script id: {script_id}")
        result.append(
            ScriptSpec(
                id=script_id,
                mode=item["mode"],
                title=item.get("title", ""),
                focus=item.get("focus", ""),
                calls_expected=item["calls_expected"],
                turns=tuple(turns),
                branches=tuple(branches),
            )
        )

    if tuple(script.id for script in result) != FIXED_IDS + INTERACTIVE_IDS:
        raise ValueError("scripts must be S1-S6 then I1-I3")
    return result


def _client_context(client: LLMClient) -> dict[str, Any]:
    return {
        "kind": type(client).__name__,
        "synthetic": bool(getattr(client, "synthetic", False)),
    }


def _model_context() -> dict[str, Any]:
    return {
        "profile_id": None,
        "profile_kind": None,
        "request_model": None,
        "upstream_response_model": None,
        "note": "null: the local fake path has no model/profile or provider response.",
    }


def _generation_context() -> dict[str, Any]:
    return {
        "requested": {
            "temperature": None,
            "max_tokens": None,
            "response_format": None,
            "timeout_seconds": None,
            "reasoning_effort": None,
            "note": "null: the fake runner sends no generation parameters.",
        },
        "response_metadata": {
            "model": None,
            "request_id": None,
            "usage": None,
            "finish_reason": None,
            "note": "null: the fake client returns only text.",
        },
    }


def _source_context(
    run_id: str,
    script_id: str,
    turn_id: str,
    branch_id: str | None,
) -> dict[str, Any]:
    def local(target: str) -> str:
        return f"{run_id}:{script_id}:{target}"

    source_turn_ids: list[str] = []
    related: list[str] = []
    event: dict[str, Any] | None = None
    if turn_id != "T1":
        source_turn_ids = ["T1"]
        related = [local("T1")]
    if script_id in {"S5", "S6"} and turn_id == "T2":
        source_turn_ids = ["T1", "Q1"]
        related = [local("T1"), local("Q1")]
        event = {
            "type": "correction" if script_id == "S5" else "revocation",
            "target_turn_ids": ["T1"],
            "target_record_ids": [local("T1")],
        }
    elif script_id in {"S5", "S6"} and turn_id == "Q2":
        source_turn_ids = ["T2"]
        related = [local("T2")]
        event = {"type": "recheck", "target_turn_ids": ["T1"]}
    elif branch_id is not None and turn_id == "T2":
        source_turn_ids = ["T1", "Q1"]
        related = [local("T1"), local("Q1")]
        event = {"type": "manual_branch", "branch": branch_id}
    return {
        "source_message_id": None,
        "source_turn_ids": source_turn_ids,
        "related_record_ids": related,
        "event": event,
        "note": "Run-local links only; source_message_id is null.",
    }


def make_record(
    *,
    run_id: str,
    script: ScriptSpec,
    turn: TurnSpec,
    turn_index: int,
    call_index: int,
    messages: Sequence[dict[str, str]],
    response: str | None,
    status: str,
    client: LLMClient,
    prompt_version_value: str,
    system_sha256: str,
    failure_reason: str | None = None,
    decision: BranchDecision | None = None,
    final_assistant: bool = False,
) -> dict[str, Any]:
    request_payload = {"messages": list(messages)}
    branch_id = decision.id if decision else None
    return {
        "schema_version": TURN_SCHEMA_VERSION,
        "record_id": f"{run_id}:{script.id}:{turn.id}",
        "run_id": run_id,
        "script_id": script.id,
        "turn_id": turn.id,
        "phase": turn.phase,
        "mode": script.mode,
        "condition": "B",
        "turn_index": turn_index,
        "call_index": call_index,
        "retry": False,
        "client": _client_context(client),
        "prompt_version": prompt_version_value,
        "system_sha256": system_sha256,
        "model": _model_context(),
        "generation": _generation_context(),
        "branch": (
            {
                "id": decision.id,
                "label": decision.label,
                "selected_by": decision.selected_by,
                "note": decision.note,
            }
            if decision
            else None
        ),
        "final_assistant": final_assistant,
        "final_for_user_turn_id": turn.id if final_assistant else None,
        "final_is_separate_user_turn": False if final_assistant else None,
        "current_user": turn.content,
        "request": {
            "messages": list(messages),
            "sha256": sha256_json(request_payload),
        },
        "response": response,
        "response_sha256": sha256_text(response) if response is not None else None,
        "status": status,
        "error": (
            {"type": "RunnerFailure", "message": failure_reason}
            if failure_reason
            else None
        ),
        "source": _source_context(run_id, script.id, turn.id, branch_id),
        "model_claims": {
            "reported_direct_teaching": None,
            "reported_inference": None,
            "reported_question": None,
            "note": "Model self-report only; not parsed or trusted by the runner.",
        },
        "human_review": {
            "direct_teaching_only": None,
            "inference_absent": None,
            "question_type": None,
            "question_target": None,
            "question_hint": None,
            "branch_increment_ok": None,
            "final_question_recorded": None,
            "notes": None,
        },
    }


def _run_meta(
    *,
    run_id: str,
    script: ScriptSpec,
    status: str,
    calls: int,
    started_at: str,
    system_sha256: str,
    prompt_version_value: str,
    error: dict[str, Any] | None,
    decision: BranchDecision | None,
    client: LLMClient,
) -> dict[str, Any]:
    return {
        "schema_version": RUN_SCHEMA_VERSION,
        "asset_version": ASSET_VERSION,
        "condition": "B",
        "run_id": run_id,
        "script_id": script.id,
        "script_mode": script.mode,
        "script_title": script.title,
        "script_focus": script.focus,
        "started_at_utc": started_at,
        "finished_at_utc": utc_now(),
        "status": status,
        "calls_min": script.calls_min,
        "calls_max": script.calls_max,
        "calls_actual": calls,
        "retry_policy": "none",
        "branch": (
            {"id": decision.id, "label": decision.label, "selected_by": "human"}
            if decision
            else None
        ),
        "error": error,
        "client": _client_context(client),
        "prompt_version": prompt_version_value,
        "system_sha256": system_sha256,
        "history_context_version": "history-context-b1",
        "max_history_messages": MAX_HISTORY_MESSAGES,
        "interactive_pause": script.mode == "interactive" and script.id in {"I2", "I3"},
        "notes": [
            "No production prompt, mode, database, or application file is used.",
            "A branch is selected by a human, never by the runner.",
            "FINAL is the assistant response to T2; it is not a separate user turn.",
        ],
    }


async def _execute_turn(
    *,
    script: ScriptSpec,
    turn: TurnSpec,
    run_id: str,
    system_prompt: str,
    system_sha256: str,
    prompt_version_value: str,
    client: LLMClient,
    storage: list[StoredMessage],
    ledger: TurnLedger,
    recorder: RunRecorder,
    records: list[dict[str, Any]],
    call_index: int,
    decision: BranchDecision | None = None,
    final_assistant: bool = False,
    allowed_contents: Sequence[str] | None = None,
) -> tuple[bool, dict[str, Any] | None, str | None]:
    ledger.begin(turn.id, turn.content, allowed_contents=allowed_contents)
    messages = build_request_messages(system_prompt, storage, turn.content)
    storage.append(StoredMessage(role="user", content=turn.content))
    try:
        raw = await client.generate(messages)
        if not isinstance(raw, str):
            raise EmptyResponseError("client returned a non-string response")
        if not raw.strip():
            raise EmptyResponseError("client returned an empty or blank response")
    except Exception as exc:  # record and stop; no retry
        failure = f"{type(exc).__name__}: {exc}"
        record = make_record(
            run_id=run_id,
            script=script,
            turn=turn,
            turn_index=ledger.index + 1,
            call_index=call_index,
            messages=messages,
            response=raw if isinstance(locals().get("raw"), str) else None,
            status="error",
            client=client,
            prompt_version_value=prompt_version_value,
            system_sha256=system_sha256,
            failure_reason=failure,
            decision=decision,
            final_assistant=final_assistant,
        )
        records.append(record)
        recorder.append(record)
        ledger.fail(failure)
        return False, record, failure

    storage.append(StoredMessage(role="assistant", content=raw))
    record = make_record(
        run_id=run_id,
        script=script,
        turn=turn,
        turn_index=ledger.index + 1,
        call_index=call_index,
        messages=messages,
        response=raw,
        status="ok",
        client=client,
        prompt_version_value=prompt_version_value,
        system_sha256=system_sha256,
        decision=decision,
        final_assistant=final_assistant,
    )
    records.append(record)
    recorder.append(record)
    ledger.commit()
    return True, record, None


async def run_fixed_script(
    script: ScriptSpec,
    *,
    system_prompt: str,
    system_sha256: str,
    prompt_version_value: str,
    client: LLMClient,
    output_root: Path,
    run_id: str | None = None,
) -> RunResult:
    if script.mode != "fixed":
        raise ValueError("run_fixed_script requires a fixed script")
    run_id = safe_run_id(
        run_id
        or f"{ASSET_VERSION}-{script.id}-{utc_now().replace(':', '').replace('+', 'Z')}-{uuid.uuid4().hex[:8]}"
    )
    recorder = RunRecorder(output_root, run_id)
    storage: list[StoredMessage] = []
    records: list[dict[str, Any]] = []
    ledger = TurnLedger(run_id, script.id, script.turns, script.calls_max)
    started_at = utc_now()
    calls = 0
    status = "completed"
    error: dict[str, str] | None = None
    try:
        for turn in script.turns:
            calls += 1
            ok, _, failure = await _execute_turn(
                script=script,
                turn=turn,
                run_id=run_id,
                system_prompt=system_prompt,
                system_sha256=system_sha256,
                prompt_version_value=prompt_version_value,
                client=client,
                storage=storage,
                ledger=ledger,
                recorder=recorder,
                records=records,
                call_index=calls,
            )
            if not ok:
                status = "failed"
                error = {"type": "RunnerFailure", "message": failure or "turn failed"}
                break
    finally:
        recorder.close()
    recorder.write_meta(
        _run_meta(
            run_id=run_id,
            script=script,
            status=status,
            calls=calls,
            started_at=started_at,
            system_sha256=system_sha256,
            prompt_version_value=prompt_version_value,
            error=error,
            decision=None,
            client=client,
        )
    )
    return RunResult(
        run_id=run_id,
        script_id=script.id,
        mode=script.mode,
        status=status,
        calls=calls,
        records=records,
        error=error,
        run_dir=str(recorder.run_dir),
    )


async def start_interactive_script(
    script: ScriptSpec,
    *,
    system_prompt: str,
    system_sha256: str,
    prompt_version_value: str,
    client: LLMClient,
    output_root: Path,
    run_id: str | None = None,
) -> tuple[PausedInteractive | None, RunResult]:
    """Run T1/Q1 and stop at the manual branch boundary.

    If T1/Q1 fails, returns (None, failed_result).  Otherwise returns a paused
    object and a paused RunResult.  No T2/FINAL call is made here.
    """
    if script.mode != "interactive" or script.id not in INTERACTIVE_IDS:
        raise ValueError("start_interactive_script requires I1/I2/I3")
    run_id = safe_run_id(
        run_id
        or f"{ASSET_VERSION}-{script.id}-{utc_now().replace(':', '').replace('+', 'Z')}-{uuid.uuid4().hex[:8]}"
    )
    recorder = RunRecorder(output_root, run_id)
    storage: list[StoredMessage] = []
    records: list[dict[str, Any]] = []
    expected = script.turns + (TurnSpec("T2", "branch_final", ""),)
    ledger = TurnLedger(run_id, script.id, expected, INTERACTIVE_MAX_CALLS[script.id])
    started_at = utc_now()
    calls = 0
    for turn in script.turns:
        calls += 1
        ok, _, failure = await _execute_turn(
            script=script,
            turn=turn,
            run_id=run_id,
            system_prompt=system_prompt,
            system_sha256=system_sha256,
            prompt_version_value=prompt_version_value,
            client=client,
            storage=storage,
            ledger=ledger,
            recorder=recorder,
            records=records,
            call_index=calls,
        )
        if not ok:
            recorder.close()
            error = {"type": "RunnerFailure", "message": failure or "turn failed"}
            recorder.write_meta(
                _run_meta(
                    run_id=run_id,
                    script=script,
                    status="failed",
                    calls=calls,
                    started_at=started_at,
                    system_sha256=system_sha256,
                    prompt_version_value=prompt_version_value,
                    error=error,
                    decision=None,
                    client=client,
                )
            )
            return None, RunResult(
                run_id=run_id,
                script_id=script.id,
                mode=script.mode,
                status="failed",
                calls=calls,
                records=records,
                error=error,
                run_dir=str(recorder.run_dir),
            )

    if script.id not in BRANCH_OPTIONS:
        recorder.close()
        recorder.write_meta(
            _run_meta(
                run_id=run_id,
                script=script,
                status="completed",
                calls=calls,
                started_at=started_at,
                system_sha256=system_sha256,
                prompt_version_value=prompt_version_value,
                error=None,
                decision=None,
                client=client,
            )
        )
        return None, RunResult(
            run_id=run_id,
            script_id=script.id,
            mode=script.mode,
            status="completed",
            calls=calls,
            records=records,
            run_dir=str(recorder.run_dir),
        )

    recorder.write_meta(
        _run_meta(
            run_id=run_id,
            script=script,
            status="paused",
            calls=calls,
            started_at=started_at,
            system_sha256=system_sha256,
            prompt_version_value=prompt_version_value,
            error=None,
            decision=None,
            client=client,
        )
    )
    paused = PausedInteractive(
        script=script,
        system_prompt=system_prompt,
        system_sha256=system_sha256,
        prompt_version_value=prompt_version_value,
        client=client,
        recorder=recorder,
        storage=storage,
        records=records,
        ledger=ledger,
        run_id=run_id,
        started_at=started_at,
        calls=calls,
    )
    return paused, RunResult(
        run_id=run_id,
        script_id=script.id,
        mode=script.mode,
        status="paused",
        calls=calls,
        records=records,
        run_dir=str(recorder.run_dir),
    )


async def resume_interactive_script(
    paused: PausedInteractive,
    branch_choice: str,
) -> RunResult:
    """Resume a paused I2/I3 exactly once with a human-selected branch.

    The terminal state is claimed before any await, history mutation, or
    client call.  A second/concurrent selection is rejected without closing
    or altering the first selection's evidence.
    """
    if paused.state != "paused":
        raise TurnOrderError(
            f"interactive run is already {paused.state}; branch selection is terminal"
        )
    script = paused.script
    if script.id not in BRANCH_OPTIONS:
        raise ValueError("this interactive script has no branch")
    if branch_choice not in BRANCH_OPTIONS[script.id]:
        raise TurnOrderError(f"unknown branch for {script.id}: {branch_choice}")
    if paused.ledger.index != 2 or paused.calls != 2:
        raise TurnOrderError("interactive run is not paused after Q1")

    decision = BranchDecision(
        id=branch_choice,
        label=(
            "no branch"
            if branch_choice == "none"
            else next(branch.label for branch in script.branches if branch.id == branch_choice)
        ),
        selected_by="human",
        note="Selected after Q1 pause; not inferred from text.",
    )

    # Claim the one-shot terminal state synchronously.  Any competing call
    # entered while the branch generation is awaiting sees "running" and is
    # rejected before it can add a call or close the recorder.
    paused.state = "running"
    paused.selection_note = decision.note
    status = "completed"
    error: dict[str, Any] | None = None
    calls = paused.calls
    cancel_exc: asyncio.CancelledError | None = None
    cleanup_failures: list[tuple[str, Exception]] = []
    try:
        if branch_choice != "none":
            branch = next(branch for branch in script.branches if branch.id == branch_choice)
            calls += 1
            ok, _, failure = await _execute_turn(
                script=script,
                turn=TurnSpec("T2", "branch_final", branch.content),
                run_id=paused.run_id,
                system_prompt=paused.system_prompt,
                system_sha256=paused.system_sha256,
                prompt_version_value=paused.prompt_version_value,
                client=paused.client,
                storage=paused.storage,
                ledger=paused.ledger,
                recorder=paused.recorder,
                records=paused.records,
                call_index=calls,
                decision=decision,
                final_assistant=True,
                allowed_contents=(branch.content,),
            )
            if not ok:
                status = "failed"
                error = {"type": "RunnerFailure", "message": failure or "turn failed"}
    except asyncio.CancelledError as exc:
        # Cancellation is not a successful branch completion.  Keep the
        # original exception for propagation after evidence cleanup.
        status = "cancelled"
        error = {"type": "CancelledError", "message": "interactive branch cancelled"}
        cancel_exc = exc
    except TurnOrderError as exc:
        status = "failed"
        error = {"type": "TurnOrderError", "message": str(exc)}
    except Exception as exc:  # preserve evidence; never retry
        status = "failed"
        error = {"type": type(exc).__name__, "message": "unexpected branch failure"}
    finally:
        try:
            paused.recorder.close()
        except Exception as exc:
            cleanup_failures.append(("cleanup", exc))
            if cancel_exc is None and status == "completed":
                status = "failed"
            if cancel_exc is None and error is None:
                error = {
                    "type": "CleanupError",
                    "message": "branch evidence cleanup failed; run is not successful",
                }
        try:
            paused.recorder.update_meta(
                _run_meta(
                    run_id=paused.run_id,
                    script=script,
                    status=status,
                    calls=calls,
                    started_at=paused.started_at,
                    system_sha256=paused.system_sha256,
                    prompt_version_value=paused.prompt_version_value,
                    error=error,
                    decision=decision,
                    client=paused.client,
                )
            )
        except Exception as exc:
            cleanup_failures.append(("metadata", exc))
            if cancel_exc is None:
                if status == "completed":
                    status = "failed"
                if error is None:
                    error = {
                        "type": "MetadataWriteError",
                        "message": (
                            "run metadata was not updated; "
                            "evidence is incomplete and disk state may still be paused"
                        ),
                    }
        if cleanup_failures:
            paused.cleanup_error = "; ".join(
                f"{kind}: {type(exc).__name__}" for kind, exc in cleanup_failures
            )
            if cancel_exc is None and error is not None:
                error = {**error, "cleanup_error": paused.cleanup_error}
        paused.state = status
    if cancel_exc is not None:
        raise cancel_exc
    return RunResult(
        run_id=paused.run_id,
        script_id=script.id,
        mode=script.mode,
        status=status,
        calls=calls,
        records=paused.records,
        error=error,
        run_dir=str(paused.recorder.run_dir),
        branch_choice=branch_choice,
    )


async def run_interactive_script(
    script: ScriptSpec,
    *,
    system_prompt: str,
    system_sha256: str,
    prompt_version_value: str,
    client: LLMClient,
    output_root: Path,
    branch_choice: str | None,
    run_id: str | None = None,
) -> RunResult:
    paused, result = await start_interactive_script(
        script,
        system_prompt=system_prompt,
        system_sha256=system_sha256,
        prompt_version_value=prompt_version_value,
        client=client,
        output_root=output_root,
        run_id=run_id,
    )
    if paused is None:
        return result
    if branch_choice is None:
        # No T2 will be sent for this paused run; close the evidence file but
        # keep the paused result for the caller.  A caller that wants to resume
        # explicitly uses start_interactive_script().
        paused.recorder.close()
        return result
    return await resume_interactive_script(paused, branch_choice)


async def run_all(
    *,
    asset_root: Path,
    output_root: Path,
    client_factory: Callable[[], LLMClient] = FakeLLMClient,
    run_prefix: str = "option-b",
    branch_choices: dict[str, str] | None = None,
) -> list[RunResult]:
    # Preflight before creating any client or run directory.
    manifest = load_and_verify_manifest(asset_root)
    if manifest.get("condition") != "B":
        raise FrozenAssetError("manifest condition is not B")
    system_prompt = load_system_prompt(asset_root)
    system_hash = sha256_text(system_prompt)
    if manifest["prompts"]["system"]["sha256"] != system_hash:
        raise FrozenAssetError("system prompt hash does not match the manifest")
    scripts = load_scripts(asset_root)
    scripts_hash = sha256_text(
        (asset_root / "scripts" / "scripts.json").read_text(encoding="utf-8")
    )
    if manifest["files"]["scripts/scripts.json"]["sha256"] != scripts_hash:
        raise FrozenAssetError("scripts.json hash does not match the manifest")
    version = prompt_version(system_hash)
    choices = dict(branch_choices or {})
    batch = utc_now().replace(":", "").replace("+", "Z")
    results: list[RunResult] = []

    def fresh_id(script_id: str) -> str:
        return safe_run_id(
            f"{run_prefix}-{batch}-{script_id}-{uuid.uuid4().hex[:6]}"
        )

    for script in scripts:
        if script.mode == "fixed":
            result = await run_fixed_script(
                script,
                system_prompt=system_prompt,
                system_sha256=system_hash,
                prompt_version_value=version,
                client=client_factory(),
                output_root=output_root,
                run_id=fresh_id(script.id),
            )
        else:
            choice = choices.get(script.id)
            if script.id == "I1":
                choice = "none"
            result = await run_interactive_script(
                script,
                system_prompt=system_prompt,
                system_sha256=system_hash,
                prompt_version_value=version,
                client=client_factory(),
                output_root=output_root,
                branch_choice=choice,
                run_id=fresh_id(script.id),
            )
        results.append(result)
        if result.status == "failed":
            break
    return results


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the isolated Option B fake-client experiment."
    )
    parser.add_argument(
        "--asset-root",
        type=Path,
        default=Path(__file__).resolve().parent,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "runs",
    )
    parser.add_argument("--run-prefix", default="option-b")
    parser.add_argument(
        "--branch",
        action="append",
        default=[],
        metavar="SCRIPT=BRANCH",
        help="manual I2/I3 decision, e.g. I2=bc; omit to pause after Q1",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    choices: dict[str, str] = {}
    for item in args.branch:
        if "=" not in item:
            raise SystemExit(f"invalid --branch: {item!r}")
        script_id, branch = item.split("=", 1)
        if script_id not in BRANCH_OPTIONS:
            raise SystemExit(f"branch not allowed for {script_id}")
        if branch not in BRANCH_OPTIONS[script_id]:
            raise SystemExit(f"invalid branch for {script_id}: {branch}")
        choices[script_id] = branch
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = asyncio.run(
        run_all(
            asset_root=args.asset_root,
            output_root=args.output_dir,
            run_prefix=args.run_prefix,
            branch_choices=choices,
        )
    )
    summary = [
        {
            "run_id": result.run_id,
            "script_id": result.script_id,
            "mode": result.mode,
            "status": result.status,
            "calls": result.calls,
            "branch": result.branch_choice,
            "run_dir": result.run_dir,
        }
        for result in results
    ]
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if all(result.status == "completed" for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
