"""Batch state for the isolated Option A+B web test instance.

The web layer is not the frozen Option A+B runner.  It uses the same frozen
scripts but drives the real ChatService, so the single-execution state machine
lives here.  A script owns exactly one session; turns are ordered; interactive
scripts pause after Q1 for a human branch choice.  Selecting a branch also
arms its one T2 exactly once.  There is deliberately no separate T2 arm entry
point.
"""

from __future__ import annotations

import hashlib
import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping


class OptionABWebError(RuntimeError):
    """A safe, user-visible test-harness error."""


class GateError(OptionABWebError):
    """A refused state transition.

    ``fatal`` is true for content/identity mismatches that compromise the
    experiment.  Duplicate or competing requests against an already consumed
    transition are non-fatal: the first legal request must keep its evidence
    and continue.
    """

    def __init__(self, message: str, *, fatal: bool = False) -> None:
        super().__init__(message)
        self.fatal = fatal


@dataclass(frozen=True)
class TurnSpec:
    script_id: str
    turn_id: str
    content: str
    content_sha256: str


@dataclass(frozen=True)
class BranchSpec:
    script_id: str
    branch_id: str
    label: str
    content: str | None
    content_sha256: str | None
    is_none: bool = False


@dataclass(frozen=True)
class ScriptSpec:
    script_id: str
    mode: str
    turns: tuple[TurnSpec, ...]
    branches: tuple[BranchSpec, ...]
    min_calls: int
    max_calls: int


@dataclass(frozen=True)
class FrozenSpec:
    asset_root: Path
    asset_version: str
    manifest_sha256: str
    system_prompt: str
    system_sha256: str
    prompt_version: str
    scripts: tuple[ScriptSpec, ...]
    total_min: int
    total_max: int
    fixed_calls: int

    def script(self, script_id: str) -> ScriptSpec:
        for script in self.scripts:
            if script.script_id == script_id:
                return script
        raise GateError(f"unknown script: {script_id}", fatal=True)

    def turn(self, script_id: str, turn_id: str) -> TurnSpec:
        script = self.script(script_id)
        for turn in script.turns:
            if turn.turn_id == turn_id:
                return turn
        raise GateError(f"unknown turn: {script_id}/{turn_id}", fatal=True)


@dataclass(frozen=True)
class ArmedTurn:
    instance_id: str
    session_id: int
    script_id: str
    turn_id: str
    content_sha256: str
    branch_id: str | None = None


@dataclass(frozen=True)
class TurnCompletion:
    instance_id: str
    session_id: int
    script_id: str
    turn_id: str
    content_sha256: str
    user_message_id: int
    assistant_message_id: int
    completed_at: str


@dataclass(frozen=True)
class BranchSelection:
    instance_id: str
    session_id: int
    script_id: str
    branch_id: str
    branch_label: str
    q1_assistant_message_id: int
    selected_at: str
    t2_content: str | None
    t2_content_sha256: str | None


@dataclass
class _ScriptRun:
    spec: ScriptSpec
    phase: str = "pending"
    session_id: int | None = None
    turn_index: int = 0
    calls: int = 0
    completions: list[TurnCompletion] = field(default_factory=list)
    branch: BranchSpec | None = None
    t2_turn: TurnSpec | None = None


def content_sha256(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _now() -> str:
    return datetime.now(UTC).isoformat()


class BatchState:
    """One instance/run with ten independent scripts.

    The state is deliberately thread-safe.  ASGI handlers and tests can call
    these methods from different threads; a transition is reserved before any
    await or side effect.
    """

    def __init__(
        self,
        *,
        instance_id: str,
        spec: FrozenSpec,
        max_sessions: int = 10,
    ) -> None:
        if not isinstance(instance_id, str) or not instance_id:
            raise ValueError("instance_id is required")
        self.instance_id = instance_id
        self.spec = spec
        self.max_sessions = max_sessions
        self._lock = threading.RLock()
        self._scripts = [_ScriptRun(script) for script in spec.scripts]
        self._script_by_id = {item.spec.script_id: item for item in self._scripts}
        self._script_order = [item.spec.script_id for item in self._scripts]
        self._current_index = 0
        self._failed_reason: str | None = None
        self._cancelled = False
        self._armed: ArmedTurn | None = None
        self._in_flight: ArmedTurn | None = None
        self._total_consumed = 0
        self._sessions_created = 0
        self._session_to_script: dict[int, str] = {}
        self._script_to_session: dict[str, int] = {}
        self._branches: list[BranchSelection] = []
        self._completions: dict[tuple[str, str], TurnCompletion] = {}

    # ------------------------------------------------------------------
    # Read-only inspection
    # ------------------------------------------------------------------
    @property
    def total_consumed(self) -> int:
        with self._lock:
            return self._total_consumed

    @property
    def total_expected_min(self) -> int:
        return self.spec.total_min

    @property
    def total_expected_max(self) -> int:
        return self.spec.total_max

    @property
    def failed(self) -> bool:
        with self._lock:
            return self._failed_reason is not None

    @property
    def cancelled(self) -> bool:
        with self._lock:
            return self._cancelled

    @property
    def in_flight(self) -> ArmedTurn | None:
        with self._lock:
            return self._in_flight

    @property
    def armed(self) -> ArmedTurn | None:
        with self._lock:
            return self._armed

    @property
    def sessions_created(self) -> int:
        with self._lock:
            return self._sessions_created

    def script_phase(self, script_id: str) -> str:
        with self._lock:
            return self._script_by_id[script_id].phase

    def completed_turn(
        self, script_id: str, turn_id: str
    ) -> TurnCompletion | None:
        with self._lock:
            return self._completions.get((script_id, turn_id))

    def turn_spec(self, script_id: str, turn_id: str) -> TurnSpec:
        with self._lock:
            script = self._script_by_id.get(script_id)
            if script is None:
                raise GateError("unknown script", fatal=True)
            if turn_id == "T2" and script.t2_turn is not None:
                return script.t2_turn
            for turn in script.spec.turns:
                if turn.turn_id == turn_id:
                    return turn
            raise GateError("unknown turn", fatal=True)

    @property
    def batch_completed(self) -> bool:
        with self._lock:
            return (
                self._failed_reason is None
                and not self._cancelled
                and self._current_index >= len(self._scripts)
            )

    def state_snapshot(self) -> dict[str, Any]:
        with self._lock:
            scripts: list[dict[str, Any]] = []
            for item in self._scripts:
                scripts.append(
                    {
                        "id": item.spec.script_id,
                        "mode": item.spec.mode,
                        "phase": item.phase,
                        "session_id": item.session_id,
                        "turn_index": item.turn_index,
                        "turns_expected": len(item.spec.turns),
                        "calls": item.calls,
                        "min_calls": item.spec.min_calls,
                        "max_calls": item.spec.max_calls,
                        "q1_assistant_message_id": (
                            item.completions[-1].assistant_message_id
                            if item.phase == "paused_human_classification"
                            and item.completions
                            else None
                        ),
                        "branches": [
                            {
                                "id": branch.branch_id,
                                "label": branch.label,
                                "is_none": branch.is_none,
                            }
                            for branch in item.spec.branches
                        ],
                        "selected_branch": item.branch.branch_id if item.branch else None,
                        "selected_t2_content": (
                            item.t2_turn.content
                            if item.t2_turn is not None
                            and item.phase in {"branch_selected", "t2_in_flight"}
                            else None
                        ),
                        "selected_t2_content_sha256": (
                            item.t2_turn.content_sha256
                            if item.t2_turn is not None
                            and item.phase in {"branch_selected", "t2_in_flight"}
                            else None
                        ),
                    }
                )
            current = (
                self._scripts[self._current_index]
                if self._current_index < len(self._scripts)
                else None
            )
            paused = bool(
                current
                and current.phase == "paused_human_classification"
            )
            return {
                "instance_id": self.instance_id,
                "condition": "AB",
                "asset_version": self.spec.asset_version,
                "prompt_version": self.spec.prompt_version,
                "system_sha256": self.spec.system_sha256,
                "status": (
                    "failed"
                    if self._failed_reason
                    else "cancelled"
                    if self._cancelled
                    else "completed"
                    if self._current_index >= len(self._scripts)
                    else "running"
                ),
                "failed_reason": self._failed_reason,
                "current_script": current.spec.script_id if current else None,
                "current_turn": (
                    current.spec.turns[current.turn_index].turn_id
                    if current
                    and current.phase == "active"
                    and current.turn_index < len(current.spec.turns)
                    else None
                ),
                "paused_human_classification": paused,
                "scripts": scripts,
                "sessions_created": self._sessions_created,
                "total_consumed": self._total_consumed,
                "total_min": self.spec.total_min,
                "total_max": self.spec.total_max,
                "fixed_calls": self.spec.fixed_calls,
            }

    # ------------------------------------------------------------------
    # Session budget
    # ------------------------------------------------------------------
    def reserve_session_creation(self) -> bool:
        with self._lock:
            if self._failed_reason is not None or self._cancelled:
                return False
            if self._sessions_created >= self.max_sessions:
                return False
            self._sessions_created += 1
            return True

    def release_session_creation(self) -> None:
        with self._lock:
            if self._sessions_created > 0:
                self._sessions_created -= 1

    def fail(self, reason: str) -> None:
        with self._lock:
            if self._failed_reason is None:
                self._failed_reason = reason
            self._armed = None
            self._in_flight = None

    def cancel(self) -> None:
        with self._lock:
            self._cancelled = True
            self._armed = None
            self._in_flight = None

    # ------------------------------------------------------------------
    # Turn gating
    # ------------------------------------------------------------------
    def _ensure_usable(self) -> None:
        if self._failed_reason is not None:
            raise GateError("batch already failed", fatal=True)
        if self._cancelled:
            raise GateError("batch already cancelled", fatal=True)

    def _current_script(self) -> _ScriptRun:
        if self._current_index >= len(self._scripts):
            raise GateError("batch is already complete", fatal=False)
        return self._scripts[self._current_index]

    def _bind_session(self, script: _ScriptRun, session_id: int) -> None:
        if not isinstance(session_id, int) or isinstance(session_id, bool) or session_id <= 0:
            raise GateError("invalid session id", fatal=True)
        previous_script = self._session_to_script.get(session_id)
        if previous_script is not None and previous_script != script.spec.script_id:
            raise GateError("session belongs to another script", fatal=True)
        bound = self._script_to_session.get(script.spec.script_id)
        if bound is not None and bound != session_id:
            raise GateError("script is bound to another session", fatal=True)
        if script.session_id is not None and script.session_id != session_id:
            raise GateError("session mismatch", fatal=True)
        script.session_id = session_id
        self._session_to_script[session_id] = script.spec.script_id
        self._script_to_session[script.spec.script_id] = session_id

    def arm(self, *, session_id: int, script_id: str, turn_id: str) -> ArmedTurn:
        """Arm the next ordinary turn; never a branch T2."""

        with self._lock:
            self._ensure_usable()
            if self._in_flight is not None:
                raise GateError("a turn is already in flight", fatal=False)
            if self._armed is not None:
                raise GateError("a turn is already armed", fatal=False)
            script = self._current_script()
            if script.spec.script_id != script_id:
                raise GateError("script is not the current script", fatal=True)
            if script.phase == "pending":
                first = script.spec.turns[0]
                if turn_id != first.turn_id:
                    raise GateError("must start with T1", fatal=True)
                self._bind_session(script, session_id)
                script.phase = "active"
            elif script.phase == "active":
                if script.turn_index >= len(script.spec.turns):
                    raise GateError("no ordinary turn remains", fatal=False)
                expected = script.spec.turns[script.turn_index]
                if turn_id != expected.turn_id:
                    raise GateError("turn is not next in frozen order", fatal=True)
            elif script.phase == "paused_human_classification":
                raise GateError(
                    "branch selection is the only entry while paused", fatal=False
                )
            elif script.phase == "branch_selected":
                raise GateError(
                    "branch T2 is already armed by branch selection", fatal=False
                )
            elif script.phase == "t2_in_flight":
                raise GateError("branch T2 is already in flight", fatal=False)
            elif script.phase in {"completed", "failed", "cancelled"}:
                raise GateError("script is terminal", fatal=False)
            else:
                raise GateError("script is not armable", fatal=False)
            if script.session_id is None:
                raise GateError("session is not bound", fatal=True)
            if script.session_id != session_id:
                raise GateError("session mismatch", fatal=True)
            spec_turn = script.spec.turns[script.turn_index]
            self._armed = ArmedTurn(
                instance_id=self.instance_id,
                session_id=session_id,
                script_id=script_id,
                turn_id=spec_turn.turn_id,
                content_sha256=spec_turn.content_sha256,
            )
            return self._armed

    def consume(self, *, session_id: int, content: str) -> ArmedTurn:
        with self._lock:
            self._ensure_usable()
            if self._in_flight is not None:
                raise GateError("a turn is already in flight", fatal=False)
            if self._armed is None:
                raise GateError("no armed turn", fatal=True)
            armed = self._armed
            if session_id != armed.session_id:
                raise GateError("session mismatch", fatal=True)
            if content_sha256(content) != armed.content_sha256:
                raise GateError("content mismatch", fatal=True)
            script = self._script_by_id[armed.script_id]
            if script.phase not in {"active", "branch_selected"}:
                raise GateError("script is not sendable", fatal=True)
            if self._total_consumed >= self.spec.total_max:
                raise GateError("total call budget exceeded", fatal=True)
            if script.calls >= script.spec.max_calls:
                raise GateError("script call budget exceeded", fatal=True)
            script.calls += 1
            self._total_consumed += 1
            self._armed = None
            self._in_flight = armed
            if script.phase == "branch_selected":
                script.phase = "t2_in_flight"
            return armed

    def complete(
        self,
        *,
        armed: ArmedTurn,
        user_message_id: int,
        assistant_message_id: int,
    ) -> TurnCompletion:
        with self._lock:
            self._ensure_usable()
            if self._in_flight is None or self._in_flight != armed:
                raise GateError("no matching in-flight turn", fatal=True)
            if (
                not isinstance(user_message_id, int)
                or isinstance(user_message_id, bool)
                or user_message_id <= 0
                or not isinstance(assistant_message_id, int)
                or isinstance(assistant_message_id, bool)
                or assistant_message_id <= 0
                or user_message_id == assistant_message_id
            ):
                raise GateError("invalid message ids", fatal=True)
            script = self._script_by_id[armed.script_id]
            completion = TurnCompletion(
                instance_id=self.instance_id,
                session_id=armed.session_id,
                script_id=armed.script_id,
                turn_id=armed.turn_id,
                content_sha256=armed.content_sha256,
                user_message_id=user_message_id,
                assistant_message_id=assistant_message_id,
                completed_at=_now(),
            )
            if script.phase == "t2_in_flight":
                if armed.turn_id != "T2":
                    raise GateError("unexpected turn while finishing branch", fatal=True)
                script.completions.append(completion)
                self._completions[(armed.script_id, armed.turn_id)] = completion
                script.phase = "completed"
                self._in_flight = None
                self._current_index += 1
                return completion
            if script.phase != "active":
                raise GateError("script is not active", fatal=True)
            expected = script.spec.turns[script.turn_index]
            if armed.turn_id != expected.turn_id:
                raise GateError("completion does not match expected turn", fatal=True)
            script.completions.append(completion)
            self._completions[(armed.script_id, armed.turn_id)] = completion
            script.turn_index += 1
            self._in_flight = None
            if script.turn_index < len(script.spec.turns):
                script.phase = "active"
                return completion
            if script.spec.branches:
                script.phase = "paused_human_classification"
            else:
                script.phase = "completed"
                self._current_index += 1
            return completion

    def select_branch(
        self,
        *,
        session_id: int,
        script_id: str,
        branch_id: str,
        q1_assistant_message_id: int,
        selected_by: str = "human",
        note: str = "",
    ) -> BranchSelection:
        """Select a branch and, for non-none, arm its one T2 at the same time."""

        with self._lock:
            self._ensure_usable()
            if self._in_flight is not None:
                raise GateError("a turn is in flight", fatal=False)
            if self._armed is not None:
                raise GateError("a turn is already armed", fatal=False)
            script = self._current_script()
            if script.spec.script_id != script_id:
                raise GateError("script is not current", fatal=True)
            if script.phase != "paused_human_classification":
                # Duplicates after a legal choice must not close the first
                # request's recorder or corrupt its evidence.
                raise GateError("script is not paused for classification", fatal=False)
            if script.session_id != session_id:
                raise GateError("session mismatch", fatal=True)
            if not isinstance(q1_assistant_message_id, int) or isinstance(
                q1_assistant_message_id, bool
            ):
                raise GateError("q1 assistant message id is required", fatal=True)
            q1 = next(
                (
                    item
                    for item in reversed(script.completions)
                    if item.turn_id == "Q1"
                ),
                None,
            )
            if q1 is None or q1.assistant_message_id != q1_assistant_message_id:
                raise GateError("q1 assistant message id mismatch", fatal=True)
            branch = next(
                (item for item in script.spec.branches if item.branch_id == branch_id),
                None,
            )
            if branch is None:
                raise GateError("unknown branch", fatal=True)
            if selected_by != "human":
                raise GateError("branch selection must be human", fatal=True)
            selection = BranchSelection(
                instance_id=self.instance_id,
                session_id=session_id,
                script_id=script_id,
                branch_id=branch.branch_id,
                branch_label=branch.label,
                q1_assistant_message_id=q1_assistant_message_id,
                selected_at=_now(),
                t2_content=branch.content,
                t2_content_sha256=branch.content_sha256,
            )
            self._branches.append(selection)
            script.branch = branch
            if branch.is_none:
                script.phase = "completed"
                self._current_index += 1
                return selection
            if branch.content is None or branch.content_sha256 is None:
                raise GateError("branch content missing", fatal=True)
            t2 = TurnSpec(
                script_id=script_id,
                turn_id="T2",
                content=branch.content,
                content_sha256=branch.content_sha256,
            )
            script.t2_turn = t2
            script.phase = "branch_selected"
            self._armed = ArmedTurn(
                instance_id=self.instance_id,
                session_id=session_id,
                script_id=script_id,
                turn_id="T2",
                content_sha256=branch.content_sha256,
                branch_id=branch.branch_id,
            )
            return selection

    def branches(self) -> list[BranchSelection]:
        with self._lock:
            return list(self._branches)

    def completions(self) -> list[TurnCompletion]:
        with self._lock:
            return [
                completion
                for item in self._scripts
                for completion in item.completions
            ]

    def script_session(self, script_id: str) -> int | None:
        with self._lock:
            item = self._script_by_id.get(script_id)
            return item.session_id if item else None


def branch_specs_from_script(script: Mapping[str, Any]) -> tuple[BranchSpec, ...]:
    branches: list[BranchSpec] = []
    script_id = str(script["id"])
    for item in script.get("branches", []):
        if not isinstance(item, Mapping):
            raise ValueError("branch must be an object")
        branch_id = str(item["id"])
        label = str(item["label"])
        content = item.get("content")
        if content is not None and not isinstance(content, str):
            raise ValueError("branch content must be text")
        branches.append(
            BranchSpec(
                script_id=script_id,
                branch_id=branch_id,
                label=label,
                content=content,
                content_sha256=content_sha256(content) if content is not None else None,
            )
        )
    return tuple(branches)


def iter_turns(spec: FrozenSpec) -> Iterable[TurnSpec]:
    for script in spec.scripts:
        yield from script.turns
