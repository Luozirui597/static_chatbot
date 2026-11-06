"""Minimal fake orchestration for the four-strategy comparison.

This module never calls a model or a network.  It validates the frozen assets
and simulates the call/turn accounting of one planned experiment.  Human branch
selection and semantic scoring are deliberately not simulated as real results.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from experiments.strategy_comparison.freeze_manifest import verify_manifest

SCHEMA_VERSION = "strategy-comparison-fake-run-v1"
TOTAL_MAX_CALLS = 264
TOTAL_MIN_CALLS = 240


class ExperimentError(RuntimeError):
    pass


class SyntheticFailure(ExperimentError):
    pass


@dataclass(frozen=True)
class Condition:
    strategy_id: str
    display_name: str
    switch_value: str
    prompt_file: str
    prompt_sha256: str
    prompt_version: str


@dataclass(frozen=True)
class Task:
    id: str
    kind: str
    calls: Any
    turns: tuple[dict[str, str], ...]


@dataclass(frozen=True)
class PlanEntry:
    session_key: str
    block_id: str
    task_id: str
    strategy_id: str
    repeat: int
    order: int


@dataclass
class CallRecord:
    index: int
    turn_id: str
    content: str
    prompt_version: str
    synthetic: bool = True


@dataclass
class SessionResult:
    session_key: str
    task_id: str
    strategy_id: str
    repeat: int
    calls: list[CallRecord] = field(default_factory=list)
    branch: str | None = None
    status: str = "active"
    failure_stage: str | None = None


class FakeSession:
    """A single isolated session; no shared model/history state."""

    def __init__(self, entry: PlanEntry, task: Task, condition: Condition):
        self.entry = entry
        self.task = task
        self.condition = condition
        self.result = SessionResult(
            session_key=entry.session_key,
            task_id=task.id,
            strategy_id=condition.strategy_id,
            repeat=entry.repeat,
        )

    def _turn(self, turn_id: str, content: str) -> str:
        if self.result.status != "active":
            raise ExperimentError("session is not active")
        call = CallRecord(
            index=len(self.result.calls) + 1,
            turn_id=turn_id,
            content=content,
            prompt_version=self.condition.prompt_version,
        )
        self.result.calls.append(call)
        # Synthetic output; this is not a provider response.
        return f"[synthetic] {self.condition.strategy_id}:{self.task.id}:{turn_id}"

    def run(
        self,
        branch: str | None = None,
        branch_content: str | None = None,
        *,
        fail_at_call: int | None = None,
    ) -> SessionResult:
        if self.result.status != "active":
            raise ExperimentError("session is not active")
        try:
            for turn in self.task.turns:
                if fail_at_call is not None and len(self.result.calls) + 1 == fail_at_call:
                    raise SyntheticFailure("synthetic_transport_failure")
                self._turn(turn["id"], turn["content"])
            if self.task.kind == "interactive":
                self.result.branch = branch
                if branch is not None:
                    if branch_content is None:
                        raise ExperimentError("missing branch content")
                    if fail_at_call is not None and len(self.result.calls) + 1 == fail_at_call:
                        raise SyntheticFailure("synthetic_transport_failure")
                    self._turn("T2", branch_content)
            self.result.status = "completed"
        except ExperimentError as exc:
            self.result.status = "failed"
            self.result.failure_stage = str(exc)
        return self.result


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _fake_branch(task_id: str, repeat: int) -> str | None:
    """A synthetic branch fixture, not a human classification.

    Branch categories are selected by task/repetition only, never by strategy.
    Repeat 1 exercises a legal missing-relation/rule branch, repeat 2 exercises
    no_extra, and repeat 3 exercises none.  Real runs leave this pending human
    review.
    """
    if task_id == "I2":
        return {1: "bc", 2: "no_extra", 3: None}[repeat]
    if task_id == "I3":
        return {1: "rule_supply", 2: "no_extra", 3: None}[repeat]
    return None


def load_assets(root: Path) -> tuple[dict[str, Condition], dict[str, Task], dict, dict]:

    verify_manifest(root)
    conditions_data = _load_json(root / "conditions.json")
    tasks_data = _load_json(root / "tasks/tasks.json")
    plan = _load_json(root / "run_plan.json")
    branches = _load_json(root / "branches/branches.json")
    conditions = {
        item["strategy_id"]: Condition(
            strategy_id=item["strategy_id"],
            display_name=item["display_name"],
            switch_value=item["switch_value"],
            prompt_file=item["prompt_file"],
            prompt_sha256=item["prompt_sha256"],
            prompt_version=item["prompt_version"],
        )
        for item in conditions_data["conditions"]
    }
    tasks = {
        item["id"]: Task(
            id=item["id"],
            kind=item["kind"],
            calls=item["calls"],
            turns=tuple(item["turns"]),
        )
        for item in tasks_data["tasks"]
    }
    if set(conditions) != {"standard", "a", "b", "ab"}:
        raise ExperimentError("condition set mismatch")
    if set(tasks) != {"S1", "S2", "S3", "S4", "S5", "S6", "I2", "I3"}:
        raise ExperimentError("task set mismatch")
    return conditions, tasks, plan, branches


def run_fake(
    root: Path,
    *,
    fail_session: str | None = None,
    fail_sessions: list[str] | None = None,
) -> dict:
    failure_keys = set(fail_sessions or [])
    if fail_session is not None:
        failure_keys.add(fail_session)
    conditions, tasks, plan, branches = load_assets(root)
    entries = [
        PlanEntry(
            session_key=item["session_key"],
            block_id=item["block_id"],
            task_id=item["task_id"],
            strategy_id=item["strategy_id"],
            repeat=item["repeat"],
            order=item["order"],
        )
        for item in plan["sessions"]
    ]
    if len(entries) != 96:
        raise ExperimentError("expected 96 sessions")
    if len({entry.session_key for entry in entries}) != 96:
        raise ExperimentError("session keys must be unique")

    results: list[SessionResult] = []
    total_calls = 0
    for entry in entries:
        task = tasks[entry.task_id]
        branch = _fake_branch(task.id, entry.repeat)
        branch_content = None
        if branch is not None:
            allowed = branches.get(task.id, {}).get(branch)
            if allowed is None:
                raise ExperimentError(f"invalid synthetic branch {task.id}:{branch}")
            branch_content = allowed["content"]
        session = FakeSession(entry, task, conditions[entry.strategy_id])
        fail_at = 2 if entry.session_key in failure_keys else None
        result = session.run(
            branch, branch_content, fail_at_call=fail_at
        )
        total_calls += len(result.calls)
        if total_calls > TOTAL_MAX_CALLS:
            raise ExperimentError("total call budget exceeded")
        results.append(result)

    completed = [r for r in results if r.status == "completed"]
    failed = [r for r in results if r.status != "completed"]
    budget_status = "partial_failure" if failed else "complete"
    if not failed and not (TOTAL_MIN_CALLS <= total_calls <= TOTAL_MAX_CALLS):
        raise ExperimentError(f"call budget out of bounds: {total_calls}")
    if failed and total_calls > TOTAL_MAX_CALLS:
        raise ExperimentError(f"call budget exceeded: {total_calls}")
    branch_counts: dict[str, int] = {}
    for result in results:
        if tasks[result.task_id].kind != "interactive":
            continue
        if result.branch is not None:
            branch_counts[result.branch] = branch_counts.get(result.branch, 0) + 1
        else:
            branch_counts["none"] = branch_counts.get("none", 0) + 1
    return {
        "schema_version": SCHEMA_VERSION,
        "synthetic": True,
        "source": "fake-branch-fixture; not a real model result",
        "planned_sessions": len(entries),
        "completed_sessions": len(completed),
        "failed_sessions": len(failed),
        "budget_status": budget_status,
        "normal_call_min": TOTAL_MIN_CALLS,
        "normal_call_max": TOTAL_MAX_CALLS,
        "total_calls": total_calls,
        "branch_counts": branch_counts,
        "sessions": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--fail-session", action="append", default=[])
    args = parser.parse_args()
    result = run_fake(args.root, fail_sessions=args.fail_session)
    serializable = {
        **{k: v for k, v in result.items() if k != "sessions"},
        "sessions": [
            {
                "session_key": r.session_key,
                "task_id": r.task_id,
                "strategy_id": r.strategy_id,
                "repeat": r.repeat,
                "branch": r.branch,
                "status": r.status,
                "failure_stage": r.failure_stage,
                "calls": len(r.calls),
            }
            for r in result["sessions"]
        ],
    }
    text = json.dumps(serializable, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        if args.output.exists():
            raise SystemExit(f"refusing to overwrite {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
