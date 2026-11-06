from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import pytest

from backend.teaching_strategies import get_teaching_prompt, get_teaching_prompt_version
from experiments.strategy_comparison.freeze_manifest import (
    FrozenAssetError,
    verify_manifest,
    write_manifest,
)
from experiments.strategy_comparison.runner import (
    ExperimentError,
    FakeSession,
    PlanEntry,
    _fake_branch,
    load_assets,
    run_fake,
)

ROOT = Path(__file__).resolve().parents[1]
TASK_IDS = ["S1", "S2", "S3", "S4", "S5", "S6", "I2", "I3"]
STRATEGIES = ["standard", "a", "b", "ab"]


def _json(name: str):
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def test_conditions_match_production_prompts_and_common_model_policy():
    conditions = _json("conditions.json")["conditions"]
    assert [c["strategy_id"] for c in conditions] == STRATEGIES
    for condition in conditions:
        prompt = (ROOT / condition["prompt_file"]).read_text(encoding="utf-8")
        assert prompt.strip() == get_teaching_prompt(condition["switch_value"]).strip()
        assert condition["profile_id"] == "default"
        assert condition["model"] == "deepseek-v4-flash"
        assert condition["generation"]["temperature"] is None
        assert condition["generation"]["max_tokens"] is None
        assert condition["generation"]["reasoning_effort"] is None
        assert condition["prompt_version"].endswith("+l1-en")
        assert len(condition["prompt_version"]) <= 50
    assert len({c["switch_value"] for c in conditions}) == 4


def test_tasks_are_common_and_have_the_frozen_counts():
    tasks = _json("tasks/tasks.json")["tasks"]
    assert [task["id"] for task in tasks] == TASK_IDS
    by_id = {task["id"]: task for task in tasks}
    for task in tasks:
        for turn in task["turns"]:
            assert turn["content"].strip()
            assert not any(
                strategy in turn["content"].lower()
                for strategy in ("standard", "strategy a", "strategy b", "a+b")
            )
    for task_id in ("S1", "S2", "S3", "S4"):
        assert len(by_id[task_id]["turns"]) == 2
    for task_id in ("S5", "S6"):
        assert len(by_id[task_id]["turns"]) == 4
    for task_id in ("I2", "I3"):
        assert by_id[task_id]["kind"] == "interactive"
        assert len(by_id[task_id]["turns"]) == 2
        assert by_id[task_id]["calls"] == {"none": 2, "branch": 3}


def test_initial_teaching_is_byte_identical_across_conditions():
    tasks = _json("tasks/tasks.json")["tasks"]
    # A condition has no independent copy of user turns; all conditions share
    # the single task table and there are no group-specific user instructions.
    user_text = " ".join(
        turn["content"] for task in tasks for turn in task["turns"]
    ).lower()
    assert "a+b" not in user_text
    assert "b is allowed" not in user_text
    assert "do not infer" not in user_text
    assert "because" not in user_text


def test_run_plan_has_24_blocks_and_96_fresh_sessions():
    plan = _json("run_plan.json")
    assert plan["seed"] == 20261106
    assert len(plan["blocks"]) == 24
    assert len(plan["sessions"]) == 96
    for block in plan["blocks"]:
        assert sorted(block["order"]) == sorted(STRATEGIES)
        assert len(set(block["order"])) == 4
    keys = [entry["session_key"] for entry in plan["sessions"]]
    assert len(keys) == len(set(keys)) == 96
    for entry in plan["sessions"]:
        assert entry["strategy_id"] in STRATEGIES
        assert entry["task_id"] in TASK_IDS


def test_budget_and_pre_pilot_are_reconciled():
    plan = _json("run_plan.json")
    assert plan["fixed_calls"] == 192
    assert plan["interactive_min_calls"] == 48
    assert plan["interactive_max_calls"] == 72
    assert (plan["total_min_calls"], plan["total_max_calls"]) == (240, 264)
    assert plan["pre_pilot"] == {"sessions": 32, "calls_min": 80, "calls_max": 88}
    # 6 fixed tasks x 4 strategies x 3 repeats x 2 = 144? Correct frozen
    # accounting uses S1-S4=2 and S5-S6=4 per repeat:
    fixed_per_strategy_repeat = (4 * 2) + (2 * 4)
    assert fixed_per_strategy_repeat == 16
    assert fixed_per_strategy_repeat * 4 * 3 == 192


def test_fake_run_is_synthetic_and_budgeted():
    result = run_fake(ROOT)
    assert result["synthetic"] is True
    assert result["planned_sessions"] == 96
    assert result["completed_sessions"] == 96
    assert result["failed_sessions"] == 0
    assert result["budget_status"] == "complete"
    assert result["normal_call_min"] == 240
    assert result["normal_call_max"] == 264
    assert result["total_calls"] == 256
    assert result["branch_counts"] == {
        "bc": 4,
        "rule_supply": 4,
        "no_extra": 8,
        "none": 8,
    }
    assert all(session.status == "completed" for session in result["sessions"])
    assert all(session.strategy_id in STRATEGIES for session in result["sessions"])


def test_sessions_are_independent_and_have_strategy_versions():
    conditions, tasks, plan, branches = load_assets(ROOT)
    entries = plan["sessions"][:2]
    sessions = []
    for raw in entries:
        entry = PlanEntry(
            session_key=raw["session_key"],
            block_id=raw["block_id"],
            task_id=raw["task_id"],
            strategy_id=raw["strategy_id"],
            repeat=raw["repeat"],
            order=raw["order"],
        )
        session = FakeSession(entry, tasks[entry.task_id], conditions[entry.strategy_id])
        session.run(None)
        sessions.append(session)
    assert sessions[0].result.calls is not sessions[1].result.calls
    assert sessions[0].entry.session_key != sessions[1].entry.session_key
    for session in sessions:
        assert all(
            call.prompt_version == session.condition.prompt_version
            for call in session.result.calls
        )


def test_none_is_two_calls_and_branch_is_one_final_without_hidden_q2():
    conditions, tasks, plan, branches = load_assets(ROOT)
    raw = plan["sessions"][0]

    def make_session():
        entry = PlanEntry(
            session_key=raw["session_key"],
            block_id=raw["block_id"],
            task_id="I2",
            strategy_id="ab",
            repeat=3,
            order=raw["order"],
        )
        return FakeSession(entry, tasks["I2"], conditions["ab"])

    none = make_session()
    none.run(None)
    assert len(none.result.calls) == 2
    assert none.result.branch is None

    branched = make_session()
    branched.run("bc", branches["I2"]["bc"]["content"])
    assert len(branched.result.calls) == 3
    assert branched.result.calls[-1].turn_id == "T2"
    assert branched.result.calls[-1].content.startswith("I teach you one missing fact")
    with pytest.raises(ExperimentError):
        branched.run("bc", branches["I2"]["bc"]["content"])


def test_direct_target_branch_is_present_and_not_used_for_full_fake_budget():
    branches = _json("branches/branches.json")
    assert branches["I2"]["direct_ac"]["request_type"] == "direct_target"
    assert branches["I3"]["direct_ac"]["request_type"] == "direct_target"
    assert "A+B" in (ROOT / "expected/strategy_expectations.md").read_text()
    assert "Legal only for B" in (ROOT / "expected/strategy_expectations.md").read_text()
    assert _fake_branch("I2", 1) == "bc"
    assert _fake_branch("I3", 1) == "rule_supply"
    assert _fake_branch("I2", 2) == "no_extra"
    assert _fake_branch("I2", 3) is None
    # The fake branch fixture does not inspect a strategy argument at all.
    assert _fake_branch("I3", 3) is None


def test_failure_is_retained_and_does_not_retry():
    conditions, tasks, plan, branches = load_assets(ROOT)
    raw = plan["sessions"][0]
    entry = PlanEntry(
        session_key=raw["session_key"],
        block_id=raw["block_id"],
        task_id=raw["task_id"],
        strategy_id=raw["strategy_id"],
        repeat=raw["repeat"],
        order=raw["order"],
    )
    session = FakeSession(entry, tasks[entry.task_id], conditions[entry.strategy_id])
    session.run(None, fail_at_call=2)
    assert session.result.status == "failed"
    assert session.result.failure_stage == "synthetic_transport_failure"
    assert len(session.result.calls) == 1
    with pytest.raises(ExperimentError):
        session.run(None)


def test_fake_batch_retains_a_failed_session():
    target = _json("run_plan.json")["sessions"][0]["session_key"]
    result = run_fake(ROOT, fail_session=target)
    assert result["failed_sessions"] == 1
    failed = [s for s in result["sessions"] if s.status == "failed"]
    assert len(failed) == 1
    assert failed[0].session_key == target
    assert failed[0].failure_stage == "synthetic_transport_failure"
    assert result["budget_status"] == "partial_failure"
    assert result["total_calls"] == 255


def test_multiple_failures_can_go_below_normal_minimum_without_replacement():
    keys = [item["session_key"] for item in _json("run_plan.json")["sessions"][:20]]
    result = run_fake(ROOT, fail_sessions=keys)
    assert result["failed_sessions"] == 20
    assert result["budget_status"] == "partial_failure"
    assert [s.session_key for s in result["sessions"] if s.status == "failed"] == keys
    assert result["total_calls"] < result["normal_call_min"]
    assert result["total_calls"] <= result["normal_call_max"]


def test_blind_sheet_has_no_strategy_labels_and_key_is_separate():
    rows = list(csv.DictReader((ROOT / "scoring/blind_coding_sheet.csv").open()))
    assert len(rows) == 96
    assert rows[0].keys() >= {"rating_id", "task_id", "repeat", "notes"}
    joined = " ".join(
        value for row in rows for value in row.values()
    ).lower()
    for label in STRATEGIES:
        assert label not in joined.split()
    assert "baseline" not in joined
    key = _json("scoring/blind_key.json")["key"]
    assert len(key) == 96
    assert {item["rating_id"] for item in key} == {
        int(row["rating_id"]) for row in rows
    }
    assert {item["strategy_id"] for item in key} == set(STRATEGIES)
    stage1_fields = set(rows[0].keys())
    assert "clarification_boundary" in stage1_fields
    assert "strategy_id" not in stage1_fields
    assert "strategy_compliance" not in stage1_fields
    assert "branch_valid" not in stage1_fields
    assert "full_chain" not in stage1_fields
    stage2 = list(csv.DictReader((ROOT / "scoring/stage2_compliance_sheet.csv").open()))
    assert len(stage2) == 96
    assert {"strategy_id", "strategy_compliance", "branch_valid", "full_chain"} <= set(stage2[0])
    lock = _json("scoring/stage1_lock.json")
    assert lock["locked"] is False and lock["stage1_sha256"] is None


def test_manifest_verifies_and_detects_mutation(tmp_path):
    root = tmp_path / "strategy"
    shutil.copytree(ROOT, root)
    write_manifest(root)
    verify_manifest(root)
    task_path = root / "tasks/S1.txt"
    task_path.write_text(task_path.read_text() + "mutated\n", encoding="utf-8")
    with pytest.raises(FrozenAssetError, match="(bytes|sha256) mismatch"):
        verify_manifest(root)


def test_manifest_has_expected_number_of_files():
    manifest = _json("manifest.json")
    included = {item["path"] for item in manifest["files"]}
    assert "conditions.json" in included
    assert "run_plan.json" in included
    assert "tasks/tasks.json" in included
    assert "scoring/blind_coding_sheet.csv" in included
    assert "scoring/blind_key.json" in included
    assert "README.md" in included
    assert "protocol.md" in included
    assert len(included) == 24
