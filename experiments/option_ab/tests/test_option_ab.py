import asyncio
import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments.option_ab.freeze_manifest import (
    ASSET_VERSION,
    SHARED_BASE_SHA256,
    load_and_verify_manifest,
    sha256_bytes,
)
from experiments.option_ab.history import (
    MAX_HISTORY_MESSAGES,
    StoredMessage,
    build_request_messages,
    recent_history,
)
from experiments.option_ab.runner import (
    RunRecorder,
    load_and_verify_manifest,
    run_all,
    run_fixed_script,
    start_interactive_script,
)

ROOT = Path(__file__).resolve().parents[1]


class CancellingClient:
    synthetic = True

    def __init__(self, cancel_on: int) -> None:
        self.cancel_on = cancel_on
        self.calls = 0

    async def generate(self, messages, **kwargs):
        self.calls += 1
        if self.calls == self.cancel_on:
            raise asyncio.CancelledError()
        return "ok"


class OkClient:
    synthetic = True

    async def generate(self, messages, **kwargs):
        return "ok"


class FailingClient:
    synthetic = True

    async def generate(self, messages, **kwargs):
        raise RuntimeError("original execution failure")


class FixedCancelClient:
    synthetic = True

    def __init__(self) -> None:
        self.exception = asyncio.CancelledError("fixed cancellation")

    async def generate(self, messages, **kwargs):
        raise self.exception


class TestFrozenAssets(unittest.TestCase):
    def test_base_is_byte_identical(self):
        self.assertEqual(sha256_bytes((ROOT / "prompts/base.txt").read_bytes()), SHARED_BASE_SHA256)

    def test_system_assembly_and_manifest(self):
        base = (ROOT / "prompts/base.txt").read_text()
        strategy = (ROOT / "prompts/ab_strategy.txt").read_text()
        self.assertEqual((ROOT / "prompts/system.txt").read_text(), base.rstrip("\n") + "\n\n" + strategy.rstrip("\n") + "\n")
        manifest = load_and_verify_manifest(ROOT)
        self.assertEqual(manifest["condition"], "AB")
        self.assertEqual(manifest["asset_version"], ASSET_VERSION)
        self.assertEqual(manifest["calls"]["logical_runs"], 10)
        self.assertEqual((manifest["calls"]["total_min"], manifest["calls"]["total_max"]), (24, 26))
        self.assertLessEqual(len(manifest["prompt_version"]), 50)

    def test_scripts_have_ten_runs_and_budget(self):
        data = json.loads((ROOT / "scripts/scripts.json").read_text())
        self.assertEqual([s["id"] for s in data["scripts"]], ["S1", "S2", "S3", "S4", "S5", "S6", "I1", "I2", "I3", "I4"])
        self.assertEqual(data["logical_runs"], 10)
        self.assertEqual(data["budget"], {"fixed_calls": 16, "interactive_min": 8, "interactive_max": 10, "total_min": 24, "total_max": 26})
        self.assertEqual({b["id"] for b in data["scripts"][7]["branches"]}, {"bc", "direct_ac", "no_extra"})
        self.assertEqual({b["id"] for b in data["scripts"][8]["branches"]}, {"rule_supply", "direct_ac", "no_extra"})
        self.assertEqual(data["scripts"][9]["calls_expected"], 2)
        self.assertEqual(data["scripts"][9].get("branches", []), [])

    def test_fixed_inputs_match_a(self):
        a = json.loads((Path(__file__).resolve().parents[2] / "option_a/scripts/scripts.json").read_text())
        b = json.loads((ROOT / "scripts/scripts.json").read_text())
        amap = {s["id"]: [t["content"] for t in s["turns"]] for s in a["scripts"]}
        bmap = {s["id"]: [t["content"] for t in s["turns"]] for s in b["scripts"] if s["id"] in amap}
        self.assertEqual(amap, bmap)


class TestHistoryWindow(unittest.TestCase):
    def test_twenty_raw_rows_then_filter(self):
        rows = [StoredMessage("user" if i % 2 == 0 else "assistant", f"row-{i}") for i in range(25)]
        window = recent_history(rows, MAX_HISTORY_MESSAGES)
        self.assertEqual([row.content for row in window], [f"row-{i}" for i in range(5, 25)])
        messages = build_request_messages("SYSTEM", rows, "CURRENT")
        self.assertEqual(messages[0], {"role": "system", "content": "SYSTEM"})
        background = json.loads(messages[1]["content"])
        self.assertEqual([m["content"] for m in background["messages"]], [f"row-{i}" for i in range(5, 25) if i % 2 == 0])
        self.assertEqual(messages[-1], {"role": "user", "content": "CURRENT"})


class TestCancellation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = load_and_verify_manifest(ROOT)
        cls.system = (ROOT / "prompts/system.txt").read_text()
        cls.system_sha = sha256_bytes(cls.system.encode())
        cls.version = cls.manifest["prompt_version"]

    def _run_fixed(self, cancel_on: int):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        client = CancellingClient(cancel_on)
        with self.assertRaises(asyncio.CancelledError):
            asyncio.run(
                run_fixed_script(
                    __import__("experiments.option_ab.runner", fromlist=["load_scripts"]).load_scripts(ROOT)[0],
                    system_prompt=self.system,
                    system_sha256=self.system_sha,
                    prompt_version_value=self.version,
                    client=client,
                    output_root=Path(tmp.name),
                )
            )
        run_dirs = list(Path(tmp.name).iterdir())
        self.assertEqual(len(run_dirs), 1)
        turns = [json.loads(line) for line in (run_dirs[0] / "turns.jsonl").read_text().splitlines()]
        meta = json.loads((run_dirs[0] / "run_meta.json").read_text())
        self.assertEqual(client.calls, cancel_on)
        self.assertEqual(len(turns), cancel_on)
        self.assertEqual(turns[-1]["status"], "cancelled")
        self.assertEqual(meta["status"], "cancelled")
        self.assertEqual(meta["calls_actual"], cancel_on)
        self.assertFalse(any(item["status"] == "completed" for item in turns))

    def test_fixed_first_turn_cancelled(self):
        self._run_fixed(1)

    def test_fixed_q1_cancelled(self):
        self._run_fixed(2)

    def test_initial_interactive_q1_cancelled(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        client = CancellingClient(2)
        with self.assertRaises(asyncio.CancelledError):
            asyncio.run(
                start_interactive_script(
                    __import__("experiments.option_ab.runner", fromlist=["load_scripts"]).load_scripts(ROOT)[7],
                    system_prompt=self.system,
                    system_sha256=self.system_sha,
                    prompt_version_value=self.version,
                    client=client,
                    output_root=Path(tmp.name),
                )
            )
        run_dirs = list(Path(tmp.name).iterdir())
        self.assertEqual(len(run_dirs), 1)
        turns = [json.loads(line) for line in (run_dirs[0] / "turns.jsonl").read_text().splitlines()]
        meta = json.loads((run_dirs[0] / "run_meta.json").read_text())
        self.assertEqual(client.calls, 2)
        self.assertEqual(len(turns), 2)
        self.assertEqual(turns[-1]["status"], "cancelled")
        self.assertEqual(meta["status"], "cancelled")
        self.assertEqual(meta["calls_actual"], 2)


class TestCloseFailure(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = load_and_verify_manifest(ROOT)
        cls.system = (ROOT / "prompts/system.txt").read_text()
        cls.system_sha = sha256_bytes(cls.system.encode())
        cls.version = cls.manifest["prompt_version"]
        cls.script = __import__("experiments.option_ab.runner", fromlist=["load_scripts"]).load_scripts(ROOT)[0]

    def test_normal_success_then_close_failure_is_failed_and_stops_batch(self):
        with tempfile.TemporaryDirectory() as tmp:
            original_close = RunRecorder.close

            def close_then_fail(recorder):
                original_close(recorder)
                raise OSError("close failed")

            with patch.object(RunRecorder, "close", new=close_then_fail):
                result = asyncio.run(run_fixed_script(
                    self.script, system_prompt=self.system, system_sha256=self.system_sha,
                    prompt_version_value=self.version, client=OkClient(),
                    output_root=Path(tmp), run_id="normal-close-fail",
                ))
            self.assertEqual(result.status, "failed")
            self.assertEqual(result.calls, 2)
            meta = json.loads((Path(tmp) / "normal-close-fail/run_meta.json").read_text())
            self.assertEqual(meta["status"], "failed")
            self.assertEqual(meta["error"]["type"], "CleanupFailure")
            self.assertIsNotNone(meta["cleanup_error"])
            turns = [json.loads(line) for line in (Path(tmp) / "normal-close-fail/turns.jsonl").read_text().splitlines()]
            self.assertEqual(len(turns), 2)
            self.assertTrue(all(t["status"] == "ok" for t in turns))
        with tempfile.TemporaryDirectory() as tmp:
            original_close = RunRecorder.close

            def close_then_fail(recorder):
                original_close(recorder)
                raise OSError("close failed")

            with patch.object(RunRecorder, "close", new=close_then_fail):
                results = asyncio.run(run_all(asset_root=ROOT, output_root=Path(tmp)))
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0].script_id, "S1")
            self.assertEqual(results[0].status, "failed")
            self.assertEqual(len(list(Path(tmp).iterdir())), 1)

    def test_existing_failure_keeps_original_error_when_close_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            original_close = RunRecorder.close

            def close_then_fail(recorder):
                original_close(recorder)
                raise OSError("close failed")

            with patch.object(RunRecorder, "close", new=close_then_fail):
                result = asyncio.run(run_fixed_script(
                    self.script, system_prompt=self.system, system_sha256=self.system_sha,
                    prompt_version_value=self.version, client=FailingClient(),
                    output_root=Path(tmp), run_id="existing-fail-close",
                ))
            self.assertEqual(result.status, "failed")
            self.assertIn("original execution failure", result.error["message"])
            meta = json.loads((Path(tmp) / "existing-fail-close/run_meta.json").read_text())
            self.assertEqual(meta["status"], "failed")
            self.assertIn("original execution failure", meta["error"]["message"])
            self.assertIsNotNone(meta["cleanup_error"])
            self.assertEqual(meta["calls_actual"], 1)

    def test_cancel_object_survives_cleanup_and_meta_failures(self):
        client = FixedCancelClient()
        with tempfile.TemporaryDirectory() as tmp:
            original_close = RunRecorder.close

            def close_then_fail(recorder):
                original_close(recorder)
                raise OSError("close failed")

            with patch.object(RunRecorder, "close", new=close_then_fail):
                with patch.object(RunRecorder, "write_meta", side_effect=OSError("meta failed")):
                    with self.assertRaises(asyncio.CancelledError) as caught:
                        asyncio.run(run_fixed_script(
                            self.script, system_prompt=self.system, system_sha256=self.system_sha,
                            prompt_version_value=self.version, client=client,
                            output_root=Path(tmp), run_id="cancel-cleanup-meta",
                        ))
            self.assertIs(caught.exception, client.exception)


class TestFakeRunner(unittest.TestCase):
    def _run(self, choices=None):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        result = asyncio.run(run_all(asset_root=ROOT, output_root=Path(tmp.name), branch_choices=choices))
        return result, Path(tmp.name)

    def test_main_batch_branches(self):
        result, out = self._run({"I2": "bc", "I3": "rule_supply"})
        self.assertEqual([r.script_id for r in result], ["S1", "S2", "S3", "S4", "S5", "S6", "I1", "I2", "I3", "I4"])
        self.assertTrue(all(r.status == "completed" for r in result))
        self.assertEqual(sum(r.calls for r in result), 26)
        self.assertEqual(result[7].branch_choice, "bc")
        self.assertEqual(result[8].branch_choice, "rule_supply")
        self.assertEqual(result[9].calls, 2)
        for run in result:
            meta = json.loads((Path(run.run_dir) / "run_meta.json").read_text())
            self.assertEqual(meta["condition"], "AB")
            self.assertEqual(meta["asset_version"], ASSET_VERSION)
            self.assertIn("turns.jsonl", {p.name for p in Path(run.run_dir).iterdir()})

    def test_none_branches_are_24(self):
        result, _ = self._run({"I2": "none", "I3": "none"})
        self.assertTrue(all(r.status == "completed" for r in result))
        self.assertEqual(sum(r.calls for r in result), 24)
        self.assertEqual(result[7].branch_choice, "none")
        self.assertEqual(result[8].branch_choice, "none")

    def test_without_branch_choices_pauses(self):
        result, _ = self._run({})
        self.assertEqual(len(result), 10)
        self.assertEqual(result[7].status, "paused")
        self.assertEqual(result[8].status, "paused")
        self.assertEqual(result[9].status, "completed")
        self.assertEqual([r.script_id for r in result], ["S1", "S2", "S3", "S4", "S5", "S6", "I1", "I2", "I3", "I4"])

    def test_all_i2_i3_routes(self):
        for i2 in ("none", "bc", "direct_ac", "no_extra"):
            for i3 in ("none", "rule_supply", "direct_ac", "no_extra"):
                result, _ = self._run({"I2": i2, "I3": i3})
                self.assertTrue(all(r.status == "completed" for r in result), (i2, i3))
                expected = 24 + (1 if i2 != "none" else 0) + (1 if i3 != "none" else 0)
                self.assertEqual(sum(r.calls for r in result), expected, (i2, i3))

    def test_t2_is_the_only_final_generation(self):
        result, _ = self._run({"I2": "bc", "I3": "none"})
        run = result[7]
        self.assertEqual(run.calls, 3)
        turns = [json.loads(line) for line in (Path(run.run_dir) / "turns.jsonl").read_text().splitlines()]
        self.assertEqual([t["turn_id"] for t in turns], ["T1", "Q1", "T2"])
        self.assertFalse(any(t.get("turn_id") == "Q2" for t in turns))
        self.assertEqual(turns[-1]["final_assistant"], True)
        self.assertEqual(turns[-1]["final_for_user_turn_id"], "T2")

    def test_coding_sheet_contract_and_blank_actuals(self):
        with (ROOT / "expected/coding_sheet.csv").open() as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
        required = {
            "script", "turn", "phase", "branch", "generation_expected",
            "expected_question_compliance", "expected_information_gain",
            "expected_model_valid_r1_completed", "expected_source_evidence_correct",
            "expected_direct_answer_correct", "expected_full_question_chain_observed",
            "expected_target_request_violation",
            "actual_execution", "actual_content_correct", "actual_policy_compliant",
            "actual_question_compliance", "actual_information_gain",
            "actual_model_valid_r1_completed", "actual_source_evidence_correct",
            "actual_direct_answer_correct", "actual_full_question_chain_observed",
            "actual_target_request_violation", "model_claims", "human_verified_evidence",
        }
        self.assertTrue(required.issubset(reader.fieldnames or []))
        self.assertEqual(len(rows), 32)
        keys = {(r["script"], r["turn"], r["branch"]) for r in rows}
        self.assertEqual(len(keys), 32)
        branch_rows = {(r["script"], r["branch"]) for r in rows if r["script"] in {"I2", "I3"} and r["branch"]}
        self.assertEqual(branch_rows, {
            ("I2", "none"), ("I2", "bc"), ("I2", "direct_ac"), ("I2", "no_extra"),
            ("I3", "none"), ("I3", "rule_supply"), ("I3", "direct_ac"), ("I3", "no_extra"),
        })
        for row in rows:
            self.assertEqual(row["actual_execution"], "")
            self.assertEqual(row["actual_policy_compliant"], "")
            self.assertEqual(row["actual_model_valid_r1_completed"], "")
            self.assertEqual(row["actual_information_gain"], "")
            self.assertEqual(row["model_claims"], "")
            self.assertEqual(row["human_verified_evidence"], "")
        by_key = {(r["script"], r["turn"], r["branch"]): r for r in rows}
        self.assertEqual(by_key[("S1", "Q1", "")]["expected_question_compliance"], "not_needed")
        self.assertEqual(by_key[("S3", "Q1", "")]["expected_target_request_violation"], "if_direct_ac")
        self.assertEqual(by_key[("I2", "Q1", "")]["expected_question_compliance"], "optional")
        self.assertEqual(by_key[("I2", "T2/FINAL", "none")]["expected_information_gain"], "N/A")
        self.assertEqual(by_key[("I2", "T2/FINAL", "none")]["generation_expected"], "no")
        self.assertEqual(by_key[("I2", "T2/FINAL", "bc")]["expected_information_gain"], "2")
        self.assertEqual(by_key[("I2", "T2/FINAL", "bc")]["expected_full_question_chain_observed"], "if_question_and_r1")
        self.assertEqual(by_key[("I2", "T2/FINAL", "direct_ac")]["expected_target_request_violation"], "inherited_from_q1")
        self.assertEqual(by_key[("I3", "T2/FINAL", "rule_supply")]["expected_model_valid_r1_completed"], "if_evidence_valid")
        self.assertEqual(by_key[("I3", "T2/FINAL", "no_extra")]["expected_full_question_chain_observed"], "no")

    def test_cleanup_and_meta_failures_preserve_cancellation(self):
        script = __import__("experiments.option_ab.runner", fromlist=["load_scripts"]).load_scripts(ROOT)[0]
        original_close = RunRecorder.close

        def close_then_fail(recorder):
            original_close(recorder)
            raise OSError("close failed")

        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(RunRecorder, "close", new=close_then_fail):
                with self.assertRaises(asyncio.CancelledError):
                    asyncio.run(run_fixed_script(
                        script, system_prompt="sys", system_sha256="0" * 64,
                        prompt_version_value="v", client=CancellingClient(1),
                        output_root=Path(tmp), run_id="cancel-close",
                    ))
            run_dir = Path(tmp) / "cancel-close"
            self.assertEqual(json.loads((run_dir / "run_meta.json").read_text())["status"], "cancelled")
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(RunRecorder, "write_meta", side_effect=OSError("meta failed")):
                with self.assertRaises(asyncio.CancelledError):
                    asyncio.run(run_fixed_script(
                        script, system_prompt="sys", system_sha256="0" * 64,
                        prompt_version_value="v", client=CancellingClient(1),
                        output_root=Path(tmp), run_id="cancel-meta",
                    ))


if __name__ == "__main__":
    unittest.main()
