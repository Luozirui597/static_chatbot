"""Direct isolation tests for the fake Option B package."""

from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments.option_b import runner as b

ASSET_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ASSET_ROOT.parents[1]
A_ROOT = REPO_ROOT / "experiments" / "option_a"
SYSTEM_PROMPT = b.load_system_prompt(ASSET_ROOT)
SYSTEM_SHA256 = b.sha256_text(SYSTEM_PROMPT)
PROMPT_VERSION = b.prompt_version(SYSTEM_SHA256)
SCRIPTS = b.load_scripts(ASSET_ROOT)


def script(script_id: str) -> b.ScriptSpec:
    return next(item for item in SCRIPTS if item.id == script_id)


def branch(script_id: str, branch_id: str) -> b.BranchSpec:
    return next(item for item in script(script_id).branches if item.id == branch_id)


def read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


class OptionBAssetTests(unittest.TestCase):
    def test_shared_base_and_fixed_turns_are_identical_to_a(self) -> None:
        a_base = (A_ROOT / "prompts" / "base.txt").read_bytes()
        b_base = (ASSET_ROOT / "prompts" / "base.txt").read_bytes()
        self.assertEqual(a_base, b_base)
        self.assertEqual(
            hashlib.sha256(b_base).hexdigest(),
            "eda835c58da3c1cbd05d83d5ac34f4ff5bbd805b8a466796957b7b489e00bb6f",
        )

        a_scripts = read_json(A_ROOT / "scripts" / "scripts.json")
        b_scripts = read_json(ASSET_ROOT / "scripts" / "scripts.json")
        a_fixed = {item["id"]: item for item in a_scripts["scripts"] if item["id"].startswith("S")}
        b_fixed = {item["id"]: item for item in b_scripts["scripts"] if item["id"].startswith("S")}
        self.assertEqual(set(a_fixed), set(b_fixed))
        for script_id in b_fixed:
            self.assertEqual(
                [turn["content"] for turn in a_fixed[script_id]["turns"]],
                [turn["content"] for turn in b_fixed[script_id]["turns"]],
            )
            text = (ASSET_ROOT / "scripts" / f"{script_id}.txt").read_text(
                encoding="utf-8"
            )
            self.assertIn("# Condition: B", text)
            self.assertNotIn("# Condition: A", text)
            if script_id == "S1":
                self.assertIn("Direct fact and unsupported relation", text)
            if script_id == "S5":
                self.assertIn("Correction of an untaught relation", text)
            if script_id == "S6":
                self.assertIn("Revocation of a rule", text)
            for turn in b_fixed[script_id]["turns"]:
                self.assertIn(turn["content"], text)

    def test_system_assembly_and_frozen_spec(self) -> None:
        base = (ASSET_ROOT / "prompts" / "base.txt").read_text(encoding="utf-8")
        strategy = (ASSET_ROOT / "prompts" / "b_strategy.txt").read_text(
            encoding="utf-8"
        )
        expected = base.rstrip("\n") + "\n\n" + strategy.rstrip("\n") + "\n"
        self.assertEqual(SYSTEM_PROMPT, expected)
        self.assertEqual(
            hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
            "44aed8837a13413565a6dee866e00671a6fcefb3cc6f5f885dd78ac8495f9113",
        )
        self.assertEqual(PROMPT_VERSION, "option-b-isolated-v0.1+sys-44aed8837a13")
        self.assertLessEqual(len(PROMPT_VERSION), 50)
        manifest = b.load_and_verify_manifest(ASSET_ROOT)
        self.assertEqual(manifest["condition"], "B")
        self.assertEqual(manifest["calls"]["fixed_total"], 16)
        self.assertEqual(manifest["calls"]["total_min"], 22)
        self.assertEqual(manifest["calls"]["total_max"], 24)
        self.assertEqual(manifest["calls"]["logical_runs"], 9)

    def test_manifest_detects_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            copy = Path(raw) / "option_b"
            shutil.copytree(
                ASSET_ROOT,
                copy,
                ignore=shutil.ignore_patterns("__pycache__", "runs"),
            )
            system = copy / "prompts" / "system.txt"
            system.write_text(system.read_text(encoding="utf-8") + "tamper\n", encoding="utf-8")
            with self.assertRaises(b.FrozenAssetError):
                b.load_and_verify_manifest(copy)

    def test_branch_text_and_increment_are_frozen(self) -> None:
        self.assertEqual(
            [item.id for item in script("I2").branches],
            ["bc", "direct_ac", "no_extra"],
        )
        self.assertEqual(
            [item.id for item in script("I3").branches],
            ["rule_request", "direct_ac", "no_extra"],
        )
        self.assertEqual(branch("I2", "bc").adds, ("B-C",))
        self.assertEqual(branch("I2", "direct_ac").adds, ("A-C",))
        self.assertIn("I have not taught B links to C", branch("I2", "direct_ac").content)
        i3_no_extra = branch("I3", "no_extra").content
        self.assertIn("I am not adding any fact or rule in this reply.", i3_no_extra)
        self.assertIn("I have not taught a rule for combining links.", i3_no_extra)
        self.assertNotIn("I have no combination rule", i3_no_extra)

    def test_history_uses_last_twenty_rows_and_filters_assistant(self) -> None:
        rows = []
        for index in range(25):
            rows.append(b.StoredMessage(role="user", content=f"u{index}"))
            rows.append(b.StoredMessage(role="assistant", content=f"a{index}"))
        recent = b.__dict__["recent_history"](rows)
        self.assertEqual(len(recent), 20)
        projected = b.__dict__["project_receive"](recent)
        self.assertEqual([item.content for item in projected], [f"u{15 + i // 2}" for i in range(0, 20, 2)])
        encoded = b.__dict__["serialize_receive_history"](projected)
        payload = json.loads(encoded)
        self.assertEqual(payload["type"], "receive_teaching_history_background")
        self.assertEqual(payload["version"], "history-context-b1")
        self.assertEqual([item["index"] for item in payload["messages"]], list(range(1, 11)))

        messages = b.build_request_messages(SYSTEM_PROMPT, rows, "CURRENT")
        self.assertEqual(messages[0], {"role": "system", "content": SYSTEM_PROMPT})
        self.assertEqual(messages[-1], {"role": "user", "content": "CURRENT"})
        self.assertEqual(sum(1 for message in messages if message["content"] == "CURRENT"), 1)
        self.assertEqual(messages[1]["role"], "user")
        self.assertIn("u15", messages[1]["content"])

    def test_history_matches_a_contract(self) -> None:
        a_history_path = A_ROOT / "history.py"
        spec = importlib.util.spec_from_file_location("option_a_history_for_b_test", a_history_path)
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        rows = [
            module.StoredMessage(role="user" if i % 2 == 0 else "assistant", content=str(i))
            for i in range(30)
        ]
        a_rows = b.__dict__["StoredMessage"]
        b_rows = [a_rows(role=item.role, content=item.content) for item in rows]
        self.assertEqual(
            module.build_request_messages("SYS", rows, "NOW"),
            b.build_request_messages("SYS", b_rows, "NOW"),
        )


class OptionBFakeRunTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.out = Path(self.tmp.name)

    async def test_fixed_group_is_six_isolated_runs_and_sixteen_calls(self) -> None:
        results = []
        for script_id in b.FIXED_IDS:
            fake = b.FakeLLMClient()
            result = await b.run_fixed_script(
                script(script_id),
                system_prompt=SYSTEM_PROMPT,
                system_sha256=SYSTEM_SHA256,
                prompt_version_value=PROMPT_VERSION,
                client=fake,
                output_root=self.out,
            )
            results.append(result)
            self.assertEqual(result.status, "completed")
            self.assertEqual(result.calls, b.FIXED_CALLS[script_id])
            self.assertEqual(len(fake.calls), b.FIXED_CALLS[script_id])
            self.assertEqual(len(result.records), b.FIXED_CALLS[script_id])
            self.assertTrue(all(record["condition"] == "B" for record in result.records))
            self.assertTrue(all(record["final_assistant"] is False for record in result.records))
        self.assertEqual(sum(result.calls for result in results), 16)
        self.assertEqual(len(results), 6)
        self.assertEqual(len(list(self.out.iterdir())), 6)

    async def test_interactive_i1_has_no_branch(self) -> None:
        fake = b.FakeLLMClient()
        result = await b.run_interactive_script(
            script("I1"),
            system_prompt=SYSTEM_PROMPT,
            system_sha256=SYSTEM_SHA256,
            prompt_version_value=PROMPT_VERSION,
            client=fake,
            output_root=self.out,
            branch_choice="none",
        )
        self.assertEqual(result.status, "completed")
        self.assertEqual(result.calls, 2)
        self.assertEqual(len(fake.calls), 2)
        self.assertEqual([record["turn_id"] for record in result.records], ["T1", "Q1"])
        self.assertFalse(any(record["turn_id"] == "Q2" for record in result.records))

    async def test_interactive_pauses_after_q1_and_branch_final_is_one_call(self) -> None:
        fake = b.FakeLLMClient()
        paused, paused_result = await b.start_interactive_script(
            script("I2"),
            system_prompt=SYSTEM_PROMPT,
            system_sha256=SYSTEM_SHA256,
            prompt_version_value=PROMPT_VERSION,
            client=fake,
            output_root=self.out,
        )
        self.assertIsNotNone(paused)
        assert paused is not None
        self.assertEqual(paused_result.status, "paused")
        self.assertEqual(paused_result.calls, 2)
        self.assertEqual(len(fake.calls), 2)
        self.assertEqual(paused.ledger.index, 2)

        final = await b.resume_interactive_script(paused, "bc")
        self.assertEqual(final.status, "completed")
        self.assertEqual(final.calls, 3)
        self.assertEqual(len(fake.calls), 3)
        self.assertEqual(len(final.records), 3)
        self.assertEqual([record["turn_id"] for record in final.records], ["T1", "Q1", "T2"])
        branch_record = final.records[-1]
        self.assertTrue(branch_record["final_assistant"])
        self.assertEqual(branch_record["final_for_user_turn_id"], "T2")
        self.assertEqual(branch_record["branch"]["id"], "bc")
        self.assertIn("B links to C", branch_record["current_user"])
        self.assertNotIn("Does A link to C", branch_record["response"])
        self.assertFalse(any(record["turn_id"] == "Q2" for record in final.records))

    async def test_none_branch_has_two_calls_and_no_t2(self) -> None:
        fake = b.FakeLLMClient()
        result = await b.run_interactive_script(
            script("I3"),
            system_prompt=SYSTEM_PROMPT,
            system_sha256=SYSTEM_SHA256,
            prompt_version_value=PROMPT_VERSION,
            client=fake,
            output_root=self.out,
            branch_choice="none",
        )
        self.assertEqual(result.status, "completed")
        self.assertEqual(result.calls, 2)
        self.assertEqual([record["turn_id"] for record in result.records], ["T1", "Q1"])
        self.assertEqual(result.branch_choice, "none")
        self.assertFalse(any(record["branch"] is not None for record in result.records))

    async def test_invalid_branch_rejected_without_extra_call(self) -> None:
        fake = b.FakeLLMClient()
        paused, _ = await b.start_interactive_script(
            script("I2"),
            system_prompt=SYSTEM_PROMPT,
            system_sha256=SYSTEM_SHA256,
            prompt_version_value=PROMPT_VERSION,
            client=fake,
            output_root=self.out,
        )
        assert paused is not None
        with self.assertRaises(b.TurnOrderError):
            await b.resume_interactive_script(paused, "leading")
        self.assertEqual(len(fake.calls), 2)
        paused.recorder.close()

    async def test_empty_or_exception_stops_without_retry(self) -> None:
        blank = b.StaticReplyClient("   ")
        blank_result = await b.run_fixed_script(
            script("S1"),
            system_prompt=SYSTEM_PROMPT,
            system_sha256=SYSTEM_SHA256,
            prompt_version_value=PROMPT_VERSION,
            client=blank,
            output_root=self.out,
        )
        self.assertEqual(blank_result.status, "failed")
        self.assertEqual(blank_result.calls, 1)
        self.assertEqual(len(blank.calls), 1)
        self.assertEqual(blank_result.records[0]["response"], "   ")
        self.assertEqual(blank_result.records[0]["status"], "error")

        raising = b.RaisingClient(RuntimeError("synthetic failure"))
        failed = await b.run_fixed_script(
            script("S1"),
            system_prompt=SYSTEM_PROMPT,
            system_sha256=SYSTEM_SHA256,
            prompt_version_value=PROMPT_VERSION,
            client=raising,
            output_root=self.out,
        )
        self.assertEqual(failed.status, "failed")
        self.assertEqual(failed.calls, 1)
        self.assertEqual(len(raising.calls), 1)
        self.assertEqual(failed.records[0]["status"], "error")

    async def test_full_batch_budget_22_or_24(self) -> None:
        no_branch = await b.run_all(
            asset_root=ASSET_ROOT,
            output_root=self.out,
            run_prefix="budget-none",
            branch_choices={"I2": "none", "I3": "none"},
        )
        self.assertEqual(len(no_branch), 9)
        self.assertTrue(all(result.status == "completed" for result in no_branch))
        self.assertEqual(sum(result.calls for result in no_branch), 22)

        with_branches = await b.run_all(
            asset_root=ASSET_ROOT,
            output_root=self.out,
            run_prefix="budget-branch",
            branch_choices={"I2": "bc", "I3": "rule_request"},
        )
        self.assertEqual(len(with_branches), 9)
        self.assertTrue(all(result.status == "completed" for result in with_branches))
        self.assertEqual(sum(result.calls for result in with_branches), 24)
        self.assertEqual(
            {result.script_id: result.calls for result in with_branches},
            {"S1": 2, "S2": 2, "S3": 2, "S4": 2, "S5": 4, "S6": 4, "I1": 2, "I2": 3, "I3": 3},
        )

    async def test_existing_run_directory_is_not_overwritten(self) -> None:
        fake = b.FakeLLMClient()
        kwargs = dict(
            script=script("S1"),
            system_prompt=SYSTEM_PROMPT,
            system_sha256=SYSTEM_SHA256,
            prompt_version_value=PROMPT_VERSION,
            client=fake,
            output_root=self.out,
            run_id="same-run",
        )
        first = await b.run_fixed_script(**kwargs)
        self.assertEqual(first.status, "completed")
        with self.assertRaises(FileExistsError):
            await b.run_fixed_script(**kwargs)


class OptionBLedgerTests(unittest.TestCase):
    def test_sequence_content_duplicate_and_budget(self) -> None:
        turns = [
            b.TurnSpec("T1", "teach", "one"),
            b.TurnSpec("Q1", "query", "two"),
        ]
        ledger = b.TurnLedger("r", "S1", turns, max_calls=2)
        with self.assertRaises(b.TurnOrderError):
            ledger.begin("Q1", "two")
        ledger.begin("T1", "one")
        ledger.commit()
        with self.assertRaises(b.TurnOrderError):
            ledger.begin("T1", "one")
        with self.assertRaises(b.TurnOrderError):
            ledger.begin("Q1", "wrong")
        ledger.begin("Q1", "two")
        ledger.commit()
        ledger.ensure_complete()
        with self.assertRaises(b.TurnOrderError):
            ledger.begin("Q1", "two")

    def test_dynamic_t2_requires_allowed_branch_content(self) -> None:
        turns = [
            b.TurnSpec("T1", "teach", "one"),
            b.TurnSpec("Q1", "query", "two"),
            b.TurnSpec("T2", "branch_final", ""),
        ]
        ledger = b.TurnLedger("r", "I2", turns, max_calls=3)
        ledger.begin("T1", "one")
        ledger.commit()
        ledger.begin("Q1", "two")
        ledger.commit()
        with self.assertRaises(b.TurnOrderError):
            ledger.begin("T2", "not the branch", allowed_contents=("allowed",))
        ledger.begin("T2", "allowed", allowed_contents=("allowed",))
        ledger.commit()
        ledger.ensure_complete()

    def test_over_budget_is_rejected_before_attempt(self) -> None:
        turns = [b.TurnSpec("T1", "teach", "one"), b.TurnSpec("Q1", "query", "two")]
        ledger = b.TurnLedger("r", "S1", turns, max_calls=1)
        ledger.begin("T1", "one")
        ledger.commit()
        with self.assertRaises(b.TurnOrderError):
            ledger.begin("Q1", "two")
        self.assertEqual(ledger.attempts, 1)

    def test_new_ledger_is_isolated(self) -> None:
        turns = [b.TurnSpec("T1", "teach", "one")]
        first = b.TurnLedger("r1", "S1", turns, max_calls=1)
        second = b.TurnLedger("r2", "S1", turns, max_calls=1)
        first.begin("T1", "one")
        second.begin("T1", "one")
        self.assertEqual(second.completed, ())
        self.assertEqual(second.attempts, 1)


class OptionBBranchTerminalTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.out = Path(self.tmp.name)

    async def _paused_i2(self, client: object) -> b.PausedInteractive:
        paused, _ = await b.start_interactive_script(
            script("I2"),
            system_prompt=SYSTEM_PROMPT,
            system_sha256=SYSTEM_SHA256,
            prompt_version_value=PROMPT_VERSION,
            client=client,  # type: ignore[arg-type]
            output_root=self.out,
        )
        assert paused is not None
        return paused

    async def test_none_completed_cannot_choose_again(self) -> None:
        fake = b.FakeLLMClient()
        paused = await self._paused_i2(fake)
        first = await b.resume_interactive_script(paused, "none")
        self.assertEqual(first.status, "completed")
        self.assertEqual(first.calls, 2)
        self.assertEqual(len(fake.calls), 2)
        with self.assertRaises(b.TurnOrderError):
            await b.resume_interactive_script(paused, "bc")
        self.assertEqual(len(fake.calls), 2)
        self.assertTrue(paused.recorder.closed)

    async def test_completed_branch_cannot_choose_again(self) -> None:
        fake = b.FakeLLMClient()
        paused = await self._paused_i2(fake)
        first = await b.resume_interactive_script(paused, "bc")
        self.assertEqual(first.status, "completed")
        self.assertEqual(len(fake.calls), 3)
        with self.assertRaises(b.TurnOrderError):
            await b.resume_interactive_script(paused, "none")
        self.assertEqual(len(fake.calls), 3)

    async def test_failed_branch_cannot_choose_again(self) -> None:
        fake = b.FakeLLMClient(replies=["first reply", "second reply", "   "])
        paused = await self._paused_i2(fake)
        failed = await b.resume_interactive_script(paused, "bc")
        self.assertEqual(failed.status, "failed")
        self.assertEqual(failed.calls, 3)
        self.assertEqual(len(fake.calls), 3)
        self.assertTrue(paused.recorder.closed)
        with self.assertRaises(b.TurnOrderError):
            await b.resume_interactive_script(paused, "none")
        self.assertEqual(len(fake.calls), 3)

    async def test_concurrent_branch_selection_has_no_extra_call(self) -> None:
        class BlockingThirdClient:
            synthetic = True

            def __init__(self) -> None:
                self.calls = 0
                self.third_started = asyncio.Event()
                self.release = asyncio.Event()

            async def generate(self, messages, **kwargs):
                self.calls += 1
                if self.calls == 3:
                    self.third_started.set()
                    await self.release.wait()
                return f"reply-{self.calls}"

        client = BlockingThirdClient()
        paused = await self._paused_i2(client)
        task = asyncio.create_task(b.resume_interactive_script(paused, "bc"))
        await asyncio.wait_for(client.third_started.wait(), timeout=1.0)

        with self.assertRaises(b.TurnOrderError):
            await b.resume_interactive_script(paused, "none")

        self.assertEqual(client.calls, 3)
        self.assertFalse(paused.recorder.closed)
        client.release.set()
        result = await asyncio.wait_for(task, timeout=1.0)
        self.assertEqual(result.status, "completed")
        self.assertEqual(result.calls, 3)
        self.assertEqual(client.calls, 3)
        with self.assertRaises(b.TurnOrderError):
            await b.resume_interactive_script(paused, "bc")
        self.assertEqual(client.calls, 3)

    async def test_all_branch_and_none_paths_are_one_t2_final(self) -> None:
        branch_sets = {
            "I2": ("none", "bc", "direct_ac", "no_extra"),
            "I3": ("none", "rule_request", "direct_ac", "no_extra"),
        }
        for script_id, choices in branch_sets.items():
            for choice in choices:
                with self.subTest(script=script_id, branch=choice):
                    fake = b.FakeLLMClient()
                    paused, _ = await b.start_interactive_script(
                        script(script_id),
                        system_prompt=SYSTEM_PROMPT,
                        system_sha256=SYSTEM_SHA256,
                        prompt_version_value=PROMPT_VERSION,
                        client=fake,
                        output_root=self.out,
                    )
                    assert paused is not None
                    result = await b.resume_interactive_script(paused, choice)
                    expected_calls = 2 if choice == "none" else 3
                    self.assertEqual(result.status, "completed")
                    self.assertEqual(result.calls, expected_calls)
                    self.assertEqual(len(fake.calls), expected_calls)
                    self.assertEqual(len(result.records), expected_calls)
                    self.assertFalse(any(record["turn_id"] == "Q2" for record in result.records))
                    if choice == "none":
                        self.assertFalse(any(record["turn_id"] == "T2" for record in result.records))
                    else:
                        branch_record = result.records[-1]
                        self.assertEqual(branch_record["turn_id"], "T2")
                        self.assertTrue(branch_record["final_assistant"])
                        self.assertEqual(branch_record["final_for_user_turn_id"], "T2")
                        self.assertFalse(branch_record["final_is_separate_user_turn"])
                        self.assertEqual(branch_record["branch"]["id"], choice)

    async def test_final_response_is_t2_generation_not_a_new_user_turn(self) -> None:
        fake = b.FakeLLMClient()
        paused, _ = await b.start_interactive_script(
            script("I2"),
            system_prompt=SYSTEM_PROMPT,
            system_sha256=SYSTEM_SHA256,
            prompt_version_value=PROMPT_VERSION,
            client=fake,
            output_root=self.out,
        )
        assert paused is not None
        result = await b.resume_interactive_script(paused, "direct_ac")
        self.assertEqual(result.calls, 3)
        self.assertEqual([record["turn_id"] for record in result.records], ["T1", "Q1", "T2"])
        self.assertEqual(
            [record["schema_version"] for record in result.records],
            [b.TURN_SCHEMA_VERSION] * 3,
        )
        self.assertFalse(any(record["final_is_separate_user_turn"] for record in result.records))


    async def test_metadata_failure_is_not_success(self) -> None:
        fake = b.FakeLLMClient()
        paused = await self._paused_i2(fake)
        with patch.object(
            paused.recorder, "update_meta", side_effect=OSError("synthetic metadata failure")
        ):
            result = await b.resume_interactive_script(paused, "bc")
        self.assertEqual(result.status, "failed")
        self.assertIsNotNone(result.error)
        assert result.error is not None
        self.assertEqual(result.error["type"], "MetadataWriteError")
        self.assertIn("evidence is incomplete", result.error["message"])
        self.assertIn("metadata", result.error["cleanup_error"])
        self.assertEqual(result.calls, 3)
        self.assertEqual(len(result.records), 3)
        self.assertEqual(len(fake.calls), 3)
        self.assertTrue(paused.recorder.closed)
        self.assertNotEqual(paused.state, "completed")
        meta = read_json(paused.recorder.run_dir / "run_meta.json")
        self.assertEqual(meta["status"], "paused")
        self.assertEqual(meta["calls_actual"], 2)

    async def test_none_metadata_failure_is_not_success(self) -> None:
        fake = b.FakeLLMClient()
        paused = await self._paused_i2(fake)
        with patch.object(
            paused.recorder, "update_meta", side_effect=OSError("synthetic metadata failure")
        ):
            result = await b.resume_interactive_script(paused, "none")
        self.assertEqual(result.status, "failed")
        self.assertIsNotNone(result.error)
        assert result.error is not None
        self.assertEqual(result.error["type"], "MetadataWriteError")
        self.assertEqual(result.calls, 2)
        self.assertEqual(len(result.records), 2)
        self.assertEqual(len(fake.calls), 2)
        self.assertNotEqual(paused.state, "completed")
        meta = read_json(paused.recorder.run_dir / "run_meta.json")
        self.assertEqual(meta["status"], "paused")
        self.assertEqual(meta["calls_actual"], 2)

    async def test_metadata_failure_stops_batch_before_i3(self) -> None:
        original = b.RunRecorder.update_meta

        def fail_for_i2(recorder, meta):
            if "-I2-" in recorder.run_dir.name:
                raise OSError("synthetic I2 metadata failure")
            return original(recorder, meta)

        with patch.object(b.RunRecorder, "update_meta", new=fail_for_i2):
            results = await b.run_all(
                asset_root=ASSET_ROOT,
                output_root=self.out,
                run_prefix="metadata-stop",
                branch_choices={"I2": "bc"},
            )
        self.assertTrue(results)
        self.assertEqual(results[-1].script_id, "I2")
        self.assertEqual(results[-1].status, "failed")
        self.assertFalse(any(result.script_id == "I3" for result in results))
        self.assertEqual(len(results), 8)
        self.assertEqual(results[-1].calls, 3)
        self.assertEqual(len(results[-1].records), 3)
        meta = read_json(Path(results[-1].run_dir) / "run_meta.json")
        self.assertEqual(meta["status"], "paused")
        self.assertEqual(meta["calls_actual"], 2)

    async def test_cancel_during_final_generation_is_not_completed(self) -> None:
        class BlockingThirdClient:
            synthetic = True

            def __init__(self) -> None:
                self.calls = 0
                self.third_started = asyncio.Event()
                self.release = asyncio.Event()

            async def generate(self, messages, **kwargs):
                self.calls += 1
                if self.calls == 3:
                    self.third_started.set()
                    await self.release.wait()
                return f"reply-{self.calls}"

        client = BlockingThirdClient()
        paused = await self._paused_i2(client)
        task = asyncio.create_task(b.resume_interactive_script(paused, "bc"))
        await asyncio.wait_for(client.third_started.wait(), timeout=1.0)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(client.calls, 3)
        self.assertEqual(paused.state, "cancelled")
        self.assertEqual(len(paused.records), 2)
        self.assertEqual(len((paused.recorder.run_dir / "turns.jsonl").read_text().splitlines()), 2)
        meta = read_json(paused.recorder.run_dir / "run_meta.json")
        self.assertEqual(meta["status"], "cancelled")
        self.assertEqual(meta["calls_actual"], 3)
        self.assertEqual(meta["error"]["type"], "CancelledError")
        with self.assertRaises(b.TurnOrderError):
            await b.resume_interactive_script(paused, "none")
        self.assertEqual(client.calls, 3)

    async def test_cancel_and_cleanup_failures_preserve_cancellation(self) -> None:
        class BlockingThirdClient:
            synthetic = True

            def __init__(self) -> None:
                self.calls = 0
                self.third_started = asyncio.Event()
                self.release = asyncio.Event()

            async def generate(self, messages, **kwargs):
                self.calls += 1
                if self.calls == 3:
                    self.third_started.set()
                    await self.release.wait()
                return f"reply-{self.calls}"

        client = BlockingThirdClient()
        paused = await self._paused_i2(client)
        task = asyncio.create_task(b.resume_interactive_script(paused, "bc"))
        await asyncio.wait_for(client.third_started.wait(), timeout=1.0)
        original_close = paused.recorder.close

        def close_then_fail() -> None:
            original_close()
            raise OSError("close failed")

        with patch.object(paused.recorder, "close", new=close_then_fail), \
             patch.object(paused.recorder, "update_meta", side_effect=OSError("meta failed")):
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        self.assertEqual(paused.state, "cancelled")
        self.assertIsNotNone(paused.cleanup_error)
        assert paused.cleanup_error is not None
        self.assertIn("cleanup", paused.cleanup_error)
        self.assertIn("metadata", paused.cleanup_error)
        self.assertEqual(len(paused.records), 2)
        self.assertEqual(client.calls, 3)

    async def test_generation_failure_not_masked_by_metadata_failure(self) -> None:
        fake = b.FakeLLMClient(replies=["first reply", "second reply", "   "])
        paused = await self._paused_i2(fake)
        with patch.object(
            paused.recorder, "update_meta", side_effect=OSError("synthetic metadata failure")
        ):
            result = await b.resume_interactive_script(paused, "bc")
        self.assertEqual(result.status, "failed")
        self.assertIsNotNone(result.error)
        assert result.error is not None
        self.assertEqual(result.error["type"], "RunnerFailure")
        self.assertIn("cleanup_error", result.error)
        self.assertEqual(result.calls, 3)
        self.assertEqual(len(result.records), 3)


class OptionBLedgerStateTests(unittest.TestCase):
    def test_begin_in_flight_and_commit_require_begin(self) -> None:
        turns = [
            b.TurnSpec("T1", "teach", "one"),
            b.TurnSpec("Q1", "query", "two"),
        ]
        ledger = b.TurnLedger("r", "S1", turns, max_calls=2)
        with self.assertRaises(b.TurnOrderError):
            ledger.commit()
        ledger.begin("T1", "one")
        with self.assertRaises(b.TurnOrderError):
            ledger.begin("Q1", "two")
        with self.assertRaises(b.TurnOrderError):
            ledger.ensure_complete()
        ledger.commit()
        with self.assertRaises(b.TurnOrderError):
            ledger.commit()
        ledger.begin("Q1", "two")
        with self.assertRaises(b.TurnOrderError):
            ledger.begin("Q1", "two")
        ledger.commit()
        ledger.ensure_complete()

    def test_failed_ledger_cannot_restart(self) -> None:
        turns = [b.TurnSpec("T1", "teach", "one")]
        ledger = b.TurnLedger("r", "S1", turns, max_calls=1)
        ledger.begin("T1", "one")
        ledger.fail("synthetic")
        with self.assertRaises(b.TurnOrderError):
            ledger.begin("T1", "one")
        with self.assertRaises(b.TurnOrderError):
            ledger.commit()
        with self.assertRaises(b.TurnOrderError):
            ledger.ensure_complete()


class OptionBScoringAssetTests(unittest.TestCase):
    def test_scoring_and_docs_do_not_retain_a_inference_language(self) -> None:
        expected = (ASSET_ROOT / "expected" / "per_turn_expectations.md").read_text(
            encoding="utf-8"
        )
        readme = (ASSET_ROOT / "README.md").read_text(encoding="utf-8")
        scripts = read_json(ASSET_ROOT / "scripts" / "scripts.json")
        self.assertNotIn("licensed inference", expected.lower())
        self.assertNotIn("licensed inference", readme.lower())
        for item in scripts["scripts"]:
            metadata = (item.get("title", "") + " " + item.get("focus", "")).lower()
            self.assertNotIn("licensed inference", metadata)
            self.assertNotIn("inferred", metadata)
        self.assertNotIn("No generated reply", expected)
        self.assertNotIn("non-generated", expected)
        self.assertIn("candidate confirmation", expected.lower())
        self.assertIn("unsupported premise", expected.lower())
        self.assertIn("generic request", expected.lower())
        self.assertIn("mixed request", expected.lower())
        self.assertNotIn("dominant request", expected.lower())
        self.assertNotIn("fits `direct_ac`", expected)

        import csv

        with (ASSET_ROOT / "expected" / "coding_sheet.csv").open(
            newline="", encoding="utf-8"
        ) as handle:
            rows = list(csv.DictReader(handle))
        self.assertTrue(rows)
        self.assertTrue(all(row["generation_expected"] == "yes" for row in rows))
        self.assertTrue(all(row["candidate_confirmation_separate"] == "yes" for row in rows))
        for column in (
            "candidate_confirmation",
            "unsupported_premise",
            "generic_request",
            "mixed_request",
        ):
            self.assertIn(column, rows[0])
        self.assertEqual(
            next(row for row in rows if row["script_id"] == "S1" and row["turn_id"] == "Q1")[
                "question_allowed"
            ],
            "optional",
        )
        for row in rows:
            self.assertNotIn("No generated reply", row["notes"])


class OptionBImportTests(unittest.TestCase):
    def test_default_import_does_not_load_backend(self) -> None:
        code = (
            "import sys; sys.path.insert(0, '.'); "
            "import experiments.option_b.runner; "
            "bad=[name for name in sys.modules if name=='backend' or name.startswith('backend.')]; "
            "print(','.join(bad)); "
            "raise SystemExit(1 if bad else 0)"
        )
        completed = subprocess.run(
            [sys.executable, "-c", code],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.assertEqual(completed.stdout.strip(), "")


if __name__ == "__main__":
    unittest.main()
