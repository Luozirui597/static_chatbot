"""Isolated tests for the Option A experiment.

These tests use only the local fake client and temporary directories. They do
not import a production client, open a network connection, read credentials,
or write the application database. The production history serializer is
compared by loading the two relevant function definitions from source with
``ast`` instead of importing ``backend.chat_service`` (which would load
``backend.config`` and dotenv).
"""

from __future__ import annotations

import ast
import csv
import json
import shutil
import sys
import tempfile
import unittest
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace

ASSET_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ASSET_ROOT.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.option_a import freeze_manifest  # noqa: E402
from experiments.option_a import runner  # noqa: E402
from experiments.option_a.history import (  # noqa: E402
    HISTORY_CONTEXT_VERSION,
    HISTORY_NOTE,
    HISTORY_TYPE,
    MAX_HISTORY_MESSAGES,
    StoredMessage,
    build_request_messages,
    project_receive,
    recent_history,
    serialize_receive_history,
)


def _literal_assignments(path: Path, names: set[str]) -> dict[str, object]:
    """Read simple literal assignments without importing the source module."""

    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: dict[str, object] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id in names:
                found[target.id] = ast.literal_eval(node.value)
    missing = names - set(found)
    if missing:
        raise AssertionError(f"missing production constants: {sorted(missing)}")
    return found


def _load_production_history_contract():
    """Load the two production functions from source, with no backend import.

    This preserves an actual comparison against the production implementation
    while avoiding backend.config/dotenv import side effects in the default test
    process. The functions are used only as a read-only contract oracle.
    """

    modes_path = REPO_ROOT / "backend" / "interaction_modes.py"
    chat_path = REPO_ROOT / "backend" / "chat_service.py"
    constants = _literal_assignments(
        modes_path,
        {
            "RECEIVE_TEACHING_MODE",
            "CORRECTIVE_MODE",
            "RECEIVE_TEACHING_HISTORY_CONTEXT_VERSION",
        },
    )
    constants.update(
        _literal_assignments(
            chat_path,
            {
                "MAX_HISTORY_MESSAGES",
                "_RECEIVE_TEACHING_HISTORY_TYPE",
                "_RECEIVE_TEACHING_HISTORY_NOTE",
            },
        )
    )
    tree = ast.parse(chat_path.read_text(encoding="utf-8"), filename=str(chat_path))
    wanted = {"_project_history_for_mode", "_serialize_receive_teaching_history"}
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in wanted
    ]
    if {node.name for node in functions} != wanted:
        raise AssertionError("production history functions were not found")
    namespace: dict[str, object] = {
        "json": json,
        "Sequence": Sequence,
        "Message": object,
        **constants,
    }
    module = ast.Module(body=functions, type_ignores=[])
    exec(compile(module, str(chat_path), "exec"), namespace)
    return (
        namespace["_project_history_for_mode"],
        namespace["_serialize_receive_teaching_history"],
    )


class AssetTests(unittest.TestCase):
    def test_prompt_files_and_exact_assembly(self) -> None:
        base = (ASSET_ROOT / "prompts" / "base.txt").read_text(encoding="utf-8")
        strategy = (ASSET_ROOT / "prompts" / "a_strategy.txt").read_text(
            encoding="utf-8"
        )
        system = (ASSET_ROOT / "prompts" / "system.txt").read_text(
            encoding="utf-8"
        )
        self.assertEqual(
            system,
            base.rstrip("\n") + "\n\n" + strategy.rstrip("\n") + "\n",
        )
        self.assertNotIn("不新增推导能力", base)
        self.assertIn("目标关系", base)
        self.assertIn("阻断状态", base)
        self.assertIn("[inferred]", strategy)
        self.assertIn("说明缺口本身不是向老师请求补答", strategy)

    def test_script_matrix_and_turn_structure(self) -> None:
        scripts = runner.load_scripts(ASSET_ROOT)
        self.assertEqual(
            [script.id for script in scripts],
            ["S1", "S2", "S3", "S4", "S5", "S6"],
        )
        self.assertEqual(
            {script.id: script.calls_expected for script in scripts},
            {"S1": 2, "S2": 2, "S3": 2, "S4": 2, "S5": 4, "S6": 4},
        )
        self.assertEqual(sum(s.calls_expected for s in scripts), 16)
        s1, s2, s3, s4, s5, s6 = scripts
        self.assertEqual([turn["id"] for turn in s5.turns], ["T1", "Q1", "T2", "Q2"])
        self.assertEqual([turn["id"] for turn in s6.turns], ["T1", "Q1", "T2", "Q2"])
        s3_text = " ".join(turn["content"] for turn in s3.turns)
        self.assertNotIn("There is no rule", s3_text)
        self.assertIn("If you need a rule", s3_text)
        self.assertIn("does not link to C", s5.turns[2]["content"])
        self.assertIn("withdraw", s6.turns[2]["content"].lower())

        # S2 must have exactly one taught relation and no active condition.
        s2_t1 = s2.turns[0]["content"]
        self.assertIn("A links to B", s2_t1)
        self.assertNotIn("B links to C", s2_t1)
        self.assertNotIn("active", s2_t1.lower())
        self.assertIn("if X links to Y and Y links to Z, then X links to Z", s2_t1)
        self.assertIn("say exactly what is missing and stop", s2.turns[1]["content"])

        # S1/S4 intentionally keep active conditions; S3 was not reverted to an
        # explicit no-rule statement.
        self.assertIn("Both A-B and B-C are marked active", s1.turns[0]["content"])
        self.assertIn("B is flagged", s4.turns[0]["content"])

    def test_s2_and_s5_expectation_corrections(self) -> None:
        expected = (ASSET_ROOT / "expected" / "per_turn_expectations.md").read_text(
            encoding="utf-8"
        )
        self.assertIn("B-C is missing", expected)
        self.assertNotIn("(and its active status)", expected)
        self.assertIn("old_support_removed", expected)
        self.assertIn("negative_fact_preserved", expected)
        self.assertIn('"B-C is no longer supported" alone', expected)
        with (ASSET_ROOT / "expected" / "coding_sheet.csv").open(
            encoding="utf-8", newline=""
        ) as handle:
            header = next(csv.reader(handle))
        self.assertIn("old_support_removed", header)
        self.assertIn("negative_fact_preserved", header)

    def test_history_constants_match_production_contract(self) -> None:
        self.assertEqual(MAX_HISTORY_MESSAGES, 20)
        self.assertEqual(HISTORY_CONTEXT_VERSION, "history-context-b1")
        self.assertEqual(HISTORY_TYPE, "receive_teaching_history_background")
        self.assertIn("老师先前 user 消息的原文", HISTORY_NOTE)


class HistoryTests(unittest.TestCase):
    def test_serializer_matches_production_for_same_rows(self) -> None:
        rows = [
            StoredMessage("user", "q1"),
            StoredMessage("assistant", "a1"),
            StoredMessage("user", "q2"),
        ]
        raw = [
            SimpleNamespace(role=row.role, content=row.content) for row in rows
        ]
        project, serialize = _load_production_history_contract()
        projected = project(raw, interaction_mode="receive_teaching")
        self.assertEqual(
            serialize_receive_history(project_receive(rows)),
            serialize(projected),
        )

    def test_production_contract_loader_has_no_backend_config_side_effect(self) -> None:
        self.assertNotIn("backend.config", sys.modules)
        self.assertNotIn("dotenv", sys.modules)
        _load_production_history_contract()
        self.assertNotIn("backend.config", sys.modules)
        self.assertNotIn("dotenv", sys.modules)

    def test_limit_is_applied_before_user_projection(self) -> None:
        rows: list[StoredMessage] = []
        for number in range(1, 26):
            rows.append(StoredMessage("user", f"q{number}"))
            rows.append(StoredMessage("assistant", f"a{number}"))
        window = recent_history(rows)
        self.assertEqual(len(window), 20)
        projected = project_receive(window)
        self.assertEqual(len(projected), 10)
        self.assertEqual([row.content for row in projected], [f"q{i}" for i in range(16, 26)])

    def test_current_user_not_in_window_and_not_duplicated(self) -> None:
        prior = [StoredMessage("user", f"q{i}") for i in range(1, 21)]
        messages = build_request_messages("SYSTEM", prior, "CURRENT")
        self.assertEqual(messages[0], {"role": "system", "content": "SYSTEM"})
        self.assertEqual(messages[1]["role"], "user")
        background = json.loads(messages[1]["content"])
        self.assertEqual(len(background["messages"]), 20)
        self.assertEqual(background["messages"][-1]["content"], "q20")
        self.assertEqual(messages[2], {"role": "user", "content": "CURRENT"})
        self.assertEqual(sum(1 for message in messages if message["content"] == "CURRENT"), 1)

    def test_empty_prior_history_has_no_background_block(self) -> None:
        messages = build_request_messages("SYSTEM", [], "CURRENT")
        self.assertEqual(
            messages,
            [
                {"role": "system", "content": "SYSTEM"},
                {"role": "user", "content": "CURRENT"},
            ],
        )


class CountingFactory:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self) -> runner.LLMClient:
        self.calls += 1
        return runner.FakeLLMClient()


class RunnerTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.system_prompt = runner.load_system_prompt(ASSET_ROOT)
        self.scripts = {script.id: script for script in runner.load_scripts(ASSET_ROOT)}

    async def test_s1_two_calls_history_projection_and_record_integrity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            client = runner.FakeLLMClient(["acknowledged", "answer"])
            result = await runner.run_script(
                self.scripts["S1"],
                system_prompt=self.system_prompt,
                client=client,
                output_root=Path(tmp),
                run_id="test-s1",
            )
            self.assertEqual(result.status, "completed")
            self.assertEqual(result.calls, 2)
            self.assertEqual(len(result.records), 2)
            self.assertEqual(len(client.calls), 2)
            first, second = result.records
            self.assertEqual([m["role"] for m in first["request"]["messages"]], ["system", "user"])
            self.assertEqual(second["request"]["messages"][0]["role"], "system")
            self.assertEqual(second["request"]["messages"][-1]["content"], self.scripts["S1"].turns[1]["content"])
            background = json.loads(second["request"]["messages"][1]["content"])
            self.assertEqual(background["type"], HISTORY_TYPE)
            self.assertEqual(background["version"], "history-context-b1")
            self.assertEqual(background["messages"], [{"index": 1, "content": self.scripts["S1"].turns[0]["content"]}])
            self.assertNotIn("acknowledged", second["request"]["messages"][1]["content"])
            self.assertEqual(
                second["request"]["sha256"],
                runner.sha256_json({"messages": second["request"]["messages"]}),
            )
            self.assertEqual(second["response_sha256"], runner.sha256_text("answer"))
            self.assertIsNone(first["model_claims"]["reported_conclusion"])
            self.assertIsNone(first["human_review"]["authorized"])
            self.assertIsNone(first["human_review"]["inference_conclusion"])
            self.assertEqual(first["current_user"], self.scripts["S1"].turns[0]["content"])
            self.assertFalse(first["retry"])
            self.assertEqual(first["client"], {"kind": "FakeLLMClient", "synthetic": True})
            self.assertIsNone(first["model"]["profile_id"])
            self.assertIsNone(first["generation"]["requested"]["temperature"])
            self.assertEqual(first["source"]["source_message_id"], None)
            self.assertIsNone(first["human_review"]["verified_source_turn_ids"])
            self.assertIsNone(first["human_review"]["verified_source_record_ids"])
            self.assertEqual(first["record_id"], "test-s1:S1:T1")
            turn_lines = (Path(result.run_dir) / "turns.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(turn_lines), 2)
            self.assertTrue(all(json.loads(line) for line in turn_lines))
            meta = json.loads((Path(result.run_dir) / "run_meta.json").read_text(encoding="utf-8"))
            self.assertEqual(meta["calls_actual"], 2)
            self.assertEqual(meta["status"], "completed")
            self.assertEqual(meta["system_sha256"], runner.sha256_text(self.system_prompt))
            self.assertEqual(meta["script_sha256"], runner._script_digest(self.scripts["S1"]))
            self.assertEqual(meta["client"], {"kind": "FakeLLMClient", "synthetic": True})
            self.assertIsNone(meta["model"]["profile_id"])
            self.assertIsNone(meta["source_message_id"])

    async def test_s5_and_s6_use_four_turns_and_link_events(self) -> None:
        for script_id in ("S5", "S6"):
            with tempfile.TemporaryDirectory() as tmp:
                client = runner.StaticReplyClient("synthetic answer")
                result = await runner.run_script(
                    self.scripts[script_id],
                    system_prompt=self.system_prompt,
                    client=client,
                    output_root=Path(tmp),
                    run_id=f"test-{script_id.lower()}",
                )
                self.assertEqual(result.status, "completed")
                self.assertEqual(result.calls, 4)
                self.assertEqual(len(result.records), 4)
                self.assertEqual(
                    [record["turn_id"] for record in result.records],
                    ["T1", "Q1", "T2", "Q2"],
                )
                self.assertEqual(
                    [record["phase"] for record in result.records],
                    ["teach", "query", "correction" if script_id == "S5" else "revocation", "recheck"],
                )
                t2 = result.records[2]
                q2 = result.records[3]
                self.assertEqual(t2["source"]["event"]["type"], "correction" if script_id == "S5" else "revocation")
                self.assertIn(result.records[1]["record_id"], t2["source"]["related_record_ids"])
                self.assertEqual(q2["source"]["event"]["type"], "recheck_after_correction" if script_id == "S5" else "recheck_after_revocation")
                self.assertIn(t2["record_id"], q2["source"]["related_record_ids"])
                self.assertTrue(all(record["client"]["synthetic"] for record in result.records))

    async def test_failure_stops_immediately_and_keeps_completed_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            client = runner.RaisingClient(RuntimeError("boom"), fail_on_call=2)
            result = await runner.run_script(
                self.scripts["S1"],
                system_prompt=self.system_prompt,
                client=client,
                output_root=Path(tmp),
                run_id="test-failure",
            )
            self.assertEqual(result.status, "failed")
            self.assertEqual(result.calls, 2)
            self.assertEqual(len(result.records), 2)
            self.assertEqual(result.records[0]["status"], "ok")
            self.assertEqual(result.records[1]["status"], "error")
            self.assertEqual(result.records[1]["error"]["type"], "RuntimeError")
            self.assertIsNone(result.records[1]["response"])
            self.assertEqual(len(client.calls), 2)
            self.assertEqual(result.error, {"type": "RuntimeError", "message": "boom"})

    async def test_blank_response_is_an_error_but_keeps_raw_string(self) -> None:
        raw_blank = "   \n"
        with tempfile.TemporaryDirectory() as tmp:
            client = runner.StaticReplyClient(raw_blank)
            result = await runner.run_script(
                self.scripts["S1"],
                system_prompt=self.system_prompt,
                client=client,
                output_root=Path(tmp),
                run_id="test-empty",
            )
            self.assertEqual(result.status, "failed")
            self.assertEqual(result.calls, 1)
            self.assertEqual(len(result.records), 1)
            self.assertEqual(result.records[0]["error"]["type"], "EmptyResponseError")
            self.assertEqual(result.records[0]["response"], raw_blank)
            self.assertEqual(result.records[0]["response_sha256"], runner.sha256_text(raw_blank))
            self.assertEqual(len(client.calls), 1)

    async def test_wrong_system_hash_fails_before_client_or_run_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            client = runner.FakeLLMClient()
            with self.assertRaises(freeze_manifest.FrozenAssetError):
                await runner.run_script(
                    self.scripts["S1"],
                    system_prompt=self.system_prompt,
                    client=client,
                    output_root=Path(tmp),
                    run_id="wrong-hash",
                    system_sha256="0" * 64,
                )
            self.assertEqual(len(client.calls), 0)
            self.assertEqual(list(Path(tmp).iterdir()), [])

    async def test_existing_run_directory_is_never_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            first = await runner.run_script(
                self.scripts["S1"],
                system_prompt=self.system_prompt,
                client=runner.FakeLLMClient(),
                output_root=Path(tmp),
                run_id="fixed-run",
            )
            before = (Path(first.run_dir) / "turns.jsonl").read_bytes()
            with self.assertRaises(FileExistsError):
                await runner.run_script(
                    self.scripts["S1"],
                    system_prompt=self.system_prompt,
                    client=runner.FakeLLMClient(),
                    output_root=Path(tmp),
                    run_id="fixed-run",
                )
            after = (Path(first.run_dir) / "turns.jsonl").read_bytes()
            self.assertEqual(before, after)

    async def test_question_in_a_reply_does_not_create_a_branch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            client = runner.StaticReplyClient("Does B link to C?")
            result = await runner.run_script(
                self.scripts["S2"],
                system_prompt=self.system_prompt,
                client=client,
                output_root=Path(tmp),
                run_id="test-no-branch",
            )
            self.assertEqual(result.calls, 2)
            self.assertEqual(len(result.records), 2)
            self.assertEqual(client.calls[1][-1]["content"], self.scripts["S2"].turns[1]["content"])
            self.assertEqual(len(client.calls), 2)

    async def test_default_run_all_is_fake_and_uses_sixteen_calls(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest_before = (ASSET_ROOT / "manifest.json").read_bytes()
            results = await runner.run_all(
                asset_root=ASSET_ROOT,
                output_root=Path(tmp),
            )
            self.assertEqual(len(results), 6)
            self.assertTrue(all(result.status == "completed" for result in results))
            self.assertEqual(sum(result.calls for result in results), 16)
            self.assertEqual(
                {result.script_id: result.calls for result in results},
                {"S1": 2, "S2": 2, "S3": 2, "S4": 2, "S5": 4, "S6": 4},
            )
            self.assertEqual(len(list(Path(tmp).iterdir())), 6)
            for result in results:
                self.assertTrue((Path(result.run_dir) / "run_meta.json").is_file())
                self.assertTrue((Path(result.run_dir) / "turns.jsonl").is_file())
            self.assertEqual(manifest_before, (ASSET_ROOT / "manifest.json").read_bytes())

    async def test_run_all_blocks_on_manifest_mismatch_before_client(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            asset_copy = Path(tmp) / "assets"
            shutil.copytree(ASSET_ROOT, asset_copy)
            manifest_before = (asset_copy / "manifest.json").read_bytes()
            (asset_copy / "scripts" / "S2.txt").write_text(
                (asset_copy / "scripts" / "S2.txt").read_text(encoding="utf-8") + "\n",
                encoding="utf-8",
                newline="\n",
            )
            factory = CountingFactory()
            output_root = Path(tmp) / "output"
            with self.assertRaises(freeze_manifest.FrozenAssetError):
                await runner.run_all(
                    asset_root=asset_copy,
                    output_root=output_root,
                    client_factory=factory,
                )
            self.assertEqual(factory.calls, 0)
            self.assertFalse(output_root.exists())
            self.assertEqual(manifest_before, (asset_copy / "manifest.json").read_bytes())

    async def test_run_all_blocks_on_system_mismatch_before_client(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            asset_copy = Path(tmp) / "assets"
            shutil.copytree(ASSET_ROOT, asset_copy)
            (asset_copy / "prompts" / "system.txt").write_text(
                (asset_copy / "prompts" / "system.txt").read_text(encoding="utf-8") + "\n",
                encoding="utf-8",
                newline="\n",
            )
            factory = CountingFactory()
            with self.assertRaises(freeze_manifest.FrozenAssetError):
                await runner.run_all(
                    asset_root=asset_copy,
                    output_root=Path(tmp) / "output",
                    client_factory=factory,
                )
            self.assertEqual(factory.calls, 0)

    def test_default_factory_is_the_local_fake(self) -> None:
        import inspect

        default = inspect.signature(runner.run_all).parameters["client_factory"].default
        self.assertIs(default, runner.FakeLLMClient)
        source = (ASSET_ROOT / "runner.py").read_text(encoding="utf-8")
        for forbidden in ("openai", "requests", "httpx", "urllib", "http.client"):
            self.assertNotIn(forbidden, source)

    def test_manifest_negative_cases(self) -> None:
        manifest_path = ASSET_ROOT / "manifest.json"
        self.assertTrue(manifest_path.is_file())
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["condition"], "A")
        self.assertEqual(manifest["calls"]["total_expected"], 16)
        self.assertIn("freeze_manifest.py", manifest["files"])
        self.assertEqual(set(manifest["files"]), set(freeze_manifest.INCLUDED_FILES))

        with tempfile.TemporaryDirectory() as tmp:
            asset_copy = Path(tmp) / "assets"
            shutil.copytree(ASSET_ROOT, asset_copy)
            missing_runner = json.loads((asset_copy / "manifest.json").read_text(encoding="utf-8"))
            del missing_runner["files"]["runner.py"]
            (asset_copy / "runner.py").write_text(
                (asset_copy / "runner.py").read_text(encoding="utf-8") + "\n",
                encoding="utf-8",
                newline="\n",
            )
            self.assertTrue(freeze_manifest.verify_manifest(asset_copy, missing_runner))

        changed_schema = json.loads(json.dumps(manifest))
        changed_schema["schema_version"] = "tampered"
        self.assertTrue(freeze_manifest.verify_manifest(ASSET_ROOT, changed_schema))

        changed_condition = json.loads(json.dumps(manifest))
        changed_condition["condition"] = "B"
        self.assertTrue(freeze_manifest.verify_manifest(ASSET_ROOT, changed_condition))

        with tempfile.TemporaryDirectory() as tmp:
            asset_copy = Path(tmp) / "assets"
            shutil.copytree(ASSET_ROOT, asset_copy)
            (asset_copy / "runner.py").write_text(
                (asset_copy / "runner.py").read_text(encoding="utf-8") + "\n",
                encoding="utf-8",
                newline="\n",
            )
            self.assertTrue(
                freeze_manifest.verify_manifest(
                    asset_copy,
                    json.loads((asset_copy / "manifest.json").read_text(encoding="utf-8")),
                )
            )

        with tempfile.TemporaryDirectory() as tmp:
            asset_copy = Path(tmp) / "assets"
            shutil.copytree(ASSET_ROOT, asset_copy)
            (asset_copy / "prompts" / "system.txt").write_text(
                (asset_copy / "prompts" / "system.txt").read_text(encoding="utf-8") + "\n",
                encoding="utf-8",
                newline="\n",
            )
            self.assertTrue(
                freeze_manifest.verify_manifest(
                    asset_copy,
                    json.loads((asset_copy / "manifest.json").read_text(encoding="utf-8")),
                )
            )

    def test_manifest_if_present_matches_assets(self) -> None:
        manifest_path = ASSET_ROOT / "manifest.json"
        if not manifest_path.is_file():
            self.skipTest("manifest has not been generated yet")
        manifest = freeze_manifest.load_and_verify_manifest(ASSET_ROOT)
        self.assertEqual(manifest["calls"]["total_expected"], 16)
        self.assertEqual(manifest["condition"], "A")


if __name__ == "__main__":
    unittest.main()
