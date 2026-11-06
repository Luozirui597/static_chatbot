from __future__ import annotations

import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import unittest
from concurrent.futures import CancelledError, ThreadPoolExecutor
from pathlib import Path
from typing import Any
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from experiments.option_a.freeze_manifest import FrozenAssetError
from experiments.option_a_web.asgi import TestApp
from experiments.option_a_web.gate import (
    ArmedTurn,
    GateError,
    RunState,
    TurnGate,
    TurnSpec,
    content_sha256,
)
from experiments.option_a_web.launcher import (
    FROZEN_ASSET_ROOT,
    REPO_ROOT,
    IsolationError,
    RunPaths,
    _normalized_launch_argv,
    _process_start_time,
    _verify_snapshot_factory,
    load_frozen_spec,
    prepare_instance,
    stop_instance,
)
from experiments.option_a_web.recording import (
    RecordingLLMClient,
    RecordingError,
    RequestContext,
    RunRecorder,
    reset_request_context,
    set_request_context,
)


def _tree_hash(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(path.rglob("*")):
        if item.is_file():
            digest.update(str(item.relative_to(path)).encode("utf-8"))
            digest.update(item.read_bytes())
    return digest.hexdigest()


class FrozenCounter:
    def __init__(self) -> None:
        self.count = 0

    def __call__(self) -> Any:
        self.count += 1
        return CountingClient()


class FailingClient:
    def __init__(self, detail: str) -> None:
        self.detail = detail
        self.calls = 0

    async def generate(self, messages, **kwargs):
        self.calls += 1
        from backend.exceptions import LLMError
        from backend.review_diagnostics import LLMCallDiagnostics

        raise LLMError(self.detail, 500, diagnostics=LLMCallDiagnostics())


class CountingClient:
    upstream_response_model = "fake-upstream-model"

    def __init__(self) -> None:
        self.calls = 0
        self.original_kwargs: list[dict[str, Any]] = []

    async def generate(
        self,
        messages,
        *,
        response_format=None,
        temperature=None,
        max_tokens=None,
        diagnostics=None,
    ):
        self.calls += 1
        self.original_kwargs.append(
            {
                "temperature": temperature,
                "max_tokens": max_tokens,
                "response_format": response_format,
                "has_diagnostics": diagnostics is not None,
            }
        )
        if diagnostics is not None:
            diagnostics.http_status = 200
            diagnostics.finish_reason = "stop"
            diagnostics.upstream_request_id = "fake-request-id"
            diagnostics.call_elapsed_ms = 1
            diagnostics.response_content_chars = len("fake reply")
        return "fake reply"


def _minimal_html() -> str:
    return "<html><head><title>Static Chatbot</title></head><body></body></html>"


class TestFreezeAndPaths(unittest.TestCase):
    def test_freeze_verification_happens_before_client_creation(self):
        source = FROZEN_ASSET_ROOT
        with tempfile.TemporaryDirectory(prefix="option-a-bad-freeze-") as tmp:
            asset = Path(tmp) / "asset"
            shutil.copytree(source, asset)
            system_path = asset / "prompts" / "system.txt"
            system_path.write_text(
                system_path.read_text(encoding="utf-8") + "\nchanged\n",
                encoding="utf-8",
            )
            counter = FrozenCounter()
            with self.assertRaises(FrozenAssetError):
                prepare_instance(
                    run_dir=Path(tmp) / "run",
                    asset_root=asset,
                    client_factory=counter,
                )
            self.assertEqual(counter.count, 0)

    def test_run_dir_must_be_outside_repo(self):
        with self.assertRaises(IsolationError):
            RunPaths.create(REPO_ROOT / "experiments" / "option_a_web" / "forbidden-run")

    def test_launcher_import_does_not_load_backend(self):
        code = (
            "import sys; "
            "import experiments.option_a_web.launcher; "
            "loaded=[n for n in sys.modules if n=='backend' or n.startswith('backend.')]; "
            "assert not loaded, loaded; print('clean')"
        )
        completed = subprocess.run(
            [sys.executable, "-c", code],
            cwd=str(REPO_ROOT),
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("clean", completed.stdout)

    def test_frozen_spec_is_verified_without_backend(self):
        spec = load_frozen_spec()
        self.assertEqual(spec.asset_version, "option-a-isolated-v0.1")
        self.assertEqual(spec.system_sha256, "16236223af1b3f813e5e27aa881e27d4b45039b58db6be2f44d4a4f9137f26a4")
        self.assertEqual(spec.prompt_version, "option-a-isolated-v0.1+sys-16236223af1b")
        self.assertEqual(spec.total_expected, 16)
        self.assertEqual(sum(len(t) for t in spec.scripts.values()), 16)


class TestGate(unittest.TestCase):
    def _gate(self) -> tuple[TurnGate, TurnSpec]:
        content = "A links to B."
        spec = TurnSpec(
            script_id="S1",
            turn_id="T1",
            content=content,
            content_sha256=content_sha256(content),
        )
        return (
            TurnGate(
                instance_id="instance-1",
                turns={("S1", "T1"): spec},
            ),
            spec,
        )

    def test_arm_consume_once_and_duplicate_rejected(self):
        gate, spec = self._gate()
        armed = gate.arm(
            instance_id="instance-1",
            session_id=7,
            script_id="S1",
            turn_id="T1",
        )
        self.assertEqual(armed.content_sha256, spec.content_sha256)
        consumed = gate.consume(session_id=7, content=spec.content)
        self.assertEqual(consumed.turn_id, "T1")
        with self.assertRaises(GateError):
            gate.consume(session_id=7, content=spec.content)

    def test_wrong_session_or_content_does_not_consume(self):
        gate, spec = self._gate()
        gate.arm(
            instance_id="instance-1",
            session_id=7,
            script_id="S1",
            turn_id="T1",
        )
        with self.assertRaises(GateError):
            gate.consume(session_id=8, content=spec.content)
        with self.assertRaises(GateError):
            gate.consume(session_id=7, content=spec.content + "x")
        self.assertEqual(gate.consume(session_id=7, content=spec.content).turn_id, "T1")

    def test_concurrent_consume_has_one_winner(self):
        gate, spec = self._gate()
        gate.arm(
            instance_id="instance-1",
            session_id=7,
            script_id="S1",
            turn_id="T1",
        )

        def attempt() -> bool:
            try:
                gate.consume(session_id=7, content=spec.content)
                return True
            except GateError:
                return False

        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(lambda _: attempt(), range(12)))
        self.assertEqual(results.count(True), 1)

    def test_second_arm_rejected_until_consumed(self):
        gate, _ = self._gate()
        gate.arm(
            instance_id="instance-1",
            session_id=7,
            script_id="S1",
            turn_id="T1",
        )
        with self.assertRaises(GateError):
            gate.arm(
                instance_id="instance-1",
                session_id=7,
                script_id="S1",
                turn_id="T1",
            )


class TestGateBatchOrder(unittest.TestCase):
    def _ordered_gate(self):
        specs = {
            "S1": [
                TurnSpec("S1", "T1", "s1-t1", content_sha256("s1-t1")),
                TurnSpec("S1", "T2", "s1-t2", content_sha256("s1-t2")),
            ],
            "S2": [
                TurnSpec("S2", "T1", "s2-t1", content_sha256("s2-t1")),
            ],
        }
        return TurnGate(instance_id="instance-1", scripts=specs), specs

    def test_script_and_turn_order_is_enforced(self):
        gate, specs = self._ordered_gate()
        with self.assertRaises(GateError):
            gate.arm(
                instance_id="instance-1",
                session_id=10,
                script_id="S2",
                turn_id="T1",
            )
        with self.assertRaises(GateError):
            gate.arm(
                instance_id="instance-1",
                session_id=10,
                script_id="S1",
                turn_id="T2",
            )
        gate.arm(
            instance_id="instance-1",
            session_id=10,
            script_id="S1",
            turn_id="T1",
        )
        gate.consume(session_id=10, content="s1-t1")
        with self.assertRaises(GateError):
            gate.arm(
                instance_id="instance-1",
                session_id=10,
                script_id="S1",
                turn_id="T2",
            )
        gate.complete(
            armed=gate.in_flight,
            user_message_id=1,
            assistant_message_id=2,
        )
        gate.arm(
            instance_id="instance-1",
            session_id=10,
            script_id="S1",
            turn_id="T2",
        )
        gate.consume(session_id=10, content="s1-t2")
        gate.complete(
            armed=gate.in_flight,
            user_message_id=3,
            assistant_message_id=4,
        )
        gate.arm(
            instance_id="instance-1",
            session_id=20,
            script_id="S2",
            turn_id="T1",
        )
        self.assertEqual(gate.session_for_script("S2"), 20)
        self.assertEqual(len(specs), 2)

    def test_session_binding_and_budget_are_enforced(self):
        gate, _ = self._ordered_gate()
        gate.arm(
            instance_id="instance-1",
            session_id=10,
            script_id="S1",
            turn_id="T1",
        )
        gate.consume(session_id=10, content="s1-t1")
        gate.complete(
            armed=gate.in_flight,
            user_message_id=1,
            assistant_message_id=2,
        )
        with self.assertRaises(GateError):
            gate.arm(
                instance_id="instance-1",
                session_id=11,
                script_id="S1",
                turn_id="T2",
            )
        gate.arm(
            instance_id="instance-1",
            session_id=10,
            script_id="S1",
            turn_id="T2",
        )
        gate.consume(session_id=10, content="s1-t2")
        gate.complete(
            armed=gate.in_flight,
            user_message_id=3,
            assistant_message_id=4,
        )
        with self.assertRaises(GateError):
            gate.arm(
                instance_id="instance-1",
                session_id=10,
                script_id="S2",
                turn_id="T1",
            )
        gate.arm(
            instance_id="instance-1",
            session_id=30,
            script_id="S2",
            turn_id="T1",
        )
        gate.consume(session_id=30, content="s2-t1")
        gate.complete(
            armed=gate.in_flight,
            user_message_id=5,
            assistant_message_id=6,
        )
        self.assertEqual(gate.total_consumed, 3)
        with self.assertRaises(GateError):
            gate.arm(
                instance_id="instance-1",
                session_id=30,
                script_id="S2",
                turn_id="T1",
            )

    def test_concurrent_arm_has_one_winner(self):
        gate, _ = self._ordered_gate()

        def attempt(index: int) -> bool:
            try:
                gate.arm(
                    instance_id="instance-1",
                    session_id=100 + index,
                    script_id="S1",
                    turn_id="T1",
                )
                return True
            except GateError:
                return False

        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(attempt, range(12)))
        self.assertEqual(results.count(True), 1)


class TestSnapshotMismatch(unittest.TestCase):
    def test_snapshot_mismatch_stops_batch_without_retry(self):
        import asyncio

        with tempfile.TemporaryDirectory(prefix="option-a-snapshot-") as tmp:
            run_paths = RunPaths.create(Path(tmp) / "run")
            recorder = RunRecorder(
                run_dir=run_paths.run_dir, evidence_dir=run_paths.evidence_dir
            )
            inner = CountingClient()
            system = "system text"
            wrapper = RecordingLLMClient(
                inner,
                recorder=recorder,
                system_prompt=system,
                system_sha256=content_sha256(system),
                profile_id="default",
                request_model="synthetic-a-model",
                synthetic=True,
            )
            gate = TurnGate(
                instance_id="instance-1",
                turns={
                    ("S1", "T1"): TurnSpec(
                        "S1", "T1", "user", content_sha256("user")
                    )
                },
            )
            app = FastAPI()

            @app.post("/api/sessions/{session_id}/messages")
            async def message(session_id: int):
                await wrapper.generate(
                    [
                        {"role": "system", "content": system},
                        {"role": "user", "content": "user"},
                    ]
                )
                return {
                    "user_message": {"id": 1},
                    "assistant_message": {"id": 2},
                }

            def verifier(session_id, message_ids, script_id, turn_id):
                return {"errors": ["forced snapshot mismatch"]}

            wrapped = TestApp(
                app,
                test_html=_minimal_html(),
                gate=gate,
                recorder=recorder,
                instance_id="instance-1",
                snapshot_verifier=verifier,
            )
            try:
                with TestClient(wrapped) as client:
                    armed = client.post(
                        "/__option_a__/arm",
                        json={
                            "instance_id": "instance-1",
                            "session_id": 7,
                            "script_id": "S1",
                            "turn_id": "T1",
                        },
                    )
                    self.assertEqual(armed.status_code, 200, armed.text)
                    response = client.post(
                        "/api/sessions/7/messages", json={"message": "user"}
                    )
                    self.assertEqual(response.status_code, 200, response.text)
                    self.assertTrue(gate.run_state.failed)
                    self.assertEqual(inner.calls, 1)
                    self.assertEqual(len(recorder.calls), 1)
                    self.assertEqual(
                        recorder.http[-1]["failure_reason"],
                        "message snapshot verification failed",
                    )
                    rejected = client.post(
                        "/__option_a__/arm",
                        json={
                            "instance_id": "instance-1",
                            "session_id": 7,
                            "script_id": "S1",
                            "turn_id": "T1",
                        },
                    )
                    self.assertEqual(rejected.status_code, 409)
                    self.assertEqual(inner.calls, 1)
            finally:
                recorder.close()


class TestUnplannedEntry(unittest.TestCase):
    def test_no_arm_rejects_before_dummy_chat(self):
        with tempfile.TemporaryDirectory(prefix="option-a-no-arm-") as tmp:
            run_paths = RunPaths.create(Path(tmp) / "run")
            recorder = RunRecorder(
                run_dir=run_paths.run_dir, evidence_dir=run_paths.evidence_dir
            )
            gate = TurnGate(
                instance_id="instance-1",
                turns={
                    ("S1", "T1"): TurnSpec(
                        "S1",
                        "T1",
                        "x",
                        content_sha256("x"),
                    )
                },
            )
            calls = {"count": 0}
            app = FastAPI()

            @app.post("/api/sessions/{session_id}/messages")
            async def fake_message(session_id: int):
                calls["count"] += 1
                return {
                    "user_message": {"id": 1},
                    "assistant_message": {"id": 2},
                }

            from experiments.option_a_web.asgi import TestApp

            wrapped = TestApp(
                app,
                test_html=_minimal_html(),
                gate=gate,
                recorder=recorder,
                instance_id="instance-1",
            )
            try:
                with TestClient(wrapped) as client:
                    response = client.post("/api/sessions/1/messages", json={"message": "x"})
                    self.assertEqual(response.status_code, 409)
                    self.assertEqual(calls["count"], 0)
                    self.assertTrue(gate.run_state.failed)
            finally:
                recorder.close()


class TestRequestProtocol(unittest.TestCase):
    def _app(self, tmp: str, route):
        run_paths = RunPaths.create(Path(tmp) / "run")
        recorder = RunRecorder(
            run_dir=run_paths.run_dir, evidence_dir=run_paths.evidence_dir
        )
        spec = TurnSpec("S1", "T1", "x", content_sha256("x"))
        gate = TurnGate(
            instance_id="instance-1",
            scripts={"S1": [spec]},
        )
        app = FastAPI()
        app.post("/api/sessions/{session_id}/messages")(route)
        wrapped = TestApp(
            app,
            test_html=_minimal_html(),
            gate=gate,
            recorder=recorder,
            instance_id="instance-1",
        )
        return wrapped, gate, recorder

    def _arm(self, client: TestClient) -> None:
        response = client.post(
            "/__option_a__/arm",
            json={
                "instance_id": "instance-1",
                "session_id": 7,
                "script_id": "S1",
                "turn_id": "T1",
            },
        )
        self.assertEqual(response.status_code, 200, response.text)

    def test_json_top_level_must_be_object(self):
        for payload in ([], None, 1, "text"):
            with self.subTest(payload=payload), tempfile.TemporaryDirectory(
                prefix="option-a-json-"
            ) as tmp:
                calls = {"count": 0}

                async def route(session_id: int):
                    calls["count"] += 1
                    return {
                        "user_message": {"id": 1},
                        "assistant_message": {"id": 2},
                    }

                wrapped, gate, recorder = self._app(tmp, route)
                try:
                    with TestClient(wrapped) as client:
                        self._arm(client)
                        response = client.post(
                            "/api/sessions/7/messages", json=payload
                        )
                        self.assertEqual(response.status_code, 400, response.text)
                    self.assertEqual(calls["count"], 0)
                    self.assertTrue(gate.run_state.failed)
                    self.assertEqual(recorder.http[-1]["status_code"], 400)
                finally:
                    recorder.close()

    def test_invalid_utf8_body_is_rejected_without_model_call(self):
        calls = {"count": 0}

        async def route(session_id: int):
            calls["count"] += 1
            return {
                "user_message": {"id": 1},
                "assistant_message": {"id": 2},
            }

        with tempfile.TemporaryDirectory(prefix="option-a-utf8-") as tmp:
            wrapped, gate, recorder = self._app(tmp, route)
            try:
                with TestClient(wrapped) as client:
                    self._arm(client)
                    response = client.post(
                        "/api/sessions/7/messages",
                        content=b"\xff",
                        headers={"content-type": "application/json"},
                    )
                    self.assertEqual(response.status_code, 400, response.text)
                self.assertEqual(calls["count"], 0)
                self.assertTrue(gate.run_state.failed)
                self.assertEqual(recorder.http[-1]["status_code"], 400)
            finally:
                recorder.close()

    def test_unhandled_exception_records_http_attempt(self):
        async def route(session_id: int):
            raise RuntimeError("secret upstream detail")

        with tempfile.TemporaryDirectory(prefix="option-a-500-") as tmp:
            wrapped, gate, recorder = self._app(tmp, route)
            try:
                with TestClient(wrapped) as client:
                    self._arm(client)
                    response = client.post(
                        "/api/sessions/7/messages", json={"message": "x"}
                    )
                    self.assertEqual(response.status_code, 500, response.text)
                    self.assertNotIn("secret upstream detail", response.text)
                self.assertTrue(gate.run_state.failed)
                self.assertEqual(len(recorder.http), 1)
                self.assertEqual(recorder.http[-1]["status_code"], 500)
            finally:
                recorder.close()

    def test_raw_exception_returns_502_and_records_http_attempt(self):
        import asyncio

        async def inner_app(scope, receive, send):
            raise RuntimeError("raw app failure")

        with tempfile.TemporaryDirectory(prefix="option-a-raw-500-") as tmp:
            run_paths = RunPaths.create(Path(tmp) / "run")
            recorder = RunRecorder(
                run_dir=run_paths.run_dir, evidence_dir=run_paths.evidence_dir
            )
            gate = TurnGate(
                instance_id="instance-1",
                scripts={
                    "S1": [TurnSpec("S1", "T1", "x", content_sha256("x"))]
                },
            )
            wrapped = TestApp(
                inner_app,
                test_html=_minimal_html(),
                gate=gate,
                recorder=recorder,
                instance_id="instance-1",
            )
            gate.arm(
                instance_id="instance-1",
                session_id=7,
                script_id="S1",
                turn_id="T1",
            )
            sent = []

            async def receive():
                return {
                    "type": "http.request",
                    "body": b'{"message": "x"}',
                    "more_body": False,
                }

            async def send(message):
                sent.append(message)

            scope = {
                "type": "http",
                "asgi": {"version": "3.0"},
                "http_version": "1.1",
                "method": "POST",
                "scheme": "http",
                "path": "/api/sessions/7/messages",
                "raw_path": b"/api/sessions/7/messages",
                "query_string": b"",
                "root_path": "",
                "headers": [(b"content-type", b"application/json")],
                "client": ("127.0.0.1", 12345),
                "server": ("127.0.0.1", 8765),
            }
            try:
                asyncio.run(wrapped(scope, receive, send))
            finally:
                recorder.close()
            starts = [m for m in sent if m.get("type") == "http.response.start"]
            self.assertEqual(starts[0]["status"], 502)
            self.assertTrue(gate.run_state.failed)
            self.assertEqual(len(recorder.http), 1)
            self.assertEqual(recorder.http[-1]["status_code"], 502)
            self.assertEqual(recorder.http[-1]["inner_error_class"], "RuntimeError")
            self.assertNotIn("raw app failure", json.dumps(recorder.http[-1]))

    def test_send_failure_records_http_attempt(self):
        import asyncio

        async def inner_app(scope, receive, send):
            await send(
                {
                    "type": "http.response.start",
                    "status": 200,
                    "headers": [(b"content-type", b"application/json")],
                }
            )
            await send(
                {
                    "type": "http.response.body",
                    "body": b'{"user_message": {"id": 1}, "assistant_message": {"id": 2}}',
                }
            )

        with tempfile.TemporaryDirectory(prefix="option-a-send-") as tmp:
            run_paths = RunPaths.create(Path(tmp) / "run")
            recorder = RunRecorder(
                run_dir=run_paths.run_dir, evidence_dir=run_paths.evidence_dir
            )
            gate = TurnGate(
                instance_id="instance-1",
                scripts={
                    "S1": [TurnSpec("S1", "T1", "x", content_sha256("x"))]
                },
            )
            wrapped = TestApp(
                inner_app,
                test_html=_minimal_html(),
                gate=gate,
                recorder=recorder,
                instance_id="instance-1",
            )
            gate.arm(
                instance_id="instance-1",
                session_id=7,
                script_id="S1",
                turn_id="T1",
            )

            async def receive():
                return {
                    "type": "http.request",
                    "body": b'{"message": "x"}',
                    "more_body": False,
                }

            async def send(message):
                raise RuntimeError("client disappeared")

            scope = {
                "type": "http",
                "asgi": {"version": "3.0"},
                "http_version": "1.1",
                "method": "POST",
                "scheme": "http",
                "path": "/api/sessions/7/messages",
                "raw_path": b"/api/sessions/7/messages",
                "query_string": b"",
                "root_path": "",
                "headers": [(b"content-type", b"application/json")],
                "client": ("127.0.0.1", 12345),
                "server": ("127.0.0.1", 8765),
            }
            try:
                asyncio.run(wrapped(scope, receive, send))
            finally:
                recorder.close()
            self.assertTrue(gate.run_state.failed)
            self.assertEqual(len(recorder.http), 1)
            self.assertTrue(recorder.http[-1]["send_failed"])
            self.assertEqual(recorder.http[-1]["status_code"], 200)

    def test_cancelled_request_records_http_attempt(self):
        import asyncio

        async def route(session_id: int):
            raise asyncio.CancelledError

        with tempfile.TemporaryDirectory(prefix="option-a-cancel-") as tmp:
            wrapped, gate, recorder = self._app(tmp, route)
            try:
                with TestClient(wrapped) as client:
                    self._arm(client)
                    with self.assertRaises(CancelledError):
                        client.post(
                            "/api/sessions/7/messages", json={"message": "x"}
                        )
                self.assertTrue(gate.run_state.failed)
                self.assertEqual(len(recorder.http), 1)
                self.assertEqual(
                    recorder.http[-1]["inner_error_class"], "CancelledError"
                )
            finally:
                recorder.close()


class TestStopIdentity(unittest.TestCase):
    def _make_run(self, tmp: str) -> tuple[Any, list[str]]:
        paths = RunPaths.create(Path(tmp) / "run")
        meta = {
            "instance_id": "instance-1",
            "paths": {"run_dir": str(paths.run_dir)},
            "host": "127.0.0.1",
            "port": 8765,
        }
        paths.meta_path.write_text(
            json.dumps(meta, ensure_ascii=False, sort_keys=True), encoding="utf-8"
        )
        launch_argv = _normalized_launch_argv(paths.run_dir, "127.0.0.1", 8765)
        payload = {
            "pid": 12345,
            "pid_start": "Mon Oct 31 10:00:00 2026",
            "instance_id": "instance-1",
            "host": "127.0.0.1",
            "port": 8765,
            "run_dir": str(paths.run_dir),
            "launch_argv": launch_argv,
            "run_meta_sha256": hashlib.sha256(
                paths.meta_path.read_bytes()
            ).hexdigest(),
        }
        paths.pid_path.write_text(
            json.dumps(payload, ensure_ascii=False, sort_keys=True), encoding="utf-8"
        )
        return paths, launch_argv

    def test_stop_refuses_when_start_identity_does_not_match(self):
        with tempfile.TemporaryDirectory(prefix="option-a-stop-start-") as tmp:
            paths, _ = self._make_run(tmp)
            with (
                patch(
                    "experiments.option_a_web.launcher._process_start_time",
                    return_value="different start",
                ),
                patch("experiments.option_a_web.launcher.os.kill") as kill,
            ):
                with self.assertRaises(IsolationError):
                    stop_instance(paths.run_dir)
                self.assertEqual(
                    [call.args[1] for call in kill.call_args_list], [0]
                )

    def test_stop_refuses_wrong_launch_argv(self):
        with tempfile.TemporaryDirectory(prefix="option-a-stop-argv-") as tmp:
            paths, _ = self._make_run(tmp)
            payload = json.loads(paths.pid_path.read_text(encoding="utf-8"))
            payload["launch_argv"] = ["python", "-m", "wrong"]
            paths.pid_path.write_text(json.dumps(payload), encoding="utf-8")
            with patch("experiments.option_a_web.launcher.os.kill") as kill:
                with self.assertRaises(IsolationError):
                    stop_instance(paths.run_dir)
                kill.assert_not_called()

    def test_stop_refuses_process_that_does_not_own_port(self):
        with tempfile.TemporaryDirectory(prefix="option-a-stop-port-") as tmp:
            paths, _ = self._make_run(tmp)
            with (
                patch(
                    "experiments.option_a_web.launcher._process_start_time",
                    return_value="Mon Oct 31 10:00:00 2026",
                ),
                patch(
                    "experiments.option_a_web.launcher._process_command",
                    return_value=(
                        "python -m experiments.option_a_web.launcher "
                        f"--run-dir {paths.run_dir} --port 8765"
                    ),
                ),
                patch(
                    "experiments.option_a_web.launcher._port_owner_pids",
                    return_value={99999},
                ),
                patch("experiments.option_a_web.launcher.os.kill") as kill,
            ):
                kill.return_value = None
                with self.assertRaises(IsolationError):
                    stop_instance(paths.run_dir)
                self.assertNotIn(
                    signal.SIGTERM,
                    [call.args[1] for call in kill.call_args_list if len(call.args) > 1],
                )

    def test_stop_sends_sigterm_after_identity_checks(self):
        with tempfile.TemporaryDirectory(prefix="option-a-stop-ok-") as tmp:
            paths, _ = self._make_run(tmp)
            with (
                patch(
                    "experiments.option_a_web.launcher._process_start_time",
                    return_value="Mon Oct 31 10:00:00 2026",
                ),
                patch(
                    "experiments.option_a_web.launcher._process_command",
                    return_value=(
                        "python -m experiments.option_a_web.launcher "
                        f"--run-dir {paths.run_dir} --port 8765"
                    ),
                ),
                patch(
                    "experiments.option_a_web.launcher._port_owner_pids",
                    return_value={12345},
                ),
                patch("experiments.option_a_web.launcher.os.kill") as kill,
            ):
                kill.side_effect = [None, None, ProcessLookupError]
                stop_instance(paths.run_dir)
                self.assertIn(
                    signal.SIGTERM,
                    [call.args[1] for call in kill.call_args_list if len(call.args) > 1],
                )


class TestIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix="option-a-web-integration-")
        cls.client_impl = CountingClient()
        cls.instance = prepare_instance(
            run_dir=Path(cls.tmp.name) / "run",
            client_factory=lambda: cls.client_impl,
        )
        cls.client_context = TestClient(cls.instance.app)
        cls.client = cls.client_context.__enter__()
        cls.sessions: list[int] = []
        cls.extra_sessions: list[int] = []

    @classmethod
    def tearDownClass(cls):
        cls.client_context.__exit__(None, None, None)
        cls.instance.close()
        cls.tmp.cleanup()

    def test_01_banner_and_static_assets(self):
        root = self.client.get("/")
        self.assertEqual(root.status_code, 200)
        self.assertIn("option-a-test-banner", root.text)
        self.assertIn("data-test-instance=\"1\"", root.text)
        self.assertIn("A Prompt 测试变体", root.text)
        self.assertIn(self.instance.frozen.system_sha256, root.text)
        self.assertIn("/static/app.js", root.text)
        static = self.client.get("/static/app.js")
        self.assertEqual(static.status_code, 200)
        self.assertGreater(len(static.text), 1000)

    def test_02_unplanned_routes_do_not_call_model(self):
        before = self.client_impl.calls
        session = self.client.post("/api/sessions", json={"title": "blocked"})
        self.assertEqual(session.status_code, 201)
        sid = session.json()["id"]
        self.extra_sessions.append(sid)
        self.assertEqual(
            self.client.post("/api/chat", json={"message": "unplanned"}).status_code,
            403,
        )
        self.assertEqual(
            self.client.get(f"/api/sessions/{sid}/history-reviews").status_code,
            403,
        )
        self.assertEqual(
            self.client.patch(
                f"/api/sessions/{sid}/interaction-mode",
                json={"interaction_mode": "corrective"},
            ).status_code,
            403,
        )
        self.assertEqual(
            self.client.patch(
                f"/api/sessions/{sid}/llm-profile",
                json={"llm_profile_id": "default", "acknowledge_remote_history": True},
            ).status_code,
            403,
        )
        self.assertEqual(self.client_impl.calls, before)

    def test_03_full_16_call_flow(self):
        self.assertEqual(self.instance.frozen.total_expected, 16)
        self.assertEqual(sum(len(turns) for turns in self.instance.frozen.scripts.values()), 16)
        for script_id, turns in self.instance.frozen.scripts.items():
            create = self.client.post(
                "/api/sessions", json={"title": f"Option A {script_id}"}
            )
            self.assertEqual(create.status_code, 201)
            session_id = create.json()["id"]
            self.sessions.append(session_id)
            self.assertEqual(create.json()["interaction_mode"], "receive_teaching")
            for turn in turns:
                arm = self.client.post(
                    "/__option_a__/arm",
                    json={
                        "instance_id": self.instance.instance_id,
                        "session_id": session_id,
                        "script_id": script_id,
                        "turn_id": turn.turn_id,
                    },
                )
                self.assertEqual(arm.status_code, 200, arm.text)
                response = self.client.post(
                    f"/api/sessions/{session_id}/messages",
                    json={"message": turn.content},
                )
                self.assertEqual(response.status_code, 200, response.text)
                body = response.json()
                self.assertEqual(body["user_message"]["content"], turn.content)
                self.assertTrue(body["assistant_message"]["content"].strip())
        self.assertEqual(self.client_impl.calls, 16)
        self.assertEqual(len(self.instance.recorder.calls), 16)
        self.assertEqual(len(self.instance.recorder.http), 16)
        self.assertEqual(len(self.sessions), 6)
        # The unplanned-route test deliberately created one additional
        # session.  Report the real database count instead of claiming that
        # the six script sessions are the whole database.
        all_sessions = self.client.get("/api/sessions").json()
        self.assertEqual(
            len(all_sessions),
            len(self.sessions) + len(self.extra_sessions),
        )
        self.assertFalse(self.instance.gate.run_state.failed)

        all_messages: list[dict[str, Any]] = []
        for session_id in self.sessions:
            messages = self.client.get(f"/api/sessions/{session_id}/messages")
            self.assertEqual(messages.status_code, 200)
            all_messages.extend(messages.json())
        self.assertEqual(len(all_messages), 32)
        for message in all_messages:
            self.assertEqual(message["interaction_mode_snapshot"], "receive_teaching")
            self.assertEqual(
                message["prompt_version_snapshot"], self.instance.frozen.prompt_version
            )
            self.assertNotEqual(
                message["prompt_version_snapshot"], "receive-teaching-v7"
            )
            self.assertEqual(message["llm_profile_id_snapshot"], "default")
            self.assertEqual(message["llm_profile_kind_snapshot"], "api")
            self.assertEqual(message["llm_model_snapshot"], self.instance.model)

    def test_04_calls_are_associated_and_parameters_are_explicit(self):
        self.assertEqual(len(self.instance.recorder.calls), 16)
        for call in self.instance.recorder.calls:
            self.assertEqual(call["request"]["system_sha256"], self.instance.frozen.system_sha256)
            self.assertEqual(call["model"]["request_model"], self.instance.model)
            self.assertEqual(call["model"]["upstream_response_model"], "fake-upstream-model")
            self.assertTrue(call["client"]["synthetic"])
            self.assertEqual(call["parameters"]["original_from_chat_service"]["temperature"], None)
            self.assertEqual(call["parameters"]["original_from_chat_service"]["max_tokens"], None)
            self.assertEqual(call["parameters"]["effective_sent_by_wrapper"]["temperature"], 0.0)
            self.assertEqual(call["parameters"]["effective_sent_by_wrapper"]["max_tokens"], 800)
            self.assertEqual(call["parameters"]["effective_sent_by_wrapper"]["timeout_seconds"], 30.0)
            self.assertFalse(call["parameters"]["effective_sent_by_wrapper"]["reasoning_effort_sent"])
            self.assertEqual(call["response"], "fake reply")
            self.assertTrue(call["call_id"])
            self.assertTrue(call["request_id"])

    def test_05_snapshot_endpoint_and_manifest_metadata(self):
        session_id = self.sessions[0]
        response = self.client.post(
            "/__option_a__/snapshot",
            json={
                "instance_id": self.instance.instance_id,
                "session_id": session_id,
                "script_id": "S1",
                "turn_id": "T1",
            },
        )
        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertEqual(payload["errors"], [])
        self.assertEqual(payload["prompt_version"], self.instance.frozen.prompt_version)
        self.assertEqual(payload["system_sha256"], self.instance.frozen.system_sha256)
        meta = json.loads(self.instance.paths.meta_path.read_text(encoding="utf-8"))
        self.assertEqual(meta["condition"], "A")
        self.assertEqual(meta["system_sha256"], self.instance.frozen.system_sha256)
        self.assertEqual(meta["database"]["database_path"], str(self.instance.paths.db_path))
        self.assertTrue(self.instance.paths.db_path.is_file())
        self.assertEqual(meta["total_expected"], 16)
        self.assertEqual(meta["frontend"]["source_index_sha256"], self.instance.source_index_sha256)

    def test_06_snapshot_verifier_rejects_missing_and_mismatched_messages(self):
        if not self.sessions:
            self.skipTest("full-flow test did not run")
        session_id = self.sessions[0]
        messages = self.client.get(f"/api/sessions/{session_id}/messages").json()
        self.assertEqual(len(messages), 4)
        user_id = messages[0]["id"]
        assistant_id = messages[1]["id"]
        verifier = _verify_snapshot_factory(self.instance.frozen, self.instance.model)

        missing = verifier(session_id, [999999, 999998], "S1", "T1")
        self.assertTrue(missing["errors"])

        wrong_roles = verifier(session_id, [assistant_id, user_id], "S1", "T1")
        self.assertTrue(wrong_roles["errors"])

        duplicate = verifier(session_id, [user_id, user_id], "S1", "T1")
        self.assertTrue(duplicate["errors"])

        wrong_turn = verifier(session_id, [user_id, assistant_id], "S1", "T2")
        self.assertTrue(wrong_turn["errors"])



class TestRecordingContract(unittest.TestCase):
    def _wrapper(self, tmp: str, inner: Any):
        run_paths = RunPaths.create(Path(tmp) / "run")
        recorder = RunRecorder(
            run_dir=run_paths.run_dir, evidence_dir=run_paths.evidence_dir
        )
        system = "system text"
        wrapper = RecordingLLMClient(
            inner,
            recorder=recorder,
            system_prompt=system,
            system_sha256=content_sha256(system),
            profile_id="default",
            request_model="synthetic-a-model",
            synthetic=True,
        )
        state = RunState("instance-1")
        arm = ArmedTurn(
            instance_id="instance-1",
            session_id=1,
            script_id="S1",
            turn_id="T1",
            content_sha256=content_sha256("user"),
        )
        context = RequestContext(
            request_id="request-1", session_id=1, arm=arm, run_state=state
        )
        return recorder, wrapper, context, state

    def test_second_generate_in_same_request_is_rejected_before_inner(self):
        import asyncio

        with tempfile.TemporaryDirectory(prefix="option-a-claim-") as tmp:
            inner = CountingClient()
            recorder, wrapper, context, state = self._wrapper(tmp, inner)
            token = set_request_context(context)
            try:
                result = asyncio.run(
                    wrapper.generate(
                        [
                            {"role": "system", "content": "system text"},
                            {"role": "user", "content": "user"},
                        ]
                    )
                )
                self.assertEqual(result, "fake reply")
                with self.assertRaises(RecordingError):
                    asyncio.run(
                        wrapper.generate(
                            [
                                {"role": "system", "content": "system text"},
                                {"role": "user", "content": "user"},
                            ]
                        )
                    )
            finally:
                reset_request_context(token)
            self.assertEqual(inner.calls, 1)
            self.assertEqual(len(recorder.calls), 1)
            self.assertTrue(state.failed)
            recorder.close()

    def test_only_stop_finish_reason_is_confirmed(self):
        import asyncio

        class FinishClient:
            def __init__(self, finish_reason):
                self.finish_reason = finish_reason
                self.calls = 0

            async def generate(self, messages, **kwargs):
                self.calls += 1
                if kwargs.get("diagnostics") is not None:
                    kwargs["diagnostics"].http_status = 200
                    kwargs["diagnostics"].finish_reason = self.finish_reason
                return "visible reply"

        for reason in (None, "content_filter", "tool_calls", "unknown"):
            with self.subTest(reason=reason), tempfile.TemporaryDirectory(
                prefix="option-a-finish-"
            ) as tmp:
                inner = FinishClient(reason)
                recorder, wrapper, context, state = self._wrapper(tmp, inner)
                token = set_request_context(context)
                try:
                    with self.assertRaises(Exception):
                        asyncio.run(
                            wrapper.generate(
                                [
                                    {"role": "system", "content": "system text"},
                                    {"role": "user", "content": "user"},
                                ]
                            )
                        )
                finally:
                    reset_request_context(token)
                self.assertEqual(inner.calls, 1)
                self.assertEqual(len(recorder.calls), 1)
                self.assertEqual(recorder.calls[0]["status"], "unconfirmed")
                self.assertEqual(recorder.calls[0]["response"], "visible reply")
                self.assertTrue(state.failed)
                recorder.close()


class TestSanitizationAndStop(unittest.TestCase):
    def test_error_recording_does_not_include_exception_detail(self):
        with tempfile.TemporaryDirectory(prefix="option-a-error-") as tmp:
            run_paths = RunPaths.create(Path(tmp) / "run")
            recorder = RunRecorder(
                run_dir=run_paths.run_dir, evidence_dir=run_paths.evidence_dir
            )
            inner = FailingClient("sk-super-secret-upstream-message")
            system = "system text"
            wrapper = RecordingLLMClient(
                inner,
                recorder=recorder,
                system_prompt=system,
                system_sha256=content_sha256(system),
                profile_id="default",
                request_model="synthetic-a-model",
                synthetic=True,
            )
            state = RunState("instance-1")
            arm = ArmedTurn(
                instance_id="instance-1",
                session_id=1,
                script_id="S1",
                turn_id="T1",
                content_sha256=content_sha256("user"),
            )
            context = RequestContext(
                request_id="request-1",
                session_id=1,
                arm=arm,
                run_state=state,
            )
            token = set_request_context(context)
            try:
                with self.assertRaises(Exception):
                    import asyncio

                    asyncio.run(
                        wrapper.generate(
                            [
                                {"role": "system", "content": system},
                                {"role": "user", "content": "user"},
                            ]
                        )
                    )
            finally:
                reset_request_context(token)
            self.assertEqual(inner.calls, 1)
            self.assertEqual(len(recorder.calls), 1)
            serialized = json.dumps(recorder.calls, ensure_ascii=False)
            self.assertNotIn("sk-super-secret", serialized)
            self.assertNotIn("upstream-message", serialized)
            self.assertTrue(state.failed)
            recorder.close()

    def _record_wrapper(self, tmp: str, inner: Any) -> tuple[Any, Any, Any, Any]:
        run_paths = RunPaths.create(Path(tmp) / "run")
        recorder = RunRecorder(run_dir=run_paths.run_dir, evidence_dir=run_paths.evidence_dir)
        system = "system text"
        wrapper = RecordingLLMClient(
            inner,
            recorder=recorder,
            system_prompt=system,
            system_sha256=content_sha256(system),
            profile_id="default",
            request_model="synthetic-a-model",
            synthetic=True,
        )
        state = RunState("instance-1")
        arm = ArmedTurn(
            instance_id="instance-1",
            session_id=1,
            script_id="S1",
            turn_id="T1",
            content_sha256=content_sha256("user"),
        )
        context = RequestContext(request_id="request-1", session_id=1, arm=arm, run_state=state)
        return recorder, wrapper, context, state

    def test_empty_response_is_recorded_and_batch_stopped(self):
        import asyncio

        class EmptyClient:
            def __init__(self): self.calls = 0
            async def generate(self, messages, **kwargs):
                self.calls += 1
                if kwargs.get("diagnostics") is not None:
                    kwargs["diagnostics"].http_status = 200
                    kwargs["diagnostics"].finish_reason = "stop"
                return ""

        with tempfile.TemporaryDirectory(prefix="option-a-empty-") as tmp:
            inner = EmptyClient()
            recorder, wrapper, context, state = self._record_wrapper(tmp, inner)
            token = set_request_context(context)
            try:
                result = asyncio.run(wrapper.generate([
                    {"role": "system", "content": "system text"},
                    {"role": "user", "content": "user"},
                ]))
            finally:
                reset_request_context(token)
            self.assertEqual(result, "")
            self.assertEqual(inner.calls, 1)
            self.assertEqual(len(recorder.calls), 1)
            self.assertEqual(recorder.calls[0]["status"], "empty")
            self.assertTrue(state.failed)
            recorder.close()

    def test_truncated_response_is_recorded_and_batch_stopped(self):
        import asyncio

        class TruncatedClient:
            def __init__(self): self.calls = 0
            async def generate(self, messages, **kwargs):
                self.calls += 1
                if kwargs.get("diagnostics") is not None:
                    kwargs["diagnostics"].http_status = 200
                    kwargs["diagnostics"].finish_reason = "length"
                return "partial"

        with tempfile.TemporaryDirectory(prefix="option-a-truncated-") as tmp:
            inner = TruncatedClient()
            recorder, wrapper, context, state = self._record_wrapper(tmp, inner)
            token = set_request_context(context)
            try:
                with self.assertRaises(Exception):
                    asyncio.run(wrapper.generate([
                        {"role": "system", "content": "system text"},
                        {"role": "user", "content": "user"},
                    ]))
            finally:
                reset_request_context(token)
            self.assertEqual(inner.calls, 1)
            self.assertEqual(len(recorder.calls), 1)
            self.assertEqual(recorder.calls[0]["status"], "truncated")
            self.assertTrue(state.failed)
            recorder.close()


if __name__ == "__main__":
    unittest.main()
