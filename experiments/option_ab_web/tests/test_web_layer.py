"""Fake-only isolation tests for the Option A+B web test harness."""

from __future__ import annotations

import asyncio
import hashlib
import json
import shutil
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
from unittest import mock

from fastapi.testclient import TestClient

from experiments.option_ab_web.asgi import TestApp
from experiments.option_ab_web.launcher import (
    IsolationError,
    load_frozen_spec,
    prepare_instance,
)
from experiments.option_ab_web.recording import (
    RecordingError,
    RecordingLLMClient,
    RequestContext,
    RunRecorder,
    reset_request_context,
    set_request_context,
)
from experiments.option_ab_web.state import (
    ArmedTurn,
    BatchState,
    GateError,
    TurnSpec,
    content_sha256,
)


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


class _Fake:
    kind = "fake"

    def __init__(self, *, finish: str = "stop", response: str = "ok") -> None:
        self.finish = finish
        self.response = response
        self.calls = 0
        self.message_batches: list[list[dict[str, str]]] = []

    async def generate(self, messages, *, response_format=None, temperature=None, max_tokens=None, diagnostics=None):
        self.calls += 1
        self.message_batches.append([dict(item) for item in messages])
        if diagnostics is not None:
            diagnostics.finish_reason = self.finish
        if self.finish == "raise":
            raise RuntimeError("synthetic failure")
        if self.finish == "cancel":
            raise asyncio.CancelledError()
        return self.response


class _DummyState:
    def __init__(self) -> None:
        self.failure: str | None = None
        self.cancelled = False

    def fail(self, reason: str) -> None:
        self.failure = self.failure or reason

    def cancel(self) -> None:
        self.cancelled = True


class _Driver:
    def __init__(self, frozen) -> None:
        self.state = BatchState(instance_id="unit", spec=frozen)
        self.session_id = 0
        self.uid = 1000
        self.aid = 2000

    def new_session(self) -> int:
        self.state.reserve_session_creation()
        self.session_id += 1
        return self.session_id

    def turn(self, script_id: str, turn_id: str, content: str) -> None:
        armed = self.state.arm(
            session_id=self.session_id, script_id=script_id, turn_id=turn_id
        )
        self.state.consume(session_id=self.session_id, content=content)
        self.uid += 1
        self.aid += 1
        self.state.complete(
            armed=armed, user_message_id=self.uid, assistant_message_id=self.aid
        )

    def branch_t2(self, script_id: str, content: str) -> None:
        armed = self.state.consume(session_id=self.session_id, content=content)
        self.uid += 1
        self.aid += 1
        self.state.complete(
            armed=armed, user_message_id=self.uid, assistant_message_id=self.aid
        )

    def previous_until(self, target: str) -> None:
        for script_id in ["S1", "S2", "S3", "S4", "S5", "S6", "I1"]:
            if script_id == target:
                return
            self.new_session()
            for turn in self.state.spec.script(script_id).turns:
                self.turn(script_id, turn.turn_id, turn.content)

    def to_pause(self, target: str = "I2") -> None:
        self.previous_until(target)
        self.new_session()
        for turn in self.state.spec.script(target).turns[:2]:
            self.turn(target, turn.turn_id, turn.content)


class TestStateMachine(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.frozen = load_frozen_spec()

    def test_sequence_skip_and_duplicate_are_rejected(self) -> None:
        d = _Driver(self.frozen)
        sid = d.new_session()
        with self.assertRaises(GateError):
            d.state.arm(session_id=sid, script_id="S2", turn_id="T1")
        d.turn("S1", "T1", self.frozen.script("S1").turns[0].content)
        with self.assertRaises(GateError):
            d.state.arm(session_id=sid, script_id="S1", turn_id="T1")
        with self.assertRaises(GateError):
            d.state.arm(session_id=sid + 1, script_id="S1", turn_id="T2")

    def test_none_is_terminal_and_has_no_t2(self) -> None:
        d = _Driver(self.frozen)
        d.to_pause("I2")
        sid = d.session_id
        q1 = d.state.completed_turn("I2", "Q1")
        assert q1 is not None
        selection = d.state.select_branch(
            session_id=sid,
            script_id="I2",
            branch_id="none",
            q1_assistant_message_id=q1.assistant_message_id,
            selected_by="human",
        )
        self.assertEqual(selection.t2_content, None)
        self.assertEqual(d.state.total_consumed, 20)
        self.assertEqual(d.state.state_snapshot()["current_script"], "I3")
        with self.assertRaises(GateError):
            d.state.select_branch(
                session_id=sid,
                script_id="I2",
                branch_id="bc",
                q1_assistant_message_id=q1.assistant_message_id,
                selected_by="human",
            )

    def test_all_i2_branches_arm_exactly_one_t2(self) -> None:
        for branch_id in ("none", "bc", "direct_ac", "no_extra"):
            d = _Driver(self.frozen)
            d.to_pause("I2")
            sid = d.session_id
            q1 = d.state.completed_turn("I2", "Q1")
            assert q1 is not None
            selection = d.state.select_branch(
                session_id=sid,
                script_id="I2",
                branch_id=branch_id,
                q1_assistant_message_id=q1.assistant_message_id,
                selected_by="human",
            )
            if branch_id == "none":
                self.assertEqual(d.state.total_consumed, 20)
                self.assertEqual(d.state.state_snapshot()["current_script"], "I3")
                continue
            self.assertTrue(selection.t2_content)
            self.assertEqual(d.state.total_consumed, 20)
            self.assertEqual(d.state.state_snapshot()["current_script"], "I2")
            with self.assertRaises(GateError):
                d.state.arm(session_id=sid, script_id="I2", turn_id="T2")
            d.branch_t2("I2", selection.t2_content)
            self.assertEqual(d.state.total_consumed, 21)
            self.assertEqual(d.state.state_snapshot()["current_script"], "I3")
            with self.assertRaises(GateError):
                d.state.consume(session_id=sid, content=selection.t2_content)

    def test_all_i3_branches(self) -> None:
        for branch_id in ("none", "rule_supply", "direct_ac", "no_extra"):
            d = _Driver(self.frozen)
            d.to_pause("I2")
            q1_i2 = d.state.completed_turn("I2", "Q1")
            assert q1_i2 is not None
            d.state.select_branch(
                session_id=d.session_id,
                script_id="I2",
                branch_id="none",
                q1_assistant_message_id=q1_i2.assistant_message_id,
                selected_by="human",
            )
            d.new_session()
            for turn in self.frozen.script("I3").turns[:2]:
                d.turn("I3", turn.turn_id, turn.content)
            self.assertTrue(d.state.state_snapshot()["paused_human_classification"])
            q1_i3 = d.state.completed_turn("I3", "Q1")
            assert q1_i3 is not None
            selection = d.state.select_branch(
                session_id=d.session_id,
                script_id="I3",
                branch_id=branch_id,
                q1_assistant_message_id=q1_i3.assistant_message_id,
                selected_by="human",
            )
            if branch_id == "none":
                self.assertEqual(d.state.total_consumed, 22)
            else:
                self.assertTrue(selection.t2_content)
                d.branch_t2("I3", selection.t2_content)
                self.assertEqual(d.state.total_consumed, 23)
            # I4 remains after I3.
            d.new_session()
            for turn in self.frozen.script("I4").turns:
                d.turn("I4", turn.turn_id, turn.content)
            self.assertTrue(d.state.batch_completed)

    def test_concurrent_selection_has_one_winner(self) -> None:
        d = _Driver(self.frozen)
        d.to_pause("I2")
        q1 = d.state.completed_turn("I2", "Q1")
        assert q1 is not None

        def choose(branch: str):
            try:
                return d.state.select_branch(
                    session_id=d.session_id,
                    script_id="I2",
                    branch_id=branch,
                    q1_assistant_message_id=q1.assistant_message_id,
                    selected_by="human",
                )
            except GateError as exc:
                return exc

        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(choose, ["bc", "direct_ac"]))
        winners = [item for item in results if not isinstance(item, GateError)]
        losers = [item for item in results if isinstance(item, GateError)]
        self.assertEqual(len(winners), 1)
        self.assertEqual(len(losers), 1)
        self.assertFalse(losers[0].fatal)
        self.assertEqual(d.state.total_consumed, 20)

    def test_session_limit_and_wrong_session(self) -> None:
        d = _Driver(self.frozen)
        for _ in range(10):
            self.assertTrue(d.state.reserve_session_creation())
        self.assertFalse(d.state.reserve_session_creation())
        d.state.release_session_creation()
        self.assertEqual(d.state.arm(session_id=1, script_id="S1", turn_id="T1").session_id, 1)

    def test_t2_cannot_be_armed_after_completed(self) -> None:
        d = _Driver(self.frozen)
        d.to_pause("I2")
        q1 = d.state.completed_turn("I2", "Q1")
        assert q1 is not None
        sel = d.state.select_branch(
            session_id=d.session_id,
            script_id="I2",
            branch_id="bc",
            q1_assistant_message_id=q1.assistant_message_id,
            selected_by="human",
        )
        d.branch_t2("I2", sel.t2_content)
        self.assertEqual(d.state.state_snapshot()["current_script"], "I3")
        with self.assertRaises(GateError):
            d.state.select_branch(
                session_id=d.session_id,
                script_id="I2",
                branch_id="no_extra",
                q1_assistant_message_id=q1.assistant_message_id,
                selected_by="human",
            )


    def test_active_next_turn_with_wrong_session_is_rejected(self) -> None:
        d = _Driver(load_frozen_spec())
        good = d.new_session()
        d.turn("S1", "T1", d.state.spec.script("S1").turns[0].content)
        wrong = d.new_session()
        q1 = d.state.spec.script("S1").turns[1]
        with self.assertRaises(GateError):
            d.state.arm(session_id=wrong, script_id="S1", turn_id="Q1")
        self.assertIsNone(d.state.completed_turn("S1", "Q1"))
        self.assertEqual(d.state.state_snapshot()["current_script"], "S1")
        self.assertEqual(d.state.state_snapshot()["total_consumed"], 1)


class TestRecording(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.root = Path(tempfile.mkdtemp(prefix="option-ab-web-unit-"))
        self.evidence = self.root / "evidence"
        self.evidence.mkdir()
        self.recorder = RunRecorder(run_dir=self.root, evidence_dir=self.evidence)
        self.state = _DummyState()
        self.arm = ArmedTurn(
            instance_id="unit",
            session_id=1,
            script_id="S1",
            turn_id="T1",
            content_sha256=content_sha256("hello"),
            branch_id=None,
        )
        self.context = RequestContext(
            request_id="req",
            session_id=1,
            armed=self.arm,
            state=self.state,
        )
        self.token = set_request_context(self.context)
        self.sys = "system"
        self.sys_sha = hashlib.sha256(self.sys.encode()).hexdigest()

    async def asyncTearDown(self) -> None:
        reset_request_context(self.token)
        self.recorder.close()
        shutil.rmtree(self.root, ignore_errors=True)

    def _client(self, inner) -> RecordingLLMClient:
        return RecordingLLMClient(
            inner,
            recorder=self.recorder,
            state=self.state,
            system_prompt=self.sys,
            system_sha256=self.sys_sha,
            prompt_version="option-ab-test",
            profile_id="default",
            request_model="fake",
            client_kind="fake",
            synthetic=True,
        )

    async def test_parameters_and_response_are_recorded(self) -> None:
        inner = _Fake()
        result = await self._client(inner).generate(
            [{"role": "system", "content": self.sys}, {"role": "user", "content": "hi"}],
            temperature=0.9,
            max_tokens=42,
        )
        self.assertEqual(result, "ok")
        record = _jsonl(self.evidence / "calls.jsonl")[0]
        self.assertEqual(record["status"], "completed")
        self.assertEqual(self.context.model_status(), "completed")
        self.assertEqual(record["original_chat_service_parameters"]["temperature"], 0.9)
        self.assertEqual(record["original_chat_service_parameters"]["max_tokens"], 42)
        effective = record["effective_parameters"]
        self.assertEqual(effective["temperature"], 0.0)
        self.assertEqual(effective["max_tokens"], 800)
        self.assertEqual(effective["timeout_seconds"], 30.0)
        self.assertIsNone(record["upstream_response_model"])
        self.assertIn("does not expose", record["upstream_model_unavailable_reason"])
        self.assertEqual(inner.calls, 1)

    async def test_second_generate_is_rejected_before_inner(self) -> None:
        inner = _Fake()
        client = self._client(inner)
        messages = [{"role": "system", "content": self.sys}, {"role": "user", "content": "hi"}]
        await client.generate(messages)
        with self.assertRaises(RecordingError):
            await client.generate(messages)
        self.assertEqual(inner.calls, 1)
        self.assertIsNotNone(self.state.failure)

    async def test_length_is_truncated_and_stops(self) -> None:
        inner = _Fake(finish="length", response="partial")
        with self.assertRaises(Exception):
            await self._client(inner).generate(
                [{"role": "system", "content": self.sys}, {"role": "user", "content": "hi"}]
            )
        record = _jsonl(self.evidence / "calls.jsonl")[0]
        self.assertEqual(record["status"], "truncated")
        self.assertEqual(self.context.model_status(), "truncated")
        self.assertIsNotNone(self.state.failure)

    async def test_empty_response_is_not_completed(self) -> None:
        inner = _Fake(finish="stop", response="")
        with self.assertRaises(Exception):
            await self._client(inner).generate(
                [{"role": "system", "content": self.sys}, {"role": "user", "content": "hi"}]
            )
        record = _jsonl(self.evidence / "calls.jsonl")[0]
        self.assertEqual(record["status"], "empty")
        self.assertEqual(self.context.model_status(), "empty")
        self.assertIsNotNone(self.state.failure)

    async def test_non_stop_finish_is_unconfirmed(self) -> None:
        inner = _Fake(finish="content_filter", response="visible text")
        with self.assertRaises(Exception):
            await self._client(inner).generate(
                [{"role": "system", "content": self.sys}, {"role": "user", "content": "hi"}]
            )
        record = _jsonl(self.evidence / "calls.jsonl")[0]
        self.assertEqual(record["status"], "unconfirmed")
        self.assertEqual(self.context.model_status(), "unconfirmed")
        self.assertEqual(record["response"], "visible text")
        self.assertIsNotNone(self.state.failure)

    async def test_cancellation_is_recorded(self) -> None:
        inner = _Fake(finish="cancel")
        with self.assertRaises(asyncio.CancelledError):
            await self._client(inner).generate(
                [{"role": "system", "content": self.sys}, {"role": "user", "content": "hi"}]
            )
        record = _jsonl(self.evidence / "calls.jsonl")[0]
        self.assertEqual(record["status"], "cancelled")
        self.assertEqual(self.context.model_status(), "cancelled")
        self.assertTrue(self.state.cancelled)

    async def test_timeout_stops_without_retry(self) -> None:
        class SlowClient:
            calls = 0

            async def generate(self, messages, **kwargs):
                self.calls += 1
                await asyncio.sleep(0.2)
                return "late"

        inner = SlowClient()
        client = RecordingLLMClient(
            inner,
            recorder=self.recorder,
            state=self.state,
            system_prompt=self.sys,
            system_sha256=self.sys_sha,
            prompt_version="option-ab-test",
            profile_id="default",
            request_model="fake",
            client_kind="fake",
            synthetic=True,
            timeout_seconds=0.01,
        )
        with self.assertRaises(asyncio.TimeoutError):
            await client.generate(
                [{"role": "system", "content": self.sys}, {"role": "user", "content": "hi"}]
            )
        self.assertEqual(inner.calls, 1)
        record = _jsonl(self.evidence / "calls.jsonl")[0]
        self.assertEqual(record["status"], "error")
        self.assertEqual(self.context.model_status(), "error")
        self.assertIsNotNone(self.state.failure)


class TestASGIProtocol(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.frozen = load_frozen_spec()

    def _app(self, raw) -> tuple[TestApp, Path, BatchState]:
        root = Path(tempfile.mkdtemp(prefix="option-ab-web-asgi-"))
        evidence = root / "evidence"
        evidence.mkdir()
        recorder = RunRecorder(run_dir=root, evidence_dir=evidence)
        state = BatchState(instance_id="asgi", spec=self.frozen)
        state.reserve_session_creation()
        turn = self.frozen.script("S1").turns[0]
        state.arm(session_id=1, script_id="S1", turn_id="T1")
        app = TestApp(
            raw,
            test_html="<html><body>B</body></html>",
            state=state,
            recorder=recorder,
            snapshot_verifier=lambda *args: {"errors": [], "ok": True},
            session_mode_getter=lambda _sid: "receive_teaching",
        )
        self.addCleanup(shutil.rmtree, root, True)
        return app, root, state

    def test_invalid_json_top_level_is_400_and_stops(self) -> None:
        async def raw(scope, receive, send):
            if scope["type"] == "lifespan":
                while True:
                    message = await receive()
                    if message["type"] == "lifespan.startup":
                        await send({"type": "lifespan.startup.complete"})
                    elif message["type"] == "lifespan.shutdown":
                        await send({"type": "lifespan.shutdown.complete"})
                        return
            raise AssertionError("real app must not be called")

        app, _, state = self._app(raw)
        with TestClient(app) as client:
            r = client.post("/api/sessions/1/messages", data=b"[]", headers={"content-type": "application/json"})
            self.assertEqual(r.status_code, 400)
            self.assertTrue(state.failed)

    def test_unhandled_app_exception_records_http_attempt(self) -> None:
        async def raw(scope, receive, send):
            if scope["type"] == "lifespan":
                while True:
                    message = await receive()
                    if message["type"] == "lifespan.startup":
                        await send({"type": "lifespan.startup.complete"})
                    elif message["type"] == "lifespan.shutdown":
                        await send({"type": "lifespan.shutdown.complete"})
                        return
            raise RuntimeError("synthetic")

        app, root, state = self._app(raw)
        content = self.frozen.script("S1").turns[0].content
        with TestClient(app) as client:
            r = client.post("/api/sessions/1/messages", json={"message": content})
            self.assertEqual(r.status_code, 502)
        records = _jsonl(root / "evidence" / "http.jsonl")
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["status"], "error")
        self.assertEqual(records[0]["outcome"], "failed")
        self.assertTrue(state.failed)
        self.assertNotIn("synthetic", json.dumps(records, ensure_ascii=False))


class ATestWebIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.parent = Path(tempfile.mkdtemp(prefix="option-ab-web-integration-"))
        cls.run_dir = cls.parent / "run"
        cls.fake = _Fake()
        cls.instance = prepare_instance(
            run_dir=cls.run_dir,
            client_factory=lambda: cls.fake,
        )
        cls.client = TestClient(cls.instance.app)
        cls.client.__enter__()
        response = cls.client.post("/api/sessions", json={})
        assert response.status_code == 201, response.text
        cls.session_s1 = response.json()["id"]

    @classmethod
    def tearDownClass(cls) -> None:
        cls.client.__exit__(None, None, None)
        cls.instance.close()
        shutil.rmtree(cls.parent, ignore_errors=True)

    def _calls(self) -> list[dict[str, Any]]:
        return _jsonl(self.run_dir / "evidence" / "calls.jsonl")

    def _http(self) -> list[dict[str, Any]]:
        return _jsonl(self.run_dir / "evidence" / "http.jsonl")

    def _state(self) -> dict[str, Any]:
        return self.client.get("/__option_ab__/state").json()

    def _arm(self, sid: int, script_id: str, turn_id: str) -> None:
        r = self.client.post(
            "/__option_ab__/arm",
            json={
                "instance_id": self.instance.instance_id,
                "session_id": sid,
                "script_id": script_id,
                "turn_id": turn_id,
            },
        )
        self.assertEqual(r.status_code, 200, r.text)

    def _send(self, sid: int, content: str) -> dict[str, Any]:
        r = self.client.post(f"/api/sessions/{sid}/messages", json={"message": content})
        self.assertEqual(r.status_code, 200, r.text)
        return r.json()

    def test_01_page_and_read_only_interfaces(self) -> None:
        page = self.client.get("/")
        self.assertEqual(page.status_code, 200)
        self.assertIn("option-ab-test-banner", page.text)
        self.assertIn("option-ab-branch-panel", page.text)
        panel_css = page.text.split("#option-ab-branch-panel {", 1)[1].split("}", 1)[0]
        self.assertIn("position: static", panel_css)
        self.assertNotIn("position: fixed", panel_css)
        self.assertLess(
            page.text.index('id="option-ab-branch-panel"'),
            page.text.index('class="app-layout"'),
        )
        self.assertIn("function clearBranchDisplay", page.text)
        self.assertIn("function syncLayout", page.text)
        self.assertIn("--option-ab-panel-height", page.text)
        self.assertIn(
            "height: calc(100vh - var(--option-ab-banner-height",
            page.text,
        )
        self.assertIn('t2pre.textContent = "";', page.text)
        self.assertIn("clearBranchDisplay();\n        panel.hidden = true;", page.text)
        self.assertIn(
            't2pre.textContent = "";\n      t2wrap.hidden = true;\n      const key',
            page.text,
        )
        self.assertEqual(self.client.get("/static/app.js").status_code, 200)
        self.assertEqual(self.client.get("/api/health").status_code, 200)
        profiles = self.client.get("/api/llm/profiles")
        self.assertEqual(profiles.status_code, 200)
        self.assertEqual(len(profiles.json()), 1)
        sessions = self.client.get("/api/sessions")
        self.assertEqual(sessions.status_code, 200)
        sid = self.session_s1
        self.assertEqual(self.client.get(f"/api/sessions/{sid}").status_code, 200)
        self.assertEqual(self.client.get(f"/api/sessions/{sid}/messages").status_code, 200)
        self.assertEqual(self.client.get(f"/api/sessions/{sid}/mode-switch-events").status_code, 200)
        self.assertEqual(self.client.get(f"/api/sessions/{sid}/history-reviews").status_code, 200)
        # The detail route is allowed; a missing review must be a real 404,
        # not a harness-synthesised success or an allow-list block.
        self.assertEqual(self.client.get(f"/api/sessions/{sid}/history-reviews/1").status_code, 404)
        self.assertEqual(self._calls(), [])

    def test_02_unplanned_writes_are_blocked(self) -> None:
        sid = self.session_s1
        cases = [
            ("post", "/api/chat", {"message": "x"}),
            ("post", f"/api/sessions/{sid}/history-reviews", {}),
            ("delete", f"/api/sessions/{sid}", None),
            ("patch", f"/api/sessions/{sid}", {"title": "x"}),
            ("patch", f"/api/sessions/{sid}/interaction-mode", {"interaction_mode": "corrective"}),
            ("patch", f"/api/sessions/{sid}/llm-profile", {"llm_profile_id": "default"}),
        ]
        for method, path, payload in cases:
            r = getattr(self.client, method)(path, json=payload) if payload is not None else getattr(self.client, method)(path)
            self.assertEqual(r.status_code, 403, (method, path, r.text))
        self.assertEqual(self._calls(), [])

    def test_03_full_run_branch_none_and_history(self) -> None:
        # Seed a mixed user/assistant history before the first model request.
        from backend.database import SessionLocal
        from backend.models import Message

        with SessionLocal() as db:
            for index in range(25):
                db.add(
                    Message(
                        session_id=self.session_s1,
                        role="user" if index % 2 == 0 else "assistant",
                        content=f"seed-{index:02d}",
                        llm_profile_id_snapshot="default",
                        llm_profile_kind_snapshot="fake",
                        llm_model_snapshot="fake",
                        interaction_mode_snapshot="receive_teaching",
                        prompt_version_snapshot="seed",
                    )
                )
            db.commit()

        # S1 uses the session created in setUpClass; the other nine are new.
        script_order = ["S1", "S2", "S3", "S4", "S5", "S6", "I1", "I2", "I3", "I4"]
        session_ids = {"S1": self.session_s1}
        branch_choices = {"I2": "bc", "I3": "none"}
        for script_id in script_order:
            if script_id != "S1":
                r = self.client.post("/api/sessions", json={})
                self.assertEqual(r.status_code, 201, r.text)
                session_ids[script_id] = r.json()["id"]
            sid = session_ids[script_id]
            script = self.instance.frozen.script(script_id)
            for turn in script.turns:
                self._arm(sid, script_id, turn.turn_id)
                self._send(sid, turn.content)
            state = self._state()
            if script_id == "I2":
                self.assertTrue(state["paused_human_classification"])
                q1_id = next(
                    item["q1_assistant_message_id"]
                    for item in state["scripts"]
                    if item["id"] == "I2"
                )
                selected = self.client.post(
                    "/__option_ab__/branch",
                    json={
                        "instance_id": self.instance.instance_id,
                        "session_id": sid,
                        "script_id": "I2",
                        "branch_id": "bc",
                        "q1_assistant_message_id": q1_id,
                        "selected_by": "human",
                    },
                )
                self.assertEqual(selected.status_code, 200, selected.text)
                payload = selected.json()
                self.assertTrue(payload["t2_content"])
                state = self._state()
                current = next(item for item in state["scripts"] if item["id"] == "I2")
                self.assertEqual(current["selected_t2_content"], payload["t2_content"])
                self._send(sid, payload["t2_content"])
            if script_id == "I3":
                self.assertTrue(state["paused_human_classification"])
                q1_id = next(
                    item["q1_assistant_message_id"]
                    for item in state["scripts"]
                    if item["id"] == "I3"
                )
                selected = self.client.post(
                    "/__option_ab__/branch",
                    json={
                        "instance_id": self.instance.instance_id,
                        "session_id": sid,
                        "script_id": "I3",
                        "branch_id": "none",
                        "q1_assistant_message_id": q1_id,
                        "selected_by": "human",
                    },
                )
                self.assertEqual(selected.status_code, 200, selected.text)
                self.assertFalse(selected.json()["t2_armed"])

        final_state = self._state()
        self.assertEqual(final_state["status"], "completed")
        self.assertEqual(final_state["total_consumed"], 25)
        self.assertEqual(final_state["sessions_created"], 10)
        self.assertEqual(self.client.post("/api/sessions", json={}).status_code, 409)

        calls = self._calls()
        self.assertEqual(len(calls), 25)
        self.assertTrue(all(item["status"] == "completed" for item in calls))
        http = self._http()
        self.assertEqual(len(http), 25)
        self.assertTrue(all(item["outcome"] == "completed" for item in http))
        branches = _jsonl(self.run_dir / "evidence" / "branches.jsonl")
        self.assertEqual([item["branch_id"] for item in branches], ["bc", "none"])

        i2_turns = [item["turn_id"] for item in calls if item["script_id"] == "I2"]
        self.assertEqual(i2_turns, ["T1", "Q1", "T2"])
        self.assertEqual(
            [item["turn_id"] for item in calls if item["script_id"] == "I3"],
            ["T1", "Q1"],
        )

        s1_request = next(
            item for item in calls if item["script_id"] == "S1" and item["turn_id"] == "T1"
        )["request_messages"]
        self.assertEqual(len(s1_request), 3)
        history = json.loads(s1_request[1]["content"])
        self.assertEqual(history["type"], "receive_teaching_history_background")
        history_contents = [item["content"] for item in history["messages"]]
        self.assertEqual(len(history_contents), 10)
        self.assertEqual(history_contents, [f"seed-{i:02d}" for i in range(6, 25, 2)])
        self.assertEqual(s1_request[2]["role"], "user")
        self.assertIn("A links to B", s1_request[2]["content"])
        http = _jsonl(self.run_dir / "evidence" / "http.jsonl")
        self.assertEqual(len(http), 25)
        self.assertTrue(all(item["model_status"] == "completed" for item in http))
        self.assertTrue(all(item["outcome"] == "completed" for item in http))
        for item in calls:
            self.assertIsNone(item["original_chat_service_parameters"]["temperature"])
            self.assertEqual(item["effective_parameters"]["temperature"], 0.0)
            self.assertEqual(item["effective_parameters"]["max_tokens"], 800)
            self.assertEqual(item["effective_parameters"]["timeout_seconds"], 30.0)
            self.assertIsNone(item["upstream_response_model"])
            self.assertIn("does not expose", item["upstream_model_unavailable_reason"])
            self.assertNotIn("Authorization", json.dumps(item, ensure_ascii=False))

        version = self.instance.frozen.prompt_version
        for sid in session_ids.values():
            messages = self.client.get(f"/api/sessions/{sid}/messages").json()
            for message in messages:
                if message["content"].startswith("seed-"):
                    continue
                self.assertEqual(message["interaction_mode_snapshot"], "receive_teaching")
                self.assertEqual(message["prompt_version_snapshot"], version)
                self.assertEqual(message["llm_model_snapshot"], "fake")

    def test_05_backend_loaded_is_rejected(self) -> None:
        with self.assertRaises(IsolationError):
            prepare_instance(
                run_dir=self.parent / "blocked-after-backend",
                client_factory=lambda: _Fake(),
            )

    def test_04_cli_requires_explicit_real_api_and_credentials(self) -> None:
        from experiments.option_ab_web.launcher import main

        with mock.patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(SystemExit):
                main(["--run-dir", str(self.parent / "cli-run")])
            with self.assertRaises(SystemExit):
                main(["--run-dir", str(self.parent / "cli-run2"), "--real-api"])


    def test_06_real_chat_service_wrong_session_has_no_side_effects(self) -> None:
        from sqlalchemy import func, select
        from backend.database import SessionLocal
        from backend.models import Message
        from experiments.option_ab_web.launcher import _session_mode_getter

        frozen = load_frozen_spec()
        root = Path(tempfile.mkdtemp(prefix="option-ab-binding-"))
        run_dir = root / "run"
        evidence = root / "evidence"
        run_dir.mkdir()
        evidence.mkdir()
        recorder = RunRecorder(run_dir=run_dir, evidence_dir=evidence)

        try:
            with TestClient(self.instance.app.app) as raw:
                first = raw.post("/api/sessions", json={})
                second = raw.post("/api/sessions", json={})
                self.assertEqual(first.status_code, 201)
                self.assertEqual(second.status_code, 201)
                session_one = first.json()["id"]
                session_wrong = second.json()["id"]

            state = BatchState(instance_id="binding-test", spec=frozen)
            app = TestApp(
                self.instance.app.app,
                test_html="<html><body>binding test</body></html>",
                state=state,
                recorder=recorder,
                snapshot_verifier=lambda *args: {"errors": [], "ok": True},
                session_mode_getter=_session_mode_getter(),
            )
            with TestClient(app) as client:
                t1 = frozen.script("S1").turns[0]
                armed = client.post(
                    "/__option_ab__/arm",
                    json={
                        "instance_id": "binding-test",
                        "session_id": session_one,
                        "script_id": "S1",
                        "turn_id": "T1",
                    },
                )
                self.assertEqual(armed.status_code, 200, armed.text)
                sent = client.post(
                    f"/api/sessions/{session_one}/messages",
                    json={"message": t1.content},
                )
                self.assertEqual(sent.status_code, 200, sent.text)
                calls_after_t1 = self.fake.calls

                wrong_arm = client.post(
                    "/__option_ab__/arm",
                    json={
                        "instance_id": "binding-test",
                        "session_id": session_wrong,
                        "script_id": "S1",
                        "turn_id": "Q1",
                    },
                )
                self.assertEqual(wrong_arm.status_code, 400, wrong_arm.text)
                self.assertEqual(self.fake.calls, calls_after_t1)
                self.assertEqual(state.state_snapshot()["status"], "failed")
                self.assertIsNone(state.completed_turn("S1", "Q1"))
                self.assertEqual(state.total_consumed, 1)

            with SessionLocal() as db:
                wrong_count = db.scalar(
                    select(func.count()).select_from(Message).where(
                        Message.session_id == session_wrong
                    )
                )
            self.assertEqual(wrong_count, 0)

            # A separate healthy state proves the normal same-session path.
            with TestClient(self.instance.app.app) as raw:
                third = raw.post("/api/sessions", json={})
                self.assertEqual(third.status_code, 201)
                session_ok = third.json()["id"]

            healthy_state = BatchState(instance_id="binding-ok", spec=frozen)
            healthy_app = TestApp(
                self.instance.app.app,
                test_html="<html><body>binding ok</body></html>",
                state=healthy_state,
                recorder=recorder,
                snapshot_verifier=lambda *args: {"errors": [], "ok": True},
                session_mode_getter=_session_mode_getter(),
            )
            with TestClient(healthy_app) as client:
                for turn in frozen.script("S1").turns:
                    armed = client.post(
                        "/__option_ab__/arm",
                        json={
                            "instance_id": "binding-ok",
                            "session_id": session_ok,
                            "script_id": "S1",
                            "turn_id": turn.turn_id,
                        },
                    )
                    self.assertEqual(armed.status_code, 200, armed.text)
                    sent = client.post(
                        f"/api/sessions/{session_ok}/messages",
                        json={"message": turn.content},
                    )
                    self.assertEqual(sent.status_code, 200, sent.text)
            self.assertEqual(healthy_state.state_snapshot()["status"], "running")
            self.assertIsNotNone(healthy_state.completed_turn("S1", "Q1"))
            self.assertEqual(self.fake.calls, calls_after_t1 + 2)
            self.assertTrue(
                any(
                    "I teach you these facts" in m["content"]
                    for m in self.fake.message_batches[-1]
                )
            )
        finally:
            recorder.close()
            shutil.rmtree(root, ignore_errors=True)


    def test_07_delivery_failure_keeps_gate_closed(self) -> None:
        from sqlalchemy import func, select
        from backend.database import SessionLocal
        from backend.models import Message
        from experiments.option_ab_web.launcher import _session_mode_getter

        frozen = load_frozen_spec()

        def asgi_scope(session_id: int) -> dict[str, Any]:
            return {
                "type": "http",
                "asgi": {"version": "3.0"},
                "http_version": "1.1",
                "method": "POST",
                "scheme": "http",
                "path": f"/api/sessions/{session_id}/messages",
                "raw_path": f"/api/sessions/{session_id}/messages".encode(),
                "query_string": b"",
                "headers": [(b"content-type", b"application/json")],
                "client": ("test", 1),
                "server": ("test", 80),
            }

        async def call_message(app, session_id: int, content: str, send) -> None:
            sent = False

            async def receive():
                nonlocal sent
                if not sent:
                    sent = True
                    return {
                        "type": "http.request",
                        "body": json.dumps({"message": content}).encode(),
                        "more_body": False,
                    }
                return {"type": "http.disconnect"}

            await app(asgi_scope(session_id), receive, send)

        with TestClient(self.instance.app.app) as raw:
            sessions = []
            for _ in range(3):
                response = raw.post("/api/sessions", json={})
                self.assertEqual(response.status_code, 201, response.text)
                sessions.append(response.json()["id"])

        roots: list[Path] = []
        try:
            for kind, session_id in zip(("start", "body", "cancel"), sessions):
                root = Path(tempfile.mkdtemp(prefix=f"option-ab-delivery-{kind}-"))
                roots.append(root)
                evidence = root / "evidence"
                evidence.mkdir()
                recorder = RunRecorder(run_dir=root / "run", evidence_dir=evidence)
                state = BatchState(instance_id=f"delivery-{kind}", spec=frozen)
                app = TestApp(
                    self.instance.app.app,
                    test_html="<html></html>",
                    state=state,
                    recorder=recorder,
                    snapshot_verifier=lambda *args: {"errors": [], "ok": True},
                    session_mode_getter=_session_mode_getter(),
                )

                t1 = frozen.script("S1").turns[0]
                state.arm(session_id=session_id, script_id="S1", turn_id="T1")
                normal_messages: list[dict[str, Any]] = []

                async def normal_send(message):
                    normal_messages.append(message)

                asyncio.run(call_message(app, session_id, t1.content, normal_send))
                self.assertEqual(state.state_snapshot()["total_consumed"], 1)
                self.assertEqual(normal_messages[0]["status"], 200)

                q1 = frozen.script("S1").turns[1]
                state.arm(session_id=session_id, script_id="S1", turn_id="Q1")
                calls_before = self.fake.calls
                sent: list[dict[str, Any]] = []

                async def failing_send(message):
                    sent.append(message)
                    if kind == "start" and message["type"] == "http.response.start":
                        raise BrokenPipeError("synthetic start failure")
                    if kind == "body" and message["type"] == "http.response.body":
                        raise BrokenPipeError("synthetic body failure")
                    if kind == "cancel" and message["type"] == "http.response.body":
                        raise asyncio.CancelledError()

                expected = asyncio.CancelledError if kind == "cancel" else BrokenPipeError
                with self.assertRaises(expected):
                    asyncio.run(call_message(app, session_id, q1.content, failing_send))

                self.assertEqual(self.fake.calls, calls_before + 1)
                self.assertIsNone(state.completed_turn("S1", "Q1"))
                self.assertIn(
                    state.state_snapshot()["status"],
                    {"failed", "cancelled"},
                )
                self.assertEqual(len(sent), 1 if kind == "start" else 2)
                with self.assertRaises(GateError):
                    state.arm(session_id=session_id, script_id="S1", turn_id="Q1")

                with SessionLocal() as db:
                    count = db.scalar(
                        select(func.count()).select_from(Message).where(
                            Message.session_id == session_id
                        )
                    )
                self.assertEqual(count, 4)
                http = _jsonl(evidence / "http.jsonl")
                q1_record = next(item for item in http if item["turn_id"] == "Q1")
                self.assertEqual(q1_record["model_status"], "completed")
                self.assertEqual(q1_record["database_status"], "saved")
                self.assertNotEqual(q1_record["delivery_status"], "delivered")
                self.assertIn(q1_record["outcome"], {"failed", "cancelled"})
                recorder.close()
        finally:
            for root in roots:
                shutil.rmtree(root, ignore_errors=True)

    def test_07b_http_model_status_uses_request_call_result(self) -> None:
        root = Path(tempfile.mkdtemp(prefix="option-ab-web-model-status-"))
        evidence = root / "evidence"
        evidence.mkdir()
        recorder = RunRecorder(run_dir=root / "run", evidence_dir=evidence)
        helper = TestApp(
            self.instance.app.app,
            test_html="<html></html>",
            state=self.instance.state,
            recorder=recorder,
            snapshot_verifier=lambda *args: {"errors": [], "ok": True},
            session_mode_getter=lambda _sid: "receive_teaching",
        )
        armed = ArmedTurn(
            instance_id="matrix",
            session_id=1,
            script_id="S1",
            turn_id="T1",
            content_sha256=content_sha256("hello"),
            branch_id=None,
        )
        body = json.dumps(
            {"user_message": {"id": 1}, "assistant_message": {"id": 2}}
        ).encode("utf-8")
        cases = [
            ("missing", [], {}, "missing", "not_delivered"),
            ("unknown", ["r:1"], {}, "unknown", "not_delivered"),
            ("duplicate", ["r:1", "r:2"], {"r:1": "completed", "r:2": "completed"}, "duplicate", "delivered"),
            ("completed", ["r:1"], {"r:1": "completed"}, "completed", "delivered"),
            ("delivery_failed", ["r:1"], {"r:1": "completed"}, "completed", "start_failed"),
            ("truncated", ["r:1"], {"r:1": "truncated"}, "truncated", "not_delivered"),
            ("empty", ["r:1"], {"r:1": "empty"}, "empty", "not_delivered"),
            ("cancelled", ["r:1"], {"r:1": "cancelled"}, "cancelled", "not_delivered"),
            ("error", ["r:1"], {"r:1": "error"}, "error", "not_delivered"),
            ("unconfirmed", ["r:1"], {"r:1": "unconfirmed"}, "unconfirmed", "not_delivered"),
        ]
        try:
            for request_id, calls, results, _, delivery in cases:
                context = RequestContext(
                    request_id=request_id,
                    session_id=1,
                    armed=armed,
                    state=self.instance.state,
                )
                context.calls = list(calls)
                context.call_results = dict(results)
                helper._record_message_http(
                    request_id=request_id,
                    armed=armed,
                    status=200,
                    body=body,
                    context=context,
                    error=None,
                    delivery_status=delivery,
                )
            records = _jsonl(evidence / "http.jsonl")
            self.assertEqual(
                [item["model_status"] for item in records],
                [case[3] for case in cases],
            )
            self.assertEqual(
                [item["model_calls"] for item in records],
                [len(case[1]) for case in cases],
            )
        finally:
            recorder.close()
            shutil.rmtree(root, ignore_errors=True)

    def test_08_snapshot_and_evidence_failures_stop_before_next_turn(self) -> None:
        from sqlalchemy import func, select
        from backend.database import SessionLocal
        from backend.models import Message
        from experiments.option_ab_web.launcher import _session_mode_getter

        frozen = load_frozen_spec()

        def new_harness(prefix: str, verifier):
            root = Path(tempfile.mkdtemp(prefix=prefix))
            evidence = root / "evidence"
            evidence.mkdir()
            recorder = RunRecorder(run_dir=root / "run", evidence_dir=evidence)
            state = BatchState(instance_id=prefix, spec=frozen)
            app = TestApp(
                self.instance.app.app,
                test_html="<html></html>",
                state=state,
                recorder=recorder,
                snapshot_verifier=verifier,
                session_mode_getter=_session_mode_getter(),
            )
            return root, recorder, state, app

        with TestClient(self.instance.app.app) as raw:
            snapshot_session = raw.post("/api/sessions", json={}).json()["id"]
            evidence_session = raw.post("/api/sessions", json={}).json()["id"]

        roots: list[Path] = []
        try:
            root, recorder, state, app = new_harness(
                "option-ab-snapshot-mismatch",
                lambda *args: {"errors": ["forced snapshot mismatch"], "ok": False},
            )
            roots.append(root)
            calls_before = self.fake.calls
            with TestClient(app) as client:
                turn = frozen.script("S1").turns[0]
                state.arm(session_id=snapshot_session, script_id="S1", turn_id="T1")
                response = client.post(
                    f"/api/sessions/{snapshot_session}/messages",
                    json={"message": turn.content},
                )
                self.assertEqual(response.status_code, 502, response.text)
                self.assertEqual(self.fake.calls, calls_before + 1)
                self.assertEqual(state.state_snapshot()["status"], "failed")
                self.assertIsNone(state.completed_turn("S1", "T1"))
                with self.assertRaises(GateError):
                    state.arm(session_id=snapshot_session, script_id="S1", turn_id="Q1")
            with SessionLocal() as db:
                count = db.scalar(
                    select(func.count()).select_from(Message).where(
                        Message.session_id == snapshot_session
                    )
                )
            self.assertEqual(count, 2)
            recorder.close()

            root, recorder, state, app = new_harness(
                "option-ab-evidence-failure",
                lambda *args: {"errors": [], "ok": True},
            )
            roots.append(root)
            calls_before = self.fake.calls
            with mock.patch.object(
                recorder,
                "record_snapshot",
                side_effect=RecordingError("forced evidence failure"),
            ):
                with TestClient(app) as client:
                    turn = frozen.script("S1").turns[0]
                    state.arm(session_id=evidence_session, script_id="S1", turn_id="T1")
                    response = client.post(
                        f"/api/sessions/{evidence_session}/messages",
                        json={"message": turn.content},
                    )
                    self.assertEqual(response.status_code, 500, response.text)
                    self.assertEqual(self.fake.calls, calls_before + 1)
                    self.assertEqual(state.state_snapshot()["status"], "failed")
                    self.assertIsNone(state.completed_turn("S1", "T1"))
                    with self.assertRaises(GateError):
                        state.arm(
                            session_id=evidence_session,
                            script_id="S1",
                            turn_id="Q1",
                        )
            with SessionLocal() as db:
                count = db.scalar(
                    select(func.count()).select_from(Message).where(
                        Message.session_id == evidence_session
                    )
                )
            self.assertEqual(count, 2)
            recorder.close()
        finally:
            for root in roots:
                shutil.rmtree(root, ignore_errors=True)

    def test_09_asgi_timeout_stops_batch(self) -> None:
        from sqlalchemy import func, select
        from backend.chat_service import ChatService, FrozenReceiveTeachingPolicy
        from backend.database import SessionLocal
        from backend.llm_profiles import LLMProfile, LLMProfileRegistry
        from backend.models import Message
        from experiments.option_ab_web.launcher import _session_mode_getter
        import backend.main as backend_main

        frozen = load_frozen_spec()
        root = Path(tempfile.mkdtemp(prefix="option-ab-timeout-"))
        evidence = root / "evidence"
        evidence.mkdir()
        recorder = RunRecorder(run_dir=root / "run", evidence_dir=evidence)
        state = BatchState(instance_id="timeout-asgi", spec=frozen)

        class SlowClient:
            calls = 0

            async def generate(self, messages, **kwargs):
                self.calls += 1
                await asyncio.sleep(0.2)
                return "late"

        slow = SlowClient()
        recording_client = RecordingLLMClient(
            slow,
            recorder=recorder,
            state=state,
            system_prompt=frozen.system_prompt,
            system_sha256=frozen.system_sha256,
            prompt_version=frozen.prompt_version,
            profile_id="default",
            request_model="fake",
            client_kind="fake",
            synthetic=True,
            timeout_seconds=0.01,
        )
        profile = LLMProfile(
            id="default",
            label="Option A+B timeout test",
            kind="fake",
            model="fake",
            client=recording_client,
            is_default=True,
        )
        service = ChatService(
            profiles=LLMProfileRegistry([profile]),
            lock_registry=backend_main._session_locks,
            frozen_receive_teaching_policy=FrozenReceiveTeachingPolicy(
                system_prompt=frozen.system_prompt,
                prompt_version=frozen.prompt_version,
            ),
        )
        old_service = backend_main.chat_service
        backend_main.chat_service = service
        try:
            with TestClient(self.instance.app.app) as raw:
                sid = raw.post("/api/sessions", json={}).json()["id"]
            app = TestApp(
                self.instance.app.app,
                test_html="<html></html>",
                state=state,
                recorder=recorder,
                snapshot_verifier=lambda *args: {"errors": [], "ok": True},
                session_mode_getter=_session_mode_getter(),
            )
            with TestClient(app) as client:
                turn = frozen.script("S1").turns[0]
                state.arm(session_id=sid, script_id="S1", turn_id="T1")
                response = client.post(
                    f"/api/sessions/{sid}/messages",
                    json={"message": turn.content},
                )
                self.assertEqual(response.status_code, 502, response.text)
                self.assertEqual(slow.calls, 1)
                self.assertEqual(state.state_snapshot()["status"], "failed")
                self.assertIsNone(state.completed_turn("S1", "T1"))
                with self.assertRaises(GateError):
                    state.arm(session_id=sid, script_id="S1", turn_id="Q1")
            http = _jsonl(evidence / "http.jsonl")
            self.assertEqual(len(http), 1)
            self.assertEqual(http[0]["model_status"], "error")
            self.assertEqual(http[0]["outcome"], "failed")
            self.assertEqual(http[0]["model_calls"], 1)
            with SessionLocal() as db:
                count = db.scalar(
                    select(func.count()).select_from(Message).where(
                        Message.session_id == sid
                    )
                )
            self.assertEqual(count, 1)
        finally:
            backend_main.chat_service = old_service
            recorder.close()
            shutil.rmtree(root, ignore_errors=True)


class TestStopIdentity(unittest.TestCase):
    def _make_run(self) -> tuple[Path, str, str, int]:
        root = Path(tempfile.mkdtemp(prefix="option-ab-web-stop-"))
        evidence = root / "evidence"
        evidence.mkdir()
        host = "127.0.0.1"
        port = 8765
        instance_id = "stop-test"
        resolved = root.resolve()
        meta = {
            "instance_id": instance_id,
            "paths": {"run_dir": str(resolved)},
            "host": host,
            "port": port,
        }
        meta_path = evidence / "run_meta.json"
        meta_path.write_text(json.dumps(meta), encoding="utf-8")
        from experiments.option_ab_web.launcher import _normalized_launch_argv

        payload = {
            "pid": 424242,
            "pid_start": "start-identity",
            "instance_id": instance_id,
            "host": host,
            "port": port,
            "launch_argv": _normalized_launch_argv(resolved, host, port),
            "run_meta_sha256": hashlib.sha256(meta_path.read_bytes()).hexdigest(),
        }
        (root / "server.pid").write_text(json.dumps(payload), encoding="utf-8")
        self.addCleanup(shutil.rmtree, root, True)
        return root, instance_id, host, port

    def test_stop_uses_verified_identity_and_sigterm(self) -> None:
        from experiments.option_ab_web import launcher

        root, instance_id, host, port = self._make_run()
        stopped = False

        def fake_kill(pid: int, sig: int) -> None:
            nonlocal stopped
            if sig == 0:
                if stopped:
                    raise ProcessLookupError
                return
            if sig == 15:
                stopped = True

        with mock.patch.object(launcher.os, "kill", side_effect=fake_kill), mock.patch.object(
            launcher, "_process_start_time", return_value="start-identity"
        ), mock.patch.object(
            launcher, "_process_command", return_value=(
                f"{launcher.sys.executable} -m experiments.option_ab_web.launcher "
                f"--run-dir {root.resolve()} --port {port}"
            )
        ), mock.patch.object(launcher, "_port_owner_pids", return_value={424242}):
            launcher.stop_instance(root)
        self.assertTrue(stopped)

    def test_stop_rejects_identity_mismatch(self) -> None:
        from experiments.option_ab_web import launcher

        root, _, _, _ = self._make_run()
        with mock.patch.object(launcher, "_process_start_time", return_value="wrong"), mock.patch.object(
            launcher.os, "kill"
        ) as kill:
            with self.assertRaises(IsolationError):
                launcher.stop_instance(root)
        kill.assert_called_once_with(424242, 0)

if __name__ == "__main__":
    unittest.main()
