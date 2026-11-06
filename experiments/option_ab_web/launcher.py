"""In-process launcher and isolation helpers for Option A+B web tests.

This module is the only place that imports ``backend.main`` for the AB web
harness.  It must be imported in a clean process: no backend module may be
loaded before the test DATABASE_URL and synthetic/real API configuration are
installed.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Iterator

from experiments.option_ab_web.asgi import TestApp
from experiments.option_ab_web.recording import RecordingLLMClient, RunRecorder
from experiments.option_ab_web.state import (
    BatchState,
    BranchSpec,
    FrozenSpec,
    ScriptSpec,
    TurnSpec,
    branch_specs_from_script,
    content_sha256,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
ASSET_ROOT = Path(__file__).resolve().parents[1] / "option_ab"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8777
TEST_MODEL = "fake"
TEST_API_KEY = "synthetic-option-ab-key"
TEST_API_BASE = "http://127.0.0.1:9/v1"
PROMPT_VERSION_MAX = 50


class IsolationError(RuntimeError):
    """A safe, actionable isolation failure."""


@dataclass(frozen=True)
class RunPaths:
    run_dir: Path
    db_path: Path
    evidence_dir: Path
    screenshots_dir: Path
    meta_path: Path
    pid_path: Path

    @classmethod
    def create(cls, run_dir: Path) -> "RunPaths":
        resolved = run_dir.expanduser().resolve()
        repo = REPO_ROOT.resolve()
        if resolved == repo or repo in resolved.parents:
            raise IsolationError("run_dir must be outside the repository")
        try:
            resolved.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            raise IsolationError("run_dir must be new and exclusive") from None
        evidence = resolved / "evidence"
        screenshots = resolved / "screenshots"
        evidence.mkdir()
        screenshots.mkdir()
        return cls(
            run_dir=resolved,
            db_path=resolved / "test.sqlite3",
            evidence_dir=evidence,
            screenshots_dir=screenshots,
            meta_path=evidence / "run_meta.json",
            pid_path=resolved / "server.pid",
        )


@dataclass
class TestInstance:
    app: TestApp
    paths: RunPaths
    state: BatchState
    recorder: RunRecorder
    frozen: FrozenSpec
    instance_id: str
    _close_callbacks: list[Callable[[], None]]

    def close(self) -> None:
        for callback in reversed(self._close_callbacks):
            with contextlib.suppress(Exception):
                callback()
        self._close_callbacks.clear()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _assert_backend_unloaded() -> None:
    loaded = sorted(
        name
        for name in sys.modules
        if name == "backend" or name.startswith("backend.")
    )
    if loaded:
        raise IsolationError(
            "backend modules were loaded before the test environment: "
            + ", ".join(loaded[:8])
        )


@contextlib.contextmanager
def _no_dotenv_reads() -> Iterator[None]:
    import dotenv

    original = dotenv.load_dotenv

    def refuse(*args: Any, **kwargs: Any) -> bool:
        return False

    dotenv.load_dotenv = refuse  # type: ignore[assignment]
    try:
        yield
    finally:
        dotenv.load_dotenv = original  # type: ignore[assignment]


@contextlib.contextmanager
def _temporary_environment(values: dict[str, str]) -> Iterator[None]:
    previous = {key: os.environ.get(key) for key in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _test_environment(
    db_path: Path,
    *,
    model: str,
    api_key: str,
    base_url: str,
) -> dict[str, str]:
    return {
        "DATABASE_URL": f"sqlite:///{db_path}",
        "LLM_MODE": "real",
        "LLM_API_KEY": api_key,
        "LLM_API_BASE_URL": base_url,
        "LLM_MODEL": model,
        "LLM_REASONING_EFFORT": "",
        "LOCAL_LLM_ENABLED": "false",
    }


def _normalise_sqlite_path(value: str) -> Path:
    from sqlalchemy.engine import make_url

    parsed = make_url(value)
    if parsed.get_backend_name() != "sqlite":
        raise IsolationError("test DATABASE_URL must be SQLite")
    database = parsed.database
    if not database or database == ":memory:":
        raise IsolationError("test DATABASE_URL must be a file-backed SQLite path")
    return Path(database).expanduser().resolve()


def _assert_db_target(paths: RunPaths) -> None:
    from backend import database

    expected = paths.db_path.resolve()
    actual = _normalise_sqlite_path(database.database_url)
    if actual != expected:
        raise IsolationError("backend database URL is not the test database")
    engine_database = database.engine.url.database
    if engine_database is None:
        raise IsolationError("engine has no test database path")
    if Path(engine_database).expanduser().resolve() != expected:
        raise IsolationError("engine is not bound to the test database")
    bind = database.SessionLocal.kw.get("bind")
    if bind is not database.engine:
        raise IsolationError("SessionLocal is not bound to the test engine")
    from backend import config

    if config.LOCAL_LLM_ENABLED:
        raise IsolationError("LOCAL_LLM_ENABLED must be false for AB web tests")


def _load_scripts(asset_root: Path) -> dict[str, Any]:
    return json.loads(
        (asset_root / "scripts" / "scripts.json").read_text(encoding="utf-8")
    )


def _build_scripts(
    asset_root: Path, manifest: dict[str, Any]
) -> tuple[ScriptSpec, ...]:
    scripts_payload = _load_scripts(asset_root)
    items = scripts_payload.get("scripts")
    if not isinstance(items, list):
        raise IsolationError("scripts.json has no scripts list")
    result: list[ScriptSpec] = []
    for item in items:
        if not isinstance(item, dict):
            raise IsolationError("script must be an object")
        script_id = str(item.get("id"))
        mode = str(item.get("mode"))
        raw_turns = item.get("turns")
        if not isinstance(raw_turns, list) or not raw_turns:
            raise IsolationError(f"{script_id} has no turns")
        turns: list[TurnSpec] = []
        for raw_turn in raw_turns:
            if not isinstance(raw_turn, dict):
                raise IsolationError(f"{script_id} turn must be an object")
            turn_id = str(raw_turn.get("id"))
            content = raw_turn.get("content")
            if not isinstance(content, str) or not content:
                raise IsolationError(f"{script_id}/{turn_id} content missing")
            if content != content.strip():
                raise IsolationError(f"{script_id}/{turn_id} has edge whitespace")
            turns.append(
                TurnSpec(
                    script_id=script_id,
                    turn_id=turn_id,
                    content=content,
                    content_sha256=content_sha256(content),
                )
            )
        branches = list(branch_specs_from_script(item)) if item.get("branches") else []
        if script_id in {"I2", "I3"}:
            if not branches:
                raise IsolationError(f"{script_id} must have branches")
            branches.insert(
                0,
                BranchSpec(
                    script_id=script_id,
                    branch_id="none",
                    label="none — no actual request",
                    content=None,
                    content_sha256=None,
                    is_none=True,
                ),
            )
        if script_id in {"I1", "I4"}:
            branches = []
        if mode == "fixed":
            calls = int(item.get("calls_expected"))
            minimum = maximum = calls
        else:
            calls = item.get("calls_expected")
            if isinstance(calls, int) and not isinstance(calls, bool):
                minimum = maximum = calls
            elif isinstance(calls, dict):
                minimum = int(calls["minimum"])
                maximum = int(calls["maximum"])
            else:
                raise IsolationError(f"{script_id} interactive calls missing")
        result.append(
            ScriptSpec(
                script_id=script_id,
                mode=mode,
                turns=tuple(turns),
                branches=tuple(branches),
                min_calls=minimum,
                max_calls=maximum,
            )
        )
    ids = [script.script_id for script in result]
    expected_ids = ["S1", "S2", "S3", "S4", "S5", "S6", "I1", "I2", "I3", "I4"]
    if ids != expected_ids:
        raise IsolationError("scripts must be S1-S6 then I1-I4")
    calls = manifest.get("calls")
    if not isinstance(calls, dict):
        raise IsolationError("manifest has no calls budget")
    return tuple(result)


def load_frozen_spec(asset_root: Path = ASSET_ROOT) -> FrozenSpec:
    """Verify the A+B package before importing backend or creating a client."""

    from experiments.option_ab.freeze_manifest import load_and_verify_manifest

    root = asset_root.expanduser().resolve()
    manifest = load_and_verify_manifest(root)
    if manifest.get("condition") != "AB":
        raise IsolationError("frozen package is not condition AB")
    asset_version = str(manifest.get("asset_version"))
    if asset_version != "option-ab-isolated-v0.1":
        raise IsolationError("unexpected A+B asset version")
    system_path = root / "prompts" / "system.txt"
    system_bytes = system_path.read_bytes()
    system_text = system_bytes.decode("utf-8")
    system_hash = _sha256_bytes(system_bytes)
    prompts = manifest.get("prompts")
    if not isinstance(prompts, dict):
        raise IsolationError("manifest prompts missing")
    expected_system = prompts.get("system", {}).get("sha256")
    if system_hash != expected_system:
        raise IsolationError("system prompt hash is not frozen")
    prompt_version = manifest.get("prompt_version")
    if not isinstance(prompt_version, str) or not prompt_version:
        raise IsolationError("manifest prompt_version missing")
    if not prompt_version.startswith(asset_version + "+sys-"):
        raise IsolationError("prompt_version does not identify the system hash")
    if len(prompt_version) > PROMPT_VERSION_MAX:
        raise IsolationError("prompt_version exceeds the message snapshot limit")
    scripts = _build_scripts(root, manifest)
    calls = manifest["calls"]
    total_min = int(calls["total_min"])
    total_max = int(calls["total_max"])
    fixed_calls = int(calls["fixed_total"])
    return FrozenSpec(
        asset_root=root,
        asset_version=asset_version,
        manifest_sha256=_sha256_file(root / "manifest.json"),
        system_prompt=system_text,
        system_sha256=system_hash,
        prompt_version=prompt_version,
        scripts=scripts,
        total_min=total_min,
        total_max=total_max,
        fixed_calls=fixed_calls,
    )


def _build_test_html(index_text: str, frozen: FrozenSpec, instance_id: str) -> str:
    banner = f"""
<style>
#option-ab-test-banner {{
  position: fixed; top: 0; left: 0; right: 0; z-index: 2147483647;
  box-sizing: border-box; padding: 8px 14px; background: #321c64; color: #fff;
  font: 13px/1.35 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  border-bottom: 2px solid #f4c542; box-shadow: 0 2px 8px rgba(0,0,0,.25);
}}
#option-ab-test-banner .b-main {{ font-weight: 700; letter-spacing: .02em; }}
#option-ab-test-banner .b-meta {{ opacity: .9; font-size: 11px; overflow-wrap: anywhere; }}
#option-ab-branch-panel {{
  position: static; margin: 8px 14px 0; width: auto; max-height: 40vh; overflow: auto;
  box-sizing: border-box; padding: 12px; background: #fff; color: #21134a;
  border: 2px solid #321c64; border-radius: 12px;
  box-shadow: 0 4px 14px rgba(0,0,0,.18); font: 13px/1.4 -apple-system, sans-serif;
}}
#option-ab-branch-panel[hidden] {{ display: none; }}
#option-ab-branch-panel h3 {{ margin: 0 0 6px; font-size: 14px; }}
#option-ab-branch-panel button {{
  display: block; width: 100%; margin: 5px 0; padding: 7px 9px;
  text-align: left; border: 1px solid #7561a8; border-radius: 7px;
  background: #f7f4ff; color: #21134a; cursor: pointer;
}}
#option-ab-branch-panel button:hover {{ background: #ebe4ff; }}
#option-ab-branch-panel pre {{
  white-space: pre-wrap; overflow-wrap: anywhere; background: #f5f2ff;
  padding: 8px; border-radius: 7px; font-size: 11px;
}}
html body {{
  padding-top: var(--option-ab-banner-height, 62px) !important;
}}
html body .app-layout {{
  height: calc(100vh - var(--option-ab-banner-height, 62px) - var(--option-ab-panel-height, 0px)) !important;
  min-height: 0;
}}
</style>
<div id="option-ab-test-banner" data-instance-id="{instance_id}">
  <div class="b-main">OPTION A+B TEST INSTANCE · condition AB · non-production</div>
  <div class="b-meta">
    instance {instance_id} · {frozen.prompt_version} · system sha256 {frozen.system_sha256}
  </div>
  <div class="b-meta" id="option-ab-state-line">loading state…</div>
</div>
<div id="option-ab-branch-panel" hidden>
  <h3>I2/I3 human branch selection</h3>
  <div id="option-ab-branch-help">Select one branch, then send its T2 through the real Send button.</div>
  <div id="option-ab-branch-list"></div>
  <div id="option-ab-t2-wrap" hidden>
    <p>Exact frozen T2 to send (FINAL is its assistant reply):</p>
    <pre id="option-ab-t2"></pre>
  </div>
</div>
<script>
(function () {{
  const instanceId = {json.dumps(instance_id)};
  const banner = document.getElementById("option-ab-test-banner");
  const bannerState = document.getElementById("option-ab-state-line");
  const panel = document.getElementById("option-ab-branch-panel");
  const list = document.getElementById("option-ab-branch-list");
  const t2wrap = document.getElementById("option-ab-t2-wrap");
  const t2pre = document.getElementById("option-ab-t2");
  let lastKey = "";
  function syncLayout() {{
    const bannerHeight = banner ? banner.offsetHeight : 62;
    const panelHeight = panel.hidden ? 0 : panel.offsetHeight + 8;
    document.documentElement.style.setProperty(
      "--option-ab-banner-height", bannerHeight + "px"
    );
    document.documentElement.style.setProperty(
      "--option-ab-panel-height", panelHeight + "px"
    );
    document.body.classList.toggle("option-ab-panel-open", !panel.hidden);
  }}
  function clearBranchDisplay() {{
    list.textContent = "";
    t2pre.textContent = "";
    t2wrap.hidden = true;
    lastKey = "";
  }}
  async function refreshState() {{
    try {{
      const response = await fetch("/__option_ab__/state", {{cache: "no-store"}});
      const state = await response.json();
      bannerState.textContent = "state " + state.status + " · calls " + state.total_consumed +
        "/" + state.total_max + " · current " + (state.current_script || "—") +
        (state.paused_human_classification ? " · PAUSED: human classification required" : "");
      const script = state.scripts.find((item) => item.id === state.current_script);
      const batchRunning = state.status === "running";
      const t2Armed = Boolean(
        batchRunning &&
        script &&
        (script.phase === "branch_selected" || script.phase === "t2_in_flight") &&
        script.selected_t2_content
      );
      if (!batchRunning || !script || (!state.paused_human_classification && !t2Armed)) {{
        clearBranchDisplay();
        panel.hidden = true;
        syncLayout();
        return;
      }}
      panel.hidden = false;
      if (t2Armed) {{
        list.textContent = "";
        t2pre.textContent = script.selected_t2_content;
        t2wrap.hidden = false;
        lastKey = "";
        syncLayout();
        return;
      }}
      t2pre.textContent = "";
      t2wrap.hidden = true;
      const key = state.current_script + ":" + script.q1_assistant_message_id;
      if (key === lastKey) return;
      lastKey = key;
      list.textContent = "";
      for (const branch of script.branches) {{
        const button = document.createElement("button");
        button.type = "button";
        button.textContent = branch.label;
        button.onclick = async () => {{
          button.disabled = true;
          try {{
            const selected = await fetch("/__option_ab__/branch", {{
              method: "POST",
              headers: {{"Content-Type": "application/json"}},
              body: JSON.stringify({{
                instance_id: instanceId,
                session_id: script.session_id,
                script_id: script.id,
                branch_id: branch.id,
                q1_assistant_message_id: script.q1_assistant_message_id,
                selected_by: "human"
              }})
            }});
            const result = await selected.json();
            if (!selected.ok) {{
              button.disabled = false;
              return;
            }}
            if (result.t2_content) {{
              t2pre.textContent = result.t2_content;
              t2wrap.hidden = false;
            }} else {{
              t2wrap.hidden = true;
            }}
            await refreshState();
          }} catch (_) {{
            button.disabled = false;
          }}
        }};
        list.appendChild(button);
      }}
      syncLayout();
    }} catch (_) {{
      bannerState.textContent = "state unavailable";
    }}
  }}
  window.addEventListener("resize", syncLayout);
  syncLayout();
  refreshState();
  setInterval(refreshState, 700);
}})();
</script>
"""
    body_match = re.search(r"<body\b[^>]*>", index_text, flags=re.IGNORECASE)
    if body_match is not None:
        return (
            index_text[: body_match.end()]
            + banner
            + index_text[body_match.end() :]
        )
    return banner + index_text


def _build_snapshot_verifier(
    frozen: FrozenSpec,
    state: BatchState,
    *,
    profile_id: str,
    profile_kind: str,
    model: str,
) -> Callable[[int, list[int], str, str], dict[str, Any]]:
    def verify(
        session_id: int, message_ids: list[int], script_id: str, turn_id: str
    ) -> dict[str, Any]:
        from backend.database import SessionLocal
        from backend.models import ChatSession, Message

        result: dict[str, Any] = {
            "instance_id": state.instance_id,
            "session_id": session_id,
            "script_id": script_id,
            "turn_id": turn_id,
            "prompt_version": frozen.prompt_version,
            "system_sha256": frozen.system_sha256,
            "model": model,
            "messages": [],
            "errors": [],
        }
        errors: list[str] = result["errors"]
        try:
            turn = state.turn_spec(script_id, turn_id)
        except Exception:
            errors.append("unknown script/turn")
            return result
        if not isinstance(message_ids, list) or len(message_ids) != 2:
            errors.append("expected exactly user and assistant message ids")
            return result
        if any(
            not isinstance(item, int) or isinstance(item, bool) or item <= 0
            for item in message_ids
        ):
            errors.append("message ids must be positive integers")
            return result
        if message_ids[0] == message_ids[1]:
            errors.append("user and assistant message ids must differ")
            return result
        try:
            with SessionLocal() as db:
                session = db.get(ChatSession, session_id)
                if session is None:
                    errors.append("session not found")
                    return result
                messages = (
                    db.query(Message).filter(Message.id.in_(message_ids)).all()
                )
                by_id = {message.id: message for message in messages}
                for expected_id, role in zip(message_ids, ("user", "assistant")):
                    message = by_id.get(expected_id)
                    if message is None:
                        errors.append(f"message {expected_id} not found")
                        continue
                    if message.session_id != session_id:
                        errors.append(f"message {expected_id} belongs to another session")
                    if message.role != role:
                        errors.append(f"message {expected_id} is not {role}")
                    if message.interaction_mode_snapshot != "receive_teaching":
                        errors.append(
                            f"message {expected_id} interaction_mode_snapshot is not receive_teaching"
                        )
                    if message.prompt_version_snapshot != frozen.prompt_version:
                        errors.append(
                            f"message {expected_id} prompt_version_snapshot mismatch"
                        )
                    if message.llm_profile_id_snapshot != profile_id:
                        errors.append(f"message {expected_id} profile id mismatch")
                    if message.llm_profile_kind_snapshot != profile_kind:
                        errors.append(f"message {expected_id} profile kind mismatch")
                    if message.llm_model_snapshot != model:
                        errors.append(f"message {expected_id} model snapshot mismatch")
                    if not isinstance(message.content, str) or not message.content.strip():
                        errors.append(f"message {expected_id} content is empty")
                    if role == "user" and message.content != turn.content:
                        errors.append(f"message {expected_id} is not the frozen user input")
                    result["messages"].append(
                        {
                            "id": message.id,
                            "role": message.role,
                            "content": message.content,
                            "interaction_mode_snapshot": message.interaction_mode_snapshot,
                            "prompt_version_snapshot": message.prompt_version_snapshot,
                            "llm_profile_id_snapshot": message.llm_profile_id_snapshot,
                            "llm_profile_kind_snapshot": message.llm_profile_kind_snapshot,
                            "llm_model_snapshot": message.llm_model_snapshot,
                        }
                    )
                if session.interaction_mode != "receive_teaching":
                    errors.append("session mode is not receive_teaching")
        except Exception:
            errors.append("snapshot database check failed")
        return result

    return verify


def _session_mode_getter() -> Callable[[int], str | None]:
    from backend.database import SessionLocal
    from backend.models import ChatSession

    def get_mode(session_id: int) -> str | None:
        with SessionLocal() as db:
            session = db.get(ChatSession, session_id)
            return session.interaction_mode if session is not None else None

    return get_mode


def prepare_instance(
    *,
    run_dir: Path,
    asset_root: Path = ASSET_ROOT,
    client_factory: Callable[[], Any] | None = None,
    real_api: bool = False,
    api_key: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
) -> TestInstance:
    """Build the A+B test ASGI app without starting a network listener."""

    frozen = load_frozen_spec(asset_root)
    _assert_backend_unloaded()
    paths = RunPaths.create(run_dir)

    resolved_model = model or ("fake" if not real_api else "")
    if real_api:
        if not api_key or not base_url or not resolved_model:
            raise IsolationError("real_api requires explicit credentials and model")
        client_kind = "api"
        synthetic = False
        client_key = api_key
        client_base = base_url
    else:
        if client_factory is None:
            raise IsolationError("fake test instances require client_factory")
        client_kind = "fake"
        synthetic = True
        client_key = TEST_API_KEY
        client_base = TEST_API_BASE

    env = _test_environment(
        paths.db_path,
        model=resolved_model,
        api_key=client_key,
        base_url=client_base,
    )
    env_guard = _temporary_environment(env)
    dotenv_guard = _no_dotenv_reads()
    env_guard.__enter__()
    dotenv_guard.__enter__()
    recorder: RunRecorder | None = None
    close_callbacks: list[Callable[[], None]] = [
        lambda: dotenv_guard.__exit__(None, None, None),
        lambda: env_guard.__exit__(None, None, None),
    ]
    try:
        import backend.main as main
        from backend import config, database
        from backend.chat_service import ChatService, FrozenReceiveTeachingPolicy
        from backend.llm_client import OpenAICompatibleLLMClient
        from backend.llm_profiles import LLMProfile, LLMProfileRegistry

        _assert_db_target(paths)
        if config.LOCAL_LLM_ENABLED:
            raise IsolationError("LOCAL_LLM_ENABLED must be false")
        instance_id = "b-" + uuid.uuid4().hex[:12]
        recorder = RunRecorder(run_dir=paths.run_dir, evidence_dir=paths.evidence_dir)
        state = BatchState(instance_id=instance_id, spec=frozen)
        if real_api:
            inner = OpenAICompatibleLLMClient(
                api_key=client_key,
                base_url=client_base,
                model=resolved_model,
                timeout=30.0,
                reasoning_effort="",
            )
        else:
            assert client_factory is not None
            inner = client_factory()
        recording_client = RecordingLLMClient(
            inner,
            recorder=recorder,
            state=state,
            system_prompt=frozen.system_prompt,
            system_sha256=frozen.system_sha256,
            prompt_version=frozen.prompt_version,
            profile_id="default",
            request_model=resolved_model,
            client_kind=client_kind,
            synthetic=synthetic,
        )
        profile = LLMProfile(
            id="default",
            label=(
                "Option A+B test API"
                if real_api
                else "Option A+B test fake"
            ),
            kind=client_kind,
            model=resolved_model,
            client=recording_client,
            is_default=True,
        )
        registry = LLMProfileRegistry([profile])
        service = ChatService(
            profiles=registry,
            lock_registry=main._session_locks,
            frozen_receive_teaching_policy=FrozenReceiveTeachingPolicy(
                system_prompt=frozen.system_prompt,
                prompt_version=frozen.prompt_version,
            ),
        )
        original_prompt = None
        original_version = None
        try:
            from backend import chat_service as chat_service_module

            original_prompt = chat_service_module.get_system_prompt_for_mode
            original_version = chat_service_module.get_prompt_version_for_mode

            def get_prompt(mode: str) -> str:
                if mode == "receive_teaching":
                    return frozen.system_prompt
                return original_prompt(mode)

            def get_version(mode: str) -> str:
                if mode == "receive_teaching":
                    return frozen.prompt_version
                return original_version(mode)

            chat_service_module.get_system_prompt_for_mode = get_prompt
            chat_service_module.get_prompt_version_for_mode = get_version
        except Exception:
            if original_prompt is not None:
                from backend import chat_service as chat_service_module

                chat_service_module.get_system_prompt_for_mode = original_prompt
            if original_version is not None:
                from backend import chat_service as chat_service_module

                chat_service_module.get_prompt_version_for_mode = original_version
            raise
        close_callbacks.append(
            lambda: _restore_prompt_getters(original_prompt, original_version)
        )
        original_profiles = main._profiles
        original_service = main.chat_service
        main._profiles = registry
        main.chat_service = service
        close_callbacks.append(
            lambda: _restore_main_services(original_profiles, original_service)
        )
        index_text = (REPO_ROOT / "frontend" / "index.html").read_text(encoding="utf-8")
        test_html = _build_test_html(index_text, frozen, instance_id)
        verifier = _build_snapshot_verifier(
            frozen,
            state,
            profile_id="default",
            profile_kind=client_kind,
            model=resolved_model,
        )
        app = TestApp(
            main.app,
            test_html=test_html,
            state=state,
            recorder=recorder,
            snapshot_verifier=verifier,
            session_mode_getter=_session_mode_getter(),
        )
        meta = {
            "kind": "option_ab_web_run",
            "instance_id": instance_id,
            "condition": "AB",
            "asset_version": frozen.asset_version,
            "manifest_sha256": frozen.manifest_sha256,
            "system_sha256": frozen.system_sha256,
            "prompt_version": frozen.prompt_version,
            "model": resolved_model,
            "receive_policy": {
                "kind": "frozen-experiment-v1",
                "reply_language": "frozen-prompt-only",
                "history_tools": False,
            },
            "profile_id": "default",
            "profile_kind": client_kind,
            "synthetic": synthetic,
            "host": host,
            "port": port,
            "paths": {
                "run_dir": str(paths.run_dir),
                "db_path": str(paths.db_path),
                "evidence_dir": str(paths.evidence_dir),
            },
            "created_at": datetime.now(UTC).isoformat(),
            "notes": {
                "temperature": 0.0,
                "max_tokens": 800,
                "timeout_seconds": 30.0,
                "reasoning_effort": "not sent",
                "timeout_source": "wrapper deadline; real client constructor timeout=30",
            },
        }
        with paths.meta_path.open("x", encoding="utf-8") as handle:
            json.dump(meta, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        close_callbacks.append(recorder.close)
        return TestInstance(
            app=app,
            paths=paths,
            state=state,
            recorder=recorder,
            frozen=frozen,
            instance_id=instance_id,
            _close_callbacks=close_callbacks,
        )
    except Exception:
        if recorder is not None:
            with contextlib.suppress(Exception):
                recorder.close()
        for callback in reversed(close_callbacks):
            with contextlib.suppress(Exception):
                callback()
        raise


def _restore_main_services(original_profiles: Any, original_service: Any) -> None:
    import backend.main as main

    main._profiles = original_profiles
    main.chat_service = original_service


def _restore_prompt_getters(original_prompt: Any, original_version: Any) -> None:
    if original_prompt is None or original_version is None:
        return
    from backend import chat_service as chat_service_module

    chat_service_module.get_system_prompt_for_mode = original_prompt
    chat_service_module.get_prompt_version_for_mode = original_version


def _process_start_time(pid: int) -> str:
    try:
        output = subprocess.check_output(
            ["ps", "-p", str(pid), "-o", "lstart="],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise IsolationError("could not read process start time") from exc
    if not output:
        raise IsolationError("process start time is empty")
    return " ".join(output.split())


def _process_command(pid: int) -> str:
    try:
        output = subprocess.check_output(
            ["ps", "-p", str(pid), "-o", "command="],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise IsolationError("could not read process command") from exc
    if not output:
        raise IsolationError("process command is empty")
    return " ".join(output.split())


def _port_owner_pids(port: int) -> set[int]:
    lsof = shutil.which("lsof")
    if not lsof:
        raise IsolationError("lsof is required to verify port ownership")
    try:
        output = subprocess.check_output(
            [lsof, "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-t"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except subprocess.CalledProcessError:
        return set()
    except OSError as exc:
        raise IsolationError("could not run lsof") from exc
    return {
        int(line.strip())
        for line in output.splitlines()
        if line.strip().isdigit()
    }


def _normalized_launch_argv(run_dir: Path, host: str, port: int) -> list[str]:
    return [
        sys.executable,
        "-m",
        "experiments.option_ab_web.launcher",
        "--real-api",
        "--run-dir",
        str(run_dir),
        "--host",
        host,
        "--port",
        str(port),
    ]


def stop_instance(run_dir: Path) -> None:
    """Stop only the launcher process identified by this run directory."""

    resolved = run_dir.expanduser().resolve()
    pid_path = resolved / "server.pid"
    meta_path = resolved / "evidence" / "run_meta.json"
    if not pid_path.is_file() or not meta_path.is_file():
        raise IsolationError("run has no server.pid/run_meta.json")
    try:
        payload = json.loads(pid_path.read_text(encoding="utf-8"))
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise IsolationError("invalid server metadata") from exc
    if not isinstance(payload, dict) or not isinstance(meta, dict):
        raise IsolationError("invalid server metadata")
    pid = payload.get("pid")
    instance_id = payload.get("instance_id")
    port = payload.get("port")
    host = payload.get("host")
    pid_start = payload.get("pid_start")
    launch_argv = payload.get("launch_argv")
    run_meta_sha256 = payload.get("run_meta_sha256")
    if (
        not isinstance(pid, int)
        or isinstance(pid, bool)
        or not isinstance(instance_id, str)
        or not isinstance(port, int)
        or isinstance(port, bool)
        or not 1 <= port <= 65535
        or not isinstance(host, str)
        or not isinstance(pid_start, str)
        or not pid_start
        or not isinstance(launch_argv, list)
        or not all(isinstance(item, str) for item in launch_argv)
        or not isinstance(run_meta_sha256, str)
        or len(run_meta_sha256) != 64
    ):
        raise IsolationError("invalid server.pid")
    if pid <= 1:
        raise IsolationError("refusing unsafe pid")
    if meta.get("instance_id") != instance_id:
        raise IsolationError("run metadata does not match server.pid")
    paths_meta = meta.get("paths")
    if not isinstance(paths_meta, dict):
        raise IsolationError("run metadata has no paths")
    try:
        meta_run_dir = Path(str(paths_meta.get("run_dir", ""))).resolve()
    except (OSError, ValueError):
        raise IsolationError("run metadata run_dir is invalid") from None
    if meta_run_dir != resolved:
        raise IsolationError("run metadata belongs to another run dir")
    if meta.get("host") != host or meta.get("port") != port:
        raise IsolationError("run metadata host/port does not match server.pid")
    if run_meta_sha256 != _sha256_file(meta_path):
        raise IsolationError("run metadata hash does not match server.pid")
    if launch_argv != _normalized_launch_argv(resolved, host, port):
        raise IsolationError("recorded launch arguments do not match this run")
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        raise IsolationError("process is not running") from None
    if _process_start_time(pid) != pid_start:
        raise IsolationError("process start identity does not match server.pid")
    command = _process_command(pid)
    if "experiments.option_ab_web.launcher" not in command:
        raise IsolationError("process command does not match this launcher")
    if str(resolved) not in command:
        raise IsolationError("process command does not contain this run dir")
    if f"--port {port}" not in command and f"--port={port}" not in command:
        raise IsolationError("process command does not contain the test port")
    if pid not in _port_owner_pids(port):
        raise IsolationError("process does not own the test listening port")
    os.kill(pid, signal.SIGTERM)
    for _ in range(50):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.1)
    raise IsolationError("launcher process did not stop after SIGTERM")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Option A+B isolated web test instance")
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", default=DEFAULT_PORT, type=int)
    parser.add_argument(
        "--real-api",
        action="store_true",
        help="use credentials from LLM_API_KEY/LLM_API_BASE_URL/LLM_MODEL",
    )
    args = parser.parse_args(argv)
    if not args.real_api:
        parser.error("--real-api is required for this launcher")
    api_key = os.environ.get("LLM_API_KEY", "").strip()
    base_url = os.environ.get("LLM_API_BASE_URL", "").strip()
    model = os.environ.get("LLM_MODEL", "").strip()
    if not api_key or not base_url or not model:
        parser.error("LLM_API_KEY, LLM_API_BASE_URL and LLM_MODEL are required")
    instance = prepare_instance(
        run_dir=args.run_dir,
        real_api=True,
        api_key=api_key,
        base_url=base_url,
        model=model,
        host=args.host,
        port=args.port,
    )
    pid_payload = {
        "pid": os.getpid(),
        "pid_start": _process_start_time(os.getpid()),
        "instance_id": instance.instance_id,
        "host": args.host,
        "port": args.port,
        "run_dir": str(instance.paths.run_dir),
        "launch_argv": _normalized_launch_argv(
            instance.paths.run_dir, args.host, args.port
        ),
        "run_meta_sha256": _sha256_file(instance.paths.meta_path),
    }
    with instance.paths.pid_path.open("x", encoding="utf-8") as handle:
        json.dump(pid_payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    try:
        import uvicorn  # noqa: PLC0415

        uvicorn.run(
            instance.app,
            host=args.host,
            port=args.port,
            workers=1,
            reload=False,
            log_level="info",
        )
    finally:
        instance.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
