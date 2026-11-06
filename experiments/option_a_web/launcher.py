"""Launch an isolated, test-only Option A web instance.

This module deliberately imports no backend module at import time.  The
launcher must set the test ``DATABASE_URL`` (and other test environment
variables) before importing ``backend.main``; it then verifies the imported
engine, replaces only the Receive prompt/version lookup in this process, and
wraps the real app with the test page and arm gate.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import html
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator

from experiments.option_a.freeze_manifest import load_and_verify_manifest
from experiments.option_a_web.asgi import TestApp
from experiments.option_a_web.gate import RunState, TurnGate, TurnSpec
from experiments.option_a_web.recording import RecordingLLMClient, RunRecorder

REPO_ROOT = Path(__file__).resolve().parents[2]
FROZEN_ASSET_ROOT = REPO_ROOT / "experiments" / "option_a"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
A_TIMEOUT_SECONDS = 30.0
A_TEMPERATURE = 0.0
A_MAX_TOKENS = 800
TEST_API_KEY = "synthetic-test-key-not-a-secret"
TEST_API_BASE_URL = "http://127.0.0.1:9/v1"
TEST_MODEL = "synthetic-a-model"
ENV_KEYS = (
    "DATABASE_URL",
    "LLM_MODE",
    "LLM_API_KEY",
    "LLM_API_BASE_URL",
    "LLM_MODEL",
    "LLM_PROFILE_LABEL",
    "LLM_REASONING_EFFORT",
    "LOCAL_LLM_ENABLED",
)


class IsolationError(RuntimeError):
    """The requested test instance is not safely isolated."""


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class FrozenSpec:
    asset_root: Path
    asset_version: str
    manifest_sha256: str
    system_prompt: str
    system_sha256: str
    prompt_version: str
    scripts: dict[str, tuple[TurnSpec, ...]]
    total_expected: int

    @property
    def turns(self) -> dict[tuple[str, str], TurnSpec]:
        return {
            (script_id, turn.turn_id): turn
            for script_id, turns in self.scripts.items()
            for turn in turns
        }


@dataclass(frozen=True)
class RunPaths:
    run_dir: Path
    evidence_dir: Path
    screenshots_dir: Path
    db_path: Path
    pid_path: Path
    meta_path: Path

    @classmethod
    def create(cls, run_dir: Path) -> "RunPaths":
        resolved = run_dir.expanduser().resolve()
        if _is_relative_to(resolved, REPO_ROOT):
            raise IsolationError("run_dir must be outside the repository")
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.mkdir(exist_ok=False)
        evidence = resolved / "evidence"
        screenshots = evidence / "screenshots"
        screenshots.mkdir(parents=True, exist_ok=True)
        return cls(
            run_dir=resolved,
            evidence_dir=evidence,
            screenshots_dir=screenshots,
            db_path=resolved / "test.sqlite3",
            pid_path=resolved / "server.pid",
            meta_path=evidence / "run_meta.json",
        )


@dataclass
class TestInstance:
    app: TestApp
    frozen: FrozenSpec
    paths: RunPaths
    gate: TurnGate
    recorder: RunRecorder
    instance_id: str
    model: str
    source_index_sha256: str
    test_html_sha256: str
    _close: Callable[[], None] = field(repr=False)

    def close(self) -> None:
        self._close()


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def load_frozen_spec(asset_root: Path = FROZEN_ASSET_ROOT) -> FrozenSpec:
    """Verify the frozen package before any backend import or client build."""

    asset_root = asset_root.resolve()
    manifest = load_and_verify_manifest(asset_root)
    if manifest.get("condition") != "A":
        raise IsolationError("frozen manifest condition is not A")
    if manifest.get("asset_version") != "option-a-isolated-v0.1":
        raise IsolationError("unexpected frozen asset_version")

    system_path = asset_root / "prompts" / "system.txt"
    system_prompt = system_path.read_text(encoding="utf-8")
    system_sha = hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()
    expected_system_sha = manifest["prompts"]["system"]["sha256"]
    if system_sha != expected_system_sha:
        raise IsolationError("system.txt does not match manifest hash")
    if len(system_prompt.encode("utf-8")) != manifest["prompts"]["system"]["bytes"]:
        raise IsolationError("system.txt byte length does not match manifest")

    scripts_payload = json.loads(
        (asset_root / "scripts" / "scripts.json").read_text(encoding="utf-8")
    )
    if scripts_payload.get("condition") != "A":
        raise IsolationError("scripts condition is not A")
    if scripts_payload.get("asset_version") != manifest["asset_version"]:
        raise IsolationError("scripts asset_version mismatch")
    per_script = manifest["calls"]["per_script"]
    total_expected = int(manifest["calls"]["total_expected"])
    scripts: dict[str, tuple[TurnSpec, ...]] = {}
    total = 0
    for script in scripts_payload.get("scripts", []):
        script_id = script.get("id")
        if not isinstance(script_id, str) or not re.fullmatch(r"S[1-6]", script_id):
            raise IsolationError("unexpected script id")
        if script_id in scripts:
            raise IsolationError(f"duplicate script id: {script_id}")
        turns_raw = script.get("turns")
        if not isinstance(turns_raw, list):
            raise IsolationError(f"script {script_id} has no turns list")
        expected = per_script.get(script_id)
        if expected is None or int(script.get("calls_expected", -1)) != int(expected):
            raise IsolationError(f"script {script_id} call count mismatch")
        turns: list[TurnSpec] = []
        seen_turns: set[str] = set()
        for raw_turn in turns_raw:
            turn_id = raw_turn.get("id")
            content = raw_turn.get("content")
            if not isinstance(turn_id, str) or not isinstance(content, str):
                raise IsolationError(f"script {script_id} has invalid turn")
            if turn_id in seen_turns:
                raise IsolationError(f"script {script_id} has duplicate turn")
            if content != content.strip():
                # ChatRequest strips the message, so a leading/trailing
                # whitespace copy could never match the frozen hash exactly.
                raise IsolationError(
                    f"script {script_id} turn {turn_id} has edge whitespace"
                )
            seen_turns.add(turn_id)
            turns.append(
                TurnSpec(
                    script_id=script_id,
                    turn_id=turn_id,
                    content=content,
                    content_sha256=sha256_text(content),
                )
            )
        if len(turns) != int(expected):
            raise IsolationError(f"script {script_id} turn count mismatch")
        scripts[script_id] = tuple(turns)
        total += len(turns)
    if total != total_expected:
        raise IsolationError(
            f"frozen total {total} does not equal manifest {total_expected}"
        )

    version = f"{manifest['asset_version']}+sys-{system_sha[:12]}"
    if len(version) > 50:
        raise IsolationError("A prompt version exceeds Message.prompt_version_snapshot")
    return FrozenSpec(
        asset_root=asset_root,
        asset_version=manifest["asset_version"],
        manifest_sha256=sha256_text(
            (asset_root / "manifest.json").read_text(encoding="utf-8")
        ),
        system_prompt=system_prompt,
        system_sha256=system_sha,
        prompt_version=version,
        scripts=scripts,
        total_expected=total_expected,
    )


def _assert_backend_unloaded() -> None:
    loaded = sorted(
        name
        for name in sys.modules
        if name == "backend" or name.startswith("backend.")
    )
    if loaded:
        raise IsolationError(
            "backend modules were already imported before test setup: "
            + ", ".join(loaded[:8])
        )


@contextlib.contextmanager
def _no_dotenv_reads() -> Iterator[None]:
    import dotenv

    original = dotenv.load_dotenv
    dotenv.load_dotenv = lambda *args, **kwargs: False
    try:
        yield
    finally:
        dotenv.load_dotenv = original


@contextlib.contextmanager
def _temporary_environment(values: dict[str, str]) -> Iterator[None]:
    previous = {key: os.environ.get(key) for key in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for key, old in previous.items():
            if old is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old


def _test_environment(db_path: Path, *, model: str, api_key: str, base_url: str) -> dict[str, str]:
    return {
        "DATABASE_URL": f"sqlite:///{db_path}",
        "LLM_MODE": "real",
        "LLM_API_KEY": api_key,
        "LLM_API_BASE_URL": base_url,
        "LLM_MODEL": model,
        "LLM_PROFILE_LABEL": "Option A Test API",
        "LLM_REASONING_EFFORT": "",
        "LOCAL_LLM_ENABLED": "false",
    }


def _assert_db_target(paths: RunPaths) -> dict[str, str]:
    from backend import config, database

    expected = paths.db_path.resolve()
    if database.engine.url.get_backend_name() != "sqlite":
        raise IsolationError("test engine is not SQLite")
    database_name = database.engine.url.database
    if not database_name:
        raise IsolationError("test engine has no database path")
    actual = Path(database_name).expanduser().resolve()
    if actual != expected:
        raise IsolationError(
            f"database engine is {actual}, expected test path {expected}"
        )
    bind = database.SessionLocal.kw.get("bind")
    if bind is not database.engine:
        raise IsolationError("SessionLocal is not bound to the test engine")
    if config.LOCAL_LLM_ENABLED:
        raise IsolationError("LOCAL_LLM_ENABLED must be false in the test instance")
    if paths.db_path == REPO_ROOT / "data" / "chatbot.db":
        raise IsolationError("refusing the production database path")
    return {
        "database_url": database.database_url,
        "database_path": str(actual),
        "local_llm_enabled": str(config.LOCAL_LLM_ENABLED),
    }


def _build_test_html(
    *,
    source_html: str,
    frozen: FrozenSpec,
    model: str,
    prompt_version: str,
    instance_id: str,
) -> str:
    if "</head>" not in source_html or "<body" not in source_html:
        raise IsolationError("frontend/index.html has no expected head/body")
    banner_text = (
        "A Prompt 测试变体 · 非生产测试实例 · "
        f"condition=A · {html.escape(prompt_version)} · "
        f"system sha256 {frozen.system_sha256} · "
        f"请求模型 {html.escape(model)}"
    )
    style = """
<style id="option-a-test-style">
#option-a-test-banner {
  position: sticky; top: 0; z-index: 2147483647;
  box-sizing: border-box; width: 100%;
  padding: 8px 14px;
  font: 600 13px/1.4 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  background: #102a43; color: #f0f7ff;
  border-bottom: 2px solid #2cb1bc;
  letter-spacing: .01em;
}
#option-a-test-banner code { color: #d9faff; font-weight: 400; }
</style>
"""
    banner = (
        f'<div id="option-a-test-banner" role="status" data-condition="A" '
        f'data-test-instance="1" data-instance-id="{html.escape(instance_id)}" '
        f'data-prompt-version="{html.escape(prompt_version)}" '
        f'data-system-sha256="{html.escape(frozen.system_sha256)}">'
        f"{banner_text}</div>"
    )
    document = source_html.replace("</head>", style + "</head>", 1)
    # Insert immediately after the opening body tag without replacing any app
    # content.  The banner is visually separate and never wraps the chat.
    body_match = re.search(r"<body[^>]*>", document, flags=re.IGNORECASE)
    if body_match is None:
        raise IsolationError("frontend/index.html has no body tag")
    insert_at = body_match.end()
    document = document[:insert_at] + banner + document[insert_at:]
    document = document.replace(
        "<title>Static Chatbot</title>",
        f"<title>Option A Test — {html.escape(prompt_version)}</title>",
        1,
    )
    document = document.replace(
        "</head>",
        '<meta name="option-a-test" content="1">\n</head>',
        1,
    )
    return document


def _verify_snapshot_factory(frozen: FrozenSpec, model: str) -> Callable:
    def verify(
        session_id: int, message_ids: list[int], script_id: str, turn_id: str
    ) -> dict[str, Any]:
        from backend.database import SessionLocal
        from backend.models import ChatSession, Message

        errors: list[str] = []
        checked: list[dict[str, Any]] = []
        turn = frozen.turns.get((script_id, turn_id))
        if turn is None:
            return {
                "session_id": session_id,
                "script_id": script_id,
                "turn_id": turn_id,
                "prompt_version": frozen.prompt_version,
                "system_sha256": frozen.system_sha256,
                "model": model,
                "errors": ["unknown script/turn"],
                "messages": [],
            }
        if not isinstance(message_ids, list) or len(message_ids) != 2:
            return {
                "session_id": session_id,
                "script_id": script_id,
                "turn_id": turn_id,
                "prompt_version": frozen.prompt_version,
                "system_sha256": frozen.system_sha256,
                "model": model,
                "errors": ["expected exactly user and assistant message ids"],
                "messages": [],
            }
        if any(
            not isinstance(item, int) or isinstance(item, bool) or item <= 0
            for item in message_ids
        ):
            errors.append("message ids must be positive integers")
        if message_ids[0] == message_ids[1]:
            errors.append("user and assistant message ids must differ")

        try:
            with SessionLocal() as db:
                session = db.get(ChatSession, session_id)
                if session is None:
                    return {
                        "session_id": session_id,
                        "script_id": script_id,
                        "turn_id": turn_id,
                        "prompt_version": frozen.prompt_version,
                        "system_sha256": frozen.system_sha256,
                        "model": model,
                        "errors": ["session not found"],
                        "messages": [],
                    }

                # Use the exact ids produced by this request.  Never ask the
                # database for a session's latest message.
                messages = (
                    db.query(Message)
                    .filter(Message.id.in_(message_ids))
                    .all()
                )
                by_id = {message.id: message for message in messages}
                for message_id in message_ids:
                    message = by_id.get(message_id)
                    if message is None:
                        errors.append(f"message {message_id} not found")
                        continue
                    if message.session_id != session_id:
                        errors.append(
                            f"message {message_id} belongs to another session"
                        )
                    if not isinstance(message.content, str):
                        errors.append(f"message {message_id} content is not text")
                    if not isinstance(message.role, str):
                        errors.append(f"message {message_id} role is not text")
                    if message.interaction_mode_snapshot != "receive_teaching":
                        errors.append(
                            f"message {message_id} interaction_mode_snapshot is not receive_teaching"
                        )
                    if message.prompt_version_snapshot != frozen.prompt_version:
                        errors.append(
                            f"message {message_id} prompt_version_snapshot is not the frozen A version"
                        )
                    if message.llm_profile_id_snapshot != "default":
                        errors.append(
                            f"message {message_id} llm_profile_id_snapshot is not default"
                        )
                    if message.llm_profile_kind_snapshot != "api":
                        errors.append(
                            f"message {message_id} llm_profile_kind_snapshot is not api"
                        )
                    if message.llm_model_snapshot != model:
                        errors.append(
                            f"message {message_id} llm_model_snapshot is not the test request model"
                        )
                    checked.append(
                        {
                            "id": message.id,
                            "role": message.role,
                            "content_sha256": sha256_text(message.content)
                            if isinstance(message.content, str)
                            else None,
                            "interaction_mode_snapshot": message.interaction_mode_snapshot,
                            "prompt_version_snapshot": message.prompt_version_snapshot,
                            "llm_profile_id_snapshot": message.llm_profile_id_snapshot,
                            "llm_profile_kind_snapshot": message.llm_profile_kind_snapshot,
                            "llm_model_snapshot": message.llm_model_snapshot,
                        }
                    )

                if len(messages) != 2:
                    errors.append(
                        f"expected exactly 2 messages for this request, found {len(messages)}"
                    )
                user_message = by_id.get(message_ids[0])
                assistant_message = by_id.get(message_ids[1])
                if user_message is not None:
                    if user_message.role != "user":
                        errors.append("first message is not the user message")
                    if user_message.content != turn.content:
                        errors.append("user message content is not the frozen turn text")
                if assistant_message is not None:
                    if assistant_message.role != "assistant":
                        errors.append("second message is not the assistant message")
                    if not isinstance(assistant_message.content, str) or not assistant_message.content.strip():
                        errors.append("assistant message is empty")
        except Exception as exc:
            errors.append(f"snapshot query failed: {type(exc).__name__}")
        return {
            "session_id": session_id,
            "script_id": script_id,
            "turn_id": turn_id,
            "prompt_version": frozen.prompt_version,
            "system_sha256": frozen.system_sha256,
            "model": model,
            "errors": errors,
            "messages": checked,
        }

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
    asset_root: Path = FROZEN_ASSET_ROOT,
    client_factory: Callable[[], Any] | None = None,
    real_api: bool = False,
    api_key: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
) -> TestInstance:
    """Build the isolated app in-process.  Does not start a network listener."""

    frozen = load_frozen_spec(asset_root)
    paths = RunPaths.create(run_dir)
    _assert_backend_unloaded()

    resolved_model = model or (
        os.environ.get("LLM_MODEL", "").strip() if real_api else TEST_MODEL
    )
    if real_api:
        if not api_key or not base_url or not resolved_model:
            raise IsolationError(
                "real_api requires explicit api_key, base_url and model"
            )
        client_key = api_key
        client_base = base_url
    else:
        if client_factory is None:
            raise IsolationError(
                "non-real test instances must supply a fake client_factory"
            )
        client_key = TEST_API_KEY
        client_base = TEST_API_BASE_URL
        resolved_model = resolved_model or TEST_MODEL

    env = _test_environment(
        paths.db_path,
        model=resolved_model,
        api_key=client_key,
        base_url=client_base,
    )
    env_guard = _temporary_environment(env)
    env_guard.__enter__()
    dotenv_guard = _no_dotenv_reads()
    dotenv_guard.__enter__()
    try:
        # The outer guard is intentionally still active during import; this
        # prevents backend.config from reading a real .env before the
        # synthetic test environment has been applied.
        import backend.main as main  # noqa: PLC0415 — after env setup
        from backend import config, database  # noqa: F401
        from backend.chat_service import ChatService, FrozenReceiveTeachingPolicy
        from backend.llm_client import create_llm_client
        from backend.llm_profiles import LLMProfile, LLMProfileRegistry

        db_info = _assert_db_target(paths)
        if real_api:
            inner = create_llm_client(
                mode="real",
                api_key=client_key,
                base_url=client_base,
                model=resolved_model,
                reasoning_effort="",
            )
        else:
            assert client_factory is not None
            inner = client_factory()

        instance_id = uuid.uuid4().hex
        recorder = RunRecorder(
            run_dir=paths.run_dir,
            evidence_dir=paths.evidence_dir,
        )
        recording_client = RecordingLLMClient(
            inner,
            recorder=recorder,
            system_prompt=frozen.system_prompt,
            system_sha256=frozen.system_sha256,
            profile_id="default",
            request_model=resolved_model,
            synthetic=not real_api,
            timeout_seconds=A_TIMEOUT_SECONDS,
            temperature=A_TEMPERATURE,
            max_tokens=A_MAX_TOKENS,
        )
        profile = LLMProfile(
            id="default",
            label="Option A Test API",
            kind="api",
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
        main._profiles = registry
        main.chat_service = service

        import backend.chat_service as chat_service_module

        original_prompt = chat_service_module.get_system_prompt_for_mode
        original_version = chat_service_module.get_prompt_version_for_mode

        def prompt_for_mode(interaction_mode: str) -> str:
            if interaction_mode == "receive_teaching":
                return frozen.system_prompt
            return original_prompt(interaction_mode)

        def version_for_mode(interaction_mode: str) -> str:
            if interaction_mode == "receive_teaching":
                return frozen.prompt_version
            return original_version(interaction_mode)

        chat_service_module.get_system_prompt_for_mode = prompt_for_mode
        chat_service_module.get_prompt_version_for_mode = version_for_mode

        source_index_path = REPO_ROOT / "frontend" / "index.html"
        source_html = source_index_path.read_text(encoding="utf-8")
        source_sha = sha256_text(source_html)
        test_html = _build_test_html(
            source_html=source_html,
            frozen=frozen,
            model=resolved_model,
            prompt_version=frozen.prompt_version,
            instance_id=instance_id,
        )
        test_html_sha = sha256_text(test_html)

        verifier = _verify_snapshot_factory(frozen, resolved_model)
        gate = TurnGate(
            instance_id=instance_id,
            scripts=frozen.scripts,
            run_state=RunState(instance_id),
        )
        test_app = TestApp(
            main.app,
            test_html=test_html,
            gate=gate,
            recorder=recorder,
            instance_id=instance_id,
            snapshot_verifier=verifier,
            session_mode_getter=_session_mode_getter(),
        )

        meta = {
            "schema_version": "option-a-web-run-v1",
            "instance_id": instance_id,
            "asset_version": frozen.asset_version,
            "manifest_sha256": frozen.manifest_sha256,
            "condition": "A",
            "total_expected": frozen.total_expected,
            "prompt_version": frozen.prompt_version,
            "system_sha256": frozen.system_sha256,
            "system_bytes": len(frozen.system_prompt.encode("utf-8")),
            "receive_policy": {
                "kind": "frozen-experiment-v1",
                "reply_language": "frozen-prompt-only",
                "history_tools": False,
            },
            "model": {
                "request_model": resolved_model,
                "upstream_response_model": None,
                "upstream_response_model_reason": (
                    "LLMClient.generate() returns only content; the production "
                    "client does not expose the upstream response model."
                ),
            },
            "parameters": {
                "original_from_chat_service": {
                    "temperature": None,
                    "max_tokens": None,
                    "response_format": None,
                    "timeout_seconds": None,
                    "reasoning_effort": None,
                },
                "effective_sent_by_wrapper": {
                    "temperature": A_TEMPERATURE,
                    "max_tokens": A_MAX_TOKENS,
                    "timeout_seconds": A_TIMEOUT_SECONDS,
                    "reasoning_effort": None,
                    "reasoning_effort_sent": False,
                },
            },
            "database": db_info,
            "paths": {
                "run_dir": str(paths.run_dir),
                "evidence_dir": str(paths.evidence_dir),
                "screenshots_dir": str(paths.screenshots_dir),
                "db_path": str(paths.db_path),
            },
            "frontend": {
                "source_index_sha256": source_sha,
                "test_html_sha256": test_html_sha,
            },
            "host": host,
            "port": port,
            "started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        with paths.meta_path.open("x", encoding="utf-8") as handle:
            json.dump(meta, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")

        def close() -> None:
            try:
                chat_service_module.get_system_prompt_for_mode = original_prompt
                chat_service_module.get_prompt_version_for_mode = original_version
            finally:
                recorder.close()
                dotenv_guard.__exit__(None, None, None)
                env_guard.__exit__(None, None, None)

        return TestInstance(
            app=test_app,
            frozen=frozen,
            paths=paths,
            gate=gate,
            recorder=recorder,
            instance_id=instance_id,
            model=resolved_model,
            source_index_sha256=source_sha,
            test_html_sha256=test_html_sha,
            _close=close,
        )
    except Exception:
        with contextlib.suppress(Exception):
            dotenv_guard.__exit__(None, None, None)
        with contextlib.suppress(Exception):
            env_guard.__exit__(None, None, None)
        raise


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _process_start_time(pid: int) -> str:
    """Return a stable process start token from ``ps``."""

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
    """Return PIDs holding a listening TCP socket on *port* via lsof."""

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
    pids: set[int] = set()
    for line in output.splitlines():
        try:
            pids.add(int(line.strip()))
        except ValueError:
            continue
    return pids


def _normalized_launch_argv(run_dir: Path, host: str, port: int) -> list[str]:
    return [
        sys.executable,
        "-m",
        "experiments.option_a_web.launcher",
        "--real-api",
        "--run-dir",
        str(run_dir),
        "--host",
        host,
        "--port",
        str(port),
    ]


def stop_instance(run_dir: Path) -> None:
    """Stop only the launcher process identified by this run dir's metadata."""

    resolved = run_dir.expanduser().resolve()
    pid_path = resolved / "server.pid"
    meta_path = resolved / "evidence" / "run_meta.json"
    if not pid_path.is_file():
        raise IsolationError("no server.pid for this run")
    if not meta_path.is_file():
        raise IsolationError("no run_meta.json for this run")
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
    if "experiments.option_a_web.launcher" not in command:
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
    parser = argparse.ArgumentParser(description="Option A isolated web test instance")
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", default=DEFAULT_PORT, type=int)
    parser.add_argument(
        "--real-api",
        action="store_true",
        help="use real API credentials from LLM_API_KEY/LLM_API_BASE_URL/LLM_MODEL",
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
        import uvicorn  # noqa: PLC0415 — only needed for a real launch

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
