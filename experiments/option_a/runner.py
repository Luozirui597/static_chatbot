"""Isolated fake-client runner for the Option A experiment.

The default and only built-in client is deterministic and local. This module
does not import a production LLM client, read environment files, or perform
network I/O.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
import sys
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Awaitable, Callable, Protocol, Sequence

try:
    from .freeze_manifest import FrozenAssetError, load_and_verify_manifest
    from .history import (
        MAX_HISTORY_MESSAGES,
        StoredMessage,
        build_request_messages,
        project_receive,
        recent_history,
        serialize_receive_history,
    )
except ImportError:  # direct script execution: python experiments/option_a/runner.py
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from freeze_manifest import (  # type: ignore[no-redef]
        FrozenAssetError,
        load_and_verify_manifest,
    )
    from history import (  # type: ignore[no-redef]
        MAX_HISTORY_MESSAGES,
        StoredMessage,
        build_request_messages,
        project_receive,
        recent_history,
        serialize_receive_history,
    )

ASSET_VERSION = "option-a-isolated-v0.1"
RUN_SCHEMA_VERSION = "option-a-run-v1"
TURN_SCHEMA_VERSION = "option-a-turn-v1"
REQUIRED_SCRIPT_IDS = ("S1", "S2", "S3", "S4", "S5", "S6")
EXPECTED_CALLS = {"S1": 2, "S2": 2, "S3": 2, "S4": 2, "S5": 4, "S6": 4}


class EmptyResponseError(RuntimeError):
    """The fake or future real client returned an empty/blank response."""


class FakeLLMClient:
    """Deterministic local fake.

    If *replies* is supplied, responses are consumed in order. If it is not,
    every call returns a synthetic marker. The fake never opens a socket.
    """

    synthetic = True

    def __init__(self, replies: Sequence[str] | None = None) -> None:
        self._replies: list[str] | None = None if replies is None else list(replies)
        self.calls: list[list[dict[str, str]]] = []

    async def generate(
        self,
        messages: Sequence[dict[str, str]],
        **_: Any,
    ) -> str:
        self.calls.append([dict(message) for message in messages])
        if self._replies is not None:
            if not self._replies:
                raise RuntimeError("fake reply queue exhausted")
            return self._replies.pop(0)
        return f"SYNTHETIC_FAKE_REPLY_{len(self.calls)}"


class StaticReplyClient:
    """Fake that returns the same reply for every call; useful for tests."""

    synthetic = True

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.calls: list[list[dict[str, str]]] = []

    async def generate(
        self,
        messages: Sequence[dict[str, str]],
        **_: Any,
    ) -> str:
        self.calls.append([dict(message) for message in messages])
        return self.reply


class RaisingClient:
    """Fake that raises on a selected call number, for fail-fast tests."""

    synthetic = True

    def __init__(self, error: BaseException, fail_on_call: int = 1) -> None:
        self.error = error
        self.fail_on_call = fail_on_call
        self.calls: list[list[dict[str, str]]] = []

    async def generate(
        self,
        messages: Sequence[dict[str, str]],
        **_: Any,
    ) -> str:
        self.calls.append([dict(message) for message in messages])
        if len(self.calls) == self.fail_on_call:
            raise self.error
        return f"SYNTHETIC_FAKE_REPLY_{len(self.calls)}"


class LLMClient(Protocol):
    async def generate(
        self,
        messages: Sequence[dict[str, str]],
        **kwargs: Any,
    ) -> str: ...


@dataclass
class RunResult:
    run_id: str
    script_id: str
    status: str
    calls: int
    records: list[dict[str, Any]] = field(default_factory=list)
    error: dict[str, str] | None = None
    run_dir: str = ""


@dataclass(frozen=True)
class ScriptSpec:
    id: str
    title: str
    focus: str
    calls_expected: int
    turns: tuple[dict[str, str], ...]


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_json(value: Any) -> str:
    canonical = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return sha256_text(canonical)


def safe_run_id(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    if not cleaned or cleaned in {".", ".."} or ".." in cleaned.split("."):
        raise ValueError("run_id must contain a safe non-empty component")
    return cleaned


def load_system_prompt(asset_root: Path) -> str:
    path = asset_root / "prompts" / "system.txt"
    return path.read_text(encoding="utf-8")


def load_scripts(asset_root: Path) -> list[ScriptSpec]:
    path = asset_root / "scripts" / "scripts.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    scripts_raw = payload.get("scripts")
    if not isinstance(scripts_raw, list):
        raise ValueError("scripts.json must contain a scripts list")

    scripts: list[ScriptSpec] = []
    seen: set[str] = set()
    for item in scripts_raw:
        script_id = item.get("id")
        if script_id in seen:
            raise ValueError(f"duplicate script id: {script_id}")
        seen.add(script_id)
        turns_raw = item.get("turns")
        if not isinstance(turns_raw, list) or not turns_raw:
            raise ValueError(f"{script_id}: turns must be a non-empty list")
        turns: list[dict[str, str]] = []
        for turn in turns_raw:
            turn_id = turn.get("id")
            phase = turn.get("phase")
            content = turn.get("content")
            if not isinstance(turn_id, str) or not turn_id:
                raise ValueError(f"{script_id}: invalid turn id")
            if not isinstance(phase, str) or not phase:
                raise ValueError(f"{script_id}/{turn_id}: invalid phase")
            if not isinstance(content, str) or not content.strip():
                raise ValueError(f"{script_id}/{turn_id}: empty content")
            turns.append({"id": turn_id, "phase": phase, "content": content})
        if item.get("calls_expected") != len(turns):
            raise ValueError(
                f"{script_id}: calls_expected must equal number of turns"
            )
        scripts.append(
            ScriptSpec(
                id=script_id,
                title=item.get("title", ""),
                focus=item.get("focus", ""),
                calls_expected=item["calls_expected"],
                turns=tuple(turns),
            )
        )

    if tuple(script.id for script in scripts) != REQUIRED_SCRIPT_IDS:
        raise ValueError(
            "scripts.json must define exactly S1, S2, S3, S4, S5, S6 in order"
        )
    for script in scripts:
        if script.calls_expected != EXPECTED_CALLS[script.id]:
            raise ValueError(f"{script.id}: unexpected calls_expected")
    return scripts


class RunRecorder:
    """Exclusive run directory and append-only JSONL turn records."""

    def __init__(self, output_root: Path, run_id: str) -> None:
        self.run_id = safe_run_id(run_id)
        self.run_dir = output_root / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self.turn_path = self.run_dir / "turns.jsonl"
        self.meta_path = self.run_dir / "run_meta.json"
        self._handle = self.turn_path.open("x", encoding="utf-8", newline="\n")

    def append(self, record: dict[str, Any]) -> None:
        self._handle.write(
            json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n"
        )
        self._handle.flush()

    def write_meta(self, meta: dict[str, Any]) -> None:
        with self.meta_path.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(
                json.dumps(meta, ensure_ascii=False, sort_keys=True, indent=2)
                + "\n"
            )

    def close(self) -> None:
        self._handle.close()

    def __enter__(self) -> "RunRecorder":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


def _script_digest(script: ScriptSpec) -> str:
    return sha256_json(
        {
            "id": script.id,
            "title": script.title,
            "focus": script.focus,
            "calls_expected": script.calls_expected,
            "turns": list(script.turns),
        }
    )


def _client_context(client: LLMClient) -> dict[str, Any]:
    return {
        "kind": type(client).__name__,
        "synthetic": bool(getattr(client, "synthetic", False)),
    }


def _model_context() -> dict[str, Any]:
    return {
        "profile_id": None,
        "profile_kind": None,
        "model": None,
        "note": "null: the local fake path has no model/profile.",
    }


def _generation_context() -> dict[str, Any]:
    return {
        "requested": {
            "temperature": None,
            "max_tokens": None,
            "note": "null: the runner sends no generation parameters on the fake path.",
        },
        "response_metadata": {
            "model": None,
            "request_id": None,
            "usage": None,
            "finish_reason": None,
            "note": "null: the fake client returns only text; no provider metadata.",
        },
    }


def _source_context(run_id: str, script_id: str, turn_id: str) -> dict[str, Any]:
    def local_record(target_turn_id: str) -> str:
        return f"{run_id}:{script_id}:{target_turn_id}"

    source_turn_ids: list[str] = [] if turn_id == "T1" else ["T1"]
    related_record_ids: list[str] = []
    event: dict[str, Any] | None = None
    if script_id in {"S5", "S6"} and turn_id == "T2":
        source_turn_ids = ["T1", "Q1"]
        related_record_ids = [local_record("T1"), local_record("Q1")]
        event = {
            "type": "correction" if script_id == "S5" else "revocation",
            "target_turn_ids": ["T1"],
            "target_record_ids": [local_record("T1")],
            "prior_inference_record_id": local_record("Q1"),
        }
    elif script_id in {"S5", "S6"} and turn_id == "Q2":
        source_turn_ids = ["T2"]
        related_record_ids = [local_record("T2")]
        event = {
            "type": (
                "recheck_after_correction"
                if script_id == "S5"
                else "recheck_after_revocation"
            ),
            "target_turn_ids": ["T1"],
            "target_record_ids": [local_record("T1")],
            "prior_inference_record_id": local_record("Q1"),
        }
    return {
        "source_message_id": None,
        "source_turn_ids": source_turn_ids,
        "related_record_ids": related_record_ids,
        "event": event,
        "note": "Source links are run-local record ids; source_message_id is null.",
    }


def make_turn_record(
    *,
    run_id: str,
    script_id: str,
    turn_id: str,
    phase: str,
    turn_index: int,
    call_index: int,
    current_user: str,
    messages: Sequence[dict[str, str]],
    response: str | None,
    status: str,
    client_kind: str,
    synthetic: bool,
    error: dict[str, str] | None = None,
) -> dict[str, Any]:
    request_payload = {"messages": list(messages)}
    source = _source_context(run_id, script_id, turn_id)
    record_id = f"{run_id}:{script_id}:{turn_id}"
    return {
        "schema_version": TURN_SCHEMA_VERSION,
        "record_id": record_id,
        "run_id": run_id,
        "script_id": script_id,
        "turn_id": turn_id,
        "phase": phase,
        "condition": "A",
        "turn_index": turn_index,
        "call_index": call_index,
        "current_user": current_user,
        "retry": False,
        "client": {"kind": client_kind, "synthetic": synthetic},
        "model": _model_context(),
        "generation": _generation_context(),
        "source": source,
        "request": {
            "messages": list(messages),
            "sha256": sha256_json(request_payload),
        },
        "response": response,
        "response_sha256": sha256_text(response) if response is not None else None,
        "status": status,
        "error": error,
        "model_claims": {
            "reported_conclusion": None,
            "reported_premises": None,
            "reported_rule": None,
            "note": "Model self-report only; not extracted by the runner. Human review verifies it.",
        },
        "human_review": {
            "authorized": None,
            "logically_valid": None,
            "source_correct": None,
            "inference_conclusion": None,
            "inference_premises": None,
            "inference_rule": None,
            "question_reason": None,
            "question_gap": None,
            "teaching_fidelity": None,
            "correction_or_revocation_effective": None,
            "old_support_removed": None,
            "negative_fact_preserved": None,
            "verified_source_turn_ids": None,
            "verified_source_record_ids": None,
            "content_question": None,
            "natural_focus": None,
            "notes": None,
        },
    }


async def run_script(
    script: ScriptSpec,
    *,
    system_prompt: str,
    client: LLMClient,
    output_root: Path,
    run_id: str | None = None,
    system_sha256: str | None = None,
) -> RunResult:
    run_id = safe_run_id(
        run_id
        or f"{ASSET_VERSION}-{script.id}-{utc_now().replace(':', '').replace('+', 'Z')}-{uuid.uuid4().hex[:8]}"
    )
    actual_system_sha = sha256_text(system_prompt)
    if system_sha256 is not None and system_sha256 != actual_system_sha:
        raise FrozenAssetError(
            "system_sha256 does not match the actual system prompt; refusing to run"
        )
    client_context = _client_context(client)
    script_digest = _script_digest(script)
    recorder = RunRecorder(output_root, run_id)
    storage: list[StoredMessage] = []
    records: list[dict[str, Any]] = []
    calls = 0
    status = "completed"
    error_info: dict[str, str] | None = None
    started_at = utc_now()

    try:
        for turn_index, turn in enumerate(script.turns, start=1):
            messages = build_request_messages(
                system_prompt,
                storage,
                turn["content"],
            )
            storage.append(StoredMessage(role="user", content=turn["content"]))
            calls += 1
            response: str | None = None
            raw_response: str | None = None
            received = False
            try:
                raw = await client.generate(messages)
                if not isinstance(raw, str):
                    raise EmptyResponseError("client returned a non-string response")
                raw_response = raw
                received = True
                if not raw.strip():
                    raise EmptyResponseError("client returned an empty or blank response")
                response = raw
            except Exception as exc:  # record and stop; never retry
                error_info = {
                    "type": type(exc).__name__,
                    "message": str(exc),
                }
                record = make_turn_record(
                    run_id=run_id,
                    script_id=script.id,
                    turn_id=turn["id"],
                    phase=turn["phase"],
                    turn_index=turn_index,
                    call_index=calls,
                    current_user=turn["content"],
                    messages=messages,
                    response=raw_response if received else None,
                    status="error",
                    client_kind=client_context["kind"],
                    synthetic=client_context["synthetic"],
                    error=error_info,
                )
                records.append(record)
                recorder.append(record)
                status = "failed"
                break

            storage.append(StoredMessage(role="assistant", content=response))
            record = make_turn_record(
                run_id=run_id,
                script_id=script.id,
                turn_id=turn["id"],
                phase=turn["phase"],
                turn_index=turn_index,
                call_index=calls,
                current_user=turn["content"],
                messages=messages,
                response=response,
                status="ok",
                client_kind=client_context["kind"],
                synthetic=client_context["synthetic"],
            )
            records.append(record)
            recorder.append(record)
    finally:
        recorder.close()

    meta = {
        "schema_version": RUN_SCHEMA_VERSION,
        "asset_version": ASSET_VERSION,
        "condition": "A",
        "run_id": run_id,
        "script_id": script.id,
        "script_title": script.title,
        "script_focus": script.focus,
        "started_at_utc": started_at,
        "finished_at_utc": utc_now(),
        "status": status,
        "calls_expected": script.calls_expected,
        "calls_actual": calls,
        "retry_policy": "none",
        "turn_count": len(records),
        "error": error_info,
        "client": client_context,
        "model": _model_context(),
        "generation": _generation_context(),
        "source_message_id": None,
        "system_sha256": actual_system_sha,
        "system_sha256_verified": system_sha256 is not None,
        "script_sha256": script_digest,
        "history_context_version": "history-context-b1",
        "max_history_messages": MAX_HISTORY_MESSAGES,
        "producer": "option_a.runner",
    }
    recorder.write_meta(meta)
    return RunResult(
        run_id=run_id,
        script_id=script.id,
        status=status,
        calls=calls,
        records=records,
        error=error_info,
        run_dir=str(recorder.run_dir),
    )


async def run_all(
    *,
    asset_root: Path,
    output_root: Path,
    client_factory: Callable[[], LLMClient] = FakeLLMClient,
    run_prefix: str = "option-a",
) -> list[RunResult]:
    # This preflight is intentionally before the first client factory call.
    # A stale/mutated manifest therefore produces zero model generations.
    manifest = load_and_verify_manifest(asset_root)
    if manifest.get("condition") != "A":
        raise FrozenAssetError("manifest condition is not A")
    system_prompt = load_system_prompt(asset_root)
    system_hash = sha256_text(system_prompt)
    if manifest["prompts"]["system"]["sha256"] != system_hash:
        raise FrozenAssetError("system prompt hash does not match the manifest")
    scripts = load_scripts(asset_root)
    scripts_file_hash = sha256_bytes(
        (asset_root / "scripts" / "scripts.json").read_bytes()
    )
    if manifest["files"]["scripts/scripts.json"]["sha256"] != scripts_file_hash:
        raise FrozenAssetError("scripts.json hash does not match the manifest")

    results: list[RunResult] = []
    batch = utc_now().replace(":", "").replace("+", "Z")
    for script in scripts:
        # A fresh client and fresh storage for every script.
        result = await run_script(
            script,
            system_prompt=system_prompt,
            client=client_factory(),
            output_root=output_root,
            run_id=safe_run_id(f"{run_prefix}-{batch}-{script.id}-{uuid.uuid4().hex[:6]}"),
            system_sha256=system_hash,
        )
        results.append(result)
    return results


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the isolated Option A fake-client experiment."
    )
    parser.add_argument(
        "--asset-root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="directory containing prompts/ and scripts/",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "runs",
        help="directory for exclusive run records",
    )
    parser.add_argument(
        "--run-prefix",
        default="option-a",
        help="prefix for generated run ids",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = asyncio.run(
        run_all(
            asset_root=args.asset_root,
            output_root=args.output_dir,
            run_prefix=args.run_prefix,
        )
    )
    summary = [
        {
            "run_id": result.run_id,
            "script_id": result.script_id,
            "status": result.status,
            "calls": result.calls,
            "run_dir": result.run_dir,
        }
        for result in results
    ]
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if all(result.status == "completed" for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
