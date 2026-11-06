"""Freeze and verify hashes for the isolated Option B experiment assets."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

MANIFEST_SCHEMA_VERSION = "option-b-manifest-v1"
ASSET_VERSION = "option-b-isolated-v0.1"
SHARED_BASE_SHA256 = (
    "eda835c58da3c1cbd05d83d5ac34f4ff5bbd805b8a466796957b7b489e00bb6f"
)
FIXED_CALLS = {"S1": 2, "S2": 2, "S3": 2, "S4": 2, "S5": 4, "S6": 4}
INTERACTIVE_CALLS = {"I1": 2, "I2": {"minimum": 2, "maximum": 3}, "I3": {"minimum": 2, "maximum": 3}}
INCLUDED_FILES = (
    "__init__.py",
    "README.md",
    "freeze_manifest.py",
    "history.py",
    "runner.py",
    "prompts/base.txt",
    "prompts/b_strategy.txt",
    "prompts/system.txt",
    "scripts/scripts.json",
    "scripts/S1.txt",
    "scripts/S2.txt",
    "scripts/S3.txt",
    "scripts/S4.txt",
    "scripts/S5.txt",
    "scripts/S6.txt",
    "scripts/I1.txt",
    "scripts/I2.txt",
    "scripts/I3.txt",
    "expected/per_turn_expectations.md",
    "expected/coding_sheet.csv",
    "tests/__init__.py",
    "tests/test_option_b.py",
)


class FrozenAssetError(RuntimeError):
    """The experiment assets do not match their frozen manifest."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_info(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    return {"bytes": len(data), "sha256": sha256_bytes(data)}


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_scripts(scripts: Any) -> None:
    if not isinstance(scripts, dict):
        raise ValueError("scripts.json must be an object")
    if scripts.get("condition") != "B":
        raise ValueError("scripts condition must be B")
    if scripts.get("asset_version") != ASSET_VERSION:
        raise ValueError("scripts asset_version mismatch")
    items = scripts.get("scripts")
    if not isinstance(items, list):
        raise ValueError("scripts must be a list")
    ids = [item.get("id") for item in items]
    if ids != ["S1", "S2", "S3", "S4", "S5", "S6", "I1", "I2", "I3"]:
        raise ValueError("scripts must define S1-S6 then I1-I3 in order")
    fixed = {}
    interactive = {}
    for item in items:
        script_id = item["id"]
        turns = item.get("turns")
        if not isinstance(turns, list) or not turns:
            raise ValueError(f"{script_id}: non-empty turns required")
        for turn in turns:
            if set(turn) != {"id", "phase", "content"}:
                raise ValueError(f"{script_id}: turn fields must be id/phase/content")
            if not isinstance(turn["content"], str) or not turn["content"].strip():
                raise ValueError(f"{script_id}: empty turn content")
        if script_id.startswith("S"):
            if item.get("mode") != "fixed":
                raise ValueError(f"{script_id}: mode must be fixed")
            expected = FIXED_CALLS[script_id]
            if item.get("calls_expected") != expected or len(turns) != expected:
                raise ValueError(f"{script_id}: fixed call count mismatch")
            fixed[script_id] = expected
        else:
            if item.get("mode") != "interactive":
                raise ValueError(f"{script_id}: mode must be interactive")
            expected = INTERACTIVE_CALLS[script_id]
            if item.get("calls_expected") != expected:
                raise ValueError(f"{script_id}: interactive call count mismatch")
            if len(turns) != 2:
                raise ValueError(f"{script_id}: interactive Q1 pause requires two turns")
            if not isinstance(item.get("branches"), list):
                raise ValueError(f"{script_id}: branches must be a list")
            interactive[script_id] = expected
    if fixed != FIXED_CALLS:
        raise ValueError("fixed call counts mismatch")
    if interactive != INTERACTIVE_CALLS:
        raise ValueError("interactive call counts mismatch")
    budget = scripts.get("budget")
    expected_budget = {
        "fixed_calls": 16,
        "interactive_min": 6,
        "interactive_max": 8,
        "total_min": 22,
        "total_max": 24,
    }
    if budget != expected_budget:
        raise ValueError("script budget mismatch")
    if scripts.get("logical_runs") != 9:
        raise ValueError("logical_runs must be 9")


def build_manifest(asset_root: Path) -> dict[str, Any]:
    base = (asset_root / "prompts" / "base.txt").read_text(encoding="utf-8")
    strategy = (asset_root / "prompts" / "b_strategy.txt").read_text(
        encoding="utf-8"
    )
    expected_system = base.rstrip("\n") + "\n\n" + strategy.rstrip("\n") + "\n"
    actual_system = (asset_root / "prompts" / "system.txt").read_text(
        encoding="utf-8"
    )
    if actual_system != expected_system:
        raise ValueError("prompts/system.txt is not the exact base + B assembly")
    base_info = file_info(asset_root / "prompts" / "base.txt")
    if base_info["sha256"] != SHARED_BASE_SHA256:
        raise ValueError("shared base does not match the frozen A base")
    system_info = file_info(asset_root / "prompts" / "system.txt")
    prompt_version = f"{ASSET_VERSION}+sys-{system_info['sha256'][:12]}"
    if len(prompt_version) > 50:
        raise ValueError("prompt version exceeds the production snapshot limit")

    files: dict[str, dict[str, Any]] = {}
    for relative in INCLUDED_FILES:
        path = asset_root / relative
        if not path.is_file():
            raise FileNotFoundError(f"missing frozen asset: {relative}")
        files[relative] = file_info(path)

    scripts = _read_json(asset_root / "scripts" / "scripts.json")
    _validate_scripts(scripts)
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "asset_version": ASSET_VERSION,
        "condition": "B",
        "generated_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "encoding": "utf-8",
        "line_endings": "LF",
        "prompt_assembly": "base.rstrip('\\n') + '\\n\\n' + b_strategy.rstrip('\\n') + '\\n'",
        "shared_base_sha256": SHARED_BASE_SHA256,
        "prompt_version": prompt_version,
        "prompts": {
            "base": base_info,
            "b_strategy": files["prompts/b_strategy.txt"],
            "system": system_info,
        },
        "calls": {
            "fixed_per_script": FIXED_CALLS,
            "fixed_total": 16,
            "interactive": INTERACTIVE_CALLS,
            "interactive_min": 6,
            "interactive_max": 8,
            "total_min": 22,
            "total_max": 24,
            "logical_runs": 9,
        },
        "files": files,
        "notes": [
            "No production prompt, mode, database, or application file is part of this manifest.",
            "The runner uses only the local fake client by default.",
            "B does not receive any R1 or derivation permission.",
            "Fixed scripts use the exact shared S1-S6 user content; interactive branches require human selection.",
        ],
    }


def _without_timestamp(manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in manifest.items()
        if key != "generated_at_utc"
    }


def verify_manifest(asset_root: Path, manifest: dict[str, Any]) -> list[str]:
    if not isinstance(manifest, dict):
        return ["manifest is not an object"]
    errors: list[str] = []
    files = manifest.get("files")
    if not isinstance(files, dict):
        return ["manifest.files is missing"]
    required = set(INCLUDED_FILES)
    actual = set(files)
    missing = sorted(required - actual)
    extra = sorted(actual - required)
    if missing:
        errors.append("manifest.files missing required entries: " + ", ".join(missing))
    if extra:
        errors.append("manifest.files has unexpected entries: " + ", ".join(extra))

    for relative in INCLUDED_FILES:
        expected = files.get(relative)
        path = asset_root / relative
        if not path.is_file():
            errors.append(f"missing: {relative}")
        elif not isinstance(expected, dict):
            errors.append(f"manifest has no file info: {relative}")
        elif file_info(path) != expected:
            errors.append(f"hash/bytes mismatch: {relative}")

    try:
        rebuilt = build_manifest(asset_root)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        errors.append(f"cannot rebuild manifest: {type(exc).__name__}: {exc}")
        return errors

    if _without_timestamp(rebuilt) != _without_timestamp(manifest):
        errors.append("complete frozen content mismatch (schema/condition/prompt/calls/files)")
    return errors


def load_manifest(asset_root: Path) -> dict[str, Any]:
    path = asset_root / "manifest.json"
    if not path.is_file():
        raise FrozenAssetError(f"missing frozen manifest: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FrozenAssetError(f"cannot read frozen manifest: {exc}") from exc
    if not isinstance(data, dict):
        raise FrozenAssetError("frozen manifest must be a JSON object")
    return data


def load_and_verify_manifest(asset_root: Path) -> dict[str, Any]:
    manifest = load_manifest(asset_root)
    errors = verify_manifest(asset_root, manifest)
    if errors:
        raise FrozenAssetError("; ".join(errors))
    return manifest


def write_manifest(asset_root: Path) -> Path:
    manifest = build_manifest(asset_root)
    path = asset_root / "manifest.json"
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze Option B asset hashes.")
    parser.add_argument(
        "--asset-root",
        type=Path,
        default=Path(__file__).resolve().parent,
    )
    parser.add_argument("--verify", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.verify:
        try:
            manifest = load_and_verify_manifest(args.asset_root)
        except FrozenAssetError as exc:
            print(str(exc))
            return 1
        print(f"manifest verified: {args.asset_root / 'manifest.json'}")
        print(f"condition: {manifest['condition']}")
        print(f"logical runs: {manifest['calls']['logical_runs']}")
        print(f"calls: {manifest['calls']['total_min']}-{manifest['calls']['total_max']}")
        return 0
    path = write_manifest(args.asset_root)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    print(f"manifest written: {path}")
    print(json.dumps(manifest["prompts"], ensure_ascii=False, indent=2))
    print(json.dumps(manifest["calls"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
