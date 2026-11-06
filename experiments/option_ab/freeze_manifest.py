"""Freeze and verify hashes for the isolated Option A+B experiment assets."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

MANIFEST_SCHEMA_VERSION = "option-ab-manifest-v1"
ASSET_VERSION = "option-ab-isolated-v0.1"
SHARED_BASE_SHA256 = "eda835c58da3c1cbd05d83d5ac34f4ff5bbd805b8a466796957b7b489e00bb6f"
FIXED_CALLS = {"S1": 2, "S2": 2, "S3": 2, "S4": 2, "S5": 4, "S6": 4}
BRANCHES = {
    "I2": {"bc", "direct_ac", "no_extra"},
    "I3": {"rule_supply", "direct_ac", "no_extra"},
}
INTERACTIVE_CALLS = {
    "I1": 2,
    "I2": {"minimum": 2, "maximum": 3},
    "I3": {"minimum": 2, "maximum": 3},
    "I4": 2,
}
INCLUDED_FILES = (
    "__init__.py",
    "README.md",
    "freeze_manifest.py",
    "history.py",
    "runner.py",
    "prompts/base.txt",
    "prompts/ab_strategy.txt",
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
    "scripts/I4.txt",
    "expected/per_turn_expectations.md",
    "expected/coding_sheet.csv",
    "tests/__init__.py",
    "tests/test_option_ab.py",
)


class FrozenAssetError(RuntimeError):
    """The experiment assets do not match their frozen manifest."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_info(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    return {"bytes": len(data), "sha256": sha256_bytes(data)}


def _validate_scripts(scripts: Any) -> None:
    if not isinstance(scripts, dict):
        raise ValueError("scripts.json must be an object")
    if scripts.get("condition") != "AB":
        raise ValueError("scripts condition must be AB")
    if scripts.get("asset_version") != ASSET_VERSION:
        raise ValueError("scripts asset_version mismatch")
    items = scripts.get("scripts")
    if not isinstance(items, list):
        raise ValueError("scripts must be a list")
    ids = [item.get("id") for item in items]
    expected_ids = ["S1", "S2", "S3", "S4", "S5", "S6", "I1", "I2", "I3", "I4"]
    if ids != expected_ids:
        raise ValueError("scripts must define S1-S6 then I1-I4 in order")
    for item in items:
        sid = item["id"]
        turns = item.get("turns")
        if not isinstance(turns, list) or not turns:
            raise ValueError(f"{sid}: non-empty turns required")
        for turn in turns:
            if set(turn) != {"id", "phase", "content"}:
                raise ValueError(f"{sid}: turn fields must be id/phase/content")
            if not isinstance(turn["content"], str) or not turn["content"].strip():
                raise ValueError(f"{sid}: empty turn content")
        branches = item.get("branches", [])
        if not isinstance(branches, list):
            raise ValueError(f"{sid}: branches must be a list")
        if sid in FIXED_CALLS:
            if item.get("mode") != "fixed" or item.get("calls_expected") != FIXED_CALLS[sid] or len(turns) != FIXED_CALLS[sid] or branches:
                raise ValueError(f"{sid}: fixed call/turn/branch mismatch")
        elif sid in {"I1", "I4"}:
            if item.get("mode") != "interactive" or item.get("calls_expected") != 2 or len(turns) != 2 or branches:
                raise ValueError(f"{sid}: must be interactive 2 turns, no branches")
        elif sid in BRANCHES:
            if item.get("mode") != "interactive" or item.get("calls_expected") != {"minimum": 2, "maximum": 3} or len(turns) != 2:
                raise ValueError(f"{sid}: must be interactive 2-3 calls")
            if {b.get("id") for b in branches} != BRANCHES[sid]:
                raise ValueError(f"{sid}: unexpected branch set")
        else:
            raise ValueError(f"unknown script id: {sid}")
    if scripts.get("logical_runs") != 10:
        raise ValueError("logical_runs must be 10")
    expected_budget = {
        "fixed_calls": 16,
        "interactive_min": 8,
        "interactive_max": 10,
        "total_min": 24,
        "total_max": 26,
    }
    if scripts.get("budget") != expected_budget:
        raise ValueError("script budget mismatch")


def build_manifest(asset_root: Path) -> dict[str, Any]:
    base = (asset_root / "prompts" / "base.txt").read_text(encoding="utf-8")
    strategy = (asset_root / "prompts" / "ab_strategy.txt").read_text(encoding="utf-8")
    expected_system = base.rstrip("\n") + "\n\n" + strategy.rstrip("\n") + "\n"
    actual_system = (asset_root / "prompts" / "system.txt").read_text(encoding="utf-8")
    if actual_system != expected_system:
        raise ValueError("prompts/system.txt is not the exact base + AB strategy assembly")
    base_info = file_info(asset_root / "prompts" / "base.txt")
    if base_info["sha256"] != SHARED_BASE_SHA256:
        raise ValueError("shared base does not match the frozen A base")
    system_info = file_info(asset_root / "prompts" / "system.txt")
    prompt_version = f"{ASSET_VERSION}+sys-{system_info['sha256'][:12]}"
    if len(prompt_version) > 50:
        raise ValueError("prompt version exceeds the production snapshot limit")
    files = {}
    for relative in INCLUDED_FILES:
        path = asset_root / relative
        if not path.is_file():
            raise FileNotFoundError(f"missing frozen asset: {relative}")
        files[relative] = file_info(path)
    scripts = json.loads((asset_root / "scripts" / "scripts.json").read_text(encoding="utf-8"))
    _validate_scripts(scripts)
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "asset_version": ASSET_VERSION,
        "condition": "AB",
        "generated_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "encoding": "utf-8",
        "line_endings": "LF",
        "prompt_assembly": "base.rstrip('\\n') + '\\n\\n' + ab_strategy.rstrip('\\n') + '\\n'",
        "shared_base_sha256": SHARED_BASE_SHA256,
        "prompt_version": prompt_version,
        "prompts": {
            "base": base_info,
            "ab_strategy": files["prompts/ab_strategy.txt"],
            "system": system_info,
        },
        "calls": {
            "fixed_per_script": FIXED_CALLS,
            "fixed_total": 16,
            "interactive": INTERACTIVE_CALLS,
            "interactive_min": 8,
            "interactive_max": 10,
            "total_min": 24,
            "total_max": 26,
            "logical_runs": 10,
        },
        "files": files,
        "notes": [
            "No production prompt, mode, database, or application file is part of this manifest.",
            "The runner uses only the local fake client by default.",
            "A+B is a separate condition with a bounded R1 and one focused question.",
            "Fixed scripts use the exact shared S1-S6 user content; interactive branches require human selection.",
        ],
    }


def _without_timestamp(manifest: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in manifest.items() if key != "generated_at_utc"}


def verify_manifest(asset_root: Path, manifest: dict[str, Any]) -> list[str]:
    if not isinstance(manifest, dict):
        return ["manifest is not an object"]
    errors: list[str] = []
    files = manifest.get("files")
    if not isinstance(files, dict):
        return ["manifest.files is missing"]
    required = set(INCLUDED_FILES)
    actual = set(files)
    if required - actual:
        errors.append("manifest.files missing required entries: " + ", ".join(sorted(required - actual)))
    if actual - required:
        errors.append("manifest.files has unexpected entries: " + ", ".join(sorted(actual - required)))
    for relative in INCLUDED_FILES:
        path = asset_root / relative
        if not path.is_file():
            errors.append(f"missing: {relative}")
        elif not isinstance(files.get(relative), dict) or file_info(path) != files[relative]:
            errors.append(f"hash/bytes mismatch: {relative}")
    try:
        rebuilt = build_manifest(asset_root)
    except Exception as exc:
        errors.append(f"cannot rebuild manifest: {type(exc).__name__}: {exc}")
        return errors
    if _without_timestamp(manifest) != _without_timestamp(rebuilt):
        errors.append("manifest content does not match the current frozen assets")
    return errors


def load_and_verify_manifest(asset_root: Path) -> dict[str, Any]:
    path = asset_root / "manifest.json"
    if not path.is_file():
        raise FrozenAssetError("manifest.json is missing")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    errors = verify_manifest(asset_root, manifest)
    if errors:
        raise FrozenAssetError("; ".join(errors))
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build or verify the Option A+B manifest.")
    parser.add_argument("--asset-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args(argv)
    if args.verify:
        manifest = load_and_verify_manifest(args.asset_root)
        print("manifest verified")
        print(f"condition: {manifest['condition']}")
        print(f"total calls: {manifest['calls']['total_min']}-{manifest['calls']['total_max']}")
        return 0
    manifest = build_manifest(args.asset_root)
    (args.asset_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {args.asset_root / 'manifest.json'}")
    print(f"prompt_version: {manifest['prompt_version']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
