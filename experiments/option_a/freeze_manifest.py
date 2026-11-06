"""Freeze and verify hashes for the isolated Option A experiment assets."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

MANIFEST_SCHEMA_VERSION = "option-a-manifest-v1"
ASSET_VERSION = "option-a-isolated-v0.1"
INCLUDED_FILES = (
    "__init__.py",
    "prompts/base.txt",
    "prompts/a_strategy.txt",
    "prompts/system.txt",
    "scripts/scripts.json",
    "scripts/S1.txt",
    "scripts/S2.txt",
    "scripts/S3.txt",
    "scripts/S4.txt",
    "scripts/S5.txt",
    "scripts/S6.txt",
    "expected/per_turn_expectations.md",
    "expected/coding_sheet.csv",
    "history.py",
    "runner.py",
    "freeze_manifest.py",
    "tests/__init__.py",
    "tests/test_option_a.py",
    "README.md",
)


class FrozenAssetError(RuntimeError):
    """The experiment assets do not match their frozen manifest."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_info(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    return {
        "bytes": len(data),
        "sha256": sha256_bytes(data),
    }


def build_manifest(asset_root: Path) -> dict[str, Any]:
    base = (asset_root / "prompts" / "base.txt").read_text(encoding="utf-8")
    strategy = (asset_root / "prompts" / "a_strategy.txt").read_text(
        encoding="utf-8"
    )
    expected_system = base.rstrip("\n") + "\n\n" + strategy.rstrip("\n") + "\n"
    actual_system = (asset_root / "prompts" / "system.txt").read_text(
        encoding="utf-8"
    )
    if actual_system != expected_system:
        raise ValueError("prompts/system.txt is not the exact base + A assembly")

    files: dict[str, dict[str, Any]] = {}
    for relative in INCLUDED_FILES:
        path = asset_root / relative
        if not path.is_file():
            raise FileNotFoundError(f"missing frozen asset: {relative}")
        files[relative] = file_info(path)

    scripts = json.loads(
        (asset_root / "scripts" / "scripts.json").read_text(encoding="utf-8")
    )
    calls = {
        script["id"]: script["calls_expected"]
        for script in scripts.get("scripts", [])
    }
    if calls != {"S1": 2, "S2": 2, "S3": 2, "S4": 2, "S5": 4, "S6": 4}:
        raise ValueError("unexpected per-script call counts")

    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "asset_version": ASSET_VERSION,
        "condition": "A",
        "generated_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "encoding": "utf-8",
        "line_endings": "LF",
        "prompt_assembly": "base.rstrip('\\n') + '\\n\\n' + a_strategy.rstrip('\\n') + '\\n'",
        "prompts": {
            "base": files["prompts/base.txt"],
            "a_strategy": files["prompts/a_strategy.txt"],
            "system": files["prompts/system.txt"],
        },
        "calls": {
            "per_script": calls,
            "total_expected": sum(calls.values()),
        },
        "files": files,
        "notes": [
            "No production prompt, mode, database, or application file is part of this manifest.",
            "The runner uses only the local fake client by default.",
            "Records are created under runs/<run_id>/ and are never overwritten.",
        ],
    }


def _content_without_timestamp(manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in manifest.items()
        if key != "generated_at_utc"
    }


def verify_manifest(asset_root: Path, manifest: dict[str, Any]) -> list[str]:
    """Compare a manifest with the complete current asset content.

    The timestamp is intentionally excluded from the comparison. Everything
    else, including the required file set, schema/condition, prompt assembly,
    call counts, and every file hash, is compared against a fresh build.
    """

    if not isinstance(manifest, dict):
        return ["manifest is not an object"]

    errors: list[str] = []
    files = manifest.get("files")
    if not isinstance(files, dict):
        return ["manifest.files is missing"]
    required = set(INCLUDED_FILES)
    actual_keys = set(files)
    missing = sorted(required - actual_keys)
    extra = sorted(actual_keys - required)
    if missing:
        errors.append("manifest.files missing required entries: " + ", ".join(missing))
    if extra:
        errors.append("manifest.files has unexpected entries: " + ", ".join(extra))

    for relative in INCLUDED_FILES:
        expected = files.get(relative)
        path = asset_root / relative
        if not path.is_file():
            errors.append(f"missing: {relative}")
            continue
        if not isinstance(expected, dict):
            errors.append(f"manifest has no file info: {relative}")
            continue
        if file_info(path) != expected:
            errors.append(f"hash/bytes mismatch: {relative}")

    try:
        rebuilt = build_manifest(asset_root)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        errors.append(f"cannot rebuild manifest: {type(exc).__name__}: {exc}")
        return errors

    if _content_without_timestamp(rebuilt) != _content_without_timestamp(manifest):
        errors.append(
            "complete frozen content mismatch (schema/condition/prompt/calls/files)"
        )
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
    """Load and fully verify the manifest before any client is created."""

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
    parser = argparse.ArgumentParser(description="Freeze Option A asset hashes.")
    parser.add_argument(
        "--asset-root",
        type=Path,
        default=Path(__file__).resolve().parent,
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="verify the existing manifest instead of rewriting it",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.verify:
        manifest_path = args.asset_root / "manifest.json"
        try:
            manifest = load_and_verify_manifest(args.asset_root)
        except FrozenAssetError as exc:
            print(str(exc))
            return 1
        print(f"manifest verified: {manifest_path}")
        print(f"condition: {manifest['condition']}")
        print(f"total calls: {manifest['calls']['total_expected']}")
        return 0

    path = write_manifest(args.asset_root)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    print(f"manifest written: {path}")
    print(json.dumps(manifest["prompts"], ensure_ascii=False, indent=2))
    print(json.dumps(manifest["calls"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
