"""Freeze and verify the four-strategy comparison assets.

The manifest covers the human/machine-readable experimental assets, not the
runner/tests that exercise them.  It contains no timestamps so repeated
verification is deterministic.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

SCHEMA_VERSION = "strategy-comparison-manifest-v1"
CONDITION_IDS = ("standard", "a", "b", "ab")
TASK_IDS = ("S1", "S2", "S3", "S4", "S5", "S6", "I2", "I3")


class FrozenAssetError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def included_files(root: Path) -> list[Path]:
    rels = [
        "README.md",
        "protocol.md",
        "conditions.json",
        "run_plan.json",
        "branches/branches.json",
        "tasks/tasks.json",
        "expected/strategy_expectations.md",
        "scoring/scoring_guide.md",
        "scoring/blind_coding_sheet.csv",
        "scoring/blind_key.json",
        "scoring/stage2_compliance_sheet.csv",
        "scoring/stage1_lock.json",
    ]
    rels += [f"tasks/{task}.txt" for task in TASK_IDS]
    rels += [f"prompts/{name}.txt" for name in ("standard", "a", "b", "ab")]
    return [root / rel for rel in rels]


def build_manifest(root: Path) -> dict:
    root = root.resolve()
    files = []
    for path in included_files(root):
        if not path.is_file():
            raise FrozenAssetError(f"missing frozen file: {path.relative_to(root)}")
        files.append({
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    try:
        conditions = json.loads((root / "conditions.json").read_text())["conditions"]
        plan = json.loads((root / "run_plan.json").read_text())
        tasks = json.loads((root / "tasks/tasks.json").read_text())["tasks"]
    except Exception as exc:  # noqa: BLE001
        raise FrozenAssetError("asset json unreadable") from exc
    if tuple(c["strategy_id"] for c in conditions) != CONDITION_IDS:
        raise FrozenAssetError("condition ids/order mismatch")
    if tuple(t["id"] for t in tasks) != TASK_IDS:
        raise FrozenAssetError("task ids/order mismatch")
    if len(plan.get("sessions", [])) != 96:
        raise FrozenAssetError("run plan must have 96 sessions")
    if plan.get("fixed_calls") != 192:
        raise FrozenAssetError("fixed call budget mismatch")
    if plan.get("total_min_calls") != 240 or plan.get("total_max_calls") != 264:
        raise FrozenAssetError("total call budget mismatch")
    return {
        "schema_version": SCHEMA_VERSION,
        "condition_ids": list(CONDITION_IDS),
        "task_ids": list(TASK_IDS),
        "session_count": 96,
        "files": files,
    }


def verify_manifest(root: Path, manifest: dict | None = None) -> dict:
    root = root.resolve()
    if manifest is None:
        manifest = json.loads((root / "manifest.json").read_text())
    expected = {item["path"]: item for item in manifest.get("files", [])}
    actual = {path.relative_to(root).as_posix(): path for path in included_files(root)}
    if set(expected) != set(actual):
        missing = sorted(set(actual) - set(expected))
        extra = sorted(set(expected) - set(actual))
        raise FrozenAssetError(f"file set mismatch missing={missing} extra={extra}")
    for rel, path in actual.items():
        if path.stat().st_size != expected[rel]["bytes"]:
            raise FrozenAssetError(f"bytes mismatch: {rel}")
        if sha256_file(path) != expected[rel]["sha256"]:
            raise FrozenAssetError(f"sha256 mismatch: {rel}")
    rebuilt = build_manifest(root)
    for key in ("schema_version", "condition_ids", "task_ids", "session_count"):
        if manifest.get(key) != rebuilt.get(key):
            raise FrozenAssetError(f"manifest field mismatch: {key}")
    if manifest.get("files") != rebuilt["files"]:
        raise FrozenAssetError("manifest file records mismatch")
    return rebuilt


def write_manifest(root: Path) -> dict:
    manifest = build_manifest(root)
    (root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.write:
        write_manifest(args.root)
        print("manifest written")
    else:
        verify_manifest(args.root)
        print("manifest verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
