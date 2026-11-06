"""Subprocess checks that the experiment CLI does not source production dotenv."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
KEY_NAME = "GENERAL_TOOL_SYNTHETIC_DOTENV_KEY"


def _env_without_key() -> dict[str, str]:
    env = os.environ.copy()
    env.pop(KEY_NAME, None)
    old_path = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        str(REPO) if not old_path else str(REPO) + os.pathsep + old_path
    )
    return env


def _run(tmp_path: Path, *args: str):
    return subprocess.run(
        [sys.executable, "-m", "experiments.general_tool_search.runner", *args],
        cwd=tmp_path,
        env=_env_without_key(),
        text=True,
        capture_output=True,
        timeout=20,
    )


def test_real_mode_does_not_load_synthetic_dotenv_key(tmp_path):
    (tmp_path / ".env").write_text(
        f"{KEY_NAME}=from-synthetic-dotenv\n", encoding="utf-8"
    )
    result = _run(
        tmp_path,
        "--real-api",
        "--api-key-env",
        KEY_NAME,
        "--base-url",
        "http://127.0.0.1:9/v1",
        "--model",
        "synthetic-model",
        "--decision-timeout",
        "0.1",
        "--overall-timeout",
        "0.1",
        "--case",
        "no_hidden",
    )
    assert result.returncode == 2
    assert "key environment variable is not set" in result.stdout
    assert "from-synthetic-dotenv" not in result.stdout + result.stderr


def test_fake_default_still_runs_with_dotenv_present(tmp_path):
    (tmp_path / ".env").write_text(
        f"{KEY_NAME}=from-synthetic-dotenv\n", encoding="utf-8"
    )
    result = _run(
        tmp_path,
        "--case",
        "no_hidden",
        "--answer",
        "The delta fold is [message 7].",
    )
    assert result.returncode == 0, result.stderr
    assert '"status": "completed_direct"' in result.stdout
    assert '"synthetic": true' in result.stdout
    assert "from-synthetic-dotenv" not in result.stdout + result.stderr
