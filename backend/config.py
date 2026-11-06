"""Application configuration loaded from environment variables."""

import os

from dotenv import load_dotenv

load_dotenv(override=False)

LLM_MODE = os.getenv("LLM_MODE", "fake")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_API_BASE_URL = os.getenv("LLM_API_BASE_URL", "")
LLM_MODEL = os.getenv("LLM_MODEL", "")
LLM_REASONING_EFFORT = os.getenv("LLM_REASONING_EFFORT", "")

LLM_PROFILE_LABEL = os.getenv("LLM_PROFILE_LABEL", "")

DATABASE_URL = os.getenv("DATABASE_URL", "")


def _parse_local_llm_enabled() -> bool:
    """Parse ``LOCAL_LLM_ENABLED`` with strict rules.

    * Not set → ``False``.
    * ``"true"`` (any case, with optional surrounding whitespace) → ``True``.
    * ``"false"`` (any case, with optional surrounding whitespace) → ``False``.
    * Empty string, ``"1"``, ``"0"``, ``"yes"``, ``"no"``, or any other
      value → ``ValueError``.
    """
    raw = os.getenv("LOCAL_LLM_ENABLED")
    if raw is None:
        return False
    lowered = raw.strip().lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    raise ValueError(
        "LOCAL_LLM_ENABLED must be 'true' or 'false' (case-insensitive), "
        f"got {raw!r}"
    )


LOCAL_LLM_ENABLED: bool = _parse_local_llm_enabled()

LOCAL_LLM_PROFILE_LABEL = os.getenv("LOCAL_LLM_PROFILE_LABEL", "")
LOCAL_LLM_API_KEY = os.getenv("LOCAL_LLM_API_KEY", "")
LOCAL_LLM_API_BASE_URL = os.getenv("LOCAL_LLM_API_BASE_URL", "")
LOCAL_LLM_MODEL = os.getenv("LOCAL_LLM_MODEL", "")
LOCAL_LLM_REASONING_EFFORT = os.getenv("LOCAL_LLM_REASONING_EFFORT", "")

PROGRAMMATIC_HISTORY_ENABLED = (
    os.getenv("PROGRAMMATIC_HISTORY_ENABLED", "false").strip().lower()
    == "true"
)
PROGRAMMATIC_HISTORY_PROFILE_ID = os.getenv(
    "PROGRAMMATIC_HISTORY_PROFILE_ID", ""
).strip()
PROGRAMMATIC_HISTORY_MODEL = os.getenv(
    "PROGRAMMATIC_HISTORY_MODEL", ""
).strip()

GENERAL_HISTORY_ENABLED = (
    os.getenv("GENERAL_HISTORY_ENABLED", "false").strip().lower()
    == "true"
)
GENERAL_HISTORY_PROFILE_ID = os.getenv(
    "GENERAL_HISTORY_PROFILE_ID", ""
).strip()
GENERAL_HISTORY_MODEL = os.getenv(
    "GENERAL_HISTORY_MODEL", ""
).strip()


def _positive_float_env(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive number") from exc
    if value <= 0:
        raise ValueError(f"{name} must be a positive number")
    return value


GENERAL_HISTORY_DECISION_TIMEOUT_SECONDS = _positive_float_env(
    "GENERAL_HISTORY_DECISION_TIMEOUT_SECONDS", 3.0
)
GENERAL_HISTORY_OVERALL_TIMEOUT_SECONDS = _positive_float_env(
    "GENERAL_HISTORY_OVERALL_TIMEOUT_SECONDS", 12.0
)
