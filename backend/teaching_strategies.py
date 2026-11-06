"""Versioned production teaching strategies, independent of interaction mode."""

from pathlib import Path

from backend.interaction_modes import (
    PROGRAMMATIC_HISTORY_PROMPT_VERSION,
    RECEIVE_TEACHING_PROMPT_VERSION,
    get_programmatic_system_prompt,
    get_system_prompt_for_mode,
)

TEACHING_STRATEGIES = ("baseline", "a", "b", "ab")
DEFAULT_TEACHING_STRATEGY = "baseline"
_PROMPTS = Path(__file__).with_name("prompts")
_BASE = (_PROMPTS / "teaching_base_v1.txt").read_text(encoding="utf-8")
_STRATEGIES = {
    name: (_PROMPTS / f"teaching_{name}_v1.txt").read_text(encoding="utf-8")
    for name in TEACHING_STRATEGIES if name != "baseline"
}


def is_valid_teaching_strategy(value: object) -> bool:
    return isinstance(value, str) and value in TEACHING_STRATEGIES


def get_teaching_prompt(strategy: str, *, history: bool = False) -> str:
    if not is_valid_teaching_strategy(strategy):
        raise ValueError("Unknown teaching strategy")
    if strategy == "baseline":
        return (
            get_programmatic_system_prompt() if history
            else get_system_prompt_for_mode("receive_teaching")
        )
    prompt = _BASE.rstrip() + "\n\n" + _STRATEGIES[strategy].rstrip() + "\n"
    return get_programmatic_system_prompt(prompt) if history else prompt


def get_teaching_prompt_version(strategy: str, route: str = "ordinary") -> str:
    if not is_valid_teaching_strategy(strategy):
        raise ValueError("Unknown teaching strategy")
    if route not in {"ordinary", "history", "clarify"}:
        raise ValueError("Unknown teaching route")
    if strategy == "baseline":
        return {
            "ordinary": RECEIVE_TEACHING_PROMPT_VERSION,
            "history": PROGRAMMATIC_HISTORY_PROMPT_VERSION,
            "clarify": "receive-teaching-v7+clarify-v1",
        }[route]
    suffix = {
        "ordinary": "history-context-b1",
        "history": "programmatic-history-v1",
        "clarify": "clarify-v1",
    }[route]
    return f"teach-{strategy}-v1+{suffix}"
