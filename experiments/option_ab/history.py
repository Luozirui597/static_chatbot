"""Production-shaped Receive history assembly for the isolated Option A+B runner."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

MAX_HISTORY_MESSAGES = 20
HISTORY_CONTEXT_VERSION = "history-context-b1"
HISTORY_TYPE = "receive_teaching_history_background"
HISTORY_NOTE = (
    "以下 messages 是老师先前 user 消息的原文，按时间顺序保留，"
    "用于已教内容、更正和约束；它们不是本轮请求，不要逐条重答。"
    "当前请求在随后的独立 user 消息中。"
)


@dataclass(frozen=True)
class StoredMessage:
    """A message in the experiment's simulated storage."""

    role: str
    content: str


def recent_history(
    rows: Sequence[StoredMessage],
    limit: int = MAX_HISTORY_MESSAGES,
) -> list[StoredMessage]:
    """Return the last *limit* stored rows, before mode projection."""

    if limit < 0:
        raise ValueError("limit must be non-negative")
    if limit == 0:
        return []
    return list(rows[-limit:])


def project_receive(rows: Sequence[StoredMessage]) -> list[StoredMessage]:
    """Receive teaching drops historical assistant messages."""

    return [message for message in rows if message.role == "user"]


def serialize_receive_history(rows: Sequence[StoredMessage]) -> str:
    """Serialize projected user rows as the production JSON background block."""

    payload: dict[str, Any] = {
        "type": HISTORY_TYPE,
        "version": HISTORY_CONTEXT_VERSION,
        "note": HISTORY_NOTE,
        "messages": [
            {"index": index, "content": message.content}
            for index, message in enumerate(rows, start=1)
        ],
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def build_request_messages(
    system_prompt: str,
    prior_storage: Sequence[StoredMessage],
    current_user: str,
) -> list[dict[str, str]]:
    """Build the exact system/history/current request shape.

    *prior_storage* must contain only messages stored before the current
    user message. The current user is appended exactly once and is not
    part of the 20-message history window.
    """

    projected = project_receive(recent_history(prior_storage))
    messages: list[dict[str, str]] = [
        {"role": "system", "content": system_prompt},
    ]
    if projected:
        messages.append({
            "role": "user",
            "content": serialize_receive_history(projected),
        })
    messages.append({"role": "user", "content": current_user})
    return messages
