"""Deterministic history-review source selection.

This module is pure: it contains no database access, no HTTP, no LLM
calls, and no prompt text.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

HISTORY_REVIEW_MAX_USER_MESSAGES = 10
HISTORY_REVIEW_MAX_SOURCE_CHARS = 4000

HISTORY_REVIEW_CONTEXT_TOKENS = 4096
HISTORY_REVIEW_RESERVED_PROMPT_TOKENS = 800
HISTORY_REVIEW_RESERVED_OUTPUT_TOKENS = 1024
HISTORY_REVIEW_RESERVED_MARGIN_TOKENS = 256
HISTORY_REVIEW_SOURCE_TOKEN_BUDGET = 2016
HISTORY_REVIEW_MESSAGE_OVERHEAD_TOKENS = 16

HISTORY_REVIEW_BUDGET_VERSION = "history-review-budget-v1"
HISTORY_REVIEW_SELECTION_POLICY_VERSION = "history-review-selection-v1"


class HistoryReviewSourceMessageTooLarge(Exception):
    """Raised when the newest eligible message cannot fit any budget."""

    def __init__(self, message_id: int, reason: str) -> None:
        self.message_id = message_id
        self.reason = reason
        super().__init__(
            f"Newest eligible message {message_id} exceeds "
            f"single-message review budget: {reason}"
        )


def estimate_history_review_tokens(text: str) -> int:
    """Estimate token cost with a fixed, model-independent formula.

    Non-ASCII characters count as one token each.  ASCII characters
    count as four characters per token, rounded up.  Every message also
    pays ``HISTORY_REVIEW_MESSAGE_OVERHEAD_TOKENS``.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    ascii_count = sum(1 for char in text if ord(char) < 128)
    non_ascii_count = len(text) - ascii_count
    ascii_tokens = math.ceil(ascii_count / 4)
    return (
        non_ascii_count
        + ascii_tokens
        + HISTORY_REVIEW_MESSAGE_OVERHEAD_TOKENS
    )


@dataclass(frozen=True)
class HistoryReviewCandidate:
    """Immutable, already-filtered eligible user message."""

    message_id: int
    content: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.message_id, int)
            or isinstance(self.message_id, bool)
            or self.message_id < 1
        ):
            raise ValueError("message_id must be a positive integer")
        if not isinstance(self.content, str):
            raise TypeError("content must be a string")


@dataclass(frozen=True)
class HistoryReviewSelection:
    """Immutable result of deterministic source selection."""

    eligible_message_count: int
    eligible_char_count: int
    eligible_from_message_id: int | None
    eligible_through_message_id: int | None

    selected_candidates: tuple[HistoryReviewCandidate, ...]
    source_message_count: int
    source_char_count: int
    source_token_estimate: int
    source_from_message_id: int | None
    source_through_message_id: int | None

    truncated: bool
    budget_version: str
    selection_policy_version: str


def _validate_candidates(
    candidates: Sequence[HistoryReviewCandidate],
) -> tuple[HistoryReviewCandidate, ...]:
    if isinstance(candidates, (list, tuple)) is False:
        raise TypeError("candidates must be a list or tuple")

    previous_message_id = 0
    for candidate in candidates:
        if isinstance(candidate, HistoryReviewCandidate) is False:
            raise TypeError(
                "every candidate must be a HistoryReviewCandidate"
            )
        if candidate.message_id <= previous_message_id:
            raise ValueError(
                "candidates must have strictly increasing message IDs"
            )
        previous_message_id = candidate.message_id
    return tuple(candidates)


def select_history_review_sources(
    candidates: Sequence[HistoryReviewCandidate],
) -> HistoryReviewSelection:
    """Select the newest contiguous run of eligible messages within budget.

    ``candidates`` must already be filtered to one session, one teaching
    cycle, ``role == "user"`` and
    ``interaction_mode_snapshot == "receive_teaching"``.
    """
    items = _validate_candidates(candidates)

    eligible_count = len(items)
    eligible_chars = sum(len(item.content) for item in items)
    eligible_from = items[0].message_id if items else None
    eligible_through = items[-1].message_id if items else None

    selected_reversed: list[HistoryReviewCandidate] = []
    source_chars = 0
    source_tokens = 0

    for candidate in reversed(items):
        candidate_chars = len(candidate.content)
        candidate_tokens = estimate_history_review_tokens(
            candidate.content
        )

        if len(selected_reversed) == 0:
            if candidate_chars > HISTORY_REVIEW_MAX_SOURCE_CHARS:
                raise HistoryReviewSourceMessageTooLarge(
                    candidate.message_id,
                    "content exceeds maximum source characters",
                )
            if candidate_tokens > HISTORY_REVIEW_SOURCE_TOKEN_BUDGET:
                raise HistoryReviewSourceMessageTooLarge(
                    candidate.message_id,
                    "content exceeds source token budget",
                )

        if len(selected_reversed) >= HISTORY_REVIEW_MAX_USER_MESSAGES:
            break
        if source_chars + candidate_chars > HISTORY_REVIEW_MAX_SOURCE_CHARS:
            break
        if (
            source_tokens + candidate_tokens
            > HISTORY_REVIEW_SOURCE_TOKEN_BUDGET
        ):
            break

        selected_reversed.append(candidate)
        source_chars += candidate_chars
        source_tokens += candidate_tokens

    selected = tuple(reversed(selected_reversed))
    return HistoryReviewSelection(
        eligible_message_count=eligible_count,
        eligible_char_count=eligible_chars,
        eligible_from_message_id=eligible_from,
        eligible_through_message_id=eligible_through,
        selected_candidates=selected,
        source_message_count=len(selected),
        source_char_count=source_chars,
        source_token_estimate=source_tokens,
        source_from_message_id=(
            selected[0].message_id if selected else None
        ),
        source_through_message_id=(
            selected[-1].message_id if selected else None
        ),
        truncated=len(selected) < eligible_count,
        budget_version=HISTORY_REVIEW_BUDGET_VERSION,
        selection_policy_version=(
            HISTORY_REVIEW_SELECTION_POLICY_VERSION
        ),
    )
