"""Strict structured decision protocol for the general search experiment."""

from __future__ import annotations

import json
from dataclasses import dataclass

from experiments.general_tool_search.types import Decision

REASONS = {
    "ordinary_recall",
    "visible_complete",
    "ambiguous",
    "not_needed",
}
FIELDS = {"need_history", "query", "visible_source_ids", "reason"}
MAX_QUERY_LENGTH = 200


class DecisionError(ValueError):
    """A decision is structurally invalid or not grounded."""


@dataclass(frozen=True)
class DecisionCheck:
    decision: Decision
    visible_ids: tuple[int, ...]


def parse_decision(raw: str, *, visible_ids: tuple[int, ...]) -> DecisionCheck:
    try:
        payload = json.loads(raw)
    except (TypeError, ValueError) as exc:
        raise DecisionError("invalid_json") from exc
    if not isinstance(payload, dict) or set(payload) != FIELDS:
        raise DecisionError("invalid_fields")
    need = payload["need_history"]
    query = payload["query"]
    ids = payload["visible_source_ids"]
    reason = payload["reason"]
    if not isinstance(need, bool):
        raise DecisionError("need_history_not_bool")
    if not isinstance(query, str):
        raise DecisionError("query_not_string")
    if not isinstance(reason, str) or reason not in REASONS:
        raise DecisionError("invalid_reason")
    if not isinstance(ids, list) or not all(
        isinstance(value, int) and not isinstance(value, bool) for value in ids
    ):
        raise DecisionError("visible_ids_not_ints")
    if len(set(ids)) != len(ids):
        raise DecisionError("duplicate_visible_id")
    if any(value not in visible_ids for value in ids):
        raise DecisionError("visible_id_not_visible")
    if need:
        if not query.strip():
            raise DecisionError("missing_query")
        if len(query) > MAX_QUERY_LENGTH:
            raise DecisionError("query_too_long")
        if ids:
            raise DecisionError("retrieval_with_visible_ids")
        if reason not in {"ordinary_recall"}:
            raise DecisionError("retrieval_reason_mismatch")
    else:
        if query:
            raise DecisionError("query_without_retrieval")
        if reason == "visible_complete" and not ids:
            raise DecisionError("visible_complete_without_ids")
        if reason in {"ambiguous", "not_needed"} and ids:
            raise DecisionError("ids_without_visible_complete")
    return DecisionCheck(
        decision=Decision(
            need_history=need,
            query=query.strip(),
            visible_source_ids=tuple(ids),
            reason=reason,
        ),
        visible_ids=tuple(ids),
    )
