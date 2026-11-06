"""Strict structured decision protocol for the general search experiment."""

from __future__ import annotations

import json
from dataclasses import dataclass

from backend.general_history.types import Decision

REASONS = {
    "visible_complete",
    "missing_fact",
    "missing_explanation",
    "ambiguous",
    "not_needed",
}
FIELDS = {"need_history", "query", "visible_source_ids", "reason"}
MAX_QUERY_LENGTH = 200


class DecisionError(ValueError):
    """A decision is structurally invalid or not grounded."""

    def __init__(
        self,
        code: str,
        *,
        reason: str | None = None,
        has_query: bool | None = None,
        visible_id_count: int | None = None,
    ) -> None:
        super().__init__(code)
        self.code = code
        self.reason = reason
        self.has_query = has_query
        self.visible_id_count = visible_id_count


@dataclass(frozen=True)
class DecisionCheck:
    decision: Decision
    visible_ids: tuple[int, ...]


def parse_decision(raw: str, *, visible_ids: tuple[int, ...]) -> DecisionCheck:
    def fail(
        code: str,
        *,
        reason: str | None = None,
        has_query: bool | None = None,
        visible_id_count: int | None = None,
    ):
        raise DecisionError(
            code,
            reason=reason,
            has_query=has_query,
            visible_id_count=visible_id_count,
        )

    try:
        payload = json.loads(raw)
    except (TypeError, ValueError):
        fail("invalid_json")
    if not isinstance(payload, dict) or set(payload) != FIELDS:
        fail("invalid_fields")
    need = payload["need_history"]
    query = payload["query"]
    ids = payload["visible_source_ids"]
    raw_reason = payload["reason"]
    reason = raw_reason if isinstance(raw_reason, str) and raw_reason in REASONS else None
    has_query = isinstance(query, str) and bool(query.strip())
    visible_id_count = len(ids) if isinstance(ids, list) else None

    if not isinstance(need, bool):
        fail("need_history_not_bool", reason=reason, has_query=has_query, visible_id_count=visible_id_count)
    if not isinstance(query, str):
        fail("query_not_string", reason=reason, has_query=has_query, visible_id_count=visible_id_count)
    if not isinstance(raw_reason, str) or reason is None:
        fail("invalid_reason", reason=None, has_query=has_query, visible_id_count=visible_id_count)
    if not isinstance(ids, list) or not all(
        isinstance(value, int) and not isinstance(value, bool) for value in ids
    ):
        fail("visible_ids_not_ints", reason=reason, has_query=has_query, visible_id_count=visible_id_count)

    # Check need_history/reason compatibility before applying the reason's ID
    # shape.  This keeps the failure code and the claimed reason distinct.
    if need and reason not in {"missing_fact", "missing_explanation"}:
        fail("retrieval_reason_mismatch", reason=reason, has_query=has_query, visible_id_count=visible_id_count)
    if not need and reason not in {"visible_complete", "ambiguous", "not_needed"}:
        fail("reason_requires_retrieval", reason=reason, has_query=has_query, visible_id_count=visible_id_count)

    if len(set(ids)) != len(ids):
        fail("duplicate_visible_id", reason=reason, has_query=has_query, visible_id_count=visible_id_count)
    if any(value not in visible_ids for value in ids):
        fail("visible_id_not_visible", reason=reason, has_query=has_query, visible_id_count=visible_id_count)

    if need:
        if not query.strip():
            fail("missing_query", reason=reason, has_query=has_query, visible_id_count=visible_id_count)
        if len(query) > MAX_QUERY_LENGTH:
            fail("query_too_long", reason=reason, has_query=has_query, visible_id_count=visible_id_count)
        if reason == "missing_fact" and ids:
            fail("missing_fact_with_visible_ids", reason=reason, has_query=has_query, visible_id_count=visible_id_count)
    else:
        if query:
            fail("query_without_retrieval", reason=reason, has_query=has_query, visible_id_count=visible_id_count)
        if reason == "visible_complete" and not ids:
            fail("visible_complete_without_ids", reason=reason, has_query=has_query, visible_id_count=visible_id_count)
        if reason in {"ambiguous", "not_needed"} and ids:
            fail("ids_without_visible_complete", reason=reason, has_query=has_query, visible_id_count=visible_id_count)

    return DecisionCheck(
        decision=Decision(
            need_history=need,
            query=query.strip(),
            visible_source_ids=tuple(ids),
            reason=reason,
        ),
        visible_ids=tuple(ids),
    )
