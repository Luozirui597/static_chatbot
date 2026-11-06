"""Read-only teacher-history search for the isolated experiment."""

from __future__ import annotations

import asyncio
import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from backend.programmatic_history.protocol import (
    MAX_QUERY_CHARS,
    validate_tool_call,
)
from backend.programmatic_history.types import (
    ToolCall,
    ToolContext,
    ToolExecutionError,
    ToolProtocolError,
    ToolResult,
    ToolTimeout,
)

MAX_SCAN_MESSAGES = 200
MAX_SCAN_CHARS = 50_000
MAX_RESULTS = 5
MAX_CONTENT_CHARS = 1_000
MAX_RESULT_CHARS = 4_000
MAX_QUERY_TOKENS = 6
DEFAULT_TOOL_TIMEOUT_SECONDS = 1.0

_CORRECTION_CUES = (
    "correction",
    "corrected",
    "actually",
    "instead",
    "not",
    "wrong",
    "replace",
    "replaced",
    "no longer",
    "revoke",
    "revoked",
)
_TOKEN_RE = re.compile(r"[a-z0-9_]+")


@dataclass(frozen=True)
class ToolLimits:
    max_scan_messages: int = MAX_SCAN_MESSAGES
    max_scan_chars: int = MAX_SCAN_CHARS
    max_results: int = MAX_RESULTS
    max_content_chars: int = MAX_CONTENT_CHARS
    max_result_chars: int = MAX_RESULT_CHARS
    timeout_seconds: float = DEFAULT_TOOL_TIMEOUT_SECONDS


def _tokens(query: str) -> list[str]:
    unique = list(dict.fromkeys(_TOKEN_RE.findall(query.lower())))
    return unique[:MAX_QUERY_TOKENS]


def _has_correction_cue(content: str) -> str | None:
    lowered = content.lower()
    for cue in _CORRECTION_CUES:
        if re.search(r"\b" + re.escape(cue) + r"\b", lowered):
            return cue
    return None


class ToolExecutor:
    """Execute one validated tool call against a bound session.

    ``connection_factory`` is a test seam.  Production experiments use the
    default sqlite3 connection and a temporary file database only.
    """

    def __init__(
        self,
        database_path: str | Path,
        *,
        limits: ToolLimits | None = None,
        connection_factory: Callable[..., Any] = sqlite3.connect,
    ) -> None:
        self.database_path = str(Path(database_path))
        self.limits = limits or ToolLimits()
        self.connection_factory = connection_factory
        self.last_connection_closed = False
        self.last_scan_rows = 0
        self.last_deadline: float | None = None

    async def execute(
        self,
        call: ToolCall,
        context: ToolContext,
        *,
        deadline: float | None = None,
        before_id: int | None = None,
    ) -> ToolResult:
        try:
            validate_tool_call(call)
        except ToolProtocolError as exc:
            return self._error(call, "invalid_arguments", str(exc))
        if not isinstance(context.session_id, int) or not isinstance(
            context.current_message_id, int
        ):
            return self._error(call, "invalid_context", "invalid_context")
        deadline = deadline or (time.monotonic() + self.limits.timeout_seconds)
        tool_deadline = min(
            deadline,
            time.monotonic() + self.limits.timeout_seconds,
        )
        remaining = tool_deadline - time.monotonic()
        if remaining <= 0:
            return self._error(call, "timeout", "turn_deadline_exceeded")
        self.last_deadline = tool_deadline
        try:
            data = await asyncio.wait_for(
                asyncio.to_thread(
                    self._execute_sync,
                    call,
                    context,
                    tool_deadline,
                    before_id,
                ),
                timeout=remaining,
            )
            return ToolResult(
                tool_call_id=call.id,
                name=call.name,
                content=data,
                is_error=data.get("status") not in {"ok", "empty"},
            )
        except asyncio.TimeoutError:
            return self._error(call, "timeout", "tool_deadline_exceeded")
        except ToolTimeout:
            return self._error(call, "timeout", "tool_deadline_exceeded")
        except ToolProtocolError as exc:
            return self._error(call, "invalid_arguments", str(exc))
        except Exception:
            return self._error(call, "error", "tool_execution_failed")

    def _error(self, call: ToolCall, status: str, code: str) -> ToolResult:
        return ToolResult(
            tool_call_id=call.id,
            name=call.name,
            content={"status": status, "error_code": code, "results": []},
            is_error=True,
        )

    def _execute_sync(
        self,
        call: ToolCall,
        context: ToolContext,
        deadline: float,
        before_id: int | None = None,
    ) -> dict[str, Any]:
        conn = None
        try:
            conn = self.connection_factory(
                self.database_path,
                timeout=min(0.25, max(0.01, deadline - time.monotonic())),
            )
            conn.execute("PRAGMA query_only=ON")

            def progress() -> int:
                return 1 if time.monotonic() >= deadline else 0

            conn.set_progress_handler(progress, 1_000)
            current = conn.execute(
                "SELECT session_id, role FROM messages WHERE id = ?",
                (context.current_message_id,),
            ).fetchone()
            if (
                current is None
                or int(current[0]) != context.session_id
                or current[1] != "user"
            ):
                return {
                    "status": "invalid_context",
                    "error_code": "current_message_mismatch",
                    "results": [],
                }

            limit = min(
                int(call.arguments.get("limit", 3)),
                self.limits.max_results,
            )
            tokens = _tokens(str(call.arguments["query"]))
            if not tokens:
                return {
                    "status": "empty",
                    "error_code": "no_search_tokens",
                    "results": [],
                }

            # SQL limits the candidate rows before Python sees them.  The
            # progress handler and the Python deadline share the same clock.
            rows = conn.execute(
                "SELECT id, content FROM messages "
                "WHERE session_id = ? AND role = 'user' AND id < ? "
                "ORDER BY id DESC LIMIT ?",
                (
                    context.session_id,
                    before_id if before_id is not None else context.current_message_id,
                    self.limits.max_scan_messages,
                ),
            ).fetchall()
            self.last_scan_rows = len(rows)
            scan_truncated = len(rows) >= self.limits.max_scan_messages
            scanned_chars = 0
            scanned_items: list[dict[str, Any]] = []

            for message_id, content in rows:
                if time.monotonic() >= deadline:
                    raise ToolTimeout("tool_deadline_exceeded")
                text = str(content)
                scanned_chars += len(text)
                if scanned_chars > self.limits.max_scan_chars:
                    scan_truncated = True
                    break
                content_tokens = set(_TOKEN_RE.findall(text.lower()))
                matched = [token for token in tokens if token in content_tokens]
                if not matched:
                    continue
                cue = _has_correction_cue(text)
                item: dict[str, Any] = {
                    "message_id": int(message_id),
                    "content": text,
                    "matched_tokens": matched,
                    "kind": "teaching_match",
                }
                if cue:
                    item["_cue"] = cue
                scanned_items.append(item)

            # A correction candidate is only a correction when an earlier,
            # already scanned, non-correction teacher message shares a query
            # token.  Older messages are in the scanned set even though the
            # SQL reads newest-first.
            for item in scanned_items:
                cue = item.get("_cue")
                if not cue:
                    continue
                support = next(
                    (
                        candidate
                        for candidate in scanned_items
                        if candidate.get("_cue") is None
                        and candidate["message_id"] < item["message_id"]
                        and set(candidate["matched_tokens"])
                        & set(item["matched_tokens"])
                    ),
                    None,
                )
                if support is not None:
                    item["kind"] = "correction_candidate"
                    item["correction_cue"] = cue
                    item["support_message_id"] = support["message_id"]
            for item in scanned_items:
                item.pop("_cue", None)

            corrections = [
                item for item in scanned_items
                if item["kind"] == "correction_candidate"
            ]
            teachings = [
                item for item in scanned_items
                if item["kind"] == "teaching_match"
            ]

            # Allocate priority slots and the total content budget before
            # ordering the visible result.  Corrections are newest-first and
            # may use up to two reserved slots; teachings use match rank.
            corrections.sort(key=lambda item: item["message_id"], reverse=True)
            selected_corrections = corrections[: min(2, limit)]
            selected: list[dict[str, Any]] = []
            total_chars = 0
            budget_truncated = False

            for item in selected_corrections:
                remaining = self.limits.max_result_chars - total_chars
                if remaining <= 0:
                    budget_truncated = True
                    continue
                content = item["content"]
                content_truncated = len(content) > self.limits.max_content_chars
                if content_truncated:
                    content = content[: self.limits.max_content_chars]
                if len(content) > remaining:
                    content = content[:remaining]
                    content_truncated = True
                if not content:
                    budget_truncated = True
                    continue
                total_chars += len(content)
                selected.append(
                    {
                        **item,
                        "content": content,
                        "truncated": content_truncated,
                    }
                )

            teachings.sort(
                key=lambda item: (-len(item["matched_tokens"]), item["message_id"])
            )
            for item in teachings:
                if len(selected) >= limit:
                    break
                remaining = self.limits.max_result_chars - total_chars
                if remaining <= 0:
                    budget_truncated = True
                    break
                content = item["content"]
                content_truncated = len(content) > self.limits.max_content_chars
                if content_truncated:
                    content = content[: self.limits.max_content_chars]
                if len(content) > remaining:
                    budget_truncated = True
                    continue
                if not content:
                    continue
                total_chars += len(content)
                selected.append(
                    {
                        **item,
                        "content": content,
                        "truncated": content_truncated,
                    }
                )

            selected.sort(key=lambda item: item["message_id"])
            results: list[dict[str, Any]] = []
            for item in selected:
                order = conn.execute(
                    "SELECT COUNT(*) FROM messages "
                    "WHERE session_id = ? AND role = 'user' AND id <= ?",
                    (context.session_id, item["message_id"]),
                ).fetchone()[0]
                result_item = {
                    "message_id": item["message_id"],
                    "order": int(order),
                    "content": item["content"],
                    "kind": item["kind"],
                    "truncated": item["truncated"],
                    **(
                        {"correction_cue": item["correction_cue"]}
                        if "correction_cue" in item
                        else {}
                    ),
                    **(
                        {"support_message_id": item["support_message_id"]}
                        if "support_message_id" in item
                        else {}
                    ),
                }
                results.append(result_item)

            selected_ids = {
                item["message_id"] for item in selected
            }
            selected_by_id = {item["message_id"]: item for item in selected}
            omitted = sum(
                1
                for item in scanned_items
                if item["message_id"] not in selected_ids
            )
            corrections_truncated = any(
                item["message_id"] not in selected_by_id
                or selected_by_id[item["message_id"]]["truncated"]
                for item in corrections
            )
            truncated = budget_truncated or any(
                item["truncated"] for item in selected
            )
            if not results:
                return {
                    "status": "empty",
                    "query": call.arguments["query"],
                    "results": [],
                    "truncated": truncated,
                    "scan_truncated": scan_truncated,
                    "corrections_truncated": corrections_truncated,
                    "omitted_count": omitted,
                }
            return {
                "status": "ok",
                "query": call.arguments["query"],
                "results": results,
                "truncated": truncated,
                "scan_truncated": scan_truncated,
                "corrections_truncated": corrections_truncated,
                "omitted_count": omitted,
            }
        except ToolTimeout:
            raise
        except sqlite3.OperationalError as exc:
            if time.monotonic() >= deadline or "interrupted" in str(exc).lower():
                raise ToolTimeout("tool_deadline_exceeded") from exc
            raise ToolExecutionError("tool_query_failed") from exc
        finally:
            if conn is not None:
                try:
                    conn.set_progress_handler(None, 0)
                except Exception:
                    pass
                conn.close()
                self.last_connection_closed = True
