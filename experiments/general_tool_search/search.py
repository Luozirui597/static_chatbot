"""Read-only anchor search over teacher messages before the visible window."""

from __future__ import annotations

import asyncio
import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from experiments.general_tool_search.anchors import (
    anchor_matches,
    anchor_weight,
)
from experiments.general_tool_search.types import Anchor, SearchOutcome

MAX_SCAN_MESSAGES = 200
MAX_SCAN_CHARS = 50_000
MAX_RESULTS = 3
MAX_CONTENT_CHARS = 1_000
MAX_RESULT_CHARS = 4_000
DEFAULT_TIMEOUT_SECONDS = 1.0

_CORRECTION_CUE = re.compile(
    r"\b(correction|corrected|actually|instead|wrong|replace|replaced|"
    r"no longer|revoke|revoked|not allowed|exception)\b",
    re.IGNORECASE,
)
_CORRECTION_WORD = re.compile(r"\b(correction|corrected|actually|instead|wrong)\b", re.I)
_EXCEPTION_WORD = re.compile(r"\b(exception|not allowed)\b", re.I)


@dataclass(frozen=True)
class SearchLimits:
    max_scan_messages: int = MAX_SCAN_MESSAGES
    max_scan_chars: int = MAX_SCAN_CHARS
    max_results: int = MAX_RESULTS
    max_content_chars: int = MAX_CONTENT_CHARS
    max_result_chars: int = MAX_RESULT_CHARS
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS


class SearchTimeout(RuntimeError):
    pass


class SearchError(RuntimeError):
    pass


class AnchorSearch:
    def __init__(
        self,
        database_path: str | Path,
        *,
        limits: SearchLimits | None = None,
        connection_factory: Callable[..., Any] = sqlite3.connect,
    ) -> None:
        self.database_path = str(Path(database_path))
        self.limits = limits or SearchLimits()
        self.connection_factory = connection_factory
        self.last_connection_closed = False
        self.last_deadline: float | None = None

    async def search(
        self,
        *,
        session_id: int,
        current_message_id: int,
        visible_start_id: int,
        anchors: list[Anchor],
        question: str,
        deadline: float | None = None,
    ) -> SearchOutcome:
        if not anchors:
            return SearchOutcome(status="empty", scan_truncated=True)
        started = time.monotonic()
        effective_deadline = min(
            deadline if deadline is not None else started + self.limits.timeout_seconds,
            started + self.limits.timeout_seconds,
        )
        self.last_deadline = effective_deadline
        self.last_connection_closed = False
        if effective_deadline <= started:
            raise SearchTimeout("search_deadline_exceeded")
        try:
            data = await asyncio.wait_for(
                asyncio.to_thread(
                    self._search_sync,
                    session_id,
                    current_message_id,
                    visible_start_id,
                    anchors,
                    question,
                    effective_deadline,
                ),
                timeout=effective_deadline - started,
            )
            return SearchOutcome(**data)
        except asyncio.TimeoutError as exc:
            raise SearchTimeout("search_deadline_exceeded") from exc

    def _search_sync(
        self,
        session_id: int,
        current_message_id: int,
        visible_start_id: int,
        anchors: list[Anchor],
        question: str,
        deadline: float,
    ) -> dict[str, Any]:
        conn = None
        try:
            uri = f"file:{self.database_path}?mode=ro"
            conn = self.connection_factory(uri, uri=True, timeout=0.25)
            conn.execute("PRAGMA query_only=ON")

            def progress() -> int:
                return 1 if time.monotonic() >= deadline else 0

            conn.set_progress_handler(progress, 1_000)
            current = conn.execute(
                "SELECT session_id, role FROM messages WHERE id = ?",
                (current_message_id,),
            ).fetchone()
            if (
                current is None
                or int(current[0]) != session_id
                or current[1] != "user"
            ):
                raise SearchError("current_message_mismatch")
            rows = conn.execute(
                "SELECT id, content FROM messages "
                "WHERE session_id = ? AND role = 'user' AND id < ? "
                "ORDER BY id DESC LIMIT ?",
                (session_id, visible_start_id, self.limits.max_scan_messages),
            ).fetchall()
            scan_truncated = len(rows) >= self.limits.max_scan_messages
            scanned_chars = 0
            why_requested = bool(
                re.search(r"\b(?:why|reason|because)\b|为什么|理由", question, re.I)
            )
            items: list[dict[str, Any]] = []
            for message_id, content in rows:
                if time.monotonic() >= deadline:
                    raise SearchTimeout("search_deadline_exceeded")
                text = str(content)
                scanned_chars += len(text)
                if scanned_chars > self.limits.max_scan_chars:
                    scan_truncated = True
                    break
                matched = [
                    anchor for anchor in anchors
                    if anchor_matches(text, anchor)
                ]
                if not matched:
                    continue
                cue = bool(_CORRECTION_CUE.search(text))
                score = sum(anchor_weight(a) for a in matched)
                if why_requested and re.search(
                    r"\b(?:reason|because|so that|why)\b|理由|因为",
                    text,
                    re.I,
                ):
                    score += 45
                item = {
                    "message_id": int(message_id),
                    "content": text,
                    "matched": matched,
                    "score": score,
                    "cue": cue,
                    "kind": "teaching_match",
                }
                items.append(item)

            for item in items:
                if not item["cue"]:
                    continue
                support = next(
                    (
                        other for other in items
                        if other["message_id"] < item["message_id"]
                        and not other["cue"]
                        and any(
                            a.kind in {"formula", "phrase", "cjk", "term"}
                            and a.normalized in {
                                b.normalized for b in item["matched"]
                            }
                            for a in other["matched"]
                        )
                    ),
                    None,
                )
                if support is not None:
                    item["kind"] = (
                        "exception_candidate"
                        if _EXCEPTION_WORD.search(item["content"])
                        else "correction_candidate"
                    )
                    item["support_message_id"] = support["message_id"]

            corrections = [
                item for item in items
                if item["kind"] in {"correction_candidate", "exception_candidate"}
            ]
            other = [item for item in items if item["kind"] == "teaching_match"]
            compare = bool(
                re.search(r"\b(original|before|previous|other|replacement)\b", question, re.I)
                and re.search(r"\b(current|replacement|correct|new|version)\b", question, re.I)
            )
            selected: list[dict[str, Any]] = []
            remaining = list(items)
            pair_found = False
            if compare:
                for correction in sorted(
                    corrections, key=lambda i: (i["score"], i["message_id"]), reverse=True
                ):
                    partner = next(
                        (
                            teaching for teaching in other
                            if teaching["message_id"] < correction["message_id"]
                            and {a.normalized for a in teaching["matched"]}
                            & {a.normalized for a in correction["matched"]}
                        ),
                        None,
                    )
                    if partner is not None:
                        selected = [correction, partner]
                        pair_found = True
                        remaining = [
                            i for i in remaining
                            if i["message_id"] not in {
                                correction["message_id"], partner["message_id"]
                            }
                        ]
                        break
            if not selected:
                for correction in sorted(
                    corrections, key=lambda i: i["message_id"], reverse=True
                )[:2]:
                    selected.append(correction)
                    remaining = [
                        i for i in remaining if i["message_id"] != correction["message_id"]
                    ]
            for item in sorted(
                remaining,
                key=lambda i: (-i["score"], -i["message_id"]),
            ):
                if len(selected) >= self.limits.max_results:
                    break
                selected.append(item)

            # Budget in priority order: corrections first, then teaching
            # matches.  Only the visible result is sorted by teacher order.
            priority = sorted(
                selected,
                key=lambda item: (
                    0 if item["kind"] in {
                        "correction_candidate", "exception_candidate"
                    } else 1,
                    -item["score"],
                    -item["message_id"],
                ),
            )
            total = 0
            results: list[dict[str, Any]] = []
            content_truncated = False
            budget_truncated = False
            for item in priority:
                content = item["content"]
                truncated = len(content) > self.limits.max_content_chars
                if truncated:
                    content = content[: self.limits.max_content_chars]
                    content_truncated = True
                remaining_budget = self.limits.max_result_chars - total
                if remaining_budget <= 0:
                    budget_truncated = True
                    continue
                is_correction = item["kind"] in {
                    "correction_candidate", "exception_candidate"
                }
                if len(content) > remaining_budget:
                    if is_correction:
                        content = content[:remaining_budget]
                        truncated = True
                        content_truncated = True
                    else:
                        budget_truncated = True
                        continue
                if not content:
                    budget_truncated = True
                    continue
                total += len(content)
                order = conn.execute(
                    "SELECT COUNT(*) FROM messages "
                    "WHERE session_id = ? AND role = 'user' AND id <= ?",
                    (session_id, item["message_id"]),
                ).fetchone()[0]
                results.append(
                    {
                        "message_id": item["message_id"],
                        "order": int(order),
                        "content": content,
                        "kind": item["kind"],
                        "truncated": truncated,
                        **(
                            {"support_message_id": item["support_message_id"]}
                            if "support_message_id" in item
                            else {}
                        ),
                    }
                )
            results.sort(key=lambda item: item["message_id"])
            selected_ids = {item["message_id"] for item in results}
            selected_by_id = {item["message_id"]: item for item in results}
            omitted = sum(
                1 for item in items if item["message_id"] not in selected_ids
            )
            corrections_truncated = any(
                item["message_id"] not in selected_by_id
                or selected_by_id[item["message_id"]].get("truncated", False)
                for item in corrections
            )
            return {
                "status": "ok" if results else "empty",
                "results": results,
                "truncated": content_truncated or budget_truncated,
                "scan_truncated": scan_truncated,
                "corrections_truncated": corrections_truncated,
                "omitted_count": omitted,
                "pair_missing": compare and not pair_found,
            }
        except SearchTimeout:
            raise
        except SearchError:
            raise
        except sqlite3.OperationalError as exc:
            if time.monotonic() >= deadline or "interrupted" in str(exc).lower():
                raise SearchTimeout("search_deadline_exceeded") from exc
            raise SearchError("search_failed") from exc
        finally:
            if conn is not None:
                try:
                    conn.set_progress_handler(None, 0)
                except Exception:
                    pass
                conn.close()
                self.last_connection_closed = True
