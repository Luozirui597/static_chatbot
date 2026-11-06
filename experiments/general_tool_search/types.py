"""Types for the general teaching-history retrieval experiment."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

AnchorType = Literal["formula", "phrase", "cjk", "term"]


@dataclass(frozen=True)
class Anchor:
    kind: AnchorType
    text: str
    normalized: str


@dataclass(frozen=True)
class Decision:
    need_history: bool
    query: str
    visible_source_ids: tuple[int, ...]
    reason: str


@dataclass
class Counts:
    decision_attempts: int = 0
    decision_requests: int = 0
    retrieval_attempts: int = 0
    retrieval_requests: int = 0
    final_attempts: int = 0
    final_requests: int = 0
    decision_http_requests: int = 0
    retrieval_http_requests: int = 0
    final_http_requests: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "decision_attempts": self.decision_attempts,
            "decision_requests": self.decision_requests,
            "retrieval_attempts": self.retrieval_attempts,
            "retrieval_requests": self.retrieval_requests,
            "final_attempts": self.final_attempts,
            "final_requests": self.final_requests,
            "decision_http_requests": self.decision_http_requests,
            "retrieval_http_requests": self.retrieval_http_requests,
            "final_http_requests": self.final_http_requests,
        }


@dataclass
class SearchOutcome:
    status: str
    results: list[dict[str, Any]] = field(default_factory=list)
    truncated: bool = False
    scan_truncated: bool = False
    corrections_truncated: bool = False
    omitted_count: int = 0
    pair_missing: bool = False
    error: str | None = None
    error_code: str | None = None
    synthetic: bool = True

    def as_history(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "results": self.results,
            "truncated": self.truncated,
            "scan_truncated": self.scan_truncated,
            "corrections_truncated": self.corrections_truncated,
            "omitted_count": self.omitted_count,
            "pair_missing": self.pair_missing,
        }


@dataclass
class EngineResult:
    status: str
    answer: str | None = None
    counts: Counts = field(default_factory=Counts)
    decision: Decision | None = None
    decision_reason: str | None = None
    search: SearchOutcome | None = None
    messages: list[dict[str, Any]] = field(default_factory=list)
    visible_source_ids: list[int] = field(default_factory=list)
    retrieved_source_ids: list[int] = field(default_factory=list)
    injected_source_ids: list[int] = field(default_factory=list)
    cited_source_ids: list[int] = field(default_factory=list)
    invalid_source_ids: list[int] = field(default_factory=list)
    error: str | None = None
    error_code: str | None = None
    synthetic: bool = True
