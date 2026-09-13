"""Strict, pure parser for history-review LLM output.

The parser never repairs, extracts, sorts, deduplicates or fills output.
It performs only validation and normalization explicitly described by the
history-review output contract.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

HISTORY_REVIEW_MAX_RAW_OUTPUT_CHARS = 16000
HISTORY_REVIEW_MAX_FINDINGS = 50
HISTORY_REVIEW_MAX_SUMMARY_CHARS = 4000
HISTORY_REVIEW_MAX_COVERAGE_NOTE_CHARS = 2000
HISTORY_REVIEW_MAX_CLAIM_TEXT_CHARS = 4000
HISTORY_REVIEW_MAX_CORRECTION_TEXT_CHARS = 4000
HISTORY_REVIEW_MAX_EXPLANATION_TEXT_CHARS = 4000

_VERDICTS = frozenset(
    {"correct", "incorrect", "uncertain", "not_a_claim"}
)
_TOP_LEVEL_KEYS = frozenset({"summary", "coverage_note", "findings"})
_FINDING_KEYS = frozenset(
    {
        "source_message_id",
        "verdict",
        "claim_text",
        "correction_text",
        "explanation_text",
    }
)


class HistoryReviewOutputError(ValueError):
    """Base class for safe, stable history-review output errors."""

    code = "history_review_output_error"
    default_message = "History review output is invalid."

    def __init__(self) -> None:
        super().__init__(self.default_message)


class HistoryReviewOutputTooLargeError(HistoryReviewOutputError):
    code = "history_review_output_too_large"
    default_message = "History review output exceeds configured limits."


class HistoryReviewJsonSyntaxError(HistoryReviewOutputError):
    code = "history_review_json_syntax_error"
    default_message = "History review output is not valid JSON."


class HistoryReviewJsonStructureError(HistoryReviewOutputError):
    code = "history_review_json_structure_error"
    default_message = "History review output has an invalid JSON structure."


class HistoryReviewJsonSemanticError(HistoryReviewOutputError):
    code = "history_review_json_semantic_error"
    default_message = "History review output violates the semantic contract."


class HistoryReviewSourceReferenceInvalidError(HistoryReviewOutputError):
    code = "history_review_source_reference_invalid"
    default_message = "History review output references an invalid source."


class HistoryReviewSourceUncoveredError(HistoryReviewOutputError):
    code = "history_review_source_uncovered"
    default_message = "History review output does not cover every source."


class _DuplicateJsonKeyError(Exception):
    """Internal marker raised by the duplicate-key object hook."""


class _InvalidJsonConstantError(Exception):
    """Internal marker raised for NaN and infinity constants."""


@dataclass(frozen=True)
class ParsedHistoryReviewFinding:
    source_message_id: int
    verdict: Literal["correct", "incorrect", "uncertain", "not_a_claim"]
    claim_text: str
    correction_text: str | None
    explanation_text: str | None


@dataclass(frozen=True)
class ParsedHistoryReviewOutput:
    summary: str
    coverage_note: str | None
    findings: tuple[ParsedHistoryReviewFinding, ...]


def _reject_constant(_value: str) -> None:
    raise _InvalidJsonConstantError()


def _object_pairs_hook(pairs: list[tuple[str, object]]) -> dict:
    result: dict = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJsonKeyError()
        result[key] = value
    return result


def _load_json(raw_output: str) -> object:
    try:
        return json.loads(
            raw_output,
            object_pairs_hook=_object_pairs_hook,
            parse_constant=_reject_constant,
        )
    except _DuplicateJsonKeyError:
        raise HistoryReviewJsonSyntaxError() from None
    except _InvalidJsonConstantError:
        raise HistoryReviewJsonSyntaxError() from None
    except json.JSONDecodeError:
        raise HistoryReviewJsonSyntaxError() from None


def _validate_source_seq_by_id(
    source_seq_by_id: Mapping[int, int],
) -> dict[int, int]:
    if not isinstance(source_seq_by_id, Mapping):
        raise TypeError("source_seq_by_id must be a Mapping")

    result: dict[int, int] = {}
    for key, value in source_seq_by_id.items():
        if isinstance(key, bool) or not isinstance(key, int) or key < 1:
            raise ValueError(
                "source_seq_by_id keys must be positive integers"
            )
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 1
        ):
            raise ValueError(
                "source_seq_by_id values must be positive integers"
            )
        result[key] = value

    if not result:
        raise ValueError("source_seq_by_id must not be empty")

    expected = list(range(1, len(result) + 1))
    if sorted(result.values()) != expected:
        raise ValueError(
            "source_seq_by_id values must be exactly 1..N"
        )
    return result


def _require_exact_keys(
    value: object,
    expected: frozenset[str],
) -> dict:
    if not isinstance(value, dict):
        raise HistoryReviewJsonStructureError()
    if set(value.keys()) != expected:
        raise HistoryReviewJsonStructureError()
    return value


def _require_positive_source_id(value: object) -> int:
    if type(value) is not int:
        raise HistoryReviewJsonStructureError()
    if value < 1:
        raise HistoryReviewSourceReferenceInvalidError()
    return value


def _normalize_required_text(
    value: object,
    *,
    max_chars: int,
) -> str:
    if not isinstance(value, str):
        raise HistoryReviewJsonStructureError()
    stripped = value.strip()
    if not stripped:
        raise HistoryReviewJsonSemanticError()
    if len(stripped) > max_chars:
        raise HistoryReviewOutputTooLargeError()
    return stripped


def _normalize_optional_text(
    value: object,
    *,
    max_chars: int,
) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise HistoryReviewJsonStructureError()
    stripped = value.strip()
    if not stripped:
        raise HistoryReviewJsonSemanticError()
    if len(stripped) > max_chars:
        raise HistoryReviewOutputTooLargeError()
    return stripped


def parse_history_review_output(
    *,
    raw_output: str,
    source_seq_by_id: Mapping[int, int],
) -> ParsedHistoryReviewOutput:
    """Parse one complete history-review JSON object strictly."""
    if not isinstance(raw_output, str):
        raise TypeError("raw_output must be a string")
    if len(raw_output) > HISTORY_REVIEW_MAX_RAW_OUTPUT_CHARS:
        raise HistoryReviewOutputTooLargeError()

    source_seqs = _validate_source_seq_by_id(source_seq_by_id)

    parsed_json = _load_json(raw_output)
    data = _require_exact_keys(parsed_json, _TOP_LEVEL_KEYS)

    summary = _normalize_required_text(
        data["summary"],
        max_chars=HISTORY_REVIEW_MAX_SUMMARY_CHARS,
    )
    coverage_note = _normalize_optional_text(
        data["coverage_note"],
        max_chars=HISTORY_REVIEW_MAX_COVERAGE_NOTE_CHARS,
    )

    findings_raw = data["findings"]
    if not isinstance(findings_raw, list):
        raise HistoryReviewJsonStructureError()
    if len(findings_raw) == 0:
        raise HistoryReviewJsonStructureError()
    if len(findings_raw) > HISTORY_REVIEW_MAX_FINDINGS:
        raise HistoryReviewOutputTooLargeError()

    previous_source_seq = 0
    seen_source_seqs: set[int] = set()
    findings: list[ParsedHistoryReviewFinding] = []

    for item in findings_raw:
        finding = _require_exact_keys(item, _FINDING_KEYS)

        source_message_id = _require_positive_source_id(
            finding["source_message_id"]
        )
        if source_message_id not in source_seqs:
            raise HistoryReviewSourceReferenceInvalidError()
        source_seq = source_seqs[source_message_id]
        if source_seq < previous_source_seq:
            raise HistoryReviewJsonSemanticError()
        previous_source_seq = source_seq
        seen_source_seqs.add(source_seq)

        verdict = finding["verdict"]
        if not isinstance(verdict, str):
            raise HistoryReviewJsonStructureError()
        if verdict not in _VERDICTS:
            raise HistoryReviewJsonSemanticError()

        claim_text = _normalize_required_text(
            finding["claim_text"],
            max_chars=HISTORY_REVIEW_MAX_CLAIM_TEXT_CHARS,
        )
        correction_text = _normalize_optional_text(
            finding["correction_text"],
            max_chars=HISTORY_REVIEW_MAX_CORRECTION_TEXT_CHARS,
        )
        explanation_text = _normalize_optional_text(
            finding["explanation_text"],
            max_chars=HISTORY_REVIEW_MAX_EXPLANATION_TEXT_CHARS,
        )

        if verdict == "correct":
            if correction_text is not None:
                raise HistoryReviewJsonSemanticError()
        elif verdict == "incorrect":
            if correction_text is None or explanation_text is None:
                raise HistoryReviewJsonSemanticError()
        elif verdict == "uncertain":
            if correction_text is not None or explanation_text is None:
                raise HistoryReviewJsonSemanticError()
        else:  # not_a_claim
            if correction_text is not None:
                raise HistoryReviewJsonSemanticError()

        findings.append(
            ParsedHistoryReviewFinding(
                source_message_id=source_message_id,
                verdict=verdict,  # type: ignore[arg-type]
                claim_text=claim_text,
                correction_text=correction_text,
                explanation_text=explanation_text,
            )
        )

    if seen_source_seqs != set(source_seqs.values()):
        raise HistoryReviewSourceUncoveredError()

    return ParsedHistoryReviewOutput(
        summary=summary,
        coverage_note=coverage_note,
        findings=tuple(findings),
    )


def truncate_raw_output(
    *,
    raw_output: str,
    max_chars: int = HISTORY_REVIEW_MAX_RAW_OUTPUT_CHARS,
) -> tuple[str, bool]:
    """Return a deterministic prefix and a truncation flag."""
    if not isinstance(raw_output, str):
        raise TypeError("raw_output must be a string")
    if isinstance(max_chars, bool) or not isinstance(max_chars, int):
        raise TypeError("max_chars must be a positive integer")
    if max_chars < 1:
        raise ValueError("max_chars must be a positive integer")

    if len(raw_output) <= max_chars:
        return raw_output, False
    return raw_output[:max_chars], True
