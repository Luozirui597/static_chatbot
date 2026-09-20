"""Strict-parser tests for history-review output."""

import json
from dataclasses import FrozenInstanceError

import pytest

from backend.history_review_parser import (
    HISTORY_REVIEW_MAX_RAW_OUTPUT_CHARS,
    HistoryReviewJsonSemanticError,
    HistoryReviewJsonStructureError,
    HistoryReviewJsonSyntaxError,
    HistoryReviewOutputError,
    HistoryReviewOutputTooLargeError,
    HistoryReviewSourceReferenceInvalidError,
    HistoryReviewSourceUncoveredError,
    parse_history_review_output,
    truncate_raw_output,
)


def _finding(
    source_message_id,
    verdict,
    claim_text="claim",
    correction_text=None,
    explanation_text=None,
):
    return {
        "source_message_id": source_message_id,
        "verdict": verdict,
        "claim_text": claim_text,
        "correction_text": correction_text,
        "explanation_text": explanation_text,
    }


def _raw(
    *,
    summary="summary",
    coverage_note=None,
    findings=None,
):
    if findings is None:
        findings = [_finding(101, "correct")]
    return json.dumps(
        {
            "summary": summary,
            "coverage_note": coverage_note,
            "findings": findings,
        },
        ensure_ascii=False,
    )


def _assert_code(exc_info, expected_code):
    assert exc_info.value.code == expected_code


def test_minimal_valid_output_is_stripped():
    result = parse_history_review_output(
        raw_output=_raw(
            summary="  summary  ",
            findings=[
                _finding(
                    101,
                    "correct",
                    claim_text="  claim text  ",
                )
            ],
        ),
        source_seq_by_id={101: 1},
    )

    assert result.summary == "summary"
    assert result.coverage_note is None
    assert len(result.findings) == 1
    assert result.findings[0].source_message_id == 101
    assert result.findings[0].claim_text == "claim text"
    assert result.findings[0].correction_text is None
    assert result.findings[0].explanation_text is None


def test_multiple_sources_and_same_source_multiple_findings():
    result = parse_history_review_output(
        raw_output=_raw(
            findings=[
                _finding(10, "correct"),
                _finding(10, "uncertain", explanation_text=" unsure "),
                _finding(20, "incorrect", correction_text=" fix ", explanation_text=" why "),
            ]
        ),
        source_seq_by_id={10: 1, 20: 2},
    )

    assert [f.source_message_id for f in result.findings] == [10, 10, 20]
    assert result.findings[1].explanation_text == "unsure"
    assert result.findings[2].correction_text == "fix"
    assert result.findings[2].explanation_text == "why"


def test_multi_claim_source_followed_by_other_sources_is_accepted():
    result = parse_history_review_output(
        raw_output=_raw(
            findings=[
                _finding(
                    101,
                    "incorrect",
                    claim_text="claim one",
                    correction_text="correction one",
                    explanation_text="explanation one",
                ),
                _finding(
                    101,
                    "correct",
                    claim_text="claim two",
                ),
                _finding(
                    102,
                    "incorrect",
                    claim_text="claim three",
                    correction_text="correction three",
                    explanation_text="explanation three",
                ),
                _finding(103, "not_a_claim", claim_text="no factual claim"),
            ]
        ),
        source_seq_by_id={101: 1, 102: 2, 103: 3},
    )

    assert [finding.source_message_id for finding in result.findings] == [
        101,
        101,
        102,
        103,
    ]
    assert [finding.verdict for finding in result.findings] == [
        "incorrect",
        "correct",
        "incorrect",
        "not_a_claim",
    ]


def test_all_four_verdicts_are_accepted():
    result = parse_history_review_output(
        raw_output=_raw(
            coverage_note="  covered  ",
            findings=[
                _finding(1, "correct"),
                _finding(2, "incorrect", correction_text="c", explanation_text="e"),
                _finding(3, "uncertain", explanation_text="e"),
                _finding(4, "not_a_claim"),
            ],
        ),
        source_seq_by_id={1: 1, 2: 2, 3: 3, 4: 4},
    )

    assert result.coverage_note == "covered"
    assert [f.verdict for f in result.findings] == [
        "correct",
        "incorrect",
        "uncertain",
        "not_a_claim",
    ]


@pytest.mark.parametrize(
    ("verdict", "correction", "explanation", "should_pass"),
    [
        ("correct", None, None, True),
        ("correct", None, "explanation", True),
        ("correct", "correction", None, False),
        ("incorrect", "correction", "explanation", True),
        ("incorrect", None, "explanation", False),
        ("incorrect", "correction", None, False),
        ("uncertain", None, "explanation", True),
        ("uncertain", None, None, False),
        ("uncertain", "correction", "explanation", False),
        ("not_a_claim", None, None, True),
        ("not_a_claim", None, "explanation", True),
        ("not_a_claim", "correction", None, False),
    ],
)
def test_verdict_text_combinations(verdict, correction, explanation, should_pass):
    raw = _raw(
        findings=[
            _finding(
                101,
                verdict,
                correction_text=correction,
                explanation_text=explanation,
            )
        ]
    )
    if should_pass:
        result = parse_history_review_output(
            raw_output=raw,
            source_seq_by_id={101: 1},
        )
        assert result.findings[0].verdict == verdict
    else:
        with pytest.raises(HistoryReviewJsonSemanticError) as exc_info:
            parse_history_review_output(
                raw_output=raw,
                source_seq_by_id={101: 1},
            )
        _assert_code(exc_info, "history_review_json_semantic_error")


def test_non_string_raw_output_is_rejected():
    with pytest.raises(TypeError):
        parse_history_review_output(
            raw_output=123,  # type: ignore[arg-type]
            source_seq_by_id={101: 1},
        )


def test_raw_output_over_limit_is_rejected_without_parsing():
    with pytest.raises(HistoryReviewOutputTooLargeError) as exc_info:
        parse_history_review_output(
            raw_output="x" * (HISTORY_REVIEW_MAX_RAW_OUTPUT_CHARS + 1),
            source_seq_by_id={101: 1},
        )
    _assert_code(exc_info, "history_review_output_too_large")


def test_raw_output_at_limit_is_parsed_not_truncated():
    valid = _raw(findings=[_finding(101, "correct")])
    padded = valid + " " * (HISTORY_REVIEW_MAX_RAW_OUTPUT_CHARS - len(valid))
    assert len(padded) == HISTORY_REVIEW_MAX_RAW_OUTPUT_CHARS

    result = parse_history_review_output(
        raw_output=padded,
        source_seq_by_id={101: 1},
    )
    assert result.summary == "summary"


@pytest.mark.parametrize(
    "raw_output",
    [
        "not json",
        "```json\n"
        '{"summary":"s","coverage_note":null,"findings":[]}\n'
        "```",
        "prefix " + _raw(),
        _raw() + " suffix",
        '{"summary":"s"}',
    ],
)
def test_non_json_shapes_are_syntax_errors(raw_output):
    with pytest.raises(HistoryReviewOutputError) as exc_info:
        parse_history_review_output(
            raw_output=raw_output,
            source_seq_by_id={101: 1},
        )
    assert exc_info.value.code in {
        "history_review_json_syntax_error",
        "history_review_json_structure_error",
    }


@pytest.mark.parametrize("raw_output", ["null", "[]", '"text"', "123", "true"])
def test_top_level_must_be_object(raw_output):
    with pytest.raises(HistoryReviewJsonStructureError) as exc_info:
        parse_history_review_output(
            raw_output=raw_output,
            source_seq_by_id={101: 1},
        )
    _assert_code(exc_info, "history_review_json_structure_error")


def test_duplicate_top_level_key_is_syntax_error():
    raw = (
        '{"summary":"one","summary":"two","coverage_note":null,'
        '"findings":[{"source_message_id":101,"verdict":"correct",'
        '"claim_text":"claim","correction_text":null,'
        '"explanation_text":null}]}'
    )
    with pytest.raises(HistoryReviewJsonSyntaxError) as exc_info:
        parse_history_review_output(
            raw_output=raw,
            source_seq_by_id={101: 1},
        )
    _assert_code(exc_info, "history_review_json_syntax_error")


def test_duplicate_nested_finding_key_is_syntax_error():
    raw = (
        '{"summary":"s","coverage_note":null,"findings":[{'
        '"source_message_id":101,"source_message_id":101,'
        '"verdict":"correct","claim_text":"claim",'
        '"correction_text":null,"explanation_text":null}]}'
    )
    with pytest.raises(HistoryReviewJsonSyntaxError) as exc_info:
        parse_history_review_output(
            raw_output=raw,
            source_seq_by_id={101: 1},
        )
    _assert_code(exc_info, "history_review_json_syntax_error")


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_non_finite_constants_are_syntax_errors(constant):
    raw = (
        '{"summary":'
        + constant
        + ',"coverage_note":null,"findings":['
        '{"source_message_id":101,"verdict":"correct",'
        '"claim_text":"claim","correction_text":null,'
        '"explanation_text":null}]}'
    )
    with pytest.raises(HistoryReviewJsonSyntaxError) as exc_info:
        parse_history_review_output(
            raw_output=raw,
            source_seq_by_id={101: 1},
        )
    _assert_code(exc_info, "history_review_json_syntax_error")


@pytest.mark.parametrize(
    "payload",
    [
        {
            "summary": "s",
            "coverage_note": None,
            "findings": [_finding(101, "correct")],
            "extra": 1,
        },
        {"summary": "s", "findings": [_finding(101, "correct")]},
        {"coverage_note": None, "findings": [_finding(101, "correct")]},
    ],
)
def test_top_level_unknown_or_missing_fields_are_structure_errors(payload):
    with pytest.raises(HistoryReviewJsonStructureError) as exc_info:
        parse_history_review_output(
            raw_output=json.dumps(payload),
            source_seq_by_id={101: 1},
        )
    _assert_code(exc_info, "history_review_json_structure_error")


@pytest.mark.parametrize(
    "finding",
    [
        {
            "source_message_id": 101,
            "verdict": "correct",
            "claim_text": "claim",
            "correction_text": None,
            "explanation_text": None,
            "extra": 1,
        },
        {
            "verdict": "correct",
            "claim_text": "claim",
            "correction_text": None,
            "explanation_text": None,
        },
    ],
)
def test_finding_unknown_or_missing_fields_are_structure_errors(finding):
    with pytest.raises(HistoryReviewJsonStructureError) as exc_info:
        parse_history_review_output(
            raw_output=_raw(findings=[finding]),
            source_seq_by_id={101: 1},
        )
    _assert_code(exc_info, "history_review_json_structure_error")


@pytest.mark.parametrize("source_id", [True, False, "101", None, 1.0])
def test_bool_or_wrong_type_source_id_is_structure_error(source_id):
    with pytest.raises(HistoryReviewJsonStructureError) as exc_info:
        parse_history_review_output(
            raw_output=_raw(
                findings=[_finding(source_id, "correct")]
            ),
            source_seq_by_id={101: 1},
        )
    _assert_code(exc_info, "history_review_json_structure_error")


@pytest.mark.parametrize("source_id", [0, -1, 999])
def test_invalid_or_unknown_source_id_is_reference_error(source_id):
    with pytest.raises(HistoryReviewSourceReferenceInvalidError) as exc_info:
        parse_history_review_output(
            raw_output=_raw(
                findings=[_finding(source_id, "correct")]
            ),
            source_seq_by_id={101: 1},
        )
    _assert_code(exc_info, "history_review_source_reference_invalid")


def test_every_source_must_be_covered():
    with pytest.raises(HistoryReviewSourceUncoveredError) as exc_info:
        parse_history_review_output(
            raw_output=_raw(
                findings=[_finding(101, "correct")]
            ),
            source_seq_by_id={101: 1, 205: 2},
        )
    _assert_code(exc_info, "history_review_source_uncovered")


def test_findings_must_not_go_back_to_earlier_source():
    with pytest.raises(HistoryReviewJsonSemanticError) as exc_info:
        parse_history_review_output(
            raw_output=_raw(
                findings=[
                    _finding(205, "correct"),
                    _finding(101, "correct"),
                ]
            ),
            source_seq_by_id={101: 1, 205: 2},
        )
    _assert_code(exc_info, "history_review_json_semantic_error")


def test_empty_findings_is_structure_error():
    with pytest.raises(HistoryReviewJsonStructureError) as exc_info:
        parse_history_review_output(
            raw_output=_raw(findings=[]),
            source_seq_by_id={101: 1},
        )
    _assert_code(exc_info, "history_review_json_structure_error")


def test_too_many_findings_is_output_too_large():
    with pytest.raises(HistoryReviewOutputTooLargeError) as exc_info:
        parse_history_review_output(
            raw_output=_raw(
                findings=[
                    _finding(101, "correct")
                    for _ in range(51)
                ]
            ),
            source_seq_by_id={101: 1},
        )
    _assert_code(exc_info, "history_review_output_too_large")


@pytest.mark.parametrize("verdict", [None, True, 1, []])
def test_non_string_verdict_is_structure_error(verdict):
    with pytest.raises(HistoryReviewJsonStructureError) as exc_info:
        parse_history_review_output(
            raw_output=_raw(
                findings=[_finding(101, verdict)]
            ),
            source_seq_by_id={101: 1},
        )
    _assert_code(exc_info, "history_review_json_structure_error")


def test_unknown_verdict_is_semantic_error():
    with pytest.raises(HistoryReviewJsonSemanticError) as exc_info:
        parse_history_review_output(
            raw_output=_raw(
                findings=[_finding(101, "maybe")]
            ),
            source_seq_by_id={101: 1},
        )
    _assert_code(exc_info, "history_review_json_semantic_error")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("summary", 1),
        ("coverage_note", []),
        ("correction_text", []),
        ("explanation_text", 1),
    ],
)
def test_optional_and_summary_wrong_types_are_structure_errors(field, value):
    payload = {
        "summary": "s",
        "coverage_note": None,
        "findings": [_finding(101, "correct")],
    }
    if field == "summary":
        payload["summary"] = value
    elif field == "coverage_note":
        payload["coverage_note"] = value
    else:
        payload["findings"][0][field] = value
    with pytest.raises(HistoryReviewJsonStructureError) as exc_info:
        parse_history_review_output(
            raw_output=json.dumps(payload),
            source_seq_by_id={101: 1},
        )
    _assert_code(exc_info, "history_review_json_structure_error")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("claim_text", True),
        ("claim_text", 1),
        ("claim_text", []),
    ],
)
def test_claim_text_wrong_type_is_structure_error(field, value):
    finding = _finding(101, "correct")
    finding[field] = value
    with pytest.raises(HistoryReviewJsonStructureError) as exc_info:
        parse_history_review_output(
            raw_output=_raw(findings=[finding]),
            source_seq_by_id={101: 1},
        )
    _assert_code(exc_info, "history_review_json_structure_error")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("summary", "   "),
        ("coverage_note", ""),
        ("claim_text", "\t\n"),
        ("correction_text", "  "),
        ("explanation_text", "\n"),
    ],
)
def test_blank_text_is_semantic_error(field, value):
    finding = _finding(
        101,
        "incorrect",
        correction_text="correction",
        explanation_text="explanation",
    )
    payload = {
        "summary": "s",
        "coverage_note": None,
        "findings": [finding],
    }
    if field in {"summary", "coverage_note"}:
        payload[field] = value
    else:
        finding[field] = value
    with pytest.raises(HistoryReviewJsonSemanticError) as exc_info:
        parse_history_review_output(
            raw_output=json.dumps(payload),
            source_seq_by_id={101: 1},
        )
    _assert_code(exc_info, "history_review_json_semantic_error")


@pytest.mark.parametrize(
    ("field", "value", "verdict"),
    [
        ("summary", "s" * 4001, "correct"),
        ("coverage_note", "c" * 2001, "correct"),
        ("claim_text", "c" * 4001, "correct"),
        ("correction_text", "c" * 4001, "incorrect"),
        ("explanation_text", "e" * 4001, "incorrect"),
    ],
)
def test_text_over_limit_is_output_too_large(field, value, verdict):
    finding = _finding(
        101,
        verdict,
        correction_text=(
            "correction" if verdict == "incorrect" else None
        ),
        explanation_text=(
            "explanation" if verdict == "incorrect" else None
        ),
    )
    payload = {
        "summary": "s",
        "coverage_note": None,
        "findings": [finding],
    }
    if field in {"summary", "coverage_note"}:
        payload[field] = value
    else:
        finding[field] = value
    with pytest.raises(HistoryReviewOutputTooLargeError) as exc_info:
        parse_history_review_output(
            raw_output=json.dumps(payload),
            source_seq_by_id={101: 1},
        )
    _assert_code(exc_info, "history_review_output_too_large")


@pytest.mark.parametrize(
    "source_seq_by_id",
    [
        [],
        "not-a-mapping",
        {},
        {True: 1},
        {101: True},
        {0: 1},
        {101: 0},
        {-1: 1},
        {101: -1},
        {101: 1, 205: 3},
        {101: 1, 205: 1},
    ],
)
def test_invalid_source_seq_by_id(source_seq_by_id):
    with pytest.raises((TypeError, ValueError)):
        parse_history_review_output(
            raw_output=_raw(
                findings=[_finding(101, "correct")]
            ),
            source_seq_by_id=source_seq_by_id,
        )


def test_input_mapping_is_not_mutated():
    source_seq_by_id = {101: 1, 205: 2}
    before = dict(source_seq_by_id)
    parse_history_review_output(
        raw_output=_raw(
            findings=[
                _finding(101, "correct"),
                _finding(205, "correct"),
            ]
        ),
        source_seq_by_id=source_seq_by_id,
    )
    assert source_seq_by_id == before


def test_returned_objects_are_frozen():
    result = parse_history_review_output(
        raw_output=_raw(findings=[_finding(101, "correct")]),
        source_seq_by_id={101: 1},
    )
    with pytest.raises(FrozenInstanceError):
        result.summary = "changed"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        result.findings[0].claim_text = "changed"  # type: ignore[misc]


def test_error_messages_do_not_contain_raw_output_marker():
    marker = "SECRET_RAW_MARKER_123"
    with pytest.raises(HistoryReviewOutputError) as exc_info:
        parse_history_review_output(
            raw_output=marker,
            source_seq_by_id={101: 1},
        )
    message = str(exc_info.value)
    assert marker not in message
    assert marker not in repr(exc_info.value)
    assert not hasattr(exc_info.value, "raw_output")


def test_truncate_rejects_non_string():
    with pytest.raises(TypeError):
        truncate_raw_output(raw_output=123, max_chars=10)  # type: ignore[arg-type]


@pytest.mark.parametrize("max_chars", [True, False, "10", None, 1.5])
def test_truncate_rejects_non_integer_max_chars(max_chars):
    with pytest.raises(TypeError):
        truncate_raw_output(raw_output="abc", max_chars=max_chars)


@pytest.mark.parametrize("max_chars", [0, -1])
def test_truncate_rejects_non_positive_max_chars(max_chars):
    with pytest.raises(ValueError):
        truncate_raw_output(raw_output="abc", max_chars=max_chars)


def test_truncate_boundaries_and_unicode():
    original = "汉字😀abc"
    assert truncate_raw_output(raw_output=original, max_chars=10) == (
        original,
        False,
    )
    assert truncate_raw_output(raw_output=original, max_chars=len(original)) == (
        original,
        False,
    )
    assert truncate_raw_output(raw_output=original, max_chars=3) == (
        "汉字😀",
        True,
    )
    assert truncate_raw_output(raw_output="", max_chars=1) == ("", False)
