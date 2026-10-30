"""Tests for the pure two-stage History Review pipeline orchestrator."""

import asyncio
import json

import pytest

from backend.exceptions import LLMError
from backend.history_review_parser import ParsedHistoryReviewOutput
from backend.history_review_prompt import HistoryReviewSourceSnapshot
from backend.history_review_selection import estimate_history_review_tokens
from backend.history_review_stages import (
    HISTORY_REVIEW_PIPELINE_VERSION,
    HISTORY_REVIEW_STAGE_1_SYSTEM_PROMPT,
    HISTORY_REVIEW_STAGE_2_SYSTEM_PROMPT,
    STAGE1_MAX_INPUT_TOKENS,
    STAGE1_MAX_SYSTEM_PROMPT_TOKENS,
    STAGE1_MAX_USER_JSON_TOKENS,
    STAGE1_RESERVED_OUTPUT_TOKENS,
    STAGE2_RESERVED_OUTPUT_TOKENS,
    HistoryReviewStage2InputTooLargeError,
    build_stage1_messages,
    validate_stage1_input_budget,
)
from backend.review_diagnostics import (
    ErrorCategory,
    FailureStage,
    HistoryReviewCancelledError,
    ReviewMode,
    ReviewStage,
)
import backend.history_review_pipeline as pipeline_module
from backend.history_review_pipeline import (
    AUDIT_MODE_EMPTY_CONFIRMATION,
    AUDIT_MODE_INITIAL,
    AUDIT_MODE_REPAIR,
    HISTORY_REVIEW_EMPTY_CONFIRMATION_TEMPLATE,
    HISTORY_REVIEW_STAGE1_STRUCTURE_COVERAGE_HINT,
    _build_empty_confirmation_messages,
    _build_stage1_retry_messages,
    _stage1_retry_system_hint,
    HISTORY_REVIEW_LLM_EMPTY_RESPONSE,
    HISTORY_REVIEW_LLM_NON_STRING_RESPONSE,
    HISTORY_REVIEW_RETRY_REPAIR_TEMPLATE,
    HistoryReviewPipelineError,
    HistoryReviewPipelineResult,
    run_history_review_pipeline,
)


class SpyClient:
    """Deterministic LLMClient spy."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.response_finish_reason = None
        self.response_http_status = None

    async def generate(
        self,
        messages,
        *,
        response_format=None,
        temperature=None,
        max_tokens=None,
        diagnostics=None,
    ):
        self.calls.append({
            "messages": [dict(message) for message in messages],
            "response_format": response_format,
            "temperature": temperature,
            "max_tokens": max_tokens,
        })
        if diagnostics is not None:
            if self.response_finish_reason is not None:
                diagnostics.finish_reason = self.response_finish_reason
            if self.response_http_status is not None:
                diagnostics.http_status = self.response_http_status
        if not self.responses:
            raise AssertionError("unexpected LLM call")
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


def _source(seq, message_id, content):
    return HistoryReviewSourceSnapshot(
        seq=seq, message_id=message_id, content=content,
    )


def _sources():
    return (
        _source(
            1,
            101,
            "Alpha is the first letter. Gamma is the third letter. "
            "Record both claims.",
        ),
        _source(
            2,
            205,
            "Beta is the second letter. Confirm this claim.",
        ),
        _source(
            3,
            309,
            "Summarize the lesson in three points.",
        ),
    )


def _stage1_raw():
    return json.dumps({
        "overflow": False,
        "sources": [
            {
                "source_message_id": 101,
                "claims": [
                    {
                        "claim_text": "Alpha is the first letter",
                        "source_quote": "Alpha is the first letter.",
                    },
                    {
                        "claim_text": "Gamma is the third letter",
                        "source_quote": "Gamma is the third letter.",
                    },
                ],
            },
            {
                "source_message_id": 205,
                "claims": [
                    {
                        "claim_text": "Beta is the second letter",
                        "source_quote": "Beta is the second letter.",
                    },
                ],
            },
            {"source_message_id": 309, "claims": []},
        ],
    }, ensure_ascii=False)


def _stage1_raw_zero_claims():
    return json.dumps({
        "overflow": False,
        "sources": [
            {"source_message_id": 101, "claims": []},
            {"source_message_id": 205, "claims": []},
            {"source_message_id": 309, "claims": []},
        ],
    }, ensure_ascii=False)


def _stage1_raw_missing_source():
    payload = json.loads(_stage1_raw())
    payload["sources"] = payload["sources"][:2]
    return json.dumps(payload, ensure_ascii=False)


def _stage1_raw_bad_quote():
    payload = json.loads(_stage1_raw())
    payload["sources"][0]["claims"][0]["source_quote"] = "not in source"
    return json.dumps(payload, ensure_ascii=False)


def _stage1_raw_top_level_claims():
    return '{"overflow":false,"sources":[],"claims":[]}'


def _stage1_raw_semantic_error():
    payload = json.loads(_stage1_raw())
    payload["sources"][0]["claims"][0]["claim_text"] = ""
    return json.dumps(payload, ensure_ascii=False)


def _stage2_raw():
    return json.dumps({
        "claims": [
            {
                "claim_id": 1,
                "verdict": "correct",
                "correction_text": None,
                "explanation_text": None,
            },
            {
                "claim_id": 2,
                "verdict": "incorrect",
                "correction_text": "Gamma is not the third letter",
                "explanation_text": "The gamma claim is incorrect.",
            },
            {
                "claim_id": 3,
                "verdict": "correct",
                "correction_text": None,
                "explanation_text": None,
            },
        ],
    }, ensure_ascii=False)


def _stage2_raw_missing_claim():
    payload = json.loads(_stage2_raw())
    payload["claims"] = payload["claims"][:1]
    return json.dumps(payload, ensure_ascii=False)


def _oversized_sources():
    return tuple(
        _source(seq, 100 + seq, "测" * 200)
        for seq in range(1, 11)
    )


def _assert_call_options(call, *, max_tokens):
    assert call["response_format"] == {"type": "json_object"}
    assert call["temperature"] == 0
    assert call["max_tokens"] == max_tokens


@pytest.mark.anyio
async def test_two_stage_success():
    sources = _sources()
    stage1_raw = _stage1_raw()
    stage2_raw = _stage2_raw()
    client = SpyClient([stage1_raw, stage2_raw])

    result = await run_history_review_pipeline(
        client=client,
        sources=sources,
    )

    assert isinstance(result, HistoryReviewPipelineResult)
    assert isinstance(result.parsed, ParsedHistoryReviewOutput)
    assert result.llm_call_count == 2
    assert [finding.source_message_id for finding in result.parsed.findings] == [
        101, 101, 205, 309,
    ]
    assert [finding.verdict for finding in result.parsed.findings] == [
        "correct", "incorrect", "correct", "not_a_claim",
    ]
    assert len(client.calls) == 2
    _assert_call_options(
        client.calls[0], max_tokens=STAGE1_RESERVED_OUTPUT_TOKENS,
    )
    _assert_call_options(
        client.calls[1], max_tokens=STAGE2_RESERVED_OUTPUT_TOKENS,
    )
    assert client.calls[0]["messages"][0]["content"] != (
        client.calls[1]["messages"][0]["content"]
    )


@pytest.mark.anyio
async def test_mixed_facts_and_instruction_pipeline_order():
    client = SpyClient([_stage1_raw(), _stage2_raw()])
    result = await run_history_review_pipeline(
        client=client,
        sources=_sources(),
    )

    assert [finding.source_message_id for finding in result.parsed.findings] == [
        101, 101, 205, 309,
    ]
    assert [finding.verdict for finding in result.parsed.findings] == [
        "correct", "incorrect", "correct", "not_a_claim",
    ]
    assert result.parsed.findings[-1].claim_text == (
        "Summarize the lesson in three points."
    )
    assert len(client.calls) == 2


@pytest.mark.anyio
async def test_stage1_empty_then_confirmation_empty_short_circuits_stage2():
    zero = _stage1_raw_zero_claims()
    client = SpyClient([zero, zero])

    result = await run_history_review_pipeline(
        client=client,
        sources=_sources(),
    )

    assert result.llm_call_count == 2
    assert len(client.calls) == 2
    assert [finding.verdict for finding in result.parsed.findings] == [
        "not_a_claim", "not_a_claim", "not_a_claim",
    ]
    assert [finding.source_message_id for finding in result.parsed.findings] == [
        101, 205, 309,
    ]
    assert len(client.calls[0]["messages"]) == 2
    assert len(client.calls[1]["messages"]) == 2
    assert client.calls[0]["messages"][1] == client.calls[1]["messages"][1]
    assert "empty_confirmation" not in client.calls[0]["messages"][0]["content"]
    assert "所有 source 的 claims 都为空" in (
        client.calls[1]["messages"][0]["content"]
    )

    audit = json.loads(result.audit_raw_output)
    assert [
        (item["stage"], item["attempt"], item["mode"])
        for item in audit["attempts"]
    ] == [
        ("stage1", 1, AUDIT_MODE_INITIAL),
        ("stage1", 2, AUDIT_MODE_EMPTY_CONFIRMATION),
    ]


@pytest.mark.anyio
async def test_stage1_empty_then_confirmation_recovers_claims_runs_stage2():
    zero = _stage1_raw_zero_claims()
    client = SpyClient([zero, _stage1_raw(), _stage2_raw()])

    result = await run_history_review_pipeline(
        client=client,
        sources=_sources(),
    )

    assert result.llm_call_count == 3
    assert len(client.calls) == 3
    assert client.calls[0]["messages"][1] == client.calls[1]["messages"][1]
    assert "所有 source 的 claims 都为空" in (
        client.calls[1]["messages"][0]["content"]
    )
    assert [finding.verdict for finding in result.parsed.findings] != [
        "not_a_claim", "not_a_claim", "not_a_claim",
    ]
    audit = json.loads(result.audit_raw_output)
    assert [
        (item["stage"], item["attempt"], item["mode"])
        for item in audit["attempts"]
    ] == [
        ("stage1", 1, AUDIT_MODE_INITIAL),
        ("stage1", 2, AUDIT_MODE_EMPTY_CONFIRMATION),
        ("stage2", 1, AUDIT_MODE_INITIAL),
    ]


@pytest.mark.parametrize(
    ("second_raw", "expected_code"),
    [
        ("{", "history_review_extraction_json_syntax_error"),
        ("[]", "history_review_extraction_json_structure_error"),
        (_stage1_raw_bad_quote(), "history_review_extraction_quote_invalid"),
    ],
)
@pytest.mark.anyio
async def test_stage1_empty_then_confirmation_error_no_third_call(
    second_raw, expected_code,
):
    client = SpyClient([_stage1_raw_zero_claims(), second_raw])

    with pytest.raises(HistoryReviewPipelineError) as exc_info:
        await run_history_review_pipeline(
            client=client,
            sources=_sources(),
        )

    error = exc_info.value
    assert error.code == expected_code
    assert error.llm_call_count == 2
    assert len(client.calls) == 2
    audit = json.loads(error.audit_raw_output)
    assert [
        (item["stage"], item["attempt"], item["mode"])
        for item in audit["attempts"]
    ] == [
        ("stage1", 1, AUDIT_MODE_INITIAL),
        ("stage1", 2, AUDIT_MODE_EMPTY_CONFIRMATION),
    ]


@pytest.mark.anyio
async def test_stage1_syntax_error_repair_never_uses_empty_confirmation():
    first_raw = "not json"
    client = SpyClient([first_raw, _stage1_raw(), _stage2_raw()])

    result = await run_history_review_pipeline(
        client=client,
        sources=_sources(),
    )

    assert result.llm_call_count == 3
    retry_system = client.calls[1]["messages"][0]["content"]
    assert "错误类别为 syntax" in retry_system
    assert HISTORY_REVIEW_STAGE1_STRUCTURE_COVERAGE_HINT not in retry_system
    assert HISTORY_REVIEW_EMPTY_CONFIRMATION_TEMPLATE not in retry_system
    audit = json.loads(result.audit_raw_output)
    assert [
        (item["stage"], item["attempt"], item["mode"])
        for item in audit["attempts"]
    ] == [
        ("stage1", 1, AUDIT_MODE_INITIAL),
        ("stage1", 2, AUDIT_MODE_REPAIR),
        ("stage2", 1, AUDIT_MODE_INITIAL),
    ]


@pytest.mark.anyio
async def test_stage1_structure_repair_hint_and_empty_claims_no_third_call():
    first_raw = _stage1_raw_top_level_claims()
    client = SpyClient([first_raw, _stage1_raw_zero_claims()])

    result = await run_history_review_pipeline(
        client=client,
        sources=_sources(),
    )

    assert result.llm_call_count == 2
    assert len(client.calls) == 2
    first_messages = client.calls[0]["messages"]
    repair_messages = client.calls[1]["messages"]
    repair_system = repair_messages[0]["content"]

    assert "错误类别为 structure" in repair_system
    assert HISTORY_REVIEW_STAGE1_STRUCTURE_COVERAGE_HINT in repair_system
    assert HISTORY_REVIEW_RETRY_REPAIR_TEMPLATE.format(
        category="structure"
    ) in repair_system
    assert first_raw not in repair_system
    assert first_messages[1] == repair_messages[1]
    assert [
        finding.source_message_id for finding in result.parsed.findings
    ] == [101, 205, 309]
    assert [
        finding.verdict for finding in result.parsed.findings
    ] == ["not_a_claim", "not_a_claim", "not_a_claim"]

    audit = json.loads(result.audit_raw_output)
    assert [
        (item["stage"], item["attempt"], item["mode"])
        for item in audit["attempts"]
    ] == [
        ("stage1", 1, AUDIT_MODE_INITIAL),
        ("stage1", 2, AUDIT_MODE_REPAIR),
    ]


@pytest.mark.anyio
async def test_stage1_semantic_repair_has_no_structure_coverage_hint():
    client = SpyClient([
        _stage1_raw_semantic_error(),
        _stage1_raw(),
        _stage2_raw(),
    ])

    result = await run_history_review_pipeline(
        client=client,
        sources=_sources(),
    )

    assert result.llm_call_count == 3
    retry_system = client.calls[1]["messages"][0]["content"]
    assert "错误类别为 semantic" in retry_system
    assert HISTORY_REVIEW_STAGE1_STRUCTURE_COVERAGE_HINT not in retry_system


@pytest.mark.anyio
async def test_stage1_reference_repair_has_no_structure_coverage_hint():
    client = SpyClient([
        _stage1_raw_bad_quote(),
        _stage1_raw(),
        _stage2_raw(),
    ])

    result = await run_history_review_pipeline(
        client=client,
        sources=_sources(),
    )

    assert result.llm_call_count == 3
    retry_system = client.calls[1]["messages"][0]["content"]
    assert "错误类别为 reference" in retry_system
    assert HISTORY_REVIEW_STAGE1_STRUCTURE_COVERAGE_HINT not in retry_system


@pytest.mark.parametrize("category", ["structure", "coverage"])
def test_stage1_retry_system_hint_present_for_structure_and_coverage(
    category,
):
    assert _stage1_retry_system_hint(category) == (
        HISTORY_REVIEW_STAGE1_STRUCTURE_COVERAGE_HINT
    )


@pytest.mark.parametrize(
    "category",
    ["syntax", "semantic", "reference", "contract"],
)
def test_stage1_retry_system_hint_absent_for_other_categories(category):
    assert _stage1_retry_system_hint(category) is None


@pytest.mark.anyio
async def test_stage2_structure_repair_has_no_stage1_hint():
    client = SpyClient([_stage1_raw(), "[]", _stage2_raw()])

    result = await run_history_review_pipeline(
        client=client,
        sources=_sources(),
    )

    assert result.llm_call_count == 3
    stage2_repair_system = client.calls[2]["messages"][0]["content"]
    assert "错误类别为 structure" in stage2_repair_system
    assert HISTORY_REVIEW_STAGE1_STRUCTURE_COVERAGE_HINT not in (
        stage2_repair_system
    )


@pytest.mark.anyio
async def test_stage1_repair_returns_valid_empty_without_third_call():
    client = SpyClient([
        _stage1_raw_missing_source(),
        _stage1_raw_zero_claims(),
    ])

    result = await run_history_review_pipeline(
        client=client,
        sources=_sources(),
    )

    assert result.llm_call_count == 2
    assert len(client.calls) == 2
    assert [finding.verdict for finding in result.parsed.findings] == [
        "not_a_claim", "not_a_claim", "not_a_claim",
    ]
    retry_system = client.calls[1]["messages"][0]["content"]
    assert "错误类别为 coverage" in retry_system
    assert HISTORY_REVIEW_STAGE1_STRUCTURE_COVERAGE_HINT in retry_system
    assert HISTORY_REVIEW_EMPTY_CONFIRMATION_TEMPLATE not in retry_system
    assert client.calls[0]["messages"][1] == client.calls[1]["messages"][1]
    audit = json.loads(result.audit_raw_output)
    assert [
        (item["stage"], item["attempt"], item["mode"])
        for item in audit["attempts"]
    ] == [
        ("stage1", 1, AUDIT_MODE_INITIAL),
        ("stage1", 2, AUDIT_MODE_REPAIR),
    ]


@pytest.mark.anyio
async def test_empty_confirmation_preserves_user_json_and_hides_first_raw():
    first_raw = _stage1_raw_zero_claims()
    client = SpyClient([first_raw, _stage1_raw(), _stage2_raw()])

    result = await run_history_review_pipeline(
        client=client,
        sources=_sources(),
    )

    assert result.llm_call_count == 3
    first_messages = client.calls[0]["messages"]
    confirmation_messages = client.calls[1]["messages"]
    assert first_messages[1] == confirmation_messages[1]
    assert first_raw not in confirmation_messages[0]["content"]
    assert first_raw not in json.dumps(
        confirmation_messages, ensure_ascii=False,
    )


@pytest.mark.anyio
async def test_empty_confirmation_budget_block_calls_only_once(monkeypatch):
    zero = _stage1_raw_zero_claims()
    client = SpyClient([zero])

    monkeypatch.setattr(
        pipeline_module,
        "_messages_token_estimate",
        lambda messages: STAGE1_MAX_INPUT_TOKENS + 1,
    )

    with pytest.raises(HistoryReviewPipelineError) as exc_info:
        await run_history_review_pipeline(
            client=client,
            sources=_sources(),
        )

    error = exc_info.value
    assert error.code == "history_review_extraction_input_too_large"
    assert error.llm_call_count == 1
    assert len(client.calls) == 1


@pytest.mark.anyio
async def test_empty_confirmation_recovered_with_stage2_repair_max_four_calls():
    client = SpyClient([
        _stage1_raw_zero_claims(),
        _stage1_raw(),
        _stage2_raw_missing_claim(),
        _stage2_raw(),
    ])

    result = await run_history_review_pipeline(
        client=client,
        sources=_sources(),
    )

    assert result.llm_call_count == 4
    assert len(client.calls) == 4
    stage2_repair_system = client.calls[3]["messages"][0]["content"]
    assert "错误类别为 coverage" in stage2_repair_system
    assert HISTORY_REVIEW_EMPTY_CONFIRMATION_TEMPLATE not in (
        stage2_repair_system
    )
    assert HISTORY_REVIEW_STAGE1_STRUCTURE_COVERAGE_HINT not in (
        stage2_repair_system
    )
    assert "X 是 Y" not in stage2_repair_system
    audit = json.loads(result.audit_raw_output)
    assert [
        (item["stage"], item["attempt"], item["mode"])
        for item in audit["attempts"]
    ] == [
        ("stage1", 1, AUDIT_MODE_INITIAL),
        ("stage1", 2, AUDIT_MODE_EMPTY_CONFIRMATION),
        ("stage2", 1, AUDIT_MODE_INITIAL),
        ("stage2", 2, AUDIT_MODE_REPAIR),
    ]


@pytest.mark.anyio
async def test_stage1_syntax_error_then_retry_success():
    first_raw = '{"BROKEN_RAW_SHOULD_NOT_LEAK":'
    client = SpyClient([first_raw, _stage1_raw(), _stage2_raw()])

    result = await run_history_review_pipeline(
        client=client,
        sources=_sources(),
    )

    assert result.llm_call_count == 3
    assert len(client.calls) == 3
    retry_system = client.calls[1]["messages"][0]["content"]
    assert "错误类别为 syntax" in retry_system
    assert first_raw not in json.dumps(client.calls[1]["messages"])
    assert client.calls[0]["messages"][1] == (
        client.calls[1]["messages"][1]
    )


@pytest.mark.anyio
async def test_stage1_coverage_error_then_retry_success():
    client = SpyClient([
        _stage1_raw_missing_source(),
        _stage1_raw(),
        _stage2_raw(),
    ])

    result = await run_history_review_pipeline(
        client=client,
        sources=_sources(),
    )

    assert result.llm_call_count == 3
    retry_system = client.calls[1]["messages"][0]["content"]
    assert "错误类别为 coverage" in retry_system


@pytest.mark.anyio
async def test_stage2_coverage_error_then_retry_success():
    client = SpyClient([
        _stage1_raw(),
        _stage2_raw_missing_claim(),
        _stage2_raw(),
    ])

    result = await run_history_review_pipeline(
        client=client,
        sources=_sources(),
    )

    assert result.llm_call_count == 3
    assert len(result.parsed.findings) == 4
    retry_system = client.calls[2]["messages"][0]["content"]
    assert "错误类别为 coverage" in retry_system
    assert HISTORY_REVIEW_STAGE1_STRUCTURE_COVERAGE_HINT not in retry_system
    assert client.calls[1]["messages"][1] == (
        client.calls[2]["messages"][1]
    )


@pytest.mark.anyio
async def test_stage1_both_attempts_fail_does_not_call_stage2():
    client = SpyClient([_stage1_raw_missing_source(), "{"])
    with pytest.raises(HistoryReviewPipelineError) as exc_info:
        await run_history_review_pipeline(
            client=client,
            sources=_sources(),
        )

    error = exc_info.value
    assert error.code == "history_review_extraction_json_syntax_error"
    assert error.called_llm is True
    assert error.llm_call_count == 2
    assert len(client.calls) == 2
    audit = json.loads(error.audit_raw_output)
    assert [item["stage"] for item in audit["attempts"]] == [
        "stage1", "stage1",
    ]
    assert [item["attempt"] for item in audit["attempts"]] == [1, 2]


@pytest.mark.anyio
async def test_stage2_both_attempts_fail_audit_contract():
    client = SpyClient([
        _stage1_raw(),
        _stage2_raw_missing_claim(),
        "{",
    ])
    with pytest.raises(HistoryReviewPipelineError) as exc_info:
        await run_history_review_pipeline(
            client=client,
            sources=_sources(),
        )

    error = exc_info.value
    assert error.code == "history_review_verification_json_syntax_error"
    assert error.called_llm is True
    assert error.llm_call_count == 3
    audit = json.loads(error.audit_raw_output)
    assert [item["stage"] for item in audit["attempts"]] == [
        "stage1", "stage2", "stage2",
    ]
    assert [item["attempt"] for item in audit["attempts"]] == [1, 1, 2]


@pytest.mark.anyio
async def test_pipeline_never_exceeds_four_calls():
    client = SpyClient([
        _stage1_raw_missing_source(),
        _stage1_raw(),
        _stage2_raw_missing_claim(),
        _stage2_raw(),
    ])
    result = await run_history_review_pipeline(
        client=client,
        sources=_sources(),
    )
    assert result.llm_call_count == 4
    assert len(client.calls) == 4


@pytest.mark.anyio
async def test_capacity_exceeded_does_not_retry_or_call_stage2():
    client = SpyClient([
        json.dumps({"overflow": True, "sources": []}),
    ])
    with pytest.raises(HistoryReviewPipelineError) as exc_info:
        await run_history_review_pipeline(
            client=client,
            sources=_sources(),
        )

    error = exc_info.value
    assert error.code == "history_review_extraction_capacity_exceeded"
    assert error.called_llm is True
    assert error.llm_call_count == 1
    assert len(client.calls) == 1
    audit = json.loads(error.audit_raw_output)
    assert len(audit["attempts"]) == 1


@pytest.mark.anyio
async def test_input_too_large_does_not_call_llm():
    client = SpyClient([])
    with pytest.raises(HistoryReviewPipelineError) as exc_info:
        await run_history_review_pipeline(
            client=client,
            sources=_oversized_sources(),
        )

    error = exc_info.value
    assert error.code == "history_review_extraction_input_too_large"
    assert error.called_llm is False
    assert error.llm_call_count == 0
    assert error.audit_raw_output is None
    assert client.calls == []


@pytest.mark.anyio
async def test_output_too_large_does_not_retry():
    client = SpyClient(["x" * 5000])
    with pytest.raises(HistoryReviewPipelineError) as exc_info:
        await run_history_review_pipeline(
            client=client,
            sources=_sources(),
        )

    error = exc_info.value
    assert error.code == "history_review_extraction_too_large"
    assert error.llm_call_count == 1
    assert len(client.calls) == 1


@pytest.mark.anyio
async def test_non_string_response_does_not_retry():
    client = SpyClient([123])
    with pytest.raises(HistoryReviewPipelineError) as exc_info:
        await run_history_review_pipeline(
            client=client,
            sources=_sources(),
        )

    error = exc_info.value
    assert error.code == HISTORY_REVIEW_LLM_NON_STRING_RESPONSE
    assert error.llm_call_count == 1
    assert error.audit_raw_output is None


@pytest.mark.anyio
async def test_empty_response_does_not_retry():
    client = SpyClient(["   "])
    with pytest.raises(HistoryReviewPipelineError) as exc_info:
        await run_history_review_pipeline(
            client=client,
            sources=_sources(),
        )

    error = exc_info.value
    assert error.code == HISTORY_REVIEW_LLM_EMPTY_RESPONSE
    assert error.llm_call_count == 1
    assert error.audit_raw_output is None


@pytest.mark.anyio
async def test_llm_error_propagates_unwrapped():
    llm_error = LLMError(detail="boom", status_code=502)
    client = SpyClient([llm_error])

    with pytest.raises(LLMError) as exc_info:
        await run_history_review_pipeline(
            client=client,
            sources=_sources(),
        )

    assert exc_info.value is llm_error
    assert len(client.calls) == 1


@pytest.mark.anyio
async def test_cancelled_error_propagates_unwrapped():
    client = SpyClient([asyncio.CancelledError()])

    with pytest.raises(asyncio.CancelledError):
        await run_history_review_pipeline(
            client=client,
            sources=_sources(),
        )

    assert len(client.calls) == 1


def test_repair_prompt_is_frozen():
    assert HISTORY_REVIEW_RETRY_REPAIR_TEMPLATE == (
        "上一次输出未通过机器校验，错误类别为 {category}。"
        "请基于完全相同的输入重新输出完整 JSON。"
        "严格遵守覆盖、顺序、引用和字段契约；不要只输出修补片段。"
    )


@pytest.mark.anyio
async def test_repair_prompt_respects_stage1_budget_and_safety():
    first_raw = '{"BROKEN_RAW_SHOULD_NOT_LEAK":'
    client = SpyClient([first_raw, _stage1_raw(), _stage2_raw()])
    result = await run_history_review_pipeline(
        client=client,
        sources=_sources(),
    )
    assert result.llm_call_count == 3

    first_messages = client.calls[0]["messages"]
    retry_messages = client.calls[1]["messages"]
    retry_system = retry_messages[0]["content"]

    assert "错误类别为 syntax" in retry_system
    assert first_raw not in retry_system
    assert first_raw not in json.dumps(retry_messages, ensure_ascii=False)
    assert "事实更正" not in retry_system
    assert first_messages[1] == retry_messages[1]
    assert sum(
        estimate_history_review_tokens(message["content"])
        for message in retry_messages
    ) <= STAGE1_MAX_INPUT_TOKENS


@pytest.mark.anyio
async def test_audit_json_contract():
    stage1_raw = _stage1_raw()
    stage2_raw = _stage2_raw()
    client = SpyClient([stage1_raw, stage2_raw])

    result = await run_history_review_pipeline(
        client=client,
        sources=_sources(),
    )

    expected = json.dumps(
        {
            "attempts": [
                {
                    "attempt": 1,
                    "mode": AUDIT_MODE_INITIAL,
                    "raw_output": stage1_raw,
                    "stage": "stage1",
                },
                {
                    "attempt": 1,
                    "mode": AUDIT_MODE_INITIAL,
                    "raw_output": stage2_raw,
                    "stage": "stage2",
                },
            ],
            "pipeline_version": HISTORY_REVIEW_PIPELINE_VERSION,
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    assert result.audit_raw_output == expected
    assert "你是 History Review" not in result.audit_raw_output
    assert '"messages"' not in result.audit_raw_output
    assert "test-model" not in result.audit_raw_output


@pytest.mark.anyio
async def test_error_audit_preserves_raw_outputs_exactly():
    first_raw = _stage1_raw_missing_source()
    second_raw = "{"
    client = SpyClient([first_raw, second_raw])

    with pytest.raises(HistoryReviewPipelineError) as exc_info:
        await run_history_review_pipeline(
            client=client,
            sources=_sources(),
        )

    audit = json.loads(exc_info.value.audit_raw_output)
    assert [item["raw_output"] for item in audit["attempts"]] == [
        first_raw, second_raw,
    ]
    assert [item["stage"] for item in audit["attempts"]] == [
        "stage1", "stage1",
    ]


@pytest.mark.anyio
async def test_same_client_is_used_for_all_stage_calls():
    client = SpyClient([
        _stage1_raw_missing_source(),
        _stage1_raw(),
        _stage2_raw_missing_claim(),
        _stage2_raw(),
    ])
    result = await run_history_review_pipeline(
        client=client,
        sources=_sources(),
    )
    assert result.llm_call_count == 4
    assert len(client.calls) == 4
    assert all(call["messages"] for call in client.calls)


@pytest.mark.anyio
async def test_stage2_input_budget_failure_wraps_with_stage1_audit(
    monkeypatch,
):
    import backend.history_review_pipeline as pipeline
    from backend.history_review_stages import (
        HistoryReviewStage2InputTooLargeError,
    )

    def fail_budget(extraction):
        raise HistoryReviewStage2InputTooLargeError()

    monkeypatch.setattr(
        pipeline,
        "validate_stage2_input_budget",
        fail_budget,
    )

    client = SpyClient([_stage1_raw()])
    with pytest.raises(HistoryReviewPipelineError) as exc_info:
        await pipeline.run_history_review_pipeline(
            client=client,
            sources=_sources(),
        )

    error = exc_info.value
    assert error.code == "history_review_verification_input_too_large"
    assert error.called_llm is True
    assert error.llm_call_count == 1
    assert len(client.calls) == 1
    audit = json.loads(error.audit_raw_output)
    assert [item["stage"] for item in audit["attempts"]] == ["stage1"]


@pytest.mark.anyio
async def test_stage2_output_too_large_does_not_retry():
    client = SpyClient([_stage1_raw(), "x" * 5000])
    with pytest.raises(HistoryReviewPipelineError) as exc_info:
        await run_history_review_pipeline(
            client=client,
            sources=_sources(),
        )

    error = exc_info.value
    assert error.code == "history_review_verification_too_large"
    assert error.llm_call_count == 2
    assert len(client.calls) == 2
    audit = json.loads(error.audit_raw_output)
    assert [item["stage"] for item in audit["attempts"]] == [
        "stage1", "stage2",
    ]


def test_empty_confirmation_suffix_fits_max_stage1_input():
    source = HistoryReviewSourceSnapshot(
        seq=1,
        message_id=101,
        content="测" * 2000,
    )
    validate_stage1_input_budget((source,))
    messages = build_stage1_messages((source,))
    confirmation_messages = _build_empty_confirmation_messages(messages)

    confirmation_tokens = sum(
        estimate_history_review_tokens(message["content"])
        for message in confirmation_messages
    )
    assert confirmation_tokens <= STAGE1_MAX_INPUT_TOKENS
    assert STAGE1_MAX_INPUT_TOKENS - confirmation_tokens >= 16

    max_valid_total = (
        estimate_history_review_tokens(
            confirmation_messages[0]["content"]
        )
        + STAGE1_MAX_USER_JSON_TOKENS
    )
    assert max_valid_total <= STAGE1_MAX_INPUT_TOKENS
    assert STAGE1_MAX_INPUT_TOKENS - max_valid_total >= 16


def test_stage1_structure_coverage_repair_prompt_budget():
    messages = build_stage1_messages(_sources())
    repair_messages = _build_stage1_retry_messages(
        messages,
        category="structure",
    )
    repair_system_tokens = estimate_history_review_tokens(
        repair_messages[0]["content"]
    )

    assert repair_system_tokens <= STAGE1_MAX_SYSTEM_PROMPT_TOKENS
    assert (
        STAGE1_MAX_SYSTEM_PROMPT_TOKENS - repair_system_tokens
    ) >= 16
    assert repair_messages[1] == messages[1]

    max_valid_total = (
        repair_system_tokens + STAGE1_MAX_USER_JSON_TOKENS
    )
    assert max_valid_total <= STAGE1_MAX_INPUT_TOKENS
    assert STAGE1_MAX_INPUT_TOKENS - max_valid_total >= 16


def test_pipeline_result_stage1_only_carries_last_call_diagnostics():
    client = SpyClient([
        _stage1_raw_zero_claims(),
        _stage1_raw_zero_claims(),
    ])

    result = asyncio.run(
        run_history_review_pipeline(client=client, sources=_sources())
    )

    diagnostics = result.last_llm_call_diagnostics
    assert diagnostics is not None
    assert diagnostics.stage == ReviewStage.STAGE1
    assert diagnostics.attempt == 2
    assert diagnostics.mode == ReviewMode.EMPTY_CONFIRMATION
    assert diagnostics.llm_call_index == 2


def test_pipeline_result_stage2_carries_last_call_diagnostics():
    client = SpyClient([_stage1_raw(), _stage2_raw()])

    result = asyncio.run(
        run_history_review_pipeline(client=client, sources=_sources())
    )

    diagnostics = result.last_llm_call_diagnostics
    assert diagnostics is not None
    assert diagnostics.stage == ReviewStage.STAGE2
    assert diagnostics.attempt == 1
    assert diagnostics.mode == ReviewMode.INITIAL
    assert diagnostics.llm_call_index == 2


def test_stage2_pre_call_budget_does_not_reuse_stage1_diagnostics(monkeypatch):
    def fail_stage2_budget(extraction):
        raise HistoryReviewStage2InputTooLargeError()

    monkeypatch.setattr(
        pipeline_module,
        "validate_stage2_input_budget",
        fail_stage2_budget,
    )
    client = SpyClient([_stage1_raw()])

    with pytest.raises(HistoryReviewPipelineError) as exc_info:
        asyncio.run(
            run_history_review_pipeline(client=client, sources=_sources())
        )

    error = exc_info.value
    assert error.code == "history_review_verification_input_too_large"
    assert error.failure_stage == FailureStage.STAGE2
    assert error.failure_attempt == 1
    assert error.failure_mode is None
    assert error.diagnostics is None
    assert error.llm_call_index is None


def test_finish_reason_length_kept_with_parser_error():
    client = SpyClient(["{", "{"])
    client.response_finish_reason = "length"

    with pytest.raises(HistoryReviewPipelineError) as exc_info:
        asyncio.run(
            run_history_review_pipeline(client=client, sources=_sources())
        )

    error = exc_info.value
    assert error.code == "history_review_extraction_json_syntax_error"
    assert error.failure_stage == FailureStage.STAGE1
    assert error.failure_attempt == 2
    assert error.failure_mode == ReviewMode.REPAIR
    assert error.diagnostics is not None
    assert error.diagnostics.finish_reason == "length"


def test_pipeline_cancellation_carries_diagnostics():
    client = SpyClient([asyncio.CancelledError()])

    with pytest.raises(asyncio.CancelledError) as exc_info:
        asyncio.run(
            run_history_review_pipeline(client=client, sources=_sources())
        )

    error = exc_info.value
    assert isinstance(error, HistoryReviewCancelledError)
    diagnostics = error.diagnostics
    assert diagnostics is not None
    assert diagnostics.error_category == (
        ErrorCategory.REVIEW_EXECUTION_INTERRUPTED
    )
    assert diagnostics.stage == ReviewStage.STAGE1
    assert diagnostics.attempt == 1
    assert diagnostics.mode == ReviewMode.INITIAL
    assert diagnostics.call_elapsed_ms is not None


def test_concurrent_pipeline_calls_use_independent_diagnostics():
    class ConcurrentClient:
        def __init__(self):
            self.counter = 0
            self.lock = asyncio.Lock()

        async def generate(
            self,
            messages,
            *,
            response_format=None,
            temperature=None,
            max_tokens=None,
            diagnostics=None,
        ):
            async with self.lock:
                self.counter += 1
                call_id = self.counter
            if diagnostics is not None:
                diagnostics.finish_reason = f"call-{call_id}"
            await asyncio.sleep(0)
            system = messages[0]["content"]
            if system.startswith(HISTORY_REVIEW_STAGE_1_SYSTEM_PROMPT):
                return _stage1_raw()
            return _stage2_raw()

    async def run_two():
        client = ConcurrentClient()
        return await asyncio.gather(
            run_history_review_pipeline(
                client=client, sources=_sources(),
            ),
            run_history_review_pipeline(
                client=client, sources=_sources(),
            ),
        )

    first, second = asyncio.run(run_two())
    first_diag = first.last_llm_call_diagnostics
    second_diag = second.last_llm_call_diagnostics
    assert first_diag is not None
    assert second_diag is not None
    assert first_diag is not second_diag
    assert first_diag.finish_reason is not None
    assert second_diag.finish_reason is not None
    assert first_diag.finish_reason != second_diag.finish_reason
    assert first_diag.llm_call_index == 2
    assert second_diag.llm_call_index == 2
