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
    STAGE1_MAX_INPUT_TOKENS,
    STAGE1_RESERVED_OUTPUT_TOKENS,
    STAGE2_RESERVED_OUTPUT_TOKENS,
)
from backend.history_review_pipeline import (
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

    async def generate(
        self,
        messages,
        *,
        response_format=None,
        temperature=None,
        max_tokens=None,
    ):
        self.calls.append({
            "messages": [dict(message) for message in messages],
            "response_format": response_format,
            "temperature": temperature,
            "max_tokens": max_tokens,
        })
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
async def test_zero_claim_stage1_short_circuits_stage2():
    client = SpyClient([_stage1_raw_zero_claims()])

    result = await run_history_review_pipeline(
        client=client,
        sources=_sources(),
    )

    assert result.llm_call_count == 1
    assert len(client.calls) == 1
    assert [finding.verdict for finding in result.parsed.findings] == [
        "not_a_claim", "not_a_claim", "not_a_claim",
    ]
    assert [finding.source_message_id for finding in result.parsed.findings] == [
        101, 205, 309,
    ]
    assert len(client.calls[0]["messages"]) == 2


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
                    "raw_output": stage1_raw,
                    "stage": "stage1",
                },
                {
                    "attempt": 1,
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
