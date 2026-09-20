"""Pure two-stage History Review orchestration.

This module is a pure async coordinator around the frozen Phase 2A
stage contracts.  It calls only the injected ``LLMClient`` and performs
no database, profile, HTTP, or service-layer work.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TypeVar

from backend.history_review_parser import ParsedHistoryReviewOutput
from backend.history_review_prompt import HistoryReviewSourceSnapshot
from backend.history_review_selection import estimate_history_review_tokens
from backend.history_review_stages import (
    HISTORY_REVIEW_PIPELINE_VERSION,
    STAGE1_MAX_INPUT_TOKENS,
    STAGE1_RESERVED_OUTPUT_TOKENS,
    STAGE2_MAX_INPUT_TOKENS,
    STAGE2_RESERVED_OUTPUT_TOKENS,
    Stage1Extraction,
    Stage2Verification,
    HistoryReviewStage1CapacityExceededError,
    HistoryReviewStage1JsonSemanticError,
    HistoryReviewStage1JsonStructureError,
    HistoryReviewStage1JsonSyntaxError,
    HistoryReviewStage1QuoteInvalidError,
    HistoryReviewStage1SourceInvalidError,
    HistoryReviewStage1SourceUncoveredError,
    HistoryReviewStage2ClaimInvalidError,
    HistoryReviewStage2ClaimUncoveredError,
    HistoryReviewStage2JsonSemanticError,
    HistoryReviewStage2JsonStructureError,
    HistoryReviewStage2JsonSyntaxError,
    HistoryReviewStageError,
    adapt_stage_results,
    build_stage1_messages,
    build_stage2_messages,
    parse_stage1_extraction,
    parse_stage2_verification,
    validate_stage1_input_budget,
    validate_stage2_input_budget,
)
from backend.llm_client import LLMClient, LLMMessage

T = TypeVar("T")

HISTORY_REVIEW_LLM_NON_STRING_RESPONSE = (
    "history_review_llm_non_string_response"
)
HISTORY_REVIEW_LLM_EMPTY_RESPONSE = "history_review_llm_empty_response"
HISTORY_REVIEW_EXTRACTION_INPUT_TOO_LARGE = (
    "history_review_extraction_input_too_large"
)
HISTORY_REVIEW_VERIFICATION_INPUT_TOO_LARGE = (
    "history_review_verification_input_too_large"
)

HISTORY_REVIEW_RETRY_REPAIR_TEMPLATE = (
    "上一次输出未通过机器校验，错误类别为 {category}。"
    "请基于完全相同的输入重新输出完整 JSON。"
    "严格遵守覆盖、顺序、引用和字段契约；不要只输出修补片段。"
)

_STAGE1_RETRYABLE_ERRORS = (
    HistoryReviewStage1JsonSyntaxError,
    HistoryReviewStage1JsonStructureError,
    HistoryReviewStage1JsonSemanticError,
    HistoryReviewStage1SourceInvalidError,
    HistoryReviewStage1SourceUncoveredError,
    HistoryReviewStage1QuoteInvalidError,
)

_STAGE2_RETRYABLE_ERRORS = (
    HistoryReviewStage2JsonSyntaxError,
    HistoryReviewStage2JsonStructureError,
    HistoryReviewStage2JsonSemanticError,
    HistoryReviewStage2ClaimInvalidError,
    HistoryReviewStage2ClaimUncoveredError,
)


@dataclass(frozen=True)
class HistoryReviewPipelineResult:
    parsed: ParsedHistoryReviewOutput
    audit_raw_output: str
    llm_call_count: int


class HistoryReviewPipelineError(ValueError):
    """Stable pipeline error with audit and call metadata."""

    def __init__(
        self,
        *,
        code: str,
        audit_raw_output: str | None,
        called_llm: bool,
        llm_call_count: int,
        message: str | None = None,
    ) -> None:
        self.code = code
        self.audit_raw_output = audit_raw_output
        self.called_llm = called_llm
        self.llm_call_count = llm_call_count
        super().__init__(message or "History review pipeline failed.")


class _PipelineAudit:
    """Collects deterministic per-call audit records."""

    def __init__(self) -> None:
        self._attempts: list[dict] = []
        self._stage_attempt: dict[str, int] = {}

    def record(self, *, stage: str, raw_output: str) -> None:
        attempt = self._stage_attempt.get(stage, 0) + 1
        self._stage_attempt[stage] = attempt
        self._attempts.append({
            "attempt": attempt,
            "raw_output": raw_output,
            "stage": stage,
        })

    def render(self) -> str | None:
        if not self._attempts:
            return None
        payload = {
            "attempts": self._attempts,
            "pipeline_version": HISTORY_REVIEW_PIPELINE_VERSION,
        }
        return json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )


class _PipelineState:
    def __init__(self) -> None:
        self.audit = _PipelineAudit()
        self.called_llm = False
        self.llm_call_count = 0


def _pipeline_error(
    state: _PipelineState,
    code: str,
) -> HistoryReviewPipelineError:
    return HistoryReviewPipelineError(
        code=code,
        audit_raw_output=state.audit.render(),
        called_llm=state.called_llm,
        llm_call_count=state.llm_call_count,
    )


def _build_retry_messages(
    messages: Sequence[LLMMessage],
    *,
    category: str,
) -> list[LLMMessage]:
    repair = HISTORY_REVIEW_RETRY_REPAIR_TEMPLATE.format(
        category=category,
    )
    copied = [dict(message) for message in messages]
    copied[0] = {
        "role": "system",
        "content": copied[0]["content"] + "\n" + repair,
    }
    return copied


def _stage1_retry_category(error: Exception) -> str:
    if isinstance(error, HistoryReviewStage1JsonSyntaxError):
        return "syntax"
    if isinstance(error, HistoryReviewStage1JsonStructureError):
        return "structure"
    if isinstance(error, HistoryReviewStage1JsonSemanticError):
        return "semantic"
    if isinstance(error, (
        HistoryReviewStage1SourceInvalidError,
        HistoryReviewStage1SourceUncoveredError,
    )):
        return "coverage"
    if isinstance(error, HistoryReviewStage1QuoteInvalidError):
        return "reference"
    return "contract"


def _stage2_retry_category(error: Exception) -> str:
    if isinstance(error, HistoryReviewStage2JsonSyntaxError):
        return "syntax"
    if isinstance(error, HistoryReviewStage2JsonStructureError):
        return "structure"
    if isinstance(error, HistoryReviewStage2JsonSemanticError):
        return "semantic"
    if isinstance(error, HistoryReviewStage2ClaimUncoveredError):
        return "coverage"
    if isinstance(error, HistoryReviewStage2ClaimInvalidError):
        return "reference"
    return "contract"


def _messages_token_estimate(messages: Sequence[LLMMessage]) -> int:
    return sum(
        estimate_history_review_tokens(message["content"])
        for message in messages
    )


async def _run_stage(
    *,
    stage: str,
    client: LLMClient,
    messages: list[LLMMessage],
    max_tokens: int,
    max_input_tokens: int,
    input_too_large_code: str,
    parser: Callable[[str], T],
    retryable_errors: tuple[type[Exception], ...],
    retry_category: Callable[[Exception], str],
    state: _PipelineState,
) -> T:
    last_error: Exception | None = None

    for attempt in (1, 2):
        current_messages = messages
        if attempt == 2:
            assert last_error is not None
            current_messages = _build_retry_messages(
                messages,
                category=retry_category(last_error),
            )
            if _messages_token_estimate(current_messages) > max_input_tokens:
                raise _pipeline_error(state, input_too_large_code)

        state.called_llm = True
        state.llm_call_count += 1
        raw_output = await client.generate(
            current_messages,
            response_format={"type": "json_object"},
            temperature=0,
            max_tokens=max_tokens,
        )

        if not isinstance(raw_output, str):
            raise _pipeline_error(
                state, HISTORY_REVIEW_LLM_NON_STRING_RESPONSE,
            )
        if not raw_output.strip():
            raise _pipeline_error(
                state, HISTORY_REVIEW_LLM_EMPTY_RESPONSE,
            )

        state.audit.record(stage=stage, raw_output=raw_output)

        try:
            return parser(raw_output)
        except retryable_errors as exc:
            last_error = exc
            if attempt == 1:
                continue
            raise _pipeline_error(state, exc.code) from exc
        except HistoryReviewStageError as exc:
            raise _pipeline_error(state, exc.code) from exc

    raise AssertionError("unreachable")


async def run_history_review_pipeline(
    *,
    client: LLMClient,
    sources: Sequence[HistoryReviewSourceSnapshot],
) -> HistoryReviewPipelineResult:
    """Run Stage 1 + optional Stage 2 and return the parsed final result."""

    state = _PipelineState()

    try:
        validate_stage1_input_budget(sources)
    except HistoryReviewStageError as exc:
        raise _pipeline_error(state, exc.code) from exc

    stage1_messages = build_stage1_messages(sources)
    extraction = await _run_stage(
        stage="stage1",
        client=client,
        messages=stage1_messages,
        max_tokens=STAGE1_RESERVED_OUTPUT_TOKENS,
        max_input_tokens=STAGE1_MAX_INPUT_TOKENS,
        input_too_large_code=HISTORY_REVIEW_EXTRACTION_INPUT_TOO_LARGE,
        parser=lambda raw: parse_stage1_extraction(
            raw_output=raw,
            sources=sources,
        ),
        retryable_errors=_STAGE1_RETRYABLE_ERRORS,
        retry_category=_stage1_retry_category,
        state=state,
    )

    if extraction.claims:
        try:
            validate_stage2_input_budget(extraction)
        except HistoryReviewStageError as exc:
            raise _pipeline_error(state, exc.code) from exc

        stage2_messages = build_stage2_messages(extraction)
        verification = await _run_stage(
            stage="stage2",
            client=client,
            messages=stage2_messages,
            max_tokens=STAGE2_RESERVED_OUTPUT_TOKENS,
            max_input_tokens=STAGE2_MAX_INPUT_TOKENS,
            input_too_large_code=(
                HISTORY_REVIEW_VERIFICATION_INPUT_TOO_LARGE
            ),
            parser=lambda raw: parse_stage2_verification(
                raw_output=raw,
                extraction=extraction,
            ),
            retryable_errors=_STAGE2_RETRYABLE_ERRORS,
            retry_category=_stage2_retry_category,
            state=state,
        )
    else:
        verification = Stage2Verification(claims=())

    try:
        parsed = adapt_stage_results(
            frozen_sources=sources,
            extraction=extraction,
            verification=verification,
        )
    except HistoryReviewStageError as exc:
        raise _pipeline_error(state, exc.code) from exc

    return HistoryReviewPipelineResult(
        parsed=parsed,
        audit_raw_output=state.audit.render() or "",
        llm_call_count=state.llm_call_count,
    )
