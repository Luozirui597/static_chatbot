"""Pure two-stage History Review orchestration.

This module is a pure async coordinator around the frozen Phase 2A
stage contracts.  It calls only the injected ``LLMClient`` and performs
no database, profile, HTTP, or service-layer work.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TypeVar

from backend.exceptions import LLMError

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
from backend.review_diagnostics import (
    ErrorCategory,
    FailureStage,
    HistoryReviewCancelledError,
    LLMCallDiagnostics,
    ReviewMode,
    ReviewStage,
)

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

HISTORY_REVIEW_STAGE1_STRUCTURE_COVERAGE_HINT = (
    "顶层仅 overflow/sources，禁止顶层 claims；"
    "sources 须按序覆盖全部输入 ID，无事实也保留该 source 且 claims=[]。"
)

HISTORY_REVIEW_EMPTY_CONFIRMATION_TEMPLATE = (
    "上一次输出 JSON 合法，但所有 source 的 claims 都为空。"
    "请基于完全相同的输入重新检查每条 source：操作指令不产生 claim，"
    "但混合在指令中的可独立验证事实必须提取；不得遗漏，不得虚构。"
    "重新输出完整 JSON。"
)

AUDIT_MODE_INITIAL = "initial"
AUDIT_MODE_REPAIR = "repair"
AUDIT_MODE_EMPTY_CONFIRMATION = "empty_confirmation"

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
    last_llm_call_diagnostics: LLMCallDiagnostics | None = None


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
        failure_stage: FailureStage | None = None,
        failure_attempt: int | None = None,
        failure_mode: ReviewMode | None = None,
        llm_call_index: int | None = None,
        diagnostics: LLMCallDiagnostics | None = None,
    ) -> None:
        self.code = code
        self.audit_raw_output = audit_raw_output
        self.called_llm = called_llm
        self.llm_call_count = llm_call_count
        self.failure_stage = failure_stage
        self.failure_attempt = failure_attempt
        self.failure_mode = failure_mode
        self.llm_call_index = llm_call_index
        self.diagnostics = diagnostics
        super().__init__(message or "History review pipeline failed.")


class _PipelineAudit:
    """Collects deterministic per-call audit records."""

    def __init__(self) -> None:
        self._attempts: list[dict] = []
        self._stage_attempt: dict[str, int] = {}

    def record(
        self,
        *,
        stage: str,
        raw_output: str,
        mode: str,
    ) -> None:
        if mode not in (
            AUDIT_MODE_INITIAL,
            AUDIT_MODE_REPAIR,
            AUDIT_MODE_EMPTY_CONFIRMATION,
        ):
            raise ValueError("invalid pipeline audit mode")
        attempt = self._stage_attempt.get(stage, 0) + 1
        self._stage_attempt[stage] = attempt
        self._attempts.append({
            "attempt": attempt,
            "mode": mode,
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
        self.last_call_diagnostics: LLMCallDiagnostics | None = None


def _pipeline_error(
    state: _PipelineState,
    code: str,
    *,
    failure_stage: FailureStage | None = None,
    failure_attempt: int | None = None,
    failure_mode: ReviewMode | None = None,
    include_last_call: bool = False,
) -> HistoryReviewPipelineError:
    diagnostics = state.last_call_diagnostics if include_last_call else None
    llm_call_index = (
        diagnostics.llm_call_index
        if diagnostics is not None
        else None
    )
    return HistoryReviewPipelineError(
        code=code,
        audit_raw_output=state.audit.render(),
        called_llm=state.called_llm,
        llm_call_count=state.llm_call_count,
        failure_stage=failure_stage,
        failure_attempt=failure_attempt,
        failure_mode=failure_mode,
        llm_call_index=llm_call_index,
        diagnostics=diagnostics,
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


def _build_empty_confirmation_messages(
    messages: Sequence[LLMMessage],
) -> list[LLMMessage]:
    copied = [dict(message) for message in messages]
    copied[0] = {
        "role": "system",
        "content": (
            copied[0]["content"]
            + "\n"
            + HISTORY_REVIEW_EMPTY_CONFIRMATION_TEMPLATE
        ),
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


def _stage1_retry_system_hint(category: str) -> str | None:
    if category in ("structure", "coverage"):
        return HISTORY_REVIEW_STAGE1_STRUCTURE_COVERAGE_HINT
    return None


def _build_stage1_retry_messages(
    messages: Sequence[LLMMessage],
    *,
    category: str,
) -> list[LLMMessage]:
    repaired_messages = _build_retry_messages(
        messages,
        category=category,
    )
    system_hint = _stage1_retry_system_hint(category)
    if system_hint is None:
        return repaired_messages
    copied = [dict(message) for message in repaired_messages]
    copied[0] = {
        "role": "system",
        "content": copied[0]["content"] + "\n" + system_hint,
    }
    return copied


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


def _elapsed_ms(start: float) -> int:
    return max(0, int((time.perf_counter() - start) * 1000))


def _ensure_diagnostics_context(
    exc: LLMError,
    diagnostics: LLMCallDiagnostics,
    start: float,
) -> None:
    if exc.diagnostics is None:
        diagnostics.call_elapsed_ms = _elapsed_ms(start)
        exc.diagnostics = diagnostics
        return
    current = exc.diagnostics
    if current.stage is None:
        current.stage = diagnostics.stage
    if current.attempt is None:
        current.attempt = diagnostics.attempt
    if current.mode is None:
        current.mode = diagnostics.mode
    if current.llm_call_index is None:
        current.llm_call_index = diagnostics.llm_call_index
    if current.call_elapsed_ms is None:
        current.call_elapsed_ms = _elapsed_ms(start)
    if current.timeout_configured_seconds is None:
        current.timeout_configured_seconds = (
            diagnostics.timeout_configured_seconds
        )


async def _call_stage_once(
    *,
    stage: str,
    attempt: int,
    client: LLMClient,
    messages: list[LLMMessage],
    max_tokens: int,
    mode: str,
    state: _PipelineState,
) -> str:
    state.called_llm = True
    state.llm_call_count += 1
    diagnostics = LLMCallDiagnostics(
        stage=ReviewStage(stage),
        attempt=attempt,
        mode=ReviewMode(mode),
        llm_call_index=state.llm_call_count,
    )
    state.last_call_diagnostics = diagnostics
    start = time.perf_counter()

    try:
        raw_output = await client.generate(
            messages,
            response_format={"type": "json_object"},
            temperature=0,
            max_tokens=max_tokens,
            diagnostics=diagnostics,
        )
    except HistoryReviewCancelledError:
        raise
    except asyncio.CancelledError:
        diagnostics.error_category = (
            ErrorCategory.REVIEW_EXECUTION_INTERRUPTED
        )
        diagnostics.call_elapsed_ms = _elapsed_ms(start)
        raise HistoryReviewCancelledError(diagnostics) from None
    except LLMError as exc:
        _ensure_diagnostics_context(exc, diagnostics, start)
        state.last_call_diagnostics = exc.diagnostics or diagnostics
        raise

    if diagnostics.call_elapsed_ms is None:
        diagnostics.call_elapsed_ms = _elapsed_ms(start)

    if not isinstance(raw_output, str):
        raise _pipeline_error(
            state,
            HISTORY_REVIEW_LLM_NON_STRING_RESPONSE,
            failure_stage=FailureStage(stage),
            failure_attempt=attempt,
            failure_mode=ReviewMode(mode),
            include_last_call=True,
        )
    if not raw_output.strip():
        raise _pipeline_error(
            state,
            HISTORY_REVIEW_LLM_EMPTY_RESPONSE,
            failure_stage=FailureStage(stage),
            failure_attempt=attempt,
            failure_mode=ReviewMode(mode),
            include_last_call=True,
        )

    state.audit.record(
        stage=stage,
        raw_output=raw_output,
        mode=mode,
    )
    return raw_output


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
        mode = AUDIT_MODE_INITIAL
        if attempt == 2:
            assert last_error is not None
            mode = AUDIT_MODE_REPAIR
            current_messages = _build_retry_messages(
                messages,
                category=retry_category(last_error),
            )
            if _messages_token_estimate(current_messages) > max_input_tokens:
                raise _pipeline_error(
                    state,
                    input_too_large_code,
                    failure_stage=FailureStage(stage),
                    failure_attempt=attempt,
                    failure_mode=ReviewMode(mode),
                    include_last_call=False,
                )

        raw_output = await _call_stage_once(
            stage=stage,
            attempt=attempt,
            client=client,
            messages=current_messages,
            max_tokens=max_tokens,
            mode=mode,
            state=state,
        )

        try:
            return parser(raw_output)
        except retryable_errors as exc:
            last_error = exc
            if attempt == 1:
                continue
            raise _pipeline_error(
                state,
                exc.code,
                failure_stage=FailureStage(stage),
                failure_attempt=attempt,
                failure_mode=ReviewMode(mode),
                include_last_call=True,
            ) from exc
        except HistoryReviewStageError as exc:
            raise _pipeline_error(
                state,
                exc.code,
                failure_stage=FailureStage(stage),
                failure_attempt=attempt,
                failure_mode=ReviewMode(mode),
                include_last_call=True,
            ) from exc

    raise AssertionError("unreachable")


async def _run_stage1(
    *,
    client: LLMClient,
    messages: list[LLMMessage],
    max_tokens: int,
    max_input_tokens: int,
    input_too_large_code: str,
    parser: Callable[[str], Stage1Extraction],
    state: _PipelineState,
) -> Stage1Extraction:
    raw_output = await _call_stage_once(
        stage="stage1",
        attempt=1,
        client=client,
        messages=messages,
        max_tokens=max_tokens,
        mode=AUDIT_MODE_INITIAL,
        state=state,
    )

    try:
        extraction = parser(raw_output)
    except _STAGE1_RETRYABLE_ERRORS as exc:
        second_mode = AUDIT_MODE_REPAIR
        category = _stage1_retry_category(exc)
        second_messages = _build_stage1_retry_messages(
            messages,
            category=category,
        )
    except HistoryReviewStageError as exc:
        raise _pipeline_error(
            state,
            exc.code,
            failure_stage=FailureStage.STAGE1,
            failure_attempt=1,
            failure_mode=ReviewMode.INITIAL,
            include_last_call=True,
        ) from exc
    else:
        if extraction.claims:
            return extraction
        second_mode = AUDIT_MODE_EMPTY_CONFIRMATION
        second_messages = _build_empty_confirmation_messages(messages)

    if _messages_token_estimate(second_messages) > max_input_tokens:
        raise _pipeline_error(
            state,
            input_too_large_code,
            failure_stage=FailureStage.STAGE1,
            failure_attempt=2,
            failure_mode=ReviewMode(second_mode),
            include_last_call=False,
        )

    second_raw_output = await _call_stage_once(
        stage="stage1",
        attempt=2,
        client=client,
        messages=second_messages,
        max_tokens=max_tokens,
        mode=second_mode,
        state=state,
    )

    try:
        return parser(second_raw_output)
    except HistoryReviewStageError as exc:
        raise _pipeline_error(
            state,
            exc.code,
            failure_stage=FailureStage.STAGE1,
            failure_attempt=2,
            failure_mode=ReviewMode(second_mode),
            include_last_call=True,
        ) from exc


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
        raise _pipeline_error(
            state,
            exc.code,
            failure_stage=FailureStage.STAGE1,
            failure_attempt=1,
            failure_mode=None,
            include_last_call=False,
        ) from exc

    stage1_messages = build_stage1_messages(sources)
    extraction = await _run_stage1(
        client=client,
        messages=stage1_messages,
        max_tokens=STAGE1_RESERVED_OUTPUT_TOKENS,
        max_input_tokens=STAGE1_MAX_INPUT_TOKENS,
        input_too_large_code=HISTORY_REVIEW_EXTRACTION_INPUT_TOO_LARGE,
        parser=lambda raw: parse_stage1_extraction(
            raw_output=raw,
            sources=sources,
        ),
        state=state,
    )

    if extraction.claims:
        try:
            validate_stage2_input_budget(extraction)
        except HistoryReviewStageError as exc:
            raise _pipeline_error(
                state,
                exc.code,
                failure_stage=FailureStage.STAGE2,
                failure_attempt=1,
                failure_mode=None,
                include_last_call=False,
            ) from exc

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
        raise _pipeline_error(
            state,
            exc.code,
            failure_stage=FailureStage.SERVICE,
            failure_attempt=None,
            failure_mode=None,
            include_last_call=True,
        ) from exc

    return HistoryReviewPipelineResult(
        parsed=parsed,
        audit_raw_output=state.audit.render() or "",
        llm_call_count=state.llm_call_count,
        last_llm_call_diagnostics=state.last_call_diagnostics,
    )
