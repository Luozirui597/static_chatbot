"""Shared diagnostics types for History Review LLM calls.

This module is intentionally dependency-free.  It must not import any
``backend.*`` modules so that it can be used from the LLM client, the
History Review pipeline and the History Review service without creating
import cycles.
"""

from __future__ import annotations

import asyncio
import math
from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum
from typing import Any


class ReviewStage(str, Enum):
    """Canonical stage values for a History Review model call."""

    STAGE1 = "stage1"
    STAGE2 = "stage2"


class FailureStage(str, Enum):
    """Canonical values for where a History Review failure occurred."""

    STAGE1 = "stage1"
    STAGE2 = "stage2"
    FINALIZE = "finalize"
    SERVICE = "service"


class ReviewMode(str, Enum):
    """Canonical values for a stage attempt mode."""

    INITIAL = "initial"
    REPAIR = "repair"
    EMPTY_CONFIRMATION = "empty_confirmation"


class ErrorCategory(str, Enum):
    """Stable internal error categories for diagnostics."""

    # Transport and timeout categories.
    UPSTREAM_CONNECT_TIMEOUT = "upstream_connect_timeout"
    UPSTREAM_READ_TIMEOUT = "upstream_read_timeout"
    UPSTREAM_WRITE_TIMEOUT = "upstream_write_timeout"
    UPSTREAM_POOL_TIMEOUT = "upstream_pool_timeout"
    UPSTREAM_TIMEOUT_UNKNOWN = "upstream_timeout_unknown"
    UPSTREAM_CONNECT_ERROR = "upstream_connect_error"
    UPSTREAM_READ_ERROR = "upstream_read_error"
    UPSTREAM_WRITE_ERROR = "upstream_write_error"
    UPSTREAM_REMOTE_PROTOCOL_ERROR = "upstream_remote_protocol_error"
    UPSTREAM_PROXY_ERROR = "upstream_proxy_error"
    UPSTREAM_UNSUPPORTED_PROTOCOL = "upstream_unsupported_protocol"
    UPSTREAM_TRANSPORT_DECODE_ERROR = "upstream_transport_decode_error"
    UPSTREAM_TRANSPORT_ERROR_UNKNOWN = "upstream_transport_error_unknown"

    # Upstream HTTP status categories.
    UPSTREAM_HTTP_400 = "upstream_http_400"
    UPSTREAM_HTTP_401 = "upstream_http_401"
    UPSTREAM_HTTP_402 = "upstream_http_402"
    UPSTREAM_HTTP_429 = "upstream_http_429"
    UPSTREAM_HTTP_5XX = "upstream_http_5xx"
    UPSTREAM_HTTP_OTHER = "upstream_http_other"

    # HTTP 200 response structure and content categories.
    UPSTREAM_RESPONSE_SHAPE_ERROR = "upstream_response_shape_error"
    UPSTREAM_RESPONSE_MISSING_CONTENT = "upstream_response_missing_content"
    UPSTREAM_RESPONSE_NULL_CONTENT = "upstream_response_null_content"
    UPSTREAM_RESPONSE_NON_STRING_CONTENT = (
        "upstream_response_non_string_content"
    )
    UPSTREAM_RESPONSE_EMPTY_CONTENT = "upstream_response_empty_content"
    UPSTREAM_RESPONSE_INVALID_JSON = "upstream_response_invalid_json"
    UPSTREAM_RESPONSE_FILTERED = "upstream_response_filtered"
    UPSTREAM_RESPONSE_UNEXPECTED_FINISH = (
        "upstream_response_unexpected_finish"
    )

    # Final History Review failure categories.
    REVIEW_OUTPUT_TRUNCATED = "review_output_truncated"
    REVIEW_OUTPUT_FILTERED = "review_output_filtered"
    REVIEW_JSON_SYNTAX = "review_json_syntax"
    REVIEW_JSON_STRUCTURE = "review_json_structure"
    REVIEW_JSON_SEMANTIC = "review_json_semantic"
    REVIEW_SOURCE_COVERAGE = "review_source_coverage"
    REVIEW_SOURCE_REFERENCE = "review_source_reference"
    REVIEW_CLAIM_COVERAGE = "review_claim_coverage"
    REVIEW_CLAIM_REFERENCE = "review_claim_reference"
    REVIEW_INPUT_TOO_LARGE = "review_input_too_large"
    REVIEW_OUTPUT_TOO_LARGE = "review_output_too_large"
    REVIEW_CAPACITY_EXCEEDED = "review_capacity_exceeded"
    REVIEW_INTERNAL_ERROR = "review_internal_error"
    REVIEW_SERVICE_ERROR = "review_service_error"
    REVIEW_FINALIZE_FAILED = "review_finalize_failed"
    REVIEW_EXECUTION_INTERRUPTED = "review_execution_interrupted"
    REVIEW_PROMPT_VERSION_UNSUPPORTED = "review_prompt_version_unsupported"


def clean_optional_string(
    value: object,
    *,
    max_length: int,
) -> str | None:
    """Return a bounded, control-character-free string or None."""

    if not isinstance(value, str):
        return None
    cleaned_chars: list[str] = []
    for char in value:
        codepoint = ord(char)
        if (
            codepoint <= 0x1F
            or 0x7F <= codepoint <= 0x9F
            or codepoint in (0x2028, 0x2029)
        ):
            continue
        cleaned_chars.append(char)
    cleaned = "".join(cleaned_chars).strip()
    if not cleaned:
        return None
    if max_length < 1:
        raise ValueError("max_length must be positive")
    return cleaned[:max_length]


def clean_optional_int(
    value: object,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int | None:
    """Return a bounded, non-bool int or None."""

    if isinstance(value, bool) or not isinstance(value, int):
        return None
    if minimum is not None and value < minimum:
        return None
    if maximum is not None and value > maximum:
        return None
    return value


def clean_optional_float(
    value: object,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float | None:
    """Return a bounded finite float or None."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    if not math.isfinite(result):
        return None
    if minimum is not None and result < minimum:
        return None
    if maximum is not None and result > maximum:
        return None
    return result


@dataclass
class TokenUsage:
    """Token accounting reported by the upstream response."""

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


@dataclass
class LLMCallDiagnostics:
    """Diagnostics for one actual History Review model call."""

    stage: ReviewStage | None = None
    attempt: int | None = None
    mode: ReviewMode | None = None
    llm_call_index: int | None = None

    error_category: ErrorCategory | None = None

    exception_type: str | None = None
    http_status: int | None = None
    retry_after_seconds: float | None = None
    timeout_configured_seconds: float | None = None
    call_elapsed_ms: int | None = None

    finish_reason: str | None = None
    token_usage: TokenUsage | None = None
    response_content_chars: int | None = None
    upstream_request_id: str | None = None

    upstream_error_code: str | None = None
    upstream_error_type: str | None = None


class HistoryReviewCancelledError(asyncio.CancelledError):
    """Cancellation carrying the diagnostics known at cancellation time."""

    def __init__(
        self,
        diagnostics: LLMCallDiagnostics | None = None,
    ) -> None:
        super().__init__()
        self.diagnostics = diagnostics


@dataclass
class HistoryReviewFailureDetails:
    """Review-level diagnostic object persisted in ``error_details``."""

    version: str = "history-review-diagnostics-v1"
    error_code: str | None = None
    error_category: ErrorCategory | None = None

    failure_stage: FailureStage | None = None
    failure_attempt: int | None = None
    failure_mode: ReviewMode | None = None

    llm_call_index: int | None = None
    llm: LLMCallDiagnostics | None = None

    review_elapsed_ms: int | None = None


def _enum_value(value: object) -> object:
    if isinstance(value, Enum):
        return value.value
    return value


def _jsonable(value: object) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return {
            key: _jsonable(item)
            for key, item in asdict(value).items()
        }
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def diagnostics_to_dict(value: object) -> dict:
    """Return a JSON-serializable dict for a dataclass object."""

    if not is_dataclass(value) or isinstance(value, type):
        raise TypeError("value must be a dataclass instance")
    converted = _jsonable(value)
    if not isinstance(converted, dict):
        raise TypeError("dataclass did not serialize to a dict")
    return converted


def error_category_value(value: ErrorCategory | None) -> str | None:
    """Return a category's string value without changing enum types."""

    return None if value is None else value.value
