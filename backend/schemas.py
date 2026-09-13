"""Request and response models with input validation."""

import re
from datetime import datetime
from typing import Annotated, Literal

from fastapi import Path
from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from backend.interaction_modes import (
    CORRECTIVE_MODE,
    RECEIVE_TEACHING_MODE,
)
from backend.llm_profiles import SessionProfileStatus
from backend.session_titles import normalize_title_whitespace

MAX_MESSAGE_LENGTH = 4000

InteractionMode = Literal[RECEIVE_TEACHING_MODE, CORRECTIVE_MODE]


class ChatRequest(BaseModel):
    """Incoming chat message with validation.

    Rejects empty strings, whitespace-only strings, and messages
    exceeding MAX_MESSAGE_LENGTH characters.  Leading / trailing
    whitespace is stripped; internal whitespace and newlines are
    preserved.
    """

    message: str = Field(
        min_length=1,
        max_length=MAX_MESSAGE_LENGTH,
        description="User message text",
    )

    @field_validator("message")
    @classmethod
    def not_blank(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("message must not be blank")
        return stripped


class ChatResponse(BaseModel):
    """Chat reply returned to the client."""

    reply: str


# ---------------------------------------------------------------------------
# Session models
# ---------------------------------------------------------------------------


class SessionResponse(BaseModel):
    """Public representation of a chat session."""

    id: int
    title: str
    interaction_mode: InteractionMode = RECEIVE_TEACHING_MODE
    created_at: datetime
    updated_at: datetime
    llm_profile_id: str
    llm_profile_label: str
    llm_profile_status: SessionProfileStatus
    llm_model_snapshot: str | None

    model_config = ConfigDict(from_attributes=True)


class MessageResponse(BaseModel):
    """Public representation of a single message."""

    id: int
    session_id: int
    role: Literal["user", "assistant"]
    content: str
    created_at: datetime
    interaction_mode_snapshot: InteractionMode | None = None
    prompt_version_snapshot: Annotated[
        str, Field(strict=True, min_length=1, max_length=50)
    ] | None = None
    llm_profile_id_snapshot: Annotated[
        str, Field(strict=True, min_length=1, max_length=50)
    ] | None = None
    llm_profile_kind_snapshot: Literal["fake", "api", "local"] | None = None
    llm_model_snapshot: Annotated[
        str, Field(strict=True, min_length=1, max_length=255)
    ] | None = None

    model_config = ConfigDict(from_attributes=True)

    @model_validator(mode="after")
    def _snapshot_triple_consistency(self) -> "MessageResponse":
        """Enforce the provenance triple invariant.

        Legal states are exactly two: all three snapshot fields are
        NULL (messages created before snapshot tracking), or all three
        are non-blank, correctly-typed values.  Anything in between —
        one or two NULLs, empty or whitespace-only strings — is
        rejected rather than silently fabricated.
        """
        snapshot_values = (
            self.llm_profile_id_snapshot,
            self.llm_profile_kind_snapshot,
            self.llm_model_snapshot,
        )
        if all(v is None for v in snapshot_values):
            return self
        if all(
            isinstance(v, str) and v.strip() != "" for v in snapshot_values
        ):
            return self
        raise ValueError(
            "llm snapshot fields must be all null or all non-blank"
        )


class SendMessageResponse(BaseModel):
    """Response for POST /api/sessions/{session_id}/messages."""

    user_message: MessageResponse
    assistant_message: MessageResponse


class RenameSessionRequest(BaseModel):
    """Request body for PATCH /api/sessions/{session_id}.

    Whitespace is collapsed before validation: newlines, tabs, and
    consecutive spaces become a single space, then leading / trailing
    whitespace is stripped.  The normalised result must be non-empty
    and at most 255 characters (the database column limit).

    Manual titles are **not** truncated — exceeding 255 characters
    after normalisation is a validation error (422).
    """

    title: str = Field(description="New session title")

    @field_validator("title", mode="before")
    @classmethod
    def _normalize_and_validate(cls, v: object) -> str:
        if not isinstance(v, str):
            raise ValueError("title must be a string")  # noqa: TRY004
        normalised = normalize_title_whitespace(v)
        if not normalised:
            raise ValueError("title must not be blank")
        if len(normalised) > 255:
            raise ValueError("title must not exceed 255 characters")
        return normalised


# ---------------------------------------------------------------------------
# Profile models
# ---------------------------------------------------------------------------

_LLM_PROFILE_ID_PATTERN = r"^[a-z][a-z0-9-]*$"


def _fullmatch_llm_profile_id(value: str) -> str:
    """Reject values that only match the pattern as a prefix (e.g. a
    trailing newline would pass ``re.match`` with ``$``)."""
    if not re.fullmatch(_LLM_PROFILE_ID_PATTERN, value):
        raise ValueError(
            f"llm_profile_id must match {_LLM_PROFILE_ID_PATTERN}"
        )
    return value


# Shared, single-source profile-id validation used by every request
# model that carries an llm_profile_id.  Strict strings only (no type
# coercion), 1-50 chars, ^[a-z][a-z0-9-]*$ enforced by fullmatch.
LLMProfileId = Annotated[
    str,
    Field(
        strict=True,
        min_length=1,
        max_length=50,
        pattern=_LLM_PROFILE_ID_PATTERN,
    ),
    AfterValidator(_fullmatch_llm_profile_id),
]


class CreateSessionRequest(BaseModel):
    """Request body for POST /api/sessions.

    *llm_profile_id* is validated strictly — no type coercion, no
    trailing-newline bypass.
    """

    llm_profile_id: LLMProfileId = "default"


class SwitchSessionProfileRequest(BaseModel):
    """Request body for PATCH /api/sessions/{id}/llm-profile."""

    llm_profile_id: LLMProfileId
    acknowledge_remote_history: bool = Field(default=False, strict=True)


class RemoteHistoryAckRequiredDetail(BaseModel):
    """Structured 409 detail for the remote-history acknowledgement."""

    code: Literal["remote_history_ack_required"]
    message: str


class LLMProfilePublic(BaseModel):
    """Public representation of an LLM profile.

    Does **not** expose API keys, base URLs, reasoning effort, clients,
    or any other request configuration.
    """

    id: str
    label: str
    kind: Literal["fake", "api", "local"]
    model: str
    is_default: bool


class DeleteResponse(BaseModel):
    """Confirmation that a resource was deleted."""

    ok: bool


class SwitchInteractionModeRequest(BaseModel):
    """Request body for PATCH /api/sessions/{id}/interaction-mode."""

    interaction_mode: InteractionMode


class ModeSwitchEventResponse(BaseModel):
    """Public representation of one interaction-mode transition."""

    id: int
    session_id: int
    from_mode: InteractionMode
    to_mode: InteractionMode
    created_at: datetime
    history_through_message_id: int | None = Field(
        default=None, gt=0, strict=True
    )
    history_boundary_version: Annotated[
        str, Field(strict=True, min_length=1, max_length=50)
    ] | None = None
    reviewable_user_message_count: int | None = Field(
        default=None, ge=0, strict=True
    )
    review_supported: bool = Field(strict=True)

    model_config = ConfigDict(from_attributes=True)


class SwitchInteractionModeResponse(BaseModel):
    """Response for PATCH /api/sessions/{id}/interaction-mode."""

    session: SessionResponse
    switch_event: ModeSwitchEventResponse | None = None


# ---------------------------------------------------------------------------
# History review models
# ---------------------------------------------------------------------------

HistoryReviewStatus = Literal[
    "pending", "running", "completed", "failed"
]
HistoryReviewVerdict = Literal[
    "correct", "incorrect", "uncertain", "not_a_claim"
]

SessionIdPath = Annotated[int, Path(gt=0)]
ReviewIdPath = Annotated[int, Path(gt=0)]


class HistoryReviewCreateRequest(BaseModel):
    """Request body for POST /api/sessions/{id}/history-reviews."""

    mode_switch_event_id: int = Field(strict=True, gt=0)
    acknowledge_remote_history: bool = Field(default=False, strict=True)

    model_config = ConfigDict(extra="forbid")


class HistoryReviewFindingResponse(BaseModel):
    """Public representation of one persisted finding."""

    seq: int = Field(strict=True, ge=1)
    source_message_id: int = Field(strict=True, gt=0)
    verdict: HistoryReviewVerdict
    claim_text: str
    correction_text: str | None
    explanation_text: str | None


class HistoryReviewBaseResponse(BaseModel):
    """Public review fields shared by list and detail responses."""

    id: int
    session_id: int
    mode_switch_event_id: int
    status: HistoryReviewStatus
    eligible_message_count: int
    source_message_count: int
    source_from_message_id: int | None
    source_through_message_id: int | None
    truncated: bool
    summary: str | None
    coverage_note: str | None
    error_code: str | None
    error_message: str | None
    created_at: datetime
    started_at: datetime | None
    completed_at: datetime | None
    updated_at: datetime


class HistoryReviewSummaryResponse(HistoryReviewBaseResponse):
    """List response without findings."""

    findings_count: int = Field(strict=True, ge=0)


class HistoryReviewDetailResponse(HistoryReviewSummaryResponse):
    """POST and GET detail response with full findings."""

    findings: list[HistoryReviewFindingResponse]

    @model_validator(mode="after")
    def _count_matches_findings(self) -> "HistoryReviewDetailResponse":
        if self.findings_count != len(self.findings):
            raise ValueError(
                "findings_count must equal len(findings)"
            )
        return self


class HistoryReviewErrorDetail(BaseModel):
    """Inner detail object for generic review API errors."""

    code: str
    message: str


class HistoryReviewAckRequiredDetail(HistoryReviewErrorDetail):
    """Inner detail object for a remote-history acknowledgement error."""

    source_message_count: int
    reviewer_profile_label: str
    reviewer_model: str
    truncated: bool


class HistoryReviewErrorResponse(BaseModel):
    """Real FastAPI error envelope for generic review API errors."""

    detail: HistoryReviewErrorDetail


class HistoryReviewAckRequiredResponse(BaseModel):
    """Real FastAPI error envelope for ack-required errors."""

    detail: HistoryReviewAckRequiredDetail
