"""Request and response models with input validation."""

import re
from datetime import datetime
from typing import Annotated, Literal

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from backend.llm_profiles import SessionProfileStatus
from backend.session_titles import normalize_title_whitespace

MAX_MESSAGE_LENGTH = 4000


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
