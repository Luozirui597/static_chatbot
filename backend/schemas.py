"""Request and response models with input validation."""

import re
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

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

    model_config = ConfigDict(from_attributes=True)


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


class CreateSessionRequest(BaseModel):
    """Request body for POST /api/sessions.

    *llm_profile_id* is validated strictly — no type coercion, no
    trailing-newline bypass.
    """

    llm_profile_id: str = Field(
        default="default",
        strict=True,
        min_length=1,
        max_length=50,
        pattern=_LLM_PROFILE_ID_PATTERN,
        description="LLM profile id for the new session",
    )

    @field_validator("llm_profile_id")
    @classmethod
    def _fullmatch_id(cls, v: str) -> str:
        if not re.fullmatch(_LLM_PROFILE_ID_PATTERN, v):
            raise ValueError(
                f"llm_profile_id must match {_LLM_PROFILE_ID_PATTERN}"
            )
        return v


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
