"""Request and response models with input validation."""

from pydantic import BaseModel, Field, field_validator

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
