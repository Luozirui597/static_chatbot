"""Application-level exceptions."""

from typing import Literal


class LLMError(Exception):
    """Raised by LLM clients when a request cannot be fulfilled.

    The *detail* is a human-readable description safe to return to the
    frontend.  *status_code* is the suggested HTTP status.
    """

    def __init__(self, detail: str, status_code: int) -> None:
        super().__init__(detail)
        self.detail = detail
        self.status_code = status_code


class LLMInvalidResponseError(LLMError):
    """Raised when the LLM returns an empty or blank response."""


class UnknownLLMProfileError(Exception):
    """The requested llm_profile_id does not exist in the registry."""

    def __init__(self, profile_id: str) -> None:
        self.profile_id = profile_id
        super().__init__(f"Unknown LLM profile: {profile_id!r}")


class SessionProfileUnavailableError(Exception):
    """The session's profile is no longer available."""

    def __init__(self, profile_id: str) -> None:
        self.profile_id = profile_id
        super().__init__(f"LLM profile {profile_id!r} is unavailable")


class SessionProfileConflictError(Exception):
    """The session's profile model has changed since creation
    or was created before model snapshots were tracked.

    *status* distinguishes ``"model_changed"`` from ``"legacy_unknown"``
    so callers can produce accurate, safe detail messages.
    """

    def __init__(self, status: str, snapshot: str | None) -> None:
        self.status = status
        self.snapshot = snapshot
        if status == "legacy_unknown":
            detail = (
                "This session was created before model tracking was "
                "added.  You can view its history but cannot send new "
                "messages.  Create a new session to continue chatting."
            )
        else:
            detail = (
                "The model for this session's profile has changed "
                "since the session was created.  Create a new session "
                "to continue."
            )
        super().__init__(detail)


class SessionProfileSwitchAckRequiredError(Exception):
    """Switching this session to a remote API profile requires an
    explicit acknowledgement that chat history will leave the machine.

    The *code* attribute is stable and machine-readable — routes map
    it to a structured 409 body and never match on message text.
    """

    code: Literal["remote_history_ack_required"] = "remote_history_ack_required"

    def __init__(self) -> None:
        super().__init__(
            "Switching to a remote API model means that on your next "
            "message, up to the 20 most recent chat messages and the "
            "new message will be sent to a remote API service.  Set "
            "acknowledge_remote_history=true to continue."
        )


class HistoryReviewError(Exception):
    """Base class for internal history-review preparation failures."""

    code = "history_review_error"

    def __init__(self, message: str | None = None) -> None:
        super().__init__(message or "History review preparation failed.")


class HistoryReviewSessionNotFound(HistoryReviewError):
    code = "history_review_session_not_found"

    def __init__(self) -> None:
        super().__init__("Session not found.")


class HistoryReviewEventNotFound(HistoryReviewError):
    code = "history_review_event_not_found"

    def __init__(self) -> None:
        super().__init__("Mode switch event not found.")


class HistoryReviewEventNotReviewable(HistoryReviewError):
    code = "history_review_event_not_reviewable"

    def __init__(self) -> None:
        super().__init__("Mode switch event is not reviewable.")


class HistoryReviewBoundaryUnavailable(HistoryReviewError):
    code = "history_review_boundary_unavailable"

    def __init__(self) -> None:
        super().__init__("History boundary is unavailable.")


class HistoryReviewNoReviewableHistory(HistoryReviewError):
    code = "history_review_no_reviewable_history"

    def __init__(self) -> None:
        super().__init__("No reviewable history is available.")


class HistoryReviewRemoteAckRequired(HistoryReviewError):
    code = "history_review_remote_ack_required"

    def __init__(
        self,
        *,
        source_message_count: int,
        reviewer_profile_label: str,
        reviewer_model: str,
        truncated: bool,
    ) -> None:
        self.source_message_count = source_message_count
        self.reviewer_profile_label = reviewer_profile_label
        self.reviewer_model = reviewer_model
        self.truncated = truncated
        super().__init__(
            "Remote API review requires explicit acknowledgement that "
            "the selected source messages will be sent to the API model."
        )
