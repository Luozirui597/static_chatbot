"""Application-level exceptions."""


class LLMError(Exception):
    """Raised by LLM clients when a request cannot be fulfilled.

    The *detail* is a human-readable description safe to return to the
    frontend.  *status_code* is the suggested HTTP status.
    """

    def __init__(self, detail: str, status_code: int) -> None:
        super().__init__(detail)
        self.detail = detail
        self.status_code = status_code
