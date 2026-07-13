"""API exception types."""

from __future__ import annotations


class APIError(Exception):
    """Base API exception with HTTP status metadata."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int = 500,
        code: str = "internal_error",
    ) -> None:
        """Initialize an API error.

        Args:
            message: Human-readable error message.
            status_code: HTTP status code to return.
            code: Stable machine-readable error code.
        """
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.code = code


class NotImplementedAPIError(APIError):
    """Raised when an endpoint contract exists but behavior is not implemented."""

    def __init__(self, message: str = "Endpoint not implemented.") -> None:
        super().__init__(message, status_code=501, code="not_implemented")


class ValidationAPIError(APIError):
    """Raised when request validation fails at the API boundary."""

    def __init__(self, message: str) -> None:
        super().__init__(message, status_code=400, code="validation_error")


class NotFoundAPIError(APIError):
    """Raised when a requested resource does not exist."""

    def __init__(self, message: str) -> None:
        super().__init__(message, status_code=404, code="not_found")
