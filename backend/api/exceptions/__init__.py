"""API exception package."""

from backend.api.exceptions.errors import APIError, NotFoundAPIError, NotImplementedAPIError, ValidationAPIError
from backend.api.exceptions.handlers import register_exception_handlers

__all__ = [
    "APIError",
    "NotFoundAPIError",
    "NotImplementedAPIError",
    "ValidationAPIError",
    "register_exception_handlers",
]
