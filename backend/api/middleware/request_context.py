"""API middleware registration."""

from fastapi import FastAPI


def register_middleware(app: FastAPI) -> None:
    """Register API middleware on the FastAPI application.

    Args:
        app: FastAPI application instance.
    """
    # Reserved for request ID, auth, CORS, and observability middleware.
