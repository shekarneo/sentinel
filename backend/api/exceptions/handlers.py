"""FastAPI exception handlers."""

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from backend.api.exceptions.errors import APIError
from backend.api.schemas.common import ErrorResponse


def register_exception_handlers(app: FastAPI) -> None:
    """Register API exception handlers on the FastAPI application."""

    @app.exception_handler(APIError)
    async def handle_api_error(_: Request, exc: APIError) -> JSONResponse:
        payload = ErrorResponse(code=exc.code, message=exc.message)
        return JSONResponse(
            status_code=exc.status_code,
            content=payload.model_dump(),
        )
