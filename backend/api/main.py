"""FastAPI application factory."""

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from backend.api.exceptions import register_exception_handlers
from backend.api.middleware import register_middleware
from backend.api.routes.v1 import router as v1_router
from backend.api.version import API_V1_PREFIX, CURRENT_API_VERSION
from backend.app.config.configuration import Configuration
from frontend.web import router as console_router

FRONTEND_STATIC_DIR = Path(__file__).resolve().parents[2] / "frontend" / "static"


def create_app() -> FastAPI:
    """Create the frontend-agnostic REST API application."""
    settings = Configuration().load()
    app = FastAPI(
        title=settings.app.name,
        version=settings.app.version,
        description=(
            "Frontend-agnostic biometric platform API. "
            "All clients must integrate exclusively through this REST surface."
        ),
    )

    register_middleware(app)
    register_exception_handlers(app)

    app.include_router(v1_router, prefix=API_V1_PREFIX)
    app.include_router(console_router)
    app.mount("/static", StaticFiles(directory=str(FRONTEND_STATIC_DIR)), name="static")

    @app.get("/health", tags=["system"])
    def root_health() -> dict[str, str]:
        """Root health probe outside versioned API prefix."""
        return {"status": "ok", "api_version": CURRENT_API_VERSION}

    return app


app = create_app()
