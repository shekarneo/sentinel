"""System status API routes."""

from fastapi import APIRouter, Depends

from backend.api.dependencies import (
    get_configuration_service,
    get_gallery_service,
)
from backend.api.schemas.configuration import PipelineProfileSummary
from backend.api.schemas.system import (
    DashboardResponse,
    ModelInfoResponse,
    SystemHealthResponse,
    SystemStatusResponse,
)
from backend.api.version import CURRENT_API_VERSION
from backend.app.services.configuration_service import ConfigurationService
from backend.app.services.gallery_service import GalleryService

router = APIRouter()


@router.get("/health", response_model=SystemHealthResponse)
def health_check() -> SystemHealthResponse:
    """Lightweight health probe."""
    return SystemHealthResponse()


@router.get("/status", response_model=SystemStatusResponse)
def system_status(
    configuration_service: ConfigurationService = Depends(get_configuration_service),
    gallery_service: GalleryService = Depends(get_gallery_service),
) -> SystemStatusResponse:
    """Return platform status and gallery summary."""
    settings = configuration_service.get_settings()
    return SystemStatusResponse(
        api_version=CURRENT_API_VERSION,
        app_name=settings.app.name,
        app_version=settings.app.version,
        environment=settings.app.environment,
        gallery_size=gallery_service.gallery_size,
    )


@router.get("/dashboard", response_model=DashboardResponse)
def dashboard(
    configuration_service: ConfigurationService = Depends(get_configuration_service),
    gallery_service: GalleryService = Depends(get_gallery_service),
) -> DashboardResponse:
    """Return aggregated dashboard metrics for the web console."""
    settings = configuration_service.get_settings()
    profiles = configuration_service.list_pipeline_profiles()
    index_stats = gallery_service.get_index_stats()
    models = [
        ModelInfoResponse(
            name=model["name"],
            path=model["path"],
            status=model["status"],
        )
        for model in configuration_service.list_models()
    ]
    return DashboardResponse(
        api_version=CURRENT_API_VERSION,
        app_name=settings.app.name,
        app_version=settings.app.version,
        environment=settings.app.environment,
        gallery_size=gallery_service.gallery_size,
        search_index_status=str(index_stats["status"]),
        search_index_vector_count=int(index_stats["vector_count"]),
        models=models,
        pipeline_profiles=[
            PipelineProfileSummary(
                name=name,
                stages=definition.stages,
                search_enabled=definition.search.enabled,
                search_top_k=definition.search.top_k,
            )
            for name, definition in profiles.items()
        ],
        recognition_summary={
            "pipeline_engine": "active",
            "profiles_configured": str(len(profiles)),
            "default_profile": "search",
        },
    )
