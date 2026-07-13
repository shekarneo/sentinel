"""Configuration API routes."""

from fastapi import APIRouter, Depends

from backend.api.dependencies import get_configuration_service
from backend.api.schemas.config_files import ConfigurationFilesResponse
from backend.api.schemas.configuration import (
    AppConfigResponse,
    ConfigurationResponse,
    PipelineProfileSummary,
)
from backend.app.services.configuration_service import ConfigurationService

router = APIRouter()


@router.get("", response_model=ConfigurationResponse)
def get_configuration(
    configuration_service: ConfigurationService = Depends(get_configuration_service),
) -> ConfigurationResponse:
    """Return read-only application and pipeline configuration."""
    settings = configuration_service.get_settings()
    profiles = configuration_service.list_pipeline_profiles()
    return ConfigurationResponse(
        app=AppConfigResponse(
            name=settings.app.name,
            version=settings.app.version,
            environment=settings.app.environment,
            debug=settings.app.debug,
        ),
        pipeline_profiles=[
            PipelineProfileSummary(
                name=name,
                stages=definition.stages,
                search_enabled=definition.search.enabled,
                search_top_k=definition.search.top_k,
            )
            for name, definition in profiles.items()
        ],
    )


@router.get("/files", response_model=ConfigurationFilesResponse)
def get_configuration_files(
    configuration_service: ConfigurationService = Depends(get_configuration_service),
) -> ConfigurationFilesResponse:
    """Return raw YAML configuration files for read-only console display."""
    files = configuration_service.read_config_files()
    return ConfigurationFilesResponse(**files)
