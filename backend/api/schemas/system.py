"""Extended system status API schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field

from backend.api.schemas.configuration import PipelineProfileSummary


class ModelInfoResponse(BaseModel):
    """Loaded model summary."""

    name: str
    path: str
    status: str = "configured"


class SystemHealthResponse(BaseModel):
    """Lightweight health probe response."""

    status: str = "ok"


class SystemStatusResponse(BaseModel):
    """Platform system status response."""

    status: str = "ok"
    api_version: str
    app_name: str
    app_version: str
    environment: str
    gallery_size: int = 0


class DashboardResponse(BaseModel):
    """Aggregated dashboard metrics for the web console."""

    status: str = "ok"
    api_version: str
    app_name: str
    app_version: str
    environment: str
    gallery_size: int = 0
    search_index_status: str
    search_index_vector_count: int = 0
    models: list[ModelInfoResponse]
    pipeline_profiles: list[PipelineProfileSummary]
    recognition_summary: dict[str, str] = Field(default_factory=dict)
