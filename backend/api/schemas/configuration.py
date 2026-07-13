"""Configuration API schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field


class AppConfigResponse(BaseModel):
    """Application settings exposed through the API."""

    name: str
    version: str
    environment: str
    debug: bool


class PipelineProfileSummary(BaseModel):
    """Summary of a configured pipeline profile."""

    name: str
    stages: list[str]
    search_enabled: bool
    search_top_k: int | None = None


class ConfigurationResponse(BaseModel):
    """Read-only configuration summary for clients."""

    app: AppConfigResponse
    pipeline_profiles: list[PipelineProfileSummary]
