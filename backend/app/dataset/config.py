"""Dataset processing configuration."""

from __future__ import annotations

from pydantic import BaseModel, Field


class DatasetProcessingConfig(BaseModel):
    """Runtime configuration for dataset processing jobs."""

    enabled: bool = False
    max_items_per_job: int = Field(default=10_000, ge=1)
    default_pipeline_profile: str = "search"
    export_enabled: bool = True
    record_observability: bool = True
