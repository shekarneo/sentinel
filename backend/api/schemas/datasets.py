"""Dataset processing API schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class DatasetProcessRequest(BaseModel):
    """Request to process a dataset."""

    dataset_type: str
    operation: str
    source_path: str
    item_count: int = Field(default=0, ge=0)
    pipeline_profile: str | None = None
    export_path: str | None = None


class DatasetResultResponse(BaseModel):
    """Aggregate dataset processing result."""

    processed: int = 0
    failed: int = 0
    duration: float = 0.0
    exports: list[str] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)


class DatasetJobSummaryResponse(BaseModel):
    """Dataset job summary counters."""

    total_items: int = 0
    processed_items: int = 0
    failed_items: int = 0
    queued_items: int = 0
    execution_time_ms: float = 0.0
    export_ready: bool = False


class DatasetJobResponse(BaseModel):
    """Dataset job response."""

    id: str
    dataset_type: str
    operation: str
    source_path: str
    status: str
    created_at: datetime
    updated_at: datetime
    dataset_id: str | None = None
    execution_task_ids: list[str] = Field(default_factory=list)
    export_path: str | None = None
    summary: DatasetJobSummaryResponse
    message: str = "Dataset processing is scaffolded only."


class DatasetJobListResponse(BaseModel):
    """List of dataset jobs."""

    count: int
    jobs: list[DatasetJobResponse]


class DatasetResultsResponse(BaseModel):
    """Dataset job results response."""

    job_id: str
    status: str
    summary: DatasetJobSummaryResponse
    result: DatasetResultResponse
    export_path: str | None = None
    execution_tasks: list[dict[str, Any]] = Field(default_factory=list)
    observability_ready: bool = False
    message: str = "Dataset results are scaffolded only."
