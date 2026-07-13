"""Job execution API schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class JobSubmitRequest(BaseModel):
    """Request to queue a background job."""

    task_type: str = Field(description="Execution type identifier.")
    payload: dict[str, Any] = Field(default_factory=dict)


class JobResponse(BaseModel):
    """Single job response."""

    id: str
    task_type: str
    priority: str
    state: str
    payload: dict[str, Any] = Field(default_factory=dict)
    result: dict[str, Any] | None = None
    error: str | None = None
    source: str
    created_at: datetime
    updated_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    message: str = "Job queued. Background execution is scaffolded only."


class JobListResponse(BaseModel):
    """List of jobs."""

    count: int
    summary: dict[str, int]
    jobs: list[JobResponse]


class JobDeleteResponse(BaseModel):
    """Job cancellation response."""

    id: str
    state: str
    message: str
