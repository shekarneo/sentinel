"""Execution observability API schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class PipelineEventResponse(BaseModel):
    """Pipeline event exposed through the API."""

    event_type: str
    timestamp: datetime
    stage: str | None = None
    message: str | None = None
    metadata: dict[str, Any] | None = None


class ExecutionStageResponse(BaseModel):
    """Per-stage execution metrics."""

    stage: str
    label: str
    status: str
    started_at: datetime
    ended_at: datetime
    duration_ms: float
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ExecutionRecordResponse(BaseModel):
    """Complete pipeline execution record."""

    id: str
    profile: str
    status: str
    started_at: datetime
    ended_at: datetime
    duration_ms: float
    face_count: int
    source: str
    stages: list[ExecutionStageResponse]
    events: list[PipelineEventResponse]
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ExecutionListResponse(BaseModel):
    """List of execution records."""

    count: int
    executions: list[ExecutionRecordResponse]
