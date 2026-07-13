"""Observability domain models."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ExecutionStatus(str, Enum):
    """Overall or per-stage execution status."""

    RUNNING = "running"
    SUCCESS = "success"
    WARNING = "warning"
    FAILED = "failed"
    SKIPPED = "skipped"


class PipelineEventType(str, Enum):
    """Pipeline lifecycle events."""

    PIPELINE_STARTED = "PipelineStarted"
    STAGE_STARTED = "StageStarted"
    STAGE_COMPLETED = "StageCompleted"
    STAGE_FAILED = "StageFailed"
    PIPELINE_COMPLETED = "PipelineCompleted"


class PipelineEvent(BaseModel):
    """A single observability event emitted during pipeline execution."""

    event_type: PipelineEventType
    timestamp: datetime
    stage: str | None = None
    message: str | None = None
    metadata: dict[str, Any] | None = None


class ExecutionStage(BaseModel):
    """Metrics captured for one pipeline stage."""

    stage: str
    status: ExecutionStatus
    started_at: datetime
    ended_at: datetime
    duration_ms: float
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ExecutionRecord(BaseModel):
    """Represents one complete pipeline execution."""

    id: str
    profile: str
    status: ExecutionStatus
    started_at: datetime
    ended_at: datetime
    duration_ms: float
    face_count: int = 0
    stages: list[ExecutionStage] = Field(default_factory=list)
    events: list[PipelineEvent] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    source: str = "api.recognition"
