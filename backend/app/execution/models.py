"""Execution engine domain models."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ExecutionType(str, Enum):
    """Supported asynchronous workload types."""

    RECOGNITION = "recognition"
    ENROLLMENT = "enrollment"
    GALLERY_REBUILD = "gallery_rebuild"
    BENCHMARK = "benchmark"
    VIDEO_PROCESSING = "video_processing"
    DATASET_EVALUATION = "dataset_evaluation"
    FRAUD_DETECTION = "fraud_detection"


class ExecutionPriority(str, Enum):
    """Task scheduling priority."""

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class ExecutionState(str, Enum):
    """Lifecycle state for an execution task."""

    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


_PRIORITY_ORDER: dict[ExecutionPriority, int] = {
    ExecutionPriority.LOW: 0,
    ExecutionPriority.NORMAL: 1,
    ExecutionPriority.HIGH: 2,
    ExecutionPriority.CRITICAL: 3,
}


class ExecutionTask(BaseModel):
    """A unit of asynchronous work managed by the execution engine."""

    id: str
    task_type: ExecutionType
    priority: ExecutionPriority = ExecutionPriority.NORMAL
    state: ExecutionState = ExecutionState.QUEUED
    payload: dict[str, Any] = Field(default_factory=dict)
    result: dict[str, Any] | None = None
    error: str | None = None
    created_at: datetime
    updated_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    source: str = "api.jobs"

    @property
    def priority_rank(self) -> int:
        """Return numeric priority for queue ordering."""
        return _PRIORITY_ORDER[self.priority]
