"""Execution engine utilities."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from backend.app.execution.models import ExecutionPriority, ExecutionTask, ExecutionType


def utc_now() -> datetime:
    """Return the current UTC timestamp."""
    return datetime.now(timezone.utc)


def new_task_id() -> str:
    """Generate a unique execution task identifier."""
    return str(uuid.uuid4())


def create_task(
    task_type: ExecutionType,
    *,
    payload: dict | None = None,
    priority: ExecutionPriority = ExecutionPriority.NORMAL,
    source: str = "api.jobs",
) -> ExecutionTask:
    """Create a new queued execution task."""
    now = utc_now()
    return ExecutionTask(
        id=new_task_id(),
        task_type=task_type,
        priority=priority,
        payload=payload or {},
        created_at=now,
        updated_at=now,
        source=source,
    )
