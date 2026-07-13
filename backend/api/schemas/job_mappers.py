"""Job API mappers."""

from __future__ import annotations

from backend.api.schemas.jobs import JobListResponse, JobResponse
from backend.app.execution.models import ExecutionTask


def map_job(task: ExecutionTask) -> JobResponse:
    """Map an execution task to an API response."""
    return JobResponse(
        id=task.id,
        task_type=task.task_type.value,
        priority=task.priority.value,
        state=task.state.value,
        payload=task.payload,
        result=task.result,
        error=task.error,
        source=task.source,
        created_at=task.created_at,
        updated_at=task.updated_at,
        started_at=task.started_at,
        completed_at=task.completed_at,
    )


def map_job_list(tasks: list[ExecutionTask], summary: dict[str, int]) -> JobListResponse:
    """Map execution tasks to a list response."""
    jobs = [map_job(task) for task in tasks]
    return JobListResponse(count=len(jobs), summary=summary, jobs=jobs)
