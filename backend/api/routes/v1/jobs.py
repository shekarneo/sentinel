"""Job execution API routes (scaffold)."""

from fastapi import APIRouter, Depends

from backend.api.dependencies import get_job_service
from backend.api.exceptions import NotFoundAPIError, ValidationAPIError
from backend.api.schemas.job_mappers import map_job, map_job_list
from backend.api.schemas.jobs import JobDeleteResponse, JobListResponse, JobResponse, JobSubmitRequest
from backend.app.execution.models import ExecutionType
from backend.app.services.job_service import JobService

router = APIRouter()


@router.post("", response_model=JobResponse)
def submit_job(
    request: JobSubmitRequest,
    job_service: JobService = Depends(get_job_service),
) -> JobResponse:
    """Queue a background job without starting worker execution."""
    try:
        task_type = ExecutionType(request.task_type.lower())
    except ValueError as exc:
        valid = ", ".join(item.value for item in ExecutionType)
        raise ValidationAPIError(
            f"Unknown task_type {request.task_type!r}. Valid types: {valid}."
        ) from exc

    if not job_service.manager.registry.has_handler(task_type):
        raise ValidationAPIError(f"No handler registered for task_type {task_type.value!r}.")

    task = job_service.submit_job(task_type, payload=request.payload)
    return map_job(task)


@router.get("", response_model=JobListResponse)
def list_jobs(
    job_service: JobService = Depends(get_job_service),
) -> JobListResponse:
    """List queued and historical jobs."""
    tasks = job_service.list_jobs()
    return map_job_list(tasks, job_service.summary())


@router.get("/{job_id}", response_model=JobResponse)
def get_job(
    job_id: str,
    job_service: JobService = Depends(get_job_service),
) -> JobResponse:
    """Return a single job by identifier."""
    task = job_service.get_job(job_id)
    if task is None:
        raise NotFoundAPIError(f"Job {job_id!r} was not found.")
    return map_job(task)


@router.delete("/{job_id}", response_model=JobDeleteResponse)
def cancel_job(
    job_id: str,
    job_service: JobService = Depends(get_job_service),
) -> JobDeleteResponse:
    """Cancel a queued job."""
    task = job_service.get_job(job_id)
    if task is None:
        raise NotFoundAPIError(f"Job {job_id!r} was not found.")

    cancelled = job_service.cancel_job(job_id)
    if cancelled is None:
        raise ValidationAPIError(f"Job {job_id!r} could not be cancelled.")

    return JobDeleteResponse(
        id=cancelled.id,
        state=cancelled.state.value,
        message="Job cancelled." if cancelled.state.value == "cancelled" else "Job is not cancellable.",
    )
