"""Dataset processing API mappers."""

from __future__ import annotations

from backend.api.schemas.datasets import (
    DatasetJobListResponse,
    DatasetJobResponse,
    DatasetJobSummaryResponse,
    DatasetResultResponse,
    DatasetResultsResponse,
)
from backend.app.dataset.models import DatasetJob, DatasetJobSummary, DatasetResult


def map_dataset_job_summary(summary: DatasetJobSummary) -> DatasetJobSummaryResponse:
    return DatasetJobSummaryResponse(
        total_items=summary.total_items,
        processed_items=summary.processed_items,
        failed_items=summary.failed_items,
        queued_items=summary.queued_items,
        execution_time_ms=summary.execution_time_ms,
        export_ready=summary.export_ready,
    )


def map_dataset_job(job: DatasetJob) -> DatasetJobResponse:
    return DatasetJobResponse(
        id=job.id,
        dataset_type=job.dataset_type.value,
        operation=job.operation.value,
        source_path=job.source_path,
        status=job.status.value,
        created_at=job.created_at,
        updated_at=job.updated_at,
        dataset_id=job.dataset_id,
        execution_task_ids=job.execution_task_ids,
        export_path=job.export_path,
        summary=map_dataset_job_summary(job.summary),
        message=job.message,
    )


def map_dataset_job_list(jobs: list[DatasetJob]) -> DatasetJobListResponse:
    mapped = [map_dataset_job(job) for job in jobs]
    return DatasetJobListResponse(count=len(mapped), jobs=mapped)


def map_dataset_result(result: DatasetResult) -> DatasetResultResponse:
    return DatasetResultResponse(
        processed=result.processed,
        failed=result.failed,
        duration=result.duration,
        exports=result.exports,
        metrics=result.metrics,
    )


def map_dataset_results(payload: dict) -> DatasetResultsResponse:
    summary = payload.get("summary", {})
    result = payload.get("result", {})
    return DatasetResultsResponse(
        job_id=payload["job_id"],
        status=payload["status"],
        summary=DatasetJobSummaryResponse(**summary),
        result=DatasetResultResponse(**result),
        export_path=payload.get("export_path"),
        execution_tasks=payload.get("execution_tasks", []),
        observability_ready=payload.get("observability_ready", False),
        message=payload.get("message", "Dataset results are scaffolded only."),
    )
