"""Dataset processing API routes (scaffold)."""

from fastapi import APIRouter, Depends

from backend.api.dependencies import get_dataset_service
from backend.api.exceptions import NotFoundAPIError, ValidationAPIError
from backend.api.schemas.dataset_mappers import (
    map_dataset_job,
    map_dataset_job_list,
    map_dataset_results,
)
from backend.api.schemas.datasets import DatasetJobListResponse, DatasetJobResponse, DatasetProcessRequest, DatasetResultsResponse
from backend.app.dataset.models import DatasetOperation, DatasetType
from backend.app.services.dataset_service import DatasetService

router = APIRouter()


@router.post("/process", response_model=DatasetJobResponse)
def process_dataset(
    request: DatasetProcessRequest,
    dataset_service: DatasetService = Depends(get_dataset_service),
) -> DatasetJobResponse:
    """Queue a dataset processing job without starting worker execution."""
    try:
        dataset_type = DatasetType(request.dataset_type.lower())
        operation = DatasetOperation(request.operation.lower())
    except ValueError as exc:
        raise ValidationAPIError(str(exc)) from exc

    try:
        job = dataset_service.process_dataset(
            dataset_type,
            operation,
            request.source_path,
            item_count=request.item_count,
            pipeline_profile=request.pipeline_profile,
        )
    except NotImplementedError as exc:
        raise ValidationAPIError(str(exc)) from exc
    except ValueError as exc:
        raise ValidationAPIError(str(exc)) from exc

    if request.export_path:
        job = dataset_service.export_job(job.id, output_path=request.export_path)

    return map_dataset_job(job)


@router.get("/jobs", response_model=DatasetJobListResponse)
def list_dataset_jobs(
    dataset_service: DatasetService = Depends(get_dataset_service),
) -> DatasetJobListResponse:
    """List dataset processing jobs."""
    return map_dataset_job_list(dataset_service.list_jobs())


@router.get("/jobs/{job_id}", response_model=DatasetJobResponse)
def get_dataset_job(
    job_id: str,
    dataset_service: DatasetService = Depends(get_dataset_service),
) -> DatasetJobResponse:
    """Return a dataset processing job."""
    job = dataset_service.get_job(job_id)
    if job is None:
        raise NotFoundAPIError(f"Dataset job {job_id!r} was not found.")
    return map_dataset_job(job)


@router.get("/results/{job_id}", response_model=DatasetResultsResponse)
def get_dataset_results(
    job_id: str,
    dataset_service: DatasetService = Depends(get_dataset_service),
) -> DatasetResultsResponse:
    """Return scaffold results for a dataset job."""
    try:
        payload = dataset_service.get_results(job_id)
    except ValueError as exc:
        raise NotFoundAPIError(str(exc)) from exc
    return map_dataset_results(payload)
