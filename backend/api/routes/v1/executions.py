"""Execution observability API routes."""

from fastapi import APIRouter, Depends

from backend.api.dependencies import get_observability_service
from backend.api.exceptions import NotFoundAPIError, ValidationAPIError
from backend.api.schemas.execution import ExecutionListResponse, ExecutionRecordResponse
from backend.api.schemas.execution_mappers import map_execution_list, map_execution_record
from backend.app.services.observability_service import ObservabilityService

router = APIRouter()


@router.get("/latest", response_model=ExecutionListResponse)
def latest_executions(
    limit: int = 10,
    observability: ObservabilityService = Depends(get_observability_service),
) -> ExecutionListResponse:
    """Return the most recent pipeline executions."""
    if limit < 1 or limit > 100:
        raise ValidationAPIError("limit must be between 1 and 100.")
    records = observability.latest_executions(limit=limit)
    return map_execution_list(records)


@router.get("", response_model=ExecutionListResponse)
def list_executions(
    observability: ObservabilityService = Depends(get_observability_service),
) -> ExecutionListResponse:
    """Return all stored pipeline executions."""
    return map_execution_list(observability.list_executions())


@router.get("/{execution_id}", response_model=ExecutionRecordResponse)
def get_execution(
    execution_id: str,
    observability: ObservabilityService = Depends(get_observability_service),
) -> ExecutionRecordResponse:
    """Return a single pipeline execution by identifier."""
    record = observability.get_execution(execution_id)
    if record is None:
        raise NotFoundAPIError(f"Execution {execution_id!r} was not found.")
    return map_execution_record(record)
