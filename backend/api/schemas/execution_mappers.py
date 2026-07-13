"""Map observability domain models to API DTOs."""

from __future__ import annotations

from backend.api.schemas.execution import (
    ExecutionListResponse,
    ExecutionRecordResponse,
    ExecutionStageResponse,
    PipelineEventResponse,
)
from backend.app.observability.models import ExecutionRecord
from backend.app.observability.recorder import STAGE_LABELS


def map_execution_record(record: ExecutionRecord) -> ExecutionRecordResponse:
    """Convert an execution record to an API response."""
    return ExecutionRecordResponse(
        id=record.id,
        profile=record.profile,
        status=record.status.value,
        started_at=record.started_at,
        ended_at=record.ended_at,
        duration_ms=record.duration_ms,
        face_count=record.face_count,
        source=record.source,
        stages=[
            ExecutionStageResponse(
                stage=stage.stage,
                label=STAGE_LABELS.get(stage.stage, stage.stage),
                status=stage.status.value,
                started_at=stage.started_at,
                ended_at=stage.ended_at,
                duration_ms=stage.duration_ms,
                warnings=stage.warnings,
                errors=stage.errors,
                metadata=stage.metadata,
            )
            for stage in record.stages
        ],
        events=[
            PipelineEventResponse(
                event_type=event.event_type.value,
                timestamp=event.timestamp,
                stage=event.stage,
                message=event.message,
                metadata=event.metadata,
            )
            for event in record.events
        ],
        warnings=record.warnings,
        errors=record.errors,
        metadata=record.metadata,
    )


def map_execution_list(records: list[ExecutionRecord]) -> ExecutionListResponse:
    """Convert execution records to a list response."""
    mapped = [map_execution_record(record) for record in records]
    return ExecutionListResponse(count=len(mapped), executions=mapped)
