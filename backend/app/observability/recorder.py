"""Build execution records from pipeline context."""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

from backend.app.observability.models import (
    ExecutionRecord,
    ExecutionStage,
    ExecutionStatus,
    PipelineEvent,
    PipelineEventType,
)
from backend.app.pipeline.context import PipelineContext

STAGE_LABELS: dict[str, str] = {
    "scrfd": "Detection",
    "alignment": "Alignment",
    "assessment": "Assessment",
    "fraud": "Fraud Detection",
    "embedding": "Embedding",
    "search": "Search",
    "verification": "Verification",
}


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _ms(seconds: float) -> float:
    return round(seconds * 1000.0, 2)


def _stage_warnings(stage_name: str, context: PipelineContext) -> list[str]:
    warnings: list[str] = []
    if stage_name == "assessment":
        for index, face in enumerate(context.faces):
            assessment = face.assessment
            if assessment is not None and assessment.is_acceptable is False:
                warnings.append(f"Face {index + 1} failed quality assessment.")
    if stage_name == "verification":
        verification = context.metadata.get("verification")
        if isinstance(verification, list) and verification:
            item = verification[0]
            if hasattr(item, "decision"):
                decision = item.decision.value
            elif isinstance(item, dict):
                decision = item.get("decision")
            else:
                decision = None
            if decision in {"reject", "unknown"}:
                warnings.append(f"Verification decision: {decision}.")
    if stage_name == "search":
        search_results = context.metadata.get("search_results")
        empty = False
        if isinstance(search_results, list) and search_results:
            first = search_results[0]
            results = first.results if hasattr(first, "results") else first.get("results", [])
            empty = not results
        elif search_results is not None:
            results = (
                search_results.results
                if hasattr(search_results, "results")
                else search_results.get("results", [])
            )
            empty = not results
        if empty:
            warnings.append("Search returned no candidates.")
    return warnings


def _stage_errors(stage_name: str, context: PipelineContext) -> list[str]:
    return [
        error["message"]
        for error in context.errors
        if error.get("stage") == stage_name
    ]


def _stage_status(stage_name: str, context: PipelineContext, duration_ms: float) -> ExecutionStatus:
    errors = _stage_errors(stage_name, context)
    if errors:
        return ExecutionStatus.FAILED
    if duration_ms <= 0 and stage_name not in context.timings:
        return ExecutionStatus.SKIPPED
    warnings = _stage_warnings(stage_name, context)
    if warnings:
        return ExecutionStatus.WARNING
    return ExecutionStatus.SUCCESS


def _overall_status(context: PipelineContext, stages: list[ExecutionStage]) -> ExecutionStatus:
    if context.errors or any(stage.status == ExecutionStatus.FAILED for stage in stages):
        return ExecutionStatus.FAILED
    if any(stage.status == ExecutionStatus.WARNING for stage in stages):
        return ExecutionStatus.WARNING
    if len(context.faces) == 0 and "scrfd" in context.timings:
        return ExecutionStatus.WARNING
    return ExecutionStatus.SUCCESS


def _public_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    public: dict[str, Any] = {}
    for key, value in metadata.items():
        if key.startswith("_"):
            continue
        if isinstance(value, (str, int, float, bool)) or value is None:
            public[key] = value
        elif isinstance(value, dict):
            public[key] = {k: v for k, v in value.items() if isinstance(v, (str, int, float, bool))}
    return public


def build_execution_record(
    context: PipelineContext,
    *,
    source: str = "api.recognition",
    started_at: datetime | None = None,
    ended_at: datetime | None = None,
) -> ExecutionRecord:
    """Create an ``ExecutionRecord`` from a completed ``PipelineContext``."""
    execution_id = str(uuid.uuid4())
    start = started_at or _utc_now()
    end = ended_at or start + timedelta(seconds=sum(context.timings.values()))
    duration_ms = _ms((end - start).total_seconds())

    ordered_stage_names = list(context.timings.keys())
    stages: list[ExecutionStage] = []
    events: list[PipelineEvent] = [
        PipelineEvent(
            event_type=PipelineEventType.PIPELINE_STARTED,
            timestamp=start,
            metadata={"profile": context.profile.value, "source": source},
        )
    ]

    cursor = start
    for stage_name in ordered_stage_names:
        seconds = context.timings.get(stage_name, 0.0)
        stage_start = cursor
        stage_end = cursor + timedelta(seconds=seconds)
        duration = _ms(seconds)
        status = _stage_status(stage_name, context, duration)
        warnings = _stage_warnings(stage_name, context)
        errors = _stage_errors(stage_name, context)

        events.append(
            PipelineEvent(
                event_type=PipelineEventType.STAGE_STARTED,
                timestamp=stage_start,
                stage=stage_name,
            )
        )
        if status == ExecutionStatus.FAILED:
            events.append(
                PipelineEvent(
                    event_type=PipelineEventType.STAGE_FAILED,
                    timestamp=stage_end,
                    stage=stage_name,
                    message=errors[0] if errors else "Stage failed.",
                )
            )
        else:
            events.append(
                PipelineEvent(
                    event_type=PipelineEventType.STAGE_COMPLETED,
                    timestamp=stage_end,
                    stage=stage_name,
                    metadata={"duration_ms": duration, "status": status.value},
                )
            )

        stages.append(
            ExecutionStage(
                stage=stage_name,
                status=status,
                started_at=stage_start,
                ended_at=stage_end,
                duration_ms=duration,
                warnings=warnings,
                errors=errors,
                metadata={
                    "label": STAGE_LABELS.get(stage_name, stage_name),
                    "face_count": len(context.faces),
                },
            )
        )
        cursor = stage_end

    overall = _overall_status(context, stages)
    record_warnings = [warning for stage in stages for warning in stage.warnings]
    record_errors = [error["message"] for error in context.errors]

    events.append(
        PipelineEvent(
            event_type=PipelineEventType.PIPELINE_COMPLETED,
            timestamp=end,
            metadata={"status": overall.value, "duration_ms": duration_ms},
        )
    )

    return ExecutionRecord(
        id=execution_id,
        profile=context.profile.value,
        status=overall,
        started_at=start,
        ended_at=end,
        duration_ms=duration_ms,
        face_count=len(context.faces),
        stages=stages,
        events=events,
        warnings=record_warnings,
        errors=record_errors,
        metadata={
            "source": source,
            "stage_count": len(stages),
            "context_metadata": _public_metadata(context.metadata),
        },
        source=source,
    )
