"""Pipeline execution recording helpers for API routes."""

from __future__ import annotations

from datetime import datetime, timezone

from backend.app.pipeline.context import PipelineContext
from backend.app.services.observability_service import ObservabilityService


def record_pipeline_execution(
    observability: ObservabilityService,
    context: PipelineContext,
    *,
    source: str,
    started_at: datetime | None = None,
) -> None:
    """Record a completed pipeline execution without modifying pipeline logic."""
    observability.record_execution(
        context,
        source=source,
        started_at=started_at,
        ended_at=datetime.now(timezone.utc),
    )
