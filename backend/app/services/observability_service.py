"""Observability application service."""

from __future__ import annotations

from datetime import datetime, timezone

from backend.app.observability.models import ExecutionRecord
from backend.app.observability.recorder import build_execution_record
from backend.app.observability.store import ExecutionStore, InMemoryExecutionStore
from backend.app.pipeline.context import PipelineContext


class ObservabilityService:
    """Records and queries pipeline execution observability data."""

    def __init__(self, store: ExecutionStore | None = None) -> None:
        self._store = store or InMemoryExecutionStore()

    @property
    def store(self) -> ExecutionStore:
        """Return the configured execution store."""
        return self._store

    def record_execution(
        self,
        context: PipelineContext,
        *,
        source: str = "api.recognition",
        started_at: datetime | None = None,
        ended_at: datetime | None = None,
    ) -> ExecutionRecord:
        """Build and persist an execution record from a pipeline context."""
        record = build_execution_record(
            context,
            source=source,
            started_at=started_at,
            ended_at=ended_at or datetime.now(timezone.utc),
        )
        self._store.add(record)
        return record

    def get_execution(self, execution_id: str) -> ExecutionRecord | None:
        """Return a stored execution by identifier."""
        return self._store.get(execution_id)

    def list_executions(self) -> list[ExecutionRecord]:
        """Return all stored executions."""
        return self._store.list_all()

    def latest_executions(self, limit: int = 10) -> list[ExecutionRecord]:
        """Return the most recent executions."""
        return self._store.latest(limit=limit)

    def clear_executions(self) -> None:
        """Clear all stored executions."""
        self._store.clear()
