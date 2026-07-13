"""Observability layer tests."""

from datetime import datetime, timezone

import numpy as np

from backend.app.observability.models import ExecutionStatus, PipelineEventType
from backend.app.observability.recorder import build_execution_record
from backend.app.observability.store import InMemoryExecutionStore
from backend.app.pipeline.context import PipelineContext
from backend.app.pipeline.profile import PipelineProfile
from backend.app.services.observability_service import ObservabilityService


def _sample_context() -> PipelineContext:
    return PipelineContext(
        image=np.zeros((64, 64, 3), dtype=np.uint8),
        profile=PipelineProfile.SEARCH,
        timings={"scrfd": 0.024, "alignment": 0.002, "embedding": 0.014},
        errors=[{"stage": "search", "message": "gallery is empty"}],
    )


def test_in_memory_execution_store() -> None:
    store = InMemoryExecutionStore()
    record = build_execution_record(_sample_context(), source="test")
    store.add(record)
    assert store.get(record.id) is not None
    assert len(store.latest(limit=5)) == 1
    assert len(store.list_all()) == 1
    store.clear()
    assert store.get(record.id) is None


def test_build_execution_record_timeline_and_events() -> None:
    started = datetime(2026, 1, 1, tzinfo=timezone.utc)
    ended = datetime(2026, 1, 1, 0, 0, 1, tzinfo=timezone.utc)
    record = build_execution_record(
        _sample_context(),
        source="test",
        started_at=started,
        ended_at=ended,
    )
    assert record.duration_ms > 0
    assert record.stages[0].started_at == started
    assert record.events[0].event_type is PipelineEventType.PIPELINE_STARTED
    assert record.events[-1].event_type is PipelineEventType.PIPELINE_COMPLETED
    assert record.status in {ExecutionStatus.FAILED, ExecutionStatus.WARNING}


def test_observability_service_records_execution() -> None:
    service = ObservabilityService(store=InMemoryExecutionStore())
    record = service.record_execution(_sample_context(), source="test")
    assert service.get_execution(record.id) is not None
    assert len(service.latest_executions()) == 1
