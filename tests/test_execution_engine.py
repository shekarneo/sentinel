"""Execution engine tests."""

import pytest

from backend.app.execution.manager import ExecutionManager
from backend.app.execution.models import ExecutionState, ExecutionType
from backend.app.execution.queue import InMemoryExecutionQueue
from backend.app.execution.worker import ExecutionWorker


def test_in_memory_queue_priority_order() -> None:
    queue = InMemoryExecutionQueue()
    manager = ExecutionManager(queue=queue)
    low = manager.submit(ExecutionType.BENCHMARK, payload={"name": "low"})
    high = manager.submit(ExecutionType.RECOGNITION, payload={"name": "high"})
    assert queue.size() == 2
    next_task = queue.next()
    assert next_task is not None
    assert next_task.id == high.id


def test_execution_manager_submit_cancel_status_list() -> None:
    manager = ExecutionManager()
    task = manager.submit(ExecutionType.ENROLLMENT, payload={"identity_id": "user-1"})
    assert manager.status(task.id) is not None
    assert manager.status(task.id).state == ExecutionState.QUEUED
    cancelled = manager.cancel(task.id)
    assert cancelled is not None
    assert cancelled.state == ExecutionState.CANCELLED
    assert len(manager.list()) == 1


def test_worker_execution_disabled() -> None:
    manager = ExecutionManager()
    worker = manager.worker
    with pytest.raises(RuntimeError, match="scaffolded only"):
        worker.poll_once()
