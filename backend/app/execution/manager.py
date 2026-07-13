"""Execution manager."""

from __future__ import annotations

from backend.app.execution.config import ExecutionEngineConfig
from backend.app.execution.models import ExecutionState, ExecutionTask, ExecutionType
from backend.app.execution.queue import ExecutionQueue, InMemoryExecutionQueue
from backend.app.execution.registry import ExecutionRegistry
from backend.app.execution.utils import create_task, utc_now
from backend.app.execution.worker import ExecutionWorker


class ExecutionManager:
    """Owns the execution queue, worker, and handler registry."""

    def __init__(
        self,
        *,
        queue: ExecutionQueue | None = None,
        registry: ExecutionRegistry | None = None,
        worker: ExecutionWorker | None = None,
        config: ExecutionEngineConfig | None = None,
    ) -> None:
        self._config = config or ExecutionEngineConfig()
        self._queue = queue or InMemoryExecutionQueue()
        self._registry = registry or ExecutionRegistry()
        self._tasks: dict[str, ExecutionTask] = {}
        self._worker = worker or ExecutionWorker(
            self._queue,
            self._registry,
            config=self._config,
            task_store=self._tasks,
        )

    @property
    def queue(self) -> ExecutionQueue:
        """Return the execution queue."""
        return self._queue

    @property
    def worker(self) -> ExecutionWorker:
        """Return the execution worker."""
        return self._worker

    @property
    def registry(self) -> ExecutionRegistry:
        """Return the execution registry."""
        return self._registry

    @property
    def config(self) -> ExecutionEngineConfig:
        """Return execution engine configuration."""
        return self._config

    def submit(
        self,
        task_type: ExecutionType,
        *,
        payload: dict | None = None,
        source: str = "api.jobs",
    ) -> ExecutionTask:
        """Submit a task to the execution queue."""
        if self._queue.size() >= self._config.max_queue_size:
            raise ValueError("Execution queue is full.")

        task = create_task(task_type, payload=payload, source=source)
        self._tasks[task.id] = task
        self._queue.submit(task)
        return task

    def cancel(self, task_id: str) -> ExecutionTask | None:
        """Cancel a queued task."""
        task = self._tasks.get(task_id)
        if task is None:
            return None

        if task.state == ExecutionState.QUEUED:
            self._queue.cancel(task_id)
            cancelled = task.model_copy(
                update={
                    "state": ExecutionState.CANCELLED,
                    "updated_at": utc_now(),
                    "completed_at": utc_now(),
                }
            )
            self._tasks[task_id] = cancelled
            return cancelled

        if task.state == ExecutionState.RUNNING:
            raise ValueError("Running tasks cannot be cancelled in the scaffolded engine.")

        return task

    def status(self, task_id: str) -> ExecutionTask | None:
        """Return the current status of a task."""
        return self._tasks.get(task_id)

    def list(self) -> list[ExecutionTask]:
        """Return all known tasks in reverse chronological order."""
        return sorted(
            self._tasks.values(),
            key=lambda task: task.created_at,
            reverse=True,
        )

    def list_by_state(self, state: ExecutionState) -> list[ExecutionTask]:
        """Return tasks filtered by lifecycle state."""
        return [task for task in self.list() if task.state == state]

    def clear(self) -> None:
        """Clear queued tasks and in-memory task history."""
        self._queue.clear()
        self._tasks.clear()
