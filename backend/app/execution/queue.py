"""Execution queue interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod

from backend.app.execution.models import ExecutionTask


class ExecutionQueue(ABC):
    """Abstract execution queue.

    ``InMemoryExecutionQueue`` is the initial implementation. A future
    persistent or distributed queue can implement the same contract.
    """

    @abstractmethod
    def submit(self, task: ExecutionTask) -> None:
        """Enqueue a task for worker processing."""

    @abstractmethod
    def next(self) -> ExecutionTask | None:
        """Return the next task eligible for execution."""

    @abstractmethod
    def cancel(self, task_id: str) -> ExecutionTask | None:
        """Cancel a queued task by identifier."""

    @abstractmethod
    def clear(self) -> None:
        """Remove all queued tasks."""

    @abstractmethod
    def size(self) -> int:
        """Return the number of queued tasks."""


class InMemoryExecutionQueue(ExecutionQueue):
    """Priority-ordered in-memory execution queue."""

    def __init__(self) -> None:
        self._tasks: list[ExecutionTask] = []

    def submit(self, task: ExecutionTask) -> None:
        self._tasks.append(task)
        self._tasks.sort(
            key=lambda item: (item.priority_rank, item.created_at),
            reverse=True,
        )

    def next(self) -> ExecutionTask | None:
        if not self._tasks:
            return None
        return self._tasks.pop(0)

    def cancel(self, task_id: str) -> ExecutionTask | None:
        for index, task in enumerate(self._tasks):
            if task.id == task_id:
                return self._tasks.pop(index)
        return None

    def clear(self) -> None:
        self._tasks.clear()

    def size(self) -> int:
        return len(self._tasks)

    def list_queued(self) -> list[ExecutionTask]:
        """Return queued tasks without dequeuing."""
        return list(self._tasks)
