"""Execution worker."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from backend.app.execution.config import ExecutionEngineConfig
from backend.app.execution.models import ExecutionState, ExecutionTask
from backend.app.execution.queue import ExecutionQueue
from backend.app.execution.registry import ExecutionRegistry
from backend.app.execution.utils import utc_now

if TYPE_CHECKING:
    from backend.app.services.observability_service import ObservabilityService

logger = logging.getLogger(__name__)


class ExecutionWorker:
    """Polls the queue and dispatches tasks to registered handlers.

    The worker is scaffolded only. Background polling is disabled until a
    future release enables asynchronous execution.
    """

    def __init__(
        self,
        queue: ExecutionQueue,
        registry: ExecutionRegistry,
        *,
        config: ExecutionEngineConfig | None = None,
        observability: ObservabilityService | None = None,
        task_store: dict[str, object] | None = None,
    ) -> None:
        self._queue = queue
        self._registry = registry
        self._config = config or ExecutionEngineConfig()
        self._observability = observability
        self._task_store = task_store

    @property
    def enabled(self) -> bool:
        """Return whether worker execution is enabled."""
        return self._config.worker_enabled

    def poll_once(self) -> bool:
        """Poll the queue and execute one task if workers are enabled.

        Returns:
            ``True`` if a task was processed, otherwise ``False``.

        Raises:
            RuntimeError: If worker execution is disabled.
        """
        if not self._config.worker_enabled:
            raise RuntimeError(
                "ExecutionWorker is scaffolded only. Background execution is disabled."
            )

        task = self._queue.next()
        if task is None:
            return False

        self.execute_task(task)
        return True

    def execute_task(self, task: ExecutionTask) -> None:
        """Execute a single task through the registered application handler.

        Args:
            task: ``ExecutionTask`` instance to execute.
        """
        if not self._config.worker_enabled:
            raise RuntimeError(
                "ExecutionWorker is scaffolded only. Background execution is disabled."
            )

        handler = self._registry.get(task.task_type)
        if handler is None:
            self._mark_failed(task, f"No handler registered for {task.task_type.value}.")
            return

        started_at = utc_now()
        task = task.model_copy(
            update={
                "state": ExecutionState.RUNNING,
                "started_at": started_at,
                "updated_at": started_at,
            }
        )
        self._persist(task)

        try:
            result = handler(task.payload)
            completed_at = utc_now()
            task = task.model_copy(
                update={
                    "state": ExecutionState.COMPLETED,
                    "result": result,
                    "completed_at": completed_at,
                    "updated_at": completed_at,
                }
            )
            self._persist(task)
            self._notify_observability(task)
        except Exception as exc:
            logger.exception("Execution task %s failed.", task.id)
            self._mark_failed(task, str(exc))

    def _mark_failed(self, task: ExecutionTask, message: str) -> None:
        completed_at = utc_now()
        task = task.model_copy(
            update={
                "state": ExecutionState.FAILED,
                "error": message,
                "completed_at": completed_at,
                "updated_at": completed_at,
            }
        )
        self._persist(task)
        self._notify_observability(task)

    def _persist(self, task: ExecutionTask) -> None:
        if self._task_store is not None:
            self._task_store[task.id] = task

    def _notify_observability(self, task: ExecutionTask) -> None:
        if self._observability is None:
            return
        logger.debug(
            "Observability hook for task %s (%s) is reserved for future integration.",
            task.id,
            task.state.value,
        )
