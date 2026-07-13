"""Execution handler registry."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from backend.app.execution.models import ExecutionType

ExecutionHandler = Callable[[dict[str, Any]], dict[str, Any]]


class ExecutionRegistry:
    """Registers execution handlers by ``ExecutionType``.

    Handlers must delegate to application services only. They must never
    import or invoke AI modules directly.
    """

    def __init__(self) -> None:
        self._handlers: dict[ExecutionType, ExecutionHandler] = {}

    def register(self, task_type: ExecutionType, handler: ExecutionHandler) -> None:
        """Register a handler for a task type."""
        self._handlers[task_type] = handler

    def get(self, task_type: ExecutionType) -> ExecutionHandler | None:
        """Return the handler for a task type."""
        return self._handlers.get(task_type)

    def list_types(self) -> list[ExecutionType]:
        """Return registered execution types."""
        return list(self._handlers.keys())

    def has_handler(self, task_type: ExecutionType) -> bool:
        """Return whether a handler is registered."""
        return task_type in self._handlers
