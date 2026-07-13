"""Execution persistence interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod

from backend.app.observability.models import ExecutionRecord


class ExecutionStore(ABC):
    """Abstract execution store.

    ``InMemoryExecutionStore`` is the initial implementation. A future database
    backend can implement the same interface without redesigning callers.
    """

    @abstractmethod
    def add(self, record: ExecutionRecord) -> None:
        """Persist a completed execution record."""

    @abstractmethod
    def get(self, execution_id: str) -> ExecutionRecord | None:
        """Return a single execution by identifier."""

    @abstractmethod
    def latest(self, limit: int = 10) -> list[ExecutionRecord]:
        """Return the most recent execution records."""

    @abstractmethod
    def list_all(self) -> list[ExecutionRecord]:
        """Return all stored execution records in reverse chronological order."""

    @abstractmethod
    def clear(self) -> None:
        """Remove all stored execution records."""


class InMemoryExecutionStore(ExecutionStore):
    """Process-local in-memory execution store."""

    def __init__(self) -> None:
        self._records: dict[str, ExecutionRecord] = {}
        self._order: list[str] = []

    def add(self, record: ExecutionRecord) -> None:
        self._records[record.id] = record
        if record.id in self._order:
            self._order.remove(record.id)
        self._order.append(record.id)

    def get(self, execution_id: str) -> ExecutionRecord | None:
        return self._records.get(execution_id)

    def latest(self, limit: int = 10) -> list[ExecutionRecord]:
        if limit < 1:
            return []
        ids = list(reversed(self._order))[:limit]
        return [self._records[execution_id] for execution_id in ids if execution_id in self._records]

    def list_all(self) -> list[ExecutionRecord]:
        return self.latest(limit=len(self._order))

    def clear(self) -> None:
        self._records.clear()
        self._order.clear()
