"""Execution engine application service."""

from __future__ import annotations

from backend.app.execution.config import ExecutionEngineConfig
from backend.app.execution.handlers import register_default_handlers
from backend.app.execution.manager import ExecutionManager
from backend.app.execution.models import ExecutionState, ExecutionTask, ExecutionType
from backend.app.execution.registry import ExecutionRegistry
from backend.app.services.enrollment_service import EnrollmentService
from backend.app.services.gallery_service import GalleryService
from backend.app.services.recognition_service import RecognitionService


class JobService:
    """Application service facade for the execution engine."""

    def __init__(
        self,
        manager: ExecutionManager | None = None,
        *,
        recognition: RecognitionService | None = None,
        enrollment: EnrollmentService | None = None,
        gallery: GalleryService | None = None,
    ) -> None:
        registry = ExecutionRegistry()
        register_default_handlers(
            registry,
            recognition=recognition,
            enrollment=enrollment,
            gallery=gallery,
        )
        self._manager = manager or ExecutionManager(
            registry=registry,
            config=ExecutionEngineConfig(),
        )

    @property
    def manager(self) -> ExecutionManager:
        """Return the underlying execution manager."""
        return self._manager

    def submit_job(
        self,
        task_type: ExecutionType,
        *,
        payload: dict | None = None,
        source: str = "api.jobs",
    ) -> ExecutionTask:
        """Queue a job without starting background execution."""
        return self._manager.submit(task_type, payload=payload, source=source)

    def get_job(self, job_id: str) -> ExecutionTask | None:
        """Return a job by identifier."""
        return self._manager.status(job_id)

    def list_jobs(self) -> list[ExecutionTask]:
        """Return all known jobs."""
        return self._manager.list()

    def cancel_job(self, job_id: str) -> ExecutionTask | None:
        """Cancel a queued job."""
        return self._manager.cancel(job_id)

    def summary(self) -> dict[str, int]:
        """Return job counts grouped by lifecycle state."""
        return {
            state.value: len(self._manager.list_by_state(state))
            for state in ExecutionState
        }
