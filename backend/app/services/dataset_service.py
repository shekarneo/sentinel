"""Dataset processing application service."""

from __future__ import annotations

from backend.app.dataset.config import DatasetProcessingConfig
from backend.app.dataset.manager import DatasetManager
from backend.app.dataset.models import DatasetJob, DatasetJobSummary, DatasetManifest, DatasetOperation, DatasetType
from backend.app.services.job_service import JobService
from backend.app.services.observability_service import ObservabilityService


class DatasetService:
    """Application facade for dataset processing."""

    def __init__(
        self,
        manager: DatasetManager | None = None,
        *,
        jobs: JobService | None = None,
        observability: ObservabilityService | None = None,
    ) -> None:
        if manager is None and jobs is None:
            raise ValueError("Either manager or job service must be provided.")
        self._manager = manager or DatasetManager(
            jobs,  # type: ignore[arg-type]
            observability,
            config=DatasetProcessingConfig(),
        )

    @property
    def manager(self) -> DatasetManager:
        return self._manager

    def load_dataset(self, dataset_type: DatasetType, source_path: str) -> DatasetManifest:
        return self._manager.load(dataset_type, source_path)

    def process_dataset(
        self,
        dataset_type: DatasetType,
        operation: DatasetOperation,
        source_path: str,
        *,
        item_count: int = 0,
        pipeline_profile: str | None = None,
    ) -> DatasetJob:
        return self._manager.process(
            dataset_type,
            operation,
            source_path,
            item_count=item_count,
            pipeline_profile=pipeline_profile,
        )

    def export_job(self, job_id: str, *, output_path: str, format_name: str = "summary") -> DatasetJob:
        return self._manager.export(job_id, output_path=output_path, format_name=format_name)

    def get_job(self, job_id: str) -> DatasetJob | None:
        return self._manager.get_job(job_id)

    def list_jobs(self) -> list[DatasetJob]:
        return self._manager.list_jobs()

    def get_results(self, job_id: str) -> dict:
        return self._manager.get_results(job_id)

    def summary(self, job_id: str | None = None) -> DatasetJobSummary | dict[str, DatasetJobSummary]:
        return self._manager.summary(job_id)
