"""Dataset processing manager."""

from __future__ import annotations

from backend.app.dataset.config import DatasetProcessingConfig
from backend.app.dataset.dataset import build_dataset_result_from_job, build_scaffold_manifest
from backend.app.dataset.exporters import ScaffoldSummaryExporter
from backend.app.dataset.loaders import COCOLoader, CSVLoader, ImageFolderLoader, ZipLoader
from backend.app.dataset.models import (
    DatasetJob,
    DatasetJobStatus,
    DatasetJobSummary,
    DatasetManifest,
    DatasetOperation,
    DatasetResult,
    DatasetType,
)
from backend.app.dataset.processor import DatasetProcessor
from backend.app.dataset.registry import (
    DatasetEvaluatorRegistry,
    DatasetExporterRegistry,
    DatasetLoaderRegistry,
)
from backend.app.dataset.utils import create_dataset_job_id, utc_now
from backend.app.services.job_service import JobService
from backend.app.services.observability_service import ObservabilityService


class DatasetManager:
    """Owns dataset loading, operation selection, and execution integration."""

    def __init__(
        self,
        job_service: JobService,
        observability_service: ObservabilityService | None = None,
        *,
        config: DatasetProcessingConfig | None = None,
        loader_registry: DatasetLoaderRegistry | None = None,
        exporter_registry: DatasetExporterRegistry | None = None,
        evaluator_registry: DatasetEvaluatorRegistry | None = None,
        processor: DatasetProcessor | None = None,
    ) -> None:
        self._config = config or DatasetProcessingConfig()
        self._job_service = job_service
        self._observability = observability_service
        self._loader_registry = loader_registry or _default_loader_registry()
        self._exporter_registry = exporter_registry or _default_exporter_registry()
        self._evaluator_registry = evaluator_registry or DatasetEvaluatorRegistry()
        self._processor = processor or DatasetProcessor(job_service)
        self._manifests: dict[str, DatasetManifest] = {}
        self._jobs: dict[str, DatasetJob] = {}

    @property
    def loader_registry(self) -> DatasetLoaderRegistry:
        return self._loader_registry

    @property
    def exporter_registry(self) -> DatasetExporterRegistry:
        return self._exporter_registry

    @property
    def evaluator_registry(self) -> DatasetEvaluatorRegistry:
        return self._evaluator_registry

    def load(self, dataset_type: DatasetType, source_path: str) -> DatasetManifest:
        """Load a dataset manifest using the registered loader."""
        loader = self._loader_registry.get(dataset_type)
        if loader is None:
            if dataset_type in {DatasetType.RTSP_CAPTURE, DatasetType.LIVE_RECORDING}:
                raise NotImplementedError(
                    f"Dataset type {dataset_type.value!r} is reserved for a future release."
                )
            manifest = build_scaffold_manifest(dataset_type, source_path, item_count=0)
        else:
            manifest = loader.load(source_path)

        self._manifests[manifest.dataset_id] = manifest
        return manifest

    def process(
        self,
        dataset_type: DatasetType,
        operation: DatasetOperation,
        source_path: str,
        *,
        item_count: int = 0,
        pipeline_profile: str | None = None,
    ) -> DatasetJob:
        """Load a manifest and queue one execution per item."""
        if operation in {
            DatasetOperation.FRAUD_EVALUATION,
            DatasetOperation.OCR_EVALUATION,
            DatasetOperation.CALIBRATION,
        }:
            raise NotImplementedError(
                f"Dataset operation {operation.value!r} is reserved for a future release."
            )

        manifest = self.load(dataset_type, source_path)
        if item_count > 0 and not manifest.items:
            manifest = build_scaffold_manifest(dataset_type, source_path, item_count=item_count)
            self._manifests[manifest.dataset_id] = manifest

        now = utc_now()
        job = DatasetJob(
            id=create_dataset_job_id(),
            dataset_type=dataset_type,
            operation=operation,
            source_path=source_path,
            status=DatasetJobStatus.QUEUED,
            created_at=now,
            updated_at=now,
            dataset_id=manifest.dataset_id,
            config=self._config,
        )
        self._jobs[job.id] = job

        profile = pipeline_profile or self._config.default_pipeline_profile
        task_ids = self._processor.queue_manifest(
            job,
            manifest,
            operation=operation,
            pipeline_profile=profile,
        )

        summary = DatasetJobSummary(
            total_items=len(manifest.items),
            queued_items=len(task_ids),
            processed_items=0,
            failed_items=0,
            execution_time_ms=0.0,
            export_ready=False,
        )
        updated = job.model_copy(
            update={
                "status": DatasetJobStatus.RUNNING if task_ids else DatasetJobStatus.COMPLETED,
                "updated_at": utc_now(),
                "execution_task_ids": task_ids,
                "summary": summary,
                "result": build_dataset_result_from_job(
                    processed=0,
                    failed=0,
                    duration=0.0,
                    metrics={"manifest_items": manifest.statistics.item_count},
                ),
            }
        )
        self._jobs[job.id] = updated
        return updated

    def export(self, job_id: str, *, output_path: str, format_name: str = "summary") -> DatasetJob:
        """Generate a scaffold export for a dataset job."""
        job = self._jobs.get(job_id)
        if job is None:
            raise ValueError(f"Dataset job {job_id!r} was not found.")

        exporter = self._exporter_registry.get(format_name)
        if exporter is None:
            raise ValueError(f"No exporter registered for format {format_name!r}.")

        export_path = exporter.export(job, output_path=output_path)
        exports = list(job.result.exports)
        if export_path not in exports:
            exports.append(export_path)
        updated = job.model_copy(
            update={
                "export_path": export_path,
                "updated_at": utc_now(),
                "summary": job.summary.model_copy(update={"export_ready": True}),
                "result": job.result.model_copy(update={"exports": exports}),
            }
        )
        self._jobs[job.id] = updated
        return updated

    def summary(self, job_id: str | None = None) -> DatasetJobSummary | dict[str, DatasetJobSummary]:
        """Return job summary or summaries for all jobs."""
        if job_id is not None:
            job = self._jobs.get(job_id)
            if job is None:
                raise ValueError(f"Dataset job {job_id!r} was not found.")
            return self._refresh_summary(job)

        return {job.id: self._refresh_summary(job) for job in self._jobs.values()}

    def get_job(self, job_id: str) -> DatasetJob | None:
        """Return a dataset job by identifier."""
        job = self._jobs.get(job_id)
        if job is None:
            return None
        return self._refresh_job(job)

    def list_jobs(self) -> list[DatasetJob]:
        """Return all dataset jobs."""
        return [self._refresh_job(job) for job in sorted(
            self._jobs.values(),
            key=lambda item: item.created_at,
            reverse=True,
        )]

    def get_manifest(self, dataset_id: str) -> DatasetManifest | None:
        """Return a loaded dataset manifest."""
        return self._manifests.get(dataset_id)

    def get_results(self, job_id: str) -> dict:
        """Return scaffold results for a dataset job."""
        job = self.get_job(job_id)
        if job is None:
            raise ValueError(f"Dataset job {job_id!r} was not found.")

        execution_tasks = [
            self._job_service.get_job(task_id)
            for task_id in job.execution_task_ids
        ]
        manifest = self._manifests.get(job.dataset_id or "")
        result = self._build_result(job)
        return {
            "job_id": job.id,
            "status": job.status.value,
            "summary": job.summary.model_dump(),
            "result": result.model_dump(),
            "export_path": job.export_path,
            "manifest": manifest.model_dump() if manifest else None,
            "execution_tasks": [
                {
                    "id": task.id,
                    "state": task.state.value,
                    "task_type": task.task_type.value,
                    "payload": task.payload,
                }
                for task in execution_tasks
                if task is not None
            ],
            "observability_ready": self._observability is not None,
            "message": "Dataset results are scaffolded only.",
        }

    def clear(self) -> None:
        """Clear in-memory manifests and jobs. Intended for tests."""
        self._manifests.clear()
        self._jobs.clear()

    def _refresh_job(self, job: DatasetJob) -> DatasetJob:
        summary = self._refresh_summary(job)
        status = self._derive_status(job, summary)
        result = self._build_result(job.model_copy(update={"summary": summary}))
        return job.model_copy(
            update={
                "summary": summary,
                "status": status,
                "result": result,
                "updated_at": utc_now(),
            }
        )

    def _build_result(self, job: DatasetJob) -> DatasetResult:
        exports = list(job.result.exports)
        if job.export_path and job.export_path not in exports:
            exports.append(job.export_path)
        return build_dataset_result_from_job(
            processed=job.summary.processed_items,
            failed=job.summary.failed_items,
            duration=job.summary.execution_time_ms,
            exports=exports,
            metrics=job.result.metrics,
        )

    def _refresh_summary(self, job: DatasetJob) -> DatasetJobSummary:
        tasks = [
            self._job_service.get_job(task_id)
            for task_id in job.execution_task_ids
        ]
        valid_tasks = [task for task in tasks if task is not None]
        processed = sum(1 for task in valid_tasks if task.state.value == "completed")
        failed = sum(1 for task in valid_tasks if task.state.value == "failed")
        queued = sum(1 for task in valid_tasks if task.state.value == "queued")
        return DatasetJobSummary(
            total_items=job.summary.total_items,
            processed_items=processed,
            failed_items=failed,
            queued_items=queued,
            execution_time_ms=job.summary.execution_time_ms,
            export_ready=job.summary.export_ready,
        )

    @staticmethod
    def _derive_status(job: DatasetJob, summary: DatasetJobSummary) -> DatasetJobStatus:
        if summary.total_items == 0:
            return DatasetJobStatus.COMPLETED
        if summary.failed_items > 0 and summary.processed_items + summary.failed_items >= summary.total_items:
            return DatasetJobStatus.FAILED
        if summary.processed_items >= summary.total_items and summary.total_items > 0:
            return DatasetJobStatus.COMPLETED
        if summary.queued_items > 0 or summary.processed_items > 0:
            return DatasetJobStatus.RUNNING
        return job.status


def _default_loader_registry() -> DatasetLoaderRegistry:
    registry = DatasetLoaderRegistry()
    registry.register(DatasetType.IMAGE_FOLDER, ImageFolderLoader())
    registry.register(DatasetType.ZIP_ARCHIVE, ZipLoader())
    registry.register(DatasetType.CSV_MANIFEST, CSVLoader())
    registry.register(DatasetType.COCO_DATASET, COCOLoader())
    return registry


def _default_exporter_registry() -> DatasetExporterRegistry:
    registry = DatasetExporterRegistry()
    registry.register(ScaffoldSummaryExporter())
    return registry
