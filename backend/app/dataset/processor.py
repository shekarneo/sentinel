"""Dataset processing orchestration."""

from __future__ import annotations

from backend.app.dataset.models import DatasetItem, DatasetJob, DatasetManifest, DatasetOperation
from backend.app.dataset.registry import map_operation_to_execution_type
from backend.app.execution.models import ExecutionType
from backend.app.services.job_service import JobService


class DatasetProcessor:
    """Queues dataset manifest items through the existing execution engine.

    The processor consumes ``DatasetManifest`` only. It has no knowledge of
    folder, CSV, ZIP, or COCO source formats. Each ``DatasetItem`` becomes one
    ``ExecutionTask``. No additional queue is introduced.
    """

    def __init__(self, job_service: JobService) -> None:
        self._job_service = job_service

    def queue_manifest(
        self,
        job: DatasetJob,
        manifest: DatasetManifest,
        *,
        operation: DatasetOperation,
        pipeline_profile: str,
    ) -> list[str]:
        """Queue one execution task per manifest item."""
        execution_type_name = map_operation_to_execution_type(operation)
        if execution_type_name is None:
            raise NotImplementedError(
                f"Dataset operation {operation.value!r} is reserved for a future release."
            )

        task_type = ExecutionType(execution_type_name)
        if not self._job_service.manager.registry.has_handler(task_type):
            raise ValueError(f"No execution handler registered for {task_type.value!r}.")

        task_ids: list[str] = []
        for item in manifest.items:
            task = self._job_service.submit_job(
                task_type,
                payload=self._build_item_payload(job, manifest, item, pipeline_profile),
                source="api.datasets",
            )
            task_ids.append(task.id)
        return task_ids

    @staticmethod
    def _build_item_payload(
        job: DatasetJob,
        manifest: DatasetManifest,
        item: DatasetItem,
        pipeline_profile: str,
    ) -> dict:
        return {
            "dataset_job_id": job.id,
            "dataset_id": manifest.dataset_id,
            "dataset_item_id": item.item_id,
            "source_path": item.source_path,
            "operation": job.operation.value,
            "pipeline_profile": pipeline_profile,
            "metadata": item.metadata,
            "ground_truth": item.ground_truth,
            "attributes": item.attributes,
        }
