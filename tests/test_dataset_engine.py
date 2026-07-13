"""Dataset processing module tests."""

import pytest

from backend.app.dataset.manager import DatasetManager
from backend.app.dataset.models import DatasetOperation, DatasetType
from backend.app.execution.config import ExecutionEngineConfig
from backend.app.execution.handlers import register_default_handlers
from backend.app.execution.manager import ExecutionManager
from backend.app.execution.registry import ExecutionRegistry
from backend.app.services.job_service import JobService


def _job_service() -> JobService:
    registry = ExecutionRegistry()
    register_default_handlers(registry)
    manager = ExecutionManager(registry=registry, config=ExecutionEngineConfig())
    return JobService(manager=manager)


def test_dataset_manager_process_queues_execution_tasks() -> None:
    jobs = _job_service()
    manager = DatasetManager(jobs)
    job = manager.process(
        DatasetType.IMAGE_FOLDER,
        DatasetOperation.RECOGNITION,
        "/data/faces",
        item_count=2,
    )
    assert job.summary.total_items == 2
    assert len(job.execution_task_ids) == 2
    assert jobs.get_job(job.execution_task_ids[0]) is not None


def test_dataset_manager_export_summary() -> None:
    jobs = _job_service()
    manager = DatasetManager(jobs)
    job = manager.process(
        DatasetType.ZIP_ARCHIVE,
        DatasetOperation.BENCHMARK,
        "/data/archive.zip",
        item_count=1,
    )
    exported = manager.export(job.id, output_path="/tmp/exports")
    assert exported.export_path is not None
    assert exported.summary.export_ready is True


def test_reserved_operation_raises() -> None:
    manager = DatasetManager(_job_service())
    with pytest.raises(NotImplementedError):
        manager.process(
            DatasetType.IMAGE_FOLDER,
            DatasetOperation.FRAUD_EVALUATION,
            "/data/faces",
        )
