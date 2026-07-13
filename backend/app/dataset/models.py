"""Dataset processing domain models."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from backend.app.dataset.config import DatasetProcessingConfig


class DatasetType(str, Enum):
    """Supported dataset source types."""

    IMAGE_FOLDER = "image_folder"
    VIDEO_FOLDER = "video_folder"
    ZIP_ARCHIVE = "zip_archive"
    CSV_MANIFEST = "csv_manifest"
    COCO_DATASET = "coco_dataset"
    GALLERY_EXPORT = "gallery_export"
    RTSP_CAPTURE = "rtsp_capture"
    LIVE_RECORDING = "live_recording"


class DatasetOperation(str, Enum):
    """Supported dataset processing operations."""

    RECOGNITION = "recognition"
    ENROLLMENT = "enrollment"
    GALLERY_IMPORT = "gallery_import"
    GALLERY_EXPORT = "gallery_export"
    BENCHMARK = "benchmark"
    MODEL_EVALUATION = "model_evaluation"
    DATASET_VALIDATION = "dataset_validation"
    FRAUD_EVALUATION = "fraud_evaluation"
    OCR_EVALUATION = "ocr_evaluation"
    CALIBRATION = "calibration"


class DatasetJobStatus(str, Enum):
    """Lifecycle state for a dataset processing job."""

    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class DatasetItem(BaseModel):
    """One unit of work inside a dataset manifest."""

    item_id: str
    source_path: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    ground_truth: dict[str, Any] | None = None
    attributes: dict[str, Any] = Field(default_factory=dict)


class DatasetStatistics(BaseModel):
    """Aggregate statistics for a loaded dataset manifest."""

    item_count: int = 0
    labeled_items: int = 0
    attributes_present: int = 0


class DatasetManifest(BaseModel):
    """Canonical representation of a loaded dataset regardless of source."""

    dataset_id: str
    dataset_type: DatasetType
    root_path: str
    items: list[DatasetItem] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    statistics: DatasetStatistics = Field(default_factory=DatasetStatistics)


class DatasetResult(BaseModel):
    """Aggregate result produced by dataset processing and evaluators."""

    processed: int = 0
    failed: int = 0
    duration: float = 0.0
    exports: list[str] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)


class DatasetJobSummary(BaseModel):
    """Aggregate counters for a dataset job."""

    total_items: int = 0
    processed_items: int = 0
    failed_items: int = 0
    queued_items: int = 0
    execution_time_ms: float = 0.0
    export_ready: bool = False


class DatasetJob(BaseModel):
    """A dataset processing job tracked by the manager."""

    id: str
    dataset_type: DatasetType
    operation: DatasetOperation
    source_path: str
    status: DatasetJobStatus = DatasetJobStatus.QUEUED
    created_at: datetime
    updated_at: datetime
    dataset_id: str | None = None
    execution_task_ids: list[str] = Field(default_factory=list)
    export_path: str | None = None
    summary: DatasetJobSummary = Field(default_factory=DatasetJobSummary)
    result: DatasetResult = Field(default_factory=DatasetResult)
    config: DatasetProcessingConfig = Field(default_factory=DatasetProcessingConfig)
    message: str = "Dataset processing is scaffolded only."
