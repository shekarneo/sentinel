"""Dataset processing registries."""

from __future__ import annotations

from backend.app.dataset.evaluators import DatasetEvaluator
from backend.app.dataset.exporters import DatasetExporter
from backend.app.dataset.loaders import DatasetLoader
from backend.app.dataset.models import DatasetOperation, DatasetType


class DatasetLoaderRegistry:
    """Registers dataset loaders by ``DatasetType``."""

    def __init__(self) -> None:
        self._loaders: dict[DatasetType, DatasetLoader] = {}

    def register(self, dataset_type: DatasetType, loader: DatasetLoader) -> None:
        self._loaders[dataset_type] = loader

    def get(self, dataset_type: DatasetType) -> DatasetLoader | None:
        return self._loaders.get(dataset_type)

    def has_loader(self, dataset_type: DatasetType) -> bool:
        return dataset_type in self._loaders


class DatasetExporterRegistry:
    """Registers dataset exporters by format name."""

    def __init__(self) -> None:
        self._exporters: dict[str, DatasetExporter] = {}

    def register(self, exporter: DatasetExporter) -> None:
        self._exporters[exporter.format_name] = exporter

    def get(self, format_name: str) -> DatasetExporter | None:
        return self._exporters.get(format_name)

    def has_exporter(self, format_name: str) -> bool:
        return format_name in self._exporters


class DatasetEvaluatorRegistry:
    """Registers dataset evaluators by name."""

    def __init__(self) -> None:
        self._evaluators: dict[str, DatasetEvaluator] = {}

    def register(self, evaluator: DatasetEvaluator) -> None:
        self._evaluators[evaluator.name] = evaluator

    def get(self, name: str) -> DatasetEvaluator | None:
        return self._evaluators.get(name)


_OPERATION_EXECUTION_MAP: dict[DatasetOperation, str] = {
    DatasetOperation.RECOGNITION: "recognition",
    DatasetOperation.ENROLLMENT: "enrollment",
    DatasetOperation.GALLERY_IMPORT: "enrollment",
    DatasetOperation.GALLERY_EXPORT: "gallery_rebuild",
    DatasetOperation.BENCHMARK: "benchmark",
    DatasetOperation.MODEL_EVALUATION: "dataset_evaluation",
    DatasetOperation.DATASET_VALIDATION: "dataset_evaluation",
}


def map_operation_to_execution_type(operation: DatasetOperation) -> str | None:
    """Map a dataset operation to an execution engine task type."""
    if operation in {
        DatasetOperation.FRAUD_EVALUATION,
        DatasetOperation.OCR_EVALUATION,
        DatasetOperation.CALIBRATION,
    }:
        return None
    return _OPERATION_EXECUTION_MAP.get(operation)
