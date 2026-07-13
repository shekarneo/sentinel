"""Dataset evaluator interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod

from backend.app.dataset.models import DatasetJob, DatasetManifest, DatasetResult


class DatasetEvaluator(ABC):
    """Abstract dataset evaluator.

    Future evaluators populate ``DatasetResult`` with operation-specific
    metrics for recognition, verification, fraud, and calibration workloads.
    """

    name: str

    @abstractmethod
    def evaluate(self, job: DatasetJob, manifest: DatasetManifest | None = None) -> DatasetResult:
        """Evaluate a completed dataset job."""


class RecognitionEvaluator(DatasetEvaluator):
    """Reserved recognition evaluator."""

    name = "recognition"

    def evaluate(self, job: DatasetJob, manifest: DatasetManifest | None = None) -> DatasetResult:
        raise NotImplementedError("RecognitionEvaluator is reserved for a future release.")


class VerificationEvaluator(DatasetEvaluator):
    """Reserved verification evaluator."""

    name = "verification"

    def evaluate(self, job: DatasetJob, manifest: DatasetManifest | None = None) -> DatasetResult:
        raise NotImplementedError("VerificationEvaluator is reserved for a future release.")


class BenchmarkEvaluator(DatasetEvaluator):
    """Reserved benchmark evaluator."""

    name = "benchmark"

    def evaluate(self, job: DatasetJob, manifest: DatasetManifest | None = None) -> DatasetResult:
        raise NotImplementedError("BenchmarkEvaluator is reserved for a future release.")


class FraudEvaluator(DatasetEvaluator):
    """Reserved fraud evaluator."""

    name = "fraud"

    def evaluate(self, job: DatasetJob, manifest: DatasetManifest | None = None) -> DatasetResult:
        raise NotImplementedError("FraudEvaluator is reserved for a future release.")


class CalibrationEvaluator(DatasetEvaluator):
    """Reserved calibration evaluator."""

    name = "calibration"

    def evaluate(self, job: DatasetJob, manifest: DatasetManifest | None = None) -> DatasetResult:
        raise NotImplementedError("CalibrationEvaluator is reserved for a future release.")
