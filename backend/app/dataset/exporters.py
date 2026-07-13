"""Dataset exporter interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from backend.app.dataset.models import DatasetJob


class DatasetExporter(ABC):
    """Abstract dataset result exporter."""

    format_name: str

    @abstractmethod
    def export(self, job: DatasetJob, *, output_path: str) -> str:
        """Export dataset job results to the target path."""


class CSVExporter(DatasetExporter):
    """Reserved CSV exporter."""

    format_name = "csv"

    def export(self, job: DatasetJob, *, output_path: str) -> str:
        raise NotImplementedError("CSVExporter is reserved for a future release.")


class JSONExporter(DatasetExporter):
    """Reserved JSON exporter."""

    format_name = "json"

    def export(self, job: DatasetJob, *, output_path: str) -> str:
        raise NotImplementedError("JSONExporter is reserved for a future release.")


class ExcelExporter(DatasetExporter):
    """Reserved Excel exporter."""

    format_name = "excel"

    def export(self, job: DatasetJob, *, output_path: str) -> str:
        raise NotImplementedError("ExcelExporter is reserved for a future release.")


class PDFExporter(DatasetExporter):
    """Reserved PDF exporter."""

    format_name = "pdf"

    def export(self, job: DatasetJob, *, output_path: str) -> str:
        raise NotImplementedError("PDFExporter is reserved for a future release.")


class ScaffoldSummaryExporter(DatasetExporter):
    """Scaffold exporter that records export metadata only."""

    format_name = "summary"

    def export(self, job: DatasetJob, *, output_path: str) -> str:
        return f"{output_path.rstrip('/')}/{job.id}.summary.json"
