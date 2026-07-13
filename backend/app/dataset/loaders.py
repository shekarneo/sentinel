"""Dataset loader interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod

from backend.app.dataset.dataset import build_scaffold_manifest
from backend.app.dataset.models import DatasetManifest, DatasetType


class DatasetLoader(ABC):
    """Abstract dataset loader.

    Concrete loaders parse a source-specific path and produce a
    source-agnostic ``DatasetManifest``.
    """

    dataset_type: DatasetType

    @abstractmethod
    def load(self, source_path: str) -> DatasetManifest:
        """Load a dataset manifest from the supplied source path."""


class ImageFolderLoader(DatasetLoader):
    """Scaffold loader for image folder datasets."""

    dataset_type = DatasetType.IMAGE_FOLDER

    def load(self, source_path: str) -> DatasetManifest:
        return build_scaffold_manifest(self.dataset_type, source_path, item_count=0)


class ZipLoader(DatasetLoader):
    """Scaffold loader for zip archive datasets."""

    dataset_type = DatasetType.ZIP_ARCHIVE

    def load(self, source_path: str) -> DatasetManifest:
        return build_scaffold_manifest(self.dataset_type, source_path, item_count=0)


class CSVLoader(DatasetLoader):
    """Scaffold loader for CSV manifest datasets."""

    dataset_type = DatasetType.CSV_MANIFEST

    def load(self, source_path: str) -> DatasetManifest:
        return build_scaffold_manifest(self.dataset_type, source_path, item_count=0)


class COCOLoader(DatasetLoader):
    """Scaffold loader for COCO-format datasets."""

    dataset_type = DatasetType.COCO_DATASET

    def load(self, source_path: str) -> DatasetManifest:
        return build_scaffold_manifest(self.dataset_type, source_path, item_count=0)
