"""Dataset manifest model tests."""

from backend.app.dataset.dataset import build_scaffold_manifest
from backend.app.dataset.models import DatasetType


def test_dataset_manifest_fields() -> None:
    manifest = build_scaffold_manifest(
        DatasetType.IMAGE_FOLDER,
        "/data/faces",
        item_count=2,
    )
    assert manifest.dataset_id
    assert manifest.root_path == "/data/faces"
    assert manifest.statistics.item_count == 2
    assert len(manifest.items) == 2
    assert manifest.items[0].item_id.startswith("item-")


def test_dataset_item_extended_fields() -> None:
    manifest = build_scaffold_manifest(DatasetType.CSV_MANIFEST, "/data/manifest.csv", item_count=1)
    item = manifest.items[0]
    assert item.metadata["scaffolded"] is True
    assert item.ground_truth is None
    assert item.attributes == {}
