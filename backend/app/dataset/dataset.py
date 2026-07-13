"""Dataset manifest builders."""

from __future__ import annotations

from backend.app.dataset.models import (
    DatasetItem,
    DatasetManifest,
    DatasetStatistics,
    DatasetType,
)
from backend.app.dataset.utils import create_dataset_id, create_dataset_item_id


def build_scaffold_manifest(
    dataset_type: DatasetType,
    root_path: str,
    *,
    item_count: int = 0,
) -> DatasetManifest:
    """Create a scaffold manifest with placeholder items."""
    items = [
        DatasetItem(
            item_id=create_dataset_item_id(index),
            source_path=f"{root_path.rstrip('/')}/item-{index:06d}",
            metadata={"scaffolded": True},
            ground_truth=None,
            attributes={},
        )
        for index in range(1, item_count + 1)
    ]
    labeled_items = sum(1 for item in items if item.ground_truth)
    attributes_present = sum(1 for item in items if item.attributes)
    return DatasetManifest(
        dataset_id=create_dataset_id(),
        dataset_type=dataset_type,
        root_path=root_path,
        items=items,
        metadata={"scaffolded": True, "item_count": item_count},
        statistics=DatasetStatistics(
            item_count=len(items),
            labeled_items=labeled_items,
            attributes_present=attributes_present,
        ),
    )


def build_dataset_result_from_job(
    *,
    processed: int,
    failed: int,
    duration: float,
    exports: list[str] | None = None,
    metrics: dict | None = None,
) -> "DatasetResult":
    """Build a ``DatasetResult`` from aggregate counters."""
    from backend.app.dataset.models import DatasetResult

    return DatasetResult(
        processed=processed,
        failed=failed,
        duration=duration,
        exports=exports or [],
        metrics=metrics or {},
    )
