"""Dataset processing helpers."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone


def utc_now() -> datetime:
    """Return the current UTC timestamp."""
    return datetime.now(timezone.utc)


def create_dataset_id() -> str:
    """Generate a dataset identifier."""
    return str(uuid.uuid4())


def create_dataset_job_id() -> str:
    """Generate a dataset job identifier."""
    return str(uuid.uuid4())


def create_dataset_item_id(index: int) -> str:
    """Generate a dataset item identifier."""
    return f"item-{index:06d}"
