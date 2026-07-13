"""Pytest configuration and shared fixtures."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.domain.embedding import EmbeddingData
from backend.app.domain.face import BoundingBox, Face, Landmark


@pytest.fixture
def unit_face() -> Face:
    """Return a probe face with a normalized unit embedding."""
    vector = np.ones(512, dtype=np.float32)
    vector /= np.linalg.norm(vector)

    return Face(
        bounding_box=BoundingBox(x=0.0, y=0.0, width=100.0, height=100.0),
        confidence=0.99,
        landmarks=[Landmark(x=10.0, y=10.0) for _ in range(5)],
        embedding=EmbeddingData(
            vector=vector,
            dimension=512,
            normalized=True,
            model_name="test",
        ),
    )
