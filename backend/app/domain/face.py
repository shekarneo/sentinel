"""
Face domain model.

This module defines the core Face object that flows through
the entire biometric pipeline.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from backend.app.domain.alignment import AlignmentData
from backend.app.domain.assessment import AssessmentData
from backend.app.domain.embedding import EmbeddingData
from backend.app.domain.fraud import FraudData


class BoundingBox(BaseModel):
    """Represents the location of a detected face.

    Coordinates use the top-left representation:

    - ``x``: left edge
    - ``y``: top edge
    - ``width``: box width
    - ``height``: box height
    """

    x: float
    y: float
    width: float
    height: float


class Landmark(BaseModel):
    """Represents a facial landmark in image coordinates."""

    x: float
    y: float


class Face(BaseModel):
    """Canonical pipeline object for the biometric engine.

    A single ``Face`` instance flows through every stage of the biometric
    pipeline. Detection fields are populated by SCRFD and must remain
    immutable. Each downstream module enriches the same object by attaching
    its own nested stage model.

    Landmark order is fixed:

    0. Left eye
    1. Right eye
    2. Nose
    3. Left mouth corner
    4. Right mouth corner

    Stage ownership:

    - ``alignment`` — Face Alignment
    - ``assessment`` — Face Assessment
    - ``fraud`` — Biometric Fraud Detection
    - ``embedding`` — Embedding Service

    Platform-level results such as search, verification, identity, tracking,
    and gallery data are stored in separate domain models, not on ``Face``.
    """

    bounding_box: BoundingBox
    confidence: float = Field(ge=0.0, le=1.0)
    landmarks: list[Landmark] = Field(min_length=5, max_length=5)

    alignment: AlignmentData | None = None
    assessment: AssessmentData | None = None
    fraud: FraudData | None = None
    embedding: EmbeddingData | None = None
