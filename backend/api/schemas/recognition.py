"""Recognition API schemas."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from backend.api.schemas.common import TimingsResponse


class BoundingBoxResponse(BaseModel):
    """Face bounding box exposed through the API."""

    x: float
    y: float
    width: float
    height: float


class LandmarkResponse(BaseModel):
    """Facial landmark coordinates exposed through the API."""

    x: float
    y: float


class FaceResponse(BaseModel):
    """Public face summary returned by recognition endpoints.

    Internal ``Face`` domain objects are never returned directly.
    """

    bounding_box: BoundingBoxResponse
    confidence: float = Field(ge=0.0, le=1.0)
    landmarks: list[LandmarkResponse] = Field(min_length=5, max_length=5)
    has_alignment: bool = False
    has_assessment: bool = False
    has_embedding: bool = False
    aligned_image_base64: str | None = Field(
        default=None,
        description="JPEG-encoded aligned face crop as base64.",
    )
    assessment: dict[str, Any] | None = None
    embedding: dict[str, Any] | None = None


class RecognitionRequest(BaseModel):
    """Recognition request contract.

    Image bytes are supplied separately as multipart upload or raw body.
    """

    profile: str = Field(
        description="Pipeline profile name (e.g. enrollment, kyc, search).",
    )


class RecognitionResponse(BaseModel):
    """Recognition response contract."""

    profile: str
    face_count: int
    faces: list[FaceResponse]
    timings: TimingsResponse
    metadata: dict[str, Any] = Field(default_factory=dict)
