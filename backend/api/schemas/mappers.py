"""DTO mappers from domain models to API schemas."""

from __future__ import annotations

import base64
from typing import Any

import cv2
import numpy as np

from backend.api.schemas.common import TimingsResponse
from backend.api.schemas.recognition import (
    BoundingBoxResponse,
    FaceResponse,
    LandmarkResponse,
    RecognitionResponse,
)
from backend.api.schemas.search import SearchCandidateResponse, SearchResponse
from backend.api.schemas.verification import VerificationResponse
from backend.app.domain.face import Face
from backend.app.domain.search import SearchResults
from backend.app.domain.verification import VerificationResult
from backend.app.pipeline.context import PipelineContext
from backend.app.pipeline.profile import PipelineProfile


def map_face(face: Face) -> FaceResponse:
    """Convert a domain ``Face`` to a public API response."""
    return FaceResponse(
        bounding_box=BoundingBoxResponse(
            x=face.bounding_box.x,
            y=face.bounding_box.y,
            width=face.bounding_box.width,
            height=face.bounding_box.height,
        ),
        confidence=face.confidence,
        landmarks=[
            LandmarkResponse(x=landmark.x, y=landmark.y)
            for landmark in face.landmarks
        ],
        has_alignment=face.alignment is not None,
        has_assessment=face.assessment is not None,
        has_embedding=face.embedding is not None,
        aligned_image_base64=_encode_aligned_image(face),
        assessment=_map_assessment(face),
        embedding=_map_embedding(face),
    )


def _encode_aligned_image(face: Face) -> str | None:
    """Encode aligned face crop as base64 JPEG."""
    if face.alignment is None or face.alignment.aligned_image is None:
        return None

    image = face.alignment.aligned_image
    if not isinstance(image, np.ndarray) or image.size == 0:
        return None

    success, encoded = cv2.imencode(".jpg", image)
    if not success:
        return None
    return base64.b64encode(encoded.tobytes()).decode("ascii")


def _map_assessment(face: Face) -> dict[str, Any] | None:
    """Map assessment domain data to a JSON-safe summary."""
    if face.assessment is None:
        return None

    assessment = face.assessment
    return {
        "overall_score": assessment.overall_score,
        "is_acceptable": assessment.is_acceptable,
        "blur_score": assessment.blur.score if assessment.blur else None,
        "brightness_score": (
            assessment.brightness.score if assessment.brightness else None
        ),
        "pose_score": assessment.pose.score if assessment.pose else None,
        "size_score": assessment.size.score if assessment.size else None,
        "visibility_score": (
            assessment.visibility.score if assessment.visibility else None
        ),
    }


def _map_embedding(face: Face) -> dict[str, Any] | None:
    """Map embedding domain data to a JSON-safe summary."""
    if face.embedding is None:
        return None

    embedding = face.embedding
    return {
        "dimension": embedding.dimension,
        "normalized": embedding.normalized,
        "model_name": embedding.model_name,
        "inference_time_ms": embedding.inference_time_ms,
        "confidence": embedding.confidence,
    }


def map_recognition_context(context: PipelineContext) -> RecognitionResponse:
    """Convert a pipeline context to a recognition API response."""
    metadata = _public_metadata(context.metadata)
    return RecognitionResponse(
        profile=context.profile.value,
        face_count=len(context.faces),
        faces=[map_face(face) for face in context.faces],
        timings=TimingsResponse(stages=dict(context.timings)),
        metadata=metadata,
    )


def map_search_results(results: SearchResults) -> SearchResponse:
    """Convert domain search results to a search API response."""
    return SearchResponse(
        results=[
            SearchCandidateResponse(
                identity_id=result.identity_id,
                score=result.score,
                rank=result.rank,
                metadata=result.metadata,
            )
            for result in results.results
        ],
        search_time_ms=results.search_time_ms,
        provider=results.provider,
    )


def map_verification_result(result: VerificationResult) -> VerificationResponse:
    """Convert a domain verification result to an API response."""
    return VerificationResponse(
        decision=result.decision.value,
        matched_identity_id=result.matched_identity_id,
        similarity_score=result.similarity_score,
        threshold=result.threshold,
        is_verified=result.is_verified,
        verification_time_ms=result.verification_time_ms,
        metadata=result.metadata,
    )


def _public_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Strip non-serializable or internal metadata before API exposure."""
    public: dict[str, Any] = {}
    for key, value in metadata.items():
        if key == "search_results":
            if isinstance(value, SearchResults):
                public[key] = map_search_results(value).model_dump()
            elif isinstance(value, list):
                public[key] = [
                    map_search_results(item).model_dump()
                    if isinstance(item, SearchResults)
                    else item
                    for item in value
                ]
            continue
        if key == "verification" and isinstance(value, list):
            public[key] = [
                map_verification_result(item).model_dump()
                if isinstance(item, VerificationResult)
                else item
                for item in value
            ]
            continue
        if isinstance(value, (str, int, float, bool)) or value is None:
            public[key] = value
        elif isinstance(value, dict):
            public[key] = value
    return public


def parse_profile(profile_name: str) -> PipelineProfile:
    """Parse a profile string into a ``PipelineProfile`` enum."""
    try:
        return PipelineProfile(profile_name.lower())
    except ValueError as exc:
        valid = ", ".join(profile.value for profile in PipelineProfile)
        raise ValueError(
            f"Unknown profile {profile_name!r}. Valid profiles: {valid}."
        ) from exc
