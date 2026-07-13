"""
Utility helpers for the face assessment module.
"""

import numpy as np

from backend.app.domain.assessment import (
    AssessmentData,
    BlurResult,
    BrightnessResult,
    PoseResult,
    SizeResult,
    VisibilityResult,
)
from backend.app.domain.face import Face


def validate_aligned_face(aligned_face: np.ndarray) -> None:
    """Validate an aligned face image for image-based analyzers.

    Args:
        aligned_face: Aligned face crop in BGR uint8 format.

    Raises:
        ValueError: If the image is invalid.
    """
    if not isinstance(aligned_face, np.ndarray):
        raise ValueError(
            f"Aligned face must be a numpy array, got {type(aligned_face)!r}."
        )

    if aligned_face.size == 0:
        raise ValueError("Aligned face image must be non-empty.")

    if aligned_face.dtype != np.uint8:
        raise ValueError(
            f"Aligned face must have dtype uint8, got {aligned_face.dtype}."
        )

    if aligned_face.ndim != 3 or aligned_face.shape[2] != 3:
        raise ValueError(
            "Aligned face must have 3 channels, "
            f"got shape {aligned_face.shape}."
        )


def validate_assessment_input(faces: list[Face]) -> None:
    """Validate inputs to the assessment pipeline.

    Args:
        faces: Faces to assess.

    Raises:
        ValueError: If the face list or alignment data is invalid.
    """
    if not isinstance(faces, list):
        raise ValueError("Faces must be provided as a list.")

    for index, face in enumerate(faces):
        if not isinstance(face, Face):
            raise ValueError(f"Expected Face at index {index}, got {type(face)!r}.")

        if face.alignment is None:
            raise ValueError(
                f"Face at index {index} must have alignment data before assessment."
            )

        if face.alignment.aligned_image is None:
            raise ValueError(
                f"Face at index {index} must have an aligned image before assessment."
            )


def build_assessment_data(
    *,
    blur: BlurResult | None = None,
    brightness: BrightnessResult | None = None,
    pose: PoseResult | None = None,
    size: SizeResult | None = None,
    visibility: VisibilityResult | None = None,
    overall_score: float | None = None,
    is_acceptable: bool | None = None,
) -> AssessmentData:
    """Combine analyzer results into a single assessment model.

    Args:
        blur: Blur analyzer output.
        brightness: Brightness analyzer output.
        pose: Pose analyzer output.
        size: Size analyzer output.
        visibility: Visibility analyzer output.
        overall_score: Combined quality score from scoring.
        is_acceptable: Whether the face passes quality thresholds.

    Returns:
        ``AssessmentData`` with nested stage results attached.
    """
    return AssessmentData(
        blur=blur,
        brightness=brightness,
        pose=pose,
        size=size,
        visibility=visibility,
        overall_score=overall_score,
        is_acceptable=is_acceptable,
    )


def clip_score(score: float) -> float:
    """Clamp a normalized score to [0, 1].

    Args:
        score: Raw score value.

    Returns:
        Score clamped to [0, 1].
    """
    return float(np.clip(score, 0.0, 1.0))
