"""
Utility helpers for the embedding engine.
"""

import numpy as np

from backend.ai.embedding.config import load_embedding_model_settings
from backend.app.domain.face import Face


def validate_aligned_face(
    aligned_face: np.ndarray,
    *,
    expected_size: int | None = None,
) -> None:
    """Validate an aligned face image for embedding extraction.

    Args:
        aligned_face: Aligned face crop in BGR uint8 format.
        expected_size: Optional square resolution requirement.

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
            f"Aligned face must have shape (H, W, 3), got {aligned_face.shape}."
        )

    if expected_size is not None:
        height, width = aligned_face.shape[:2]
        if height != expected_size or width != expected_size:
            raise ValueError(
                f"Aligned face must be {expected_size}x{expected_size}x3, "
                f"got {height}x{width}x3."
            )


def validate_embedding_input(faces: list[Face]) -> None:
    """Validate inputs to the embedding pipeline.

    Args:
        faces: Faces to embed.

    Raises:
        ValueError: If the face list or alignment data is invalid.
    """
    if not isinstance(faces, list):
        raise ValueError("Faces must be provided as a list.")

    expected_size = load_embedding_model_settings().input_size

    for index, face in enumerate(faces):
        if not isinstance(face, Face):
            raise ValueError(f"Expected Face at index {index}, got {type(face)!r}.")

        if face.alignment is None:
            raise ValueError(
                f"Face at index {index} must have alignment data before embedding."
            )

        if face.alignment.aligned_image is None:
            raise ValueError(
                f"Face at index {index} must have an aligned image before embedding."
            )

        validate_aligned_face(
            face.alignment.aligned_image,
            expected_size=expected_size,
        )
