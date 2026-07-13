"""
Utility helpers for the search engine.
"""

import numpy as np

from backend.app.domain.face import Face

_NORM_TOLERANCE = 1e-5


def validate_probe_vector(
    vector: np.ndarray,
    *,
    expected_dimension: int | None = None,
) -> None:
    """Validate a probe embedding vector for similarity search.

    Args:
        vector: Probe embedding vector.
        expected_dimension: Optional required vector length.

    Raises:
        ValueError: If the vector is invalid.
    """
    if not isinstance(vector, np.ndarray):
        raise ValueError(
            f"Probe vector must be a numpy array, got {type(vector)!r}."
        )

    if vector.size == 0:
        raise ValueError("Probe vector must be non-empty.")

    if vector.dtype not in (np.float32, np.float64):
        raise ValueError(
            f"Probe vector must be float32 or float64, got {vector.dtype}."
        )

    flat_vector = vector.reshape(-1)
    if expected_dimension is not None and flat_vector.size != expected_dimension:
        raise ValueError(
            "Embedding dimension does not match index: "
            f"expected {expected_dimension}, got {flat_vector.size}."
        )

    norm = float(np.linalg.norm(flat_vector))
    if abs(norm - 1.0) > _NORM_TOLERANCE:
        raise ValueError(
            "Probe vector must be unit-normalized for cosine search: "
            f"expected norm 1.0 ± {_NORM_TOLERANCE}, got {norm:.8f}."
        )


def validate_gallery_vector(
    vector: np.ndarray,
    *,
    expected_dimension: int | None = None,
) -> None:
    """Validate a gallery embedding vector before indexing.

    Args:
        vector: Gallery embedding vector.
        expected_dimension: Optional required vector length.

    Raises:
        ValueError: If the vector is invalid.
    """
    validate_probe_vector(vector, expected_dimension=expected_dimension)


def validate_search_input(faces: list[Face]) -> None:
    """Validate inputs to the search pipeline.

    Args:
        faces: Faces to search.

    Raises:
        ValueError: If the face list or embedding data is invalid.
    """
    if not isinstance(faces, list):
        raise ValueError("Faces must be provided as a list.")

    for index, face in enumerate(faces):
        if not isinstance(face, Face):
            raise ValueError(f"Expected Face at index {index}, got {type(face)!r}.")

        if face.embedding is None:
            raise ValueError(
                f"Face at index {index} must have embedding data before search."
            )

        if face.embedding.vector is None:
            raise ValueError(
                f"Face at index {index} must have an embedding vector before search."
            )

        if not face.embedding.normalized:
            raise ValueError(
                f"Face at index {index} must have a normalized embedding before search."
            )

        validate_probe_vector(face.embedding.vector)
