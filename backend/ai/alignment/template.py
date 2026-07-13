"""
Reference landmark template for face alignment.

Provides canonical landmark positions that detected landmarks are aligned to
when warping a face to standard orientation and scale.
"""

import numpy as np

from backend.ai.alignment.constants import CANONICAL_FACE_SIZE

# Standard ArcFace 112x112 reference template.
# Landmark order matches the Face domain model:
# 0 left eye, 1 right eye, 2 nose, 3 left mouth, 4 right mouth.
REFERENCE_LANDMARKS: tuple[tuple[float, float], ...] = (
    (38.2946, 51.6963),
    (73.5318, 51.5014),
    (56.0252, 71.7366),
    (41.5493, 92.3655),
    (70.7299, 92.2041),
)


def get_reference_landmarks(
    face_size: int | None = None,
) -> np.ndarray:
    """Return the canonical landmark template for alignment.

    The template defines target (x, y) positions for each of the five
    facial landmarks in the aligned face coordinate system. Coordinates
    follow the standard ArcFace 112x112 reference layout and are scaled
    proportionally when a different ``face_size`` is requested.

    Args:
        face_size: Canonical face size. Defaults to ``CANONICAL_FACE_SIZE``.

    Returns:
        Reference landmark array with shape ``(5, 2)`` and dtype ``float32``.
        A copy is returned to prevent accidental mutation of the template.
    """
    size = face_size if face_size is not None else CANONICAL_FACE_SIZE

    template = np.array(REFERENCE_LANDMARKS, dtype=np.float32)

    if size != CANONICAL_FACE_SIZE:
        template = template * (size / CANONICAL_FACE_SIZE)

    return template.copy()
