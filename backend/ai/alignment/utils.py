"""
Utility helpers for the face alignment module.
"""

import numpy as np

from backend.app.domain.face import Face, Landmark


def crop_face_region(image: np.ndarray, face: Face) -> np.ndarray:
    """Crop the bounding-box region for a detected face.

    Args:
        image: Source image in HWC BGR layout.
        face: Detected face with a bounding box.

    Returns:
        Cropped face region from the source image.
    """
    bbox = face.bounding_box
    image_height, image_width = image.shape[:2]

    x1 = max(0, int(round(bbox.x)))
    y1 = max(0, int(round(bbox.y)))
    x2 = min(image_width, int(round(bbox.x + bbox.width)))
    y2 = min(image_height, int(round(bbox.y + bbox.height)))

    return image[y1:y2, x1:x2].copy()


def landmarks_to_array(landmarks: list[Landmark]) -> np.ndarray:
    """Convert Face landmarks to a NumPy array.

    Args:
        landmarks: Five facial landmarks in SCRFD order.

    Returns:
        Landmark array with shape ``(5, 2)`` and dtype ``float32``.
    """
    return np.array(
        [[landmark.x, landmark.y] for landmark in landmarks],
        dtype=np.float32,
    )


def validate_alignment_input(image: np.ndarray, faces: list[Face]) -> None:
    """Validate inputs to the alignment pipeline.

    Args:
        image: Source image passed to the aligner.
        faces: Detected faces to align.

    Raises:
        ValueError: If the image or face list is invalid.
    """
    if image is None:
        raise ValueError("Input image must not be None.")

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Input image must be an HWC BGR image with 3 channels.")

    if image.size == 0:
        raise ValueError("Input image must not be empty.")

    if not isinstance(faces, list):
        raise ValueError("Faces must be provided as a list.")

    for index, face in enumerate(faces):
        if not isinstance(face, Face):
            raise ValueError(f"Expected Face at index {index}, got {type(face)!r}.")

        if len(face.landmarks) != 5:
            raise ValueError(
                f"Face at index {index} must have 5 landmarks, "
                f"got {len(face.landmarks)}."
            )
