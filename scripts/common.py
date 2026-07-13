"""Shared bootstrap helpers for development scripts."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.config.configuration import resolve_scrfd_model_path
from backend.app.domain.face import Face

SCRFD_MODEL_PATH = resolve_scrfd_model_path()


def load_image(image_path: Path) -> np.ndarray:
    """Load an image from disk using OpenCV.

    Args:
        image_path: Path to the input image.

    Returns:
        Loaded BGR image.

    Raises:
        RuntimeError: If the image cannot be read.
    """
    image = cv2.imread(str(image_path))

    if image is None:
        raise RuntimeError(f"Failed to load image: {image_path}")

    return image


def validate_detected_faces(faces: list[Face]) -> None:
    """Validate that SCRFD detected at least one face.

    Args:
        faces: Faces returned by SCRFD.

    Raises:
        ValueError: If no faces were detected.
    """
    if not faces:
        raise ValueError("No faces detected in the input image.")


def validate_aligned_faces(faces: list[Face]) -> None:
    """Validate that every face has alignment data populated.

    Args:
        faces: Faces after alignment.

    Raises:
        ValueError: If alignment data is missing.
    """
    for index, face in enumerate(faces):
        if face.alignment is None:
            raise ValueError(f"Face at index {index} is missing alignment data.")

        if face.alignment.aligned_image is None:
            raise ValueError(f"Face at index {index} is missing aligned image.")
