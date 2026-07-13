"""
Geometric transform utilities for face alignment.

Handles similarity transform estimation and face warping.

Transform estimation follows the InsightFace/ArcFace preprocessing pipeline:
all five SCRFD landmarks are used with ``skimage.transform.SimilarityTransform``
(Umeyama least-squares). This matches the reference implementation used during
ArcFace training and is preferred over partial 3-point OpenCV estimation,
which ignores mouth geometry and diverges from InsightFace normalization.
"""

import cv2
import numpy as np
from skimage import transform as trans

from backend.ai.alignment.constants import NUM_LANDMARKS

_EXPECTED_LANDMARK_SHAPE = (NUM_LANDMARKS, 2)
_EXPECTED_TRANSFORM_SHAPE = (2, 3)


def _validate_landmarks(landmarks: np.ndarray, name: str) -> np.ndarray:
    """Validate landmark array shape, dtype, and values.

    Args:
        landmarks: Landmark coordinates to validate.
        name: Argument name used in validation error messages.

    Returns:
        Landmark array as ``float32`` with shape ``(5, 2)``.

    Raises:
        ValueError: If the landmark array is invalid.
    """
    if not isinstance(landmarks, np.ndarray):
        raise ValueError(f"{name} must be a NumPy ndarray.")

    if landmarks.shape != _EXPECTED_LANDMARK_SHAPE:
        raise ValueError(
            f"{name} must have shape {_EXPECTED_LANDMARK_SHAPE}, "
            f"got {landmarks.shape}."
        )

    converted = np.asarray(landmarks, dtype=np.float32)

    if not np.isfinite(converted).all():
        raise ValueError(f"{name} must contain only finite values.")

    return converted


def _validate_warp_image(image: np.ndarray) -> None:
    """Validate an image passed to the warping routine.

    Args:
        image: Source image in HWC BGR layout.

    Raises:
        ValueError: If the image is invalid.
    """
    if image is None:
        raise ValueError("Input image must not be None.")

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Input image must be an HWC BGR image with 3 channels.")

    if image.size == 0:
        raise ValueError("Input image must not be empty.")


def _validate_transform_matrix(transform_matrix: np.ndarray) -> np.ndarray:
    """Validate an affine transform matrix.

    Args:
        transform_matrix: Affine transform matrix to validate.

    Returns:
        Transform matrix as ``float32`` with shape ``(2, 3)``.

    Raises:
        ValueError: If the transform matrix is invalid.
    """
    if not isinstance(transform_matrix, np.ndarray):
        raise ValueError("transform_matrix must be a NumPy ndarray.")

    if transform_matrix.shape != _EXPECTED_TRANSFORM_SHAPE:
        raise ValueError(
            "transform_matrix must have shape "
            f"{_EXPECTED_TRANSFORM_SHAPE}, got {transform_matrix.shape}."
        )

    converted = np.asarray(transform_matrix, dtype=np.float32)

    if not np.isfinite(converted).all():
        raise ValueError("transform_matrix must contain only finite values.")

    return converted


def estimate_similarity_transform(
    source_landmarks: np.ndarray,
    destination_landmarks: np.ndarray,
) -> np.ndarray:
    """Estimate a 2x3 similarity transform matrix.

    Maps ``source_landmarks`` (detected SCRFD landmarks) to
    ``destination_landmarks`` (ArcFace reference template) using a
    similarity transform (rotation, uniform scale, translation).

    Estimation uses ``skimage.transform.SimilarityTransform`` on all five
    landmarks. This is the standard InsightFace/ArcFace approach and keeps
    preprocessing mathematically compatible with ArcFace embedding models.

    Compared with a 3-point OpenCV workaround, the 5-point similarity
    transform balances error across eyes, nose, and mouth so the full face
    geometry matches the template used during ArcFace training.

    Args:
        source_landmarks: Detected landmark coordinates with shape ``(5, 2)``.
        destination_landmarks: Reference landmark coordinates with shape
            ``(5, 2)``.

    Returns:
        Affine transform matrix with shape ``(2, 3)`` and dtype ``float32``.

    Raises:
        ValueError: If either landmark array is invalid.
        RuntimeError: If similarity transform estimation fails.
    """
    source = _validate_landmarks(source_landmarks, "source_landmarks")
    destination = _validate_landmarks(
        destination_landmarks,
        "destination_landmarks",
    )

    similarity = trans.SimilarityTransform()
    estimated = similarity.estimate(source, destination)

    if not estimated:
        raise RuntimeError("Similarity transform estimation failed.")

    transform = similarity.params[:2, :].astype(np.float32)

    if not np.isfinite(transform).all():
        raise RuntimeError("Similarity transform estimation failed.")

    return transform


def warp_face(
    image: np.ndarray,
    transform_matrix: np.ndarray,
    output_size: tuple[int, int],
) -> np.ndarray:
    """Warp an image using the given similarity transform.

    Produces a canonical aligned face crop at the specified output size.

    Args:
        image: Source image in HWC BGR layout.
        transform_matrix: Affine transform matrix with shape ``(2, 3)``.
        output_size: Target crop size as ``(width, height)``.

    Returns:
        Aligned face crop with shape ``(height, width, 3)`` and the same
        dtype as the input image.

    Raises:
        ValueError: If the image, transform matrix, or output size is invalid.
    """
    _validate_warp_image(image)
    transform = _validate_transform_matrix(transform_matrix)

    output_width, output_height = output_size

    if output_width <= 0 or output_height <= 0:
        raise ValueError("output_size dimensions must be positive.")

    aligned_face = cv2.warpAffine(
        image,
        transform,
        (output_width, output_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    return aligned_face.astype(image.dtype, copy=False)
