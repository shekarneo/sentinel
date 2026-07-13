import numpy as np
import cv2

import common
from backend.ai.alignment.constants import NUM_LANDMARKS
from backend.ai.alignment.template import get_reference_landmarks
from backend.ai.alignment.transform import estimate_similarity_transform

_EXPECTED_TRANSFORM_SHAPE = (2, 3)
_LANDMARK_TOLERANCE = 1e-4


def validate_transform_matrix(transform: np.ndarray) -> None:
    """Validate the estimated affine transform matrix."""
    if transform.shape != _EXPECTED_TRANSFORM_SHAPE:
        raise RuntimeError(
            f"Unexpected transform shape: {transform.shape}. "
            f"Expected: {_EXPECTED_TRANSFORM_SHAPE}."
        )

    if transform.dtype != np.float32:
        raise RuntimeError(
            f"Unexpected transform dtype: {transform.dtype}. "
            "Expected: float32."
        )


def validate_round_trip(
    reference_landmarks: np.ndarray,
    transform: np.ndarray,
) -> None:
    """Verify the transform maps reference landmarks back to themselves."""
    points = reference_landmarks.reshape(NUM_LANDMARKS, 1, 2)
    transformed = cv2.transform(points, transform).reshape(NUM_LANDMARKS, 2)

    if not np.allclose(transformed, reference_landmarks, atol=_LANDMARK_TOLERANCE):
        raise RuntimeError(
            "Transformed landmarks do not match the reference template "
            f"within tolerance {_LANDMARK_TOLERANCE}."
        )


def validate_template_alignment(
    source_landmarks: np.ndarray,
    transform: np.ndarray,
    reference_landmarks: np.ndarray,
) -> None:
    """Verify all five landmarks align to the ArcFace template."""
    points = source_landmarks.reshape(NUM_LANDMARKS, 1, 2)
    transformed = cv2.transform(points, transform).reshape(NUM_LANDMARKS, 2)
    max_error = np.max(np.abs(transformed - reference_landmarks))

    if not np.allclose(transformed, reference_landmarks, atol=_LANDMARK_TOLERANCE):
        raise RuntimeError(
            "Five-point similarity transform did not reproduce the ArcFace "
            f"template within tolerance {_LANDMARK_TOLERANCE}. "
            f"Maximum landmark error: {max_error:.6f}."
        )


def main() -> None:
    reference_landmarks = get_reference_landmarks()

    transform = estimate_similarity_transform(
        reference_landmarks,
        reference_landmarks,
    )

    validate_transform_matrix(transform)
    validate_round_trip(reference_landmarks, transform)
    validate_template_alignment(
        reference_landmarks,
        transform,
        reference_landmarks,
    )

    print("--------------------------------")
    print("Similarity Transform (InsightFace 5-point)")
    print("--------------------------------")
    print()
    print("Transformation Matrix:")
    print(transform)
    print()
    print(f"Shape : {transform.shape}")
    print(f"Dtype : {transform.dtype}")
    print()
    print("Five-point round-trip validation passed.")
    print("--------------------------------")


if __name__ == "__main__":
    main()
