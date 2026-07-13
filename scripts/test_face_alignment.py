from pathlib import Path

import cv2
import numpy as np

import common
from common import load_image
from backend.ai.alignment.aligner import FaceAligner
from backend.ai.alignment.constants import CANONICAL_FACE_SIZE
from backend.ai.alignment.template import get_reference_landmarks
from backend.ai.alignment.utils import landmarks_to_array
from backend.ai.detection.scrfd.detector import SCRFDDetector
from backend.app.domain.face import Face

_EXPECTED_ALIGNED_SHAPE = (CANONICAL_FACE_SIZE, CANONICAL_FACE_SIZE, 3)
_EXPECTED_TRANSFORM_SHAPE = (2, 3)
_LANDMARK_ALIGNMENT_TOLERANCE = 15.0




def validate_detected_faces(faces: list[Face]) -> None:
    """Validate detector output."""
    if not isinstance(faces, list):
        raise RuntimeError("Detector output must be a list of Face objects.")

    for face in faces:
        if not isinstance(face, Face):
            raise RuntimeError("Detector output must contain Face objects only.")


def validate_aligned_faces(faces: list[Face], detected_count: int) -> None:
    """Validate aligner output."""
    if len(faces) != detected_count:
        raise RuntimeError(
            f"Aligned face count mismatch: expected {detected_count}, "
            f"got {len(faces)}."
        )

    for index, face in enumerate(faces, start=1):
        if face.alignment is None:
            raise RuntimeError(f"Face {index} is missing alignment data.")

        aligned_image = face.alignment.aligned_image
        transform_matrix = face.alignment.transformation_matrix

        if aligned_image is None:
            raise RuntimeError(f"Face {index} is missing aligned_image.")

        if transform_matrix is None:
            raise RuntimeError(f"Face {index} is missing transformation_matrix.")

        if aligned_image.shape != _EXPECTED_ALIGNED_SHAPE:
            raise RuntimeError(
                f"Face {index} aligned_image shape mismatch: "
                f"expected {_EXPECTED_ALIGNED_SHAPE}, got {aligned_image.shape}."
            )

        if transform_matrix.shape != _EXPECTED_TRANSFORM_SHAPE:
            raise RuntimeError(
                f"Face {index} transformation_matrix shape mismatch: "
                f"expected {_EXPECTED_TRANSFORM_SHAPE}, "
                f"got {transform_matrix.shape}."
            )

        if transform_matrix.dtype != np.float32:
            raise RuntimeError(
                f"Face {index} transformation_matrix dtype must be float32, "
                f"got {transform_matrix.dtype}."
            )

        validate_landmark_alignment(face, index)


def validate_landmark_alignment(face: Face, index: int) -> None:
    """Verify all five landmarks align to the ArcFace template."""
    assert face.alignment is not None
    assert face.alignment.transformation_matrix is not None

    reference_landmarks = get_reference_landmarks()
    source_landmarks = landmarks_to_array(face.landmarks)
    transformed = cv2.transform(
        source_landmarks.reshape(5, 1, 2),
        face.alignment.transformation_matrix,
    ).reshape(5, 2)
    landmark_errors = np.max(np.abs(transformed - reference_landmarks), axis=1)
    max_error = float(np.max(landmark_errors))

    if max_error > _LANDMARK_ALIGNMENT_TOLERANCE:
        raise RuntimeError(
            f"Face {index} landmarks did not align to the ArcFace template "
            f"within tolerance {_LANDMARK_ALIGNMENT_TOLERANCE}px. "
            f"Maximum error: {max_error:.2f}px."
        )


def print_results(faces: list[Face]) -> None:
    """Print alignment validation results."""
    print("--------------------------------")
    print(f"Faces detected : {len(faces)}")
    print(f"Faces aligned  : {len(faces)}")
    print("--------------------------------")
    print()

    for index, face in enumerate(faces, start=1):
        assert face.alignment is not None
        assert face.alignment.transformation_matrix is not None
        assert face.alignment.aligned_image is not None

        print(f"Face {index}")
        print(f"  Confidence            : {face.confidence:.4f}")
        print(
            "  Aligned Image Shape   : "
            f"{face.alignment.aligned_image.shape}"
        )
        reference_landmarks = get_reference_landmarks()
        source_landmarks = landmarks_to_array(face.landmarks)
        transformed = cv2.transform(
            source_landmarks.reshape(5, 1, 2),
            face.alignment.transformation_matrix,
        ).reshape(5, 2)
        max_error = float(np.max(np.abs(transformed - reference_landmarks)))
        print(f"  Max Landmark Error    : {max_error:.2f}px")
        print("  Transformation Matrix :")
        print(face.alignment.transformation_matrix)
        print()

    print("--------------------------------")


def main() -> None:
    import sys

    if len(sys.argv) != 2:
        raise RuntimeError(
            "Usage: python scripts/test_face_alignment.py <image_path>"
        )

    image_path = Path(sys.argv[1])

    if not image_path.exists():
        raise RuntimeError(f"Image not found: {image_path}")

    image = load_image(image_path)
    detector = SCRFDDetector(common.SCRFD_MODEL_PATH)
    faces = detector.detect(image)

    validate_detected_faces(faces)
    detected_count = len(faces)

    aligner = FaceAligner()
    aligned_faces = aligner.align(image, faces)

    validate_aligned_faces(aligned_faces, detected_count)
    print_results(aligned_faces)


if __name__ == "__main__":
    main()
