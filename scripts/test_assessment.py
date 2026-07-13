"""Run SCRFD detection, alignment, and full face assessment on an image."""

from pathlib import Path

import cv2

import common
from common import load_image
from backend.ai.alignment.aligner import FaceAligner
from backend.ai.assessment.assessor import FaceAssessor
from backend.ai.detection.scrfd.detector import SCRFDDetector
from backend.app.domain.face import Face




def validate_detected_faces(faces: list[Face]) -> None:
    """Validate detector output.

    Args:
        faces: Faces returned by SCRFD.

    Raises:
        ValueError: If no faces were detected.
    """
    if not faces:
        raise ValueError("No faces detected in the input image.")


def validate_assessment_output(face: Face) -> None:
    """Validate that all assessment fields are populated.

    Args:
        face: Face after assessment.

    Raises:
        ValueError: If assessment data is incomplete.
    """
    if face.assessment is None:
        raise ValueError("Face is missing assessment data.")

    assessment = face.assessment
    required_sections = (
        ("blur", assessment.blur),
        ("brightness", assessment.brightness),
        ("pose", assessment.pose),
        ("visibility", assessment.visibility),
        ("size", assessment.size),
    )

    for name, value in required_sections:
        if value is None:
            raise ValueError(f"Assessment section '{name}' was not populated.")

    if assessment.overall_score is None:
        raise ValueError("Overall score was not populated.")

    if assessment.is_acceptable is None:
        raise ValueError("Acceptability flag was not populated.")


def print_face_assessment(index: int, face: Face) -> None:
    """Print assessment metrics for a single face.

    Args:
        index: One-based face index.
        face: Assessed face.
    """
    assert face.assessment is not None
    assert face.assessment.blur is not None
    assert face.assessment.brightness is not None
    assert face.assessment.pose is not None
    assert face.assessment.visibility is not None
    assert face.assessment.size is not None

    assessment = face.assessment

    print(f"Face Index: {index}")
    print(f"Blur Score: {assessment.blur.score:.4f}")
    print(f"Brightness Score: {assessment.brightness.score:.4f}")
    print(
        "Pose: "
        f"yaw={assessment.pose.yaw:.2f} "
        f"pitch={assessment.pose.pitch:.2f} "
        f"roll={assessment.pose.roll:.2f} "
        f"score={assessment.pose.score:.4f}"
    )
    print(
        "Visibility: "
        f"ratio={assessment.visibility.visible_ratio:.4f} "
        f"score={assessment.visibility.score:.4f}"
    )
    print(
        "Face Size: "
        f"width={assessment.size.width:.1f} "
        f"height={assessment.size.height:.1f} "
        f"score={assessment.size.score:.4f}"
    )
    print(f"Overall Score: {assessment.overall_score:.4f}")
    print(f"Acceptable: {assessment.is_acceptable}")
    print()


def main() -> None:
    """Run detection, alignment, and assessment on every detected face."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run SCRFD detection, alignment, and face assessment."
    )
    parser.add_argument("image", type=Path, help="Path to input image")
    args = parser.parse_args()

    image = load_image(args.image)
    detector = SCRFDDetector(common.SCRFD_MODEL_PATH)
    aligner = FaceAligner()
    assessor = FaceAssessor()

    faces = detector.detect(image)
    validate_detected_faces(faces)

    aligner.align(image, faces)
    assessor.assess(image, faces)

    print(f"Detected Faces: {len(faces)}")
    print()

    for index, face in enumerate(faces, start=1):
        validate_assessment_output(face)
        print_face_assessment(index, face)


if __name__ == "__main__":
    main()
