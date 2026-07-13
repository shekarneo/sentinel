"""Run SCRFD detection, alignment, and blur analysis on an image."""

from pathlib import Path

import cv2

import common
from backend.ai.alignment.aligner import FaceAligner
from backend.ai.assessment.blur import BlurAnalyzer
from backend.ai.detection.scrfd.detector import SCRFDDetector
from backend.app.domain.face import Face


def load_image(image_path: Path):
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
    """Validate detector output.

    Args:
        faces: Faces returned by SCRFD.

    Raises:
        ValueError: If no faces were detected.
    """
    if not faces:
        raise ValueError("No faces detected in the input image.")


def validate_aligned_faces(faces: list[Face]) -> None:
    """Validate alignment output.

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


def main() -> None:
    """Run detection, alignment, and blur analysis on every detected face."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run SCRFD detection, alignment, and blur analysis."
    )
    parser.add_argument("image", type=Path, help="Path to input image")
    args = parser.parse_args()

    image = load_image(args.image)
    detector = SCRFDDetector(common.SCRFD_MODEL_PATH)
    aligner = FaceAligner()
    blur_analyzer = BlurAnalyzer()

    faces = detector.detect(image)
    validate_detected_faces(faces)

    aligner.align(image, faces)
    validate_aligned_faces(faces)

    print(f"Detected Faces: {len(faces)}")
    print()

    for index, face in enumerate(faces, start=1):
        assert face.alignment is not None
        assert face.alignment.aligned_image is not None

        result = blur_analyzer.evaluate(face.alignment.aligned_image)

        print(f"Face Index: {index}")
        print(f"Variance: {result.variance:.2f}")
        print(f"Score: {result.score:.4f}")
        print()


if __name__ == "__main__":
    main()
