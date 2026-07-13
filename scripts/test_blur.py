"""Run SCRFD detection, alignment, and blur analysis on an image."""

from pathlib import Path

import common
from common import load_image, validate_aligned_faces, validate_detected_faces
from backend.ai.alignment.aligner import FaceAligner
from backend.ai.assessment.blur import BlurAnalyzer
from backend.ai.detection.scrfd.detector import SCRFDDetector


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
