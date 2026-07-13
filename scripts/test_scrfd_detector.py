from pathlib import Path

import cv2

import common
from common import load_image
from backend.ai.detection.scrfd.detector import SCRFDDetector
from backend.app.domain.face import Face

LANDMARK_COLORS = [
    (0, 255, 255),
    (0, 255, 0),
    (255, 0, 0),
    (255, 255, 0),
    (255, 0, 255),
]




def validate_faces(faces: list[Face]) -> None:
    """Validate detector output."""
    if not isinstance(faces, list):
        raise RuntimeError("Detector output must be a list of Face objects.")

    for face in faces:
        if not isinstance(face, Face):
            raise RuntimeError("Detector output must contain Face objects only.")

        if not 0.0 <= face.confidence <= 1.0:
            raise RuntimeError(
                f"Confidence must be within [0, 1], got {face.confidence}."
            )

        if len(face.landmarks) != 5:
            raise RuntimeError(
                f"Each face must have 5 landmarks, got {len(face.landmarks)}."
            )


def print_faces(faces: list[Face]) -> None:
    """Print detected face details."""
    print("--------------------------------")
    print(f"Faces detected: {len(faces)}")
    print("--------------------------------")
    print()

    for index, face in enumerate(faces, start=1):
        bbox = face.bounding_box

        print(f"Face {index}")
        print(f"  Confidence   : {face.confidence:.4f}")
        print(
            "  Bounding Box : "
            f"x={bbox.x:.1f}, y={bbox.y:.1f}, "
            f"width={bbox.width:.1f}, height={bbox.height:.1f}"
        )

        for landmark_index, landmark in enumerate(face.landmarks, start=1):
            print(
                f"  Landmark {landmark_index} : "
                f"x={landmark.x:.1f}, y={landmark.y:.1f}"
            )

        print()

    print("--------------------------------")


def main() -> None:
    import sys

    if len(sys.argv) != 2:
        raise RuntimeError(
            "Usage: python scripts/test_scrfd_detector.py <image_path>"
        )

    image_path = Path(sys.argv[1])

    if not image_path.exists():
        raise RuntimeError(f"Image not found: {image_path}")

    detector = SCRFDDetector(common.SCRFD_MODEL_PATH)
    faces = detector.detect(load_image(image_path))

    validate_faces(faces)
    print_faces(faces)


if __name__ == "__main__":
    main()
