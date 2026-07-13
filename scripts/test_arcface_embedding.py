"""Run SCRFD detection, alignment, and ArcFace embedding on an image."""

from pathlib import Path

import cv2
import numpy as np

import common
from common import load_image
from backend.ai.alignment.aligner import FaceAligner
from backend.ai.detection.scrfd.detector import SCRFDDetector
from backend.ai.embedding.embedder import FaceEmbedder
from backend.app.domain.face import Face

_NORM_TOLERANCE = 1e-5
_EXPECTED_DIMENSION = 512




def validate_detected_faces(faces: list[Face]) -> None:
    """Validate detector output.

    Args:
        faces: Faces returned by SCRFD.

    Raises:
        ValueError: If no faces were detected.
    """
    if not faces:
        raise ValueError("No faces detected in the input image.")


def validate_embedding_output(face: Face) -> None:
    """Validate embedding fields for a single face.

    Args:
        face: Face after embedding.

    Raises:
        ValueError: If embedding data is missing or invalid.
    """
    if face.embedding is None:
        raise ValueError("Face is missing embedding data.")

    embedding = face.embedding

    if embedding.vector is None:
        raise ValueError("Embedding vector was not populated.")

    if embedding.dimension != _EXPECTED_DIMENSION:
        raise ValueError(
            f"Expected embedding dimension {_EXPECTED_DIMENSION}, "
            f"got {embedding.dimension}."
        )

    if not embedding.normalized:
        raise ValueError("Embedding must be marked as normalized.")

    vector_norm = float(np.linalg.norm(embedding.vector))
    if abs(vector_norm - 1.0) > _NORM_TOLERANCE:
        raise ValueError(
            f"Expected unit norm 1.0 ± {_NORM_TOLERANCE}, got {vector_norm:.8f}."
        )

    if embedding.model_name is None or not embedding.model_name:
        raise ValueError("Model name was not populated.")


def print_face_embedding(index: int, face: Face) -> None:
    """Print embedding metrics for a single face.

    Args:
        index: One-based face index.
        face: Embedded face.
    """
    assert face.embedding is not None
    assert face.embedding.vector is not None

    embedding = face.embedding
    vector_norm = float(np.linalg.norm(embedding.vector))

    print(f"Face Index: {index}")
    print(f"Embedding Dimension: {embedding.dimension}")
    print(f"Vector Norm: {vector_norm:.8f}")
    print(
        "Inference Time: "
        f"{embedding.inference_time_ms:.3f} ms"
        if embedding.inference_time_ms is not None
        else "Inference Time: n/a"
    )
    print(f"Model Name: {embedding.model_name}")
    print()


def main() -> None:
    """Run detection, alignment, and embedding on every detected face."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run SCRFD detection, alignment, and ArcFace embedding."
    )
    parser.add_argument("image", type=Path, help="Path to input image")
    args = parser.parse_args()

    image = load_image(args.image)
    detector = SCRFDDetector(common.SCRFD_MODEL_PATH)
    aligner = FaceAligner()
    embedder = FaceEmbedder()

    faces = detector.detect(image)
    validate_detected_faces(faces)

    aligner.align(image, faces)
    embedder.embed(faces)

    print(f"Detected Faces: {len(faces)}")
    print()

    for index, face in enumerate(faces, start=1):
        validate_embedding_output(face)
        print_face_embedding(index, face)


if __name__ == "__main__":
    main()
