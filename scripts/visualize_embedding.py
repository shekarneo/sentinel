"""Visualize ArcFace embeddings for every detected face."""

from pathlib import Path

import cv2
import numpy as np

import common
from backend.ai.alignment.aligner import FaceAligner
from backend.ai.alignment.constants import CANONICAL_FACE_SIZE
from backend.ai.alignment.utils import crop_face_region
from backend.ai.detection.scrfd.detector import SCRFDDetector
from backend.ai.embedding.embedder import FaceEmbedder
from backend.app.domain.face import Face

DISPLAY_SCALE = 3
PANEL_BORDER = 4
_PREVIEW_VALUES = 10


def load_image(image_path: Path) -> np.ndarray:
    """Load an image from disk using OpenCV."""
    image = cv2.imread(str(image_path))

    if image is None:
        raise RuntimeError(f"Failed to load image: {image_path}")

    return image


def build_square_panel(image: np.ndarray) -> np.ndarray:
    """Resize an image patch to the canonical face display size."""
    return cv2.resize(
        image,
        (CANONICAL_FACE_SIZE, CANONICAL_FACE_SIZE),
        interpolation=cv2.INTER_LINEAR,
    )


def upscale_panel(panel: np.ndarray) -> np.ndarray:
    """Upscale a panel for easier visual inspection."""
    display_size = CANONICAL_FACE_SIZE * DISPLAY_SCALE
    return cv2.resize(
        panel,
        (display_size, display_size),
        interpolation=cv2.INTER_NEAREST,
    )


def add_panel_border(panel: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    """Add a colored border around a panel."""
    return cv2.copyMakeBorder(
        panel,
        PANEL_BORDER,
        PANEL_BORDER,
        PANEL_BORDER,
        PANEL_BORDER,
        borderType=cv2.BORDER_CONSTANT,
        value=color,
    )


def draw_embedding_block(panel: np.ndarray, face: Face) -> np.ndarray:
    """Overlay embedding metadata on a visualization panel."""
    output = panel.copy()

    if face.embedding is None or face.embedding.vector is None:
        raise RuntimeError("Face is missing embedding data.")

    embedding = face.embedding
    vector_norm = float(np.linalg.norm(embedding.vector))
    preview = ", ".join(f"{value:.4f}" for value in embedding.vector[:_PREVIEW_VALUES])

    lines = [
        f"Dimension: {embedding.dimension}",
        f"Vector Norm: {vector_norm:.6f}",
        f"Model: {embedding.model_name}",
        (
            "Inference: "
            f"{embedding.inference_time_ms:.2f} ms"
            if embedding.inference_time_ms is not None
            else "Inference: n/a"
        ),
        f"First {_PREVIEW_VALUES}: [{preview}]",
    ]

    y_offset = 18
    for line in lines:
        cv2.putText(
            output,
            line,
            (8, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        y_offset += 16

    return output


def build_embedding_panel(image: np.ndarray, face: Face) -> np.ndarray:
    """Build a side-by-side original crop and aligned face embedding panel."""
    if face.alignment is None or face.alignment.aligned_image is None:
        raise RuntimeError("Face is missing aligned image.")

    original_crop = build_square_panel(crop_face_region(image, face))
    aligned_face = face.alignment.aligned_image.copy()

    original_panel = add_panel_border(original_crop, (0, 0, 255))
    aligned_panel = add_panel_border(aligned_face, (0, 255, 0))
    aligned_panel = draw_embedding_block(aligned_panel, face)

    original_panel = upscale_panel(original_panel)
    aligned_panel = upscale_panel(aligned_panel)

    panel = np.hstack([original_panel, aligned_panel])

    cv2.putText(
        panel,
        "Original Face",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        panel,
        "Aligned Face + Embedding",
        (original_panel.shape[1] + 12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )

    return panel


def save_embedding_panels(
    image: np.ndarray,
    faces: list[Face],
    image_path: Path,
) -> list[Path]:
    """Save embedding visualization panels for each face."""
    saved_paths: list[Path] = []

    for index, face in enumerate(faces, start=1):
        panel = build_embedding_panel(image, face)
        output_path = image_path.with_name(
            f"{image_path.stem}_embedding_{index}{image_path.suffix}"
        )

        if not cv2.imwrite(str(output_path), panel):
            raise RuntimeError(f"Failed to save embedding panel: {output_path}")

        saved_paths.append(output_path)
        print(f"Saved embedding panel to: {output_path}")

    return saved_paths


def display_faces(
    image: np.ndarray,
    faces: list[Face],
    image_path: Path,
) -> None:
    """Display embedding visualization for each detected face."""
    window_name = "ArcFace Embedding"

    try:
        for index, face in enumerate(faces, start=1):
            panel = build_embedding_panel(image, face)
            cv2.imshow(window_name, panel)

            assert face.embedding is not None
            assert face.embedding.vector is not None

            vector_norm = float(np.linalg.norm(face.embedding.vector))
            print(
                f"Face {index}/{len(faces)} - "
                f"dimension={face.embedding.dimension} "
                f"norm={vector_norm:.6f} - "
                "Press ESC to exit, N for next face, S to save outputs."
            )

            while True:
                key = cv2.waitKey(0) & 0xFF

                if key == 27:
                    cv2.destroyAllWindows()
                    return

                if key in (ord("n"), ord("N")):
                    break

                if key in (ord("s"), ord("S")):
                    save_embedding_panels(image, faces, image_path)

        cv2.destroyAllWindows()
    except cv2.error:
        save_embedding_panels(image, faces, image_path)
        print("OpenCV GUI is unavailable in this environment.")


def main() -> None:
    """Run detection, alignment, embedding, and visualization."""
    import sys

    if len(sys.argv) != 2:
        raise RuntimeError(
            "Usage: python scripts/visualize_embedding.py <image_path>"
        )

    image_path = Path(sys.argv[1])

    if not image_path.exists():
        raise RuntimeError(f"Image not found: {image_path}")

    image = load_image(image_path)
    detector = SCRFDDetector(common.SCRFD_MODEL_PATH)
    faces = detector.detect(image)

    print(f"Detected {len(faces)} face(s)")

    aligner = FaceAligner()
    aligner.align(image, faces)

    embedder = FaceEmbedder()
    embedder.embed(faces)

    save_embedding_panels(image, faces, image_path)
    display_faces(image, faces, image_path)


if __name__ == "__main__":
    main()
