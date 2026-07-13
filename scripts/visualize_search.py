"""Visualize FAISS search matches for every detected face."""

from pathlib import Path

import cv2
import numpy as np

import common
from backend.ai.alignment.aligner import FaceAligner
from backend.ai.alignment.constants import CANONICAL_FACE_SIZE
from backend.ai.alignment.utils import crop_face_region
from backend.ai.detection.scrfd.detector import SCRFDDetector
from backend.ai.embedding.embedder import FaceEmbedder
from backend.ai.search.config import get_search_index_path, get_search_mapping_path
from backend.ai.search.searcher import FaceSearcher, create_search_engine_components
from backend.app.services.identity_service import IdentityService
from backend.app.domain.face import Face
from backend.app.domain.search import SearchResults

DISPLAY_SCALE = 3
PANEL_BORDER = 4
TOP_K = 5


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


def draw_match_block(panel: np.ndarray, results: SearchResults) -> np.ndarray:
    """Overlay top search matches on a visualization panel."""
    output = panel.copy()

    lines = [f"Top-{TOP_K} Matches", f"Search: {results.search_time_ms:.2f} ms"]

    for match in results.results[:TOP_K]:
        lines.append(
            f"#{match.rank} {match.identity_id} sim={match.score:.4f}"
        )

    if len(results.results) == 0:
        lines.append("(no matches)")

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


def build_search_panel(
    image: np.ndarray,
    face: Face,
    results: SearchResults,
) -> np.ndarray:
    """Build a side-by-side original crop and aligned face search panel."""
    if face.alignment is None or face.alignment.aligned_image is None:
        raise RuntimeError("Face is missing aligned image.")

    original_crop = build_square_panel(crop_face_region(image, face))
    aligned_face = face.alignment.aligned_image.copy()

    original_panel = add_panel_border(original_crop, (0, 0, 255))
    aligned_panel = add_panel_border(aligned_face, (0, 255, 0))
    aligned_panel = draw_match_block(aligned_panel, results)

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
        "Aligned Face + Search",
        (original_panel.shape[1] + 12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )

    return panel


def save_search_panels(
    image: np.ndarray,
    faces: list[Face],
    search_results: list[SearchResults],
    image_path: Path,
) -> list[Path]:
    """Save search visualization panels for each face."""
    saved_paths: list[Path] = []

    for index, (face, results) in enumerate(zip(faces, search_results), start=1):
        panel = build_search_panel(image, face, results)
        output_path = image_path.with_name(
            f"{image_path.stem}_search_{index}{image_path.suffix}"
        )

        if not cv2.imwrite(str(output_path), panel):
            raise RuntimeError(f"Failed to save search panel: {output_path}")

        saved_paths.append(output_path)
        print(f"Saved search panel to: {output_path}")

    return saved_paths


def display_faces(
    image: np.ndarray,
    faces: list[Face],
    search_results: list[SearchResults],
    image_path: Path,
) -> None:
    """Display search visualization for each detected face."""
    window_name = "FAISS Search"

    try:
        for index, (face, results) in enumerate(zip(faces, search_results), start=1):
            panel = build_search_panel(image, face, results)
            cv2.imshow(window_name, panel)

            top_match = results.results[0] if results.results else None
            summary = (
                f"identity={top_match.identity_id} sim={top_match.score:.4f}"
                if top_match is not None
                else "no matches"
            )
            print(
                f"Face {index}/{len(faces)} - {summary} - "
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
                    save_search_panels(image, faces, search_results, image_path)

        cv2.destroyAllWindows()
    except cv2.error:
        save_search_panels(image, faces, search_results, image_path)
        print("OpenCV GUI is unavailable in this environment.")


def main() -> None:
    """Run detection, embedding, gallery enrollment, search, and visualization."""
    import sys

    if len(sys.argv) != 2:
        raise RuntimeError("Usage: python scripts/visualize_search.py <image_path>")

    image_path = Path(sys.argv[1])

    if not image_path.exists():
        raise RuntimeError(f"Image not found: {image_path}")

    for path in (get_search_index_path(), get_search_mapping_path()):
        if path.exists():
            path.unlink()

    image = load_image(image_path)
    detector = SCRFDDetector(common.SCRFD_MODEL_PATH)
    faces = detector.detect(image)

    print(f"Detected {len(faces)} face(s)")

    aligner = FaceAligner()
    aligner.align(image, faces)

    embedder = FaceEmbedder()
    embedder.embed(faces)

    repository, search_index = create_search_engine_components()
    identity_service = IdentityService(repository=repository, search_index=search_index)
    searcher = FaceSearcher(repository=repository, search_index=search_index)

    for index, face in enumerate(faces, start=1):
        assert face.embedding is not None
        assert face.embedding.vector is not None
        identity_service.enroll(f"gallery_face_{index}", face.embedding.vector)

    identity_service.save_gallery()
    search_results = searcher.search(faces, top_k=TOP_K)

    save_search_panels(image, faces, search_results, image_path)
    display_faces(image, faces, search_results, image_path)


if __name__ == "__main__":
    main()
