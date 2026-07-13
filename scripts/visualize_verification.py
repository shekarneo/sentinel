"""Visualize verification decisions for every detected face."""

from pathlib import Path

import cv2
import numpy as np

import common
from common import load_image
from backend.ai.alignment.aligner import FaceAligner
from backend.ai.alignment.constants import CANONICAL_FACE_SIZE
from backend.ai.alignment.utils import crop_face_region
from backend.ai.detection.scrfd.detector import SCRFDDetector
from backend.ai.embedding.embedder import FaceEmbedder
from backend.ai.search.config import get_search_index_path, get_search_mapping_path
from backend.ai.search.searcher import FaceSearcher, create_search_engine_components
from backend.ai.verification.config import load_verification_thresholds
from backend.ai.verification.verifier import FaceVerifier
from backend.app.domain.face import Face
from backend.app.domain.search import SearchResults
from backend.app.domain.verification import VerificationResult
from backend.app.services.identity_service import IdentityService

DISPLAY_SCALE = 3
PANEL_BORDER = 4
TOP_K = 5

_DECISION_COLORS = {
    "accept": (0, 200, 0),
    "reject": (0, 0, 220),
    "unknown": (0, 165, 255),
}




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


def draw_verification_block(
    panel: np.ndarray,
    search_results: SearchResults,
    verification_result: VerificationResult,
) -> np.ndarray:
    """Overlay verification details on a visualization panel."""
    output = panel.copy()
    top_match = search_results.results[0] if search_results.results else None
    decision = verification_result.decision.value.upper()
    identity = verification_result.matched_identity_id or "(none)"

    lines = [
        "Verification",
        f"Top Match: {identity}",
        f"Similarity: {verification_result.similarity_score:.4f}",
        f"Threshold: {verification_result.threshold:.4f}",
        f"Decision: {decision}",
        f"Time: {verification_result.verification_time_ms:.2f} ms",
    ]

    if top_match is not None and top_match.identity_id != identity:
        lines.insert(2, f"Search #1: {top_match.identity_id}")

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


def build_verification_panel(
    image: np.ndarray,
    face: Face,
    search_results: SearchResults,
    verification_result: VerificationResult,
) -> np.ndarray:
    """Build a verification visualization panel for one face."""
    if face.alignment is None or face.alignment.aligned_image is None:
        raise RuntimeError("Face is missing aligned image.")

    original_crop = build_square_panel(crop_face_region(image, face))
    aligned_face = face.alignment.aligned_image.copy()
    decision_color = _DECISION_COLORS.get(
        verification_result.decision.value,
        (255, 255, 255),
    )

    original_panel = add_panel_border(original_crop, (0, 0, 255))
    aligned_panel = add_panel_border(aligned_face, (0, 255, 0))
    aligned_panel = draw_verification_block(
        aligned_panel,
        search_results,
        verification_result,
    )

    original_panel = upscale_panel(original_panel)
    aligned_panel = upscale_panel(aligned_panel)
    panel = np.hstack([original_panel, aligned_panel])
    panel = add_panel_border(panel, decision_color)

    cv2.putText(
        panel,
        "Detected Face",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        panel,
        "Aligned Face + Verification",
        (original_panel.shape[1] + 12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )

    return panel


def save_verification_panels(
    image: np.ndarray,
    faces: list[Face],
    search_results: list[SearchResults],
    verification_results: list[VerificationResult],
    image_path: Path,
) -> list[Path]:
    """Save verification visualization panels for each face."""
    saved_paths: list[Path] = []

    for index, (face, results, verification) in enumerate(
        zip(faces, search_results, verification_results),
        start=1,
    ):
        panel = build_verification_panel(image, face, results, verification)
        output_path = image_path.with_name(
            f"{image_path.stem}_verification_{index}{image_path.suffix}"
        )

        if not cv2.imwrite(str(output_path), panel):
            raise RuntimeError(f"Failed to save verification panel: {output_path}")

        saved_paths.append(output_path)
        print(f"Saved verification panel to: {output_path}")

    return saved_paths


def display_faces(
    image: np.ndarray,
    faces: list[Face],
    search_results: list[SearchResults],
    verification_results: list[VerificationResult],
    image_path: Path,
) -> None:
    """Display verification visualization for each detected face."""
    window_name = "Verification"

    try:
        for index, (face, results, verification) in enumerate(
            zip(faces, search_results, verification_results),
            start=1,
        ):
            panel = build_verification_panel(image, face, results, verification)
            cv2.imshow(window_name, panel)

            print(
                f"Face {index}/{len(faces)} - "
                f"identity={verification.matched_identity_id or '(none)'} "
                f"sim={verification.similarity_score:.4f} "
                f"threshold={verification.threshold:.4f} "
                f"decision={verification.decision.value.upper()} - "
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
                    save_verification_panels(
                        image,
                        faces,
                        search_results,
                        verification_results,
                        image_path,
                    )

        cv2.destroyAllWindows()
    except cv2.error:
        save_verification_panels(
            image,
            faces,
            search_results,
            verification_results,
            image_path,
        )
        print("OpenCV GUI is unavailable in this environment.")


def main() -> None:
    """Run detection, embedding, search, verification, and visualization."""
    import sys

    if len(sys.argv) != 2:
        raise RuntimeError("Usage: python scripts/visualize_verification.py <image_path>")

    image_path = Path(sys.argv[1])

    if not image_path.exists():
        raise RuntimeError(f"Image not found: {image_path}")

    for path in (get_search_index_path(), get_search_mapping_path()):
        path.unlink(missing_ok=True)

    image = load_image(image_path)
    detector = SCRFDDetector(common.SCRFD_MODEL_PATH)
    faces = detector.detect(image)

    print(f"Detected {len(faces)} face(s)")
    print(f"Verification Threshold: {load_verification_thresholds().similarity_threshold:.4f}")

    aligner = FaceAligner()
    aligner.align(image, faces)

    embedder = FaceEmbedder()
    embedder.embed(faces)

    repository, search_index = create_search_engine_components()
    identity_service = IdentityService(repository=repository, search_index=search_index)
    searcher = FaceSearcher(repository=repository, search_index=search_index)
    verifier = FaceVerifier()

    for index, face in enumerate(faces, start=1):
        assert face.embedding is not None
        assert face.embedding.vector is not None
        identity_service.enroll(f"gallery_face_{index}", face.embedding.vector)

    identity_service.save_gallery()
    search_results = searcher.search(faces, top_k=TOP_K)
    verification_results = verifier.verify(faces, search_results)

    save_verification_panels(
        image,
        faces,
        search_results,
        verification_results,
        image_path,
    )
    display_faces(
        image,
        faces,
        search_results,
        verification_results,
        image_path,
    )


if __name__ == "__main__":
    main()
