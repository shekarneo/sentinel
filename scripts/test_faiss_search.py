"""Run SCRFD detection, alignment, embedding, and FAISS gallery search."""

from pathlib import Path

import cv2
import numpy as np

import common
from backend.ai.alignment.aligner import FaceAligner
from backend.ai.detection.scrfd.detector import SCRFDDetector
from backend.ai.embedding.embedder import FaceEmbedder
from backend.ai.search.config import get_search_index_path, get_search_mapping_path
from backend.ai.search.searcher import FaceSearcher, create_search_engine_components
from backend.app.services.identity_service import IdentityService
from backend.app.domain.face import Face


def load_image(image_path: Path) -> np.ndarray:
    """Load an image from disk using OpenCV."""
    image = cv2.imread(str(image_path))

    if image is None:
        raise RuntimeError(f"Failed to load image: {image_path}")

    return image


def build_gallery_identities(faces: list[Face]) -> list[str]:
    """Create deterministic gallery identity identifiers for detected faces."""
    return [f"gallery_face_{index}" for index in range(1, len(faces) + 1)]


def print_search_results(
    face_index: int,
    results,
    *,
    top_k: int,
) -> None:
    """Print top-k search matches for a probe face."""
    print(f"Probe Face Index: {face_index}")
    print(f"Search Time: {results.search_time_ms:.3f} ms")
    print(f"Provider: {results.provider}")
    print(f"Top-{top_k} Matches:")

    if not results.results:
        print("  (no matches)")
        print()
        return

    for match in results.results[:top_k]:
        print(
            f"  Rank {match.rank}: identity={match.identity_id} "
            f"similarity={match.score:.6f} distance={1.0 - match.score:.6f}"
        )

    print()


def main() -> None:
    """Run detection, embedding, gallery enrollment, and FAISS search."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run SCRFD, alignment, embedding, and FAISS search."
    )
    parser.add_argument("image", type=Path, help="Path to input image")
    parser.add_argument("--top-k", type=int, default=5, help="Top-K matches")
    parser.add_argument(
        "--reset-gallery",
        action="store_true",
        help="Delete existing gallery index and mapping before enrollment",
    )
    args = parser.parse_args()

    if args.reset_gallery:
        for path in (get_search_index_path(), get_search_mapping_path()):
            if path.exists():
                path.unlink()

    image = load_image(args.image)
    detector = SCRFDDetector(common.SCRFD_MODEL_PATH)
    aligner = FaceAligner()
    embedder = FaceEmbedder()
    repository, search_index = create_search_engine_components()
    identity_service = IdentityService(repository=repository, search_index=search_index)
    searcher = FaceSearcher(repository=repository, search_index=search_index)

    faces = detector.detect(image)
    if not faces:
        raise ValueError("No faces detected in the input image.")

    aligner.align(image, faces)
    embedder.embed(faces)

    identity_ids = build_gallery_identities(faces)
    for face, identity_id in zip(faces, identity_ids):
        assert face.embedding is not None
        assert face.embedding.vector is not None
        identity_service.enroll(identity_id, face.embedding.vector)

    identity_service.save_gallery()

    print(f"Detected Faces: {len(faces)}")
    print(f"Gallery Size: {identity_service.gallery_size}")
    print()

    search_results = searcher.search(faces, top_k=args.top_k)

    for index, results in enumerate(search_results, start=1):
        print_search_results(index, results, top_k=args.top_k)


if __name__ == "__main__":
    main()
