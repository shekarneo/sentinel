"""Run SCRFD detection, alignment, embedding, search, and verification."""

from pathlib import Path

import cv2
import numpy as np

import common
from common import load_image
from backend.ai.alignment.aligner import FaceAligner
from backend.ai.detection.scrfd.detector import SCRFDDetector
from backend.ai.embedding.embedder import FaceEmbedder
from backend.ai.search.config import get_search_index_path, get_search_mapping_path
from backend.ai.search.searcher import FaceSearcher, create_search_engine_components
from backend.ai.verification.config import load_verification_thresholds
from backend.ai.verification.engine import ThresholdVerificationEngine
from backend.ai.verification.verifier import FaceVerifier
from backend.app.domain.face import Face
from backend.app.domain.search import SearchResult, SearchResults
from backend.app.domain.verification import VerificationDecision, VerificationResult
from backend.app.services.identity_service import IdentityService




def print_verification_result(
    face_index: int,
    result: VerificationResult,
) -> None:
    """Print verification output for a probe face."""
    identity = result.matched_identity_id or "(none)"
    print(f"Probe Face Index: {face_index}")
    print(f"Identity: {identity}")
    print(f"Similarity: {result.similarity_score:.6f}")
    print(f"Threshold: {result.threshold:.6f}")
    print(f"Decision: {result.decision.value.upper()}")
    print(f"Verified: {result.is_verified}")
    print(f"Verification Time: {result.verification_time_ms:.3f} ms")
    print()


def assert_decision(
    result: VerificationResult,
    expected: VerificationDecision,
    *,
    label: str,
) -> None:
    """Assert that a verification result matches the expected decision."""
    if result.decision is not expected:
        raise AssertionError(
            f"{label} expected decision {expected.value}, got {result.decision.value}."
        )


def run_scenario_tests(face: Face) -> None:
    """Validate threshold verification behavior for controlled scenarios."""
    threshold = load_verification_thresholds().similarity_threshold
    engine = ThresholdVerificationEngine()

    empty_results = SearchResults(results=[], search_time_ms=0.0, provider="test")
    unknown_result = engine.verify(face, empty_results)
    assert_decision(unknown_result, VerificationDecision.UNKNOWN, label="No match")
    if unknown_result.matched_identity_id is not None:
        raise AssertionError("No-match scenario must not populate matched_identity_id.")
    print("Scenario: no match")
    print_verification_result(0, unknown_result)

    rejected_results = SearchResults(
        results=[
            SearchResult(
                identity_id="gallery_face_2",
                score=0.20,
                rank=1,
            )
        ],
        search_time_ms=1.0,
        provider="test",
    )
    rejected_result = engine.verify(face, rejected_results)
    assert_decision(rejected_result, VerificationDecision.REJECT, label="Rejected match")
    if rejected_result.similarity_score >= threshold:
        raise AssertionError("Rejected-match scenario must be below threshold.")
    print("Scenario: rejected match")
    print_verification_result(0, rejected_result)

    accepted_results = SearchResults(
        results=[
            SearchResult(
                identity_id="gallery_face_1",
                score=0.98,
                rank=1,
            )
        ],
        search_time_ms=1.0,
        provider="test",
    )
    accepted_result = engine.verify(face, accepted_results)
    assert_decision(accepted_result, VerificationDecision.ACCEPT, label="Accepted match")
    if not accepted_result.is_verified:
        raise AssertionError("Accepted-match scenario must set is_verified=True.")
    print("Scenario: accepted match")
    print_verification_result(0, accepted_result)

    print("Scenario: empty gallery")
    print(
        "Empty gallery prevents FaceSearcher from running; verification handles "
        "zero candidates as UNKNOWN when search_results.results is empty."
    )
    print()


def main() -> None:
    """Run detection, embedding, gallery enrollment, search, and verification."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run SCRFD, alignment, embedding, search, and verification."
    )
    parser.add_argument("image", type=Path, help="Path to input image")
    parser.add_argument("--top-k", type=int, default=5, help="Top-K search matches")
    parser.add_argument(
        "--reset-gallery",
        action="store_true",
        help="Delete existing gallery index and mapping before enrollment",
    )
    parser.add_argument(
        "--skip-scenarios",
        action="store_true",
        help="Skip controlled scenario tests and run only the full pipeline",
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
    verifier = FaceVerifier()

    faces = detector.detect(image)
    if not faces:
        raise ValueError("No faces detected in the input image.")

    aligner.align(image, faces)
    embedder.embed(faces)

    if not args.skip_scenarios:
        print("=== Controlled Verification Scenarios ===")
        run_scenario_tests(faces[0])
        print("=== Full Pipeline Verification ===")

    identity_ids = [f"gallery_face_{index}" for index in range(1, len(faces) + 1)]
    for face, identity_id in zip(faces, identity_ids):
        assert face.embedding is not None
        assert face.embedding.vector is not None
        identity_service.enroll(identity_id, face.embedding.vector)

    identity_service.save_gallery()

    print(f"Detected Faces: {len(faces)}")
    print(f"Gallery Size: {identity_service.gallery_size}")
    print(f"Verification Threshold: {load_verification_thresholds().similarity_threshold:.6f}")
    print()

    search_results = searcher.search(faces, top_k=args.top_k)
    verification_results = verifier.verify(faces, search_results)

    for index, result in enumerate(verification_results, start=1):
        print_verification_result(index, result)


if __name__ == "__main__":
    main()
