"""Verification engine policy tests."""

import pytest

from backend.ai.verification.engine import ThresholdVerificationEngine
from backend.app.domain.face import Face
from backend.app.domain.search import SearchResult, SearchResults
from backend.app.domain.verification import VerificationDecision


def test_unknown_when_no_candidates(unit_face: Face) -> None:
    """Empty search results should produce UNKNOWN."""
    engine = ThresholdVerificationEngine()
    result = engine.verify(
        unit_face,
        SearchResults(results=[], search_time_ms=0.0, provider="test"),
    )

    assert result.decision is VerificationDecision.UNKNOWN
    assert result.is_verified is False
    assert result.matched_identity_id is None
    assert result.metadata["policy"] == "threshold"
    assert result.metadata["engine_version"] == "1.0.0"


def test_accept_above_threshold(unit_face: Face) -> None:
    """Scores above threshold should produce ACCEPT."""
    engine = ThresholdVerificationEngine(similarity_threshold=0.45)
    result = engine.verify(
        unit_face,
        SearchResults(
            results=[
                SearchResult(identity_id="user_a", score=0.98, rank=1),
            ],
            search_time_ms=1.0,
            provider="test",
        ),
    )

    assert result.decision is VerificationDecision.ACCEPT
    assert result.is_verified is True
    assert result.matched_identity_id == "user_a"


def test_reject_below_threshold(unit_face: Face) -> None:
    """Scores below threshold should produce REJECT."""
    engine = ThresholdVerificationEngine(similarity_threshold=0.45)
    result = engine.verify(
        unit_face,
        SearchResults(
            results=[
                SearchResult(identity_id="user_b", score=0.20, rank=1),
            ],
            search_time_ms=1.0,
            provider="test",
        ),
    )

    assert result.decision is VerificationDecision.REJECT
    assert result.is_verified is False
    assert result.matched_identity_id == "user_b"


def test_duplicate_candidate_identity_rejected(unit_face: Face) -> None:
    """Duplicate candidate identities should fail validation."""
    engine = ThresholdVerificationEngine()
    with pytest.raises(ValueError, match="Duplicate search candidate identity_id"):
        engine.verify(
            unit_face,
            SearchResults(
                results=[
                    SearchResult(identity_id="dup", score=0.9, rank=1),
                    SearchResult(identity_id="dup", score=0.8, rank=2),
                ],
                search_time_ms=1.0,
                provider="test",
            ),
        )
