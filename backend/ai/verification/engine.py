"""
Verification engine interface.

Concrete providers such as threshold, adaptive, and multi-template engines
must implement this interface.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

from backend.ai.verification.config import load_verification_thresholds
from backend.ai.verification.utils import (
    resolve_is_verified,
    validate_face_embedding,
    validate_search_results,
    validate_threshold,
)
from backend.app.domain.face import Face
from backend.app.domain.search import SearchResult, SearchResults
from backend.app.domain.verification import VerificationDecision, VerificationResult

logger = logging.getLogger(__name__)

THRESHOLD_POLICY = "threshold"
ENGINE_VERSION = "1.0.0"


class VerificationEngine(ABC):
    """Abstract verification engine used by the verification module.

    ``VerificationEngine`` owns verification policy and decision logic only.
    It must not depend on search providers, embedding providers, or gallery
    storage implementations.
    """

    @abstractmethod
    def verify(
        self,
        face: Face,
        search_results: SearchResults,
    ) -> VerificationResult:
        """Verify a probe face against search candidates.

        Args:
            face: Probe face with populated embedding data.
            search_results: Ranked gallery matches for the probe face.

        Returns:
            Verification decision and supporting scores.

        Raises:
            ValueError: If inputs are invalid.
        """


class ThresholdVerificationEngine(VerificationEngine):
    """Threshold-based verification engine.

    Evaluates an ordered candidate list and applies a fixed similarity
    threshold to the best-ranked candidate.
    """

    def __init__(self, *, similarity_threshold: float | None = None) -> None:
        """Initialize the threshold verification engine.

        Args:
            similarity_threshold: Optional threshold override for dependency
                injection or testing. When omitted, the value is loaded from
                configuration.
        """
        self._similarity_threshold = similarity_threshold

    def verify(
        self,
        face: Face,
        search_results: SearchResults,
    ) -> VerificationResult:
        """Verify a probe face using a fixed similarity threshold.

        Args:
            face: Probe face with populated embedding data.
            search_results: Ranked gallery matches for the probe face.

        Returns:
            Verification decision and supporting scores.

        Raises:
            ValueError: If inputs are invalid.
        """
        validate_face_embedding(face)
        validate_search_results(search_results)

        threshold = self._resolve_threshold()
        ordered_candidates = self._order_candidates(search_results.results)

        return self._evaluate_threshold_policy(
            ordered_candidates,
            threshold=threshold,
            search_results=search_results,
        )

    def _resolve_threshold(self) -> float:
        """Return the configured similarity threshold."""
        threshold = (
            self._similarity_threshold
            if self._similarity_threshold is not None
            else load_verification_thresholds().similarity_threshold
        )
        validate_threshold(threshold)
        return float(threshold)

    @staticmethod
    def _order_candidates(candidates: list[SearchResult]) -> list[SearchResult]:
        """Return search candidates ordered from best to worst rank."""
        return sorted(candidates, key=lambda candidate: candidate.rank)

    def _evaluate_threshold_policy(
        self,
        ordered_candidates: list[SearchResult],
        *,
        threshold: float,
        search_results: SearchResults,
    ) -> VerificationResult:
        """Evaluate ordered candidates using the threshold verification policy.

        The threshold policy inspects only the best-ranked candidate. Future
        policies may evaluate multiple candidates from the same ordered list.
        """
        if not ordered_candidates:
            logger.debug("No search candidates available; returning UNKNOWN.")
            return self._build_result(
                decision=VerificationDecision.UNKNOWN,
                matched_identity_id=None,
                similarity_score=0.0,
                threshold=threshold,
                search_results=search_results,
                evaluated_candidate=None,
                candidate_count=0,
            )

        best_match = ordered_candidates[0]
        similarity_score = float(best_match.score)
        decision = (
            VerificationDecision.ACCEPT
            if similarity_score >= threshold
            else VerificationDecision.REJECT
        )

        logger.debug(
            "Threshold verification decision=%s identity=%s score=%.6f threshold=%.6f.",
            decision.value,
            best_match.identity_id,
            similarity_score,
            threshold,
        )

        return self._build_result(
            decision=decision,
            matched_identity_id=best_match.identity_id,
            similarity_score=similarity_score,
            threshold=threshold,
            search_results=search_results,
            evaluated_candidate=best_match,
            candidate_count=len(ordered_candidates),
        )

    def _build_result(
        self,
        *,
        decision: VerificationDecision,
        matched_identity_id: str | None,
        similarity_score: float,
        threshold: float,
        search_results: SearchResults,
        evaluated_candidate: SearchResult | None,
        candidate_count: int,
    ) -> VerificationResult:
        """Build a ``VerificationResult`` with enforced decision invariants."""
        return VerificationResult(
            decision=decision,
            matched_identity_id=matched_identity_id,
            similarity_score=similarity_score,
            threshold=threshold,
            is_verified=resolve_is_verified(decision),
            verification_time_ms=0.0,
            metadata=self._build_metadata(
                search_results,
                evaluated_candidate=evaluated_candidate,
                candidate_count=candidate_count,
            ),
        )

    @staticmethod
    def _build_metadata(
        search_results: SearchResults,
        *,
        evaluated_candidate: SearchResult | None,
        candidate_count: int,
    ) -> dict[str, Any]:
        """Build optional verification metadata for downstream stages."""
        metadata: dict[str, Any] = {
            "policy": THRESHOLD_POLICY,
            "engine_version": ENGINE_VERSION,
            "candidate_count": candidate_count,
            "search_provider": search_results.provider,
            "search_time_ms": search_results.search_time_ms,
        }

        if evaluated_candidate is not None:
            metadata["rank"] = evaluated_candidate.rank
            if evaluated_candidate.metadata is not None:
                metadata["candidate_metadata"] = evaluated_candidate.metadata

        return metadata
