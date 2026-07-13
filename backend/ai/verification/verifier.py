"""
Face verifier orchestration.

Consumes probe faces and search outputs, delegates verification policy to a
configured ``VerificationEngine``, and returns ``VerificationResult`` objects.
"""

from __future__ import annotations

import logging
import time

from backend.ai.verification.engine import ThresholdVerificationEngine, VerificationEngine
from backend.ai.verification.utils import validate_verification_input
from backend.app.domain.face import Face
from backend.app.domain.search import SearchResults
from backend.app.domain.verification import VerificationResult

logger = logging.getLogger(__name__)


def create_verification_engine(
    engine: VerificationEngine | None = None,
) -> VerificationEngine:
    """Create the configured verification engine provider.

    Args:
        engine: Optional pre-initialized verification engine for dependency
            injection.

    Returns:
        Verification engine implementation.

    Raises:
        ValueError: If the configured provider is unsupported.
    """
    if engine is not None:
        return engine

    return ThresholdVerificationEngine()


class FaceVerifier:
    """Orchestrates face verification against search candidates.

    Consumes ``Face`` objects with populated embeddings and corresponding
    ``SearchResults``, then delegates decision policy to ``VerificationEngine``.
    The verifier is provider-agnostic and must not depend on search or
    embedding implementations.
    """

    def __init__(self, engine: VerificationEngine | None = None) -> None:
        """Initialize the face verifier.

        Args:
            engine: Optional verification engine for dependency injection.
        """
        self._engine = create_verification_engine(engine)

    def verify(
        self,
        faces: list[Face],
        search_results: list[SearchResults],
    ) -> list[VerificationResult]:
        """Verify probe faces against search candidates.

        Args:
            faces: Faces with populated ``Face.embedding`` data.
            search_results: Search outputs aligned with ``faces``.

        Returns:
            One ``VerificationResult`` per input face, in order.

        Raises:
            ValueError: If inputs are invalid or misaligned.
            NotImplementedError: If the configured engine is not yet implemented.
        """
        validate_verification_input(faces, search_results)

        if not faces:
            logger.debug("No faces to verify.")
            return []

        all_results: list[VerificationResult] = []

        for index, (face, results) in enumerate(zip(faces, search_results)):
            start_time = time.perf_counter()
            result = self._engine.verify(face, results)
            verification_time_ms = (time.perf_counter() - start_time) * 1000.0

            if result.verification_time_ms == 0.0:
                result = result.model_copy(
                    update={"verification_time_ms": verification_time_ms}
                )

            all_results.append(result)
            logger.debug("Verified face %d.", index)

        logger.info("Verified %d face(s).", len(faces))
        return all_results


def verify(
    faces: list[Face],
    search_results: list[SearchResults],
    *,
    engine: VerificationEngine | None = None,
) -> list[VerificationResult]:
    """Verify probe faces using the default face verifier.

    Args:
        faces: Faces with populated ``Face.embedding`` data.
        search_results: Search outputs aligned with ``faces``.
        engine: Optional verification engine for dependency injection.

    Returns:
        One ``VerificationResult`` per input face, in order.
    """
    return FaceVerifier(engine=engine).verify(faces, search_results)
