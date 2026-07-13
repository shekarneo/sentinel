"""Verification application service.

The API layer must call ``VerificationService`` instead of ``FaceVerifier``
directly.
"""

from __future__ import annotations

from backend.ai.verification.verifier import FaceVerifier
from backend.app.domain.face import Face
from backend.app.domain.search import SearchResults
from backend.app.domain.verification import VerificationResult


class VerificationService:
    """Application service wrapper around the face verification orchestrator."""

    def __init__(self, verifier: FaceVerifier | None = None) -> None:
        """Initialize the verification service.

        Args:
            verifier: Optional pre-built verifier for dependency injection.
        """
        self._verifier = verifier or FaceVerifier()

    def verify(
        self,
        faces: list[Face],
        search_results: list[SearchResults],
    ) -> list[VerificationResult]:
        """Verify probe faces against search candidates.

        Args:
            faces: Probe faces with populated embeddings.
            search_results: Search outputs aligned with ``faces``.

        Returns:
            Verification results for each probe face.
        """
        return self._verifier.verify(faces, search_results)
