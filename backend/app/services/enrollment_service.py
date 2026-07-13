"""Enrollment application service.

Enrollment operations are delegated to ``IdentityService``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from backend.app.services.identity_service import IdentityService


class EnrollmentService:
    """Application service for gallery enrollment operations."""

    def __init__(self, identity_service: IdentityService) -> None:
        """Initialize the enrollment service.

        Args:
            identity_service: Shared identity gallery service.
        """
        self._identity = identity_service

    def enroll(
        self,
        identity_id: str,
        vector: np.ndarray,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """Enroll a gallery identity embedding.

        Args:
            identity_id: Unique gallery identity identifier.
            vector: Unit-normalized embedding vector.
            metadata: Optional identity metadata.

        Returns:
            Assigned embedding identifier.
        """
        return self._identity.enroll(identity_id, vector, metadata=metadata)

    def update(
        self,
        identity_id: str,
        vector: np.ndarray,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Update an enrolled gallery identity."""
        self._identity.update(identity_id, vector, metadata=metadata)

    def delete(self, identity_id: str) -> None:
        """Delete an enrolled gallery identity."""
        self._identity.delete(identity_id)

    def enroll_and_persist(
        self,
        identity_id: str,
        vector: np.ndarray,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """Enroll an identity and persist gallery assets to disk."""
        embedding_id = self.enroll(identity_id, vector, metadata=metadata)
        self._identity.save_gallery()
        return embedding_id

    def delete_and_persist(self, identity_id: str) -> None:
        """Delete an identity and persist gallery assets to disk."""
        self.delete(identity_id)
        self._identity.save_gallery()
