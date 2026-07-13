"""Gallery application service.

Gallery operations are delegated to ``IdentityService``, which owns gallery
lifecycle orchestration.
"""

from __future__ import annotations

from typing import Any

from backend.app.services.identity_service import IdentityService


class GalleryService:
    """Read and maintenance operations for the identity gallery."""

    def __init__(self, identity_service: IdentityService) -> None:
        """Initialize the gallery service.

        Args:
            identity_service: Shared identity gallery service.
        """
        self._identity = identity_service

    @property
    def gallery_size(self) -> int:
        """Return the number of enrolled identities."""
        self._identity.load_gallery()
        return self._identity.gallery_size

    def list_identities(self) -> list[str]:
        """Return enrolled identity identifiers."""
        self._identity.load_gallery()
        return self._identity.list_identities()

    def get_identity_metadata(self, identity_id: str) -> dict[str, Any] | None:
        """Return metadata for an enrolled identity."""
        self._identity.load_gallery()
        return self._identity.get_identity_metadata(identity_id)

    def list_entries(self) -> list[dict[str, Any]]:
        """Return gallery entries with metadata for console display."""
        self._identity.load_gallery()
        entries: list[dict[str, Any]] = []
        for identity_id in self._identity.list_identities():
            entries.append(
                {
                    "identity_id": identity_id,
                    "metadata": self._identity.get_identity_metadata(identity_id),
                    "enrollment_count": 1,
                }
            )
        return entries

    def get_index_stats(self) -> dict[str, int | str]:
        """Return search index statistics."""
        self._identity.load_gallery()
        vector_count = self._identity.index_vector_count
        identity_count = self.gallery_size
        if identity_count == 0:
            status = "empty"
        elif vector_count == identity_count:
            status = "ready"
        else:
            status = "sync_required"
        return {
            "status": status,
            "vector_count": vector_count,
            "identity_count": identity_count,
        }

    def rebuild(self) -> int:
        """Rebuild the vector index from enrolled gallery vectors."""
        self._identity.rebuild_gallery()
        return self.gallery_size
