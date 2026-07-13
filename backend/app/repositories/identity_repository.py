"""
Identity repository interface.

Persistence and lookup of gallery identity metadata. Vector storage
belongs to ``SearchIndex`` implementations such as FAISS.
"""

from abc import ABC, abstractmethod
from typing import Any


class IdentityRepository(ABC):
    """Abstract identity mapping repository used by the search engine.

    ``IdentityRepository`` is the only owner of identity metadata and
    embedding-to-identity mappings. Vector storage belongs to ``SearchIndex``.
    """

    @abstractmethod
    def load(self) -> None:
        """Load identity mappings from configured storage."""

    @abstractmethod
    def save(self) -> None:
        """Persist identity mappings to configured storage."""

    @abstractmethod
    def add(
        self,
        identity_id: str,
        embedding_id: int,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Add an identity mapping.

        Args:
            identity_id: Unique gallery identity identifier.
            embedding_id: Provider-level embedding identifier.
            metadata: Optional identity metadata.

        Raises:
            ValueError: If the identity or embedding identifier is invalid.
        """

    @abstractmethod
    def update(
        self,
        identity_id: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Update identity metadata.

        Args:
            identity_id: Unique gallery identity identifier.
            metadata: Optional identity metadata to store.

        Raises:
            ValueError: If the identity is not enrolled.
        """

    @abstractmethod
    def remove(self, identity_id: str) -> int:
        """Remove an identity mapping.

        Args:
            identity_id: Unique gallery identity identifier.

        Returns:
            The embedding identifier that was removed.

        Raises:
            ValueError: If the identity is not enrolled.
        """

    @abstractmethod
    def lookup_identity(self, embedding_id: int) -> str | None:
        """Resolve an embedding identifier to an identity identifier."""

    @abstractmethod
    def lookup_embedding(self, identity_id: str) -> int | None:
        """Resolve an identity identifier to an embedding identifier."""

    @abstractmethod
    def lookup_metadata(self, identity_id: str) -> dict[str, Any] | None:
        """Return optional metadata for an enrolled identity."""

    @abstractmethod
    def allocate_embedding_id(self) -> int:
        """Allocate the next embedding identifier for enrollment."""

    @abstractmethod
    def count(self) -> int:
        """Return the number of enrolled identities."""

    @abstractmethod
    def list_identities(self) -> list[str]:
        """Return all enrolled identity identifiers."""

    @abstractmethod
    def list_embedding_ids(self) -> list[int]:
        """Return all embedding identifiers tracked by the repository."""
