"""
Search index interface.

Concrete providers such as FAISS, Milvus, Qdrant, and pgvector must
implement this interface.

Index providers operate on embedding identifiers and vectors only.
Identity resolution belongs to ``IdentityRepository`` via ``FaceSearcher``.
"""

from abc import ABC, abstractmethod

import numpy as np

from backend.ai.search.types import RawSearchOutput


class SearchIndex(ABC):
    """Abstract vector search index used by the search engine.

    ``SearchIndex`` owns embedding vectors and raw nearest-neighbor search
    only. It must never store, return, or modify identity metadata.
    """

    @abstractmethod
    def load(self) -> None:
        """Load index assets required for search.

        Raises:
            FileNotFoundError: If required index assets are missing when expected.
            ValueError: If the index file is invalid.
        """

    @abstractmethod
    def add(self, embedding_id: int, vector: np.ndarray) -> None:
        """Add an embedding vector to the index.

        Args:
            embedding_id: Provider-level embedding identifier.
            vector: Unit-normalized embedding vector.

        Raises:
            ValueError: If the embedding identifier or vector is invalid.
        """

    @abstractmethod
    def remove(self, embedding_id: int) -> None:
        """Remove an embedding vector from the index.

        Args:
            embedding_id: Provider-level embedding identifier.

        Raises:
            ValueError: If the embedding identifier is invalid or not found.
        """

    @abstractmethod
    def update(self, embedding_id: int, vector: np.ndarray) -> None:
        """Replace an existing embedding vector in the index.

        Args:
            embedding_id: Provider-level embedding identifier.
            vector: Unit-normalized embedding vector.

        Raises:
            ValueError: If the embedding identifier or vector is invalid.
        """

    @abstractmethod
    def search(
        self,
        vector: np.ndarray,
        *,
        top_k: int = 5,
    ) -> RawSearchOutput:
        """Search the index for nearest embedding matches.

        Args:
            vector: Unit-normalized probe embedding vector.
            top_k: Maximum number of matches to return.

        Returns:
            Raw embedding identifiers and inner-product distances only.
            ``SearchIndex`` must never return identity metadata.

        Raises:
            ValueError: If the probe vector or ``top_k`` is invalid.
        """

    @abstractmethod
    def save(self) -> None:
        """Persist the index to configured storage."""

    @property
    @abstractmethod
    def count(self) -> int:
        """Return the number of embeddings stored in the index."""

    @property
    @abstractmethod
    def dimension(self) -> int | None:
        """Return the embedding dimension when known."""

    @abstractmethod
    def list_embedding_ids(self) -> list[int]:
        """Return all embedding identifiers stored in the index."""

    def rebuild(self) -> None:
        """Rebuild the index from stored vectors.

        Providers that support vector reconstruction should override this
        method. The default implementation is a no-op.
        """
