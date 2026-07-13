"""
Identity gallery service.

Owns gallery lifecycle orchestration across the identity repository and
the vector search index.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from backend.ai.search.config import (
    get_search_index_path,
    get_search_mapping_path,
)
from backend.ai.search.index import SearchIndex
from backend.ai.search.searcher import create_identity_repository, create_search_index
from backend.ai.search.utils import validate_gallery_vector
from backend.app.repositories.identity_repository import IdentityRepository

logger = logging.getLogger(__name__)


class IdentityService:
    """Orchestrates gallery enrollment and persistence.

    ``IdentityService`` is the only owner of gallery lifecycle operations.
    It coordinates identity mappings in ``IdentityRepository`` with vector
    storage in ``SearchIndex``.
    """

    def __init__(
        self,
        repository: IdentityRepository | None = None,
        search_index: SearchIndex | None = None,
        *,
        index: SearchIndex | None = None,
    ) -> None:
        """Initialize the identity service.

        Args:
            repository: Identity repository for dependency injection.
            search_index: Search index for dependency injection.
            index: Deprecated alias for ``search_index``.
        """
        resolved_index = search_index if search_index is not None else index
        self._repository = create_identity_repository(repository)
        self._index = create_search_index(resolved_index)
        self._loaded = False
        self._initialized = False

    @property
    def gallery_size(self) -> int:
        """Return the number of enrolled gallery identities."""
        return self._repository.count()

    @property
    def index_vector_count(self) -> int:
        """Return the number of vectors stored in the search index."""
        self.load_gallery()
        return self._index.count

    def initialize(self) -> None:
        """Load gallery assets and validate repository/index consistency.

        This lifecycle hook is idempotent and must be invoked explicitly by
        the caller when production gallery readiness should be verified.
        """
        if self._initialized:
            return

        self._repository.load()
        self._index.load()
        self._validate_gallery_consistency()
        self._loaded = True
        self._initialized = True
        logger.info(
            "Initialized gallery: identities=%d index_count=%d",
            self.gallery_size,
            self._index.count,
        )

    def load_gallery(self) -> None:
        """Load the identity repository and search index."""
        if self._loaded:
            return

        self._repository.load()
        self._index.load()
        self._loaded = True
        logger.info(
            "Loaded gallery: identities=%d index_count=%d",
            self.gallery_size,
            self._index.count,
        )

    def save_gallery(self) -> None:
        """Persist the identity repository and search index."""
        self.load_gallery()
        self._index.save()
        self._repository.save()
        logger.info("Saved gallery to %s.", get_search_index_path())

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
            metadata: Optional identity metadata stored in ``IdentityRepository``.

        Returns:
            Assigned embedding identifier.
        """
        self.load_gallery()
        validate_gallery_vector(vector, expected_dimension=self._index.dimension)

        embedding_id = self._repository.allocate_embedding_id()
        self._repository.add(identity_id, embedding_id, metadata=metadata)
        self._index.add(embedding_id, vector)

        logger.debug("Enrolled identity %s as embedding_id=%d.", identity_id, embedding_id)
        return embedding_id

    def update(
        self,
        identity_id: str,
        vector: np.ndarray,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Update an enrolled gallery identity embedding."""
        self.load_gallery()

        embedding_id = self._repository.lookup_embedding(identity_id)
        if embedding_id is None:
            raise ValueError(f"identity_id {identity_id!r} is not enrolled.")

        validate_gallery_vector(vector, expected_dimension=self._index.dimension)
        self._index.update(embedding_id, vector)

        if metadata is not None:
            self._repository.update(identity_id, metadata=metadata)

        logger.debug("Updated identity %s (embedding_id=%d).", identity_id, embedding_id)

    def delete(self, identity_id: str) -> None:
        """Delete an enrolled gallery identity."""
        self.load_gallery()

        embedding_id = self._repository.remove(identity_id)
        self._index.remove(embedding_id)
        logger.debug("Deleted identity %s (embedding_id=%d).", identity_id, embedding_id)

    def list_identities(self) -> list[str]:
        """Return all enrolled identity identifiers."""
        self.load_gallery()
        return self._repository.list_identities()

    def get_identity_metadata(self, identity_id: str) -> dict[str, Any] | None:
        """Return optional metadata for an enrolled identity."""
        self.load_gallery()
        return self._repository.lookup_metadata(identity_id)

    def rebuild_gallery(self) -> None:
        """Rebuild the vector index from vectors currently stored in the index."""
        self.load_gallery()

        if self.gallery_size == 0:
            logger.info("Gallery rebuild skipped because no identities are enrolled.")
            return

        self._index.rebuild()
        logger.info("Rebuilt gallery index for %d identities.", self.gallery_size)

    def _validate_gallery_consistency(self) -> None:
        """Validate that repository mappings and index vectors are synchronized."""
        mapping_path = get_search_mapping_path()
        index_path = get_search_index_path()

        repository_count = self._repository.count()
        index_count = self._index.count
        mapping_exists = mapping_path.exists()
        index_exists = index_path.exists()

        if repository_count > 0 and not mapping_exists:
            raise ValueError(
                "Identity mappings are loaded but the mapping file is missing at "
                f"{mapping_path}."
            )

        if index_count > 0 and not index_exists:
            raise ValueError(
                "Search index vectors are loaded but the index file is missing at "
                f"{index_path}."
            )

        if mapping_exists and repository_count > 0 and not index_exists:
            raise ValueError(
                "Identity mapping exists but the search index file is missing at "
                f"{index_path}."
            )

        if index_exists and index_count > 0 and not mapping_exists:
            raise ValueError(
                "Search index exists but the identity mapping file is missing at "
                f"{mapping_path}."
            )

        repository_embedding_ids = set(self._repository.list_embedding_ids())
        index_embedding_ids = set(self._index.list_embedding_ids())

        if repository_embedding_ids != index_embedding_ids:
            missing_in_index = sorted(repository_embedding_ids - index_embedding_ids)
            missing_in_mapping = sorted(index_embedding_ids - repository_embedding_ids)
            raise ValueError(
                "Gallery mapping and search index are out of sync: "
                f"missing_in_index={missing_in_index} "
                f"missing_in_mapping={missing_in_mapping}"
            )

        if repository_count != index_count:
            raise ValueError(
                "Gallery mapping count does not match search index count: "
                f"mapping={repository_count} index={index_count}."
            )
