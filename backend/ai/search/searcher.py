"""
Face searcher orchestration.

Queries a configured ``SearchIndex`` using probe embeddings from ``Face``
objects, then resolves raw matches through ``IdentityRepository``.
"""

from __future__ import annotations

import logging
import time

import numpy as np

from backend.ai.search.config import load_search_model_settings
from backend.ai.search.index import SearchIndex
from backend.ai.search.types import RawSearchOutput
from backend.ai.search.utils import validate_gallery_vector, validate_search_input
from backend.app.domain.face import Face
from backend.app.domain.search import SearchResult, SearchResults
from backend.app.repositories.identity_repository import IdentityRepository
from backend.app.repositories.json_identity_repository import JsonIdentityRepository

logger = logging.getLogger(__name__)


def create_search_index(index: SearchIndex | None = None) -> SearchIndex:
    """Create the configured search index provider.

    Args:
        index: Optional pre-initialized search index for dependency injection.

    Returns:
        Search index implementation selected from configuration.

    Raises:
        ValueError: If the configured provider is unsupported.
    """
    if index is not None:
        return index

    settings = load_search_model_settings()
    if settings.provider == "faiss":
        from backend.ai.search.faiss.index import FaissSearchIndex

        return FaissSearchIndex()

    raise ValueError(f"Unsupported search provider: {settings.provider!r}.")


def create_identity_repository(
    repository: IdentityRepository | None = None,
) -> IdentityRepository:
    """Create the configured identity repository.

    Args:
        repository: Optional repository for dependency injection.

    Returns:
        Identity repository implementation.
    """
    if repository is not None:
        return repository

    return JsonIdentityRepository()


def create_search_engine_components(
    repository: IdentityRepository | None = None,
    search_index: SearchIndex | None = None,
    *,
    index: SearchIndex | None = None,
) -> tuple[IdentityRepository, SearchIndex]:
    """Create shared search engine components for dependency injection.

    Returns:
        A repository and search index pair that can be reused by
        ``IdentityService`` and ``FaceSearcher``.
    """
    resolved_index = search_index if search_index is not None else index
    return (
        create_identity_repository(repository),
        create_search_index(resolved_index),
    )


class FaceSearcher:
    """Orchestrates face similarity search against a gallery index.

    Consumes ``Face`` objects with populated embeddings, delegates raw vector
    search to ``SearchIndex``, and resolves identities through
    ``IdentityRepository``. Gallery lifecycle belongs to ``IdentityService``.

    ``FaceSearcher`` resolves identity metadata only after ``SearchIndex``
    returns ``RawSearchOutput``. It must never read metadata from the index.
    """

    def __init__(
        self,
        repository: IdentityRepository | None = None,
        search_index: SearchIndex | None = None,
        *,
        index: SearchIndex | None = None,
    ) -> None:
        """Initialize the face searcher.

        Args:
            repository: Identity repository for dependency injection.
            search_index: Search index for dependency injection.
            index: Deprecated alias for ``search_index``.
        """
        resolved_index = search_index if search_index is not None else index
        self._repository = create_identity_repository(repository)
        self._index = create_search_index(resolved_index)
        self._provider = load_search_model_settings().provider
        self._loaded = False

    def _ensure_ready(self) -> None:
        """Load index and repository assets required for search."""
        if self._loaded:
            return

        self._repository.load()
        self._index.load()
        self._loaded = True

    def search(
        self,
        faces: list[Face],
        *,
        top_k: int = 5,
        score_threshold: float | None = None,
    ) -> list[SearchResults]:
        """Search gallery identities for each probe face embedding.

        Args:
            faces: Faces with populated ``Face.embedding`` data.
            top_k: Maximum number of matches to return per face.
            score_threshold: Optional minimum inner-product score to keep.

        Returns:
            One ``SearchResults`` object per input face, in order.

        Raises:
            ValueError: If inputs are invalid or the gallery is empty.
        """
        validate_search_input(faces)

        if not faces:
            logger.debug("No faces to search.")
            return []

        if top_k < 1:
            raise ValueError(f"top_k must be at least 1, got {top_k}.")

        self._ensure_ready()

        if self._repository.count() == 0:
            raise ValueError("Cannot search because the gallery is empty.")

        all_results: list[SearchResults] = []

        for index, face in enumerate(faces):
            embedding = face.embedding
            if embedding is None or embedding.vector is None:
                raise ValueError(
                    f"Face at index {index} is missing embedding data after validation."
                )

            validate_gallery_vector(
                embedding.vector,
                expected_dimension=self._index.dimension,
            )

            start_time = time.perf_counter()
            raw_matches = self._index.search(embedding.vector, top_k=top_k)
            search_time_ms = (time.perf_counter() - start_time) * 1000.0

            matches = self._resolve_search_results(
                raw_matches,
                score_threshold=score_threshold,
            )

            all_results.append(
                SearchResults(
                    results=matches,
                    search_time_ms=search_time_ms,
                    provider=self._provider,
                )
            )

            logger.debug("Searched face %d.", index)

        logger.info("Searched %d face(s).", len(faces))
        return all_results

    def _resolve_search_results(
        self,
        raw_matches: RawSearchOutput,
        *,
        score_threshold: float | None,
    ) -> list[SearchResult]:
        """Convert raw index matches into domain ``SearchResult`` objects."""
        resolved: list[SearchResult] = []

        for embedding_id, distance in zip(raw_matches.indices, raw_matches.distances):
            score = float(distance)
            if score_threshold is not None and score < score_threshold:
                continue

            identity_id = self._repository.lookup_identity(int(embedding_id))
            if identity_id is None:
                logger.warning(
                    "Skipping unknown embedding_id %s with no identity mapping.",
                    embedding_id,
                )
                continue

            resolved.append(
                SearchResult(
                    identity_id=identity_id,
                    score=score,
                    rank=len(resolved) + 1,
                    metadata=self._repository.lookup_metadata(identity_id),
                )
            )

        return resolved


def search(
    faces: list[Face],
    *,
    repository: IdentityRepository | None = None,
    search_index: SearchIndex | None = None,
    index: SearchIndex | None = None,
    top_k: int = 5,
    score_threshold: float | None = None,
) -> list[SearchResults]:
    """Search gallery identities using the default face searcher.

    Args:
        faces: Faces with populated ``Face.embedding`` data.
        repository: Optional identity repository for dependency injection.
        search_index: Optional search index for dependency injection.
        index: Deprecated alias for ``search_index``.
        top_k: Maximum number of matches to return per face.
        score_threshold: Optional minimum inner-product score to keep.

    Returns:
        One ``SearchResults`` object per input face.
    """
    return FaceSearcher(
        repository=repository,
        search_index=search_index,
        index=index,
    ).search(
        faces,
        top_k=top_k,
        score_threshold=score_threshold,
    )
