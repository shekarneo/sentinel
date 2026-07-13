"""Search application service.

The API layer must call ``SearchService`` instead of ``FaceSearcher`` directly.
"""

from __future__ import annotations

from backend.ai.search.index import SearchIndex
from backend.ai.search.searcher import FaceSearcher, create_search_engine_components
from backend.app.domain.face import Face
from backend.app.domain.search import SearchResults
from backend.app.repositories.identity_repository import IdentityRepository


class SearchService:
    """Application service wrapper around the face search orchestrator."""

    def __init__(
        self,
        searcher: FaceSearcher | None = None,
        *,
        repository: IdentityRepository | None = None,
        search_index: SearchIndex | None = None,
    ) -> None:
        """Initialize the search service.

        Args:
            searcher: Optional pre-built searcher for dependency injection.
            repository: Optional identity repository shared with gallery services.
            search_index: Optional search index shared with gallery services.
        """
        if searcher is not None:
            self._searcher = searcher
            return

        resolved_repository, resolved_index = create_search_engine_components(
            repository,
            search_index,
        )
        self._searcher = FaceSearcher(
            repository=resolved_repository,
            search_index=resolved_index,
        )

    def search(self, faces: list[Face], *, top_k: int = 1) -> list[SearchResults]:
        """Search the gallery for probe faces.

        Args:
            faces: Probe faces with populated embeddings.
            top_k: Maximum number of candidates to return per face.

        Returns:
            One aggregated search result per input face, in order.
        """
        return self._searcher.search(faces, top_k=top_k)
