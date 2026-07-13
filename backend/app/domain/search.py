"""
Search stage domain models.

Stores outputs produced by the Search Engine. Search results are kept
separate from ``Face``; the pipeline attaches them to
``PipelineContext.metadata``.
"""

from typing import Any

from pydantic import BaseModel, Field


class SearchResult(BaseModel):
    """A single gallery match candidate.

    Represents one identity match returned by a similarity search after
    ``IdentityRepository`` resolves a raw ``embedding_id`` from
    ``SearchIndex``. ``SearchIndex`` never knows identity identifiers.

    This model must not include probe embeddings, gallery vectors, or
    pipeline state.
    """

    identity_id: str = Field(
        description=(
            "Gallery identity resolved by IdentityRepository after raw search."
        ),
    )
    score: float
    rank: int = Field(ge=1)
    metadata: dict[str, Any] | None = None


class SearchResults(BaseModel):
    """Aggregated search output for one probe embedding query.

    Attached to ``PipelineContext.metadata['search_results']`` rather than
    stored on ``Face``.
    """

    results: list[SearchResult]
    search_time_ms: float
    provider: str
