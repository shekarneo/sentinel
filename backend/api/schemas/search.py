"""Search API schemas."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class SearchCandidateResponse(BaseModel):
    """Single search match candidate."""

    identity_id: str
    score: float
    rank: int = Field(ge=1)
    metadata: dict[str, Any] | None = None


class SearchRequest(BaseModel):
    """Search request contract."""

    top_k: int = Field(default=1, ge=1, description="Maximum candidates to return.")


class SearchResponse(BaseModel):
    """Search response contract.

    Internal ``SearchResults`` domain objects are never returned directly.
    """

    results: list[SearchCandidateResponse]
    search_time_ms: float
    provider: str
