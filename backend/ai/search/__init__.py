"""
Search engine module.

Queries gallery indexes using unit-normalized face embeddings and returns
``SearchResults`` for downstream verification and decision stages.
"""

from backend.ai.search.index import SearchIndex
from backend.ai.search.searcher import (
    FaceSearcher,
    create_search_engine_components,
    create_search_index,
    search,
)
from backend.ai.search.types import RawSearchOutput

__all__ = [
    "FaceSearcher",
    "RawSearchOutput",
    "SearchIndex",
    "create_search_engine_components",
    "create_search_index",
    "search",
]
