"""
JSON-backed identity repository.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from backend.ai.search.config import get_search_mapping_path
from backend.app.repositories.identity_repository import IdentityRepository

logger = logging.getLogger(__name__)


class JsonIdentityRepository(IdentityRepository):
    """Persist identity mappings in a JSON file alongside the search index."""

    def __init__(self, mapping_path=None) -> None:
        """Initialize the JSON identity repository.

        Args:
            mapping_path: Optional override for the mapping file path.
        """
        self._mapping_path = mapping_path or get_search_mapping_path()
        self._embedding_to_identity: dict[int, str] = {}
        self._identity_to_embedding: dict[str, int] = {}
        self._identity_metadata: dict[str, dict[str, Any]] = {}
        self._next_embedding_id = 1
        self._loaded = False

    def load(self) -> None:
        """Load identity mappings from disk."""
        if self._loaded:
            return

        if not self._mapping_path.exists():
            self._embedding_to_identity = {}
            self._identity_to_embedding = {}
            self._identity_metadata = {}
            self._next_embedding_id = 1
            self._loaded = True
            return

        with self._mapping_path.open("r", encoding="utf-8") as file:
            payload = json.load(file)

        self._embedding_to_identity = {
            int(key): value
            for key, value in payload.get("embedding_to_identity", {}).items()
        }
        self._identity_to_embedding = {
            key: int(value)
            for key, value in payload.get("identity_to_embedding", {}).items()
        }
        self._identity_metadata = payload.get("identity_metadata", {})
        self._next_embedding_id = int(payload.get("next_embedding_id", 1))
        self._loaded = True
        logger.info(
            "Loaded identity repository from %s (identities=%d).",
            self._mapping_path,
            self.count(),
        )

    def save(self) -> None:
        """Persist identity mappings to disk."""
        self._mapping_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "embedding_to_identity": {
                str(key): value for key, value in self._embedding_to_identity.items()
            },
            "identity_to_embedding": self._identity_to_embedding,
            "identity_metadata": self._identity_metadata,
            "next_embedding_id": self._next_embedding_id,
        }

        with self._mapping_path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2, sort_keys=True)

        logger.info("Saved identity repository to %s.", self._mapping_path)

    def add(
        self,
        identity_id: str,
        embedding_id: int,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Add an identity mapping."""
        if not identity_id:
            raise ValueError("identity_id must be a non-empty string.")

        if embedding_id < 0:
            raise ValueError(f"embedding_id must be non-negative, got {embedding_id}.")

        if identity_id in self._identity_to_embedding:
            raise ValueError(f"identity_id {identity_id!r} is already enrolled.")

        if embedding_id in self._embedding_to_identity:
            raise ValueError(f"embedding_id {embedding_id} is already mapped.")

        self._embedding_to_identity[embedding_id] = identity_id
        self._identity_to_embedding[identity_id] = embedding_id
        if metadata is not None:
            self._identity_metadata[identity_id] = metadata

        self._next_embedding_id = max(self._next_embedding_id, embedding_id + 1)

    def update(
        self,
        identity_id: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Update identity metadata."""
        if identity_id not in self._identity_to_embedding:
            raise ValueError(f"identity_id {identity_id!r} is not enrolled.")

        if metadata is not None:
            self._identity_metadata[identity_id] = metadata

    def remove(self, identity_id: str) -> int:
        """Remove an identity mapping."""
        embedding_id = self._identity_to_embedding.pop(identity_id, None)
        if embedding_id is None:
            raise ValueError(f"identity_id {identity_id!r} is not enrolled.")

        self._embedding_to_identity.pop(embedding_id, None)
        self._identity_metadata.pop(identity_id, None)
        return embedding_id

    def lookup_identity(self, embedding_id: int) -> str | None:
        """Resolve an embedding identifier to an identity identifier."""
        return self._embedding_to_identity.get(embedding_id)

    def lookup_embedding(self, identity_id: str) -> int | None:
        """Resolve an identity identifier to an embedding identifier."""
        return self._identity_to_embedding.get(identity_id)

    def lookup_metadata(self, identity_id: str) -> dict[str, Any] | None:
        """Return optional metadata for an enrolled identity."""
        return self._identity_metadata.get(identity_id)

    def allocate_embedding_id(self) -> int:
        """Allocate the next embedding identifier for enrollment."""
        embedding_id = self._next_embedding_id
        self._next_embedding_id += 1
        return embedding_id

    def count(self) -> int:
        """Return the number of enrolled identities."""
        return len(self._identity_to_embedding)

    def list_identities(self) -> list[str]:
        """Return all enrolled identity identifiers."""
        return list(self._identity_to_embedding.keys())

    def list_embedding_ids(self) -> list[int]:
        """Return all embedding identifiers tracked by the repository."""
        return sorted(self._embedding_to_identity.keys())
