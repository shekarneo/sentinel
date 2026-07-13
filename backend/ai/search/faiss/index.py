"""
FAISS search index provider.

Uses ``IndexFlatIP`` for inner-product search on unit-normalized embeddings.
For unit vectors, inner product is equivalent to cosine similarity.

Identity metadata is not stored in FAISS. ``IdentityRepository`` and
``IdentityService`` own gallery identity mappings.
"""

from __future__ import annotations

import logging
import threading

import faiss
import numpy as np

from backend.ai.search.config import get_search_index_path, load_search_model_settings
from backend.ai.search.index import SearchIndex
from backend.ai.search.types import RawSearchOutput
from backend.ai.search.utils import validate_gallery_vector, validate_probe_vector

logger = logging.getLogger(__name__)

_DEFAULT_DIMENSION = 512


class FaissSearchIndex(SearchIndex):
    """FAISS-backed vector index using ``IndexFlatIP``.

    Embeddings are assumed to be L2-normalized before indexing and search.
    This provider must not re-normalize vectors.

    ``IndexIDMap2`` wraps the flat inner-product index so embeddings can be
    addressed by stable integer identifiers for add, remove, and update.
    """

    def __init__(self) -> None:
        """Initialize FAISS provider settings from configuration."""
        settings = load_search_model_settings()
        self._provider = settings.provider
        self._index_path = get_search_index_path()
        self._metric = settings.metric
        self._index: faiss.IndexIDMap2 | None = None
        self._dimension: int | None = None
        self._known_ids: set[int] = set()
        self._load_lock = threading.Lock()

        if self._metric != "cosine":
            raise ValueError(
                "FaissSearchIndex only supports cosine search via IndexFlatIP, "
                f"got metric {self._metric!r}."
            )

    @property
    def count(self) -> int:
        """Return the number of embeddings stored in the index."""
        if self._index is None:
            return 0
        return int(self._index.ntotal)

    @property
    def dimension(self) -> int | None:
        """Return the embedding dimension when known."""
        return self._dimension

    def list_embedding_ids(self) -> list[int]:
        """Return all embedding identifiers stored in the FAISS index."""
        self._ensure_loaded()
        return sorted(self._known_ids)

    def load(self) -> None:
        """Load the FAISS index from disk or create an empty index.

        Raises:
            ValueError: If the configured index file is invalid.
        """
        if self._index is not None:
            return

        with self._load_lock:
            if self._index is not None:
                return

            if self._index_path.exists():
                self._index = faiss.read_index(str(self._index_path))
                if not isinstance(self._index, faiss.IndexIDMap2):
                    raise ValueError(
                        "Expected a FAISS IndexIDMap2 index file at "
                        f"{self._index_path}."
                    )
                self._dimension = int(self._index.d)
                self._known_ids = self._rebuild_known_ids()
                logger.info(
                    "Loaded FAISS index from %s: count=%d dimension=%d",
                    self._index_path,
                    self.count,
                    self._dimension,
                )
                return

            self._dimension = _DEFAULT_DIMENSION
            self._index = self._create_index(self._dimension)
            logger.info(
                "Created empty FAISS index: dimension=%d path=%s",
                self._dimension,
                self._index_path,
            )

    def save(self) -> None:
        """Persist the FAISS index to disk."""
        if self._index is None:
            raise RuntimeError("FAISS index has not been loaded.")

        self._index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(self._index_path))
        logger.info("Saved FAISS index to %s (count=%d).", self._index_path, self.count)

    def add(self, embedding_id: int, vector: np.ndarray) -> None:
        """Add an embedding vector to the FAISS index.

        Args:
            embedding_id: Stable embedding identifier.
            vector: Unit-normalized embedding vector.

        Raises:
            ValueError: If the identifier or vector is invalid.
        """
        self._ensure_loaded()
        assert self._index is not None

        if embedding_id < 0:
            raise ValueError(f"embedding_id must be non-negative, got {embedding_id}.")

        prepared_vector = self._prepare_vector(vector)
        validate_gallery_vector(prepared_vector, expected_dimension=self._dimension)

        if embedding_id in self._known_ids:
            raise ValueError(f"embedding_id {embedding_id} already exists in the index.")

        vectors = prepared_vector.reshape(1, -1).astype(np.float32)
        ids = np.array([embedding_id], dtype=np.int64)
        self._index.add_with_ids(vectors, ids)
        self._known_ids.add(embedding_id)
        logger.debug("Added embedding_id=%d to FAISS index.", embedding_id)

    def remove(self, embedding_id: int) -> None:
        """Remove an embedding vector from the FAISS index.

        Args:
            embedding_id: Stable embedding identifier.

        Raises:
            ValueError: If the identifier is invalid or not found.
        """
        self._ensure_loaded()
        assert self._index is not None

        if embedding_id < 0:
            raise ValueError(f"embedding_id must be non-negative, got {embedding_id}.")

        ids = np.array([embedding_id], dtype=np.int64)
        if embedding_id not in self._known_ids:
            raise ValueError(f"embedding_id {embedding_id} was not found in the index.")

        removed = self._index.remove_ids(ids)
        if removed != 1:
            raise ValueError(f"Failed to remove embedding_id {embedding_id} from the index.")

        self._known_ids.remove(embedding_id)

        logger.debug("Removed embedding_id=%d from FAISS index.", embedding_id)

    def update(self, embedding_id: int, vector: np.ndarray) -> None:
        """Replace an existing embedding vector in the FAISS index."""
        self.remove(embedding_id)
        self.add(embedding_id, vector)

    def search(
        self,
        vector: np.ndarray,
        *,
        top_k: int = 5,
    ) -> RawSearchOutput:
        """Search the FAISS index for nearest embedding matches.

        Returns raw embedding identifiers and inner-product distances only.
        Inner product equals cosine similarity for unit-normalized vectors.

        Args:
            vector: Unit-normalized probe embedding vector.
            top_k: Maximum number of matches to return.

        Returns:
            Raw FAISS indices and distances without identity resolution.

        Raises:
            ValueError: If the probe vector, index state, or ``top_k`` is invalid.
        """
        self._ensure_loaded()
        assert self._index is not None

        if top_k < 1:
            raise ValueError(f"top_k must be at least 1, got {top_k}.")

        if self.count == 0:
            raise ValueError("Cannot search an empty FAISS index.")

        prepared_vector = self._prepare_vector(vector)
        validate_probe_vector(prepared_vector, expected_dimension=self._dimension)

        effective_top_k = min(top_k, self.count)
        vectors = prepared_vector.reshape(1, -1).astype(np.float32)
        distances, indices = self._index.search(vectors, effective_top_k)

        valid_mask = indices[0] >= 0
        return RawSearchOutput(
            indices=indices[0][valid_mask].astype(np.int64),
            distances=distances[0][valid_mask].astype(np.float32),
        )

    def _ensure_loaded(self) -> None:
        if self._index is None:
            self.load()

    @staticmethod
    def _create_index(dimension: int) -> faiss.IndexIDMap2:
        """Create a new FAISS index backed by ``IndexFlatIP``."""
        if dimension < 1:
            raise ValueError(f"Embedding dimension must be positive, got {dimension}.")

        base_index = faiss.IndexFlatIP(dimension)
        return faiss.IndexIDMap2(base_index)

    def _prepare_vector(self, vector: np.ndarray) -> np.ndarray:
        """Return a contiguous 1-D float vector."""
        prepared = np.asarray(vector, dtype=np.float32).reshape(-1)
        if self._dimension is None:
            self._dimension = int(prepared.size)
            if self._index is not None and self._index.ntotal == 0 and self._index.d != self._dimension:
                self._index = self._create_index(self._dimension)
        return prepared

    def _rebuild_known_ids(self) -> set[int]:
        """Rebuild the local embedding-id registry from a loaded FAISS index."""
        assert self._index is not None

        if self.count == 0:
            return set()

        batch = faiss.vector_to_array(self._index.id_map)
        if batch.size == 0:
            return set()
        return {int(value) for value in batch.tolist()}

    def rebuild(self) -> None:
        """Rebuild the FAISS index from vectors currently stored in memory."""
        self._ensure_loaded()
        assert self._index is not None

        snapshot = [
            (embedding_id, self._reconstruct(embedding_id))
            for embedding_id in sorted(self._known_ids)
        ]

        if not snapshot:
            if self._dimension is None:
                self._dimension = _DEFAULT_DIMENSION
            self._index = self._create_index(self._dimension)
            self._known_ids = set()
            logger.info("Rebuilt empty FAISS index.")
            return

        dimension = int(snapshot[0][1].size)
        self._dimension = dimension
        self._index = self._create_index(dimension)
        self._known_ids = set()

        for embedding_id, vector in snapshot:
            self.add(embedding_id, vector)

        logger.info("Rebuilt FAISS index with %d embeddings.", len(snapshot))

    def _reconstruct(self, embedding_id: int) -> np.ndarray:
        """Reconstruct a stored embedding vector by identifier.

        Args:
            embedding_id: Provider-level embedding identifier.

        Returns:
            Reconstructed embedding vector.

        Raises:
            ValueError: If the embedding identifier is not found.
        """
        self._ensure_loaded()
        assert self._index is not None

        if embedding_id not in self._known_ids:
            raise ValueError(f"embedding_id {embedding_id} was not found in the index.")

        vector = self._index.reconstruct(int(embedding_id))
        return np.asarray(vector, dtype=np.float32).reshape(-1)
