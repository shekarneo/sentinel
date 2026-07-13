"""
Embedding model interface.

Concrete providers such as ArcFace, AdaFace, MagFace, and ElasticFace must
implement this interface.
"""

from abc import ABC, abstractmethod

import numpy as np

from backend.app.domain.embedding import EmbeddingData


class EmbeddingModel(ABC):
    """Abstract embedding model used by the embedding engine."""

    @abstractmethod
    def load(self) -> None:
        """Load model assets required for inference.

        Raises:
            NotImplementedError: If the provider is not yet implemented.
        """

    def warmup(self) -> None:
        """Load model assets and perform optional provider warm-up.

        The default implementation ensures the model is loaded. Providers may
        override this method to run additional one-time initialization.
        """
        self.load()

    @abstractmethod
    def embed(self, aligned_face: np.ndarray) -> EmbeddingData:
        """Extract an embedding vector from an aligned face crop.

        Every provider must return a unit-length (L2-normalized) embedding.
        Normalization is the provider's responsibility; downstream modules
        must not re-normalize vectors stored in ``Face.embedding.vector``.

        Args:
            aligned_face: Aligned face image in HWC BGR uint8 layout.

        Returns:
            ``EmbeddingData`` containing a unit-length feature vector with
            ``normalized=True``.

        Raises:
            NotImplementedError: If the provider is not yet implemented.
            ValueError: If the aligned face image is invalid.
        """

    def embed_batch(self, aligned_faces: list[np.ndarray]) -> list[EmbeddingData]:
        """Extract embeddings for a batch of aligned face crops.

        Future providers may override this method to run batched inference.
        The default implementation is not provided.

        Args:
            aligned_faces: Aligned face images in HWC BGR uint8 layout.

        Returns:
            One ``EmbeddingData`` per input face, each unit-normalized.

        Raises:
            NotImplementedError: If the provider does not implement batching.
        """
        raise NotImplementedError(
            "Batch embedding is not implemented for this provider."
        )
