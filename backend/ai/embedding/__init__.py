"""
Embedding engine module.

Extracts biometric feature vectors from aligned faces and populates
``Face.embedding`` with ``EmbeddingData`` for downstream search and
verification stages.
"""

from backend.ai.embedding.embedder import FaceEmbedder, embed
from backend.ai.embedding.model import EmbeddingModel

__all__ = [
    "EmbeddingModel",
    "FaceEmbedder",
    "embed",
]
