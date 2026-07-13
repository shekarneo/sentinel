"""
Face embedder orchestration.

Enriches existing ``Face`` objects with embedding data. The embedder does
not create separate domain objects such as ``EmbeddingResult``.
"""

import logging

from backend.ai.embedding.config import load_embedding_model_settings
from backend.ai.embedding.model import EmbeddingModel
from backend.ai.embedding.utils import validate_embedding_input
from backend.app.domain.face import Face

logger = logging.getLogger(__name__)


def create_embedding_model(model: EmbeddingModel | None = None) -> EmbeddingModel:
    """Create the configured embedding model provider.

    Args:
        model: Optional pre-initialized embedding model for dependency injection.

    Returns:
        Embedding model implementation selected from configuration.

    Raises:
        ValueError: If the configured provider is unsupported.
    """
    if model is not None:
        return model

    settings = load_embedding_model_settings()
    if settings.provider == "arcface":
        from backend.ai.embedding.arcface.model import ArcFaceEmbeddingModel

        return ArcFaceEmbeddingModel()

    raise ValueError(f"Unsupported embedding provider: {settings.provider!r}.")


class FaceEmbedder:
    """Orchestrates the face embedding pipeline.

    Consumes aligned ``Face`` objects and enriches each face with
    ``EmbeddingData``. The embedder is provider-agnostic and delegates
    inference to a configured ``EmbeddingModel`` implementation.
    """

    def __init__(self, model: EmbeddingModel | None = None) -> None:
        """Initialize the face embedder.

        Args:
            model: Optional embedding model for dependency injection.
        """
        self._model = create_embedding_model(model)
        self._warmed_up = False

    def warmup(self) -> None:
        """Ensure the configured embedding provider is loaded and ready.

        Warm-up is idempotent. Repeated calls do not reload the provider or
        repeat expensive provider initialization.
        """
        if self._warmed_up:
            logger.debug("Embedding provider warm-up already completed.")
            return

        self._model.warmup()
        self._warmed_up = True
        logger.info("Embedding provider warm-up complete.")

    def embed(self, faces: list[Face]) -> list[Face]:
        """Extract embeddings and enrich Face objects with embedding data.

        Pipeline:
            1. Validate aligned faces
            2. Load the configured embedding model
            3. Extract embeddings for each aligned face crop
            4. Attach ``EmbeddingData`` to each Face

        Args:
            faces: Faces with populated ``Face.alignment`` data.

        Returns:
            The same Face objects enriched with embedding information.

        Raises:
            ValueError: If the input face list or alignment data is invalid.
            NotImplementedError: If the configured provider is not implemented.
        """
        validate_embedding_input(faces)

        if not faces:
            logger.debug("No faces to embed.")
            return faces

        self._model.load()

        for index, face in enumerate(faces):
            alignment = face.alignment
            if alignment is None or alignment.aligned_image is None:
                raise ValueError(
                    f"Face at index {index} is missing alignment data after validation."
                )

            face.embedding = self._model.embed(alignment.aligned_image)

            logger.debug("Embedded face %d.", index)

        logger.info("Embedded %d face(s).", len(faces))
        return faces


def embed(faces: list[Face], model: EmbeddingModel | None = None) -> list[Face]:
    """Embed aligned faces using the default face embedder.

    Public entry point for the embedding module. Enriches existing ``Face``
    objects with embedding data without creating new domain types.

    Args:
        faces: Faces with populated ``Face.alignment`` data.
        model: Optional embedding model for dependency injection.

    Returns:
        The same Face objects enriched with embedding information.
    """
    return FaceEmbedder(model=model).embed(faces)
