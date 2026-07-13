"""
ArcFace embedding model.

Loads the ArcFace ONNX provider and extracts unit-normalized embeddings
from aligned face crops.
"""

from __future__ import annotations

import logging
import threading
import time

import numpy as np

from backend.ai.common.onnx_engine import ONNXRuntimeEngine
from backend.ai.embedding.config import (
    get_embedding_model_path,
    load_embedding_model_settings,
)
from backend.ai.embedding.model import EmbeddingModel
from backend.ai.embedding.postprocess import postprocess_embedding
from backend.ai.embedding.preprocess import preprocess_aligned_face
from backend.app.domain.embedding import EmbeddingData

logger = logging.getLogger(__name__)


class ArcFaceEmbeddingModel(EmbeddingModel):
    """ArcFace embedding provider.

    This class is the first concrete ``EmbeddingModel`` implementation.
    Additional providers such as AdaFace, MagFace, and ElasticFace should
    follow the same interface without changing ``FaceEmbedder``.
    """

    def __init__(self) -> None:
        """Initialize ArcFace provider settings from configuration."""
        settings = load_embedding_model_settings()
        self._provider = settings.provider
        self._model_name = settings.model
        self._input_size = settings.input_size
        self._engine: ONNXRuntimeEngine | None = None
        self._input_name: str | None = None
        self._embedding_dim: int | None = None
        self._load_lock = threading.Lock()

    def load(self) -> None:
        """Load and validate the ArcFace ONNX model.

        Loading is idempotent and thread-safe. Concurrent callers block until
        the first initialization completes, then reuse the loaded session.

        Raises:
            FileNotFoundError: If the configured model file does not exist.
            ValueError: If the model IO contract does not match expectations.
        """
        if self._engine is not None:
            return

        with self._load_lock:
            if self._engine is not None:
                return

            model_path = get_embedding_model_path()

            if not model_path.exists():
                raise FileNotFoundError(
                    f"ArcFace embedding model not found: {model_path}"
                )

            engine = ONNXRuntimeEngine(model_path)
            engine.load()
            self._validate_engine(engine)

            session = engine.session
            if session is None:
                raise RuntimeError("ONNX session is not initialized after load.")

            input_shape = session.get_inputs()[0].shape
            output_shape = session.get_outputs()[0].shape

            input_name = engine.input_names[0]
            output_name = engine.output_names[0]

            self._input_name = input_name
            self._embedding_dim = self._resolve_embedding_dimension(engine)
            self._engine = engine

            logger.info(
                "Loaded ArcFace embedding model: provider=%s model=%s "
                "input_name=%s input_shape=%s output_name=%s output_shape=%s "
                "embedding_dim=%s",
                self._provider,
                self._model_name,
                input_name,
                input_shape,
                output_name,
                output_shape,
                self._embedding_dim if self._embedding_dim is not None else "unknown",
            )

    def _validate_engine(self, engine: ONNXRuntimeEngine) -> None:
        """Validate ArcFace ONNX input and output tensor contracts."""
        if len(engine.input_names) != 1:
            raise ValueError(
                "ArcFace model must have exactly one input tensor, "
                f"got {len(engine.input_names)}: {engine.input_names}"
            )

        if len(engine.output_names) != 1:
            raise ValueError(
                "ArcFace model must have exactly one output tensor, "
                f"got {len(engine.output_names)}: {engine.output_names}"
            )

        session = engine.session
        if session is None:
            raise RuntimeError("ONNX session is not initialized after load.")

        input_shape = session.get_inputs()[0].shape
        if len(input_shape) != 4:
            raise ValueError(
                "ArcFace input must be NCHW with 4 dimensions, "
                f"got shape {input_shape}"
            )

        _, channels, height, width = input_shape
        if channels != 3:
            raise ValueError(
                "ArcFace input must be NCHW with 3 channels, "
                f"got shape {input_shape}"
            )

        for axis_name, axis_value in (("height", height), ("width", width)):
            if not isinstance(axis_value, int):
                raise ValueError(
                    "ArcFace input height and width must be fixed integers "
                    f"matching input_size ({self._input_size}), got shape "
                    f"{input_shape}"
                )
            if axis_value != self._input_size:
                raise ValueError(
                    f"ArcFace input {axis_name} ({axis_value}) does not match "
                    f"configured input_size ({self._input_size})"
                )

    @staticmethod
    def _resolve_embedding_dimension(engine: ONNXRuntimeEngine) -> int | None:
        """Return the embedding dimension when declared by the ONNX graph."""
        session = engine.session
        if session is None:
            return None

        output_shape = session.get_outputs()[0].shape
        if not output_shape:
            return None

        embedding_dim = output_shape[-1]
        return embedding_dim if isinstance(embedding_dim, int) else None

    def embed(self, aligned_face: np.ndarray) -> EmbeddingData:
        """Extract an ArcFace embedding from an aligned face crop.

        Args:
            aligned_face: Aligned face image in HWC BGR uint8 layout.

        Returns:
            ``EmbeddingData`` containing a unit-length feature vector.

        Raises:
            ValueError: If the aligned face image is invalid.
            RuntimeError: If the model has not been loaded.
        """
        if self._engine is None or self._input_name is None:
            self.load()

        if self._engine is None or self._input_name is None:
            raise RuntimeError("ArcFace model failed to initialize after load.")

        input_name = self._input_name
        engine = self._engine

        input_tensor = preprocess_aligned_face(
            aligned_face,
            input_size=self._input_size,
        )

        start_time = time.perf_counter()
        outputs = engine.infer({input_name: input_tensor})
        inference_time_ms = (time.perf_counter() - start_time) * 1000.0

        return postprocess_embedding(
            outputs[0],
            model_name=self._model_name,
            inference_time_ms=inference_time_ms,
            expected_dimension=self._embedding_dim,
        )

    def embed_batch(self, aligned_faces: list[np.ndarray]) -> list[EmbeddingData]:
        """Extract ArcFace embeddings for a batch of aligned face crops.

        Args:
            aligned_faces: Aligned face images in HWC BGR uint8 layout.

        Returns:
            One ``EmbeddingData`` per input face.

        Raises:
            NotImplementedError: ArcFace batch embedding is not yet implemented.
        """
        raise NotImplementedError("ArcFace batch embedding is not yet implemented.")
