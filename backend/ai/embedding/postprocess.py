"""
Embedding postprocessing utilities.

Converts raw ONNX outputs into domain ``EmbeddingData`` with mandatory
L2 normalization.
"""

import numpy as np

from backend.app.domain.embedding import EmbeddingData

_NORM_TOLERANCE = 1e-5


def postprocess_embedding(
    raw_output: np.ndarray,
    *,
    model_name: str,
    inference_time_ms: float | None = None,
    expected_dimension: int | None = None,
) -> EmbeddingData:
    """Convert raw model output into a unit-normalized ``EmbeddingData``.

    Args:
        raw_output: Raw embedding tensor from the model runtime.
        model_name: Provider model identifier.
        inference_time_ms: Optional inference latency in milliseconds.
        expected_dimension: Optional expected vector length from the ONNX graph.

    Returns:
        Populated ``EmbeddingData`` with ``normalized=True``.

    Raises:
        ValueError: If the raw output is invalid or normalization fails.
    """
    if not isinstance(raw_output, np.ndarray):
        raise ValueError(
            f"Raw embedding output must be a numpy array, got {type(raw_output)!r}."
        )

    if raw_output.size == 0:
        raise ValueError("Raw embedding output must be non-empty.")

    vector = np.asarray(raw_output, dtype=np.float32).reshape(-1)

    if expected_dimension is not None and vector.size != expected_dimension:
        raise ValueError(
            "Embedding dimension does not match model: "
            f"expected {expected_dimension}, got {vector.size}."
        )

    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        raise ValueError("Cannot L2-normalize a zero embedding vector.")

    normalized_vector = vector / norm
    final_norm = float(np.linalg.norm(normalized_vector))
    if abs(final_norm - 1.0) > _NORM_TOLERANCE:
        raise ValueError(
            "L2 normalization failed: "
            f"expected unit norm, got {final_norm:.8f}."
        )

    return EmbeddingData(
        vector=normalized_vector,
        dimension=int(normalized_vector.size),
        normalized=True,
        model_name=model_name,
        inference_time_ms=inference_time_ms,
        confidence=None,
    )
