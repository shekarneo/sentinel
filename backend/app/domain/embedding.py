"""
Embedding stage domain model.

Stores outputs produced by the Embedding Service.
"""

import numpy as np
from pydantic import BaseModel, ConfigDict, Field


class EmbeddingData(BaseModel):
    """Outputs from the Embedding Service stage.

    Populated by the embedding module and attached to ``Face.embedding``.
    Stores the biometric feature vector extracted from an aligned face crop.

    **Embedding invariant:** ``vector`` always contains a unit-length
    (L2-normalized) embedding for successful extractions. Downstream modules
    such as FAISS indexing, similarity search, verification, and clustering
    may assume normalization and must not re-normalize vectors.

    This object represents embedding output only. It must not include
    similarity scores, identity labels, search results, or gallery metadata.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    vector: np.ndarray | None = Field(
        default=None,
        description=(
            "Unit-length L2-normalized feature vector. Providers must "
            "normalize before populating this field."
        ),
    )
    dimension: int | None = None
    normalized: bool = Field(
        default=False,
        description=(
            "Guaranteed ``True`` for successful embeddings. Indicates the "
            "provider applied L2 normalization before returning "
            "``EmbeddingData``. Downstream modules may safely assume "
            "``vector`` is normalized when this is ``True``."
        ),
    )
    model_name: str
    inference_time_ms: float | None = None
    confidence: float | None = None
