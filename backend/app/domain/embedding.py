"""
Embedding stage domain model.

Stores outputs produced by the Embedding Service.
"""

import numpy as np
from pydantic import BaseModel, ConfigDict


class EmbeddingData(BaseModel):
    """Outputs from the Embedding Service stage.

    Populated by the embedding module and attached to ``Face.embedding``.
    Stores the biometric feature vector extracted from an aligned face crop.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    vector: np.ndarray | None = None
    model_name: str | None = None
    dimension: int | None = None
