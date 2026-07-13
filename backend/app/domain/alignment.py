"""
Alignment stage domain model.

Stores outputs produced by the Face Alignment module.
"""

import numpy as np
from pydantic import BaseModel, ConfigDict


class AlignmentData(BaseModel):
    """Outputs from the Face Alignment stage.

    Populated by the alignment module and attached to ``Face.alignment``.
    Alignment normalizes a detected face to a canonical orientation and
    scale for downstream assessment, fraud detection, and embedding.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    aligned_image: np.ndarray | None = None
    transformation_matrix: np.ndarray | None = None
