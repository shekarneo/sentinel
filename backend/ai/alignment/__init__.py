"""
Face alignment module.

Normalizes detected faces using landmark geometry and enriches existing
``Face`` domain objects with aligned face crops for downstream assessment,
fraud detection, and embedding stages.
"""

from backend.ai.alignment.aligner import FaceAligner, align

__all__ = [
    "FaceAligner",
    "align",
]
