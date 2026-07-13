"""
Search engine shared types.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RawSearchOutput:
    """Raw nearest-neighbor search output from a vector index.

    ``indices`` are embedding identifiers assigned by the index provider.
    ``distances`` are inner-product scores when using ``IndexFlatIP`` with
    unit-normalized vectors (equivalent to cosine similarity).
    """

    indices: np.ndarray
    distances: np.ndarray
