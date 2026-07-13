"""
Face detector interface.
"""

from abc import ABC, abstractmethod

import numpy as np

from backend.app.domain.face import Face


class FaceDetector(ABC):
    """
    Base interface for all face detectors.
    """

    @abstractmethod
    def detect(self, image: np.ndarray) -> list[Face]:
        """
        Detect faces in an image.

        Parameters
        ----------
        image:
            Input image.

        Returns
        -------
        List of detected Face objects.
        """
        pass