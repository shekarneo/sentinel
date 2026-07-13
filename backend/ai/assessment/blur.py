"""
Blur analysis for face assessment.
"""

import logging

import cv2
import numpy as np

from backend.ai.assessment.config import load_assessment_thresholds
from backend.ai.assessment.utils import clip_score, validate_aligned_face
from backend.app.domain.assessment import BlurResult

logger = logging.getLogger(__name__)


class BlurAnalyzer:
    """Estimates blur on an aligned face crop using variance of Laplacian."""

    def __init__(
        self,
        warning_threshold: float | None = None,
        acceptable_threshold: float | None = None,
        excellent_threshold: float | None = None,
    ) -> None:
        """Initialize the blur analyzer.

        Args:
            warning_threshold: Variance at or below which score is 0. Loaded
                from configuration when omitted.
            acceptable_threshold: Variance mapped to a score of 0.5. Loaded
                from configuration when omitted.
            excellent_threshold: Variance at or above which score is 1. Loaded
                from configuration when omitted.

        Raises:
            ValueError: If thresholds are not strictly increasing.
        """
        blur = load_assessment_thresholds().blur
        self._warning_threshold = (
            blur.warning if warning_threshold is None else warning_threshold
        )
        self._acceptable_threshold = (
            blur.acceptable if acceptable_threshold is None else acceptable_threshold
        )
        self._excellent_threshold = (
            blur.excellent if excellent_threshold is None else excellent_threshold
        )

        if not (
            self._warning_threshold
            < self._acceptable_threshold
            < self._excellent_threshold
        ):
            raise ValueError(
                "Blur thresholds must satisfy "
                "warning < acceptable < excellent."
            )

    def evaluate(self, aligned_face: np.ndarray) -> BlurResult:
        """Evaluate blur on an aligned face crop.

        Args:
            aligned_face: Aligned face image in HWC BGR uint8 layout.

        Returns:
            ``BlurResult`` containing Laplacian variance and normalized score.

        Raises:
            ValueError: If the aligned face image is invalid.
        """
        validate_aligned_face(aligned_face)

        variance = self._compute_variance(aligned_face)
        score = self._normalize_score(variance)

        logger.debug(
            "Blur analysis: variance=%.2f score=%.4f",
            variance,
            score,
        )

        return BlurResult(variance=variance, score=score)

    def _compute_variance(self, aligned_face: np.ndarray) -> float:
        """Compute Laplacian variance for a BGR aligned face.

        Args:
            aligned_face: Validated aligned face crop in BGR uint8 format.

        Returns:
            Variance of the Laplacian operator applied to the grayscale image.
        """
        gray = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        return float(laplacian.var())

    def _normalize_score(self, variance: float) -> float:
        """Map Laplacian variance to a score in [0, 1] via piecewise linear interpolation.

        Scoring segments:
            - ``variance <= warning`` → 0.0
            - ``warning < variance < acceptable`` → 0.0 to 0.5
            - ``acceptable <= variance < excellent`` → 0.5 to 1.0
            - ``variance >= excellent`` → 1.0

        Args:
            variance: Laplacian variance of the aligned face.

        Returns:
            Normalized blur score clamped to [0, 1].
        """
        warning = self._warning_threshold
        acceptable = self._acceptable_threshold
        excellent = self._excellent_threshold

        if variance <= warning:
            return 0.0

        if variance >= excellent:
            return 1.0

        if variance < acceptable:
            ratio = (variance - warning) / (acceptable - warning)
            return clip_score(0.5 * ratio)

        ratio = (variance - acceptable) / (excellent - acceptable)
        return clip_score(0.5 + 0.5 * ratio)
