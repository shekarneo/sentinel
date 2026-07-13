"""
Brightness analysis for face assessment.
"""

import logging

import cv2
import numpy as np

from backend.ai.assessment.config import load_assessment_thresholds
from backend.ai.assessment.utils import clip_score, validate_aligned_face
from backend.app.domain.assessment import BrightnessResult

logger = logging.getLogger(__name__)


class BrightnessAnalyzer:
    """Estimates brightness and exposure on an aligned face crop."""

    def __init__(
        self,
        too_dark_threshold: float | None = None,
        acceptable_low_threshold: float | None = None,
        acceptable_high_threshold: float | None = None,
        too_bright_threshold: float | None = None,
    ) -> None:
        """Initialize the brightness analyzer.

        Args:
            too_dark_threshold: Mean brightness at or below which score is 0.
                Loaded from configuration when omitted.
            acceptable_low_threshold: Lower bound of the acceptable range
                (score 1). Loaded from configuration when omitted.
            acceptable_high_threshold: Upper bound of the acceptable range
                (score 1). Loaded from configuration when omitted.
            too_bright_threshold: Mean brightness at or above which score is 0.
                Loaded from configuration when omitted.

        Raises:
            ValueError: If thresholds are not strictly increasing.
        """
        brightness = load_assessment_thresholds().brightness
        self._too_dark_threshold = (
            brightness.too_dark if too_dark_threshold is None else too_dark_threshold
        )
        self._acceptable_low_threshold = (
            brightness.acceptable_low
            if acceptable_low_threshold is None
            else acceptable_low_threshold
        )
        self._acceptable_high_threshold = (
            brightness.acceptable_high
            if acceptable_high_threshold is None
            else acceptable_high_threshold
        )
        self._too_bright_threshold = (
            brightness.too_bright
            if too_bright_threshold is None
            else too_bright_threshold
        )

        if not (
            self._too_dark_threshold
            < self._acceptable_low_threshold
            < self._acceptable_high_threshold
            < self._too_bright_threshold
        ):
            raise ValueError(
                "Brightness thresholds must satisfy "
                "too_dark < acceptable_low < acceptable_high < too_bright."
            )

    def evaluate(self, aligned_face: np.ndarray) -> BrightnessResult:
        """Evaluate brightness on an aligned face crop.

        Args:
            aligned_face: Aligned face image in HWC BGR uint8 layout.

        Returns:
            ``BrightnessResult`` containing mean brightness and normalized score.

        Raises:
            ValueError: If the aligned face image is invalid.
        """
        validate_aligned_face(aligned_face)

        mean_brightness = self._compute_mean_brightness(aligned_face)
        score = self._normalize_score(mean_brightness)

        logger.debug(
            "Brightness analysis: mean=%.2f score=%.4f",
            mean_brightness,
            score,
        )

        return BrightnessResult(
            mean_brightness=mean_brightness,
            score=score,
        )

    def _compute_mean_brightness(self, aligned_face: np.ndarray) -> float:
        """Compute mean grayscale intensity for a BGR aligned face.

        Args:
            aligned_face: Validated aligned face crop in BGR uint8 format.

        Returns:
            Mean pixel intensity on the 0–255 grayscale scale.
        """
        gray = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2GRAY)
        return float(gray.mean())

    def _normalize_score(self, mean_brightness: float) -> float:
        """Map mean brightness to a score in [0, 1] via piecewise linear interpolation.

        Scoring segments:
            - ``mean <= too_dark`` → 0.0
            - ``too_dark < mean < acceptable_low`` → 0.0 to 1.0
            - ``acceptable_low <= mean <= acceptable_high`` → 1.0
            - ``acceptable_high < mean < too_bright`` → 1.0 to 0.0
            - ``mean >= too_bright`` → 0.0

        Args:
            mean_brightness: Mean grayscale intensity of the aligned face.

        Returns:
            Normalized brightness score clamped to [0, 1].
        """
        too_dark = self._too_dark_threshold
        acceptable_low = self._acceptable_low_threshold
        acceptable_high = self._acceptable_high_threshold
        too_bright = self._too_bright_threshold

        if mean_brightness <= too_dark or mean_brightness >= too_bright:
            return 0.0

        if acceptable_low <= mean_brightness <= acceptable_high:
            return 1.0

        if mean_brightness < acceptable_low:
            ratio = (mean_brightness - too_dark) / (acceptable_low - too_dark)
            return clip_score(ratio)

        ratio = (too_bright - mean_brightness) / (too_bright - acceptable_high)
        return clip_score(ratio)
