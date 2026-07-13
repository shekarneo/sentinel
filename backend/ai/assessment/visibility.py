"""
Visibility analysis for face assessment.
"""

import logging

import numpy as np

from backend.ai.assessment.config import load_assessment_thresholds
from backend.ai.assessment.utils import clip_score
from backend.app.domain.assessment import VisibilityResult
from backend.app.domain.face import Face

logger = logging.getLogger(__name__)


class VisibilityAnalyzer:
    """Estimates how much of the face bounding box is visible in the image."""

    def __init__(self, min_visible_ratio: float | None = None) -> None:
        """Initialize the visibility analyzer.

        Args:
            min_visible_ratio: Ratio at or above which visibility score is 1.
                Loaded from configuration when omitted.

        Raises:
            ValueError: If the minimum visible ratio is not in (0, 1].
        """
        visibility = load_assessment_thresholds().visibility
        self._min_visible_ratio = (
            visibility.minimum_visible_ratio
            if min_visible_ratio is None
            else min_visible_ratio
        )

        if not 0 < self._min_visible_ratio <= 1.0:
            raise ValueError(
                "Minimum visible ratio must be in the range (0, 1]."
            )

    def evaluate(self, face: Face, image: np.ndarray) -> VisibilityResult:
        """Evaluate face visibility against image boundaries.

        Args:
            face: Face with a detection bounding box.
            image: Source image in HWC BGR layout.

        Returns:
            ``VisibilityResult`` containing visible ratio and normalized score.

        Raises:
            ValueError: If the image or bounding box is invalid.
        """
        visible_ratio = self._compute_visible_ratio(face, image)
        score = self._normalize_score(visible_ratio)

        logger.debug(
            "Visibility analysis: ratio=%.4f score=%.4f",
            visible_ratio,
            score,
        )

        return VisibilityResult(visible_ratio=visible_ratio, score=score)

    def _compute_visible_ratio(self, face: Face, image: np.ndarray) -> float:
        """Compute the clipped bounding-box area ratio.

        Args:
            face: Face with a detection bounding box.
            image: Source image in HWC BGR layout.

        Returns:
            Ratio of visible bounding-box area to full bounding-box area.

        Raises:
            ValueError: If the image or bounding box is invalid.
        """
        if not isinstance(image, np.ndarray) or image.size == 0:
            raise ValueError("Image must be a non-empty numpy array.")

        if image.ndim < 2:
            raise ValueError(f"Image must be at least 2D, got shape {image.shape}.")

        image_height, image_width = image.shape[:2]
        bbox = face.bounding_box

        bbox_width = float(bbox.width)
        bbox_height = float(bbox.height)
        bbox_area = bbox_width * bbox_height

        if bbox_area <= 0:
            raise ValueError("Bounding box area must be positive.")

        x1 = max(0.0, float(bbox.x))
        y1 = max(0.0, float(bbox.y))
        x2 = min(float(image_width), float(bbox.x) + bbox_width)
        y2 = min(float(image_height), float(bbox.y) + bbox_height)

        visible_width = max(0.0, x2 - x1)
        visible_height = max(0.0, y2 - y1)
        visible_area = visible_width * visible_height

        return float(visible_area / bbox_area)

    def _normalize_score(self, visible_ratio: float) -> float:
        """Map visible ratio to a score in [0, 1].

        Args:
            visible_ratio: Fraction of the bounding box visible in-frame.

        Returns:
            Normalized visibility score clamped to [0, 1].
        """
        if visible_ratio >= self._min_visible_ratio:
            return 1.0

        return clip_score(visible_ratio / self._min_visible_ratio)
