"""
Face size analysis for face assessment.
"""

import logging

from backend.ai.assessment.config import load_assessment_thresholds
from backend.ai.assessment.utils import clip_score
from backend.app.domain.assessment import SizeResult
from backend.app.domain.face import Face

logger = logging.getLogger(__name__)


class SizeAnalyzer:
    """Estimates whether the detected face occupies sufficient image area."""

    def __init__(
        self,
        min_face_width: float | None = None,
        min_face_height: float | None = None,
    ) -> None:
        """Initialize the size analyzer.

        Args:
            min_face_width: Minimum acceptable bounding-box width in pixels.
                Loaded from configuration when omitted.
            min_face_height: Minimum acceptable bounding-box height in pixels.
                Loaded from configuration when omitted.

        Raises:
            ValueError: If minimum dimensions are not positive.
        """
        size = load_assessment_thresholds().size
        self._min_face_width = (
            size.min_face_width if min_face_width is None else min_face_width
        )
        self._min_face_height = (
            size.min_face_height if min_face_height is None else min_face_height
        )

        if self._min_face_width <= 0 or self._min_face_height <= 0:
            raise ValueError("Minimum face dimensions must be positive.")

    def evaluate(self, face: Face) -> SizeResult:
        """Evaluate face size from the detection bounding box.

        Args:
            face: Face with populated detection bounding box.

        Returns:
            ``SizeResult`` containing width, height, and normalized score.

        Raises:
            ValueError: If the bounding box is invalid.
        """
        width, height = self._extract_dimensions(face)
        score = self._normalize_score(width, height)

        logger.debug(
            "Size analysis: width=%.2f height=%.2f score=%.4f",
            width,
            height,
            score,
        )

        return SizeResult(width=width, height=height, score=score)

    def _extract_dimensions(self, face: Face) -> tuple[float, float]:
        """Read width and height from the face bounding box.

        Args:
            face: Face with a detection bounding box.

        Returns:
            Bounding-box width and height in pixels.

        Raises:
            ValueError: If dimensions are non-positive.
        """
        width = float(face.bounding_box.width)
        height = float(face.bounding_box.height)

        if width <= 0 or height <= 0:
            raise ValueError(
                f"Bounding box dimensions must be positive, got "
                f"width={width}, height={height}."
            )

        return width, height

    def _normalize_score(self, width: float, height: float) -> float:
        """Map face dimensions to a score in [0, 1].

        The score uses the limiting dimension so both width and height must
        meet the minimum thresholds for a perfect score.

        Args:
            width: Bounding-box width in pixels.
            height: Bounding-box height in pixels.

        Returns:
            Normalized size score clamped to [0, 1].
        """
        width_score = width / self._min_face_width
        height_score = height / self._min_face_height
        return clip_score(min(width_score, height_score))
