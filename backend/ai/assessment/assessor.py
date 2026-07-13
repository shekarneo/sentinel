"""
Face assessor orchestration.

Enriches existing ``Face`` objects with quality assessment data. The assessor
does not create separate domain objects such as ``AssessmentResult``.
"""

import logging

import numpy as np

from backend.ai.assessment.blur import BlurAnalyzer
from backend.ai.assessment.brightness import BrightnessAnalyzer
from backend.ai.assessment.pose import PoseAnalyzer
from backend.ai.assessment.scoring import compute_overall_score
from backend.ai.assessment.size import SizeAnalyzer
from backend.ai.assessment.utils import build_assessment_data, validate_assessment_input
from backend.ai.assessment.visibility import VisibilityAnalyzer
from backend.app.domain.face import Face

logger = logging.getLogger(__name__)


class FaceAssessor:
    """Orchestrates the face assessment pipeline.

    Consumes aligned ``Face`` objects and enriches each face with
    ``AssessmentData``. Assessment operates on aligned faces only and
    must not modify detection, alignment, or embedding fields.
    """

    def __init__(self) -> None:
        """Initialize analyzers used by the assessment pipeline."""
        self._blur_analyzer = BlurAnalyzer()
        self._brightness_analyzer = BrightnessAnalyzer()
        self._pose_analyzer = PoseAnalyzer()
        self._visibility_analyzer = VisibilityAnalyzer()
        self._size_analyzer = SizeAnalyzer()

    def assess(
        self,
        image: np.ndarray,
        faces: list[Face],
    ) -> list[Face]:
        """Assess aligned faces and enrich Face objects with assessment data.

        Pipeline:
            1. Validate aligned faces
            2. Run blur, brightness, size, visibility, and pose analyzers
            3. Combine analyzer results into ``AssessmentData``
            4. Compute overall score and acceptability
            5. Attach ``AssessmentData`` to each Face

        Args:
            image: Source image in HWC BGR layout.
            faces: Faces with populated ``Face.alignment`` data.

        Returns:
            The same Face objects enriched with assessment information.

        Raises:
            ValueError: If the input face list or alignment data is invalid.
        """
        validate_assessment_input(faces)

        if not faces:
            logger.debug("No faces to assess.")
            return faces

        for index, face in enumerate(faces):
            assert face.alignment is not None
            assert face.alignment.aligned_image is not None

            aligned_face = face.alignment.aligned_image
            blur = self._blur_analyzer.evaluate(aligned_face)
            brightness = self._brightness_analyzer.evaluate(aligned_face)
            size = self._size_analyzer.evaluate(face)
            visibility = self._visibility_analyzer.evaluate(face, image)
            pose = self._pose_analyzer.evaluate(face)

            assessment = build_assessment_data(
                blur=blur,
                brightness=brightness,
                pose=pose,
                size=size,
                visibility=visibility,
            )
            scoring = compute_overall_score(assessment)

            face.assessment = assessment.model_copy(
                update={
                    "overall_score": scoring.overall_score,
                    "is_acceptable": scoring.is_acceptable,
                }
            )

            logger.debug(
                "Assessed face %d: overall=%.4f acceptable=%s",
                index,
                face.assessment.overall_score,
                face.assessment.is_acceptable,
            )

        logger.info("Assessed %d face(s).", len(faces))
        return faces


def assess(
    image: np.ndarray,
    faces: list[Face],
) -> list[Face]:
    """Assess aligned faces using the default face assessor.

    Public entry point for the assessment module. Enriches existing ``Face``
    objects with assessment data without creating new domain types.

    Args:
        image: Source image in HWC BGR layout.
        faces: Faces with populated ``Face.alignment`` data.

    Returns:
        The same Face objects enriched with assessment information.
    """
    return FaceAssessor().assess(image, faces)
