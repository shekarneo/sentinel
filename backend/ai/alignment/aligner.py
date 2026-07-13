"""
Face aligner orchestration.

Enriches existing ``Face`` objects with aligned face crops. The aligner does
not create separate domain objects such as ``AlignedFace`` or
``AlignmentResult``.
"""

import logging

import numpy as np

from backend.ai.alignment.constants import CANONICAL_FACE_SIZE
from backend.ai.alignment.template import get_reference_landmarks
from backend.ai.alignment.transform import estimate_similarity_transform, warp_face
from backend.ai.alignment.utils import landmarks_to_array, validate_alignment_input
from backend.app.domain.alignment import AlignmentData
from backend.app.domain.face import Face

logger = logging.getLogger(__name__)


class FaceAligner:
    """Orchestrates the face alignment pipeline.

    Consumes detected ``Face`` objects and the source image, then enriches
    each face with alignment data (aligned crop and transform metadata).

    Face Alignment is shared by both Face Assessment and Embedding stages.
    """

    def __init__(self, face_size: int = CANONICAL_FACE_SIZE) -> None:
        """Initialize the face aligner.

        Args:
            face_size: Canonical width and height for aligned face crops.
        """
        self._face_size = face_size
        self._output_size = (face_size, face_size)

    def align(
        self,
        image: np.ndarray,
        faces: list[Face],
    ) -> list[Face]:
        """Align detected faces and enrich Face objects with alignment data.

        Pipeline:
            1. Extract landmark coordinates from each Face
            2. Estimate similarity transform to the ArcFace reference template
            3. Warp the source image to canonical face size
            4. Attach ``AlignmentData`` to each Face

        Args:
            image: Source image in HWC BGR layout.
            faces: Faces detected by the upstream detection stage.

        Returns:
            The same Face objects enriched with alignment information.

        Raises:
            ValueError: If the input image or face list is invalid.
        """
        validate_alignment_input(image, faces)

        if not faces:
            logger.debug("No faces to align.")
            return faces

        reference_landmarks = get_reference_landmarks(self._face_size)

        for face in faces:
            self._align_face(image, face, reference_landmarks)

        logger.debug("Aligned %d faces", len(faces))
        return faces

    def _align_face(
        self,
        image: np.ndarray,
        face: Face,
        reference_landmarks: np.ndarray,
    ) -> None:
        """Align a single face and populate ``Face.alignment``."""
        source_landmarks = landmarks_to_array(face.landmarks)
        transform_matrix = estimate_similarity_transform(
            source_landmarks,
            reference_landmarks,
        )
        aligned_image = warp_face(
            image,
            transform_matrix,
            self._output_size,
        )

        face.alignment = AlignmentData(
            aligned_image=aligned_image,
            transformation_matrix=transform_matrix,
        )


def align(
    image: np.ndarray,
    faces: list[Face],
) -> list[Face]:
    """Align detected faces using the default face aligner.

    Public entry point for the alignment module. Enriches existing ``Face``
    objects with alignment data without creating new domain types.

    Args:
        image: Source image in HWC BGR layout.
        faces: Faces detected by the upstream detection stage.

    Returns:
        The same Face objects enriched with alignment information.
    """
    return FaceAligner().align(image, faces)
