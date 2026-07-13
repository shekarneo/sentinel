"""
Pose analysis for face assessment.

``PoseAnalyzer`` is **not** a calibrated head pose estimator. It applies a
lightweight landmark-based quality heuristic used only to score whether a face
is sufficiently frontal for biometric processing.

The reported yaw, pitch, and roll values are approximate geometric indicators.
They must never be interpreted as precise head pose measurements or used
outside face quality assessment.
"""

import logging
import math

import numpy as np

from backend.ai.alignment.utils import landmarks_to_array
from backend.ai.assessment.config import load_assessment_thresholds
from backend.ai.assessment.utils import clip_score
from backend.app.domain.assessment import PoseResult
from backend.app.domain.face import Face, Landmark

logger = logging.getLogger(__name__)


class PoseAnalyzer:
    """Scores face frontalness using lightweight landmark geometry.

    This analyzer provides a quality heuristic only. It does not perform
    calibrated head pose estimation and its angle outputs must not be treated
    as precise yaw, pitch, or roll measurements.
    """

    def __init__(
        self,
        max_yaw: float | None = None,
        max_pitch: float | None = None,
        max_roll: float | None = None,
    ) -> None:
        """Initialize the pose analyzer.

        Args:
            max_yaw: Maximum acceptable absolute yaw in degrees. Loaded from
                configuration when omitted.
            max_pitch: Maximum acceptable absolute pitch in degrees. Loaded
                from configuration when omitted.
            max_roll: Maximum acceptable absolute roll in degrees. Loaded from
                configuration when omitted.

        Raises:
            ValueError: If any pose limit is not positive.
        """
        pose = load_assessment_thresholds().pose
        self._max_yaw = pose.max_yaw if max_yaw is None else max_yaw
        self._max_pitch = pose.max_pitch if max_pitch is None else max_pitch
        self._max_roll = pose.max_roll if max_roll is None else max_roll

        if self._max_yaw <= 0 or self._max_pitch <= 0 or self._max_roll <= 0:
            raise ValueError("Pose limits must be positive.")

    def evaluate(self, face: Face) -> PoseResult:
        """Estimate pose quality from five-point facial landmarks.

        The returned yaw, pitch, and roll values are heuristic quality
        indicators derived from landmark geometry. They are not calibrated
        pose estimates.

        Args:
            face: Face with populated SCRFD landmarks.

        Returns:
            ``PoseResult`` containing heuristic yaw, pitch, roll, and score.

        Raises:
            ValueError: If landmarks are missing or degenerate.
        """
        yaw, pitch, roll = self._estimate_pose(face.landmarks)
        score = self._normalize_score(yaw, pitch, roll)

        logger.debug(
            "Pose analysis: yaw=%.2f pitch=%.2f roll=%.2f score=%.4f",
            yaw,
            pitch,
            roll,
            score,
        )

        return PoseResult(yaw=yaw, pitch=pitch, roll=roll, score=score)

    def _estimate_pose(self, landmarks: list[Landmark]) -> tuple[float, float, float]:
        """Estimate heuristic yaw, pitch, and roll from landmark geometry.

        Args:
            landmarks: Five facial landmarks in SCRFD order.

        Returns:
            Heuristic yaw, pitch, and roll in degrees.

        Raises:
            ValueError: If landmark geometry is degenerate.
        """
        points = landmarks_to_array(landmarks)
        left_eye = points[0]
        right_eye = points[1]
        nose = points[2]
        left_mouth = points[3]
        right_mouth = points[4]

        eye_vector = right_eye - left_eye
        inter_eye_distance = float(np.linalg.norm(eye_vector))
        if inter_eye_distance <= 1e-6:
            raise ValueError("Eye landmarks are too close to estimate pose.")

        eye_center = (left_eye + right_eye) / 2.0
        mouth_center = (left_mouth + right_mouth) / 2.0
        vertical_span = float(mouth_center[1] - eye_center[1])

        roll = math.degrees(math.atan2(eye_vector[1], eye_vector[0]))
        yaw = math.degrees(math.atan2(nose[0] - eye_center[0], inter_eye_distance))

        if abs(vertical_span) <= 1e-6:
            pitch = 0.0
        else:
            expected_nose_y = eye_center[1] + 0.55 * vertical_span
            pitch = math.degrees(
                math.atan2(nose[1] - expected_nose_y, abs(vertical_span))
            )

        return yaw, pitch, roll

    def _normalize_score(self, yaw: float, pitch: float, roll: float) -> float:
        """Map pose angles to a conservative score in [0, 1].

        Args:
            yaw: Heuristic yaw in degrees.
            pitch: Heuristic pitch in degrees.
            roll: Heuristic roll in degrees.

        Returns:
            Normalized pose score clamped to [0, 1].
        """
        yaw_score = clip_score(1.0 - abs(yaw) / self._max_yaw)
        pitch_score = clip_score(1.0 - abs(pitch) / self._max_pitch)
        roll_score = clip_score(1.0 - abs(roll) / self._max_roll)
        return min(yaw_score, pitch_score, roll_score)
