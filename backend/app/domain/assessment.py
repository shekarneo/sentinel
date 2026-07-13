"""
Assessment stage domain model.

Stores outputs produced by the Face Assessment module.
"""

from pydantic import BaseModel


class AssessmentData(BaseModel):
    """Outputs from the Face Assessment stage.

    Populated by the assessment module and attached to ``Face.assessment``.
    Evaluates whether an aligned face is suitable for biometric processing.
    """

    quality_score: float | None = None
    blur_score: float | None = None
    pose_yaw: float | None = None
    pose_pitch: float | None = None
    pose_roll: float | None = None
    visibility_score: float | None = None
    brightness_score: float | None = None
