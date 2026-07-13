"""
Assessment stage domain model.

Stores outputs produced by the Face Assessment module.
"""

from pydantic import BaseModel


class BlurResult(BaseModel):
    """Blur metrics from the blur analyzer."""

    variance: float
    score: float


class BrightnessResult(BaseModel):
    """Brightness metrics from the brightness analyzer."""

    mean_brightness: float
    score: float


class PoseResult(BaseModel):
    """Pose metrics from the pose analyzer."""

    yaw: float
    pitch: float
    roll: float
    score: float


class SizeResult(BaseModel):
    """Face size metrics from the size analyzer."""

    width: float
    height: float
    score: float


class VisibilityResult(BaseModel):
    """Visibility metrics from the visibility analyzer."""

    visible_ratio: float
    score: float


class AssessmentData(BaseModel):
    """Outputs from the Face Assessment stage.

    Populated by the assessment module and attached to ``Face.assessment``.
    Evaluates whether an aligned face is suitable for biometric processing.

    Each analyzer returns its own result model. ``FaceAssessor`` combines
    analyzer outputs into a single ``AssessmentData`` instance and attaches
    it to the same ``Face`` object.

    Face Assessment enriches ``Face`` only. It must not modify detection,
    alignment, or embedding data.
    """

    blur: BlurResult | None = None
    brightness: BrightnessResult | None = None
    pose: PoseResult | None = None
    size: SizeResult | None = None
    visibility: VisibilityResult | None = None

    overall_score: float | None = None
    is_acceptable: bool | None = None
