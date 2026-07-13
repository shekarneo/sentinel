"""
Overall assessment scoring utilities.
"""

import numpy as np

from backend.ai.assessment.config import load_assessment_thresholds
from backend.app.domain.assessment import AssessmentData


def compute_overall_score(assessment: AssessmentData) -> AssessmentData:
    """Combine analyzer metrics into an overall quality decision.

    Scoring weights and the minimum acceptable score are loaded from
    ``configs/thresholds.yaml``.

    Args:
        assessment: Combined analyzer results for a single face.

    Returns:
        Partial ``AssessmentData`` containing ``overall_score`` and
        ``is_acceptable`` only.

    Raises:
        ValueError: If any analyzer result is missing.
    """
    if assessment.blur is None:
        raise ValueError("Blur result is required for overall scoring.")
    if assessment.brightness is None:
        raise ValueError("Brightness result is required for overall scoring.")
    if assessment.pose is None:
        raise ValueError("Pose result is required for overall scoring.")
    if assessment.visibility is None:
        raise ValueError("Visibility result is required for overall scoring.")
    if assessment.size is None:
        raise ValueError("Size result is required for overall scoring.")

    overall = load_assessment_thresholds().overall
    overall_score = (
        overall.blur_weight * assessment.blur.score
        + overall.brightness_weight * assessment.brightness.score
        + overall.pose_weight * assessment.pose.score
        + overall.visibility_weight * assessment.visibility.score
        + overall.size_weight * assessment.size.score
    )
    overall_score = float(np.clip(overall_score, 0.0, 1.0))
    is_acceptable = overall_score >= overall.minimum_acceptable_score

    return AssessmentData(
        overall_score=overall_score,
        is_acceptable=is_acceptable,
    )
