"""
Assessment configuration accessors.

Thresholds and scoring weights are loaded from ``configs/thresholds.yaml``.
Implementation modules must not define production threshold defaults.
"""

from functools import lru_cache

from backend.app.config.configuration import (
    AssessmentThresholdSettings,
    Configuration,
)


@lru_cache(maxsize=1)
def load_assessment_thresholds() -> AssessmentThresholdSettings:
    """Load face assessment thresholds from application configuration.

    Returns:
        Validated assessment threshold settings.
    """
    return Configuration().load_thresholds().assessment


def clear_assessment_threshold_cache() -> None:
    """Clear the cached assessment threshold settings.

    Intended for tests that modify configuration at runtime.
    """
    load_assessment_thresholds.cache_clear()
