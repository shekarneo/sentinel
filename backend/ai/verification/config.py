"""
Verification engine configuration accessors.

Thresholds are loaded from ``configs/thresholds.yaml``.
"""

from functools import lru_cache

from backend.app.config.configuration import (
    Configuration,
    VerificationThresholdSettings,
)


@lru_cache(maxsize=1)
def load_verification_thresholds() -> VerificationThresholdSettings:
    """Load verification thresholds from application configuration.

    Returns:
        Validated verification threshold settings.
    """
    return Configuration().load_thresholds().verification


def clear_verification_config_cache() -> None:
    """Clear cached verification configuration.

    Intended for tests that modify configuration at runtime.
    """
    load_verification_thresholds.cache_clear()
