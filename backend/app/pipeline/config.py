"""
Pipeline configuration accessors.

Profile stage sequences are loaded from ``configs/pipeline_profiles.yaml``.
"""

from functools import lru_cache

from backend.app.config.configuration import Configuration, PipelineProfileSettings


@lru_cache(maxsize=1)
def load_pipeline_profile_settings() -> PipelineProfileSettings:
    """Load pipeline profile definitions from application configuration.

    Returns:
        Validated pipeline profile settings.
    """
    return Configuration().load_pipeline_profiles()


def clear_pipeline_profile_cache() -> None:
    """Clear cached pipeline profile settings.

    Intended for tests that modify configuration at runtime.
    """
    load_pipeline_profile_settings.cache_clear()
