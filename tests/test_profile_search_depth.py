"""Pipeline profile search depth tests."""

import pytest

from backend.app.pipeline import PipelineBuilder, PipelineProfile, create_default_registry
from backend.app.pipeline.config import clear_pipeline_profile_cache, load_pipeline_profile_settings


@pytest.fixture(autouse=True)
def _clear_profile_cache() -> None:
    """Ensure profile settings are reloaded for each test."""
    clear_pipeline_profile_cache()


@pytest.fixture
def builder() -> PipelineBuilder:
    """Return a builder backed by the default stage registry."""
    return PipelineBuilder(create_default_registry())


@pytest.mark.parametrize(
    ("profile", "expected_top_k"),
    [
        (PipelineProfile.SURVEILLANCE, 1),
        (PipelineProfile.ATTENDANCE, 5),
        (PipelineProfile.ACCESS_CONTROL, 3),
        (PipelineProfile.KYC, 5),
        (PipelineProfile.SEARCH, 1),
    ],
)
def test_profile_search_top_k(
    builder: PipelineBuilder,
    profile: PipelineProfile,
    expected_top_k: int,
) -> None:
    """Built-in profiles should expose configured search depth."""
    pipeline = builder.build(profile)

    assert pipeline.search is not None
    assert pipeline.search.enabled is True
    assert pipeline.search.top_k == expected_top_k


def test_enrollment_search_disabled(builder: PipelineBuilder) -> None:
    """Enrollment should disable profile search."""
    pipeline = builder.build(PipelineProfile.ENROLLMENT)

    assert pipeline.search is not None
    assert pipeline.search.enabled is False
    assert pipeline.search.top_k is None


def test_profile_settings_loaded_from_yaml() -> None:
    """YAML profile definitions should expose stages and search policy."""
    profiles = load_pipeline_profile_settings().profiles

    assert profiles["attendance"].stages[-1] == "search"
    assert profiles["attendance"].search.top_k == 5
    assert profiles["surveillance"].search.top_k == 1


def test_invalid_top_k_rejected() -> None:
    """Search depth below 1 should fail validation."""
    from backend.app.config.configuration import PipelineProfileSearchSettings

    with pytest.raises(ValueError, match="top_k must be at least 1"):
        PipelineProfileSearchSettings(enabled=True, top_k=0)
