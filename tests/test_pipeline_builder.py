"""Pipeline builder dependency validation tests."""

import pytest

from backend.app.pipeline import PipelineBuilder, PipelineProfile, create_default_registry


@pytest.fixture
def builder() -> PipelineBuilder:
    """Return a builder backed by the default stage registry."""
    return PipelineBuilder(create_default_registry())


@pytest.mark.parametrize(
    "profile",
    [
        PipelineProfile.ENROLLMENT,
        PipelineProfile.ATTENDANCE,
        PipelineProfile.SURVEILLANCE,
        PipelineProfile.KYC,
        PipelineProfile.ACCESS_CONTROL,
        PipelineProfile.SEARCH,
    ],
)
def test_builtin_profiles_construct(builder: PipelineBuilder, profile: PipelineProfile) -> None:
    """Built-in profiles should satisfy stage dependency validation."""
    pipeline = builder.build(profile)
    assert pipeline.stages


def test_verification_requires_search(builder: PipelineBuilder) -> None:
    """Verification without search should fail during construction."""
    with pytest.raises(ValueError, match="requires \\['search'\\]"):
        builder.build(
            PipelineProfile.CUSTOM,
            custom_stages=["scrfd", "alignment", "embedding", "verification"],
        )


def test_search_requires_embedding(builder: PipelineBuilder) -> None:
    """Search before embedding should fail during construction."""
    with pytest.raises(ValueError, match="requires \\['embedding'\\]"):
        builder.build(
            PipelineProfile.CUSTOM,
            custom_stages=["scrfd", "alignment", "search", "embedding"],
        )
