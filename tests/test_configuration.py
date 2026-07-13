"""Configuration loading tests."""

from backend.app.config.configuration import Configuration
from backend.app.core.constants import (
    DEFAULT_LOGGING_FILE,
    DEFAULT_MODELS_FILE,
    DEFAULT_PIPELINE_PROFILES_FILE,
    DEFAULT_SETTINGS_FILE,
    DEFAULT_THRESHOLDS_FILE,
)


def test_all_configuration_files_exist() -> None:
    """Every default configuration file should be present."""
    for path in (
        DEFAULT_SETTINGS_FILE,
        DEFAULT_MODELS_FILE,
        DEFAULT_THRESHOLDS_FILE,
        DEFAULT_PIPELINE_PROFILES_FILE,
        DEFAULT_LOGGING_FILE,
    ):
        assert path.exists(), f"Missing configuration file: {path}"


def test_configuration_models_load() -> None:
    """Configuration loader should validate all YAML groups."""
    configuration = Configuration()

    settings = configuration.load()
    models = configuration.load_models()
    thresholds = configuration.load_thresholds()
    profiles = configuration.load_pipeline_profiles().profiles

    assert settings.app.name == "Sentinel"
    assert models.embedding.provider == "arcface"
    assert models.search.provider == "faiss"
    assert thresholds.verification.similarity_threshold == 0.45
    assert "kyc" in profiles
    assert profiles["kyc"].search.top_k == 5
    assert "search" in profiles["kyc"].stages
