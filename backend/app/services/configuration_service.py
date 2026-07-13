"""Configuration application service."""

from __future__ import annotations

from backend.app.config.configuration import (
    Configuration,
    PipelineProfileDefinition,
    Settings,
)
from backend.app.pipeline.config import load_pipeline_profile_settings


class ConfigurationService:
    """Read-only access to validated application configuration."""

    def get_settings(self) -> Settings:
        """Return validated application settings."""
        return Configuration().load()

    def list_pipeline_profiles(self) -> dict[str, PipelineProfileDefinition]:
        """Return configured pipeline profiles."""
        return load_pipeline_profile_settings().profiles

    def read_config_files(self) -> dict[str, str]:
        """Return raw YAML configuration file contents."""
        from backend.app.core.constants import (
            DEFAULT_MODELS_FILE,
            DEFAULT_PIPELINE_PROFILES_FILE,
            DEFAULT_THRESHOLDS_FILE,
        )

        return {
            "models_yaml": DEFAULT_MODELS_FILE.read_text(encoding="utf-8"),
            "thresholds_yaml": DEFAULT_THRESHOLDS_FILE.read_text(encoding="utf-8"),
            "pipeline_profiles_yaml": DEFAULT_PIPELINE_PROFILES_FILE.read_text(
                encoding="utf-8"
            ),
        }

    def list_models(self) -> list[dict[str, str]]:
        """Return configured model paths for dashboard display."""
        models = Configuration().load_models()
        return [
            {
                "name": "SCRFD Detection",
                "path": models.detection.scrfd.model,
                "status": "configured",
            },
            {
                "name": "ArcFace Embedding",
                "path": models.embedding.model,
                "status": "configured",
            },
            {
                "name": f"Search ({models.search.provider})",
                "path": models.search.index_path,
                "status": "configured",
            },
        ]
