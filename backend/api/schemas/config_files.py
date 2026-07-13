"""Configuration file API schemas."""

from __future__ import annotations

from pydantic import BaseModel


class ConfigurationFilesResponse(BaseModel):
    """Raw configuration file contents for read-only console display."""

    models_yaml: str
    thresholds_yaml: str
    pipeline_profiles_yaml: str
