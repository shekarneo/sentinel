"""
Application configuration.

This module is responsible for loading and validating the
application configuration from YAML files.
"""

from pathlib import Path

import yaml
from pydantic import BaseModel

from backend.app.core.constants import (
    DEFAULT_MODELS_FILE,
    DEFAULT_SETTINGS_FILE,
    DEFAULT_THRESHOLDS_FILE,
    PROJECT_ROOT,
)


# -------------------------------------------------------------------------
# Configuration Models
# -------------------------------------------------------------------------


class AppSettings(BaseModel):
    name: str
    version: str
    environment: str
    debug: bool


class RuntimeSettings(BaseModel):
    device: str


class PathSettings(BaseModel):
    models_dir: str
    logs_dir: str
    data_dir: str


class Settings(BaseModel):
    app: AppSettings
    runtime: RuntimeSettings
    paths: PathSettings


class ScrfdThresholdSettings(BaseModel):
    score_threshold: float = 0.5
    nms_iou_threshold: float = 0.4


class DetectionThresholdSettings(BaseModel):
    scrfd: ScrfdThresholdSettings = ScrfdThresholdSettings()


class ThresholdSettings(BaseModel):
    detection: DetectionThresholdSettings = DetectionThresholdSettings()


class InputSizeSettings(BaseModel):
    width: int = 640
    height: int = 640


class ScrfdModelSettings(BaseModel):
    model: str = "detection/scrfd/det_10g.onnx"
    input: InputSizeSettings = InputSizeSettings()


class DetectionModelSettings(BaseModel):
    scrfd: ScrfdModelSettings = ScrfdModelSettings()


class ModelSettings(BaseModel):
    detection: DetectionModelSettings = DetectionModelSettings()


# -------------------------------------------------------------------------
# Configuration Loader
# -------------------------------------------------------------------------


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    return data or {}


class Configuration:
    """Loads and validates application configuration."""

    def __init__(self, config_path: Path = DEFAULT_SETTINGS_FILE):
        self.config_path = config_path

    def load(self) -> Settings:
        """Load application settings from YAML."""
        return Settings.model_validate(_load_yaml(self.config_path))

    def load_models(self, models_path: Path = DEFAULT_MODELS_FILE) -> ModelSettings:
        """Load model path settings from YAML."""
        return ModelSettings.model_validate(_load_yaml(models_path))

    def load_thresholds(
        self,
        thresholds_path: Path = DEFAULT_THRESHOLDS_FILE,
    ) -> ThresholdSettings:
        """Load detection thresholds from YAML."""
        return ThresholdSettings.model_validate(_load_yaml(thresholds_path))


def resolve_scrfd_model_path(
    settings: Settings | None = None,
    model_settings: ModelSettings | None = None,
) -> Path:
    """Resolve the SCRFD ONNX model path from configuration."""
    configuration = Configuration()

    if settings is None:
        settings = configuration.load()

    if model_settings is None:
        model_settings = configuration.load_models()

    return (
        PROJECT_ROOT
        / settings.paths.models_dir
        / model_settings.detection.scrfd.model
    )
