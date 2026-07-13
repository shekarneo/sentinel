"""
Application configuration.

This module is responsible for loading and validating the
application configuration from YAML files.
"""

from pathlib import Path

import yaml
from pydantic import BaseModel, model_validator

from backend.app.core.constants import (
    DEFAULT_MODELS_FILE,
    DEFAULT_PIPELINE_PROFILES_FILE,
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


class BlurThresholdSettings(BaseModel):
    warning: float
    acceptable: float
    excellent: float


class BrightnessThresholdSettings(BaseModel):
    too_dark: float
    acceptable_low: float
    acceptable_high: float
    too_bright: float


class PoseThresholdSettings(BaseModel):
    max_yaw: float
    max_pitch: float
    max_roll: float


class SizeThresholdSettings(BaseModel):
    min_face_width: float
    min_face_height: float


class VisibilityThresholdSettings(BaseModel):
    minimum_visible_ratio: float


class OverallThresholdSettings(BaseModel):
    blur_weight: float
    brightness_weight: float
    pose_weight: float
    visibility_weight: float
    size_weight: float
    minimum_acceptable_score: float

    @model_validator(mode="after")
    def validate_weights_sum_to_one(self) -> "OverallThresholdSettings":
        """Ensure analyzer weights form a valid convex combination."""
        total = (
            self.blur_weight
            + self.brightness_weight
            + self.pose_weight
            + self.visibility_weight
            + self.size_weight
        )
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"Assessment weights must sum to 1.0, got {total:.6f}."
            )
        return self


class AssessmentThresholdSettings(BaseModel):
    blur: BlurThresholdSettings
    brightness: BrightnessThresholdSettings
    pose: PoseThresholdSettings
    size: SizeThresholdSettings
    visibility: VisibilityThresholdSettings
    overall: OverallThresholdSettings


class ThresholdSettings(BaseModel):
    detection: DetectionThresholdSettings = DetectionThresholdSettings()
    assessment: AssessmentThresholdSettings


class InputSizeSettings(BaseModel):
    width: int = 640
    height: int = 640


class ScrfdModelSettings(BaseModel):
    model: str = "detection/scrfd/det_10g.onnx"
    input: InputSizeSettings = InputSizeSettings()


class DetectionModelSettings(BaseModel):
    scrfd: ScrfdModelSettings = ScrfdModelSettings()


class EmbeddingModelSettings(BaseModel):
    provider: str
    model: str
    input_size: int


class ModelSettings(BaseModel):
    detection: DetectionModelSettings = DetectionModelSettings()
    embedding: EmbeddingModelSettings


class PipelineProfileSettings(BaseModel):
    """Pipeline profile stage sequences loaded from YAML."""

    profiles: dict[str, list[str]]


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
        """Load detection and assessment thresholds from YAML."""
        return ThresholdSettings.model_validate(_load_yaml(thresholds_path))

    def load_pipeline_profiles(
        self,
        profiles_path: Path = DEFAULT_PIPELINE_PROFILES_FILE,
    ) -> PipelineProfileSettings:
        """Load pipeline profile stage sequences from YAML."""
        return PipelineProfileSettings.model_validate(_load_yaml(profiles_path))


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


def resolve_embedding_model_path(
    settings: Settings | None = None,
    model_settings: ModelSettings | None = None,
) -> Path:
    """Resolve the embedding ONNX model path from configuration."""
    configuration = Configuration()

    if settings is None:
        settings = configuration.load()

    if model_settings is None:
        model_settings = configuration.load_models()

    return (
        PROJECT_ROOT
        / settings.paths.models_dir
        / model_settings.embedding.model
    )
