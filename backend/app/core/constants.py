from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]

CONFIG_DIR = PROJECT_ROOT / "configs"

DEFAULT_SETTINGS_FILE = CONFIG_DIR / "settings.yaml"
DEFAULT_MODELS_FILE = CONFIG_DIR / "models.yaml"
DEFAULT_THRESHOLDS_FILE = CONFIG_DIR / "thresholds.yaml"
DEFAULT_PIPELINE_PROFILES_FILE = CONFIG_DIR / "pipeline_profiles.yaml"
DEFAULT_LOGGING_FILE = CONFIG_DIR / "logging.yaml"
