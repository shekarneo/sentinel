"""Shared bootstrap helpers for development scripts."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.config.configuration import Configuration, resolve_scrfd_model_path

SCRFD_MODEL_PATH = resolve_scrfd_model_path()
