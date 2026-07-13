"""
Application logging configuration.

Loads logging defaults from ``configs/logging.yaml`` and ensures the
configured logs directory exists.
"""

from __future__ import annotations

import logging
from pathlib import Path

import yaml

from backend.app.config.configuration import Configuration
from backend.app.core.constants import DEFAULT_LOGGING_FILE, PROJECT_ROOT

_CONFIGURED = False


def configure_logging(
    logging_path: Path = DEFAULT_LOGGING_FILE,
    *,
    force: bool = False,
) -> None:
    """Configure root logging from YAML settings.

    Args:
        logging_path: Path to the logging configuration file.
        force: Reconfigure logging even if already configured.
    """
    global _CONFIGURED

    if _CONFIGURED and not force:
        return

    if not logging_path.exists():
        raise FileNotFoundError(f"Logging configuration not found: {logging_path}")

    with logging_path.open("r", encoding="utf-8") as file:
        payload = yaml.safe_load(file) or {}

    logging_settings = payload.get("logging", {})
    level_name = str(logging_settings.get("level", "INFO")).upper()
    log_format = logging_settings.get(
        "format",
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    settings = Configuration().load()
    logs_dir = PROJECT_ROOT / settings.paths.logs_dir
    logs_dir.mkdir(parents=True, exist_ok=True)

    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(level=level, format=log_format, force=True)
    _CONFIGURED = True
