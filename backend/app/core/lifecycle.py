"""Application startup and shutdown lifecycle hooks."""

import logging

from backend.app.config.configuration import Configuration
from backend.app.core.logging_config import configure_logging

logger = logging.getLogger(__name__)


def startup_app() -> None:
    """Initialize logging and emit application startup information."""
    configure_logging()
    settings = Configuration().load()

    logger.info("=" * 60)
    logger.info("%s", settings.app.name)
    logger.info("Version      : %s", settings.app.version)
    logger.info("Environment  : %s", settings.app.environment)
    logger.info("Device       : %s", settings.runtime.device)
    logger.info("Logs Dir     : %s", settings.paths.logs_dir)
    logger.info("=" * 60)
    logger.info("System Status : READY")


def shutdown_app() -> None:
    """Release application resources during shutdown."""
    logger.info("Application shutdown complete.")
