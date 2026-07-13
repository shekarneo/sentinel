"""Logging configuration tests."""

import logging

from backend.app.core.logging_config import configure_logging


def test_configure_logging_from_yaml() -> None:
    """Logging should initialize from configs/logging.yaml."""
    configure_logging(force=True)

    root_logger = logging.getLogger()
    assert root_logger.level == logging.INFO
    assert root_logger.handlers
