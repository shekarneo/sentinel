"""REST API package for the Sentinel Biometric Platform."""

from backend.api.main import app, create_app
from backend.api.version import API_V1_PREFIX, CURRENT_API_VERSION, SUPPORTED_API_VERSIONS

__all__ = [
    "API_V1_PREFIX",
    "CURRENT_API_VERSION",
    "SUPPORTED_API_VERSIONS",
    "app",
    "create_app",
]
