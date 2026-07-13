"""Application repositories."""

from backend.app.repositories.identity_repository import IdentityRepository
from backend.app.repositories.json_identity_repository import JsonIdentityRepository

__all__ = [
    "IdentityRepository",
    "JsonIdentityRepository",
]
