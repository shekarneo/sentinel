"""
Embedding engine configuration accessors.
"""

from functools import lru_cache
from pathlib import Path

from backend.app.config.configuration import (
    Configuration,
    EmbeddingModelSettings,
    resolve_embedding_model_path,
)


@lru_cache(maxsize=1)
def load_embedding_model_settings() -> EmbeddingModelSettings:
    """Load embedding model settings from application configuration.

    Returns:
        Validated embedding model settings.
    """
    return Configuration().load_models().embedding


def get_embedding_model_path() -> Path:
    """Resolve the configured embedding ONNX model path.

    Returns:
        Absolute path to the embedding model file.
    """
    return resolve_embedding_model_path()


def clear_embedding_config_cache() -> None:
    """Clear cached embedding configuration.

    Intended for tests that modify configuration at runtime.
    """
    load_embedding_model_settings.cache_clear()
