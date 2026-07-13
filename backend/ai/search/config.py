"""
Search engine configuration accessors.
"""

from functools import lru_cache
from pathlib import Path

from backend.app.config.configuration import (
    Configuration,
    SearchModelSettings,
    resolve_search_index_path,
)


@lru_cache(maxsize=1)
def load_search_model_settings() -> SearchModelSettings:
    """Load search index settings from application configuration.

    Returns:
        Validated search model settings.
    """
    return Configuration().load_models().search


def get_search_index_path() -> Path:
    """Resolve the configured search index path.

    Returns:
        Absolute path to the search index file.
    """
    return resolve_search_index_path()


def get_search_mapping_path() -> Path:
    """Resolve the identity mapping path for the configured search index.

    Returns:
        Absolute path to the gallery identity mapping file.
    """
    index_path = get_search_index_path()
    return index_path.with_suffix(f"{index_path.suffix}.mapping.json")


def clear_search_config_cache() -> None:
    """Clear cached search configuration.

    Intended for tests that modify configuration at runtime.
    """
    load_search_model_settings.cache_clear()
