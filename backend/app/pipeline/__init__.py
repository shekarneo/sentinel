"""
Biometric pipeline orchestration layer.

AI modules never invoke one another directly. Execution order and use-case
selection are owned by this pipeline layer.
"""

from backend.app.pipeline.builder import PipelineBuilder
from backend.app.pipeline.config import (
    clear_pipeline_profile_cache,
    load_pipeline_profile_settings,
)
from backend.app.pipeline.context import PipelineContext
from backend.app.pipeline.executor import PipelineExecutor
from backend.app.pipeline.pipeline import Pipeline
from backend.app.pipeline.metadata import StageMetadata
from backend.app.pipeline.profile import PipelineProfile
from backend.app.pipeline.registry import PipelineRegistry, create_default_registry
from backend.app.pipeline.stage import PipelineStage

__all__ = [
    "Pipeline",
    "PipelineBuilder",
    "PipelineContext",
    "PipelineExecutor",
    "PipelineProfile",
    "PipelineRegistry",
    "PipelineStage",
    "StageMetadata",
    "clear_pipeline_profile_cache",
    "create_default_registry",
    "load_pipeline_profile_settings",
]
