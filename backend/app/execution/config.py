"""Execution engine configuration."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ExecutionEngineConfig(BaseModel):
    """Runtime configuration for the execution engine.

    The engine is scaffolded only. Worker execution remains disabled until a
    future release enables background processing.
    """

    enabled: bool = False
    worker_enabled: bool = False
    max_queue_size: int = Field(default=1000, ge=1)
    poll_interval_ms: int = Field(default=250, ge=1)
    default_priority: str = "normal"
