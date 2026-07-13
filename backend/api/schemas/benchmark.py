"""Benchmark API schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field


class BenchmarkRequest(BaseModel):
    """Benchmark request contract."""

    profile: str = Field(default="search")
    iterations: int = Field(default=1, ge=1, le=1000)


class BenchmarkStageTimings(BaseModel):
    """Per-stage benchmark timings in milliseconds."""

    detection_time_ms: float = 0.0
    alignment_time_ms: float = 0.0
    assessment_time_ms: float = 0.0
    embedding_time_ms: float = 0.0
    search_time_ms: float = 0.0
    verification_time_ms: float = 0.0
    total_pipeline_time_ms: float = 0.0


class BenchmarkResponse(BaseModel):
    """Benchmark response contract."""

    profile: str
    iterations: int
    timings: BenchmarkStageTimings
    message: str = "Benchmark completed."
