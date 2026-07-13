"""Live camera domain models."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from backend.app.live.config import LiveCameraConfig


class SessionStatus(str, Enum):
    """Lifecycle state for a live camera session."""

    STOPPED = "stopped"
    RUNNING = "running"
    PAUSED = "paused"


class LivePerformanceMetrics(BaseModel):
    """Performance counters for a live session."""

    capture_fps: float = 0.0
    processing_fps: float = 0.0
    pipeline_latency_ms: float = 0.0
    average_latency_ms: float = 0.0
    dropped_frames: int = 0
    recognition_rate: float = 0.0


class LiveFaceOverlay(BaseModel):
    """Overlay data for a single detected face."""

    bounding_box: dict[str, float]
    confidence: float
    identity_id: str | None = None
    similarity: float | None = None
    verification_decision: str | None = None


class LiveFrameOverlay(BaseModel):
    """Recognition overlay for one processed frame."""

    faces: list[LiveFaceOverlay] = Field(default_factory=list)
    pipeline_time_ms: float = 0.0
    profile: str = ""
    face_count: int = 0


class LiveTimelineEntry(BaseModel):
    """Recognition timeline event for the console."""

    timestamp: datetime
    face_count: int
    pipeline_time_ms: float
    recognized_identities: list[str] = Field(default_factory=list)


class LiveSession(BaseModel):
    """A live camera recognition session."""

    id: str
    started_at: datetime
    updated_at: datetime
    status: SessionStatus
    pipeline_profile: str
    fps: float = 0.0
    processed_frames: int = 0
    recognized_faces: int = 0
    dropped_frames: int = 0
    config: LiveCameraConfig
    metrics: LivePerformanceMetrics = Field(default_factory=LivePerformanceMetrics)
    timeline: list[LiveTimelineEntry] = Field(default_factory=list)
    last_overlay: LiveFrameOverlay | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
