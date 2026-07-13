"""Live camera API schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class LiveExtensionHooksResponse(BaseModel):
    """Reserved future extension flags."""

    tracking_enabled: bool = False
    multi_camera_enabled: bool = False
    rtsp_url: str | None = None
    video_recording_enabled: bool = False
    fraud_detection_enabled: bool = False


class LiveRecognitionPolicyRequest(BaseModel):
    """Recognition scheduling policy."""

    policy_type: str = Field(
        default="every_n_frames",
        description="every_frame | every_n_frames | adaptive | tracker_assisted | motion_triggered",
    )
    frame_interval: int = Field(default=3, ge=1)


class LiveConfigRequest(BaseModel):
    """Live session configuration."""

    camera_index: int = Field(default=0, ge=0)
    resolution_width: int = Field(default=640, ge=160)
    resolution_height: int = Field(default=480, ge=120)
    target_fps: int = Field(default=15, ge=1, le=60)
    pipeline_profile: str = "surveillance"
    overlay_enabled: bool = True
    submission_interval_ms: int = Field(default=500, ge=50)
    recognition_policy: LiveRecognitionPolicyRequest = Field(
        default_factory=LiveRecognitionPolicyRequest
    )
    recognition_interval_ms: int | None = Field(
        default=None,
        ge=50,
        description="Deprecated. Use submission_interval_ms.",
    )
    frame_skip: int | None = Field(
        default=None,
        ge=0,
        description="Deprecated. Use recognition_policy.frame_interval.",
    )
    extensions: LiveExtensionHooksResponse = Field(default_factory=LiveExtensionHooksResponse)


class LivePerformanceResponse(BaseModel):
    """Performance metrics for a live session."""

    capture_fps: float = 0.0
    processing_fps: float = 0.0
    pipeline_latency_ms: float = 0.0
    average_latency_ms: float = 0.0
    dropped_frames: int = 0
    recognition_rate: float = 0.0


class LiveFaceOverlayResponse(BaseModel):
    """Overlay data for one detected face."""

    bounding_box: dict[str, float]
    confidence: float
    identity_id: str | None = None
    similarity: float | None = None
    verification_decision: str | None = None


class LiveTimelineEntryResponse(BaseModel):
    """Recognition timeline entry."""

    timestamp: datetime
    face_count: int
    pipeline_time_ms: float
    recognized_identities: list[str] = Field(default_factory=list)


class LiveSessionResponse(BaseModel):
    """Live session status response."""

    id: str
    started_at: datetime
    updated_at: datetime
    status: str
    pipeline_profile: str
    fps: float
    processed_frames: int
    recognized_faces: int
    dropped_frames: int
    config: LiveConfigRequest
    metrics: LivePerformanceResponse
    timeline: list[LiveTimelineEntryResponse] = Field(default_factory=list)
    last_overlay: dict[str, Any] | None = None
    message: str = "Live camera session scaffold."


class LiveStartRequest(BaseModel):
    """Request to start a live session."""

    config: LiveConfigRequest = Field(default_factory=LiveConfigRequest)


class LiveStopRequest(BaseModel):
    """Request to stop a live session."""

    session_id: str | None = None


class LiveStatusResponse(BaseModel):
    """Active session status wrapper."""

    active: bool
    session: LiveSessionResponse | None = None


class LiveSessionListResponse(BaseModel):
    """List of live sessions."""

    count: int
    sessions: list[LiveSessionResponse]


class LiveFrameProcessResponse(BaseModel):
    """Frame processing response for console overlays."""

    session_id: str
    overlay: dict[str, Any] | None = None
    metrics: LivePerformanceResponse
    dropped: bool = False
    render_spec: dict[str, Any] | None = None
