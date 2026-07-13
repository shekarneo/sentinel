"""Live camera configuration."""

from __future__ import annotations

from pydantic import BaseModel, Field

from backend.app.live.policy import RecognitionPolicy


class LiveExtensionHooks(BaseModel):
    """Reserved extension points for future live camera capabilities."""

    tracking_enabled: bool = False
    multi_camera_enabled: bool = False
    rtsp_url: str | None = None
    video_recording_enabled: bool = False
    fraud_detection_enabled: bool = False


class LiveCameraConfig(BaseModel):
    """Runtime configuration for a live camera session."""

    camera_index: int = Field(default=0, ge=0)
    resolution_width: int = Field(default=640, ge=160)
    resolution_height: int = Field(default=480, ge=120)
    target_fps: int = Field(default=15, ge=1, le=60)
    pipeline_profile: str = "surveillance"
    overlay_enabled: bool = True
    submission_interval_ms: int = Field(
        default=500,
        ge=50,
        description="Client frame submission interval in milliseconds.",
    )
    recognition_policy: RecognitionPolicy = Field(default_factory=RecognitionPolicy)
    extensions: LiveExtensionHooks = Field(default_factory=LiveExtensionHooks)
