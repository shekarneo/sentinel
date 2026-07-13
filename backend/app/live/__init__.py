"""Live camera package."""

from backend.app.live.config import LiveCameraConfig, LiveExtensionHooks
from backend.app.live.manager import LiveCameraManager
from backend.app.live.models import (
    LiveFaceOverlay,
    LiveFrameOverlay,
    LivePerformanceMetrics,
    LiveSession,
    LiveTimelineEntry,
    SessionStatus,
)
from backend.app.live.overlay import (
    CanvasOverlayRenderer,
    OpenCVOverlayRenderer,
    OverlayRenderer,
    VideoOverlayRenderer,
)
from backend.app.live.policy import RecognitionPolicy, RecognitionPolicyType
from backend.app.live.recording import NoOpRecordingHook, RecordingHook
from backend.app.live.stream import (
    BrowserWebcamStream,
    LiveStream,
    OpenCVCameraStream,
    RTSPStream,
    USBStream,
    VideoFileStream,
)

__all__ = [
    "BrowserWebcamStream",
    "CanvasOverlayRenderer",
    "LiveCameraConfig",
    "LiveCameraManager",
    "LiveExtensionHooks",
    "LiveFaceOverlay",
    "LiveFrameOverlay",
    "LivePerformanceMetrics",
    "LiveSession",
    "LiveStream",
    "LiveTimelineEntry",
    "NoOpRecordingHook",
    "OpenCVCameraStream",
    "OpenCVOverlayRenderer",
    "OverlayRenderer",
    "RTSPStream",
    "RecognitionPolicy",
    "RecognitionPolicyType",
    "RecordingHook",
    "SessionStatus",
    "USBStream",
    "VideoFileStream",
    "VideoOverlayRenderer",
]
