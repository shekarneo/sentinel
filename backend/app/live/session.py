"""Live camera session helpers."""

from __future__ import annotations

from backend.app.live.config import LiveCameraConfig
from backend.app.live.models import LivePerformanceMetrics, LiveSession, SessionStatus
from backend.app.live.policy import RecognitionPolicyType
from backend.app.live.utils import LatencyTracker, create_session_id, utc_now


class LiveSessionState:
    """Mutable runtime state for an active live session."""

    def __init__(self, config: LiveCameraConfig) -> None:
        now = utc_now()
        self.session = LiveSession(
            id=create_session_id(),
            started_at=now,
            updated_at=now,
            status=SessionStatus.RUNNING,
            pipeline_profile=config.pipeline_profile,
            config=config,
            metrics=LivePerformanceMetrics(),
        )
        self._latency = LatencyTracker()
        self._capture_frame_count = 0
        self._capture_started_at = now

    def touch(self) -> None:
        self.session.updated_at = utc_now()

    def update_capture_fps(self) -> None:
        elapsed = max((utc_now() - self._capture_started_at).total_seconds(), 0.001)
        self.session.metrics.capture_fps = round(self._capture_frame_count / elapsed, 2)
        self.session.fps = self.session.metrics.capture_fps

    def record_capture_frame(self) -> None:
        self._capture_frame_count += 1
        self.update_capture_fps()

    def should_process_frame(self, *, force: bool = False) -> bool:
        policy = self.session.config.recognition_policy
        if policy.policy_type in {
            RecognitionPolicyType.ADAPTIVE,
            RecognitionPolicyType.TRACKER_ASSISTED,
            RecognitionPolicyType.MOTION_TRIGGERED,
        }:
            raise NotImplementedError(
                f"Recognition policy {policy.policy_type.value!r} is reserved for a future release."
            )
        return policy.should_recognize(self._capture_frame_count, force=force)

    def record_drop(self) -> None:
        self.session.dropped_frames += 1
        self.session.metrics.dropped_frames = self.session.dropped_frames

    def record_processing(self, *, latency_ms: float, recognized_faces: int) -> None:
        self._latency.record(latency_ms)
        self.session.processed_frames += 1
        self.session.recognized_faces += recognized_faces
        self.session.metrics.pipeline_latency_ms = latency_ms
        self.session.metrics.average_latency_ms = self._latency.average_ms
        self.session.metrics.processing_fps = self._latency.processing_fps()
        processed = max(self.session.processed_frames, 1)
        self.session.metrics.recognition_rate = round(
            self.session.recognized_faces / processed,
            3,
        )
