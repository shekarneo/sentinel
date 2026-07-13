"""Live camera module tests."""

import numpy as np
import pytest

from backend.app.live.config import LiveCameraConfig
from backend.app.live.manager import LiveCameraManager
from backend.app.live.models import SessionStatus
from backend.app.pipeline.profile import PipelineProfile


class _StubRecognitionService:
    def recognize(self, image: np.ndarray, profile: PipelineProfile):
        return type(
            "Context",
            (),
            {
                "profile": profile,
                "faces": [],
                "timings": {"total": 0.01},
                "metadata": {},
            },
        )()

    @staticmethod
    def decode_image(image_bytes: bytes) -> np.ndarray:
        return np.zeros((48, 64, 3), dtype=np.uint8)


def test_live_session_lifecycle() -> None:
    manager = LiveCameraManager(_StubRecognitionService())
    session = manager.start_session(LiveCameraConfig(pipeline_profile="search"))
    assert session.status == SessionStatus.RUNNING

    paused = manager.pause_session(session.id)
    assert paused is not None
    assert paused.status == SessionStatus.PAUSED

    resumed = manager.resume_session(session.id)
    assert resumed is not None
    assert resumed.status == SessionStatus.RUNNING

    stopped = manager.stop_session(session.id)
    assert stopped is not None
    assert stopped.status == SessionStatus.STOPPED


def test_live_frame_processing_updates_metrics() -> None:
    manager = LiveCameraManager(_StubRecognitionService())
    session = manager.start_session()
    image = np.zeros((48, 64, 3), dtype=np.uint8)

    overlay = manager.process_frame(image, session_id=session.id, force=True)
    assert overlay is not None
    updated = manager.status(session.id)
    assert updated is not None
    assert updated.processed_frames == 1
    assert updated.metrics.pipeline_latency_ms > 0


def test_live_start_requires_single_active_session() -> None:
    manager = LiveCameraManager(_StubRecognitionService())
    manager.start_session()
    with pytest.raises(ValueError, match="already active"):
        manager.start_session()
