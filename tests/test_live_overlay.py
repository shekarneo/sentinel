"""Overlay renderer tests."""

import pytest

from backend.app.live.models import LiveFaceOverlay, LiveFrameOverlay
from backend.app.live.overlay import (
    CanvasOverlayRenderer,
    OpenCVOverlayRenderer,
    VideoOverlayRenderer,
)


def test_canvas_overlay_renderer_builds_spec() -> None:
    overlay = LiveFrameOverlay(
        profile="surveillance",
        face_count=1,
        pipeline_time_ms=12.5,
        faces=[
            LiveFaceOverlay(
                bounding_box={"x": 1, "y": 2, "width": 3, "height": 4},
                confidence=0.9,
                identity_id="user-1",
                similarity=0.88,
                verification_decision="accept",
            )
        ],
    )
    spec = CanvasOverlayRenderer().build_render_spec(overlay, enabled=True, width=640, height=480)
    assert spec["renderer"] == "canvas"
    assert spec["enabled"] is True
    assert len(spec["faces"]) == 1
    assert "user-1" in spec["faces"][0]["label"]


def test_reserved_renderers_raise() -> None:
    with pytest.raises(NotImplementedError):
        OpenCVOverlayRenderer().build_render_spec(None)
    with pytest.raises(NotImplementedError):
        VideoOverlayRenderer().build_render_spec(None)
