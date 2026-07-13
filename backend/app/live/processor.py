"""Shared live frame processing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from backend.app.live.models import LiveFrameOverlay, LivePerformanceMetrics
from backend.app.live.manager import LiveCameraManager
from backend.app.live.overlay import CanvasOverlayRenderer, OverlayRenderer


@dataclass(frozen=True)
class LiveFrameResult:
    """Domain result for one processed live frame."""

    session_id: str
    overlay: LiveFrameOverlay | None
    dropped: bool
    metrics: LivePerformanceMetrics
    render_spec: dict[str, Any]


def process_live_frame(
    live_manager: LiveCameraManager,
    *,
    session_id: str,
    image_bytes: bytes,
    force: bool = False,
    renderer: OverlayRenderer | None = None,
) -> LiveFrameResult:
    """Process one frame and build overlay render artifacts."""
    session = live_manager.status(session_id)
    if session is None:
        raise ValueError(f"Live session {session_id!r} was not found.")

    before_dropped = session.dropped_frames
    overlay = live_manager.process_frame_bytes(
        image_bytes,
        session_id=session_id,
        force=force,
    )
    updated = live_manager.status(session_id)
    if updated is None:
        raise ValueError(f"Live session {session_id!r} was not found.")

    dropped = (
        updated.dropped_frames > before_dropped
        and overlay == updated.last_overlay
        and not force
    )
    resolved_renderer = renderer or CanvasOverlayRenderer()
    render_spec = resolved_renderer.build_render_spec(
        overlay,
        enabled=updated.config.overlay_enabled,
        width=updated.config.resolution_width,
        height=updated.config.resolution_height,
    )
    return LiveFrameResult(
        session_id=session_id,
        overlay=overlay,
        dropped=dropped,
        metrics=updated.metrics,
        render_spec=render_spec,
    )
