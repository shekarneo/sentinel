"""Overlay rendering abstractions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from backend.app.live.models import LiveFaceOverlay, LiveFrameOverlay


class OverlayRenderer(ABC):
    """Abstract overlay renderer.

    Renderers translate recognition overlays into client-specific drawing
  instructions. ``RecognitionService`` remains unaware of rendering.
    """

    @abstractmethod
    def build_render_spec(
        self,
        overlay: LiveFrameOverlay | None,
        *,
        enabled: bool = True,
        width: int | None = None,
        height: int | None = None,
    ) -> dict[str, Any]:
        """Build a renderer-specific overlay specification."""


class CanvasOverlayRenderer(OverlayRenderer):
    """Canvas overlay renderer for the Web Console."""

    def build_render_spec(
        self,
        overlay: LiveFrameOverlay | None,
        *,
        enabled: bool = True,
        width: int | None = None,
        height: int | None = None,
    ) -> dict[str, Any]:
        if not enabled or overlay is None:
            return {
                "renderer": "canvas",
                "enabled": False,
                "faces": [],
                "width": width,
                "height": height,
            }

        return {
            "renderer": "canvas",
            "enabled": True,
            "width": width,
            "height": height,
            "profile": overlay.profile,
            "face_count": overlay.face_count,
            "pipeline_time_ms": overlay.pipeline_time_ms,
            "faces": [_face_spec(face) for face in overlay.faces],
        }


def _face_spec(face: LiveFaceOverlay) -> dict[str, Any]:
    return {
        "bounding_box": face.bounding_box,
        "confidence": face.confidence,
        "identity_id": face.identity_id,
        "similarity": face.similarity,
        "verification_decision": face.verification_decision,
        "stroke_color": "#20c997" if face.identity_id else "#3d8bfd",
        "label": " | ".join(
            part
            for part in [
                face.identity_id or "unknown",
                f"{round(face.similarity * 100)}%" if face.similarity is not None else None,
                face.verification_decision,
            ]
            if part
        ),
    }


class OpenCVOverlayRenderer(OverlayRenderer):
    """Reserved server-side OpenCV overlay renderer."""

    def build_render_spec(
        self,
        overlay: LiveFrameOverlay | None,
        *,
        enabled: bool = True,
        width: int | None = None,
        height: int | None = None,
    ) -> dict[str, Any]:
        raise NotImplementedError("OpenCVOverlayRenderer is reserved for a future release.")


class VideoOverlayRenderer(OverlayRenderer):
    """Reserved video stream overlay renderer."""

    def build_render_spec(
        self,
        overlay: LiveFrameOverlay | None,
        *,
        enabled: bool = True,
        width: int | None = None,
        height: int | None = None,
    ) -> dict[str, Any]:
        raise NotImplementedError("VideoOverlayRenderer is reserved for a future release.")
