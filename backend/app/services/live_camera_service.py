"""Live camera application service."""

from __future__ import annotations

import numpy as np

from backend.app.live.config import LiveCameraConfig
from backend.app.live.manager import LiveCameraManager
from backend.app.live.models import LiveFrameOverlay, LiveSession
from backend.app.services.recognition_service import RecognitionService


class LiveCameraService:
    """Application facade for live camera sessions and frame processing."""

    def __init__(
        self,
        manager: LiveCameraManager | None = None,
        *,
        recognition: RecognitionService | None = None,
    ) -> None:
        if manager is None and recognition is None:
            raise ValueError("Either manager or recognition service must be provided.")
        self._manager = manager or LiveCameraManager(recognition)  # type: ignore[arg-type]

    @property
    def manager(self) -> LiveCameraManager:
        """Return the underlying live camera manager."""
        return self._manager

    def start_session(self, config: LiveCameraConfig | None = None) -> LiveSession:
        """Start a live camera session."""
        return self._manager.start_session(config)

    def stop_session(self, session_id: str | None = None) -> LiveSession | None:
        """Stop a live camera session."""
        return self._manager.stop_session(session_id)

    def pause_session(self, session_id: str | None = None) -> LiveSession | None:
        """Pause a live camera session."""
        return self._manager.pause_session(session_id)

    def resume_session(self, session_id: str | None = None) -> LiveSession | None:
        """Resume a paused live camera session."""
        return self._manager.resume_session(session_id)

    def process_frame(
        self,
        image: np.ndarray,
        *,
        session_id: str | None = None,
        force: bool = False,
    ) -> LiveFrameOverlay | None:
        """Process one frame through the recognition pipeline."""
        return self._manager.process_frame(image, session_id=session_id, force=force)

    def process_frame_bytes(
        self,
        image_bytes: bytes,
        *,
        session_id: str | None = None,
        force: bool = False,
    ) -> LiveFrameOverlay | None:
        """Decode and process one frame."""
        return self._manager.process_frame_bytes(
            image_bytes,
            session_id=session_id,
            force=force,
        )

    def status(self, session_id: str | None = None) -> LiveSession | None:
        """Return session status."""
        return self._manager.status(session_id)

    def list_sessions(self) -> list[LiveSession]:
        """Return all sessions."""
        return self._manager.list_sessions()
