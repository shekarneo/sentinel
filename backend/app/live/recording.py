"""Recording extension hooks."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from backend.app.live.models import LiveFrameOverlay, LiveSession


class RecordingHook(ABC):
    """Extension point for live session recording.

    Future implementations may support MP4 recording, snapshots, and evidence
    storage. The default release ships with a no-op hook only.
    """

    @abstractmethod
    def on_session_start(self, session: LiveSession) -> None:
        """Called when a live session starts."""

    @abstractmethod
    def on_session_stop(self, session: LiveSession) -> None:
        """Called when a live session stops."""

    @abstractmethod
    def on_frame(
        self,
        session: LiveSession,
        *,
        image_bytes: bytes,
        overlay: LiveFrameOverlay | None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Called after a frame is captured or processed."""


class NoOpRecordingHook(RecordingHook):
    """Default recording hook that performs no work."""

    def on_session_start(self, session: LiveSession) -> None:
        return None

    def on_session_stop(self, session: LiveSession) -> None:
        return None

    def on_frame(
        self,
        session: LiveSession,
        *,
        image_bytes: bytes,
        overlay: LiveFrameOverlay | None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        return None
