"""Live camera manager."""

from __future__ import annotations

import numpy as np

from backend.app.live.config import LiveCameraConfig
from backend.app.live.models import LiveFrameOverlay, LiveSession, SessionStatus
from backend.app.live.session import LiveSessionState
from backend.app.live.recording import NoOpRecordingHook, RecordingHook
from backend.app.live.stream import BrowserWebcamStream, LiveStream
from backend.app.live.utils import build_overlay, build_timeline_entry, utc_now
from backend.app.pipeline.profile import PipelineProfile
from backend.app.services.recognition_service import RecognitionService


class LiveCameraManager:
    """Owns live camera sessions and frame processing.

    All recognition work is delegated to ``RecognitionService``. This module
    must never import or invoke AI modules directly.
    """

    def __init__(
        self,
        recognition_service: RecognitionService,
        *,
        stream: LiveStream | None = None,
        recording_hook: RecordingHook | None = None,
    ) -> None:
        self._recognition = recognition_service
        self._stream = stream or BrowserWebcamStream()
        self._recording_hook = recording_hook or NoOpRecordingHook()
        self._sessions: dict[str, LiveSessionState] = {}
        self._active_session_id: str | None = None

    @property
    def stream(self) -> LiveStream:
        """Return the configured live stream."""
        return self._stream

    def start_session(self, config: LiveCameraConfig | None = None) -> LiveSession:
        """Start a new live camera session."""
        if self._active_session_id is not None:
            active = self._sessions[self._active_session_id]
            if active.session.status in {SessionStatus.RUNNING, SessionStatus.PAUSED}:
                raise ValueError("A live session is already active.")

        resolved = config or LiveCameraConfig()
        state = LiveSessionState(resolved)
        self._stream.open(resolved)
        self._sessions[state.session.id] = state
        self._active_session_id = state.session.id
        self._recording_hook.on_session_start(state.session)
        return state.session

    def stop_session(self, session_id: str | None = None) -> LiveSession | None:
        """Stop an active or paused session."""
        resolved_id = session_id or self._active_session_id
        if resolved_id is None:
            return None

        state = self._sessions.get(resolved_id)
        if state is None:
            return None

        state.session.status = SessionStatus.STOPPED
        state.touch()
        self._stream.close()
        self._recording_hook.on_session_stop(state.session)
        if self._active_session_id == resolved_id:
            self._active_session_id = None
        return state.session

    def pause_session(self, session_id: str | None = None) -> LiveSession | None:
        """Pause an active session."""
        state = self._resolve_active_state(session_id)
        if state is None:
            return None
        if state.session.status != SessionStatus.RUNNING:
            return state.session

        state.session.status = SessionStatus.PAUSED
        state.touch()
        return state.session

    def resume_session(self, session_id: str | None = None) -> LiveSession | None:
        """Resume a paused session."""
        state = self._resolve_active_state(session_id)
        if state is None:
            return None
        if state.session.status != SessionStatus.PAUSED:
            return state.session

        state.session.status = SessionStatus.RUNNING
        state.touch()
        return state.session

    def process_frame(
        self,
        image: np.ndarray,
        *,
        session_id: str | None = None,
        force: bool = False,
    ) -> LiveFrameOverlay | None:
        """Process one frame through ``RecognitionService``."""
        state = self._resolve_active_state(session_id)
        if state is None or state.session.status != SessionStatus.RUNNING:
            return None

        state.record_capture_frame()
        if not force and not state.should_process_frame(force=force):
            state.record_drop()
            return state.session.last_overlay

        profile = PipelineProfile(state.session.config.pipeline_profile)
        context = self._recognition.recognize(image, profile)
        overlay = build_overlay(context)
        recognized_faces = sum(1 for face in overlay.faces if face.identity_id)

        state.record_processing(
            latency_ms=overlay.pipeline_time_ms,
            recognized_faces=recognized_faces,
        )
        state.session.last_overlay = overlay
        state.session.timeline.append(build_timeline_entry(context))
        if len(state.session.timeline) > 50:
            state.session.timeline = state.session.timeline[-50:]
        state.touch()
        return overlay

    def process_frame_bytes(
        self,
        image_bytes: bytes,
        *,
        session_id: str | None = None,
        force: bool = False,
    ) -> LiveFrameOverlay | None:
        """Decode frame bytes and process through the live pipeline."""
        before = self.status(session_id)
        before_dropped = before.dropped_frames if before else 0
        image = self._recognition.decode_image(image_bytes)
        overlay = self.process_frame(image, session_id=session_id, force=force)
        state = self._resolve_active_state(session_id)
        if state is not None:
            dropped = state.session.dropped_frames > before_dropped
            self._recording_hook.on_frame(
                state.session,
                image_bytes=image_bytes,
                overlay=overlay,
                metadata={"dropped": dropped, "processed": not dropped},
            )
        return overlay

    def status(self, session_id: str | None = None) -> LiveSession | None:
        """Return the active or requested session status."""
        resolved_id = session_id or self._active_session_id
        if resolved_id is None:
            return None
        state = self._sessions.get(resolved_id)
        return state.session if state else None

    def list_sessions(self) -> list[LiveSession]:
        """Return all sessions in reverse chronological order."""
        return sorted(
            (state.session for state in self._sessions.values()),
            key=lambda session: session.started_at,
            reverse=True,
        )

    def clear(self) -> None:
        """Clear all sessions. Intended for tests."""
        self._stream.close()
        self._sessions.clear()
        self._active_session_id = None

    def _resolve_active_state(self, session_id: str | None) -> LiveSessionState | None:
        resolved_id = session_id or self._active_session_id
        if resolved_id is None:
            return None
        return self._sessions.get(resolved_id)
