"""Live camera API mappers."""

from __future__ import annotations

from backend.api.schemas.live import (
    LiveConfigRequest,
    LiveExtensionHooksResponse,
    LiveFaceOverlayResponse,
    LiveFrameProcessResponse,
    LivePerformanceResponse,
    LiveRecognitionPolicyRequest,
    LiveSessionListResponse,
    LiveSessionResponse,
    LiveStatusResponse,
    LiveTimelineEntryResponse,
)
from backend.app.live.config import LiveCameraConfig, LiveExtensionHooks
from backend.app.live.models import LiveFrameOverlay, LiveSession
from backend.app.live.policy import RecognitionPolicy, RecognitionPolicyType
from backend.app.live.processor import LiveFrameResult


def map_recognition_policy(policy: RecognitionPolicy) -> LiveRecognitionPolicyRequest:
    """Map domain recognition policy to API schema."""
    return LiveRecognitionPolicyRequest(
        policy_type=policy.policy_type.value,
        frame_interval=policy.frame_interval,
    )


def map_recognition_policy_request(request: LiveRecognitionPolicyRequest) -> RecognitionPolicy:
    """Map API recognition policy to domain model."""
    try:
        policy_type = RecognitionPolicyType(request.policy_type.lower())
    except ValueError as exc:
        valid = ", ".join(item.value for item in RecognitionPolicyType)
        raise ValueError(
            f"Unknown recognition policy {request.policy_type!r}. Valid policies: {valid}."
        ) from exc
    return RecognitionPolicy(
        policy_type=policy_type,
        frame_interval=request.frame_interval,
    )


def map_live_config(config: LiveCameraConfig) -> LiveConfigRequest:
    """Map live config to API schema."""
    return LiveConfigRequest(
        camera_index=config.camera_index,
        resolution_width=config.resolution_width,
        resolution_height=config.resolution_height,
        target_fps=config.target_fps,
        pipeline_profile=config.pipeline_profile,
        overlay_enabled=config.overlay_enabled,
        submission_interval_ms=config.submission_interval_ms,
        recognition_policy=map_recognition_policy(config.recognition_policy),
        extensions=LiveExtensionHooksResponse(
            tracking_enabled=config.extensions.tracking_enabled,
            multi_camera_enabled=config.extensions.multi_camera_enabled,
            rtsp_url=config.extensions.rtsp_url,
            video_recording_enabled=config.extensions.video_recording_enabled,
            fraud_detection_enabled=config.extensions.fraud_detection_enabled,
        ),
    )


def map_live_config_request(request: LiveConfigRequest) -> LiveCameraConfig:
    """Map API config request to domain config."""
    policy = map_recognition_policy_request(request.recognition_policy)
    if request.frame_skip is not None:
        policy = RecognitionPolicy.from_legacy(frame_skip=request.frame_skip)

    submission_interval_ms = request.submission_interval_ms
    if request.recognition_interval_ms is not None:
        submission_interval_ms = request.recognition_interval_ms

    return LiveCameraConfig(
        camera_index=request.camera_index,
        resolution_width=request.resolution_width,
        resolution_height=request.resolution_height,
        target_fps=request.target_fps,
        pipeline_profile=request.pipeline_profile,
        overlay_enabled=request.overlay_enabled,
        submission_interval_ms=submission_interval_ms,
        recognition_policy=policy,
        extensions=LiveExtensionHooks(
            tracking_enabled=request.extensions.tracking_enabled,
            multi_camera_enabled=request.extensions.multi_camera_enabled,
            rtsp_url=request.extensions.rtsp_url,
            video_recording_enabled=request.extensions.video_recording_enabled,
            fraud_detection_enabled=request.extensions.fraud_detection_enabled,
        ),
    )


def map_live_overlay(overlay: LiveFrameOverlay | None) -> dict | None:
    """Map overlay model to JSON-safe dict."""
    if overlay is None:
        return None
    return {
        "profile": overlay.profile,
        "face_count": overlay.face_count,
        "pipeline_time_ms": overlay.pipeline_time_ms,
        "faces": [
            LiveFaceOverlayResponse(
                bounding_box=face.bounding_box,
                confidence=face.confidence,
                identity_id=face.identity_id,
                similarity=face.similarity,
                verification_decision=face.verification_decision,
            ).model_dump()
            for face in overlay.faces
        ],
    }


def map_frame_result(result: LiveFrameResult) -> LiveFrameProcessResponse:
    """Map a processed frame result to the API response."""
    return LiveFrameProcessResponse(
        session_id=result.session_id,
        overlay=map_live_overlay(result.overlay),
        metrics=LivePerformanceResponse(
            capture_fps=result.metrics.capture_fps,
            processing_fps=result.metrics.processing_fps,
            pipeline_latency_ms=result.metrics.pipeline_latency_ms,
            average_latency_ms=result.metrics.average_latency_ms,
            dropped_frames=result.metrics.dropped_frames,
            recognition_rate=result.metrics.recognition_rate,
        ),
        dropped=result.dropped,
        render_spec=result.render_spec,
    )


def map_live_session(session: LiveSession) -> LiveSessionResponse:
    """Map live session to API response."""
    return LiveSessionResponse(
        id=session.id,
        started_at=session.started_at,
        updated_at=session.updated_at,
        status=session.status.value,
        pipeline_profile=session.pipeline_profile,
        fps=session.fps,
        processed_frames=session.processed_frames,
        recognized_faces=session.recognized_faces,
        dropped_frames=session.dropped_frames,
        config=map_live_config(session.config),
        metrics=LivePerformanceResponse(
            capture_fps=session.metrics.capture_fps,
            processing_fps=session.metrics.processing_fps,
            pipeline_latency_ms=session.metrics.pipeline_latency_ms,
            average_latency_ms=session.metrics.average_latency_ms,
            dropped_frames=session.metrics.dropped_frames,
            recognition_rate=session.metrics.recognition_rate,
        ),
        timeline=[
            LiveTimelineEntryResponse(
                timestamp=entry.timestamp,
                face_count=entry.face_count,
                pipeline_time_ms=entry.pipeline_time_ms,
                recognized_identities=entry.recognized_identities,
            )
            for entry in session.timeline
        ],
        last_overlay=map_live_overlay(session.last_overlay),
    )


def map_live_status(session: LiveSession | None) -> LiveStatusResponse:
    """Map active session status."""
    if session is None:
        return LiveStatusResponse(active=False, session=None)
    return LiveStatusResponse(active=True, session=map_live_session(session))


def map_live_session_list(sessions: list[LiveSession]) -> LiveSessionListResponse:
    """Map session list response."""
    mapped = [map_live_session(session) for session in sessions]
    return LiveSessionListResponse(count=len(mapped), sessions=mapped)
