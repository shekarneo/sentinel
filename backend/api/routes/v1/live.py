"""Live camera API routes (scaffold)."""

from fastapi import APIRouter, Depends, File, Form, UploadFile, WebSocket

from backend.api.dependencies import get_live_camera_service
from backend.api.exceptions import NotFoundAPIError, ValidationAPIError
from backend.api.schemas.live import (
    LiveFrameProcessResponse,
    LivePerformanceResponse,
    LiveSessionListResponse,
    LiveStartRequest,
    LiveStatusResponse,
    LiveStopRequest,
)
from backend.api.schemas.live_mappers import (
    map_frame_result,
    map_live_config_request,
    map_live_session,
    map_live_session_list,
    map_live_status,
)
from backend.app.live.processor import process_live_frame as run_live_frame
from backend.app.live.ws_handler import LiveWebSocketHandler
from backend.app.services.live_camera_service import LiveCameraService

router = APIRouter()


@router.post("/start")
def start_live_session(
    request: LiveStartRequest,
    live_service: LiveCameraService = Depends(get_live_camera_service),
):
    """Start a live camera session."""
    try:
        session = live_service.start_session(map_live_config_request(request.config))
    except ValueError as exc:
        raise ValidationAPIError(str(exc)) from exc
    return map_live_session(session)


@router.post("/stop")
def stop_live_session(
    request: LiveStopRequest,
    live_service: LiveCameraService = Depends(get_live_camera_service),
):
    """Stop a live camera session."""
    session = live_service.stop_session(request.session_id)
    if session is None:
        raise NotFoundAPIError("No active live session was found.")
    return map_live_session(session)


@router.get("/status", response_model=LiveStatusResponse)
def live_status(
    live_service: LiveCameraService = Depends(get_live_camera_service),
) -> LiveStatusResponse:
    """Return the active live session status."""
    return map_live_status(live_service.status())


@router.get("/sessions", response_model=LiveSessionListResponse)
def list_live_sessions(
    live_service: LiveCameraService = Depends(get_live_camera_service),
) -> LiveSessionListResponse:
    """List live camera sessions."""
    return map_live_session_list(live_service.list_sessions())


@router.post("/pause")
def pause_live_session(
    request: LiveStopRequest,
    live_service: LiveCameraService = Depends(get_live_camera_service),
):
    """Pause a live camera session."""
    session = live_service.pause_session(request.session_id)
    if session is None:
        raise NotFoundAPIError("No active live session was found.")
    return map_live_session(session)


@router.post("/resume")
def resume_live_session(
    request: LiveStopRequest,
    live_service: LiveCameraService = Depends(get_live_camera_service),
):
    """Resume a paused live camera session."""
    session = live_service.resume_session(request.session_id)
    if session is None:
        raise NotFoundAPIError("No active live session was found.")
    return map_live_session(session)


@router.post("/frame", response_model=LiveFrameProcessResponse)
async def process_live_frame(
    session_id: str = Form(...),
    image: UploadFile = File(...),
    force: bool = Form(False),
    live_service: LiveCameraService = Depends(get_live_camera_service),
) -> LiveFrameProcessResponse:
    """Process one webcam frame through the live recognition pipeline."""
    image_bytes = await image.read()
    if not image_bytes:
        raise ValidationAPIError("Frame image is required.")

    try:
        result = run_live_frame(
            live_service.manager,
            session_id=session_id,
            image_bytes=image_bytes,
            force=force,
        )
    except ValueError as exc:
        raise NotFoundAPIError(str(exc)) from exc

    return map_frame_result(result)


@router.websocket("/ws")
async def live_websocket(
    websocket: WebSocket,
    live_service: LiveCameraService = Depends(get_live_camera_service),
) -> None:
    """Stream live frames and receive recognition overlays over WebSocket."""
    handler = LiveWebSocketHandler(live_service)
    await handler.handle(websocket)
