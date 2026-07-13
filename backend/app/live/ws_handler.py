"""WebSocket protocol handler for live camera streaming."""

from __future__ import annotations

import base64
import json
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect

from backend.app.live.processor import process_live_frame
from backend.app.services.live_camera_service import LiveCameraService


class LiveWebSocketHandler:
    """Handle live camera WebSocket messages."""

    def __init__(self, live_service: LiveCameraService) -> None:
        self._live_service = live_service

    async def handle(self, websocket: WebSocket) -> None:
        await websocket.accept()
        try:
            while True:
                message = await websocket.receive()
                if message.get("type") == "websocket.disconnect":
                    break

                try:
                    payload = self._parse_message(message)
                    response = self._dispatch(payload)
                except (ValueError, NotImplementedError) as exc:
                    response = {"type": "error", "message": str(exc)}
                await websocket.send_json(response)
        except WebSocketDisconnect:
            return

    def _parse_message(self, message: dict[str, Any]) -> dict[str, Any]:
        if "text" in message and message["text"] is not None:
            return json.loads(message["text"])

        if "bytes" in message and message["bytes"] is not None:
            raw = message["bytes"]
            separator = raw.find(b"\n")
            if separator == -1:
                raise ValueError("Binary WebSocket frame must contain a JSON header.")
            header = json.loads(raw[:separator].decode("utf-8"))
            header["image_bytes"] = raw[separator + 1 :]
            return header

        raise ValueError("Unsupported WebSocket message type.")

    def _dispatch(self, payload: dict[str, Any]) -> dict[str, Any]:
        message_type = payload.get("type", "frame")

        if message_type == "ping":
            return {"type": "pong"}

        if message_type == "frame":
            return self._handle_frame(payload)

        if message_type == "status":
            from backend.api.schemas.live_mappers import map_live_session

            session_id = payload.get("session_id")
            session = self._live_service.status(session_id)
            if session is None:
                raise ValueError("No live session was found.")
            return {
                "type": "status",
                "session": map_live_session(session).model_dump(mode="json"),
            }

        raise ValueError(f"Unsupported WebSocket message type {message_type!r}.")

    def _handle_frame(self, payload: dict[str, Any]) -> dict[str, Any]:
        from backend.api.schemas.live_mappers import map_frame_result

        session_id = payload.get("session_id")
        if not session_id:
            raise ValueError("session_id is required for frame messages.")

        image_bytes = payload.get("image_bytes")
        if image_bytes is None:
            encoded = payload.get("image_base64")
            if not encoded:
                raise ValueError("Frame image is required.")
            image_bytes = base64.b64decode(encoded)

        result = process_live_frame(
            self._live_service.manager,
            session_id=session_id,
            image_bytes=image_bytes,
            force=bool(payload.get("force", False)),
        )
        response = map_frame_result(result)
        return {
            "type": "frame_result",
            **response.model_dump(mode="json"),
        }
