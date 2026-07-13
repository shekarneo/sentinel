"""Live camera API tests."""

import base64

import numpy as np
from fastapi.testclient import TestClient

from backend.api.dependencies import get_app_services
from backend.api.main import create_app
from backend.api.version import API_V1_PREFIX


def test_live_api_scaffold() -> None:
    client = TestClient(create_app())
    get_app_services().live.manager.clear()

    start = client.post(
        f"{API_V1_PREFIX}/live/start",
        json={
            "config": {
                "pipeline_profile": "surveillance",
                "target_fps": 10,
                "recognition_policy": {"policy_type": "every_n_frames", "frame_interval": 2},
            }
        },
    )
    assert start.status_code == 200
    session_id = start.json()["id"]
    assert start.json()["status"] == "running"
    assert start.json()["config"]["recognition_policy"]["policy_type"] == "every_n_frames"

    status = client.get(f"{API_V1_PREFIX}/live/status")
    assert status.status_code == 200
    assert status.json()["active"] is True

    paused = client.post(
        f"{API_V1_PREFIX}/live/pause",
        json={"session_id": session_id},
    )
    assert paused.status_code == 200
    assert paused.json()["status"] == "paused"

    resumed = client.post(
        f"{API_V1_PREFIX}/live/resume",
        json={"session_id": session_id},
    )
    assert resumed.status_code == 200
    assert resumed.json()["status"] == "running"

    sessions = client.get(f"{API_V1_PREFIX}/live/sessions")
    assert sessions.status_code == 200
    assert sessions.json()["count"] == 1

    stopped = client.post(
        f"{API_V1_PREFIX}/live/stop",
        json={"session_id": session_id},
    )
    assert stopped.status_code == 200
    assert stopped.json()["status"] == "stopped"


def test_live_frame_rest_compatibility() -> None:
    client = TestClient(create_app())
    get_app_services().live.manager.clear()

    start = client.post(
        f"{API_V1_PREFIX}/live/start",
        json={"config": {"recognition_policy": {"policy_type": "every_frame"}}},
    )
    session_id = start.json()["id"]
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    import cv2

    ok, encoded = cv2.imencode(".jpg", image)
    assert ok

    response = client.post(
        f"{API_V1_PREFIX}/live/frame",
        data={"session_id": session_id, "force": "true"},
        files={"image": ("frame.jpg", encoded.tobytes(), "image/jpeg")},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["session_id"] == session_id
    assert "render_spec" in payload
    assert payload["render_spec"]["renderer"] == "canvas"


def test_live_websocket_frame_processing() -> None:
    client = TestClient(create_app())
    get_app_services().live.manager.clear()

    start = client.post(
        f"{API_V1_PREFIX}/live/start",
        json={"config": {"recognition_policy": {"policy_type": "every_frame"}}},
    )
    session_id = start.json()["id"]
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    import cv2

    ok, encoded = cv2.imencode(".jpg", image)
    assert ok

    with client.websocket_connect(f"{API_V1_PREFIX}/live/ws") as websocket:
        websocket.send_json(
            {
                "type": "frame",
                "session_id": session_id,
                "force": True,
                "image_base64": base64.b64encode(encoded.tobytes()).decode("ascii"),
            }
        )
        payload = websocket.receive_json()
        assert payload["type"] == "frame_result"
        assert payload["session_id"] == session_id
        assert payload["render_spec"]["renderer"] == "canvas"


def test_live_console_page() -> None:
    client = TestClient(create_app())
    response = client.get("/live")
    assert response.status_code == 200
    assert "Live Camera" in response.text
    assert "Recognition Timeline" in response.text

