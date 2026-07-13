"""Recognition console partial tests."""

import json

from fastapi.testclient import TestClient

from backend.api.main import create_app
from frontend.pipeline_view import build_stages, build_timeline


def test_pipeline_view_builders() -> None:
    result = {
        "profile": "kyc",
        "face_count": 1,
        "faces": [
            {
                "confidence": 0.99,
                "bounding_box": {"x": 1, "y": 2, "width": 3, "height": 4},
                "has_alignment": True,
                "assessment": {"is_acceptable": True, "overall_score": 0.9},
                "embedding": {"model_name": "arcface", "dimension": 512, "normalized": True},
            }
        ],
        "timings": {"stages": {"scrfd": 0.024, "alignment": 0.002, "embedding": 0.014}},
        "metadata": {},
    }
    profile_config = {"stages": ["scrfd", "alignment", "embedding"], "search": {"enabled": False}}
    stages = build_stages(result, profile_config)
    assert stages[0]["status"] == "success"
    timeline = build_timeline(result)
    assert timeline["total_ms"] > 0
    assert timeline["entries"][0]["width_pct"] >= 12


def test_recognition_render_partial() -> None:
    client = TestClient(create_app())
    payload = {
        "result": {
            "profile": "search",
            "face_count": 0,
            "faces": [],
            "timings": {"stages": {}},
            "metadata": {},
        },
        "original_image_base64": "aGVsbG8=",
        "api_latency_ms": 12.3,
    }
    response = client.post("/partials/recognition/render", json=payload)
    assert response.status_code == 200
    assert "Pipeline Stages" in response.text
    assert "recognition-payload" in response.text


def test_recognition_page_has_htmx_form() -> None:
    client = TestClient(create_app())
    response = client.get("/recognition")
    assert response.status_code == 200
    assert 'hx-post="/partials/recognition/run"' in response.text
    assert "Recognition History" in response.text
