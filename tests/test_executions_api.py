"""Execution API tests."""

import numpy as np
from fastapi.testclient import TestClient

from backend.api.dependencies import get_app_services
from backend.api.main import create_app
from backend.api.version import API_V1_PREFIX
from backend.app.pipeline.context import PipelineContext
from backend.app.pipeline.profile import PipelineProfile


def test_executions_endpoints() -> None:
    client = TestClient(create_app())
    observability = get_app_services().observability
    observability.clear_executions()

    context = PipelineContext(
        image=np.zeros((8, 8, 3), dtype=np.uint8),
        profile=PipelineProfile.SEARCH,
        timings={"scrfd": 0.01},
    )
    record = observability.record_execution(context, source="test.api")

    list_response = client.get(f"{API_V1_PREFIX}/executions")
    assert list_response.status_code == 200
    payload = list_response.json()
    assert payload["count"] == 1

    get_response = client.get(f"{API_V1_PREFIX}/executions/{record.id}")
    assert get_response.status_code == 200
    assert get_response.json()["id"] == record.id

    latest_response = client.get(f"{API_V1_PREFIX}/executions/latest?limit=5")
    assert latest_response.status_code == 200
    assert latest_response.json()["count"] == 1

    missing = client.get(f"{API_V1_PREFIX}/executions/does-not-exist")
    assert missing.status_code == 404


def test_executions_console_page() -> None:
    client = TestClient(create_app())
    response = client.get("/executions")
    assert response.status_code == 200
    assert "Execution History" in response.text
    assert "Pipeline Timeline" in response.text
