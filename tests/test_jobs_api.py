"""Jobs API scaffold tests."""

from fastapi.testclient import TestClient

from backend.api.dependencies import get_app_services
from backend.api.main import create_app
from backend.api.version import API_V1_PREFIX


def test_jobs_api_scaffold() -> None:
    client = TestClient(create_app())
    get_app_services().jobs.manager.clear()

    submit = client.post(
        f"{API_V1_PREFIX}/jobs",
        json={"task_type": "recognition", "payload": {"profile": "search"}},
    )
    assert submit.status_code == 200
    job_id = submit.json()["id"]
    assert submit.json()["state"] == "queued"

    listing = client.get(f"{API_V1_PREFIX}/jobs")
    assert listing.status_code == 200
    assert listing.json()["count"] == 1
    assert listing.json()["summary"]["queued"] == 1

    detail = client.get(f"{API_V1_PREFIX}/jobs/{job_id}")
    assert detail.status_code == 200

    cancelled = client.delete(f"{API_V1_PREFIX}/jobs/{job_id}")
    assert cancelled.status_code == 200
    assert cancelled.json()["state"] == "cancelled"


def test_jobs_console_page() -> None:
    client = TestClient(create_app())
    response = client.get("/jobs")
    assert response.status_code == 200
    assert "Queued" in response.text
    assert "Running" in response.text
