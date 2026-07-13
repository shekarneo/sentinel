"""Dataset processing API tests."""

from fastapi.testclient import TestClient

from backend.api.dependencies import get_app_services
from backend.api.main import create_app
from backend.api.version import API_V1_PREFIX


def test_datasets_api_scaffold() -> None:
    client = TestClient(create_app())
    get_app_services().datasets.manager.clear()
    get_app_services().jobs.manager.clear()

    process = client.post(
        f"{API_V1_PREFIX}/datasets/process",
        json={
            "dataset_type": "image_folder",
            "operation": "recognition",
            "source_path": "/data/faces",
            "item_count": 2,
            "export_path": "/tmp/exports",
        },
    )
    assert process.status_code == 200
    job_id = process.json()["id"]
    assert process.json()["summary"]["total_items"] == 2
    assert len(process.json()["execution_task_ids"]) == 2

    listing = client.get(f"{API_V1_PREFIX}/datasets/jobs")
    assert listing.status_code == 200
    assert listing.json()["count"] == 1

    detail = client.get(f"{API_V1_PREFIX}/datasets/jobs/{job_id}")
    assert detail.status_code == 200

    results = client.get(f"{API_V1_PREFIX}/datasets/results/{job_id}")
    assert results.status_code == 200
    assert results.json()["job_id"] == job_id
    assert len(results.json()["execution_tasks"]) == 2
    assert "result" in results.json()
    assert results.json()["result"]["processed"] == 0


def test_datasets_console_page() -> None:
    client = TestClient(create_app())
    response = client.get("/datasets")
    assert response.status_code == 200
    assert "Dataset Processing" in response.text
    assert "Processed" in response.text
