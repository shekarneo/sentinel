"""API layer tests."""

from fastapi.testclient import TestClient

from backend.api.main import create_app
from backend.api.version import API_V1_PREFIX


def test_api_app_creates() -> None:
    client = TestClient(create_app())
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_v1_system_status() -> None:
    client = TestClient(create_app())
    response = client.get(f"{API_V1_PREFIX}/system/status")
    assert response.status_code == 200
    payload = response.json()
    assert "api_version" in payload
    assert "gallery_size" in payload


def test_v1_configuration() -> None:
    client = TestClient(create_app())
    response = client.get(f"{API_V1_PREFIX}/configuration")
    assert response.status_code == 200
    payload = response.json()
    assert "app" in payload
    assert "pipeline_profiles" in payload
    assert len(payload["pipeline_profiles"]) > 0


def test_v1_configuration_files() -> None:
    client = TestClient(create_app())
    response = client.get(f"{API_V1_PREFIX}/configuration/files")
    assert response.status_code == 200
    payload = response.json()
    assert "models_yaml" in payload
    assert "thresholds_yaml" in payload
    assert "pipeline_profiles_yaml" in payload


def test_v1_dashboard() -> None:
    client = TestClient(create_app())
    response = client.get(f"{API_V1_PREFIX}/system/dashboard")
    assert response.status_code == 200
    payload = response.json()
    assert "models" in payload
    assert "pipeline_profiles" in payload


def test_v1_gallery_list() -> None:
    client = TestClient(create_app())
    response = client.get(f"{API_V1_PREFIX}/gallery")
    assert response.status_code == 200
    payload = response.json()
    assert "count" in payload
    assert "identities" in payload
    assert "entries" in payload


def test_console_dashboard_page() -> None:
    client = TestClient(create_app())
    response = client.get("/dashboard")
    assert response.status_code == 200
    assert "Sentinel" in response.text


def test_console_recognition_page() -> None:
    client = TestClient(create_app())
    response = client.get("/recognition")
    assert response.status_code == 200
    assert "Run Pipeline" in response.text
