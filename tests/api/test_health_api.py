"""API tests for health, readiness, and metrics endpoints."""

from fastapi.testclient import TestClient
from unified_llm.core.registry import ProviderRegistry


def test_healthz(client: TestClient):
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_readyz_healthy(client: TestClient):
    response = client.get("/readyz")
    assert response.status_code == 200
    data = response.json()
    assert data["ready"] is True
    assert "fake_echo" in data["active_providers"]


def test_readyz_deep_check(client: TestClient):
    response = client.get("/readyz?deep=true")
    assert response.status_code == 200
    data = response.json()
    assert data["ready"] is True
    assert "fake_echo" in data["details"]
    assert data["details"]["fake_echo"]["ok"] is True


def test_readyz_unready_when_no_active_providers(client: TestClient, app):
    # Temporarily clear active adapters
    reg: ProviderRegistry = app.state.registry
    original = reg._active_adapters
    reg._active_adapters = {}

    try:
        response = client.get("/readyz")
        assert response.status_code == 503
        data = response.json()
        assert data["ready"] is False
    finally:
        reg._active_adapters = original


def test_metrics_endpoint(client: TestClient):
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]
    assert "ulil_requests_total" in response.text
