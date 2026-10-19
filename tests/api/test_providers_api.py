"""API tests for providers inspection endpoint."""

from fastapi.testclient import TestClient


def test_list_providers(client: TestClient):
    response = client.get("/v1/providers")
    assert response.status_code == 200
    providers = response.json()
    assert isinstance(providers, list)
    names = [p["name"] for p in providers]
    assert "fake_echo" in names

    fake_p = next(p for p in providers if p["name"] == "fake_echo")
    assert fake_p["enabled"] is True
    assert fake_p["configured"] is True
    assert "capabilities" in fake_p
    assert fake_p["allowed_models"] == ["echo-v1", "echo-v2"]
