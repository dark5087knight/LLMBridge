"""API tests for /v1/chat endpoints, validation policies, and authentication."""

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from unified_llm.config.settings import ServiceSettings
from unified_llm.core.capabilities import Capabilities
from unified_llm.main import create_app
from tests.conftest import FakeEchoAdapter


def test_chat_completion_success(client: TestClient):
    payload = {
        "provider": "fake_echo",
        "model": "echo-v1",
        "messages": [
            {"role": "user", "content": "Hello LLMBridge!"}
        ],
        "params": {
            "temperature": 0.5,
            "max_tokens": 100,
        },
        "options": {
            "include_raw": True,
        },
    }

    response = client.post("/v1/chat", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["provider"] == "fake_echo"
    assert data["model"] == "echo-v1"
    assert data["message"]["content"] == "Echo: Hello LLMBridge!"
    assert data["finish_reason"] == "stop"
    assert data["usage"]["total_tokens"] == 25
    assert data["raw"] == {"echo": True}
    assert "X-Request-ID" in response.headers
    assert data["id"] == response.headers["X-Request-ID"]


def test_chat_unsupported_parameter_lenient(client: TestClient, app):
    # Temporarily remove temperature from fake adapter capabilities to test warning
    fake_adapter = app.state.registry.get("fake_echo")
    original_caps = fake_adapter.capabilities
    fake_adapter.capabilities = Capabilities(
        supported_params=frozenset({"max_tokens"}),
    )

    payload = {
        "provider": "fake_echo",
        "model": "echo-v1",
        "messages": [{"role": "user", "content": "Hello"}],
        "params": {
            "temperature": 0.7,
            "max_tokens": 50,
        },
        "options": {
            "strict_params": False,
        },
    }

    try:
        response = client.post("/v1/chat", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert len(data["warnings"]) >= 1
        assert data["warnings"][0]["code"] == "param_ignored"
        assert data["warnings"][0]["param"] == "temperature"
    finally:
        fake_adapter.capabilities = original_caps


def test_chat_unsupported_parameter_strict(client: TestClient, app):
    fake_adapter = app.state.registry.get("fake_echo")
    original_caps = fake_adapter.capabilities
    fake_adapter.capabilities = Capabilities(
        supported_params=frozenset({"max_tokens"}),
    )

    payload = {
        "provider": "fake_echo",
        "model": "echo-v1",
        "messages": [{"role": "user", "content": "Hello"}],
        "params": {
            "temperature": 0.7,
        },
        "options": {
            "strict_params": True,
        },
    }

    try:
        response = client.post("/v1/chat", json=payload)
        assert response.status_code == 422
        data = response.json()
        assert data["error"]["type"] == "unsupported_parameter"
        assert "temperature" in data["error"]["message"]
    finally:
        fake_adapter.capabilities = original_caps


def test_chat_unknown_provider(client: TestClient):
    payload = {
        "provider": "non_existent_provider",
        "model": "foo",
        "messages": [{"role": "user", "content": "Hello"}],
    }
    response = client.post("/v1/chat", json=payload)
    assert response.status_code == 404
    data = response.json()
    assert data["error"]["type"] == "unknown_provider"


def test_chat_provider_not_configured(client: TestClient):
    payload = {
        "provider": "unconfigured",
        "model": "foo",
        "messages": [{"role": "user", "content": "Hello"}],
    }
    response = client.post("/v1/chat", json=payload)
    assert response.status_code == 503
    data = response.json()
    assert data["error"]["type"] == "provider_not_configured"


def test_chat_model_not_allowed(client: TestClient):
    payload = {
        "provider": "fake_echo",
        "model": "forbidden-model",
        "messages": [{"role": "user", "content": "Hello"}],
    }
    response = client.post("/v1/chat", json=payload)
    assert response.status_code == 403
    data = response.json()
    assert data["error"]["type"] == "model_not_allowed"


def test_chat_streaming_phase2_endpoint(client: TestClient):
    payload = {
        "provider": "fake_echo",
        "model": "echo-v1",
        "messages": [{"role": "user", "content": "Hello"}],
    }
    response = client.post("/v1/chat/stream", json=payload)
    assert response.status_code == 422
    data = response.json()
    assert data["error"]["type"] == "unsupported_feature"


def test_inbound_auth_enforcement():
    # App configured with an API key
    settings = ServiceSettings(
        api_keys=[SecretStr("secret-token-123")],
    )
    auth_app = create_app(settings)
    auth_client = TestClient(auth_app)

    # 1. Missing header -> 401
    resp1 = auth_client.post(
        "/v1/chat",
        json={"provider": "p", "model": "m", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp1.status_code == 401
    assert resp1.json()["error"]["type"] == "unauthorized"

    # 2. Invalid header -> 401
    resp2 = auth_client.post(
        "/v1/chat",
        headers={"Authorization": "Bearer wrong-key"},
        json={"provider": "p", "model": "m", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp2.status_code == 401

    # 3. Health checks remain accessible without auth
    health_resp = auth_client.get("/healthz")
    assert health_resp.status_code == 200


def test_payload_size_limit():
    settings = ServiceSettings(max_request_bytes=50)
    small_app = create_app(settings)
    small_client = TestClient(small_app)

    large_payload = {
        "provider": "fake_echo",
        "model": "echo-v1",
        "messages": [{"role": "user", "content": "A" * 100}],
    }
    resp = small_client.post("/v1/chat", json=large_payload)
    assert resp.status_code == 413
    assert resp.json()["error"]["type"] == "payload_too_large"
