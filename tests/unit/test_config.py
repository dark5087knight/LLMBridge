"""Unit tests for configuration loading, interpolation, and validation."""

import os
import pytest
from pydantic import SecretStr

from unified_llm.config.loader import (
    interpolate_env_vars,
    load_yaml_config,
    validate_provider_credentials,
)
from unified_llm.config.settings import ProviderConfig, ServiceSettings


def test_interpolate_env_vars(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("TEST_API_KEY", "secret-12345")
    monkeypatch.delenv("MISSING_VAR", raising=False)

    text = "key: ${TEST_API_KEY}, fallback: ${MISSING_VAR:default_val}"
    result = interpolate_env_vars(text)

    assert result == "key: secret-12345, fallback: default_val"


def test_yaml_config_loading(tmp_path):
    yaml_file = tmp_path / "test_providers.yaml"
    yaml_file.write_text(
        """
providers:
  openai:
    enabled: true
    timeout_s: 45
    allowed_models: ["gpt-4o"]
""",
        encoding="utf-8",
    )

    data = load_yaml_config(yaml_file)
    assert "providers" in data
    assert data["providers"]["openai"]["timeout_s"] == 45
    assert data["providers"]["openai"]["allowed_models"] == ["gpt-4o"]


def test_validate_provider_credentials():
    # Enabled cloud provider with missing key should fail fast
    bad_config = ProviderConfig(enabled=True, api_key=None)
    with pytest.raises(ValueError) as exc_info:
        validate_provider_credentials("openai", bad_config)
    assert "missing required credentials" in str(exc_info.value)

    # Disabled provider does not fail
    disabled_config = ProviderConfig(enabled=False, api_key=None)
    validate_provider_credentials("openai", disabled_config)

    # Enabled provider with valid key succeeds
    good_config = ProviderConfig(enabled=True, api_key=SecretStr("sk-test"))
    validate_provider_credentials("openai", good_config)

    # Ollama does not require an api_key
    ollama_config = ProviderConfig(enabled=True, api_key=None)
    validate_provider_credentials("ollama", ollama_config)


def test_secret_str_masking():
    cfg = ProviderConfig(api_key=SecretStr("super-secret-key"))
    # Repr must not leak the raw key
    assert "super-secret-key" not in repr(cfg)
    assert "super-secret-key" not in str(cfg)
    assert cfg.api_key.get_secret_value() == "super-secret-key"
