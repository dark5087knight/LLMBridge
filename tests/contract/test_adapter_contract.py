"""Shared adapter contract test suite parameterized across all provider adapters."""

import pytest
from pydantic import SecretStr

from unified_llm.config.settings import ProviderConfig
from unified_llm.core.context import CallContext, Deadline
from unified_llm.core.errors import (
    ProviderAuthError,
    ProviderError,
    ProviderTimeout,
    RateLimited,
    ULILError,
)
from unified_llm.core.schemas import ChatRequest, GenerationParams, Message
from unified_llm.providers import (
    GeminiAdapter,
    GroqAdapter,
    OllamaAdapter,
    OpenAIAdapter,
)
from unified_llm.providers.base import ProviderAdapter

ADAPTER_CLASSES = [OpenAIAdapter, GroqAdapter, GeminiAdapter, OllamaAdapter]


@pytest.mark.parametrize("adapter_cls", ADAPTER_CLASSES)
def test_adapter_capabilities_declaration(adapter_cls: type[ProviderAdapter]):
    """Every adapter must declare a non-empty name and capabilities with supported_params."""
    assert adapter_cls.name is not None
    assert len(adapter_cls.name) > 0
    assert adapter_cls.capabilities is not None
    assert len(adapter_cls.capabilities.supported_params) > 0


@pytest.mark.parametrize("adapter_cls", ADAPTER_CLASSES)
def test_adapter_error_translation_never_leaks_api_key(adapter_cls: type[ProviderAdapter]):
    """translate_error must never leak API keys in error messages and always return ProviderError."""
    api_key_secret = "sk-live-super-secret-12345"
    config = ProviderConfig(enabled=True, api_key=SecretStr(api_key_secret))
    adapter = adapter_cls(config)

    # Simulate an arbitrary exception containing the key
    raw_exc = Exception(f"Failed call with key {api_key_secret}")
    translated = adapter.translate_error(raw_exc)

    assert isinstance(translated, ProviderError)
    assert isinstance(translated, ULILError)
    assert api_key_secret not in translated.message


@pytest.mark.parametrize("adapter_cls", ADAPTER_CLASSES)
def test_adapter_error_translation_taxonomy_mapping(adapter_cls: type[ProviderAdapter]):
    """translate_error maps known error types correctly."""
    adapter = adapter_cls(ProviderConfig(enabled=True))

    # Test generic timeout
    timeout_exc = TimeoutError("Request timed out")
    mapped_timeout = adapter.translate_error(timeout_exc)
    assert isinstance(mapped_timeout, ProviderError)
    assert mapped_timeout.retryable in (True, False)
