"""Unit tests for canonical request and response schemas."""

import pytest
from pydantic import ValidationError

from unified_llm.core.schemas import (
    ChatRequest,
    ChatResponse,
    GenerationParams,
    Message,
    RequestOptions,
    ResponseMeta,
    Usage,
)


def test_valid_message():
    msg = Message(role="user", content="Hello world")
    assert msg.role == "user"
    assert msg.content == "Hello world"


def test_invalid_message_role():
    with pytest.raises(ValidationError):
        Message(role="invalid_role", content="Hello")  # type: ignore


def test_generation_params_forbid_extra():
    # Canonical parameters succeed
    params = GenerationParams(temperature=0.7, max_tokens=100, top_p=0.9)
    assert params.temperature == 0.7

    # Extra parameters must be strictly forbidden
    with pytest.raises(ValidationError) as exc_info:
        GenerationParams(temperature=0.7, unknown_foo="bar")  # type: ignore
    assert "extra_forbidden" in str(exc_info.value)


def test_generation_params_value_ranges():
    # temperature < 0 should fail
    with pytest.raises(ValidationError):
        GenerationParams(temperature=-0.1)

    # top_p <= 0 or > 1 should fail
    with pytest.raises(ValidationError):
        GenerationParams(top_p=0.0)
    with pytest.raises(ValidationError):
        GenerationParams(top_p=1.1)

    # max_tokens < 1 should fail
    with pytest.raises(ValidationError):
        GenerationParams(max_tokens=0)


def test_request_options():
    opts = RequestOptions(timeout_s=30.0, max_retries=2, strict_params=True, include_raw=True)
    assert opts.timeout_s == 30.0
    assert opts.max_retries == 2
    assert opts.strict_params is True
    assert opts.include_raw is True

    # Negative timeout or retries > 5 should fail
    with pytest.raises(ValidationError):
        RequestOptions(timeout_s=-1.0)
    with pytest.raises(ValidationError):
        RequestOptions(max_retries=6)


def test_chat_request_validation():
    # Requires at least one message
    with pytest.raises(ValidationError):
        ChatRequest(provider="openai", model="gpt-4o", messages=[])

    req = ChatRequest(
        provider="openai",
        model="gpt-4o",
        messages=[Message(role="user", content="Hi")],
        params=GenerationParams(temperature=0.5),
        options=RequestOptions(timeout_s=15.0),
        provider_options={"num_ctx": 4096},
        metadata={"caller": "test-agent"},
    )
    assert req.provider == "openai"
    assert req.model == "gpt-4o"
    assert len(req.messages) == 1
    assert req.provider_options["num_ctx"] == 4096


def test_chat_response_construction():
    resp = ChatResponse(
        id="req_test_123",
        provider="openai",
        model="gpt-4o-mini",
        model_served="gpt-4o-mini-2024-07-18",
        message=Message(role="assistant", content="Response text"),
        finish_reason="stop",
        usage=Usage(input_tokens=10, output_tokens=20, total_tokens=30),
        meta=ResponseMeta(latency_ms=500, attempts=1),
    )
    data = resp.model_dump()
    assert data["id"] == "req_test_123"
    assert data["finish_reason"] == "stop"
    assert data["usage"]["total_tokens"] == 30
