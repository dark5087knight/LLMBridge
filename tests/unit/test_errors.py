"""Unit tests for error taxonomy, error envelope formatting, and secret redaction."""

from unified_llm.core.errors import (
    ContextLengthExceeded,
    InvalidRequest,
    ModelNotFound,
    ProviderAuthError,
    ProviderConnectionError,
    ProviderTimeout,
    ProviderUnavailable,
    RateLimited,
    UnknownProvider,
    UnsupportedParameter,
)
from unified_llm.observability.redaction import redact_dict, redact_text


def test_error_taxonomy_properties():
    # Client errors
    inv = InvalidRequest("Bad format")
    assert inv.type == "invalid_request"
    assert inv.http_status == 422
    assert inv.retryable is False

    unsupp = UnsupportedParameter("Param not supported")
    assert unsupp.type == "unsupported_parameter"
    assert unsupp.http_status == 422
    assert unsupp.retryable is False

    unk = UnknownProvider("No such provider")
    assert unk.type == "unknown_provider"
    assert unk.http_status == 404
    assert unk.retryable is False

    # Provider errors
    auth = ProviderAuthError("Auth failed")
    assert auth.type == "provider_auth_error"
    assert auth.http_status == 502
    assert auth.retryable is False

    nf = ModelNotFound("Model missing")
    assert nf.type == "model_not_found"
    assert nf.http_status == 404
    assert nf.retryable is False

    ctx = ContextLengthExceeded("Prompt too long")
    assert ctx.type == "context_length_exceeded"
    assert ctx.http_status == 413
    assert ctx.retryable is False

    rl = RateLimited("Rate limit hit", retry_after_s=2.5)
    assert rl.type == "rate_limited"
    assert rl.http_status == 429
    assert rl.retryable is True
    assert rl.retry_after_s == 2.5

    timeout = ProviderTimeout("Call timed out")
    assert timeout.type == "provider_timeout"
    assert timeout.http_status == 504
    assert timeout.retryable is True

    conn = ProviderConnectionError("DNS failure")
    assert conn.type == "provider_connection_error"
    assert conn.http_status == 502
    assert conn.retryable is True

    unavail = ProviderUnavailable("503 Overloaded")
    assert unavail.type == "provider_unavailable"
    assert unavail.http_status == 503
    assert unavail.retryable is True


def test_to_envelope():
    err = RateLimited(
        message="Too many requests",
        provider="groq",
        provider_status=429,
        retry_after_s=3.0,
        details={"limit": 60},
    )
    envelope = err.to_envelope(request_id="req_test_abc")

    assert "error" in envelope
    e = envelope["error"]
    assert e["type"] == "rate_limited"
    assert e["message"] == "Too many requests"
    assert e["provider"] == "groq"
    assert e["retryable"] is True
    assert e["retry_after_s"] == 3.0
    assert e["provider_status"] == 429
    assert e["request_id"] == "req_test_abc"
    assert e["details"] == {"limit": 60}


def test_redact_text():
    raw = "My OpenAI key is sk-1234567890abcdef and token is Bearer secret_token_xyz."
    redacted = redact_text(raw)
    assert "sk-1234567890abcdef" not in redacted
    assert "secret_token_xyz" not in redacted
    assert "[REDACTED]" in redacted


def test_redact_dict():
    data = {
        "api_key": "sk-secret123456",
        "nested": {
            "authorization": "Bearer token123",
            "safe_field": "hello world",
        },
        "list_field": ["safe", "sk-anotherkey12345"],
    }
    cleaned = redact_dict(data)
    assert cleaned["api_key"] == "[REDACTED]"
    assert cleaned["nested"]["authorization"] == "[REDACTED]"
    assert cleaned["nested"]["safe_field"] == "hello world"
    assert cleaned["list_field"][0] == "safe"
    assert "sk-anotherkey12345" not in cleaned["list_field"][1]
