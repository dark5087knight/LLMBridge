"""Redaction utilities to prevent credentials and secrets from leaking into logs or errors."""

import re
from typing import Any

# Patterns matching standard provider API keys and authorization headers
_SECRET_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9_-]{10,}", re.IGNORECASE),
    re.compile(r"gsk_[A-Za-z0-9_-]{10,}", re.IGNORECASE),
    re.compile(r"AIzaSy[A-Za-z0-9_-]{20,}", re.IGNORECASE),
    re.compile(r"Bearer\s+[A-Za-z0-9_.\-]+", re.IGNORECASE),
]

_SENSITIVE_KEYS = {
    "api_key",
    "apikey",
    "authorization",
    "password",
    "secret",
    "token",
    "access_token",
}


def redact_text(text: str) -> str:
    """Redact known secret patterns in a string."""
    if not isinstance(text, str):
        return text

    sanitized = text
    for pattern in _SECRET_PATTERNS:
        sanitized = pattern.sub("[REDACTED]", sanitized)
    return sanitized


def redact_dict(data: dict[str, Any]) -> dict[str, Any]:
    """Recursively redact dictionary keys and values."""
    redacted: dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(key, str) and key.lower() in _SENSITIVE_KEYS:
            redacted[key] = "[REDACTED]"
        elif isinstance(value, dict):
            redacted[key] = redact_dict(value)
        elif isinstance(value, list):
            redacted[key] = [
                redact_dict(item) if isinstance(item, dict)
                else (redact_text(item) if isinstance(item, str) else item)
                for item in value
            ]
        elif isinstance(value, str):
            redacted[key] = redact_text(value)
        else:
            redacted[key] = value
    return redacted


def redaction_processor(_: Any, __: str, event_dict: dict[str, Any]) -> dict[str, Any]:
    """Structlog processor that redacts secrets from event dictionary."""
    return redact_dict(event_dict)
