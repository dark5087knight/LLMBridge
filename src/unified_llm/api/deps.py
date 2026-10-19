"""Dependency injection providers for FastAPI routes."""

from typing import cast
from fastapi import Request

from unified_llm.config.settings import ServiceSettings
from unified_llm.core.context import RequestContext
from unified_llm.core.registry import ProviderRegistry
from unified_llm.core.service import LLMService


def get_service(request: Request) -> LLMService:
    """Retrieve the LLMService singleton from application state."""
    return cast(LLMService, request.app.state.service)


def get_registry(request: Request) -> ProviderRegistry:
    """Retrieve the ProviderRegistry singleton from application state."""
    return cast(ProviderRegistry, request.app.state.registry)


def get_settings(request: Request) -> ServiceSettings:
    """Retrieve the ServiceSettings singleton from application state."""
    return cast(ServiceSettings, request.app.state.settings)


def get_request_context(request: Request) -> RequestContext:
    """Build a RequestContext from the inbound HTTP request."""
    req_id = getattr(request.state, "request_id", "req_unknown")
    trace_id = request.headers.get("X-Trace-ID")
    caller = request.headers.get("X-Caller-ID")

    # Extract provider key from headers if passed
    provider_key = (
        request.headers.get("X-Provider-Key")
        or request.headers.get("X-API-Key")
        or request.headers.get("X-Groq-API-Key")
    )
    if not provider_key:
        auth = request.headers.get("Authorization")
        if auth and auth.startswith("Bearer "):
            provider_key = auth.split(" ", 1)[1].strip()

    return RequestContext(
        request_id=req_id,
        trace_id=trace_id,
        caller=caller,
        provider_key=provider_key,
    )
