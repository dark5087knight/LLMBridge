"""FastAPI application factory and lifecycle management for ULIL."""

from contextlib import asynccontextmanager
from typing import AsyncIterator
from fastapi import FastAPI

from unified_llm.api.error_handlers import register_error_handlers
from unified_llm.api.middleware import (
    InboundAuthMiddleware,
    PayloadSizeLimitMiddleware,
    RequestIdMiddleware,
)
from unified_llm.api.routes import health_router, messages_router, providers_router
from unified_llm.config.loader import load_settings
from unified_llm.config.settings import ServiceSettings
from unified_llm.core.registry import ProviderRegistry
from unified_llm.core.retry import RetryPolicy
from unified_llm.core.service import LLMService
from unified_llm.observability.logging import configure_logging


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifecycle: startup initialization and graceful shutdown."""
    settings: ServiceSettings = getattr(app.state, "settings", None) or load_settings()
    app.state.settings = settings

    configure_logging(log_level=settings.log_level, log_format=settings.log_format)

    # Build or reuse registry
    registry: ProviderRegistry | None = getattr(app.state, "registry", None)
    if registry is None:
        registry = ProviderRegistry.from_settings(settings)
        app.state.registry = registry

    await registry.startup_all()

    # Build or reuse service
    if getattr(app.state, "service", None) is None:
        app.state.service = LLMService(
            registry=registry,
            policy=RetryPolicy.from_settings(settings),
            settings=settings,
        )

    yield

    await registry.shutdown_all()


def create_app(settings: ServiceSettings | None = None) -> FastAPI:
    """Create and configure the FastAPI application."""
    if settings is None:
        try:
            settings = load_settings()
        except Exception:
            # Fallback to defaults if env/yaml has startup validation issues before test overrides
            settings = ServiceSettings()

    app = FastAPI(
        title="Unified LLM Integration Layer (ULIL)",
        version="0.1.0",
        description="Unified middleware service for multi-provider LLM integrations",
        lifespan=lifespan,
    )
    app.state.settings = settings

    registry = ProviderRegistry.from_settings(settings)
    app.state.registry = registry
    app.state.service = LLMService(
        registry=registry,
        policy=RetryPolicy.from_settings(settings),
        settings=settings,
    )

    # Add middlewares (executed in reverse of addition order: RequestId -> Auth -> PayloadSize)
    app.add_middleware(PayloadSizeLimitMiddleware, max_bytes=settings.max_request_bytes)
    app.add_middleware(InboundAuthMiddleware, settings=settings)
    app.add_middleware(RequestIdMiddleware)

    # Error handling
    register_error_handlers(app)

    # Routers
    app.include_router(health_router)
    app.include_router(providers_router)
    app.include_router(messages_router)

    return app


app = create_app()
