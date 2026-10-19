"""API routes package."""

from unified_llm.api.routes.health import router as health_router
from unified_llm.api.routes.messages import router as messages_router
from unified_llm.api.routes.providers import router as providers_router

# Backwards compatibility alias
chat_router = messages_router

__all__ = ["messages_router", "chat_router", "health_router", "providers_router"]
