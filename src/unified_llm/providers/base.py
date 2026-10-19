"""ProviderAdapter abstract base class and health status contract."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import AsyncIterator, ClassVar

from unified_llm.config.settings import ProviderConfig
from unified_llm.core.capabilities import Capabilities
from unified_llm.core.context import CallContext
from unified_llm.core.errors import ProviderError, UnsupportedFeature
from unified_llm.core.schemas import MessageRequest, MessageResponse, StreamEvent


@dataclass
class HealthStatus:
    """Provider health status representation."""

    ok: bool
    message: str = "ok"
    details: dict[str, str] | None = None

    @classmethod
    def healthy(cls, message: str = "healthy") -> "HealthStatus":
        return cls(ok=True, message=message)

    @classmethod
    def unhealthy(cls, message: str) -> "HealthStatus":
        return cls(ok=False, message=message)

    @classmethod
    def unknown(cls) -> "HealthStatus":
        return cls(ok=True, message="health check not implemented")


class ProviderAdapter(ABC):
    """Abstract contract that every provider adapter must implement.

    Rules:
    1. Raise only ProviderError subclasses from generate/stream.
    2. Set SDK-internal retries to zero (max_retries=0).
    3. Apply per-attempt timeout from CallContext.
    4. Own a long-lived SDK client created in startup(), closed in shutdown().
    5. Never log prompts, completions, or credentials.
    6. Declare capabilities honestly.
    """

    name: ClassVar[str]
    capabilities: ClassVar[Capabilities]

    def __init__(self, config: ProviderConfig) -> None:
        self.config = config

    async def startup(self) -> None:
        """Initialize connection pools and SDK clients."""
        pass

    async def shutdown(self) -> None:
        """Close connection pools and SDK clients."""
        pass

    @abstractmethod
    async def generate(self, req: MessageRequest, ctx: CallContext) -> MessageResponse:
        """Map canonical request -> execute provider call -> normalize response.

        Must raise only ProviderError subclasses on failure.
        """
        ...

    def stream(self, req: MessageRequest, ctx: CallContext) -> AsyncIterator[StreamEvent]:
        """Stream completion tokens as Server-Sent Events."""
        raise UnsupportedFeature(f"Streaming is not supported by provider '{self.name}'.", provider=self.name)

    async def health_check(self) -> HealthStatus:
        """Execute a deep health check for the provider backend."""
        return HealthStatus.unknown()

    @abstractmethod
    def translate_error(self, exc: Exception) -> ProviderError:
        """Map SDK-specific exceptions to the standardized ULIL error taxonomy."""
        ...
