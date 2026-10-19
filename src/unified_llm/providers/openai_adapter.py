"""OpenAI provider adapter implementation using official openai SDK."""

import time
from typing import ClassVar
import httpx
import openai
from openai import AsyncOpenAI

from unified_llm.config.settings import ProviderConfig
from unified_llm.core.capabilities import Capabilities
from unified_llm.core.context import CallContext
from unified_llm.core.errors import (
    BadProviderResponse,
    ContentRejected,
    ContextLengthExceeded,
    InvalidRequest,
    ModelNotFound,
    ProviderAuthError,
    ProviderConnectionError,
    ProviderError,
    ProviderTimeout,
    ProviderUnavailable,
    RateLimited,
)
from unified_llm.core.schemas import MessageRequest, MessageResponse
from unified_llm.providers._openai_style import from_openai_response, to_openai_payload
from unified_llm.providers.base import HealthStatus, ProviderAdapter


class OpenAIAdapter(ProviderAdapter):
    """Adapter for OpenAI models using AsyncOpenAI."""

    name: ClassVar[str] = "openai"
    capabilities: ClassVar[Capabilities] = Capabilities(
        system_role=True,
        streaming=True,
        json_mode=True,
        seed=True,
        top_k=False,
        stop_sequences=True,
        penalties=True,
        multimodal_input=False,
        tool_calling=False,
        supported_params=frozenset({
            "temperature",
            "top_p",
            "max_tokens",
            "stop",
            "seed",
            "presence_penalty",
            "frequency_penalty",
            "response_format",
        }),
    )

    def __init__(self, config: ProviderConfig) -> None:
        super().__init__(config)
        self._client: AsyncOpenAI | None = None

    async def startup(self) -> None:
        api_key = self.config.api_key.get_secret_value() if self.config.api_key else "missing"
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=self.config.base_url,
            max_retries=0,  # Retries are governed by ULIL core
            timeout=self.config.timeout_s,
        )

    async def shutdown(self) -> None:
        if self._client:
            await self._client.close()
            self._client = None

    async def generate(self, req: MessageRequest, ctx: CallContext) -> MessageResponse:
        if not self._client:
            await self.startup()
            assert self._client is not None

        start_time = time.monotonic()
        payload = to_openai_payload(req)

        try:
            native = await self._client.chat.completions.create(
                **payload,
                timeout=ctx.remaining_s(),
            )
        except Exception as exc:
            raise self.translate_error(exc) from exc

        latency_ms = int((time.monotonic() - start_time) * 1000)
        return from_openai_response(native, req, ctx, latency_ms=latency_ms)

    async def health_check(self) -> HealthStatus:
        if not self._client:
            await self.startup()
            assert self._client is not None
        try:
            # Query models list with low timeout
            await self._client.models.list(timeout=5.0)
            return HealthStatus.healthy("OpenAI API reachable")
        except Exception as exc:
            return HealthStatus.unhealthy(f"OpenAI health check failed: {type(exc).__name__}")

    def translate_error(self, exc: Exception) -> ProviderError:
        """Map OpenAI SDK exceptions to ULIL error taxonomy."""
        if isinstance(exc, openai.AuthenticationError):
            return ProviderAuthError("OpenAI authentication failed. Invalid API key or credentials.", provider=self.name, provider_status=401)

        if isinstance(exc, openai.PermissionDeniedError):
            return ProviderAuthError("OpenAI permission denied.", provider=self.name, provider_status=403)

        if isinstance(exc, openai.NotFoundError):
            return ModelNotFound("Requested model not found on OpenAI.", provider=self.name, provider_status=404)

        if isinstance(exc, openai.RateLimitError):
            retry_after: float | None = None
            if hasattr(exc, "response") and exc.response is not None:
                header_val = exc.response.headers.get("retry-after")
                if header_val:
                    try:
                        retry_after = float(header_val)
                    except ValueError:
                        pass
            return RateLimited(
                "OpenAI rate limit exceeded.",
                provider=self.name,
                provider_status=429,
                retry_after_s=retry_after,
            )

        if isinstance(exc, openai.BadRequestError):
            msg = str(exc).lower()
            if any(k in msg for k in ("context_length", "maximum context", "too many tokens")):
                return ContextLengthExceeded("Prompt exceeds model context length.", provider=self.name, provider_status=400)
            if any(k in msg for k in ("content filter", "safety policy", "policy violation")):
                return ContentRejected("Prompt rejected by OpenAI content filter.", provider=self.name, provider_status=400)
            return InvalidRequest("Bad request sent to OpenAI.", provider=self.name, provider_status=400)

        if isinstance(exc, openai.APITimeoutError) or isinstance(exc, httpx.TimeoutException):
            return ProviderTimeout("OpenAI request timed out.", provider=self.name, provider_status=504)

        if isinstance(exc, openai.APIConnectionError) or isinstance(exc, httpx.ConnectError):
            return ProviderConnectionError("Could not connect to OpenAI API.", provider=self.name, provider_status=502)

        if isinstance(exc, openai.InternalServerError):
            return ProviderUnavailable("OpenAI internal server error.", provider=self.name, provider_status=500)

        if isinstance(exc, openai.APIStatusError):
            status = getattr(exc, "status_code", 502)
            if status == 429:
                return RateLimited("OpenAI rate limit exceeded.", provider=self.name, provider_status=429)
            if status >= 500:
                return ProviderUnavailable("OpenAI service unavailable.", provider=self.name, provider_status=status)
            return BadProviderResponse(f"OpenAI error with status {status}.", provider=self.name, provider_status=status)

        return BadProviderResponse(f"Unexpected OpenAI error: {type(exc).__name__}", provider=self.name)
