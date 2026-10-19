"""OpenAI provider adapter implementation using official openai SDK."""

import time
from typing import AsyncIterator, ClassVar
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
    ModelNotFound,
    ProviderAuthError,
    ProviderConnectionError,
    ProviderError,
    ProviderTimeout,
    ProviderUnavailable,
    RateLimited,
)
from unified_llm.core.schemas import MessageRequest, MessageResponse, StreamEvent
from unified_llm.providers._openai_style import (
    from_openai_finish_reason,
    from_openai_response,
    to_openai_payload,
)
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
        tool_calling=True,
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

    def _resolve_api_key(self, req: MessageRequest, ctx: CallContext) -> str | None:
        if req.api_key:
            return req.api_key
        if req.provider_options and req.provider_options.get("api_key"):
            return str(req.provider_options["api_key"])
        if ctx.provider_key:
            return ctx.provider_key
        if self.config.api_key:
            return self.config.api_key.get_secret_value()
        return None

    async def generate(self, req: MessageRequest, ctx: CallContext) -> MessageResponse:
        api_key = self._resolve_api_key(req, ctx)
        if not api_key or api_key == "missing":
            raise ProviderAuthError(
                "No OpenAI API key found. Provide it in the request via 'api_key', 'provider_options.api_key', or 'Authorization: Bearer <key>' / 'X-Provider-Key' header.",
                provider="openai",
                provider_status=401,
            )

        client = (
            self._client
            if (self._client and self.config.api_key and api_key == self.config.api_key.get_secret_value())
            else AsyncOpenAI(
                api_key=api_key,
                base_url=self.config.base_url,
                max_retries=0,
                timeout=self.config.timeout_s,
            )
        )

        start_time = time.monotonic()
        payload = to_openai_payload(req)

        try:
            native = await client.chat.completions.create(
                **payload,
                timeout=ctx.remaining_s(),
            )
        except Exception as exc:
            raise self.translate_error(exc) from exc
        finally:
            if client != self._client:
                await client.close()

        latency_ms = int((time.monotonic() - start_time) * 1000)
        return from_openai_response(native, req, ctx, latency_ms=latency_ms)

    async def stream(self, req: MessageRequest, ctx: CallContext) -> AsyncIterator[StreamEvent]:
        api_key = self._resolve_api_key(req, ctx)
        if not api_key or api_key == "missing":
            raise ProviderAuthError(
                "No OpenAI API key found. Provide it in the request via 'api_key', 'provider_options.api_key', or 'Authorization: Bearer <key>' / 'X-Provider-Key' header.",
                provider="openai",
                provider_status=401,
            )

        client = (
            self._client
            if (self._client and self.config.api_key and api_key == self.config.api_key.get_secret_value())
            else AsyncOpenAI(
                api_key=api_key,
                base_url=self.config.base_url,
                max_retries=0,
                timeout=self.config.timeout_s,
            )
        )

        payload = to_openai_payload(req)
        payload["stream"] = True
        payload["stream_options"] = {"include_usage": True}

        yield StreamEvent(
            event="start",
            data={
                "id": ctx.request_id,
                "provider": self.name,
                "model": req.model,
            },
        )

        finish_reason = "stop"
        try:
            response_stream = await client.chat.completions.create(
                **payload,
                timeout=ctx.remaining_s(),
            )
            async for chunk in response_stream:
                if chunk.choices:
                    choice = chunk.choices[0]
                    delta = choice.delta
                    if delta and delta.content:
                        yield StreamEvent(
                            event="delta",
                            data={
                                "content": delta.content,
                                "role": delta.role or "assistant",
                            },
                        )
                    if choice.finish_reason:
                        finish_reason = from_openai_finish_reason(choice.finish_reason)

                if getattr(chunk, "usage", None):
                    usage = chunk.usage
                    yield StreamEvent(
                        event="usage",
                        data={
                            "input_tokens": usage.prompt_tokens,
                            "output_tokens": usage.completion_tokens,
                            "total_tokens": usage.total_tokens,
                        },
                    )
        except Exception as exc:
            raise self.translate_error(exc) from exc
        finally:
            if client != self._client:
                await client.close()

        yield StreamEvent(
            event="end",
            data={
                "finish_reason": finish_reason,
            },
        )

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
            return BadProviderResponse("Bad request sent to OpenAI.", provider=self.name, provider_status=400, retryable=False)

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
