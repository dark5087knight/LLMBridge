"""Ollama provider adapter implementation using HTTPX async client."""

import time
from typing import Any, ClassVar, Literal
import httpx

from unified_llm.config.settings import ProviderConfig
from unified_llm.core.capabilities import Capabilities, Warning
from unified_llm.core.context import CallContext
from unified_llm.core.errors import (
    BadProviderResponse,
    ContextLengthExceeded,
    InvalidRequest,
    ModelNotFound,
    ProviderConnectionError,
    ProviderError,
    ProviderTimeout,
    ProviderUnavailable,
    RateLimited,
)
from unified_llm.core.schemas import (
    ChatRequest,
    ChatResponse,
    Message,
    ResponseMeta,
    Usage,
)
from unified_llm.providers.base import HealthStatus, ProviderAdapter


def from_ollama_finish_reason(
    done_reason: str | None,
) -> Literal["stop", "length", "content_filter", "error", "other"]:
    """Map Ollama done_reason to canonical finish reason."""
    if not done_reason:
        return "stop"
    dr = done_reason.lower()
    if dr == "stop":
        return "stop"
    if dr in ("length", "max_tokens"):
        return "length"
    return "other"


def to_ollama_payload(req: ChatRequest, keep_alive: str | None = None) -> dict[str, Any]:
    """Pure function mapping a ChatRequest to Ollama /api/chat payload."""
    payload: dict[str, Any] = {
        "model": req.model,
        "messages": [{"role": m.role, "content": m.content} for m in req.messages],
        "stream": False,
    }

    if keep_alive:
        payload["keep_alive"] = keep_alive

    options: dict[str, Any] = {}
    p = req.params
    if p.temperature is not None:
        options["temperature"] = p.temperature
    if p.top_p is not None:
        options["top_p"] = p.top_p
    if p.top_k is not None:
        options["top_k"] = p.top_k
    if p.max_tokens is not None:
        options["num_predict"] = p.max_tokens
    if p.seed is not None:
        options["seed"] = p.seed
    if p.stop is not None:
        options["stop"] = p.stop

    if options:
        payload["options"] = options

    if p.response_format == "json":
        payload["format"] = "json"

    # Merge provider_options escape hatch
    if req.provider_options:
        payload.update(req.provider_options)

    return payload


def from_ollama_response(
    native: dict[str, Any],
    req: ChatRequest,
    ctx: CallContext,
    latency_ms: int = 0,
    warnings: list[Warning] | None = None,
) -> ChatResponse:
    """Pure function normalizing Ollama response to ChatResponse."""
    msg_dict = native.get("message", {})
    content = msg_dict.get("content", "")
    role = msg_dict.get("role", "assistant")

    finish_reason = from_ollama_finish_reason(native.get("done_reason"))

    prompt_eval_count = native.get("prompt_eval_count")
    eval_count = native.get("eval_count")
    total_tokens = None
    if prompt_eval_count is not None and eval_count is not None:
        total_tokens = prompt_eval_count + eval_count

    return ChatResponse(
        id=ctx.request_id,
        provider=req.provider,
        model=req.model,
        model_served=native.get("model"),
        message=Message(role=role, content=content),
        finish_reason=finish_reason,
        usage=Usage(
            input_tokens=prompt_eval_count,
            output_tokens=eval_count,
            total_tokens=total_tokens,
        ),
        warnings=warnings or [],
        meta=ResponseMeta(latency_ms=latency_ms, attempts=ctx.attempt),
        raw=native if req.options.include_raw else None,
    )


class OllamaAdapter(ProviderAdapter):
    """Adapter for Ollama local backend using HTTPX."""

    name: ClassVar[str] = "ollama"
    capabilities: ClassVar[Capabilities] = Capabilities(
        system_role=True,
        streaming=True,
        json_mode=True,
        seed=True,
        top_k=True,
        stop_sequences=True,
        penalties=False,
        multimodal_input=False,
        tool_calling=False,
        supported_params=frozenset({
            "temperature",
            "top_p",
            "top_k",
            "max_tokens",
            "stop",
            "seed",
            "response_format",
        }),
    )

    def __init__(self, config: ProviderConfig) -> None:
        super().__init__(config)
        self._client: httpx.AsyncClient | None = None

    async def startup(self) -> None:
        base_url = self.config.base_url or "http://localhost:11434"
        self._client = httpx.AsyncClient(
            base_url=str(base_url).rstrip("/"),
            timeout=httpx.Timeout(self.config.timeout_s, connect=self.config.connect_timeout_s),
        )

    async def shutdown(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    async def generate(self, req: ChatRequest, ctx: CallContext) -> ChatResponse:
        if not self._client:
            await self.startup()
            assert self._client is not None

        keep_alive = self.config.extra.get("keep_alive")
        payload = to_ollama_payload(req, keep_alive=keep_alive)
        start_time = time.monotonic()

        try:
            resp = await self._client.post(
                "/api/chat",
                json=payload,
                timeout=ctx.remaining_s(),
            )
            if resp.is_error:
                resp.read()
                raise httpx.HTTPStatusError(
                    message=f"Ollama returned HTTP {resp.status_code}: {resp.text}",
                    request=resp.request,
                    response=resp,
                )
            data = resp.json()
        except Exception as exc:
            raise self.translate_error(exc) from exc

        latency_ms = int((time.monotonic() - start_time) * 1000)
        return from_ollama_response(data, req, ctx, latency_ms=latency_ms)

    async def health_check(self) -> HealthStatus:
        if not self._client:
            await self.startup()
            assert self._client is not None
        try:
            resp = await self._client.get("/api/tags", timeout=5.0)
            if resp.status_code == 200:
                return HealthStatus.healthy("Ollama service reachable")
            return HealthStatus.unhealthy(f"Ollama returned HTTP {resp.status_code}")
        except Exception as exc:
            return HealthStatus.unhealthy(f"Ollama health check failed: {type(exc).__name__}")

    def translate_error(self, exc: Exception) -> ProviderError:
        """Map HTTPX and Ollama errors to ULIL error taxonomy."""
        if isinstance(exc, httpx.ConnectError):
            return ProviderConnectionError("Could not connect to Ollama server.", provider=self.name, provider_status=502)

        if isinstance(exc, httpx.TimeoutException):
            return ProviderTimeout("Ollama request timed out.", provider=self.name, provider_status=504)

        if isinstance(exc, httpx.HTTPStatusError):
            status = exc.response.status_code
            if status == 404:
                return ModelNotFound("Model not found in Ollama. Ensure model has been pulled.", provider=self.name, provider_status=404)
            if status == 429:
                return RateLimited("Ollama rate limit exceeded.", provider=self.name, provider_status=429)
            if status >= 500:
                return ProviderUnavailable("Ollama server error.", provider=self.name, provider_status=status)
            return BadProviderResponse(f"Ollama returned HTTP {status}.", provider=self.name, provider_status=status)

        return BadProviderResponse(f"Unexpected Ollama error: {type(exc).__name__}", provider=self.name)
