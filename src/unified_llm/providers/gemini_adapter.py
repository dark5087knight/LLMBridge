"""Gemini provider adapter implementation using official google-genai SDK."""

import time
from typing import Any, ClassVar, Literal, cast
from google import genai
from google.genai import errors as genai_errors, types

from unified_llm.config.settings import ProviderConfig
from unified_llm.core.capabilities import Capabilities, Warning
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
from unified_llm.core.schemas import (
    Message,
    MessageRequest,
    MessageResponse,
    ResponseMeta,
    Usage,
)
from unified_llm.providers.base import HealthStatus, ProviderAdapter


def from_gemini_finish_reason(
    reason: str | None,
) -> Literal["stop", "length", "content_filter", "error", "other"]:
    """Map Gemini finish reason enum string to canonical finish reason."""
    if not reason:
        return "stop"
    r = str(reason).upper()
    if r in ("STOP", "FINISH_REASON_STOP"):
        return "stop"
    if r in ("MAX_TOKENS", "LENGTH", "FINISH_REASON_MAX_TOKENS"):
        return "length"
    if any(k in r for k in ("SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED")):
        return "content_filter"
    if "OTHER" in r or "ERROR" in r:
        return "error"
    return "other"


def to_gemini_payload(
    req: MessageRequest,
) -> tuple[list[types.Content], types.GenerateContentConfig, list[Warning]]:
    """Pure function mapping a MessageRequest to Gemini contents and GenerateContentConfig."""
    warnings: list[Warning] = []
    system_texts: list[str] = []
    contents: list[types.Content] = []

    for msg in req.messages:
        content_str = msg.content if isinstance(msg.content, str) else str(msg.content)
        if msg.role == "system":
            system_texts.append(content_str)
        elif msg.role == "assistant":
            contents.append(
                types.Content(
                    role="model",
                    parts=[types.Part.from_text(text=content_str)],
                )
            )
        else:  # user
            contents.append(
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=content_str)],
                )
            )

    if len(system_texts) > 1:
        warnings.append(
            Warning(
                code="multiple_system_prompts_joined",
                param="messages",
                message="Multiple system messages were concatenated with newlines for Gemini.",
            )
        )

    system_instruction = "\n\n".join(system_texts) if system_texts else None

    # Config options
    p = req.params
    config_kwargs: dict[str, Any] = {}
    if system_instruction:
        config_kwargs["system_instruction"] = system_instruction
    if p.temperature is not None:
        config_kwargs["temperature"] = p.temperature
    if p.top_p is not None:
        config_kwargs["top_p"] = p.top_p
    if p.top_k is not None:
        config_kwargs["top_k"] = p.top_k
    if p.max_tokens is not None:
        config_kwargs["max_output_tokens"] = p.max_tokens
    if p.stop is not None:
        config_kwargs["stop_sequences"] = p.stop
    if p.response_format == "json":
        config_kwargs["response_mime_type"] = "application/json"

    # Merge provider_options if provided
    if req.provider_options:
        config_kwargs.update(req.provider_options)

    config = types.GenerateContentConfig(**config_kwargs)
    return contents, config, warnings


def from_gemini_response(
    native: Any,
    req: MessageRequest,
    ctx: CallContext,
    latency_ms: int = 0,
    warnings: list[Warning] | None = None,
) -> MessageResponse:
    """Pure function normalizing a Gemini response to canonical MessageResponse."""
    content_text = ""
    finish_reason_raw = None

    candidates = getattr(native, "candidates", None)
    if candidates and len(candidates) > 0:
        candidate = candidates[0]
        finish_reason_raw = getattr(candidate, "finish_reason", None)
        if candidate.content and candidate.content.parts:
            parts_text = []
            for part in candidate.content.parts:
                if hasattr(part, "text") and part.text:
                    parts_text.append(part.text)
            content_text = "".join(parts_text)

    finish_reason = from_gemini_finish_reason(finish_reason_raw)

    # Usage
    usage_meta = getattr(native, "usage_metadata", None)
    input_tokens = getattr(usage_meta, "prompt_token_count", None) if usage_meta else None
    output_tokens = getattr(usage_meta, "candidates_token_count", None) if usage_meta else None
    total_tokens = getattr(usage_meta, "total_token_count", None) if usage_meta else None

    # model served
    model_version = getattr(native, "model_version", None)

    raw_dict = None
    if req.options.include_raw:
        if hasattr(native, "model_dump"):
            raw_dict = native.model_dump()
        elif hasattr(native, "__dict__"):
            raw_dict = dict(native.__dict__)
        else:
            raw_dict = {"response": str(native)}

    return MessageResponse(
        id=ctx.request_id,
        provider=req.provider,
        model=req.model,
        model_served=model_version,
        message=Message(role="assistant", content=content_text),
        finish_reason=finish_reason,
        usage=Usage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
        ),
        warnings=warnings or [],
        meta=ResponseMeta(latency_ms=latency_ms, attempts=ctx.attempt),
        raw=raw_dict,
    )


class GeminiAdapter(ProviderAdapter):
    """Adapter for Google Gemini models using google-genai SDK."""

    name: ClassVar[str] = "gemini"
    capabilities: ClassVar[Capabilities] = Capabilities(
        system_role=True,
        streaming=True,
        json_mode=True,
        seed=False,
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
            "response_format",
        }),
    )

    def __init__(self, config: ProviderConfig) -> None:
        super().__init__(config)
        self._client: genai.Client | None = None

    async def startup(self) -> None:
        api_key = self.config.api_key.get_secret_value() if self.config.api_key else "missing"
        self._client = genai.Client(api_key=api_key)

    async def shutdown(self) -> None:
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
                "No Gemini API key found. Provide it in the request via 'api_key', 'provider_options.api_key', or 'Authorization: Bearer <key>' / 'X-Provider-Key' header.",
                provider="gemini",
                provider_status=401,
            )

        client = (
            self._client
            if (self._client and self.config.api_key and api_key == self.config.api_key.get_secret_value())
            else genai.Client(api_key=api_key)
        )

        contents, config, warnings = to_gemini_payload(req)
        start_time = time.monotonic()

        try:
            response = await client.aio.models.generate_content(
                model=req.model,
                contents=cast(types.ContentListUnion, contents),
                config=config,
            )
        except Exception as exc:
            raise self.translate_error(exc) from exc

        latency_ms = int((time.monotonic() - start_time) * 1000)
        return from_gemini_response(response, req, ctx, latency_ms=latency_ms, warnings=warnings)

    async def health_check(self) -> HealthStatus:
        if not self._client:
            await self.startup()
            assert self._client is not None
        try:
            # Query models
            await self._client.aio.models.get(model="gemini-1.5-flash")
            return HealthStatus.healthy("Gemini API reachable")
        except Exception as exc:
            return HealthStatus.unhealthy(f"Gemini health check failed: {type(exc).__name__}")

    def translate_error(self, exc: Exception) -> ProviderError:
        """Map google-genai SDK exceptions to ULIL error taxonomy."""
        code = getattr(exc, "code", None)
        msg = str(exc).lower()

        if code in (401, 403) or "api_key_invalid" in msg or "permission_denied" in msg:
            return ProviderAuthError("Gemini authentication failed. Invalid API key or credentials.", provider=self.name, provider_status=code or 401)

        if code == 404 or "not_found" in msg:
            return ModelNotFound("Requested model not found on Gemini.", provider=self.name, provider_status=404)

        if code == 429 or "resource_exhausted" in msg or "quota" in msg:
            return RateLimited("Gemini quota or rate limit exceeded.", provider=self.name, provider_status=429)

        if code == 400 or isinstance(exc, genai_errors.ClientError):
            if any(k in msg for k in ("context", "token count", "too large")):
                return ContextLengthExceeded("Prompt exceeds Gemini context length.", provider=self.name, provider_status=400)
            if any(k in msg for k in ("safety", "blocked", "policy")):
                return ContentRejected("Prompt rejected by Gemini safety policy.", provider=self.name, provider_status=400)
            return BadProviderResponse("Bad request sent to Gemini.", provider=self.name, provider_status=400, retryable=False)

        if code in (500, 503) or isinstance(exc, genai_errors.ServerError):
            return ProviderUnavailable("Gemini service unavailable.", provider=self.name, provider_status=code or 503)

        if "timeout" in msg:
            return ProviderTimeout("Gemini request timed out.", provider=self.name, provider_status=504)

        if "connection" in msg:
            return ProviderConnectionError("Could not connect to Gemini API.", provider=self.name, provider_status=502)

        return BadProviderResponse(f"Unexpected Gemini error: {type(exc).__name__}", provider=self.name)
