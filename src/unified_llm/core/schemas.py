"""Canonical schemas for request, response, and provider information."""

from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field

from unified_llm.core.capabilities import Warning


class Message(BaseModel):
    """Canonical message."""

    role: Literal["system", "user", "assistant"]
    content: str | list[Any]


class GenerationParams(BaseModel):
    """Canonical generation parameters. Extra parameters are strictly forbidden."""

    model_config = ConfigDict(extra="forbid")

    temperature: float | None = Field(None, ge=0.0)
    top_p: float | None = Field(None, gt=0.0, le=1.0)
    top_k: int | None = Field(None, ge=1)
    max_tokens: int | None = Field(None, ge=1)
    stop: list[str] | None = Field(None, max_length=16)
    seed: int | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    response_format: Literal["text", "json"] | None = None


class RequestOptions(BaseModel):
    """Execution options requested by the caller."""

    timeout_s: float | None = Field(None, gt=0.0, le=600.0)
    max_retries: int | None = Field(None, ge=0, le=5)
    strict_params: bool = False
    include_raw: bool = False


class MessageRequest(BaseModel):
    """Canonical message execution request for dialogues, tool-use, and agent loops."""

    provider: str
    model: str
    messages: list[Message] = Field(min_length=1)
    params: GenerationParams = Field(default_factory=GenerationParams)
    options: RequestOptions = Field(default_factory=RequestOptions)
    provider_options: dict[str, Any] = Field(default_factory=dict)
    api_key: str | None = None
    metadata: dict[str, str] = Field(default_factory=dict)


class Usage(BaseModel):
    """Normalized token usage counters."""

    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None


class ResponseMeta(BaseModel):
    """Metadata regarding call execution (latency and retries)."""

    latency_ms: int
    attempts: int


class MessageResponse(BaseModel):
    """Canonical message execution response."""

    id: str
    provider: str
    model: str
    model_served: str | None = None
    message: Message
    finish_reason: Literal["stop", "length", "content_filter", "error", "other"]
    usage: Usage
    warnings: list[Warning] = Field(default_factory=list)
    meta: ResponseMeta
    raw: dict[str, Any] | None = None


class StreamEvent(BaseModel):
    """Event structure for Server-Sent Events streaming."""

    event: Literal["start", "delta", "usage", "end", "error"]
    data: dict[str, Any]


class ProviderInfo(BaseModel):
    """Provider metadata returned by /v1/providers."""

    name: str
    enabled: bool
    configured: bool
    capabilities: dict[str, Any]
    allowed_models: list[str] | None = None
