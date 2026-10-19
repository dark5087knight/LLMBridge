"""Shared pure mapping and normalization functions for OpenAI-compatible providers."""

import json
from typing import Any, Literal
from unified_llm.core.capabilities import Warning
from unified_llm.core.context import CallContext
from unified_llm.core.schemas import (
    FunctionCall,
    Message,
    MessageRequest,
    MessageResponse,
    ResponseMeta,
    ToolCall,
    Usage,
)


def to_openai_payload(req: MessageRequest) -> dict[str, Any]:
    """Pure function mapping a canonical MessageRequest to an OpenAI-compatible JSON payload."""
    openai_messages: list[dict[str, Any]] = []
    for m in req.messages:
        msg_dict: dict[str, Any] = {"role": m.role}
        if m.content is not None:
            msg_dict["content"] = m.content
        if m.name is not None:
            msg_dict["name"] = m.name
        if m.tool_call_id is not None:
            msg_dict["tool_call_id"] = m.tool_call_id
        if m.tool_calls:
            msg_dict["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": (
                            tc.function.arguments
                            if isinstance(tc.function.arguments, str)
                            else json.dumps(tc.function.arguments)
                        ),
                    },
                }
                for tc in m.tool_calls
            ]
        openai_messages.append(msg_dict)

    payload: dict[str, Any] = {
        "model": req.model,
        "messages": openai_messages,
    }

    if req.tools:
        payload["tools"] = [
            {
                "type": t.type,
                "function": {
                    "name": t.function.name,
                    "description": t.function.description,
                    "parameters": t.function.parameters,
                },
            }
            for t in req.tools
        ]
        if req.tool_choice is not None:
            payload["tool_choice"] = req.tool_choice

    p = req.params
    if p.temperature is not None:
        payload["temperature"] = p.temperature
    if p.top_p is not None:
        payload["top_p"] = p.top_p
    if p.max_tokens is not None:
        payload["max_tokens"] = p.max_tokens
    if p.stop is not None:
        payload["stop"] = p.stop
    if p.seed is not None:
        payload["seed"] = p.seed
    if p.presence_penalty is not None:
        payload["presence_penalty"] = p.presence_penalty
    if p.frequency_penalty is not None:
        payload["frequency_penalty"] = p.frequency_penalty
    if p.response_format == "json":
        payload["response_format"] = {"type": "json_object"}
    elif p.response_format == "text":
        payload["response_format"] = {"type": "text"}

    # Merge provider_options escape hatch (excluding internal routing/auth parameters)
    if req.provider_options:
        extra = {k: v for k, v in req.provider_options.items() if k != "api_key"}
        payload.update(extra)

    return payload


def from_openai_finish_reason(
    reason: str | None,
) -> Literal["stop", "length", "tool_calls", "content_filter", "error", "other"]:
    """Map OpenAI finish reason to canonical taxonomy."""
    if not reason:
        return "stop"
    reason_lower = str(reason).lower()
    if reason_lower in ("tool_calls", "function_call"):
        return "tool_calls"
    if reason_lower in ("stop",):
        return "stop"
    if reason_lower in ("length", "max_tokens"):
        return "length"
    if reason_lower == "content_filter":
        return "content_filter"
    return "other"


def from_openai_response(
    native: Any,
    req: MessageRequest,
    ctx: CallContext,
    latency_ms: int = 0,
    warnings: list[Warning] | None = None,
) -> MessageResponse:
    """Pure function normalizing an OpenAI SDK response object or dictionary to MessageResponse."""
    parsed_tool_calls: list[ToolCall] | None = None

    # Handle dict or object attributes
    if isinstance(native, dict):
        choices = native.get("choices", [])
        choice = choices[0] if choices else {}
        msg_dict = choice.get("message", {})
        content = msg_dict.get("content") or ""
        role = msg_dict.get("role", "assistant")
        finish_reason_raw = choice.get("finish_reason")

        raw_tc = msg_dict.get("tool_calls")
        if raw_tc:
            parsed_tool_calls = [
                ToolCall(
                    id=tc.get("id") or "call_unknown",
                    type="function",
                    function=FunctionCall(
                        name=tc.get("function", {}).get("name", ""),
                        arguments=tc.get("function", {}).get("arguments", ""),
                    ),
                )
                for tc in raw_tc
            ]

        model_served = native.get("model")
        usage_data = native.get("usage") or {}
        input_tokens = usage_data.get("prompt_tokens")
        output_tokens = usage_data.get("completion_tokens")
        total_tokens = usage_data.get("total_tokens")
        raw_dict = native if req.options.include_raw else None
    else:
        choice = native.choices[0] if getattr(native, "choices", None) else None
        if choice and hasattr(choice, "message"):
            content = choice.message.content or ""
            role = choice.message.role or "assistant"
            finish_reason_raw = getattr(choice, "finish_reason", None)
            raw_tc = getattr(choice.message, "tool_calls", None)
            if raw_tc:
                parsed_tool_calls = [
                    ToolCall(
                        id=getattr(tc, "id", None) or "call_unknown",
                        type="function",
                        function=FunctionCall(
                            name=getattr(tc.function, "name", ""),
                            arguments=getattr(tc.function, "arguments", ""),
                        ),
                    )
                    for tc in raw_tc
                ]
        else:
            content = ""
            role = "assistant"
            finish_reason_raw = None

        model_served = getattr(native, "model", None)
        usage = getattr(native, "usage", None)
        input_tokens = getattr(usage, "prompt_tokens", None) if usage else None
        output_tokens = getattr(usage, "completion_tokens", None) if usage else None
        total_tokens = getattr(usage, "total_tokens", None) if usage else None

        if req.options.include_raw:
            raw_dict = native.model_dump() if hasattr(native, "model_dump") else dict(native)
        else:
            raw_dict = None

    finish_reason = from_openai_finish_reason(finish_reason_raw)
    if parsed_tool_calls and finish_reason == "stop":
        finish_reason = "tool_calls"

    return MessageResponse(
        id=ctx.request_id,
        provider=req.provider,
        model=req.model,
        model_served=model_served,
        message=Message(role=role, content=content, tool_calls=parsed_tool_calls),
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
