"""Unit tests for pure response normalization functions across providers."""

from types import SimpleNamespace
from unified_llm.core.context import CallContext, Deadline
from unified_llm.core.schemas import ChatRequest, Message, RequestOptions
from unified_llm.providers._openai_style import (
    from_openai_finish_reason,
    from_openai_response,
)
from unified_llm.providers.gemini_adapter import (
    from_gemini_finish_reason,
    from_gemini_response,
)
from unified_llm.providers.ollama_adapter import (
    from_ollama_finish_reason,
    from_ollama_response,
)


def _make_context(request_id: str = "req_test_1") -> CallContext:
    return CallContext(
        deadline=Deadline(total_timeout_s=30.0),
        attempt=1,
        request_id=request_id,
    )


def test_openai_finish_reason_mapping():
    assert from_openai_finish_reason("stop") == "stop"
    assert from_openai_finish_reason("tool_calls") == "stop"
    assert from_openai_finish_reason("length") == "length"
    assert from_openai_finish_reason("content_filter") == "content_filter"
    assert from_openai_finish_reason("something_novel") == "other"
    assert from_openai_finish_reason(None) == "stop"


def test_openai_response_normalization():
    req = ChatRequest(
        provider="openai",
        model="gpt-4o",
        messages=[Message(role="user", content="Hi")],
        options=RequestOptions(include_raw=True),
    )
    ctx = _make_context()

    native = {
        "model": "gpt-4o-2024-08-06",
        "choices": [
            {
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": "Hello there!"},
            }
        ],
        "usage": {
            "prompt_tokens": 12,
            "completion_tokens": 8,
            "total_tokens": 20,
        },
    }

    norm = from_openai_response(native, req, ctx, latency_ms=150)

    assert norm.id == ctx.request_id
    assert norm.provider == "openai"
    assert norm.model == "gpt-4o"
    assert norm.model_served == "gpt-4o-2024-08-06"
    assert norm.message.content == "Hello there!"
    assert norm.finish_reason == "stop"
    assert norm.usage.input_tokens == 12
    assert norm.usage.output_tokens == 8
    assert norm.usage.total_tokens == 20
    assert norm.raw is not None
    assert norm.meta.latency_ms == 150


def test_gemini_finish_reason_mapping():
    assert from_gemini_finish_reason("STOP") == "stop"
    assert from_gemini_finish_reason("MAX_TOKENS") == "length"
    assert from_gemini_finish_reason("SAFETY") == "content_filter"
    assert from_gemini_finish_reason("RECITATION") == "content_filter"
    assert from_gemini_finish_reason("OTHER") == "error"
    assert from_gemini_finish_reason("UNKNOWN_REASON") == "other"


def test_gemini_response_normalization():
    req = ChatRequest(
        provider="gemini",
        model="gemini-1.5-flash",
        messages=[Message(role="user", content="Hi")],
    )
    ctx = _make_context()

    # Create dummy response structure mimicking google-genai
    native = SimpleNamespace(
        model_version="gemini-1.5-flash-002",
        candidates=[
            SimpleNamespace(
                finish_reason="STOP",
                content=SimpleNamespace(
                    parts=[SimpleNamespace(text="Gemini answer")],
                    role="model",
                ),
            )
        ],
        usage_metadata=SimpleNamespace(
            prompt_token_count=15,
            candidates_token_count=25,
            total_token_count=40,
        ),
    )

    norm = from_gemini_response(native, req, ctx, latency_ms=220)

    assert norm.provider == "gemini"
    assert norm.model_served == "gemini-1.5-flash-002"
    assert norm.message.content == "Gemini answer"
    assert norm.finish_reason == "stop"
    assert norm.usage.input_tokens == 15
    assert norm.usage.output_tokens == 25
    assert norm.usage.total_tokens == 40
    assert norm.raw is None  # include_raw is False


def test_ollama_response_normalization():
    req = ChatRequest(
        provider="ollama",
        model="llama3.2",
        messages=[Message(role="user", content="Hi")],
    )
    ctx = _make_context()

    native = {
        "model": "llama3.2:latest",
        "message": {"role": "assistant", "content": "Ollama answer"},
        "done_reason": "stop",
        "prompt_eval_count": 22,
        "eval_count": 33,
    }

    norm = from_ollama_response(native, req, ctx, latency_ms=300)

    assert norm.provider == "ollama"
    assert norm.model_served == "llama3.2:latest"
    assert norm.message.content == "Ollama answer"
    assert norm.finish_reason == "stop"
    assert norm.usage.input_tokens == 22
    assert norm.usage.output_tokens == 33
    assert norm.usage.total_tokens == 55
