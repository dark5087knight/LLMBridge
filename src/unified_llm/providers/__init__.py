"""Provider adapters for ULIL."""

from unified_llm.providers.base import HealthStatus, ProviderAdapter
from unified_llm.providers.gemini_adapter import GeminiAdapter
from unified_llm.providers.groq_adapter import GroqAdapter
from unified_llm.providers.ollama_adapter import OllamaAdapter
from unified_llm.providers.openai_adapter import OpenAIAdapter

BUILTIN_ADAPTERS: dict[str, type[ProviderAdapter]] = {
    "openai": OpenAIAdapter,
    "groq": GroqAdapter,
    "gemini": GeminiAdapter,
    "ollama": OllamaAdapter,
}

__all__ = [
    "BUILTIN_ADAPTERS",
    "GeminiAdapter",
    "GroqAdapter",
    "HealthStatus",
    "OllamaAdapter",
    "OpenAIAdapter",
    "ProviderAdapter",
]
