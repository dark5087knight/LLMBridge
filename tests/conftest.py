"""Pytest configuration and shared test fixtures."""

import pytest
from typing import AsyncIterator, ClassVar
from fastapi import FastAPI
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient

from unified_llm.config.settings import ProviderConfig, ServiceSettings
from unified_llm.core.capabilities import Capabilities
from unified_llm.core.context import CallContext
from unified_llm.core.errors import ProviderError
from unified_llm.core.registry import ProviderRegistry
from unified_llm.core.retry import RetryPolicy
from unified_llm.core.schemas import (
    ChatRequest,
    ChatResponse,
    Message,
    ResponseMeta,
    Usage,
)
from unified_llm.core.service import LLMService
from unified_llm.main import create_app
from unified_llm.providers.base import HealthStatus, ProviderAdapter


class FakeEchoAdapter(ProviderAdapter):
    """Test fake adapter that echoes input messages."""

    name: ClassVar[str] = "fake_echo"
    capabilities: ClassVar[Capabilities] = Capabilities(
        system_role=True,
        streaming=True,
        json_mode=True,
        seed=True,
        top_k=True,
        stop_sequences=True,
        penalties=True,
        multimodal_input=False,
        tool_calling=False,
        supported_params=frozenset({
            "temperature",
            "top_p",
            "top_k",
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
        self.call_count = 0
        self.should_fail_error: ProviderError | None = None

    async def generate(self, req: ChatRequest, ctx: CallContext) -> ChatResponse:
        self.call_count += 1
        if self.should_fail_error:
            raise self.should_fail_error

        last_content = req.messages[-1].content
        content_str = last_content if isinstance(last_content, str) else str(last_content)

        return ChatResponse(
            id=ctx.request_id,
            provider=req.provider,
            model=req.model,
            model_served=f"{req.model}-served",
            message=Message(role="assistant", content=f"Echo: {content_str}"),
            finish_reason="stop",
            usage=Usage(input_tokens=10, output_tokens=15, total_tokens=25),
            warnings=[],
            meta=ResponseMeta(latency_ms=42, attempts=ctx.attempt),
            raw={"echo": True} if req.options.include_raw else None,
        )

    async def health_check(self) -> HealthStatus:
        return HealthStatus.healthy("Fake echo adapter is healthy")

    def translate_error(self, exc: Exception) -> ProviderError:
        if isinstance(exc, ProviderError):
            return exc
        return ProviderError(str(exc), provider=self.name)


@pytest.fixture
def fake_adapter() -> FakeEchoAdapter:
    return FakeEchoAdapter(config=ProviderConfig(enabled=True))


@pytest.fixture
def test_settings() -> ServiceSettings:
    return ServiceSettings(
        log_level="DEBUG",
        log_format="console",
        max_request_bytes=100_000,
        max_total_timeout_s=30.0,
        max_attempts_cap=3,
        api_keys=[],
        providers={
            "fake_echo": ProviderConfig(
                enabled=True,
                allowed_models=["echo-v1", "echo-v2"],
                timeout_s=10.0,
            ),
            "unconfigured": ProviderConfig(enabled=False),
        },
    )


@pytest.fixture
def test_registry(test_settings: ServiceSettings, fake_adapter: FakeEchoAdapter) -> ProviderRegistry:
    registry = ProviderRegistry()
    registry.register(FakeEchoAdapter)
    registry.build(test_settings.providers)
    return registry


@pytest.fixture
def test_service(test_registry: ProviderRegistry, test_settings: ServiceSettings) -> LLMService:
    return LLMService(
        registry=test_registry,
        policy=RetryPolicy.from_settings(test_settings),
        settings=test_settings,
    )


@pytest.fixture
def app(test_settings: ServiceSettings, test_registry: ProviderRegistry, test_service: LLMService) -> FastAPI:
    application = create_app(test_settings)
    application.state.settings = test_settings
    application.state.registry = test_registry
    application.state.service = test_service
    return application


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    return TestClient(app)


@pytest.fixture
async def async_client(app: FastAPI) -> AsyncIterator[AsyncClient]:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
