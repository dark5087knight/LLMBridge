"""ProviderRegistry for adapter registration, discovery, lifecycle, and concurrency control."""

import asyncio
import importlib.metadata
from typing import Mapping
from structlog import get_logger

from unified_llm.config.settings import ProviderConfig, ServiceSettings
from unified_llm.core.errors import ProviderNotConfigured, UnknownProvider
from unified_llm.core.schemas import ProviderInfo
from unified_llm.providers.base import ProviderAdapter
from unified_llm.providers import BUILTIN_ADAPTERS

logger = get_logger(__name__)


class ProviderRegistry:
    """Registry maintaining available and initialized provider adapters."""

    def __init__(self) -> None:
        self._adapter_classes: dict[str, type[ProviderAdapter]] = {}
        self._active_adapters: dict[str, ProviderAdapter] = {}
        self._concurrency_semaphores: dict[str, asyncio.Semaphore] = {}
        self._configs: dict[str, ProviderConfig] = {}

    def register(self, cls: type[ProviderAdapter]) -> None:
        """Register a provider adapter class."""
        self._adapter_classes[cls.name.lower()] = cls

    def discover_entry_points(self) -> None:
        """Discover third-party adapter classes declared via entry points."""
        try:
            selected = importlib.metadata.entry_points(group="unified_llm.providers")
            for ep in selected:
                try:
                    adapter_cls = ep.load()
                    if issubclass(adapter_cls, ProviderAdapter):
                        self.register(adapter_cls)
                        logger.info("discovered_plugin_provider", name=adapter_cls.name)
                except Exception as exc:
                    logger.warning("plugin_provider_load_failed", name=ep.name, error=str(exc))
        except Exception as exc:
            logger.warning("entry_point_discovery_failed", error=str(exc))

    @classmethod
    def from_settings(cls, settings: ServiceSettings) -> "ProviderRegistry":
        """Instantiate and configure registry from ServiceSettings."""
        registry = cls()

        # Register built-ins
        for name, adapter_cls in BUILTIN_ADAPTERS.items():
            registry.register(adapter_cls)

        # Discover plugins
        registry.discover_entry_points()

        # Build enabled and configured adapters
        registry.build(settings.providers)
        return registry

    def build(self, configs: Mapping[str, ProviderConfig]) -> None:
        """Instantiate enabled adapters with their configuration."""
        self._configs = dict(configs)

        for name, adapter_cls in self._adapter_classes.items():
            config = configs.get(name)
            if not config:
                # Default provider config - enabled by default so per-request keys work
                config = ProviderConfig(enabled=True)

            if config.enabled:
                try:
                    adapter = adapter_cls(config)
                    self._active_adapters[name] = adapter
                    self._concurrency_semaphores[name] = asyncio.Semaphore(config.max_concurrency)
                    logger.info("provider_adapter_initialized", provider=name)
                except Exception as exc:
                    logger.error("provider_adapter_init_failed", provider=name, error=str(exc))

    def get(self, name: str) -> ProviderAdapter:
        """Retrieve an active provider adapter."""
        name_lower = name.lower()
        if name_lower not in self._adapter_classes:
            raise UnknownProvider(f"Unknown provider '{name}'.", provider=name)

        adapter = self._active_adapters.get(name_lower)
        if not adapter:
            raise ProviderNotConfigured(
                f"Provider '{name}' is not configured or disabled.", provider=name
            )

        return adapter

    def get_semaphore(self, name: str) -> asyncio.Semaphore | None:
        """Retrieve the concurrency limit semaphore for a provider."""
        return self._concurrency_semaphores.get(name.lower())

    def get_config(self, name: str) -> ProviderConfig | None:
        """Retrieve configuration for a provider."""
        return self._configs.get(name.lower())

    def describe(self) -> list[ProviderInfo]:
        """Describe all registered and active providers."""
        infos: list[ProviderInfo] = []
        for name, adapter_cls in self._adapter_classes.items():
            config = self._configs.get(name)
            enabled = config.enabled if config else False
            configured = name in self._active_adapters
            caps = adapter_cls.capabilities
            caps_dict = {
                "system_role": caps.system_role,
                "streaming": caps.streaming,
                "json_mode": caps.json_mode,
                "seed": caps.seed,
                "top_k": caps.top_k,
                "stop_sequences": caps.stop_sequences,
                "penalties": caps.penalties,
                "multimodal_input": caps.multimodal_input,
                "tool_calling": caps.tool_calling,
                "supported_params": sorted(list(caps.supported_params)),
            }
            allowed_models = config.allowed_models if config else None

            infos.append(
                ProviderInfo(
                    name=name,
                    enabled=enabled,
                    configured=configured,
                    capabilities=caps_dict,
                    allowed_models=allowed_models,
                )
            )
        return infos

    async def startup_all(self) -> None:
        """Call startup() on all active adapters."""
        for name, adapter in self._active_adapters.items():
            try:
                await adapter.startup()
            except Exception as exc:
                logger.error("provider_startup_failed", provider=name, error=str(exc))

    async def shutdown_all(self) -> None:
        """Call shutdown() on all active adapters."""
        for name, adapter in self._active_adapters.items():
            try:
                await adapter.shutdown()
            except Exception as exc:
                logger.error("provider_shutdown_failed", provider=name, error=str(exc))
