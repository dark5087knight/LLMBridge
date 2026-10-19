"""Settings and configuration models for ULIL."""

from pathlib import Path
from typing import Any, Literal
from pydantic import BaseModel, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class ProviderConfig(BaseModel):
    """Configuration for an individual provider adapter."""

    enabled: bool = True
    api_key: SecretStr | None = None
    base_url: str | None = None
    timeout_s: float = 60.0
    connect_timeout_s: float = 5.0
    max_concurrency: int = 32
    allowed_models: list[str] | None = None  # None = any model allowed
    extra: dict[str, Any] = Field(default_factory=dict)


class ServiceSettings(BaseSettings):
    """Global service settings for the Unified LLM Integration Layer."""

    model_config = SettingsConfigDict(
        env_prefix="ULIL_",
        env_nested_delimiter="__",
        extra="ignore",
    )

    config_file: Path | None = None
    log_level: str = "INFO"
    log_format: Literal["json", "console"] = "json"
    api_keys: list[SecretStr] = Field(default_factory=list)
    max_request_bytes: int = 1_000_000
    max_total_timeout_s: float = 300.0
    max_attempts_cap: int = 3
    debug_log_payloads: bool = False
    providers: dict[str, ProviderConfig] = Field(default_factory=dict)
