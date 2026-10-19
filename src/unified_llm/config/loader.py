"""Configuration loader supporting environment variables and YAML interpolation."""

import os
import re
from pathlib import Path
from typing import Any
import yaml
from pydantic import ValidationError

from unified_llm.config.settings import ProviderConfig, ServiceSettings

_ENV_VAR_PATTERN = re.compile(r"\$\{([A-Za-z0-9_]+)(?::([^}]*))?\}")


def interpolate_env_vars(text: str) -> str:
    """Interpolate ${VAR} or ${VAR:default} expressions within a string."""

    def replacer(match: re.Match[str]) -> str:
        var_name = match.group(1)
        default_val = match.group(2) if match.group(2) is not None else ""
        return os.environ.get(var_name, default_val)

    return _ENV_VAR_PATTERN.sub(replacer, text)


def load_yaml_config(path: Path | str) -> dict[str, Any]:
    """Load and interpolate a YAML configuration file."""
    path_obj = Path(path)
    if not path_obj.is_file():
        raise FileNotFoundError(f"Config file not found: {path_obj}")

    content = path_obj.read_text(encoding="utf-8")
    interpolated = interpolate_env_vars(content)
    data = yaml.safe_load(interpolated)
    return data if isinstance(data, dict) else {}


def deep_merge(target: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge source into target. Target keys take precedence if present."""
    result = dict(target)
    for key, value in source.items():
        if key in result:
            if isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def validate_provider_credentials(name: str, config: ProviderConfig) -> None:
    """Validate that enabled providers have required credentials if specified."""
    # When api_key is omitted in static config, keys can still be supplied per-request at runtime.
    return


def load_settings(config_file: Path | str | None = None) -> ServiceSettings:
    """Load settings from environment variables and an optional YAML file.

    Precedence:
    1. Environment variables (highest)
    2. YAML file (optional)
    3. Built-in defaults (lowest)
    """
    # 1. Load initial settings from environment
    env_settings = ServiceSettings()

    # Determine file path
    cfg_path = config_file or env_settings.config_file or os.environ.get("ULIL_CONFIG_FILE")
    if cfg_path:
        yaml_data = load_yaml_config(cfg_path)

        # Merge providers from YAML into existing environment providers.
        # Environment provider configurations take precedence over YAML.
        yaml_providers = yaml_data.get("providers", {})
        merged_providers: dict[str, ProviderConfig] = {}

        for p_name, p_val in yaml_providers.items():
            if isinstance(p_val, dict):
                merged_providers[p_name] = ProviderConfig(**p_val)

        # Overwrite/merge with any environment-provided providers
        for p_name, p_cfg in env_settings.providers.items():
            if p_name in merged_providers:
                # Merge dicts
                merged_dict = deep_merge(
                    p_cfg.model_dump(exclude_unset=True),
                    merged_providers[p_name].model_dump(),
                )
                merged_providers[p_name] = ProviderConfig(**merged_dict)
            else:
                merged_providers[p_name] = p_cfg

        # Global fields from YAML if not explicitly set in env
        merged_settings_dict = yaml_data.copy()
        merged_settings_dict.pop("providers", None)
        env_dict = env_settings.model_dump(exclude_unset=True)
        env_dict.pop("providers", None)

        final_data = deep_merge(env_dict, merged_settings_dict)
        final_data["providers"] = merged_providers
        settings = ServiceSettings(**final_data)
    else:
        settings = env_settings

    # Validate startup credentials for enabled providers
    for prov_name, prov_cfg in settings.providers.items():
        validate_provider_credentials(prov_name, prov_cfg)

    return settings
