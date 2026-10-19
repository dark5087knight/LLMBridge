"""Capabilities model and capability validation logic."""

from dataclasses import dataclass
from typing import TYPE_CHECKING
from pydantic import BaseModel

if TYPE_CHECKING:
    from unified_llm.core.schemas import ChatRequest


class Warning(BaseModel):
    """Warning item reported when an unsupported parameter is ignored in lenient mode."""

    code: str
    param: str | None = None
    message: str


@dataclass(frozen=True)
class Capabilities:
    """Declared capabilities of a provider adapter."""

    system_role: bool = True
    streaming: bool = True
    json_mode: bool = True
    seed: bool = True
    top_k: bool = False
    stop_sequences: bool = True
    penalties: bool = True
    multimodal_input: bool = False
    tool_calling: bool = False
    supported_params: frozenset[str] = frozenset({
        "temperature",
        "top_p",
        "max_tokens",
        "stop",
    })


def validate_against_capabilities(
    req: "ChatRequest",
    capabilities: Capabilities,
    strict: bool = False,
) -> list[Warning]:
    """Validate request generation parameters against provider capabilities.

    In lenient mode (strict=False), unsupported parameters are omitted and warnings returned.
    In strict mode (strict=True), raises UnsupportedParameter on the first unsupported parameter.
    """
    from unified_llm.core.errors import UnsupportedParameter

    warnings: list[Warning] = []

    # Map of parameter name to value
    params_dict = req.params.model_dump(exclude_unset=True, exclude_none=True)

    for param_name, param_val in params_dict.items():
        if param_val is None:
            continue

        # Check against supported_params
        is_supported = param_name in capabilities.supported_params

        # Specific feature capability checks
        if param_name == "response_format" and param_val == "json" and not capabilities.json_mode:
            is_supported = False
        elif param_name == "seed" and not capabilities.seed:
            is_supported = False
        elif param_name == "top_k" and not capabilities.top_k:
            is_supported = False
        elif param_name == "stop" and not capabilities.stop_sequences:
            is_supported = False
        elif param_name in ("presence_penalty", "frequency_penalty") and not capabilities.penalties:
            is_supported = False

        if not is_supported:
            if strict:
                raise UnsupportedParameter(
                    f"Parameter '{param_name}' is not supported by provider '{req.provider}'.",
                    provider=req.provider,
                )
            warnings.append(
                Warning(
                    code="param_ignored",
                    param=param_name,
                    message=f"Parameter '{param_name}' is not supported by provider '{req.provider}' and was ignored.",
                )
            )

    return warnings
