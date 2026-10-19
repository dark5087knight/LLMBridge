"""Providers inspection endpoints."""

from fastapi import APIRouter, Depends

from unified_llm.api.deps import get_registry
from unified_llm.core.registry import ProviderRegistry
from unified_llm.core.schemas import ProviderInfo

router = APIRouter(prefix="/v1", tags=["providers"])


@router.get("/providers", response_model=list[ProviderInfo])
async def list_providers(
    registry: ProviderRegistry = Depends(get_registry),
) -> list[ProviderInfo]:
    """List configured and available providers, capabilities, and allowed models."""
    return registry.describe()
