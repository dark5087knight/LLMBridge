"""Health, readiness, and metrics endpoints."""

from typing import Any
from fastapi import APIRouter, Depends, Query, Response
from fastapi.responses import JSONResponse

from unified_llm.api.deps import get_registry
from unified_llm.core.registry import ProviderRegistry
from unified_llm.observability.metrics import metrics

router = APIRouter(tags=["health"])


@router.get("/healthz")
async def healthz() -> dict[str, str]:
    """Liveness probe: verifies process is alive and receiving traffic."""
    return {"status": "ok"}


@router.get("/readyz")
async def readyz(
    deep: bool = Query(False, description="Perform deep health checks against provider APIs"),
    registry: ProviderRegistry = Depends(get_registry),
) -> JSONResponse:
    """Readiness probe: verifies at least one provider is configured and available."""
    active_providers = list(registry._active_adapters.keys())
    if not active_providers:
        return JSONResponse(
            status_code=503,
            content={
                "ready": False,
                "reason": "No provider adapters are currently configured and active.",
            },
        )

    if deep:
        deep_results: dict[str, dict[str, Any]] = {}
        all_healthy = True

        for name, adapter in registry._active_adapters.items():
            status = await adapter.health_check()
            deep_results[name] = {"ok": status.ok, "message": status.message}
            if not status.ok:
                all_healthy = False

        if not all_healthy:
            return JSONResponse(
                status_code=503,
                content={
                    "ready": False,
                    "reason": "One or more provider health checks failed.",
                    "details": deep_results,
                },
            )
        return JSONResponse(
            status_code=200,
            content={
                "ready": True,
                "active_providers": active_providers,
                "details": deep_results,
            },
        )

    return JSONResponse(
        status_code=200,
        content={
            "ready": True,
            "active_providers": active_providers,
        },
    )


@router.get("/metrics")
async def get_metrics() -> Response:
    """Prometheus metrics endpoint."""
    return Response(
        content=metrics.render(),
        media_type="text/plain; version=0.0.4",
    )
