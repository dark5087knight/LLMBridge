"""Core LLMService orchestrating validation, concurrency limits, retries, and normalization."""

import asyncio
from structlog import get_logger

from unified_llm.config.settings import ServiceSettings
from unified_llm.core.capabilities import validate_against_capabilities
from unified_llm.core.context import CallContext, Deadline, RequestContext
from unified_llm.core.errors import (
    ModelNotAllowed,
    ProviderError,
    ProviderTimeout,
)
from unified_llm.core.registry import ProviderRegistry
from unified_llm.core.retry import RetryPolicy, run_with_retries
from unified_llm.core.schemas import ChatRequest, ChatResponse
from unified_llm.observability.metrics import metrics

logger = get_logger(__name__)


class LLMService:
    """Service orchestrating execution of a canonical LLM chat completion."""

    def __init__(
        self,
        registry: ProviderRegistry,
        policy: RetryPolicy,
        settings: ServiceSettings,
    ) -> None:
        self._registry = registry
        self._policy = policy
        self._settings = settings

    async def chat(self, req: ChatRequest, ctx: RequestContext) -> ChatResponse:
        """Execute a validated chat completion request."""
        # 1. Resolve provider adapter
        adapter = self._registry.get(req.provider)
        config = self._registry.get_config(req.provider)

        # 2. Model allow-list check
        if config and config.allowed_models is not None:
            if req.model not in config.allowed_models:
                raise ModelNotAllowed(
                    f"Model '{req.model}' is not in the allowed models for provider '{req.provider}'.",
                    provider=req.provider,
                )

        # 3. Capability validation
        warnings = validate_against_capabilities(
            req,
            adapter.capabilities,
            strict=req.options.strict_params,
        )

        # 4. Compute deadline
        configured_timeout = config.timeout_s if config else 60.0
        requested_timeout = req.options.timeout_s or configured_timeout
        effective_timeout = min(requested_timeout, self._settings.max_total_timeout_s)
        deadline = Deadline(total_timeout_s=effective_timeout)

        # 5. Concurrency semaphore
        semaphore = self._registry.get_semaphore(req.provider)
        request_policy = self._policy.for_request(req)

        # 6. Execute with bounded retries
        async def _attempt_call(attempt: int) -> ChatResponse:
            call_ctx = CallContext(
                deadline=deadline,
                attempt=attempt,
                request_id=ctx.request_id,
                per_attempt_timeout_s=configured_timeout,
            )

            # Acquire concurrency slot bounded by remaining deadline
            if semaphore is not None:
                try:
                    await asyncio.wait_for(semaphore.acquire(), timeout=deadline.remaining_s())
                except asyncio.TimeoutError:
                    raise ProviderTimeout(
                        f"Concurrency limit reached and deadline expired waiting for slot on provider '{req.provider}'.",
                        provider=req.provider,
                    )

            metrics.inc_inflight(req.provider)
            try:
                metrics.inc_attempts(req.provider, "started")
                response = await adapter.generate(req, call_ctx)
                metrics.inc_attempts(req.provider, "success")
                return response
            except Exception as exc:
                metrics.inc_attempts(req.provider, "failed")
                raise exc
            finally:
                metrics.dec_inflight(req.provider)
                if semaphore is not None:
                    semaphore.release()

        try:
            resp, attempts_made = await run_with_retries(
                _attempt_call,
                policy=request_policy,
                deadline=deadline,
            )
        except ProviderError as exc:
            metrics.inc_requests(req.provider, req.model, "error")
            metrics.inc_errors(req.provider, exc.type)
            raise exc
        except Exception as exc:
            metrics.inc_requests(req.provider, req.model, "error")
            metrics.inc_errors(req.provider, type(exc).__name__)
            raise exc

        # 7. Post-process response
        resp.warnings.extend(warnings)
        resp.meta.attempts = attempts_made

        # 8. Record metrics
        metrics.inc_requests(req.provider, req.model, "success")
        metrics.observe_duration(req.provider, req.model, deadline.elapsed_s())

        # 9. Structured completion log
        logger.info(
            "llm_call_completed",
            request_id=ctx.request_id,
            provider=req.provider,
            model=req.model,
            attempts=attempts_made,
            latency_ms=resp.meta.latency_ms,
            finish_reason=resp.finish_reason,
        )

        return resp
