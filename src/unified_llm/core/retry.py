"""Retry and deadline policy implementation using Tenacity and async retry loops."""

import asyncio
import random
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, TypeVar

from structlog import get_logger
from tenacity import (
    AsyncRetrying,
    RetryCallState,
    retry_if_exception,
    stop_after_attempt,
)

from unified_llm.core.context import Deadline
from unified_llm.core.errors import ProviderError, ProviderTimeout
from unified_llm.core.schemas import ChatRequest

logger = get_logger(__name__)

T = TypeVar("T")


@dataclass
class RetryPolicy:
    """Configurable retry policy."""

    max_attempts: int = 3
    initial_backoff_s: float = 0.5
    max_backoff_s: float = 8.0
    max_retry_after_s: float = 10.0

    @classmethod
    def from_settings(cls, settings: Any) -> "RetryPolicy":
        """Build a RetryPolicy from global ServiceSettings."""
        return cls(max_attempts=settings.max_attempts_cap)

    def for_request(self, req: ChatRequest) -> "RetryPolicy":
        """Produce request-specific policy.

        options.max_retries can lower the attempt count but never raise it above server cap.
        """
        if req.options.max_retries is not None:
            # max_attempts = 1 initial try + requested retries
            desired_attempts = req.options.max_retries + 1
            effective_attempts = min(desired_attempts, self.max_attempts)
        else:
            effective_attempts = self.max_attempts

        return RetryPolicy(
            max_attempts=max(1, effective_attempts),
            initial_backoff_s=self.initial_backoff_s,
            max_backoff_s=self.max_backoff_s,
            max_retry_after_s=self.max_retry_after_s,
        )


def log_retry(retry_state: RetryCallState) -> None:
    """Log retry attempt information."""
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    logger.warning(
        "llm_call_retry",
        attempt=retry_state.attempt_number,
        error_type=type(exc).__name__ if exc else None,
        error=str(exc) if exc else None,
    )


async def run_with_retries(
    fn: Callable[[int], Awaitable[T]],
    policy: RetryPolicy,
    deadline: Deadline,
) -> tuple[T, int]:
    """Execute fn(attempt) with bounded retries and total deadline enforcement.

    Returns:
        tuple of (result, total_attempts_made)

    Raises:
        ProviderTimeout: If the overall deadline expires.
        ProviderError: When an error is non-retryable or retry attempts are exhausted.
    """
    attempt = 1
    last_exception: Exception | None = None

    while True:
        # Check if deadline is already expired before starting attempt
        if deadline.is_expired():
            raise ProviderTimeout(
                f"Total deadline ({deadline.total_timeout_s}s) exhausted before attempt {attempt}."
            )

        try:
            result = await fn(attempt)
            return result, attempt
        except ProviderError as exc:
            last_exception = exc

            # Non-retryable errors fail immediately
            if not exc.retryable:
                raise exc

            # If attempts exhausted, fail fast
            if attempt >= policy.max_attempts:
                raise exc

            # Calculate backoff delay
            if exc.retry_after_s is not None:
                if exc.retry_after_s > policy.max_retry_after_s:
                    # Reject sleeping longer than cap; fail fast
                    raise exc
                delay = exc.retry_after_s
            else:
                # Exponential backoff with full jitter: 0.5s -> 8.0s
                exp = policy.initial_backoff_s * (2 ** (attempt - 1))
                capped = min(policy.max_backoff_s, exp)
                delay = random.uniform(capped * 0.5, capped)

            # Check if sleeping this delay exceeds remaining deadline
            if delay >= deadline.remaining_s():
                raise ProviderTimeout(
                    f"Deadline remaining ({deadline.remaining_s():.2f}s) insufficient for backoff ({delay:.2f}s).",
                    provider=exc.provider,
                )

            logger.info(
                "llm_retry_scheduled",
                attempt=attempt,
                next_attempt=attempt + 1,
                delay_s=round(delay, 3),
                error_type=exc.type,
                provider=exc.provider,
            )

            await asyncio.sleep(delay)
            attempt += 1

        except Exception as exc:
            # Any unhandled exception is not retryable
            raise exc
