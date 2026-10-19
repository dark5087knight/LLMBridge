"""Unit tests for Deadline math, RetryPolicy, and run_with_retries."""

import time
import pytest

from unified_llm.core.context import Deadline
from unified_llm.core.errors import (
    InvalidRequest,
    ProviderAuthError,
    ProviderTimeout,
    RateLimited,
)
from unified_llm.core.retry import RetryPolicy, run_with_retries
from unified_llm.core.schemas import ChatRequest, Message, RequestOptions


def test_deadline_math():
    deadline = Deadline(total_timeout_s=1.0)
    assert not deadline.is_expired()
    assert deadline.remaining_s() <= 1.0
    assert deadline.elapsed_s() >= 0.0

    time.sleep(0.05)
    assert deadline.elapsed_s() >= 0.04
    assert deadline.remaining_s() < 1.0


def test_retry_policy_for_request():
    policy = RetryPolicy(max_attempts=3)

    # options.max_retries = 1 -> effective max_attempts = 2
    req1 = ChatRequest(
        provider="p",
        model="m",
        messages=[Message(role="user", content="hi")],
        options=RequestOptions(max_retries=1),
    )
    req_policy1 = policy.for_request(req1)
    assert req_policy1.max_attempts == 2

    # options.max_retries = 5 -> capped at server max 3
    req2 = ChatRequest(
        provider="p",
        model="m",
        messages=[Message(role="user", content="hi")],
        options=RequestOptions(max_retries=5),
    )
    req_policy2 = policy.for_request(req2)
    assert req_policy2.max_attempts == 3

    # options.max_retries = 0 -> effective max_attempts = 1
    req3 = ChatRequest(
        provider="p",
        model="m",
        messages=[Message(role="user", content="hi")],
        options=RequestOptions(max_retries=0),
    )
    req_policy3 = policy.for_request(req3)
    assert req_policy3.max_attempts == 1


@pytest.mark.asyncio
async def test_run_with_retries_non_retryable_fails_fast():
    attempts = 0

    async def fail_auth(attempt: int):
        nonlocal attempts
        attempts = attempt
        raise ProviderAuthError("Invalid credentials", provider="test")

    deadline = Deadline(total_timeout_s=5.0)
    policy = RetryPolicy(max_attempts=3)

    with pytest.raises(ProviderAuthError):
        await run_with_retries(fail_auth, policy=policy, deadline=deadline)

    # Must fail on attempt 1 without retrying
    assert attempts == 1


@pytest.mark.asyncio
async def test_run_with_retries_success_after_retry():
    call_count = 0

    async def transient_fail(attempt: int):
        nonlocal call_count
        call_count += 1
        if attempt == 1:
            raise RateLimited("Too many requests", provider="test", retry_after_s=0.01)
        return "success"

    deadline = Deadline(total_timeout_s=5.0)
    policy = RetryPolicy(max_attempts=3, initial_backoff_s=0.01, max_backoff_s=0.02)

    res, total_attempts = await run_with_retries(transient_fail, policy=policy, deadline=deadline)
    assert res == "success"
    assert total_attempts == 2
    assert call_count == 2


@pytest.mark.asyncio
async def test_run_with_retries_exhaustion():
    call_count = 0

    async def persistent_fail(attempt: int):
        nonlocal call_count
        call_count += 1
        raise RateLimited("Overloaded", provider="test", retry_after_s=0.01)

    deadline = Deadline(total_timeout_s=5.0)
    policy = RetryPolicy(max_attempts=2, initial_backoff_s=0.01)

    with pytest.raises(RateLimited):
        await run_with_retries(persistent_fail, policy=policy, deadline=deadline)

    assert call_count == 2


@pytest.mark.asyncio
async def test_run_with_retries_retry_after_exceeds_cap():
    async def high_retry_after(attempt: int):
        # 30s > max_retry_after_s (10s)
        raise RateLimited("Rate limited", provider="test", retry_after_s=30.0)

    deadline = Deadline(total_timeout_s=60.0)
    policy = RetryPolicy(max_attempts=3, max_retry_after_s=10.0)

    # Should fail fast without waiting 30s
    with pytest.raises(RateLimited):
        await run_with_retries(high_retry_after, policy=policy, deadline=deadline)


@pytest.mark.asyncio
async def test_run_with_retries_deadline_exhaustion():
    # Deadline already expired
    deadline = Deadline(total_timeout_s=0.001)
    time.sleep(0.01)

    async def will_not_run(attempt: int):
        return "ok"

    policy = RetryPolicy(max_attempts=3)
    with pytest.raises(ProviderTimeout):
        await run_with_retries(will_not_run, policy=policy, deadline=deadline)
