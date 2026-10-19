"""Context objects for request lifecycle, deadline tracking, and attempt tracking."""

import time
from dataclasses import dataclass, field


@dataclass(frozen=True)
class RequestContext:
    """Context bound to the inbound HTTP request."""

    request_id: str
    trace_id: str | None = None
    caller: str | None = None


@dataclass
class Deadline:
    """Tracks time budget across attempts and backoffs."""

    total_timeout_s: float
    start_time: float = field(default_factory=time.monotonic)

    def elapsed_s(self) -> float:
        """Seconds elapsed since deadline creation."""
        return time.monotonic() - self.start_time

    def remaining_s(self) -> float:
        """Seconds remaining before deadline is exhausted."""
        remaining = self.total_timeout_s - self.elapsed_s()
        return max(0.0, remaining)

    def is_expired(self) -> bool:
        """Check whether the deadline has passed."""
        return self.remaining_s() <= 0.0


@dataclass
class CallContext:
    """Context provided to an individual provider adapter attempt."""

    deadline: Deadline
    attempt: int
    request_id: str
    per_attempt_timeout_s: float = 60.0

    def remaining_s(self) -> float:
        """Compute the per-attempt timeout bounded by the total remaining deadline."""
        return max(0.001, min(self.per_attempt_timeout_s, self.deadline.remaining_s()))
