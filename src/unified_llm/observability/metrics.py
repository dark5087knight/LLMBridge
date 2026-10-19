"""In-memory Prometheus-compatible metrics registry."""

from collections import defaultdict
from threading import Lock


class MetricsRegistry:
    """Thread-safe Prometheus metrics registry."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._requests_total: dict[tuple[str, str, str], int] = defaultdict(int)
        self._attempts_total: dict[tuple[str, str], int] = defaultdict(int)
        self._errors_total: dict[tuple[str, str], int] = defaultdict(int)
        self._inflight: dict[str, int] = defaultdict(int)
        self._durations: dict[tuple[str, str], list[float]] = defaultdict(list)

    def inc_requests(self, provider: str, model: str, status: str) -> None:
        with self._lock:
            self._requests_total[(provider, model, status)] += 1

    def inc_attempts(self, provider: str, outcome: str) -> None:
        with self._lock:
            self._attempts_total[(provider, outcome)] += 1

    def inc_errors(self, provider: str, error_type: str) -> None:
        with self._lock:
            self._errors_total[(provider, error_type)] += 1

    def inc_inflight(self, provider: str) -> None:
        with self._lock:
            self._inflight[provider] += 1

    def dec_inflight(self, provider: str) -> None:
        with self._lock:
            self._inflight[provider] = max(0, self._inflight[provider] - 1)

    def observe_duration(self, provider: str, model: str, duration_s: float) -> None:
        with self._lock:
            self._durations[(provider, model)].append(duration_s)
            if len(self._durations[(provider, model)]) > 1000:
                self._durations[(provider, model)] = self._durations[(provider, model)][-500:]

    def render(self) -> str:
        """Format metrics in Prometheus text exposition format."""
        lines: list[str] = [
            "# HELP ulil_requests_total Total number of LLM completion requests.",
            "# TYPE ulil_requests_total counter",
        ]
        with self._lock:
            for (p, m, s), val in self._requests_total.items():
                lines.append(f'ulil_requests_total{{provider="{p}",model="{m}",status="{s}"}} {val}')

            lines.extend([
                "# HELP ulil_attempts_total Total attempts made to providers.",
                "# TYPE ulil_attempts_total counter",
            ])
            for (p, o), val in self._attempts_total.items():
                lines.append(f'ulil_attempts_total{{provider="{p}",outcome="{o}"}} {val}')

            lines.extend([
                "# HELP ulil_errors_total Total errors encountered by provider and error type.",
                "# TYPE ulil_errors_total counter",
            ])
            for (p, e), val in self._errors_total.items():
                lines.append(f'ulil_errors_total{{provider="{p}",error_type="{e}"}} {val}')

            lines.extend([
                "# HELP ulil_inflight_requests Current number of requests being processed.",
                "# TYPE ulil_inflight_requests gauge",
            ])
            for p, val in self._inflight.items():
                lines.append(f'ulil_inflight_requests{{provider="{p}"}} {val}')

        return "\n".join(lines) + "\n"


metrics = MetricsRegistry()
