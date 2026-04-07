"""Per-lane performance counters with reservoir-sampled latency (Phase 8).

Thread-safe metrics for each LLM lane (infer/review/record). Feeds:
  - ``maxim doctor`` — per-lane p50/p99/counts
  - LeaderProxy admission control (Phase 7b) — queue depth, failure rate
  - InferenceRouter health checks (Phase 7d) — latency + failure signals
  - ``MAXIM_LANE_TRACE=1`` — per-request trace enrichment

Design: no external deps, thread-safe via threading.Lock, O(1) record_call.
Latency percentiles use a fixed-size reservoir (last 200 samples) — good
enough for operational decisions without unbounded memory growth.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any


@dataclass
class LaneMetrics:
    """Per-lane performance counters with reservoir-sampled latency."""

    lane_name: str = ""

    # Monotonic counters
    jobs_submitted: int = 0
    jobs_completed: int = 0
    jobs_failed: int = 0
    failover_count: int = 0

    # Backend attribution
    local_calls: int = 0
    remote_calls: int = 0
    peer_calls: int = 0
    cloud_calls: int = 0

    # Queue pressure
    current_in_flight: int = 0
    peak_in_flight: int = 0

    # Token + cost accumulators
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0

    # Latency reservoir (last 200 samples)
    _latencies: deque[float] = field(
        default_factory=lambda: deque(maxlen=200),
        repr=False,
    )
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    # Timestamp tracking
    _first_call_time: float = 0.0
    _last_call_time: float = 0.0

    def record_start(self) -> float:
        """Record a job starting. Returns a start timestamp for record_end."""
        with self._lock:
            self.jobs_submitted += 1
            self.current_in_flight += 1
            if self.current_in_flight > self.peak_in_flight:
                self.peak_in_flight = self.current_in_flight
        return time.time()

    def record_end(
        self,
        start_time: float,
        *,
        success: bool = True,
        kind: str = "local",
        input_tokens: int = 0,
        output_tokens: int = 0,
        cost_usd: float = 0.0,
        failover: bool = False,
    ) -> float:
        """Record a completed job. Returns the latency in ms."""
        latency_ms = (time.time() - start_time) * 1000
        now = time.time()

        with self._lock:
            self.current_in_flight = max(0, self.current_in_flight - 1)
            self._latencies.append(latency_ms)
            self._last_call_time = now
            if self._first_call_time == 0.0:
                self._first_call_time = now

            if success:
                self.jobs_completed += 1
            else:
                self.jobs_failed += 1

            if failover:
                self.failover_count += 1

            # Attribution
            if kind == "local":
                self.local_calls += 1
            elif kind == "remote" or kind == "self-hosted":
                self.remote_calls += 1
            elif kind == "peer":
                self.peer_calls += 1
            elif kind == "cloud":
                self.cloud_calls += 1

            self.total_input_tokens += input_tokens
            self.total_output_tokens += output_tokens
            self.total_cost_usd += cost_usd

        return latency_ms

    def record_call(
        self,
        latency_ms: float,
        *,
        success: bool = True,
        kind: str = "local",
        input_tokens: int = 0,
        output_tokens: int = 0,
        cost_usd: float = 0.0,
    ) -> None:
        """Record a call with a known latency (e.g. from the LeaderProxy).

        Use record_start/record_end for calls where you control both ends.
        Use this for externally-timed calls (proxy logs, etc.).
        """
        now = time.time()
        with self._lock:
            self.jobs_submitted += 1
            self._latencies.append(latency_ms)
            self._last_call_time = now
            if self._first_call_time == 0.0:
                self._first_call_time = now

            if success:
                self.jobs_completed += 1
            else:
                self.jobs_failed += 1

            if kind == "local":
                self.local_calls += 1
            elif kind == "remote" or kind == "self-hosted":
                self.remote_calls += 1
            elif kind == "peer":
                self.peer_calls += 1
            elif kind == "cloud":
                self.cloud_calls += 1

            self.total_input_tokens += input_tokens
            self.total_output_tokens += output_tokens
            self.total_cost_usd += cost_usd

    # ─── derived properties ───────────────────────────────────────────

    @property
    def avg_latency_ms(self) -> float:
        with self._lock:
            if not self._latencies:
                return 0.0
            return sum(self._latencies) / len(self._latencies)

    @property
    def p50_latency_ms(self) -> float:
        return self._percentile(50)

    @property
    def p99_latency_ms(self) -> float:
        return self._percentile(99)

    @property
    def failure_rate(self) -> float:
        with self._lock:
            total = self.jobs_completed + self.jobs_failed
            if total == 0:
                return 0.0
            return self.jobs_failed / total

    @property
    def remote_ratio(self) -> float:
        with self._lock:
            total = self.local_calls + self.remote_calls + self.peer_calls + self.cloud_calls
            if total == 0:
                return 0.0
            return (self.remote_calls + self.peer_calls + self.cloud_calls) / total

    @property
    def total_tokens(self) -> int:
        with self._lock:
            return self.total_input_tokens + self.total_output_tokens

    @property
    def uptime_s(self) -> float:
        with self._lock:
            if self._first_call_time == 0.0:
                return 0.0
            return time.time() - self._first_call_time

    def _percentile(self, pct: int) -> float:
        with self._lock:
            if not self._latencies:
                return 0.0
            sorted_lat = sorted(self._latencies)
            idx = max(0, min(len(sorted_lat) - 1, int(len(sorted_lat) * pct / 100)))
            return sorted_lat[idx]

    # ─── snapshot / serialization ─────────────────────────────────────

    def snapshot(self) -> dict[str, Any]:
        """Thread-safe snapshot of all metrics for logging / doctor / API."""
        with self._lock:
            return {
                "lane": self.lane_name,
                "jobs_submitted": self.jobs_submitted,
                "jobs_completed": self.jobs_completed,
                "jobs_failed": self.jobs_failed,
                "failover_count": self.failover_count,
                "failure_rate": round(self.failure_rate, 3),
                "in_flight": self.current_in_flight,
                "peak_in_flight": self.peak_in_flight,
                "avg_latency_ms": round(self.avg_latency_ms, 1),
                "p50_latency_ms": round(self.p50_latency_ms, 1),
                "p99_latency_ms": round(self.p99_latency_ms, 1),
                "local_calls": self.local_calls,
                "remote_calls": self.remote_calls,
                "peer_calls": self.peer_calls,
                "cloud_calls": self.cloud_calls,
                "total_input_tokens": self.total_input_tokens,
                "total_output_tokens": self.total_output_tokens,
                "total_cost_usd": round(self.total_cost_usd, 4),
            }

    def format_compact(self) -> str:
        """One-line summary for doctor / banner output."""
        parts = [self.lane_name or "?"]
        with self._lock:
            total = self.jobs_completed + self.jobs_failed
            if total == 0:
                parts.append("no calls yet")
            else:
                parts.append(f"{total} calls")
                parts.append(f"p50={self.p50_latency_ms:.0f}ms")
                parts.append(f"p99={self.p99_latency_ms:.0f}ms")
                if self.jobs_failed > 0:
                    parts.append(f"fail={self.failure_rate:.0%}")
                parts.append(f"tokens={self.total_input_tokens}in+{self.total_output_tokens}out")
                if self.total_cost_usd > 0:
                    parts.append(f"${self.total_cost_usd:.4f}")
        return " | ".join(parts)


class MetricsRegistry:
    """Central registry of per-lane LaneMetrics instances.

    One instance per process, shared between LaneBackendManager and
    LeaderProxy. Thread-safe creation of per-lane metrics on first access.

    Supports legacy lane name aliases (infer→large, review→small, etc.)
    for backward compatibility during the tier migration.
    """

    # Legacy lane names → tier names. Callers using old names get
    # redirected to the tier's metrics transparently.
    _LANE_ALIASES: dict[str, str] = {
        "infer": "large",
        "infer_net": "large",
        "review": "small",
        "record": "small",
    }

    def __init__(self) -> None:
        self._metrics: dict[str, LaneMetrics] = {}
        self._lock = threading.Lock()

    def get(self, lane_name: str) -> LaneMetrics:
        """Get or create metrics for a lane. Thread-safe.

        Accepts tier names (large/medium/small) or legacy lane names
        (infer/review/record), which are resolved via alias.
        """
        resolved = self._LANE_ALIASES.get(lane_name, lane_name)
        with self._lock:
            if resolved not in self._metrics:
                self._metrics[resolved] = LaneMetrics(lane_name=resolved)
            return self._metrics[resolved]

    def all_metrics(self) -> dict[str, LaneMetrics]:
        """Snapshot of all lane metrics."""
        with self._lock:
            return dict(self._metrics)

    def snapshot(self) -> dict[str, dict[str, Any]]:
        """Full snapshot for doctor / API."""
        with self._lock:
            return {name: m.snapshot() for name, m in self._metrics.items()}


# Module-level singleton — shared between lane manager + proxy
_global_registry: MetricsRegistry | None = None
_registry_lock = threading.Lock()


def get_metrics_registry() -> MetricsRegistry:
    """Get the global MetricsRegistry singleton."""
    global _global_registry
    with _registry_lock:
        if _global_registry is None:
            _global_registry = MetricsRegistry()
        return _global_registry


__all__ = [
    "LaneMetrics",
    "MetricsRegistry",
    "get_metrics_registry",
]
