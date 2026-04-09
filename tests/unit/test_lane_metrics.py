"""Unit tests for Phase 8: LaneMetrics per-lane performance counters."""

from __future__ import annotations

import threading
import time

import pytest

from maxim.models.language.lane_metrics import LaneMetrics, MetricsRegistry


class TestLaneMetrics:
    """Core LaneMetrics functionality."""

    def test_record_call_basic(self) -> None:
        m = LaneMetrics(lane_name="infer")
        m.record_call(100.0, success=True, kind="local", input_tokens=50, output_tokens=10)
        assert m.jobs_submitted == 1
        assert m.jobs_completed == 1
        assert m.jobs_failed == 0
        assert m.local_calls == 1
        assert m.total_input_tokens == 50
        assert m.total_output_tokens == 10

    def test_record_call_failure(self) -> None:
        m = LaneMetrics(lane_name="infer")
        m.record_call(200.0, success=False, kind="cloud")
        assert m.jobs_failed == 1
        assert m.jobs_completed == 0
        assert m.cloud_calls == 1

    def test_record_start_end(self) -> None:
        m = LaneMetrics(lane_name="review")
        t = m.record_start()
        assert m.current_in_flight == 1
        assert m.peak_in_flight == 1
        time.sleep(0.01)
        latency = m.record_end(t, success=True, kind="remote", input_tokens=100)
        assert latency >= 10  # at least 10ms from sleep
        assert m.current_in_flight == 0
        assert m.jobs_completed == 1
        assert m.remote_calls == 1

    def test_in_flight_tracking(self) -> None:
        m = LaneMetrics(lane_name="infer")
        t1 = m.record_start()
        t2 = m.record_start()
        assert m.current_in_flight == 2
        assert m.peak_in_flight == 2
        m.record_end(t1)
        assert m.current_in_flight == 1
        assert m.peak_in_flight == 2  # peak stays
        m.record_end(t2)
        assert m.current_in_flight == 0

    def test_kind_attribution(self) -> None:
        m = LaneMetrics(lane_name="infer")
        m.record_call(10, kind="local")
        m.record_call(20, kind="remote")
        m.record_call(30, kind="self-hosted")
        m.record_call(40, kind="peer")
        m.record_call(50, kind="cloud")
        assert m.local_calls == 1
        assert m.remote_calls == 2  # remote + self-hosted
        assert m.peer_calls == 1
        assert m.cloud_calls == 1

    def test_latency_percentiles(self) -> None:
        m = LaneMetrics(lane_name="infer")
        for i in range(100):
            m.record_call(float(i + 1))
        assert m.avg_latency_ms == pytest.approx(50.5, abs=0.1)
        assert m.p50_latency_ms == 51.0  # idx=50 in sorted [1..100]
        assert m.p99_latency_ms == 100.0  # idx=99 in sorted [1..100]

    def test_percentiles_empty(self) -> None:
        m = LaneMetrics(lane_name="infer")
        assert m.avg_latency_ms == 0.0
        assert m.p50_latency_ms == 0.0
        assert m.p99_latency_ms == 0.0

    def test_failure_rate(self) -> None:
        m = LaneMetrics(lane_name="infer")
        m.record_call(10, success=True)
        m.record_call(10, success=True)
        m.record_call(10, success=False)
        assert m.failure_rate == pytest.approx(1 / 3, abs=0.01)

    def test_failure_rate_empty(self) -> None:
        m = LaneMetrics(lane_name="infer")
        assert m.failure_rate == 0.0

    def test_remote_ratio(self) -> None:
        m = LaneMetrics(lane_name="infer")
        m.record_call(10, kind="local")
        m.record_call(10, kind="local")
        m.record_call(10, kind="cloud")
        assert m.remote_ratio == pytest.approx(1 / 3, abs=0.01)

    def test_cost_accumulation(self) -> None:
        m = LaneMetrics(lane_name="infer")
        m.record_call(10, cost_usd=0.05)
        m.record_call(10, cost_usd=0.10)
        assert m.total_cost_usd == pytest.approx(0.15)

    def test_total_tokens(self) -> None:
        m = LaneMetrics(lane_name="infer")
        m.record_call(10, input_tokens=100, output_tokens=50)
        m.record_call(10, input_tokens=200, output_tokens=75)
        assert m.total_tokens == 425

    def test_snapshot(self) -> None:
        m = LaneMetrics(lane_name="infer")
        m.record_call(100, success=True, kind="cloud", input_tokens=50, output_tokens=10)
        snap = m.snapshot()
        assert snap["lane"] == "infer"
        assert snap["jobs_completed"] == 1
        assert snap["cloud_calls"] == 1
        assert snap["total_input_tokens"] == 50
        assert snap["avg_latency_ms"] == 100.0

    def test_format_compact(self) -> None:
        m = LaneMetrics(lane_name="infer")
        assert "no calls yet" in m.format_compact()
        m.record_call(100, input_tokens=50, output_tokens=10)
        compact = m.format_compact()
        assert "infer" in compact
        assert "1 calls" in compact
        assert "p50=" in compact

    def test_failover_count(self) -> None:
        m = LaneMetrics(lane_name="infer")
        t = m.record_start()
        m.record_end(t, failover=True)
        assert m.failover_count == 1

    def test_reservoir_bounded(self) -> None:
        m = LaneMetrics(lane_name="infer")
        for i in range(500):
            m.record_call(float(i))
        # Reservoir maxlen is 200
        assert len(m._latencies) == 200

    def test_thread_safety(self) -> None:
        """Concurrent record_call from multiple threads shouldn't crash."""
        m = LaneMetrics(lane_name="infer")
        errors: list[Exception] = []

        def worker(n: int) -> None:
            try:
                for i in range(100):
                    m.record_call(float(i), kind="local", input_tokens=1)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert m.jobs_submitted == 800
        assert m.local_calls == 800
        assert m.total_input_tokens == 800


class TestMetricsRegistry:
    """MetricsRegistry thread-safe lane management."""

    def test_get_creates_on_first_access(self) -> None:
        r = MetricsRegistry()
        m = r.get("large")
        assert m.lane_name == "large"
        assert r.get("large") is m  # same instance

    def test_all_metrics(self) -> None:
        r = MetricsRegistry()
        r.get("large")
        r.get("small")
        all_m = r.all_metrics()
        assert "large" in all_m
        assert "small" in all_m

    def test_snapshot(self) -> None:
        r = MetricsRegistry()
        m = r.get("large")
        m.record_call(100, input_tokens=50)
        snap = r.snapshot()
        assert snap["large"]["jobs_completed"] == 1

    def test_thread_safe_creation(self) -> None:
        """Concurrent get() for the same lane returns the same instance."""
        r = MetricsRegistry()
        instances: list[LaneMetrics] = []

        def worker() -> None:
            instances.append(r.get("large"))

        threads = [threading.Thread(target=worker) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(instances) == 20
        assert all(inst is instances[0] for inst in instances)


# ─── MetricsRegistry tier names ────────────────────────────────────────


class TestMetricsRegistryTierNames:
    def test_tier_names_create_distinct_metrics(self):
        registry = MetricsRegistry()
        large = registry.get("large")
        medium = registry.get("medium")
        small = registry.get("small")
        assert large is not medium
        assert medium is not small
        assert large.lane_name == "large"
        assert medium.lane_name == "medium"
        assert small.lane_name == "small"
