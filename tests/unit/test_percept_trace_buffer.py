"""Tests for PerceptTraceBuffer — F0.2 foundation."""

from __future__ import annotations

import math
import threading

import pytest

from maxim.memory.percept_trace_buffer import PerceptTraceBuffer


class TestRecordAndSnapshot:
    def test_record_and_snapshot(self) -> None:
        buf = PerceptTraceBuffer()
        buf.record("a", "p1")
        buf.record("a", "p2")
        buf.record("a", "p3")
        snap = buf.snapshot()
        assert len(snap) == 3
        assert all(e.activation_strength == 1.0 for e in snap)

    def test_record_default_activation(self) -> None:
        buf = PerceptTraceBuffer()
        buf.record("a", "p1")
        assert buf.snapshot()[0].activation_strength == 1.0


class TestDecay:
    @pytest.mark.parametrize("n_ticks", [0, 5, 10, 20, 30])
    def test_decay_shape(self, n_ticks: int) -> None:
        tau = 10.0
        buf = PerceptTraceBuffer(tau=tau)
        buf.record("a", "p1")
        for _ in range(n_ticks):
            buf.tick()
        expected = math.exp(-n_ticks / tau)
        snap = buf.snapshot(min_activation=0.0)
        if expected < 0.01 and not snap:
            return  # evicted, that's fine
        assert len(snap) == 1
        assert snap[0].activation_strength == pytest.approx(expected, rel=1e-9)

    def test_eviction_below_threshold(self) -> None:
        buf = PerceptTraceBuffer(tau=10.0)
        buf.record("a", "p1")
        # After ~50 ticks, activation ≈ exp(-50/10) ≈ 0.0067 < 0.01
        for _ in range(50):
            buf.tick()
        assert len(buf) == 0
        assert buf.snapshot() == []


class TestRingBuffer:
    def test_ring_buffer_overflow(self) -> None:
        buf = PerceptTraceBuffer(max_entries=5)
        for i in range(10):
            buf.record("a", f"p{i}")
        assert len(buf) == 5
        snap = buf.snapshot()
        ids = {e.percept_id for e in snap}
        # Oldest 5 should be evicted, newest 5 survive
        for i in range(5, 10):
            assert f"p{i}" in ids
        for i in range(5):
            assert f"p{i}" not in ids


class TestReset:
    def test_reset_all(self) -> None:
        buf = PerceptTraceBuffer()
        buf.record("a", "p1")
        buf.record("b", "p2")
        buf.reset()
        assert len(buf) == 0

    def test_reset_per_agent(self) -> None:
        buf = PerceptTraceBuffer()
        buf.record("a", "p1")
        buf.record("a", "p2")
        buf.record("b", "p3")
        buf.reset(agent_id="a")
        assert len(buf) == 1
        snap = buf.snapshot()
        assert snap[0].agent_id == "b"


class TestAgentIsolation:
    def test_snapshot_filters_by_agent(self) -> None:
        buf = PerceptTraceBuffer()
        buf.record("a", "p1")
        buf.record("a", "p2")
        buf.record("b", "p3")
        snap_a = buf.snapshot(agent_id="a")
        snap_b = buf.snapshot(agent_id="b")
        assert len(snap_a) == 2
        assert all(e.agent_id == "a" for e in snap_a)
        assert len(snap_b) == 1
        assert snap_b[0].agent_id == "b"


class TestRecent:
    def test_recent_returns_k_most_recent(self) -> None:
        buf = PerceptTraceBuffer()
        for i in range(10):
            buf.record("a", f"p{i}")
        recent = buf.recent(k=3)
        assert len(recent) == 3
        # Most recent first
        assert recent[0].percept_id == "p9"
        assert recent[1].percept_id == "p8"
        assert recent[2].percept_id == "p7"


class TestTick:
    def test_tick_advances_counter(self) -> None:
        buf = PerceptTraceBuffer()
        assert buf.current_tick == 0
        buf.tick()
        assert buf.current_tick == 1
        buf.tick()
        buf.tick()
        assert buf.current_tick == 3


class TestSnapshotSorting:
    def test_snapshot_sorted_by_activation(self) -> None:
        buf = PerceptTraceBuffer(tau=10.0)
        buf.record("a", "old")
        for _ in range(5):
            buf.tick()
        buf.record("a", "new")
        snap = buf.snapshot()
        assert len(snap) == 2
        assert snap[0].percept_id == "new"
        assert snap[1].percept_id == "old"
        assert snap[0].activation_strength > snap[1].activation_strength


class TestEmptyBuffer:
    def test_empty_buffer(self) -> None:
        buf = PerceptTraceBuffer()
        assert buf.snapshot() == []
        assert buf.recent() == []
        assert len(buf) == 0


class TestThreadSafety:
    def test_thread_safety(self) -> None:
        buf = PerceptTraceBuffer(max_entries=200)
        errors: list[Exception] = []
        barrier = threading.Barrier(8)

        def writer(agent_id: str) -> None:
            try:
                barrier.wait(timeout=5)
                for i in range(100):
                    buf.record(agent_id, f"p{i}")
                    if i % 10 == 0:
                        buf.tick()
            except Exception as exc:
                errors.append(exc)

        def reader(agent_id: str) -> None:
            try:
                barrier.wait(timeout=5)
                for _ in range(100):
                    buf.snapshot(agent_id=agent_id)
                    buf.recent(agent_id=agent_id, k=5)
                    len(buf)
            except Exception as exc:
                errors.append(exc)

        threads = []
        for i in range(4):
            threads.append(threading.Thread(target=writer, args=(f"w{i}",)))
            threads.append(threading.Thread(target=reader, args=(f"w{i}",)))
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)
        assert not errors, f"Thread errors: {errors}"
