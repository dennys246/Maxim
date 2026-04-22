"""Tests for WorkingMemorySet (0.8 Stage 1).

Covers:
- Basic add/query interface
- Thread safety (concurrent add from 5 threads × 100 adds)
- Kind filtering
- Capacity bounds
- Persistence migration (tier="working" → "short_term")
- Monotonic tick counter
"""

from __future__ import annotations

import threading
import time
from typing import Any

import pytest

from maxim.agents.working_memory import WMEntry, WorkingMemoryKind, WorkingMemorySet


# ─────────────────────────────────────────────────────────────────────────────
# Basic interface
# ─────────────────────────────────────────────────────────────────────────────


class TestWorkingMemorySetBasic:
    def test_add_returns_entry(self):
        wms = WorkingMemorySet(agent_id="test")
        entry = wms.add(WorkingMemoryKind.PERCEPT, content="hello", ref="p-1")
        assert isinstance(entry, WMEntry)
        assert entry.kind == WorkingMemoryKind.PERCEPT
        assert entry.content == "hello"
        assert entry.ref == "p-1"
        assert entry.agent_id == "test"
        assert entry.tick == 0

    def test_add_increments_tick(self):
        wms = WorkingMemorySet(agent_id="test")
        e1 = wms.add(WorkingMemoryKind.PERCEPT, content="a")
        e2 = wms.add(WorkingMemoryKind.OUTCOME, content="b")
        assert e1.tick == 0
        assert e2.tick == 1
        assert wms.current_tick == 2

    def test_recent_returns_oldest_first(self):
        wms = WorkingMemorySet(agent_id="test")
        wms.add(WorkingMemoryKind.PERCEPT, content="first")
        wms.add(WorkingMemoryKind.PERCEPT, content="second")
        wms.add(WorkingMemoryKind.PERCEPT, content="third")
        items = wms.recent(limit=3)
        assert [e.content for e in items] == ["first", "second", "third"]

    def test_recent_respects_limit(self):
        wms = WorkingMemorySet(agent_id="test")
        for i in range(10):
            wms.add(WorkingMemoryKind.PERCEPT, content=f"item-{i}")
        items = wms.recent(limit=3)
        assert len(items) == 3
        assert items[-1].content == "item-9"

    def test_by_kind_filters(self):
        wms = WorkingMemorySet(agent_id="test")
        wms.add(WorkingMemoryKind.PERCEPT, content="p1")
        wms.add(WorkingMemoryKind.OUTCOME, content="o1")
        wms.add(WorkingMemoryKind.PERCEPT, content="p2")
        wms.add(WorkingMemoryKind.THOUGHT, content="t1")

        percepts = wms.by_kind({WorkingMemoryKind.PERCEPT})
        assert len(percepts) == 2
        assert all(e.kind == WorkingMemoryKind.PERCEPT for e in percepts)
        # Most-recent first
        assert percepts[0].content == "p2"

    def test_by_kind_multi(self):
        wms = WorkingMemorySet(agent_id="test")
        wms.add(WorkingMemoryKind.PERCEPT, content="p")
        wms.add(WorkingMemoryKind.OUTCOME, content="o")
        wms.add(WorkingMemoryKind.THOUGHT, content="t")

        results = wms.by_kind({WorkingMemoryKind.PERCEPT, WorkingMemoryKind.THOUGHT})
        assert len(results) == 2

    def test_since_tick(self):
        wms = WorkingMemorySet(agent_id="test")
        wms.add(WorkingMemoryKind.PERCEPT, content="a")
        wms.add(WorkingMemoryKind.PERCEPT, content="b")
        wms.add(WorkingMemoryKind.PERCEPT, content="c")
        items = wms.since_tick(1)
        assert len(items) == 2
        assert items[0].content == "b"

    def test_for_agent(self):
        wms = WorkingMemorySet(agent_id="agent-1")
        wms.add(WorkingMemoryKind.PERCEPT, content="a", agent_id="agent-1")
        wms.add(WorkingMemoryKind.PERCEPT, content="b", agent_id="agent-2")
        items = wms.for_agent("agent-1")
        assert len(items) == 1
        assert items[0].content == "a"

    def test_size(self):
        wms = WorkingMemorySet(agent_id="test")
        assert wms.size == 0
        wms.add(WorkingMemoryKind.PERCEPT, content="a")
        assert wms.size == 1

    def test_clear(self):
        wms = WorkingMemorySet(agent_id="test")
        wms.add(WorkingMemoryKind.PERCEPT, content="a")
        wms.add(WorkingMemoryKind.PERCEPT, content="b")
        old_tick = wms.current_tick
        wms.clear()
        assert wms.size == 0
        # tick is NOT reset — monotonic
        assert wms.current_tick == old_tick


# ─────────────────────────────────────────────────────────────────────────────
# Capacity bounds
# ─────────────────────────────────────────────────────────────────────────────


class TestWorkingMemorySetCapacity:
    def test_bounded_deque(self):
        wms = WorkingMemorySet(agent_id="test", capacity=5)
        for i in range(10):
            wms.add(WorkingMemoryKind.PERCEPT, content=f"item-{i}")
        assert wms.size == 5
        items = wms.recent()
        assert items[0].content == "item-5"  # oldest surviving
        assert items[-1].content == "item-9"  # newest


# ─────────────────────────────────────────────────────────────────────────────
# Thread safety (F3)
# ─────────────────────────────────────────────────────────────────────────────


class TestWorkingMemorySetConcurrency:
    def test_concurrent_add_5_threads_100_each(self):
        """5 threads × 100 adds = 500 entries. All must be present."""
        wms = WorkingMemorySet(agent_id="test", capacity=1000)
        barrier = threading.Barrier(5)
        errors: list[Exception] = []

        def worker(thread_id: int) -> None:
            try:
                barrier.wait(timeout=5)
                for i in range(100):
                    wms.add(
                        WorkingMemoryKind.PERCEPT,
                        content=f"t{thread_id}-{i}",
                        ref=f"ref-{thread_id}-{i}",
                    )
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not errors, f"Thread errors: {errors}"
        assert wms.size == 500
        # Verify tick counter is exactly 500
        assert wms.current_tick == 500
        # Verify all entries have unique ticks
        items = wms.recent(limit=500)
        ticks = [e.tick for e in items]
        assert len(set(ticks)) == 500

    def test_concurrent_add_and_read(self):
        """Reads interleaved with writes must not crash."""
        wms = WorkingMemorySet(agent_id="test", capacity=500)
        stop = threading.Event()
        errors: list[Exception] = []

        def writer() -> None:
            try:
                for i in range(200):
                    wms.add(WorkingMemoryKind.PERCEPT, content=f"w-{i}")
            except Exception as e:
                errors.append(e)
            finally:
                stop.set()

        def reader() -> None:
            try:
                while not stop.is_set():
                    _ = wms.recent(limit=10)
                    _ = wms.by_kind({WorkingMemoryKind.PERCEPT}, limit=5)
                    _ = wms.size
            except Exception as e:
                errors.append(e)

        w = threading.Thread(target=writer)
        r = threading.Thread(target=reader)
        w.start()
        r.start()
        w.join(timeout=10)
        r.join(timeout=10)
        assert not errors, f"Thread errors: {errors}"


# ─────────────────────────────────────────────────────────────────────────────
# Persistence migration (F4): tier="working" → "short_term" on load
# ─────────────────────────────────────────────────────────────────────────────


class TestTierPersistenceMigration:
    def _make_ep(self, ep_id: str) -> Any:
        from maxim.memory.types import EpisodicMemory, Perception

        return EpisodicMemory(
            id=ep_id,
            timestamp=time.time(),
            run_id=f"run-{ep_id}",
            perception=Perception(salience=0.8),
        )

    def test_working_tier_migrates_to_short_term(self):
        """Old serialized data with tier='working' loads as SHORT_TERM."""
        from maxim.agents.bus import MemoryTier, WorkingMemoryEntry

        ep = self._make_ep("ep-migrate")
        data = {
            "record_type": "EpisodicMemory",
            "record": ep.to_dict(),
            "salience": 0.7,
            "decay_rate": 0.1,
            "source": "percept",
            "tier": "working",  # OLD tier value
            "predicted_outcomes": None,
            "prediction_confidence": 0.0,
        }

        entry = WorkingMemoryEntry.from_dict(data)
        assert entry.tier == MemoryTier.SHORT_TERM  # Migrated!
        assert entry.salience == 0.7

    def test_short_tier_loads_normally(self):
        """Current tier='short' still works."""
        from maxim.agents.bus import MemoryTier, WorkingMemoryEntry

        ep = self._make_ep("ep-short")
        data = {
            "record_type": "EpisodicMemory",
            "record": ep.to_dict(),
            "tier": "short",
        }
        entry = WorkingMemoryEntry.from_dict(data)
        assert entry.tier == MemoryTier.SHORT_TERM

    def test_forming_tier_loads_normally(self):
        """tier='forming' still works."""
        from maxim.agents.bus import MemoryTier, WorkingMemoryEntry

        ep = self._make_ep("ep-forming")
        data = {
            "record_type": "EpisodicMemory",
            "record": ep.to_dict(),
            "tier": "forming",
        }
        entry = WorkingMemoryEntry.from_dict(data)
        assert entry.tier == MemoryTier.FORMING


# ─────────────────────────────────────────────────────────────────────────────
# WMEntry frozen semantics
# ─────────────────────────────────────────────────────────────────────────────


class TestWMEntry:
    def test_frozen(self):
        entry = WMEntry(
            kind=WorkingMemoryKind.PERCEPT,
            ref="p-1",
            content="hello",
            timestamp=time.time(),
            tick=0,
        )
        with pytest.raises(AttributeError):
            entry.content = "modified"  # type: ignore[misc]

    def test_all_kinds_exist(self):
        expected = {"percept", "outcome", "reaction", "pain", "activation", "thought", "recall", "conversation"}
        actual = {k.value for k in WorkingMemoryKind}
        assert actual == expected
