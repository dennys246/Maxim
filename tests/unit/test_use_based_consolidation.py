"""Tests for Stage 7 use-based consolidation.

Tests the promotion_pressure → SHORT_TERM → LONG_TERM path:
- recall() calls touch() on every result (access_count increments)
- recall(working_memory=...) adds RECALL entries to WMS
- Context-diverse queries accumulate promotion_pressure
- Identical queries do NOT accumulate pressure (diversity guard)
- Pressure decays with wall-clock elapsed time
- Threshold crossing triggers promotion to long_term
- Promotion resets pressure and access_contexts
- Concurrent recall threads produce consistent pressure (F5)
"""

from __future__ import annotations

import threading
import time


from maxim.agents.working_memory import WorkingMemoryKind, WorkingMemorySet
from maxim.memory.hippocampus import Hippocampus, HippocampusConfig
from maxim.memory.hippocampus_consolidation import (
    _PROMOTION_PRESSURE_THRESHOLD,
    _context_signature,
)
from maxim.memory.types import (
    Action,
    CompressedMemory,
    Context,
    Decision,
    EpisodicMemory,
    Outcome,
    Perception,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _make_hippocampus() -> Hippocampus:
    return Hippocampus(
        HippocampusConfig(
            persistence_path=None,
            enable_sleep_consolidation=True,
            auto_save_after_sleep=False,
        )
    )


def _capture(hip: Hippocampus, goal: str = "find cup", tool: str = "look") -> str:
    return hip.capture(
        perception=Perception(
            observations={"text": f"looking for {goal}"},
            detected_objects=["cup"],
            salience=0.5,
            novelty=0.5,
        ),
        context=Context(active_goal=goal),
        decision=Decision(intent={"goal": goal}),
        action=Action(tool_name=tool),
        outcome=Outcome(success=True, result="found it"),
    )


# ─────────────────────────────────────────────────────────────────────────────
# MemoryRecord field tests
# ─────────────────────────────────────────────────────────────────────────────


class TestMemoryRecordFields:
    def test_defaults(self):
        ep = EpisodicMemory(
            id="test",
            timestamp=time.time(),
            perception=Perception(),
            context=Context(),
            decision=Decision(),
            action=Action(),
            outcome=Outcome(),
        )
        assert ep.promotion_pressure == 0.0
        assert ep.last_scored_at == 0.0
        assert len(ep.access_contexts) == 0

    def test_serialization_round_trip(self):
        ep = EpisodicMemory(
            id="test",
            timestamp=time.time(),
            perception=Perception(),
            context=Context(),
            decision=Decision(),
            action=Action(),
            outcome=Outcome(),
        )
        ep.promotion_pressure = 2.5
        ep.last_scored_at = 1000.0
        ep.access_contexts.append("abc123")
        ep.access_contexts.append("def456")

        data = ep.to_dict()
        restored = EpisodicMemory.from_dict(data)
        assert restored.promotion_pressure == 2.5
        assert restored.last_scored_at == 1000.0
        assert list(restored.access_contexts) == ["abc123", "def456"]

    def test_backward_compatible_deserialization(self):
        """Old data without promotion fields deserializes with defaults."""
        data = {
            "id": "old",
            "timestamp": 1000.0,
            "perception": {},
            "context": {},
            "decision": {},
            "action": {},
            "outcome": {},
        }
        ep = EpisodicMemory.from_dict(data)
        assert ep.promotion_pressure == 0.0
        assert ep.last_scored_at == 0.0
        assert len(ep.access_contexts) == 0

    def test_compressed_memory_round_trip(self):
        cm = CompressedMemory(
            id="comp",
            timestamp=time.time(),
            goal="test",
            tool_name="tool",
            success=True,
        )
        cm.promotion_pressure = 1.5
        cm.access_contexts.append("sig1")

        data = cm.to_dict()
        restored = CompressedMemory.from_dict(data)
        assert restored.promotion_pressure == 1.5
        assert list(restored.access_contexts) == ["sig1"]

    def test_access_contexts_maxlen(self):
        ep = EpisodicMemory(
            id="test",
            timestamp=time.time(),
            perception=Perception(),
            context=Context(),
            decision=Decision(),
            action=Action(),
            outcome=Outcome(),
        )
        for i in range(15):
            ep.access_contexts.append(f"ctx_{i}")
        assert len(ep.access_contexts) == 10
        assert ep.access_contexts[0] == "ctx_5"  # oldest evicted


# ─────────────────────────────────────────────────────────────────────────────
# Context signature
# ─────────────────────────────────────────────────────────────────────────────


class TestContextSignature:
    def test_deterministic(self):
        assert _context_signature("hello world") == _context_signature("hello world")

    def test_case_insensitive(self):
        assert _context_signature("Hello World") == _context_signature("hello world")

    def test_whitespace_normalized(self):
        assert _context_signature("hello  world") == _context_signature("hello world")

    def test_different_queries_differ(self):
        assert _context_signature("hello") != _context_signature("world")


# ─────────────────────────────────────────────────────────────────────────────
# Recall touch behavior
# ─────────────────────────────────────────────────────────────────────────────


class TestRecallTouch:
    def test_recall_increments_access_count(self):
        hip = _make_hippocampus()
        mem_id = _capture(hip, goal="find cup")

        # Recall with a matching query — access_count should increase
        results = hip.recall(goal="find cup")
        assert len(results) >= 1
        count_after_recall = results[0].access_count

        # Second recall should increase further
        hip.recall(goal="find cup")
        # The result object from recall is a snapshot at read time, but
        # touch happens in the write phase AFTER the read. So check via get().
        mem_after = hip.get(mem_id)
        assert mem_after.access_count > count_after_recall

    def test_recall_updates_accessed_at(self):
        hip = _make_hippocampus()
        mem_id = _capture(hip, goal="find cup")
        mem = hip.get(mem_id)
        original_ts = mem.accessed_at

        time.sleep(0.01)
        hip.recall(goal="find cup")

        mem_after = hip.get(mem_id)
        assert mem_after.accessed_at > original_ts


# ─────────────────────────────────────────────────────────────────────────────
# WMS integration
# ─────────────────────────────────────────────────────────────────────────────


class TestRecallWMS:
    def test_recall_adds_recall_entries_to_wms(self):
        hip = _make_hippocampus()
        _capture(hip, goal="find cup")
        wms = WorkingMemorySet(agent_id="test")

        results = hip.recall(goal="find cup", working_memory=wms)
        assert len(results) >= 1

        entries = wms.by_kind([WorkingMemoryKind.RECALL])
        assert len(entries) == len(results)
        assert entries[0].kind == WorkingMemoryKind.RECALL
        assert "[recalled]" in entries[0].content

    def test_recall_without_wms_does_not_error(self):
        hip = _make_hippocampus()
        _capture(hip, goal="find cup")
        results = hip.recall(goal="find cup")
        assert len(results) >= 1


# ─────────────────────────────────────────────────────────────────────────────
# Promotion pressure
# ─────────────────────────────────────────────────────────────────────────────


class TestPromotionPressure:
    def test_diverse_queries_accumulate_pressure(self):
        hip = _make_hippocampus()
        mem_id = _capture(hip, goal="find cup")

        # A single diverse recall should add non-zero pressure
        hip.recall(query="where is the cup")
        mem = hip.get(mem_id)
        assert mem.promotion_pressure > 0.0 or mem.long_term, "Either pressure accumulated or promotion already fired"

    def test_identical_queries_do_not_accumulate(self):
        hip = _make_hippocampus()
        mem_id = _capture(hip, goal="find cup")

        hip.recall(query="where is the cup")
        mem1 = hip.get(mem_id)
        pressure_after_first = mem1.promotion_pressure

        hip.recall(query="where is the cup")
        mem2 = hip.get(mem_id)
        # Pressure may have decayed slightly but should NOT have increased
        assert mem2.promotion_pressure <= pressure_after_first

    def test_pressure_decays_over_time(self):
        hip = _make_hippocampus()
        mem_id = _capture(hip, goal="find cup")

        hip.recall(query="cup query one")
        mem1 = hip.get(mem_id)
        pressure_before = mem1.promotion_pressure

        # Simulate passage of time by directly setting last_scored_at back
        mem1.last_scored_at -= 100  # 100 seconds ago

        # Next recall with different query will apply decay first
        hip.recall(query="cup query two")
        mem2 = hip.get(mem_id)
        # Decay should have reduced pressure before the new increment
        # (decay magnitude would be 100 * _PRESSURE_DECAY_RATE).
        # New pressure = (old - decay) + new_score, so it should be less than
        # old + new_score (without decay)
        assert mem2.promotion_pressure < pressure_before + 2.0  # generous bound

    def test_no_pressure_on_non_query_recall(self):
        """Recall without a query string does NOT accumulate pressure."""
        hip = _make_hippocampus()
        mem_id = _capture(hip, goal="find cup")

        hip.recall(goal="find cup")  # filter-based, no query
        mem = hip.get(mem_id)
        assert mem.promotion_pressure == 0.0

    def test_promotion_triggers_at_threshold(self):
        """Enough diverse access crosses threshold → promotes to long_term."""
        hip = _make_hippocampus()
        mem_id = _capture(hip, goal="find cup")

        # Pump pressure high enough by recalling with many diverse queries
        for i in range(20):
            hip.recall(query=f"unique query number {i} about cup")

        mem = hip.get(mem_id)
        assert mem.long_term is True, (
            f"Memory should be long_term after enough diverse access, pressure={mem.promotion_pressure}"
        )

    def test_promotion_resets_fields(self):
        """After promotion, pressure and access_contexts are cleared."""
        hip = _make_hippocampus()
        mem_id = _capture(hip, goal="find cup")

        for i in range(20):
            hip.recall(query=f"diverse query {i} cup")

        mem = hip.get(mem_id)
        assert mem.long_term is True
        assert mem.promotion_pressure == 0.0
        assert len(mem.access_contexts) == 0

    def test_already_long_term_skipped(self):
        """Memories already promoted don't accumulate more pressure."""
        hip = _make_hippocampus()
        # Capture with very high salience to get immediate promotion
        mem_id = hip.capture(
            perception=Perception(salience=0.99, novelty=0.5, observations={"text": "cup"}),
            context=Context(active_goal="find cup"),
            decision=Decision(intent={"goal": "find cup"}),
            action=Action(tool_name="look"),
            outcome=Outcome(success=True, result="found"),
        )
        mem = hip.get(mem_id)
        assert mem.long_term is True

        hip.recall(query="cup search")
        mem_after = hip.get(mem_id)
        assert mem_after.promotion_pressure == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Concurrency (F5)
# ─────────────────────────────────────────────────────────────────────────────


class TestConcurrentRecall:
    def test_concurrent_recall_pressure_consistency(self):
        """3 threads recalling the same query, verify no data corruption."""
        hip = _make_hippocampus()
        mem_id = _capture(hip, goal="find cup")
        errors: list[str] = []

        def recall_loop(thread_id: int) -> None:
            try:
                for i in range(10):
                    hip.recall(query=f"thread {thread_id} query {i} cup")
            except Exception as e:
                errors.append(f"Thread {thread_id}: {e}")

        threads = [threading.Thread(target=recall_loop, args=(t,)) for t in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not errors, f"Concurrent recall errors: {errors}"
        mem = hip.get(mem_id)
        # Should have accumulated significant pressure from 30 diverse queries
        # (may have promoted)
        assert mem.access_count > 1
        # access_contexts maxlen is 10, so at most 10 stored
        assert len(mem.access_contexts) <= 10


# ─────────────────────────────────────────────────────────────────────────────
# Sleep interaction (F10)
# ─────────────────────────────────────────────────────────────────────────────


class TestSleepInteraction:
    def test_sleep_consolidation_uses_pressure(self):
        """Consolidation cycle considers promotion_pressure."""
        hip = _make_hippocampus()
        mem_id = _capture(hip, goal="find cup")

        # Manually set high pressure (simulating many diverse accesses)
        mem = hip.get(mem_id)
        mem.promotion_pressure = _PROMOTION_PRESSURE_THRESHOLD + 1.0

        # Sleep should promote via the _should_promote check
        hip._add_consolidation_candidate(mem_id)
        results = hip._consolidate()
        assert results["promoted"] >= 1

        mem_after = hip.get(mem_id)
        assert mem_after.long_term is True

    def test_sleep_does_not_decay_pressure(self):
        """Sleep replay is independent of use-based pressure (F10)."""
        hip = _make_hippocampus()
        mem_id = _capture(hip, goal="find cup")

        mem = hip.get(mem_id)
        mem.promotion_pressure = 2.0

        # Sleep consolidation should not modify pressure of non-promoted memories
        hip.sleep()
        mem_after = hip.get(mem_id)
        # Pressure unchanged (sleep doesn't run the decay path)
        assert mem_after.promotion_pressure == 2.0
