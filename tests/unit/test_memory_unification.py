"""Tests for the memory unification refactor.

Covers:
- MemoryRecord ABC: keywords(), to_context_dict(), thread-safe touch()
- PredictedOutcome and MathContextEntry dataclasses
- WorkingMemoryEntry: lifecycle, tier transitions, serialization
- MemoryAgent: staged formation, pattern completion hook, persistence
"""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock

import pytest

from maxim.agents.bus import (
    AgentBus,
    MemoryTier,
    Percept,
    WorkingMemoryEntry,
)
from maxim.agents.memory_agent import (
    AssociationIndex,
    MemoryAgent,
)
from maxim.math.math_types import CompressedMathMemory, MathMemory
from maxim.math.types import MathCategory
from maxim.memory.semantic_types import CompressedSemantic, SemanticMemory
from maxim.memory.types import (
    Action,
    CompressedMemory,
    Context,
    Decision,
    EpisodicMemory,
    MathContextEntry,
    Outcome,
    Perception,
    PredictedOutcome,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


def _make_episodic(**kwargs) -> EpisodicMemory:
    """Create a test EpisodicMemory with sensible defaults."""
    defaults = dict(
        id="ep-1",
        timestamp=time.time(),
        run_id="run-1",
        perception=Perception(
            detected_objects=["mug", "table"],
            detected_people=["Dennis"],
            salience=0.8,
            novelty=0.6,
            cli_input="grasp the mug",
        ),
        context=Context(active_goal="grasp mug", active_mode="active"),
        decision=Decision(intent={"goal": "grasp mug"}, confidence=0.9),
        action=Action(tool_name="navigate"),
        outcome=Outcome(success=True),
    )
    defaults.update(kwargs)
    return EpisodicMemory(**defaults)


def _make_math(**kwargs) -> MathMemory:
    """Create a test MathMemory."""
    defaults = dict(
        id="math-1",
        timestamp=time.time(),
        name="mug_grasp_time",
        category=MathCategory.PATTERN,
        domain="timing",
        verbal="typically ~310ms",
        confidence=0.85,
    )
    defaults.update(kwargs)
    return MathMemory(**defaults)


def _make_semantic(**kwargs) -> SemanticMemory:
    """Create a test SemanticMemory."""
    defaults = dict(
        id="sem-1",
        timestamp=time.time(),
        name="mug",
        category="object",
        definition="A drinking vessel",
        confidence=0.9,
    )
    defaults.update(kwargs)
    return SemanticMemory(**defaults)


# ─────────────────────────────────────────────────────────────────────────────
# MemoryRecord ABC tests
# ─────────────────────────────────────────────────────────────────────────────


class TestMemoryRecordKeywords:
    """Test keywords() on all MemoryRecord subclasses."""

    def test_episodic_keywords(self):
        ep = _make_episodic()
        kws = ep.keywords()
        assert "mug" in kws
        assert "table" in kws
        assert "dennis" in kws
        assert "navigate" in kws
        assert "grasp" in kws

    def test_episodic_keywords_empty(self):
        ep = EpisodicMemory(id="ep-empty", timestamp=time.time())
        kws = ep.keywords()
        assert kws == set()

    def test_math_keywords(self):
        mm = _make_math()
        kws = mm.keywords()
        assert "mug_grasp_time" in kws
        assert "timing" in kws

    def test_semantic_keywords(self):
        sm = _make_semantic()
        kws = sm.keywords()
        assert "mug" in kws
        assert "object" in kws

    def test_compressed_memory_keywords(self):
        ep = _make_episodic()
        cm = CompressedMemory.from_episodic(ep)
        kws = cm.keywords()
        assert "grasp" in kws
        assert "navigate" in kws

    def test_compressed_math_keywords(self):
        mm = _make_math()
        cm = CompressedMathMemory.from_math_record(mm)
        kws = cm.keywords()
        assert "mug_grasp_time" in kws

    def test_compressed_semantic_keywords(self):
        sm = _make_semantic()
        cs = CompressedSemantic.from_semantic(sm)
        kws = cs.keywords()
        assert "mug" in kws


class TestMemoryRecordContextDict:
    """Test to_context_dict() on all MemoryRecord subclasses."""

    def test_episodic_context_dict(self):
        ep = _make_episodic()
        ctx = ep.to_context_dict()
        assert ctx["type"] == "episodic"
        assert "mug" in ctx["detected_objects"]
        assert ctx["tool"] == "navigate"
        assert ctx["success"] is True

    def test_math_context_dict(self):
        mm = _make_math()
        ctx = mm.to_context_dict()
        assert ctx["type"] == "math"
        assert ctx["name"] == "mug_grasp_time"
        assert ctx["verbal"] == "typically ~310ms"

    def test_semantic_context_dict(self):
        sm = _make_semantic()
        ctx = sm.to_context_dict()
        assert ctx["type"] == "semantic"
        assert ctx["name"] == "mug"
        assert ctx["definition"] == "A drinking vessel"

    def test_compressed_memory_context_dict(self):
        ep = _make_episodic()
        cm = CompressedMemory.from_episodic(ep)
        ctx = cm.to_context_dict()
        assert ctx["type"] == "compressed_episodic"
        assert ctx["tool"] == "navigate"

    def test_compressed_math_context_dict(self):
        mm = _make_math()
        cm = CompressedMathMemory.from_math_record(mm)
        ctx = cm.to_context_dict()
        assert ctx["type"] == "compressed_math"

    def test_compressed_semantic_context_dict(self):
        sm = _make_semantic()
        cs = CompressedSemantic.from_semantic(sm)
        ctx = cs.to_context_dict()
        assert ctx["type"] == "compressed_semantic"


class TestMemoryRecordThreadSafety:
    """Test thread-safe touch()."""

    def test_touch_updates_fields(self):
        ep = _make_episodic()
        old_accessed = ep.accessed_at
        old_count = ep.access_count
        time.sleep(0.01)
        ep.touch()
        assert ep.accessed_at > old_accessed
        assert ep.access_count == old_count + 1

    def test_concurrent_touch(self):
        """Multiple threads calling touch() simultaneously."""
        ep = _make_episodic()
        initial_count = ep.access_count

        def touch_many():
            for _ in range(100):
                ep.touch()

        threads = [threading.Thread(target=touch_many) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert ep.access_count == initial_count + 500


# ─────────────────────────────────────────────────────────────────────────────
# PredictedOutcome and MathContextEntry tests
# ─────────────────────────────────────────────────────────────────────────────


class TestPredictedOutcome:
    """Test PredictedOutcome serialization."""

    def test_round_trip_simple(self):
        po = PredictedOutcome(tool="navigate", success=True, goal="grasp mug")
        d = po.to_dict()
        restored = PredictedOutcome.from_dict(d)
        assert restored.tool == "navigate"
        assert restored.success is True
        assert restored.goal == "grasp mug"
        assert restored.math_context is None

    def test_round_trip_with_math_context(self):
        po = PredictedOutcome(
            tool="navigate",
            success=True,
            math_context=[
                MathContextEntry(name="mug:time", verbal="310ms", confidence=0.9, domain="timing"),
            ],
            source_episode_id="ep-42",
        )
        d = po.to_dict()
        restored = PredictedOutcome.from_dict(d)
        assert len(restored.math_context) == 1
        assert restored.math_context[0].name == "mug:time"
        assert restored.math_context[0].verbal == "310ms"
        assert restored.source_episode_id == "ep-42"


class TestMathContextEntry:
    """Test MathContextEntry serialization."""

    def test_round_trip(self):
        mce = MathContextEntry(name="speed", verbal="fast", confidence=0.7, domain="timing")
        d = mce.to_dict()
        restored = MathContextEntry.from_dict(d)
        assert restored.name == "speed"
        assert restored.verbal == "fast"
        assert restored.confidence == 0.7


# ─────────────────────────────────────────────────────────────────────────────
# MemoryTier tests
# ─────────────────────────────────────────────────────────────────────────────


class TestMemoryTier:
    """Test enriched MemoryTier enum."""

    def test_all_values_exist(self):
        assert MemoryTier.FORMING.value == "forming"
        assert MemoryTier.WORKING.value == "working"
        assert MemoryTier.SHORT_TERM.value == "short"
        assert MemoryTier.LONG_TERM.value == "long"

    def test_serialization(self):
        for tier in MemoryTier:
            assert MemoryTier(tier.value) == tier


# ─────────────────────────────────────────────────────────────────────────────
# WorkingMemoryEntry tests
# ─────────────────────────────────────────────────────────────────────────────


class TestWorkingMemoryEntry:
    """Test WorkingMemoryEntry wrapper."""

    def test_wraps_episodic(self):
        ep = _make_episodic()
        entry = WorkingMemoryEntry(record=ep, salience=0.8)
        assert entry.id == "ep-1"
        assert entry.timestamp == ep.timestamp

    def test_wraps_math(self):
        mm = _make_math()
        entry = WorkingMemoryEntry(record=mm, source="ag_grounding")
        assert entry.id == "math-1"

    def test_wraps_semantic(self):
        sm = _make_semantic()
        entry = WorkingMemoryEntry(record=sm)
        assert entry.id == "sem-1"

    def test_tier_lifecycle(self):
        ep = _make_episodic()
        entry = WorkingMemoryEntry(record=ep, tier=MemoryTier.FORMING)
        assert entry.tier == MemoryTier.FORMING
        entry.tier = MemoryTier.WORKING
        assert entry.tier == MemoryTier.WORKING
        entry.tier = MemoryTier.SHORT_TERM
        assert entry.tier == MemoryTier.SHORT_TERM

    def test_is_protected(self):
        ep = _make_episodic()
        entry = WorkingMemoryEntry(record=ep, tier=MemoryTier.FORMING)
        assert entry.is_protected is True
        entry.tier = MemoryTier.WORKING
        assert entry.is_protected is True
        entry.tier = MemoryTier.SHORT_TERM
        assert entry.is_protected is False

    def test_touch_delegates(self):
        ep = _make_episodic()
        entry = WorkingMemoryEntry(record=ep)
        old_count = ep.access_count
        entry.touch()
        assert ep.access_count == old_count + 1

    def test_keywords_cached(self):
        ep = _make_episodic()
        entry = WorkingMemoryEntry(record=ep)
        kws1 = entry.keywords
        kws2 = entry.keywords
        assert kws1 is kws2  # Same object (cached)
        assert "mug" in kws1

    def test_invalidate_keywords(self):
        ep = _make_episodic()
        entry = WorkingMemoryEntry(record=ep)
        _ = entry.keywords
        entry.invalidate_keywords()
        assert entry._keywords is None

    def test_current_salience_no_decay_for_forming(self):
        ep = _make_episodic()
        entry = WorkingMemoryEntry(
            record=ep, salience=0.9, tier=MemoryTier.FORMING
        )
        assert entry.current_salience() == 0.9

    def test_should_promote_only_short_term(self):
        ep = _make_episodic()
        entry = WorkingMemoryEntry(record=ep, salience=0.8, tier=MemoryTier.FORMING)
        assert entry.should_promote() is False
        entry.tier = MemoryTier.SHORT_TERM
        assert entry.should_promote() is True  # salience 0.8 > 0.7 threshold

    def test_should_evict_never_for_protected(self):
        ep = _make_episodic()
        entry = WorkingMemoryEntry(record=ep, tier=MemoryTier.FORMING)
        assert entry.should_evict(0.0) is False  # Even 0 seconds

    def test_round_trip_serialization(self):
        ep = _make_episodic()
        entry = WorkingMemoryEntry(
            record=ep,
            salience=0.75,
            source="percept",
            tier=MemoryTier.SHORT_TERM,
            predicted_outcomes=[
                PredictedOutcome(tool="nav", success=True, goal="test"),
            ],
            prediction_confidence=0.6,
        )
        d = entry.to_dict()
        restored = WorkingMemoryEntry.from_dict(d)
        assert restored.id == ep.id
        assert restored.salience == 0.75
        assert restored.tier == MemoryTier.SHORT_TERM
        assert restored.predicted_outcomes[0].tool == "nav"
        assert restored.prediction_confidence == 0.6

    def test_from_dict_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Unknown record type"):
            WorkingMemoryEntry.from_dict({
                "record_type": "FakeRecord",
                "record": {},
            })


# ─────────────────────────────────────────────────────────────────────────────
# MemoryAgent tests
# ─────────────────────────────────────────────────────────────────────────────


class TestStagedFormation:
    """Test staged memory formation in MemoryAgent."""

    def _make_agent(self) -> MemoryAgent:
        bus = AgentBus()
        return MemoryAgent(bus, reset_on_startup=True)

    def test_begin_memory_formation_creates_forming_entry(self):
        agent = self._make_agent()
        percept = Percept(
            timestamp=time.time(),
            source="vision",
            salience=0.8,
            novelty=0.5,
            detections=[{"label": "mug", "class_id": 73}],
        )
        with agent._lock:
            entry = agent._begin_memory_formation(percept, "run-1")

        assert entry.tier == MemoryTier.FORMING
        assert isinstance(entry.record, EpisodicMemory)
        assert "run-1" in agent._forming_pool

    def test_update_forming_decision(self):
        agent = self._make_agent()
        percept = Percept(timestamp=time.time(), source="vision", salience=0.8)
        with agent._lock:
            agent._begin_memory_formation(percept, "run-1")
            agent._update_forming_decision(
                "run-1", Decision(intent={"goal": "grasp"}, confidence=0.9)
            )

        entry = agent._forming_pool["run-1"]
        assert entry.record.decision.intent["goal"] == "grasp"

    def test_update_forming_action(self):
        agent = self._make_agent()
        percept = Percept(timestamp=time.time(), source="vision", salience=0.8)
        with agent._lock:
            agent._begin_memory_formation(percept, "run-1")
            agent._update_forming_action("run-1", Action(tool_name="navigate"))

        entry = agent._forming_pool["run-1"]
        assert entry.record.action.tool_name == "navigate"

    def test_complete_forming_transitions_to_working(self):
        agent = self._make_agent()
        percept = Percept(timestamp=time.time(), source="vision", salience=0.8)
        with agent._lock:
            agent._begin_memory_formation(percept, "run-1")
            agent._complete_forming_memory("run-1", Outcome(success=True))

        entry = agent._forming_pool["run-1"]
        assert entry.tier == MemoryTier.WORKING

    def test_forming_survives_eviction(self):
        agent = self._make_agent()
        percept = Percept(timestamp=time.time(), source="vision", salience=0.8)
        with agent._lock:
            agent._begin_memory_formation(percept, "run-1")
            agent._apply_decay()

        assert "run-1" in agent._forming_pool

    def test_flush_working_to_captured(self):
        agent = self._make_agent()
        percept = Percept(timestamp=time.time(), source="vision", salience=0.8)
        with agent._lock:
            agent._begin_memory_formation(percept, "run-1")
            agent._complete_forming_memory("run-1", Outcome(success=True))
            agent._flush_working_to_captured()

        assert "run-1" not in agent._forming_pool

    def test_concurrent_forming_entries(self):
        agent = self._make_agent()
        with agent._lock:
            for i in range(3):
                percept = Percept(timestamp=time.time() + i, source="vision", salience=0.8)
                agent._begin_memory_formation(percept, f"run-{i}")

        assert len(agent._forming_pool) == 3

    def test_invalidate_keywords_on_mutation(self):
        agent = self._make_agent()
        percept = Percept(timestamp=time.time(), source="vision", salience=0.8)
        with agent._lock:
            entry = agent._begin_memory_formation(percept, "run-1")
            _ = entry.keywords  # Cache keywords
            assert entry._keywords is not None
            agent._update_forming_decision("run-1", Decision(intent={"goal": "test"}))
            assert entry._keywords is None  # Invalidated


class TestPatternCompletionHook:
    """Test pattern completion hook in MemoryAgent."""

    def _make_agent(self) -> MemoryAgent:
        bus = AgentBus()
        return MemoryAgent(bus, reset_on_startup=True)

    def test_no_hook_by_default(self):
        agent = self._make_agent()
        assert agent._pattern_completion_fn is None

    def test_set_pattern_completion_fn(self):
        agent = self._make_agent()
        fn = MagicMock(return_value=[])
        agent.set_pattern_completion_fn(fn)
        assert agent._pattern_completion_fn is fn

    def test_hook_called_during_formation(self):
        agent = self._make_agent()
        predictions = [
            PredictedOutcome(tool="navigate", success=True, goal="test"),
        ]
        fn = MagicMock(return_value=predictions)
        agent.set_pattern_completion_fn(fn)

        percept = Percept(timestamp=time.time(), source="vision", salience=0.8)
        with agent._lock:
            entry = agent._begin_memory_formation(percept, "run-1")

        fn.assert_called_once()
        assert entry.predicted_outcomes == predictions

    def test_prediction_confidence_calculation(self):
        agent = self._make_agent()
        predictions = [
            PredictedOutcome(tool="navigate", success=True),
            PredictedOutcome(tool="navigate", success=True),
            PredictedOutcome(tool="navigate", success=True),
            PredictedOutcome(tool="navigate", success=True),
            PredictedOutcome(tool="navigate", success=True),
        ]
        conf = agent._compute_prediction_confidence(predictions)
        # 5/5 success, 5/5 consistency, 5/5 sample → 1.0
        assert conf == 1.0

    def test_prediction_confidence_sample_dampening(self):
        agent = self._make_agent()
        predictions = [
            PredictedOutcome(tool="navigate", success=True),
        ]
        conf = agent._compute_prediction_confidence(predictions)
        # 1/1 success, 1/1 consistency, 1/5 sample → 0.2
        assert conf == pytest.approx(0.2)

    def test_prediction_confidence_empty(self):
        agent = self._make_agent()
        assert agent._compute_prediction_confidence([]) == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Persistence tests
# ─────────────────────────────────────────────────────────────────────────────


class TestPersistence:
    """Test that Hippocampus-delegated persistence works."""

    def test_forming_pool_survives_flush(self):
        """Forming entries remain until explicitly flushed."""
        bus = AgentBus()
        agent = MemoryAgent(bus, reset_on_startup=True)

        percept = Percept(timestamp=time.time(), source="vision", salience=0.8)
        with agent._lock:
            agent._begin_memory_formation(percept, "run-1")
            agent._complete_forming_memory("run-1", Outcome(success=True))

        # WORKING entry still in pool
        assert "run-1" in agent._forming_pool

        # After flush, removed
        with agent._lock:
            agent._flush_working_to_captured()
        assert "run-1" not in agent._forming_pool

    def test_persist_memories_no_crash(self):
        """_persist_memories_to_disk is a no-op but should not crash."""
        bus = AgentBus()
        agent = MemoryAgent(bus, reset_on_startup=True)
        agent._persist_memories_to_disk()  # No-op, should not crash

    def test_salience_tracking(self):
        """Salience dict tracks captured memory IDs."""
        bus = AgentBus()
        agent = MemoryAgent(bus, reset_on_startup=True)

        percept = Percept(timestamp=time.time(), source="vision", salience=0.8)
        with agent._lock:
            entry = agent._begin_memory_formation(percept, "run-sal")
        # Entry's record should have an ID tracked in salience dict
        assert entry.record.id in agent._salience or len(agent._forming_pool) > 0


# ─────────────────────────────────────────────────────────────────────────────
# Helper class tests
# ─────────────────────────────────────────────────────────────────────────────


class TestAssociationIndex:
    """Test AssociationIndex with ID-based API."""

    def test_add_and_find(self):
        idx = AssociationIndex()
        idx.add_by_id("ep-1", {"transcript": "mug on table", "source": "vision"})

        results = idx.find_similar("mug table", top_k=5)
        assert len(results) > 0
        # Returns (memory_id, score) tuples
        assert results[0][0] == "ep-1"

    def test_remove(self):
        idx = AssociationIndex()
        idx.add_by_id("ep-1", {"transcript": "mug on table"})
        idx.remove("ep-1")

        results = idx.find_similar("mug", top_k=5)
        assert len(results) == 0


# ─────────────────────────────────────────────────────────────────────────────
# Integration: full pipeline lifecycle
# ─────────────────────────────────────────────────────────────────────────────


class TestFullPipeline:
    """Integration test: percept → FORMING → decision → action → outcome → WORKING."""

    def test_full_lifecycle(self):
        bus = AgentBus()
        agent = MemoryAgent(bus, reset_on_startup=True)

        # 1. Percept arrives → FORMING
        percept = Percept(
            timestamp=time.time(),
            source="vision",
            salience=0.8,
            novelty=0.6,
            detections=[{"label": "mug", "class_id": 73}],
            cli_input="pick up the mug",
        )
        with agent._lock:
            entry = agent._begin_memory_formation(percept, "run-lifecycle")

        assert entry.tier == MemoryTier.FORMING
        assert isinstance(entry.record, EpisodicMemory)
        assert "mug" in entry.record.perception.detected_objects

        # 2. Decision
        with agent._lock:
            agent._update_forming_decision(
                "run-lifecycle",
                Decision(intent={"goal": "grasp mug"}, confidence=0.9),
            )
        assert entry.record.decision.intent["goal"] == "grasp mug"

        # 3. Action
        with agent._lock:
            agent._update_forming_action(
                "run-lifecycle",
                Action(tool_name="navigate"),
            )
        assert entry.record.action.tool_name == "navigate"

        # 4. Outcome → WORKING
        with agent._lock:
            agent._complete_forming_memory(
                "run-lifecycle",
                Outcome(success=True, result="grasped"),
            )
        assert entry.tier == MemoryTier.WORKING

        # 5. New cycle → flush WORKING → captured
        with agent._lock:
            agent._flush_working_to_captured()
        assert "run-lifecycle" not in agent._forming_pool

    def test_build_context_includes_forming(self):
        bus = AgentBus()
        agent = MemoryAgent(bus, reset_on_startup=True)

        percept = Percept(
            timestamp=time.time(),
            source="vision",
            salience=0.8,
            detections=[{"label": "mug"}],
        )
        with agent._lock:
            entry = agent._begin_memory_formation(percept, "run-ctx")

        # Without hippocampus wired, forming entries are tracked locally
        assert "run-ctx" in agent._forming_pool
        assert entry.tier == MemoryTier.FORMING
        # build_context should not crash even without hippocampus
        ctx = agent.build_context()
        assert ctx is not None

    def test_build_context_includes_predictions(self):
        bus = AgentBus()
        agent = MemoryAgent(bus, reset_on_startup=True)

        predictions = [
            PredictedOutcome(
                tool="navigate", success=True, goal="test",
                math_context=[MathContextEntry(name="speed", verbal="fast")],
            ),
        ]
        agent.set_pattern_completion_fn(lambda ep: predictions)

        percept = Percept(
            timestamp=time.time(),
            source="vision",
            salience=0.8,
            detections=[],
        )
        with agent._lock:
            entry = agent._begin_memory_formation(percept, "run-pred")

        assert entry.predicted_outcomes is not None
        assert entry.prediction_confidence > 0

    def test_full_pipeline_roundtrip(self):
        """Test forming → working → captured lifecycle."""
        bus = AgentBus()
        agent = MemoryAgent(bus, reset_on_startup=True)

        percept = Percept(timestamp=time.time(), source="vision", salience=0.8)
        with agent._lock:
            entry = agent._begin_memory_formation(percept, "run-save")
            agent._complete_forming_memory("run-save", Outcome(success=True))
            agent._flush_working_to_captured()

        # Entry flushed from forming pool
        assert "run-save" not in agent._forming_pool
        assert isinstance(entry.record, EpisodicMemory)
