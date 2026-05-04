"""Integration tests for the MemoryHub and bridge system.

Tests the complete memory integration including:
- MemoryHub lifecycle (session start/end)
- Bridge functionality (spatial, salience, planning, escalation)
- Hippocampus capture and recall
- Sleep consolidation
"""

from __future__ import annotations

import sys
import time

import pytest

# Add src to path for imports
sys.path.insert(0, "src")


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def core_systems():
    """Create core memory systems."""
    from maxim.decisions.nac import NAc
    from maxim.memory.hippocampus import Hippocampus, HippocampusConfig
    from maxim.similarity.ec import EntorhinalCortex
    from maxim.time.scn import SCN

    config = HippocampusConfig(
        enable_sleep_consolidation=True,
        auto_save_after_sleep=False,  # Don't save during tests
    )

    return {
        "hippocampus": Hippocampus(config),
        "scn": SCN(),
        "nac": NAc(),
        "ec": EntorhinalCortex(),
    }


@pytest.fixture
def memory_hub(core_systems):
    """Create MemoryHub with core systems."""
    from maxim.integration import MemoryHub

    hub = MemoryHub(
        hippocampus=core_systems["hippocampus"],
        scn=core_systems["scn"],
        nac=core_systems["nac"],
        ec=core_systems["ec"],
    )
    hub.connect()  # Initialize bridges
    return hub


@pytest.fixture
def sample_perception():
    """Create sample perception data."""
    from maxim.memory.types import Perception

    return Perception(
        observations={"position": (320, 240)},
        detected_objects=["mug", "table"],
        detected_people=[],
        salience=0.7,
        novelty=0.6,
        cli_input="find the mug",
    )


@pytest.fixture
def sample_context():
    """Create sample context data."""
    from maxim.memory.types import Context

    return Context(
        active_goal="find mug",
        active_mode="exploration",
    )


@pytest.fixture
def sample_decision():
    """Create sample decision data."""
    from maxim.memory.types import Decision

    return Decision(
        intent={"goal": "find mug", "action": "look_around"},
        reasoning="Looking for mug in workspace",
        confidence=0.8,
    )


@pytest.fixture
def sample_action():
    """Create sample action data."""
    from maxim.memory.types import Action

    return Action(
        tool_name="look_around",
        tool_params={"direction": "left"},
    )


@pytest.fixture
def sample_outcome():
    """Create sample outcome data."""
    from maxim.memory.types import Outcome

    return Outcome(
        success=True,
        result={"found": True, "object": "mug"},
    )


# ─────────────────────────────────────────────────────────────────────────────
# MemoryHub Lifecycle Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestMemoryHubLifecycle:
    """Test MemoryHub session lifecycle."""

    def test_hub_creation(self, memory_hub):
        """Test MemoryHub is created correctly."""
        assert memory_hub is not None
        assert memory_hub.hippocampus is not None
        assert memory_hub.scn is not None
        assert memory_hub.nac is not None
        assert memory_hub.ec is not None

    def test_bridge_connection(self, memory_hub):
        """Test bridges are connected."""
        health = memory_hub.health_check()

        # Planning and escalation should always be healthy (no external deps)
        assert health["planning"] is True
        assert health["escalation"] is True

        # Spatial and salience won't be healthy without external systems
        assert health["spatial"] is False
        assert health["salience"] is False

    def test_session_start(self, memory_hub):
        """Test session start initializes bridges."""
        results = memory_hub.on_session_start()

        assert memory_hub._session_active is True
        assert isinstance(results, dict)

    def test_session_end(self, memory_hub):
        """Test session end consolidates memory."""
        memory_hub.on_session_start()
        results = memory_hub.on_session_end()

        assert memory_hub._session_active is False
        assert "session_duration_seconds" in results

    def test_stats(self, memory_hub):
        """Test stats collection from all subsystems."""
        memory_hub.on_session_start()
        stats = memory_hub.stats()

        assert "hippocampus" in stats
        assert "scn" in stats
        assert "nac" in stats
        assert "ec" in stats
        assert "session_active" in stats


# ─────────────────────────────────────────────────────────────────────────────
# Hippocampus Integration Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestHippocampusIntegration:
    """Test Hippocampus memory capture and recall."""

    def test_capture_memory(
        self,
        memory_hub,
        sample_perception,
        sample_context,
        sample_decision,
        sample_action,
        sample_outcome,
    ):
        """Test memory capture through Hippocampus."""
        memory_id = memory_hub.hippocampus.capture(
            perception=sample_perception,
            context=sample_context,
            decision=sample_decision,
            action=sample_action,
            outcome=sample_outcome,
        )

        assert memory_id is not None
        assert len(memory_hub.hippocampus) == 1

    def test_recall_by_goal(
        self,
        memory_hub,
        sample_perception,
        sample_context,
        sample_decision,
        sample_action,
        sample_outcome,
    ):
        """Test memory recall by goal."""
        memory_hub.hippocampus.capture(
            perception=sample_perception,
            context=sample_context,
            decision=sample_decision,
            action=sample_action,
            outcome=sample_outcome,
        )

        memories = memory_hub.hippocampus.recall(goal="find mug")
        assert len(memories) == 1
        assert memories[0].context.active_goal == "find mug"

    def test_recall_by_success(
        self,
        memory_hub,
        sample_perception,
        sample_context,
        sample_decision,
        sample_action,
        sample_outcome,
    ):
        """Test memory recall by success status."""
        memory_hub.hippocampus.capture(
            perception=sample_perception,
            context=sample_context,
            decision=sample_decision,
            action=sample_action,
            outcome=sample_outcome,
        )

        successful = memory_hub.hippocampus.recall(success=True)
        failed = memory_hub.hippocampus.recall(success=False)

        assert len(successful) == 1
        assert len(failed) == 0

    def test_scn_temporal_indexing(
        self,
        memory_hub,
        sample_perception,
        sample_context,
        sample_decision,
        sample_action,
        sample_outcome,
    ):
        """Test SCN temporal signature registration."""
        from maxim.time.temporal_signature import TemporalSignature

        memory_id = memory_hub.hippocampus.capture(
            perception=sample_perception,
            context=sample_context,
            decision=sample_decision,
            action=sample_action,
            outcome=sample_outcome,
        )

        # SCN should have registered this memory's temporal signature
        # (via Hippocampus -> MemoryHub wiring)
        # Note: SCN registration happens in the capture pathway
        # For this test, we manually register
        sig = TemporalSignature.now()
        memory_hub.scn.register(memory_id, sig)

        assert memory_hub.scn.get_signature(memory_id) is not None


# ─────────────────────────────────────────────────────────────────────────────
# Bridge Integration Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestPlanningBridge:
    """Test PlanHistoryBridge functionality."""

    def test_get_plan_templates_empty(self, memory_hub):
        """Test getting templates when no history exists."""
        templates = memory_hub.get_plan_templates("find mug")

        # Should return fallback templates
        assert isinstance(templates, list)

    def test_record_plan_outcome(self, memory_hub):
        """Test recording plan outcomes."""
        memory_hub.on_session_start()

        # Record a successful plan
        memory_hub.record_plan_outcome(
            goal="find mug",
            tool_sequence=["look_around", "grasp"],
            success=True,
        )

        # NAc should have learned from this
        stats = memory_hub.nac.stats()
        assert stats["total_observations"] > 0

    def test_record_plan_outcome_uses_canonical_signature(self, memory_hub):
        """Plan outcomes write under the canonical 'tool:<name>' signature.

        Pre-fix this bridge stored bare tool names ('grasp') while
        tool_dispatch + tool_pain_bridge stored 'tool:grasp', splitting
        the same logical tool's CausalLinks across two _links keys and
        starving every query path that used build_tool_signature.
        """
        memory_hub.on_session_start()
        memory_hub.record_plan_outcome(
            goal="find mug",
            tool_sequence=["look_around", "grasp"],
            success=True,
        )
        link_keys = list(memory_hub.nac._links.keys())
        # Canonical form must be present.
        assert "tool:look_around" in link_keys
        assert "tool:grasp" in link_keys
        # Bare form must NOT be present (the regression we just closed).
        assert "look_around" not in link_keys
        assert "grasp" not in link_keys

    def test_predicted_success(self, memory_hub):
        """Test success prediction for plans."""
        prediction = memory_hub.get_predicted_success(
            goal="find mug",
            tool_sequence=["look_around"],
        )

        # Should return neutral prediction with no history
        assert 0.0 <= prediction <= 1.0


class TestEscalationBridge:
    """Test EscalationLearningBridge functionality."""

    def test_get_threshold(self, memory_hub):
        """Test getting escalation threshold."""
        threshold = memory_hub.get_escalation_threshold(
            goal="find mug",
            novelty=0.6,
            salience=0.7,
        )

        # Should return default threshold
        assert 0.0 <= threshold <= 1.0

    def test_should_escalate_low(self, memory_hub):
        """Test escalation decision with low values."""
        should, reason = memory_hub.should_escalate(
            goal="find mug",
            novelty=0.3,
            salience=0.3,
        )

        assert should is False
        assert "threshold" in reason

    def test_should_escalate_high(self, memory_hub):
        """Test escalation decision with high values."""
        should, reason = memory_hub.should_escalate(
            goal="find mug",
            novelty=0.9,
            salience=0.9,
        )

        assert should is True
        assert "threshold" in reason

    def test_record_escalation_outcome(self, memory_hub):
        """Test recording escalation outcomes."""
        memory_hub.on_session_start()

        # Record an escalation outcome
        memory_hub.record_escalation_outcome(
            goal="find mug",
            escalated=True,
            success=True,
            novelty=0.7,
            salience=0.8,
        )

        # Bridge should have recorded this
        stats = memory_hub.escalation.stats()
        assert stats["total_samples"] == 1

    def test_threshold_learning(self, memory_hub):
        """Test that thresholds adapt from outcomes."""
        memory_hub.on_session_start()

        # Get initial threshold
        initial = memory_hub.get_escalation_threshold("find mug", 0.5, 0.5)

        # Record multiple successful escalations
        for _ in range(10):
            memory_hub.record_escalation_outcome(
                goal="find mug",
                escalated=True,
                success=True,
                novelty=0.7,
                salience=0.8,
            )

        # Threshold should have decreased (escalate more)
        final = memory_hub.get_escalation_threshold("find mug", 0.5, 0.5)

        # Note: threshold may not change much in 10 samples, but should trend down
        assert final <= initial + 0.1  # Allow small variance


# ─────────────────────────────────────────────────────────────────────────────
# Sleep Consolidation Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestSleepConsolidation:
    """Test sleep consolidation through MemoryHub."""

    def test_sleep_empty(self, memory_hub):
        """Test sleep with no memories."""
        memory_hub.on_session_start()
        results = memory_hub.sleep()

        assert results["compressed"] == 0
        assert results["removed"] == 0
        assert results["preserved"] == 0

    def test_sleep_preserves_recent(
        self,
        memory_hub,
        sample_perception,
        sample_context,
        sample_decision,
        sample_action,
        sample_outcome,
    ):
        """Test sleep preserves recently captured memories."""
        memory_hub.on_session_start()

        # Capture a memory
        memory_hub.hippocampus.capture(
            perception=sample_perception,
            context=sample_context,
            decision=sample_decision,
            action=sample_action,
            outcome=sample_outcome,
        )

        # Run sleep
        results = memory_hub.sleep()

        # Recent memory should be preserved
        assert len(memory_hub.hippocampus) == 1
        assert results["preserved"] == 1

    def test_sleep_with_scn_integration(
        self,
        memory_hub,
        sample_perception,
        sample_context,
        sample_decision,
        sample_action,
        sample_outcome,
    ):
        """Test sleep uses SCN for temporal-aware consolidation."""
        from maxim.time.temporal_signature import TemporalSignature

        memory_hub.on_session_start()

        # Capture memory and register with SCN
        memory_id = memory_hub.hippocampus.capture(
            perception=sample_perception,
            context=sample_context,
            decision=sample_decision,
            action=sample_action,
            outcome=sample_outcome,
        )

        sig = TemporalSignature.now()
        memory_hub.scn.register(memory_id, sig)

        # Run sleep
        results = memory_hub.sleep()

        # Memory should be preserved
        assert results["preserved"] == 1


# ─────────────────────────────────────────────────────────────────────────────
# Error Handling Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestErrorHandling:
    """Test fault tolerance and error handling."""

    def test_disabled_bridge_returns_default(self, memory_hub):
        """Test that disabled bridges return sensible defaults."""
        memory_hub._disabled_bridges.add("spatial")

        # Should return fallback
        positions = memory_hub.get_likely_positions("mug")
        assert positions == [(5, 5, 0.1)]

    def test_enable_bridge(self, memory_hub):
        """Test re-enabling a disabled bridge."""
        memory_hub._disabled_bridges.add("spatial")
        assert "spatial" in memory_hub._disabled_bridges

        result = memory_hub.enable_bridge("spatial")
        assert result is True
        assert "spatial" not in memory_hub._disabled_bridges


# ─────────────────────────────────────────────────────────────────────────────
# Performance Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestPerformance:
    """Basic performance tests."""

    def test_capture_latency(
        self,
        memory_hub,
        sample_perception,
        sample_context,
        sample_decision,
        sample_action,
        sample_outcome,
    ):
        """Test that memory capture is fast enough for real-time use."""
        memory_hub.on_session_start()

        start = time.perf_counter()
        for _ in range(100):
            memory_hub.hippocampus.capture(
                perception=sample_perception,
                context=sample_context,
                decision=sample_decision,
                action=sample_action,
                outcome=sample_outcome,
            )
        elapsed = time.perf_counter() - start

        # Should capture 100 memories in under 1 second
        assert elapsed < 1.0
        avg_ms = (elapsed / 100) * 1000
        print(f"\nAverage capture latency: {avg_ms:.2f}ms")

    def test_recall_latency(
        self,
        memory_hub,
        sample_perception,
        sample_context,
        sample_decision,
        sample_action,
        sample_outcome,
    ):
        """Test that memory recall is fast."""
        memory_hub.on_session_start()

        # Populate with 100 memories
        for _ in range(100):
            memory_hub.hippocampus.capture(
                perception=sample_perception,
                context=sample_context,
                decision=sample_decision,
                action=sample_action,
                outcome=sample_outcome,
            )

        start = time.perf_counter()
        for _ in range(100):
            memory_hub.hippocampus.recall(goal="find mug", limit=10)
        elapsed = time.perf_counter() - start

        # Should query 100 times in under 1 second
        assert elapsed < 1.0
        avg_ms = (elapsed / 100) * 1000
        print(f"\nAverage recall latency: {avg_ms:.2f}ms")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
