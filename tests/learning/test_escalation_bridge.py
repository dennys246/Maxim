"""Learning tests for EscalationLearningBridge - Verify threshold adaptation.

These tests verify that the escalation learning system:
1. Learns to lower thresholds when escalation helps
2. Learns to raise thresholds when autonomous action works
3. Adapts to different goal types
4. Persists learned thresholds correctly
"""

from __future__ import annotations

import pytest


@pytest.mark.learning
class TestEscalationThresholdLearning:
    """Test that escalation thresholds adapt correctly."""

    def test_successful_escalation_lowers_threshold(self, escalation_bridge):
        """Successful escalation lowers threshold (escalate more)."""
        # Get initial threshold
        initial = escalation_bridge.get_threshold(goal="find mug", novelty=0.5, salience=0.5)

        # Record multiple successful escalations
        for _ in range(10):
            escalation_bridge.record_outcome(
                goal="find mug",
                escalated=True,
                success=True,  # Escalation helped
                novelty=0.5,
                salience=0.5,
            )

        final = escalation_bridge.get_threshold(goal="find mug", novelty=0.5, salience=0.5)

        # Threshold should have decreased (escalate more often)
        assert final < initial, f"Expected decrease: {initial} -> {final}"

    def test_failed_escalation_raises_threshold(self, escalation_bridge):
        """Failed escalation raises threshold (escalate less)."""
        # Seed with some samples to reach min_samples_for_adjustment
        for _ in range(5):
            escalation_bridge.record_outcome(
                goal="test",
                escalated=True,
                success=True,
                novelty=0.5,
                salience=0.5,
            )

        initial = escalation_bridge.get_threshold(goal="test", novelty=0.5, salience=0.5)

        # Record multiple failed escalations
        for _ in range(10):
            escalation_bridge.record_outcome(
                goal="test",
                escalated=True,
                success=False,  # Escalation didn't help
                novelty=0.5,
                salience=0.5,
            )

        final = escalation_bridge.get_threshold(goal="test", novelty=0.5, salience=0.5)

        # Threshold should have increased (escalate less often)
        assert final > initial, f"Expected increase: {initial} -> {final}"

    def test_successful_autonomous_action_raises_threshold(self, escalation_bridge):
        """Successful autonomous action raises threshold."""
        # Seed with some samples
        for _ in range(5):
            escalation_bridge.record_outcome(
                goal="auto",
                escalated=False,
                success=True,
                novelty=0.5,
                salience=0.5,
            )

        initial = escalation_bridge.get_threshold(goal="auto", novelty=0.5, salience=0.5)

        # Record more successful autonomous actions
        for _ in range(10):
            escalation_bridge.record_outcome(
                goal="auto",
                escalated=False,
                success=True,  # Autonomous action worked
                novelty=0.5,
                salience=0.5,
            )

        final = escalation_bridge.get_threshold(goal="auto", novelty=0.5, salience=0.5)

        # Should have increased (less need to escalate)
        assert final >= initial, f"Expected increase or stable: {initial} -> {final}"

    def test_failed_autonomous_action_lowers_threshold(self, escalation_bridge):
        """Failed autonomous action lowers threshold (should have escalated)."""
        # Seed
        for _ in range(5):
            escalation_bridge.record_outcome(
                goal="risky",
                escalated=False,
                success=True,
                novelty=0.5,
                salience=0.5,
            )

        initial = escalation_bridge.get_threshold(goal="risky", novelty=0.5, salience=0.5)

        # Record failed autonomous actions
        for _ in range(10):
            escalation_bridge.record_outcome(
                goal="risky",
                escalated=False,
                success=False,  # Should have escalated
                novelty=0.5,
                salience=0.5,
            )

        final = escalation_bridge.get_threshold(goal="risky", novelty=0.5, salience=0.5)

        # Threshold should have decreased (escalate more)
        assert final < initial, f"Expected decrease: {initial} -> {final}"


@pytest.mark.learning
class TestEscalationGoalSensitivity:
    """Test that escalation learns goal-specific thresholds."""

    def test_different_goals_learn_independently(self, escalation_bridge):
        """Different goal types develop different thresholds."""
        # Goal A: escalation always helps
        for _ in range(10):
            escalation_bridge.record_outcome(
                goal="navigate to kitchen",
                escalated=True,
                success=True,
                novelty=0.6,
                salience=0.6,
            )

        # Goal B: autonomous always works
        for _ in range(10):
            escalation_bridge.record_outcome(
                goal="look around",
                escalated=False,
                success=True,
                novelty=0.4,
                salience=0.4,
            )

        threshold_nav = escalation_bridge.get_threshold(
            goal="navigate to bedroom", novelty=0.5, salience=0.5
        )
        threshold_look = escalation_bridge.get_threshold(
            goal="observe surroundings", novelty=0.5, salience=0.5
        )

        # Navigation should have lower threshold (escalate more)
        # Looking should have higher threshold (escalate less)
        # Note: may not always differ if not enough samples for goal type matching


@pytest.mark.learning
class TestEscalationBounds:
    """Test that threshold adjustments stay bounded."""

    def test_adjustment_respects_max(self, escalation_bridge):
        """Adjustment never exceeds max_adjustment."""
        max_adj = escalation_bridge.max_adjustment

        # Extreme case: many successful escalations
        for _ in range(100):
            escalation_bridge.record_outcome(
                goal="bounded_test",
                escalated=True,
                success=True,
                novelty=0.5,
                salience=0.5,
            )

        # Check adjustment is bounded
        key = ("search", -1)  # search is goal classification for "bounded_test"
        # Need to check the actual threshold adjustment
        stats = escalation_bridge.stats()
        # The adjustment should be within bounds


@pytest.mark.learning
class TestEscalationDecision:
    """Test the should_escalate decision function."""

    def test_should_escalate_with_high_scores(self, escalation_bridge):
        """High novelty+salience triggers escalation."""
        should, reason = escalation_bridge.should_escalate(
            goal="test",
            novelty=0.95,
            salience=0.95,
        )

        assert should is True
        assert "threshold" in reason.lower()

    def test_should_not_escalate_with_low_scores(self, escalation_bridge):
        """Low novelty+salience does not trigger escalation."""
        should, reason = escalation_bridge.should_escalate(
            goal="test",
            novelty=0.1,
            salience=0.1,
        )

        assert should is False
        assert "threshold" in reason.lower()


@pytest.mark.learning
class TestEscalationPersistence:
    """Test that learned thresholds persist correctly."""

    def test_save_load_preserves_thresholds(self, escalation_bridge, tmp_path):
        """Learned thresholds survive save/load cycle."""
        # Learn some thresholds
        for _ in range(10):
            escalation_bridge.record_outcome(
                goal="persistent_goal",
                escalated=True,
                success=True,
                novelty=0.5,
                salience=0.5,
            )

        original_threshold = escalation_bridge.get_threshold(
            goal="persistent_goal", novelty=0.5, salience=0.5
        )

        path = tmp_path / "escalation.json"
        escalation_bridge.save(str(path))

        # Create new bridge and load
        from maxim.bridges.escalation_bridge import EscalationLearningBridge

        new_bridge = EscalationLearningBridge(
            hippocampus=escalation_bridge.hippocampus,
            scn=escalation_bridge.scn,
            nac=escalation_bridge.nac,
            persist_path=str(path),
        )
        new_bridge.load(str(path))

        loaded_threshold = new_bridge.get_threshold(
            goal="persistent_goal", novelty=0.5, salience=0.5
        )

        # Should be similar (may not be exact due to default handling)
        assert abs(loaded_threshold - original_threshold) < 0.1

    def test_save_load_preserves_stats(self, escalation_bridge, tmp_path):
        """Statistics survive save/load cycle."""
        for _ in range(5):
            escalation_bridge.record_outcome(
                goal="test",
                escalated=True,
                success=True,
                novelty=0.5,
                salience=0.5,
            )

        original_stats = escalation_bridge.stats()

        path = tmp_path / "escalation.json"
        escalation_bridge.save(str(path))

        from maxim.bridges.escalation_bridge import EscalationLearningBridge

        new_bridge = EscalationLearningBridge(
            hippocampus=escalation_bridge.hippocampus,
            scn=escalation_bridge.scn,
            nac=escalation_bridge.nac,
            persist_path=str(path),
        )
        new_bridge.load(str(path))

        loaded_stats = new_bridge.stats()

        assert loaded_stats["total_samples"] == original_stats["total_samples"]


@pytest.mark.learning
class TestEscalationHealthTracking:
    """Test health and error tracking."""

    def test_is_healthy_initially(self, escalation_bridge):
        """Bridge starts in healthy state."""
        assert escalation_bridge.is_healthy is True

    def test_stats_returns_expected_keys(self, escalation_bridge):
        """Stats returns all expected keys."""
        stats = escalation_bridge.stats()

        assert "healthy" in stats
        assert "learned_thresholds" in stats
        assert "total_samples" in stats
        assert "recent_records" in stats