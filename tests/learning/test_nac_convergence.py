"""Learning tests for NAc - Verify TD learning converges correctly.

These tests verify that the Rescorla-Wagner learning algorithm:
1. Converges toward correct predictions
2. Learns at appropriate rates
3. Is stable (doesn't oscillate)
4. Handles mixed signals correctly
"""

from __future__ import annotations

import pytest


@pytest.mark.learning
class TestNAcConvergence:
    """Test that NAc predictions converge toward true values."""

    def test_consistent_positive_increases_prediction(self, nac, valence_positive):
        """Consistent positive outcomes increase predicted value."""

        # Observe consistently positive outcomes
        for i in range(20):
            link = nac.observe(
                event_type="tool",
                event_signature="always_works",
                outcome_type="result",
                outcome_signature="success",
                outcome_valence=valence_positive,
                delta_seconds=1.0,
            )

        # Predicted value should have increased toward 1.0
        final_value = link.predicted_value
        assert final_value > 0.8, f"Expected prediction > 0.8, got {final_value}"

    def test_consistent_negative_decreases_prediction(self, nac, valence_negative):
        """Consistent negative outcomes decrease predicted value."""
        # Observe consistently negative outcomes
        for i in range(20):
            link = nac.observe(
                event_type="tool",
                event_signature="always_fails",
                outcome_type="result",
                outcome_signature="failure",
                outcome_valence=valence_negative,
                delta_seconds=1.0,
            )

        # Predicted value should have decreased toward 0.0
        final_value = link.predicted_value
        assert final_value < 0.2, f"Expected prediction < 0.2, got {final_value}"

    def test_mixed_outcomes_converge_to_proportion(
        self, nac, valence_positive, valence_negative
    ):
        """Mixed outcomes converge to approximate success rate."""
        # 70% positive, 30% negative
        for i in range(100):
            valence = valence_positive if i % 10 < 7 else valence_negative

            link = nac.observe(
                event_type="tool",
                event_signature="mostly_works",
                outcome_type="result",
                outcome_signature="result",
                outcome_valence=valence,
                delta_seconds=1.0,
            )

        # Should converge toward ~0.7
        final_value = link.predicted_value
        assert 0.5 < final_value < 0.9, f"Expected ~0.7, got {final_value}"

    def test_learning_is_monotonic_for_consistent_signal(
        self, nac, valence_positive
    ):
        """Consistent positive signal produces monotonic increase."""
        predictions = []

        for i in range(30):
            link = nac.observe(
                event_type="tool",
                event_signature="consistent",
                outcome_type="result",
                outcome_signature="success",
                outcome_valence=valence_positive,
                delta_seconds=1.0,
            )
            predictions.append(link.predicted_value)

        # Check mostly monotonic (allow 1-2 small dips due to R-W dynamics)
        increases = sum(
            1 for i in range(1, len(predictions)) if predictions[i] >= predictions[i - 1]
        )
        monotonicity_ratio = increases / (len(predictions) - 1)

        assert monotonicity_ratio > 0.9, f"Expected mostly monotonic, got {monotonicity_ratio}"

    def test_learning_rate_affects_convergence_speed(self, valence_positive):
        """Higher learning rate = faster convergence."""
        from maxim.decisions.nac import NAc, NACConfig

        slow_nac = NAc(config=NACConfig(base_learning_rate=0.05))
        fast_nac = NAc(config=NACConfig(base_learning_rate=0.5))

        # Same observations
        for i in range(10):
            slow_nac.observe(
                event_type="tool",
                event_signature="test",
                outcome_type="result",
                outcome_signature="success",
                outcome_valence=valence_positive,
                delta_seconds=1.0,
            )
            fast_nac.observe(
                event_type="tool",
                event_signature="test",
                outcome_type="result",
                outcome_signature="success",
                outcome_valence=valence_positive,
                delta_seconds=1.0,
            )

        slow_links = slow_nac.get_links_for_event("test")
        fast_links = fast_nac.get_links_for_event("test")

        slow_value = slow_links[0].predicted_value
        fast_value = fast_links[0].predicted_value

        # Fast learner should have higher value after same observations
        assert fast_value > slow_value, f"Fast {fast_value} should > Slow {slow_value}"

    def test_prediction_stabilizes_after_many_observations(
        self, nac, valence_positive, valence_negative
    ):
        """Predictions stabilize (low variance) after many observations."""
        # 60% positive rate
        for i in range(100):
            valence = valence_positive if i % 10 < 6 else valence_negative
            link = nac.observe(
                event_type="tool",
                event_signature="stable",
                outcome_type="result",
                outcome_signature="result",
                outcome_valence=valence,
                delta_seconds=1.0,
            )

        # Last 20 predictions should be stable
        history = link.prediction_history[-20:]
        if len(history) >= 10:
            variance = sum((v - sum(history) / len(history)) ** 2 for v in history) / len(
                history
            )
            assert variance < 0.02, f"Predictions not stable, variance={variance}"


@pytest.mark.learning
class TestNAcContextSensitivity:
    """Test that NAc learns context-dependent predictions."""

    def test_different_contexts_learn_independently(
        self, nac, valence_positive, valence_negative
    ):
        """Same action, different contexts, different predictions."""
        # Context A: always positive
        for _ in range(10):
            nac.observe(
                event_type="tool",
                event_signature="grasp",
                outcome_type="result",
                outcome_signature="result",
                outcome_valence=valence_positive,
                delta_seconds=1.0,
                context={"object": "mug"},
            )

        # Context B: always negative
        for _ in range(10):
            nac.observe(
                event_type="tool",
                event_signature="grasp",
                outcome_type="result",
                outcome_signature="result",
                outcome_valence=valence_negative,
                delta_seconds=1.0,
                context={"object": "heavy_box"},
            )

        # Predictions should differ by context
        pred_mug = nac.predict("tool", "grasp", context={"object": "mug"})
        pred_box = nac.predict("tool", "grasp", context={"object": "heavy_box"})

        assert pred_mug is not None
        assert pred_box is not None
        assert pred_mug.predicted_value > pred_box.predicted_value


@pytest.mark.learning
class TestNAcTemporalLearning:
    """Test that NAc learns temporal patterns correctly."""

    def test_temporal_delta_converges_to_mean(self, nac, valence_positive):
        """Temporal delta statistics converge to true distribution."""
        delays = [1.0, 2.0, 3.0, 2.0, 1.5, 2.5]

        for delay in delays:
            link = nac.observe(
                event_type="tool",
                event_signature="timed_action",
                outcome_type="result",
                outcome_signature="success",
                outcome_valence=valence_positive,
                delta_seconds=delay,
            )

        expected_mean = sum(delays) / len(delays)
        actual_mean = link.temporal_delta.mean

        assert abs(actual_mean - expected_mean) < 0.1, (
            f"Expected mean ~{expected_mean}, got {actual_mean}"
        )

    def test_temporal_prediction_provides_bounds(self, nac, valence_positive):
        """Temporal predictions include confidence bounds."""
        delays = [1.0, 1.5, 2.0, 2.5, 3.0, 1.2, 1.8, 2.2]

        for delay in delays:
            link = nac.observe(
                event_type="tool",
                event_signature="variable_timing",
                outcome_type="result",
                outcome_signature="success",
                outcome_valence=valence_positive,
                delta_seconds=delay,
            )

        expected, lower, upper = link.temporal_delta.predict_delay()

        assert lower <= expected <= upper
        assert lower >= min(delays) * 0.8
        assert upper <= max(delays) * 1.2


@pytest.mark.learning
class TestNAcRecovery:
    """Test that NAc recovers from incorrect predictions."""

    def test_recovers_from_initial_bias(self, nac, valence_positive, valence_negative):
        """NAc recovers from initial positive bias when outcomes are negative."""
        # Start with positive expectations - use same outcome signature throughout
        for _ in range(10):
            nac.observe(
                event_type="tool",
                event_signature="changing",
                outcome_type="result",
                outcome_signature="outcome",  # Same signature for all
                outcome_valence=valence_positive,
                delta_seconds=1.0,
            )

        links = nac.get_links_for_event("changing")
        initial_prediction = links[0].predicted_value

        # Now consistently negative - same outcome signature updates the same link
        for _ in range(30):
            nac.observe(
                event_type="tool",
                event_signature="changing",
                outcome_type="result",
                outcome_signature="outcome",  # Same signature so same link is updated
                outcome_valence=valence_negative,
                delta_seconds=1.0,
            )

        final_prediction = links[0].predicted_value

        assert final_prediction < initial_prediction, "Should have decreased"
        assert final_prediction < 0.6, "Should have recovered toward low prediction"

    def test_adapts_to_regime_change(self, nac, valence_positive, valence_negative):
        """NAc adapts when outcome distribution changes."""
        # Phase 1: 80% positive
        for i in range(30):
            valence = valence_positive if i % 10 < 8 else valence_negative
            nac.observe(
                event_type="tool",
                event_signature="regime",
                outcome_type="result",
                outcome_signature="result",
                outcome_valence=valence,
                delta_seconds=1.0,
            )

        links = nac.get_links_for_event("regime")
        phase1_prediction = links[0].predicted_value

        # Phase 2: 20% positive (regime change)
        for i in range(30):
            valence = valence_positive if i % 10 < 2 else valence_negative
            nac.observe(
                event_type="tool",
                event_signature="regime",
                outcome_type="result",
                outcome_signature="result",
                outcome_valence=valence,
                delta_seconds=1.0,
            )

        phase2_prediction = links[0].predicted_value

        assert phase2_prediction < phase1_prediction, "Should adapt to new regime"