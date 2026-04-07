"""Learning tests for FocusLearner - Verify Rescorla-Wagner gain adaptation.

These tests verify that the FocusLearner:
1. Converges to correct gain values
2. Adapts to overshoot/undershoot patterns
3. Detects and handles oscillation
4. Persists learned gains correctly
"""

from __future__ import annotations

import pytest


@pytest.mark.learning
class TestFocusLearnerConvergence:
    """Test that FocusLearner gains converge correctly."""

    def test_consistent_overshoot_reduces_gain(self, focus_learner):
        """Consistent overshoot reduces movement gain."""
        initial_gain = focus_learner.get_average_gain()

        # Simulate consistent overshoot: we command 100px, target ends up past center
        for i in range(20):
            focus_learner.record_intent(du=100.0, dv=0.0)

            # Wait for min delay
            import time

            time.sleep(0.01)

            # Target overshot - ended up at -20 (past center by 20px)
            # When we commanded 100px toward center, we went too far
            focus_learner.record_result(target_u=300.0, target_v=240.0)  # Error = -20

        final_gain = focus_learner.get_average_gain()

        # Gain should have decreased to compensate for overshoot
        assert final_gain < initial_gain, f"Expected gain decrease: {initial_gain} -> {final_gain}"

    def test_consistent_undershoot_increases_gain(self, focus_learner):
        """Consistent undershoot increases movement gain."""
        initial_gain = focus_learner.get_average_gain()

        # Simulate consistent undershoot: we command movement but don't reach center
        for i in range(20):
            focus_learner.record_intent(du=100.0, dv=0.0)

            import time

            time.sleep(0.01)

            # Target undershot - still 50px away from center
            focus_learner.record_result(target_u=370.0, target_v=240.0)  # Error = +50

        final_gain = focus_learner.get_average_gain()

        # Gain should have increased to compensate for undershoot
        assert final_gain > initial_gain, f"Expected gain increase: {initial_gain} -> {final_gain}"

    def test_perfect_targeting_stabilizes_gain(self, focus_learner):
        """Perfect targeting keeps gain stable."""
        # First, let it learn a bit
        for _ in range(5):
            focus_learner.record_intent(du=100.0, dv=0.0)
            import time

            time.sleep(0.01)
            focus_learner.record_result(target_u=320.0, target_v=240.0)  # Perfect

        initial_gain = focus_learner.get_average_gain()

        # More perfect results
        for _ in range(10):
            focus_learner.record_intent(du=100.0, dv=0.0)
            import time

            time.sleep(0.01)
            focus_learner.record_result(target_u=320.0, target_v=240.0)

        final_gain = focus_learner.get_average_gain()

        # Gain should be very stable
        assert abs(final_gain - initial_gain) < 0.1, f"Gain unstable: {initial_gain} -> {final_gain}"


@pytest.mark.learning
class TestFocusLearnerDirectionAwareness:
    """Test that FocusLearner learns direction-specific gains."""

    def test_learns_asymmetric_gains(self, focus_learner):
        """Different directions can have different optimal gains."""
        # Positive direction: consistent overshoot -> reduce gain
        for _ in range(15):
            focus_learner.record_intent(du=100.0, dv=0.0)
            import time

            time.sleep(0.01)
            # Overshoot in positive direction
            focus_learner.record_result(target_u=280.0, target_v=240.0)

        # Negative direction: consistent undershoot -> increase gain
        for _ in range(15):
            focus_learner.record_intent(du=-100.0, dv=0.0)
            import time

            time.sleep(0.01)
            # Undershoot in negative direction
            focus_learner.record_result(target_u=360.0, target_v=240.0)

        # Check that direction-specific gains differ
        gain_h_pos, _ = focus_learner.get_gain(du=100.0, dv=0.0)
        gain_h_neg, _ = focus_learner.get_gain(du=-100.0, dv=0.0)

        assert gain_h_neg > gain_h_pos, f"Expected asymmetric gains: pos={gain_h_pos}, neg={gain_h_neg}"


@pytest.mark.learning
class TestFocusLearnerOscillationHandling:
    """Test oscillation detection and damping."""

    def test_detects_oscillation(self, focus_learner):
        """Detects rapid direction changes as oscillation."""
        initial_oscillation_count = focus_learner._oscillation_count

        # Simulate oscillation: rapid left-right-left-right
        directions = [100.0, -100.0, 100.0, -100.0, 100.0, -100.0, 100.0, -100.0]

        for du in directions:
            focus_learner.record_intent(du=du, dv=0.0)
            import time

            time.sleep(0.01)
            focus_learner.record_result(target_u=340.0, target_v=240.0)

        # Should have detected oscillation
        assert focus_learner._oscillation_count > initial_oscillation_count

    def test_oscillation_reduces_gains(self, focus_learner):
        """Oscillation detection reduces all gains."""
        # Get initial gains
        gain_before, _ = focus_learner.get_gain(100.0, 0.0)

        # Trigger oscillation
        for _ in range(3):
            for du in [100.0, -100.0]:
                focus_learner.record_intent(du=du, dv=0.0)
                import time

                time.sleep(0.01)
                focus_learner.record_result(target_u=340.0, target_v=240.0)

        gain_after, _ = focus_learner.get_gain(100.0, 0.0)

        # Gains should have been reduced due to oscillation
        assert gain_after <= gain_before, f"Expected reduction: {gain_before} -> {gain_after}"


@pytest.mark.learning
class TestFocusLearnerBounds:
    """Test that gains stay within configured bounds."""

    def test_gain_respects_minimum(self, focus_learner):
        """Gain never goes below min_gain."""
        min_gain = focus_learner.config.min_gain

        # Extreme overshoot to try to reduce gain below minimum
        for _ in range(50):
            focus_learner.record_intent(du=100.0, dv=0.0)
            import time

            time.sleep(0.01)
            # Massive overshoot
            focus_learner.record_result(target_u=100.0, target_v=240.0)

        gain_h, gain_v = focus_learner.get_gain(100.0, 100.0)

        assert gain_h >= min_gain, f"Horizontal gain {gain_h} below min {min_gain}"
        assert gain_v >= min_gain, f"Vertical gain {gain_v} below min {min_gain}"

    def test_gain_respects_maximum(self, focus_learner):
        """Gain never goes above max_gain."""
        max_gain = focus_learner.config.max_gain

        # Extreme undershoot to try to increase gain above maximum
        for _ in range(50):
            focus_learner.record_intent(du=100.0, dv=0.0)
            import time

            time.sleep(0.01)
            # Massive undershoot
            focus_learner.record_result(target_u=400.0, target_v=300.0)

        gain_h, gain_v = focus_learner.get_gain(100.0, 100.0)

        assert gain_h <= max_gain, f"Horizontal gain {gain_h} above max {max_gain}"
        assert gain_v <= max_gain, f"Vertical gain {gain_v} above max {max_gain}"


@pytest.mark.learning
class TestFocusLearnerPersistence:
    """Test that learned gains persist correctly."""

    def test_save_load_preserves_gains(self, focus_learner, tmp_path):
        """Gains survive save/load cycle."""
        # Learn some gains
        for _ in range(10):
            focus_learner.record_intent(du=100.0, dv=50.0)
            import time

            time.sleep(0.01)
            focus_learner.record_result(target_u=330.0, target_v=250.0)

        original_gain = focus_learner.get_average_gain()

        # Save
        path = tmp_path / "focus.json"
        focus_learner.save(str(path))

        # Create new instance and load
        from maxim.proprioception.focus_learner import FocusLearner, FocusLearnerConfig

        new_learner = FocusLearner(config=FocusLearnerConfig(persist_path=None, auto_save_interval=0))
        new_learner.load(str(path))

        loaded_gain = new_learner.get_average_gain()

        assert abs(loaded_gain - original_gain) < 0.001, f"Gain mismatch: {original_gain} vs {loaded_gain}"

    def test_save_load_preserves_stats(self, focus_learner, tmp_path):
        """Statistics survive save/load cycle."""
        # Generate some samples
        for _ in range(10):
            focus_learner.record_intent(du=100.0, dv=0.0)
            import time

            time.sleep(0.01)
            focus_learner.record_result(target_u=320.0, target_v=240.0)

        original_stats = focus_learner.get_stats()

        path = tmp_path / "focus.json"
        focus_learner.save(str(path))

        from maxim.proprioception.focus_learner import FocusLearner, FocusLearnerConfig

        new_learner = FocusLearner(config=FocusLearnerConfig(persist_path=None, auto_save_interval=0))
        new_learner.load(str(path))

        loaded_stats = new_learner.get_stats()

        assert loaded_stats["total_samples"] == original_stats["total_samples"]
        assert loaded_stats["successful_focuses"] == original_stats["successful_focuses"]


@pytest.mark.learning
class TestFocusLearnerReset:
    """Test reset functionality."""

    def test_reset_restores_initial_gains(self, focus_learner):
        """Reset restores initial gain values."""
        initial = focus_learner.config.initial_gain

        # Learn different gains
        for _ in range(20):
            focus_learner.record_intent(du=100.0, dv=0.0)
            import time

            time.sleep(0.01)
            focus_learner.record_result(target_u=280.0, target_v=240.0)

        # Should have changed
        pre_reset = focus_learner.get_average_gain()
        assert pre_reset != initial

        # Reset
        focus_learner.reset()

        # Should be back to initial
        post_reset = focus_learner.get_average_gain()
        assert post_reset == initial

    def test_reset_clears_stats(self, focus_learner):
        """Reset clears all statistics."""
        # Generate samples
        for _ in range(5):
            focus_learner.record_intent(du=100.0, dv=0.0)
            import time

            time.sleep(0.01)
            focus_learner.record_result(target_u=320.0, target_v=240.0)

        assert focus_learner._total_samples > 0

        focus_learner.reset()

        assert focus_learner._total_samples == 0
        assert focus_learner._successful_focuses == 0
        assert focus_learner._oscillation_count == 0
