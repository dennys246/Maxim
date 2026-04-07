"""Unit tests for Reachy connection management."""

from __future__ import annotations

import time


class TestFailureState:
    """Test FailureState dataclass."""

    def test_default_values(self):
        """Default values are zero."""
        from maxim.conscience.connection import FailureState

        state = FailureState()

        assert state.count == 0
        assert state.last_ts == 0.0


class TestFailureTracker:
    """Test failure tracking across subsystems."""

    def test_record_failure_increments_count(self):
        """Recording failure increments count."""
        from maxim.conscience.connection import FailureTracker

        tracker = FailureTracker()
        tracker.record_failure("motor")

        assert tracker.motor.count == 1

    def test_record_failure_returns_false_below_threshold(self):
        """Returns False when below threshold."""
        from maxim.conscience.connection import FailureTracker

        tracker = FailureTracker(thresholds={"motor": 3})

        assert tracker.record_failure("motor") is False
        assert tracker.record_failure("motor") is False

    def test_record_failure_returns_true_at_threshold(self):
        """Returns True when threshold reached."""
        from maxim.conscience.connection import FailureTracker

        tracker = FailureTracker(thresholds={"motor": 3})

        tracker.record_failure("motor")
        tracker.record_failure("motor")
        result = tracker.record_failure("motor")

        assert result is True

    def test_window_resets_count(self):
        """Count resets after window expires."""
        from maxim.conscience.connection import FailureTracker

        tracker = FailureTracker(window_s=0.01)

        tracker.record_failure("motor")
        tracker.record_failure("motor")

        time.sleep(0.02)

        # Should reset, starting fresh
        assert tracker.motor.count == 2  # Still has old count
        tracker.record_failure("motor")  # This resets and adds 1
        assert tracker.motor.count == 1

    def test_reset_clears_all(self):
        """Reset clears all failure states."""
        from maxim.conscience.connection import FailureTracker

        tracker = FailureTracker()
        tracker.record_failure("motor")
        tracker.record_failure("video")
        tracker.record_failure("audio")

        tracker.reset()

        assert tracker.motor.count == 0
        assert tracker.video.count == 0
        assert tracker.audio.count == 0

    def test_reset_subsystem(self):
        """Can reset individual subsystem."""
        from maxim.conscience.connection import FailureTracker

        tracker = FailureTracker()
        tracker.record_failure("motor")
        tracker.record_failure("video")

        tracker.reset_subsystem("motor")

        assert tracker.motor.count == 0
        assert tracker.video.count == 1


class TestConnectionConfig:
    """Test connection configuration."""

    def test_default_values(self):
        """Default values are sensible."""
        from maxim.conscience.connection import ConnectionConfig

        config = ConnectionConfig()

        assert config.robot_name == "reachy_mini"
        assert config.timeout == 30.0
        assert config.media_backend == "default"
        assert config.reconnect_cooldown_s == 20.0

    def test_custom_values(self):
        """Can set custom values."""
        from maxim.conscience.connection import ConnectionConfig

        config = ConnectionConfig(
            robot_name="test_robot",
            timeout=60.0,
            motor_failure_threshold=5,
        )

        assert config.robot_name == "test_robot"
        assert config.timeout == 60.0
        assert config.motor_failure_threshold == 5
