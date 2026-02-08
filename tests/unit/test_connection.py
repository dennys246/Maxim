"""Unit tests for Reachy connection management."""

from __future__ import annotations

import threading
import time

import pytest


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


class TestReachyConnection:
    """Test ReachyConnection class."""

    def test_creation(self):
        """Can create connection manager."""
        from maxim.conscience.connection import ReachyConnection

        conn = ReachyConnection()

        assert conn.is_connected is False
        assert conn.mini is None

    def test_set_media_lock(self):
        """Can set media lock."""
        from maxim.conscience.connection import ReachyConnection

        conn = ReachyConnection()
        lock = threading.Lock()

        conn.set_media_lock(lock)

        assert conn._media_lock is lock

    def test_on_reconnect_callback(self):
        """Can register reconnect callbacks."""
        from maxim.conscience.connection import ReachyConnection

        conn = ReachyConnection()
        called = []

        conn.on_reconnect(lambda: called.append(True))

        assert len(conn._on_reconnect) == 1

    def test_note_failure_ignores_non_connection_errors(self):
        """Non-connection errors are ignored."""
        from maxim.conscience.connection import ReachyConnection

        conn = ReachyConnection()

        result = conn.note_failure("motor", ValueError("not a connection error"))

        assert result is False

    def test_note_failure_ignores_during_shutdown(self):
        """Failures during shutdown are ignored."""
        from maxim.conscience.connection import ReachyConnection

        conn = ReachyConnection()
        stop_event = threading.Event()
        stop_event.set()

        result = conn.note_failure(
            "motor",
            Exception("lost connection"),
            stop_event=stop_event,
        )

        assert result is False

    def test_mark_woke_up(self):
        """Can mark robot as woken up."""
        from maxim.conscience.connection import ReachyConnection

        conn = ReachyConnection()

        conn.mark_woke_up()
        assert conn._woke_up is True

        conn.mark_sleeping()
        assert conn._woke_up is False
