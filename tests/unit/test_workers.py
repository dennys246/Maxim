"""Unit tests for maxim.conscience.workers."""

from __future__ import annotations

import logging
import queue
import threading
import time
from unittest.mock import MagicMock, PropertyMock

import pytest

from maxim.conscience.workers import (
    IK_FAILURE_THRESHOLD,
    IKWarningHandler,
    frame_capture_worker,
    motor_worker,
)


# ---------------------------------------------------------------------------
# IKWarningHandler tests
# ---------------------------------------------------------------------------


class TestIKWarningHandler:
    """Tests for the IKWarningHandler logging handler."""

    def _make_record(self, msg: str, level: int = logging.WARNING) -> logging.LogRecord:
        return logging.LogRecord(
            name="test",
            level=level,
            pathname="",
            lineno=0,
            msg=msg,
            args=(),
            exc_info=None,
        )

    def test_initial_state(self):
        handler = IKWarningHandler()
        assert handler.ik_failure_detected is False
        assert handler.failure_message is None

    def test_detects_ik_error(self):
        handler = IKWarningHandler()
        handler.emit(self._make_record("Some IK error occurred in joint"))
        assert handler.ik_failure_detected is True
        assert "IK error" in handler.failure_message

    def test_detects_pose_not_achievable(self):
        handler = IKWarningHandler()
        handler.emit(self._make_record("Pose not achievable for target"))
        assert handler.ik_failure_detected is True
        assert "Pose not achievable" in handler.failure_message

    def test_detects_collision_detected(self):
        handler = IKWarningHandler()
        handler.emit(self._make_record("Collision detected during movement"))
        assert handler.ik_failure_detected is True
        assert "Collision detected" in handler.failure_message

    def test_ignores_unrelated_warnings(self):
        handler = IKWarningHandler()
        handler.emit(self._make_record("Motor voltage low"))
        assert handler.ik_failure_detected is False
        assert handler.failure_message is None

    def test_ignores_unrelated_then_detects(self):
        handler = IKWarningHandler()
        handler.emit(self._make_record("All good"))
        assert handler.ik_failure_detected is False
        handler.emit(self._make_record("ik error on servo"))
        assert handler.ik_failure_detected is True

    def test_case_insensitive_detection(self):
        handler = IKWarningHandler()
        handler.emit(self._make_record("IK ERROR uppercase"))
        assert handler.ik_failure_detected is True

    def test_failure_message_preserves_original_case(self):
        handler = IKWarningHandler()
        original = "IK Error On Joint 3"
        handler.emit(self._make_record(original))
        assert handler.failure_message == original


# ---------------------------------------------------------------------------
# IK_FAILURE_THRESHOLD constant
# ---------------------------------------------------------------------------


def test_ik_failure_threshold_equals_3():
    assert IK_FAILURE_THRESHOLD == 3


# ---------------------------------------------------------------------------
# motor_worker tests
# ---------------------------------------------------------------------------


def _make_mock_maxim():
    """Create a mock maxim object with the attributes motor_worker expects."""
    maxim = MagicMock()
    maxim._default_network = None
    maxim.yaw = 0.0
    maxim.pitch = 0.0
    maxim.y = 0.0
    maxim.z = 0.0
    maxim.roll = 0.0
    maxim.x = 0.0
    return maxim


class TestMotorWorker:
    """Tests for the motor_worker function."""

    def test_executes_queued_command(self):
        """motor_worker should call the function pulled from the queue."""
        mq = queue.Queue()
        stop = threading.Event()
        maxim = _make_mock_maxim()

        called = threading.Event()
        fn = MagicMock(side_effect=lambda: called.set())

        mq.put((fn, (), {}))

        t = threading.Thread(target=motor_worker, args=(mq, stop, maxim), daemon=True)
        t.start()

        assert called.wait(timeout=2.0), "Queued function was not called"
        stop.set()
        t.join(timeout=2.0)

        fn.assert_called_once()

    def test_stops_when_event_set(self):
        """motor_worker should exit once stop_event is set."""
        mq = queue.Queue()
        stop = threading.Event()
        maxim = _make_mock_maxim()

        stop.set()  # Set before starting
        t = threading.Thread(target=motor_worker, args=(mq, stop, maxim), daemon=True)
        t.start()
        t.join(timeout=2.0)
        assert not t.is_alive(), "motor_worker did not exit after stop_event was set"

    def test_handles_command_exception(self):
        """motor_worker should call _note_connection_failure on exception."""
        mq = queue.Queue()
        stop = threading.Event()
        maxim = _make_mock_maxim()

        error = RuntimeError("servo fault")
        fn = MagicMock(side_effect=error)

        mq.put((fn, (), {}))

        t = threading.Thread(target=motor_worker, args=(mq, stop, maxim), daemon=True)
        t.start()

        # Give it time to process the item
        time.sleep(0.5)
        stop.set()
        t.join(timeout=2.0)

        maxim._note_connection_failure.assert_called_once_with("motor", error)

    def test_executes_multiple_commands_in_order(self):
        """motor_worker should process commands sequentially."""
        mq = queue.Queue()
        stop = threading.Event()
        maxim = _make_mock_maxim()

        results = []
        fn1 = MagicMock(side_effect=lambda: results.append(1))
        fn2 = MagicMock(side_effect=lambda: results.append(2))

        mq.put((fn1, (), {}))
        mq.put((fn2, (), {}))

        t = threading.Thread(target=motor_worker, args=(mq, stop, maxim), daemon=True)
        t.start()

        # Wait for both items to be processed
        deadline = time.time() + 2.0
        while len(results) < 2 and time.time() < deadline:
            time.sleep(0.05)

        stop.set()
        t.join(timeout=2.0)

        assert results == [1, 2]

    def test_ik_failure_detection_via_log(self):
        """motor_worker should detect IK failures emitted as log warnings."""
        mq = queue.Queue()
        stop = threading.Event()
        maxim = _make_mock_maxim()

        def emit_ik_warning():
            logger = logging.getLogger("reachy_mini")
            logger.warning("IK error on joint 5")

        mq.put((emit_ik_warning, (), {}))

        t = threading.Thread(target=motor_worker, args=(mq, stop, maxim), daemon=True)
        t.start()

        # Give it time to process
        time.sleep(0.5)
        stop.set()
        t.join(timeout=2.0)

        # The worker should have logged a warning about the IK failure
        maxim.log.warning.assert_called()
        call_args = maxim.log.warning.call_args_list
        assert any("IK failure detected" in str(c) for c in call_args)

    def test_passes_args_and_kwargs(self):
        """motor_worker should forward args and kwargs to the command function."""
        mq = queue.Queue()
        stop = threading.Event()
        maxim = _make_mock_maxim()

        fn = MagicMock()
        mq.put((fn, (1, "two"), {"three": 3}))

        t = threading.Thread(target=motor_worker, args=(mq, stop, maxim), daemon=True)
        t.start()

        time.sleep(0.5)
        stop.set()
        t.join(timeout=2.0)

        fn.assert_called_once_with(1, "two", three=3)


# ---------------------------------------------------------------------------
# frame_capture_worker tests
# ---------------------------------------------------------------------------


class TestFrameCaptureWorker:
    """Tests for the frame_capture_worker function."""

    def test_stops_when_event_set(self):
        """frame_capture_worker should exit once stop_event is set."""
        stop = threading.Event()
        media_lock = threading.Lock()
        maxim = MagicMock()
        maxim.video_fps = 20.0

        # get_frame returns None so the worker just loops
        maxim.mini.media.get_frame.return_value = None

        frame_save_q = queue.Queue()
        frame_obs_q = queue.Queue()

        # Set stop after a brief delay so the worker runs at least one iteration
        def set_stop():
            time.sleep(0.15)
            stop.set()

        threading.Thread(target=set_stop, daemon=True).start()

        t = threading.Thread(
            target=frame_capture_worker,
            args=(stop, media_lock, maxim, frame_save_q, frame_obs_q),
            daemon=True,
        )
        t.start()
        t.join(timeout=3.0)
        assert not t.is_alive(), "frame_capture_worker did not exit after stop_event was set"

    def test_exits_cleanly_with_none_frames(self):
        """frame_capture_worker should not crash when get_frame returns None."""
        stop = threading.Event()
        media_lock = threading.Lock()
        maxim = MagicMock()
        maxim.video_fps = 20.0
        maxim.mini.media.get_frame.return_value = None

        frame_save_q = queue.Queue()
        frame_obs_q = queue.Queue()

        stop.set()  # Immediate stop

        t = threading.Thread(
            target=frame_capture_worker,
            args=(stop, media_lock, maxim, frame_save_q, frame_obs_q),
            daemon=True,
        )
        t.start()
        t.join(timeout=2.0)

        assert not t.is_alive()
        assert frame_save_q.empty()
