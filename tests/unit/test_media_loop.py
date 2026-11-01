"""Unit tests for MediaLoopMixin helper methods."""

from __future__ import annotations

import logging
import queue
from unittest.mock import MagicMock, patch

import pytest


from maxim.embodied_runtime.media_loop import MediaLoopMixin


class _TestMedia(MediaLoopMixin):
    """Minimal concrete instance of MediaLoopMixin for testing."""

    def __init__(self):
        self.log = logging.getLogger("test")
        self._motor_queue = None
        self.actions = {}
        self.mini = MagicMock()
        self.audio = False
        self.mode = "active"
        self.sleeping = False
        self._robot = None
        self.verbosity = 0

    def _note_connection_failure(self, subsystem, error):
        pass

    def _release_cv2(self):
        pass

    def _release_media(self):
        pass


# -- _enqueue_motor tests --------------------------------------------------


class TestEnqueueMotorWithoutQueue:
    """When _motor_queue is None, fn runs now and its result is the resolved ticket's outcome (#1499 5.6)."""

    def test_calls_fn_directly_and_returns_a_resolved_ticket(self):
        from maxim.hardware.controller import MotionOutcome

        media = _TestMedia()
        assert media._motor_queue is None
        calls = []

        def fn(x, y):
            calls.append((x, y))
            return MotionOutcome(ok=True, chunks_planned=1, chunks_done=1)

        ticket = media._enqueue_motor(fn, 3, 7)
        assert calls == [(3, 7)]
        assert ticket.completed.done() and ticket.completed.result().ok
        assert ticket.accepted.result(timeout=0) is None

    def test_passes_kwargs(self):
        media = _TestMedia()
        seen = {}

        def fn(a, *, b=0):
            seen["product"] = a * b
            return True

        assert media._enqueue_motor(fn, 5, b=4).completed.result(timeout=0).ok
        assert seen == {"product": 20}

    def test_a_raising_fn_resolves_the_ticket_instead_of_raising(self):
        media = _TestMedia()

        def fn():
            raise RuntimeError("boom")

        out = media._enqueue_motor(fn).completed.result(timeout=0)
        assert not out.ok and out.kind == "error" and "boom" in out.reason


class TestEnqueueMotorWithQueue:
    """When _motor_queue exists, the command is queued and its pending ticket returned."""

    def test_puts_command_on_queue_and_returns_its_ticket(self):
        from maxim.embodied_runtime.workers import MotorQueue

        media = _TestMedia()
        media._motor_queue = MotorQueue(maxsize=4)

        sentinel = MagicMock()
        ticket = media._enqueue_motor(sentinel, "a", key="val")

        assert not ticket.completed.done()
        sentinel.assert_not_called()

        item = media._motor_queue.get(timeout=0.1)
        assert item.ticket is ticket
        assert item.fn is sentinel
        assert item.args == ("a",)
        assert item.kwargs == {"key": "val"}

    def test_a_newer_command_supersedes_the_waiting_one(self):
        from maxim.embodied_runtime.workers import MotorQueue

        media = _TestMedia()
        media._motor_queue = MotorQueue(maxsize=4)

        old = media._enqueue_motor(MagicMock())
        new_fn = MagicMock()
        media._enqueue_motor(new_fn, 1)

        assert old.completed.result(timeout=0).kind == "superseded"
        item = media._motor_queue.get(timeout=0.1)
        assert item.fn is new_fn and item.args == (1,)
        with pytest.raises(queue.Empty):
            media._motor_queue.get(timeout=0.01)


# -- _clear_motor_queue tests -----------------------------------------------


class TestClearMotorQueueNoQueue:
    """When _motor_queue is None, returns 0."""

    def test_returns_zero(self):
        media = _TestMedia()
        assert media._motor_queue is None
        assert media._clear_motor_queue() == 0


class TestClearMotorQueueWithItems:
    """A waiting command is cleared and its ticket resolved superseded."""

    def test_returns_count_and_resolves_the_tickets(self):
        from maxim.embodied_runtime.workers import MotorQueue

        media = _TestMedia()
        media._motor_queue = MotorQueue(maxsize=8)
        ticket = media._enqueue_motor(MagicMock())

        assert media._clear_motor_queue() == 1
        assert ticket.completed.result(timeout=0).kind == "superseded"
        assert media._clear_motor_queue() == 0


class TestClearMotorQueueEmpty:
    """When queue exists but is empty, returns 0."""

    def test_returns_zero_for_empty_queue(self):
        from maxim.embodied_runtime.workers import MotorQueue

        media = _TestMedia()
        media._motor_queue = MotorQueue(maxsize=4)

        assert media._clear_motor_queue() == 0


# -- sleep tests ------------------------------------------------------------


class TestSleep:
    """sleep() sets state and delegates to live()."""

    def test_sets_audio_and_mode(self):
        media = _TestMedia()
        media.audio = False
        media.mode = "active"

        with patch.object(media, "live"):
            media.sleep("/tmp/test_home", parallel=False, run_id="test-run")

        assert media.audio is True
        assert media.mode == "sleep"

    def test_calls_goto_sleep_when_not_sleeping(self):
        media = _TestMedia()
        media.sleeping = False
        robot = MagicMock()
        media._robot = robot

        with patch.object(media, "live"):
            media.sleep("/tmp/test_home", parallel=True, run_id="r1")

        robot.goto_sleep.assert_called_once()
        assert media.sleeping is True

    def test_skips_goto_sleep_when_already_sleeping(self):
        media = _TestMedia()
        media.sleeping = True
        robot = MagicMock()
        media._robot = robot

        with patch.object(media, "live"):
            media.sleep("/tmp/test_home", parallel=True, run_id="r1")

        robot.goto_sleep.assert_not_called()

    def test_delegates_to_live_with_correct_args(self):
        media = _TestMedia()

        with patch.object(media, "live", return_value="live_result") as mock_live:
            result = media.sleep("/tmp/home", parallel=False, run_id="run-42")

        mock_live.assert_called_once_with(
            home_dir="/tmp/home",
            parallel=False,
            vision=False,
            motor=False,
            wake_up=False,
            run_id="run-42",
        )
        assert result == "live_result"

    def test_no_robot_does_not_raise(self):
        media = _TestMedia()
        media._robot = None
        media.sleeping = False

        with patch.object(media, "live"):
            media.sleep("/tmp/home", parallel=True, run_id="r")

        assert media.sleeping is True
