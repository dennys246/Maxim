"""Unit tests for MediaLoopMixin helper methods."""

from __future__ import annotations

import logging
import queue
from unittest.mock import MagicMock, patch


from maxim.conscience.media_loop import MediaLoopMixin


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
    """When _motor_queue is None, fn is called directly."""

    def test_calls_fn_directly_and_returns_result(self):
        media = _TestMedia()
        assert media._motor_queue is None

        result = media._enqueue_motor(lambda x, y: x + y, 3, 7)
        assert result == 10

    def test_passes_kwargs(self):
        media = _TestMedia()

        def fn(a, *, b=0):
            return a * b

        assert media._enqueue_motor(fn, 5, b=4) == 20


class TestEnqueueMotorWithQueue:
    """When _motor_queue exists, command is queued and None is returned."""

    def test_puts_command_on_queue_and_returns_none(self):
        media = _TestMedia()
        media._motor_queue = queue.Queue(maxsize=4)

        sentinel = MagicMock()
        result = media._enqueue_motor(sentinel, "a", key="val")

        assert result is None
        sentinel.assert_not_called()

        fn, args, kwargs = media._motor_queue.get_nowait()
        assert fn is sentinel
        assert args == ("a",)
        assert kwargs == {"key": "val"}

    def test_drains_one_item_before_putting(self):
        media = _TestMedia()
        media._motor_queue = queue.Queue(maxsize=4)

        # Pre-fill one item
        old_fn = MagicMock()
        media._motor_queue.put_nowait((old_fn, (), {}))

        new_fn = MagicMock()
        media._enqueue_motor(new_fn, 1)

        # The old item was drained; only the new item should remain
        fn, args, kwargs = media._motor_queue.get_nowait()
        assert fn is new_fn
        assert args == (1,)

        # Queue should now be empty
        assert media._motor_queue.empty()

    def test_full_queue_silently_drops(self):
        media = _TestMedia()
        media._motor_queue = queue.Queue(maxsize=1)

        # Fill queue so drain removes one, then put succeeds
        media._motor_queue.put_nowait(("dummy", (), {}))
        # Second put fills it again
        media._enqueue_motor(lambda: None)

        # Now the queue is full (maxsize=1).  A third enqueue should
        # drain one and put the new one, so it still works at maxsize=1.
        result = media._enqueue_motor(lambda: "ignored")
        assert result is None


# -- _clear_motor_queue tests -----------------------------------------------


class TestClearMotorQueueNoQueue:
    """When _motor_queue is None, returns 0."""

    def test_returns_zero(self):
        media = _TestMedia()
        assert media._motor_queue is None
        assert media._clear_motor_queue() == 0


class TestClearMotorQueueWithItems:
    """When queue has items, returns the count of cleared items."""

    def test_returns_count_of_cleared_items(self):
        media = _TestMedia()
        q = queue.Queue(maxsize=8)
        media._motor_queue = q

        for i in range(5):
            q.put_nowait((f"fn_{i}", (), {}))

        cleared = media._clear_motor_queue()
        assert cleared == 5
        assert q.empty()


class TestClearMotorQueueEmpty:
    """When queue exists but is empty, returns 0."""

    def test_returns_zero_for_empty_queue(self):
        media = _TestMedia()
        media._motor_queue = queue.Queue(maxsize=4)

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
