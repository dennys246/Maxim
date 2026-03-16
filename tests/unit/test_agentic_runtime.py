"""Unit tests for AgenticRuntimeMixin lifecycle methods."""

from __future__ import annotations

import logging
import threading
from unittest.mock import MagicMock, patch


from maxim.conscience.agentic_runtime import AgenticRuntimeMixin


class _TestRuntime(AgenticRuntimeMixin):
    """Minimal concrete class that satisfies AgenticRuntimeMixin dependencies."""

    def __init__(self):
        self.log = logging.getLogger("test.agentic_runtime")
        self._agentic_thread = None
        self._agentic_stop_event = None
        self._agentic_agent = None
        self._agentic_state = None
        self._capture_manager = None
        self._default_network = None
        self._fear_agent = None
        self._voice_agentic_enabled = False
        self._woke_up = False
        self.sleeping = False
        self.mode = "sleep"
        self.mini = None
        self.home_dir = "data"
        self._motor_queue = None
        self._state_manager = MagicMock()
        self._vision_event_logger = None
        self._vision_event_thread = None
        self._vision_event_stop_event = None

    def _enqueue_motor(self, fn, *args, **kwargs):
        fn(*args, **kwargs)

    def _start_vision_event_stream(self):
        pass

    def _stop_vision_event_stream(self, *, timeout=2.0):
        pass

    def _request_mode(self, mode):
        self.mode = mode


# ---------------------------------------------------------------------------
# _stop_agentic_runtime
# ---------------------------------------------------------------------------


class TestStopAgenticRuntime:
    """Tests for _stop_agentic_runtime."""

    def test_noop_when_nothing_running(self):
        """Calling stop when no runtime is active should not raise."""
        rt = _TestRuntime()
        rt._stop_agentic_runtime()

        assert rt._agentic_thread is None
        assert rt._agentic_stop_event is None
        assert rt._agentic_agent is None
        assert rt._agentic_state is None
        assert rt._default_network is None
        assert rt._capture_manager is None
        assert rt._fear_agent is None

    def test_clears_references(self):
        """After stop, all agentic references should be set to None."""
        rt = _TestRuntime()
        # Simulate a previously-running runtime by setting fake references.
        rt._agentic_thread = MagicMock(is_alive=MagicMock(return_value=False))
        rt._agentic_stop_event = threading.Event()
        rt._agentic_agent = MagicMock()
        rt._agentic_state = MagicMock()
        rt._fear_agent = MagicMock()
        rt._default_network = MagicMock()
        rt._capture_manager = MagicMock()

        rt._stop_agentic_runtime()

        assert rt._agentic_thread is None
        assert rt._agentic_stop_event is None
        assert rt._agentic_agent is None
        assert rt._agentic_state is None
        assert rt._default_network is None
        assert rt._capture_manager is None
        assert rt._fear_agent is None
        rt._state_manager.set_agent.assert_called_with(None)

    def test_sets_stop_event(self):
        """Stop should signal the stop event so the worker thread exits."""
        rt = _TestRuntime()
        ev = threading.Event()
        rt._agentic_stop_event = ev

        rt._stop_agentic_runtime()

        assert ev.is_set()

    def test_joins_agentic_thread(self):
        """Stop should join the agentic thread with a timeout."""
        rt = _TestRuntime()
        fake_thread = MagicMock(spec=threading.Thread)
        fake_thread.is_alive.return_value = False
        rt._agentic_thread = fake_thread
        rt._agentic_stop_event = threading.Event()

        rt._stop_agentic_runtime(timeout=5.0)

        # Thread was alive check happens, join is called
        assert rt._agentic_thread is None


# ---------------------------------------------------------------------------
# wake_up_agentic
# ---------------------------------------------------------------------------


class TestWakeUpAgentic:
    """Tests for wake_up_agentic."""

    def test_sets_voice_agentic_enabled(self):
        """wake_up_agentic should set _voice_agentic_enabled to True
        when the runtime thread starts successfully."""
        rt = _TestRuntime()
        # Simulate _start_agentic_runtime producing a live thread
        fake_thread = MagicMock(spec=threading.Thread)
        fake_thread.is_alive.return_value = True

        def fake_start():
            rt._agentic_thread = fake_thread

        with patch.object(rt, "_start_agentic_runtime", side_effect=fake_start):
            rt.wake_up_agentic()

        assert rt._voice_agentic_enabled is True

    def test_handles_no_mini(self):
        """When mini is None, wake_up_agentic should not crash."""
        rt = _TestRuntime()
        rt.mini = None

        with patch.object(rt, "_start_agentic_runtime"):
            rt.wake_up_agentic()

        # Should not have set _woke_up since mini is None
        assert rt._woke_up is False

    def test_wakes_robot_when_mini_present(self):
        """When mini is set, wake_up should call mini.wake_up via _enqueue_motor."""
        rt = _TestRuntime()
        rt.mini = MagicMock()

        with patch.object(rt, "_start_agentic_runtime"):
            rt.wake_up_agentic()

        rt.mini.wake_up.assert_called_once()
        assert rt._woke_up is True
        assert rt.sleeping is False

    def test_mode_switch_from_sleep(self):
        """When mode is 'sleep', wake_up should request 'exploration'."""
        rt = _TestRuntime()
        rt.mode = "sleep"

        with patch.object(rt, "_start_agentic_runtime"):
            rt.wake_up_agentic()

        assert rt.mode == "exploration"

    def test_no_mode_switch_when_not_sleeping(self):
        """When mode is not 'sleep', wake_up should not change mode."""
        rt = _TestRuntime()
        rt.mode = "active"

        with patch.object(rt, "_start_agentic_runtime"):
            rt.wake_up_agentic()

        assert rt.mode == "active"

    def test_disables_voice_if_runtime_fails(self):
        """If _start_agentic_runtime fails to produce a live thread,
        _voice_agentic_enabled should be reset to False."""
        rt = _TestRuntime()
        rt._agentic_thread = None

        with patch.object(rt, "_start_agentic_runtime"):
            # _start_agentic_runtime is mocked to do nothing,
            # so _agentic_thread stays None -> voice should be disabled.
            rt.wake_up_agentic()

        assert rt._voice_agentic_enabled is False


# ---------------------------------------------------------------------------
# _start_agentic_runtime (early-exit path)
# ---------------------------------------------------------------------------


class TestStartAgenticRuntime:
    """Tests for the early-exit guard in _start_agentic_runtime."""

    def test_skips_if_thread_already_alive(self):
        """If an agentic thread is already alive, _start_agentic_runtime
        should return immediately without creating a new one."""
        rt = _TestRuntime()
        existing_thread = MagicMock(spec=threading.Thread)
        existing_thread.is_alive.return_value = True
        rt._agentic_thread = existing_thread

        with patch("maxim.conscience.agentic_runtime.is_gpu_available") as mock_gpu:
            rt._start_agentic_runtime()
            # is_gpu_available should never be reached because of early return
            mock_gpu.assert_not_called()

        # Thread reference should still be the original
        assert rt._agentic_thread is existing_thread
