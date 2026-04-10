"""Unit tests for VisionStreamMixin."""

from __future__ import annotations

import logging
import os
import threading
from unittest.mock import MagicMock


from maxim.embodied_runtime.vision_stream import VisionStreamMixin


class _TestVision(VisionStreamMixin):
    def __init__(self, tmp_path):
        self.log = logging.getLogger("test")
        self.home_dir = str(tmp_path)
        self._vision_event_logger = None
        self._vision_event_thread = None
        self._vision_event_stop_event = None
        self._vision_event_last_frame_ts = None
        self.vision_events_path = ""
        self.run_id = "test-run"
        self._last_frame = None
        self._last_frame_ts = None
        self.segmenter = None
        self._observation_lock = threading.Lock()
        self._segmenter_model = None
        self.interests = []
        self._threads = {}

    def _ensure_segmenter(self):
        pass

    def register_thread(self, name, thread):
        self._threads[name] = thread


# ---------------------------------------------------------------------------
# _ensure_vision_logger
# ---------------------------------------------------------------------------


class TestEnsureVisionLogger:
    def test_returns_none_initially_then_creates_logger(self, tmp_path, monkeypatch):
        """Logger is None before first call; first call creates one."""
        obj = _TestVision(tmp_path)
        assert obj._vision_event_logger is None

        mock_logger = MagicMock()
        mock_cls = MagicMock(return_value=mock_logger)
        monkeypatch.setattr("maxim.embodied_runtime.vision_stream.VisionEventLogger", mock_cls)

        result = obj._ensure_vision_logger()

        assert result is mock_logger
        assert obj._vision_event_logger is mock_logger
        mock_cls.assert_called_once()
        mock_logger.start.assert_called_once()

    def test_sets_vision_events_path_when_empty(self, tmp_path, monkeypatch):
        """When vision_events_path is empty, it gets set from home_dir + run_id."""
        obj = _TestVision(tmp_path)
        assert obj.vision_events_path == ""

        mock_logger = MagicMock()
        monkeypatch.setattr(
            "maxim.embodied_runtime.vision_stream.VisionEventLogger",
            MagicMock(return_value=mock_logger),
        )

        obj._ensure_vision_logger()

        expected = os.path.join(str(tmp_path), "vision", "vision_events_test-run.jsonl")
        assert obj.vision_events_path == expected

    def test_returns_cached_logger_on_second_call(self, tmp_path, monkeypatch):
        """Second call returns the cached logger without creating a new one."""
        obj = _TestVision(tmp_path)

        mock_logger = MagicMock()
        mock_cls = MagicMock(return_value=mock_logger)
        monkeypatch.setattr("maxim.embodied_runtime.vision_stream.VisionEventLogger", mock_cls)

        first = obj._ensure_vision_logger()
        second = obj._ensure_vision_logger()

        assert first is second
        assert mock_cls.call_count == 1

    def test_returns_none_on_logger_creation_failure(self, tmp_path, monkeypatch):
        """Returns None when VisionEventLogger raises."""
        obj = _TestVision(tmp_path)

        monkeypatch.setattr(
            "maxim.embodied_runtime.vision_stream.VisionEventLogger",
            MagicMock(side_effect=RuntimeError("boom")),
        )

        result = obj._ensure_vision_logger()

        assert result is None
        assert obj._vision_event_logger is None


# ---------------------------------------------------------------------------
# _stop_vision_event_stream
# ---------------------------------------------------------------------------


class TestStopVisionEventStream:
    def test_sets_stop_event(self, tmp_path):
        """Stop event is set so the worker thread exits."""
        obj = _TestVision(tmp_path)
        stop_event = threading.Event()
        obj._vision_event_stop_event = stop_event

        mock_thread = MagicMock()
        obj._vision_event_thread = mock_thread

        mock_logger = MagicMock()
        obj._vision_event_logger = mock_logger

        obj._stop_vision_event_stream()

        assert stop_event.is_set()

    def test_clears_thread_and_logger_references(self, tmp_path):
        """After stop, thread, stop event, and logger are set to None."""
        obj = _TestVision(tmp_path)
        obj._vision_event_stop_event = threading.Event()
        obj._vision_event_thread = MagicMock()
        obj._vision_event_logger = MagicMock()

        obj._stop_vision_event_stream()

        assert obj._vision_event_thread is None
        assert obj._vision_event_stop_event is None
        assert obj._vision_event_logger is None

    def test_calls_logger_stop(self, tmp_path):
        """Logger.stop() is called with the timeout."""
        obj = _TestVision(tmp_path)
        mock_logger = MagicMock()
        obj._vision_event_logger = mock_logger

        obj._stop_vision_event_stream(timeout=3.0)

        mock_logger.stop.assert_called_once_with(timeout=3.0)

    def test_noop_when_no_stream_running(self, tmp_path):
        """No error when called with nothing running."""
        obj = _TestVision(tmp_path)
        assert obj._vision_event_thread is None
        assert obj._vision_event_stop_event is None
        assert obj._vision_event_logger is None

        # Should not raise
        obj._stop_vision_event_stream()

        assert obj._vision_event_thread is None
        assert obj._vision_event_stop_event is None
        assert obj._vision_event_logger is None


# ---------------------------------------------------------------------------
# _start_vision_event_stream
# ---------------------------------------------------------------------------


class TestStartVisionEventStream:
    def test_does_not_start_if_logger_creation_fails(self, tmp_path, monkeypatch):
        """When _ensure_vision_logger returns None, no thread is created."""
        obj = _TestVision(tmp_path)

        monkeypatch.setattr(
            "maxim.embodied_runtime.vision_stream.VisionEventLogger",
            MagicMock(side_effect=RuntimeError("boom")),
        )

        obj._start_vision_event_stream()

        assert obj._vision_event_thread is None
        assert "maxim.vision.events" not in obj._threads

    def test_does_not_start_duplicate_thread(self, tmp_path, monkeypatch):
        """If a thread is already alive, a second call is a no-op."""
        obj = _TestVision(tmp_path)

        alive_thread = MagicMock()
        alive_thread.is_alive.return_value = True
        obj._vision_event_thread = alive_thread

        mock_cls = MagicMock()
        monkeypatch.setattr("maxim.embodied_runtime.vision_stream.VisionEventLogger", mock_cls)

        obj._start_vision_event_stream()

        # Logger constructor should not have been called
        mock_cls.assert_not_called()
        # Thread reference unchanged
        assert obj._vision_event_thread is alive_thread

    def test_creates_and_registers_thread(self, tmp_path, monkeypatch):
        """A daemon thread is created, registered, and started."""
        obj = _TestVision(tmp_path)

        mock_logger = MagicMock()
        monkeypatch.setattr(
            "maxim.embodied_runtime.vision_stream.VisionEventLogger",
            MagicMock(return_value=mock_logger),
        )

        obj._start_vision_event_stream()

        assert obj._vision_event_thread is not None
        assert obj._vision_event_thread.daemon is True
        assert obj._vision_event_thread.name == "maxim.vision.events"
        assert obj._vision_event_thread.is_alive()
        assert "maxim.vision.events" in obj._threads
        assert obj._vision_event_stop_event is not None

        # Clean up
        obj._stop_vision_event_stream(timeout=2.0)
