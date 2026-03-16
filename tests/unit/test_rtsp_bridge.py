"""Unit tests for RTSPBridge."""

from __future__ import annotations

import subprocess
import threading
import time
from unittest.mock import Mock, patch, MagicMock

import numpy as np
import pytest

from maxim.tools.rtsp_bridge import RTSPBridge, RTSPBridgeConfig


@pytest.fixture
def config():
    return RTSPBridgeConfig(
        rtsp_url="rtsp://localhost:8554/test",
        fps=10,
        preset="ultrafast",
        tune="zerolatency",
    )


@pytest.fixture
def mock_maxim():
    m = Mock()
    m.log = Mock()
    m._live_stop_event = None  # standalone mode
    m._last_frame = None
    m._last_frame_ts = None
    m.mini = Mock()
    return m


@pytest.fixture
def bridge(mock_maxim, config):
    return RTSPBridge(mock_maxim, config)


class TestRTSPBridgeConfig:
    def test_defaults(self):
        cfg = RTSPBridgeConfig()
        assert cfg.rtsp_url == "rtsp://localhost:8554/reachy"
        assert cfg.fps == 20
        assert cfg.bitrate == "2M"
        assert cfg.auto_start_mediamtx is True
        assert cfg.mediamtx_path is None

    def test_custom(self, config):
        assert config.fps == 10
        assert config.rtsp_url == "rtsp://localhost:8554/test"


class TestRTSPBridgeProperties:
    def test_not_running_initially(self, bridge):
        assert bridge.is_running is False

    def test_is_running_when_active(self, bridge):
        bridge._running = True
        assert bridge.is_running is True

    def test_not_running_after_stop(self, bridge):
        bridge._running = True
        bridge._stop_event.set()
        assert bridge.is_running is False


class TestRTSPBridgeStop:
    def test_stop_sets_event(self, bridge):
        bridge.stop()
        assert bridge._stop_event.is_set()

    def test_stop_kills_ffmpeg(self, bridge):
        proc = Mock()
        proc.stdin = Mock()
        proc.wait = Mock()
        bridge._proc = proc

        bridge.stop()

        proc.stdin.close.assert_called()


class TestRTSPBridgeFrameGrabbing:
    def test_standalone_mode_uses_get_frame(self, bridge, mock_maxim):
        """When no live loop, grab from mini.media directly."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_maxim.mini.media.get_frame.return_value = frame

        result = bridge._grab_frame()
        assert result is frame

    def test_coexist_mode_uses_last_frame(self, bridge, mock_maxim):
        """When live loop running, read _last_frame."""
        mock_maxim._live_stop_event = Mock()
        mock_maxim._live_stop_event.is_set.return_value = False
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_maxim._last_frame = frame
        mock_maxim._last_frame_ts = 1.0

        result = bridge._grab_frame()
        assert result is frame

    def test_coexist_mode_dedup(self, bridge, mock_maxim):
        """Same timestamp means duplicate frame — return None."""
        mock_maxim._live_stop_event = Mock()
        mock_maxim._live_stop_event.is_set.return_value = False
        mock_maxim._last_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_maxim._last_frame_ts = 1.0

        bridge._grab_frame()  # first call sets _last_sent_ts
        result = bridge._grab_frame()  # second call with same ts
        assert result is None


class TestRTSPBridgeFFmpeg:
    @patch("subprocess.Popen")
    def test_start_ffmpeg_returns_process(self, mock_popen, bridge):
        proc = bridge._start_ffmpeg(640, 480)
        assert proc is not None
        mock_popen.assert_called_once()

    @patch("subprocess.Popen", side_effect=FileNotFoundError)
    def test_start_ffmpeg_missing_binary(self, mock_popen, bridge):
        proc = bridge._start_ffmpeg(640, 480)
        assert proc is None

    @patch("subprocess.Popen")
    def test_ffmpeg_command_structure(self, mock_popen, bridge, config):
        bridge._start_ffmpeg(640, 480)
        cmd = mock_popen.call_args[0][0]

        assert cmd[0] == "ffmpeg"
        assert "-f" in cmd
        assert "rawvideo" in cmd
        assert config.rtsp_url in cmd
        assert str(config.fps) in cmd


class TestMediaMTXAutoStart:
    def test_port_detection(self, bridge):
        """Extracts port from RTSP URL."""
        assert bridge._get_rtsp_port() == 8554

    def test_port_detection_custom(self, mock_maxim):
        cfg = RTSPBridgeConfig(rtsp_url="rtsp://host:9999/stream")
        b = RTSPBridge(mock_maxim, cfg)
        assert b._get_rtsp_port() == 9999

    @patch("maxim.tools.rtsp_bridge.socket.socket")
    def test_port_in_use_returns_true(self, mock_socket_cls, bridge):
        mock_sock = Mock()
        mock_socket_cls.return_value.__enter__ = Mock(return_value=mock_sock)
        mock_socket_cls.return_value.__exit__ = Mock(return_value=False)
        mock_sock.connect_ex.return_value = 0
        assert bridge._is_port_in_use(8554) is True

    @patch("maxim.tools.rtsp_bridge.socket.socket")
    def test_port_not_in_use_returns_false(self, mock_socket_cls, bridge):
        mock_sock = Mock()
        mock_socket_cls.return_value.__enter__ = Mock(return_value=mock_sock)
        mock_socket_cls.return_value.__exit__ = Mock(return_value=False)
        mock_sock.connect_ex.return_value = 1
        assert bridge._is_port_in_use(8554) is False

    @patch.object(RTSPBridge, "_is_port_in_use", return_value=True)
    def test_ensure_skips_if_already_running(self, mock_port, bridge):
        """If port is in use, don't start a new MediaMTX."""
        result = bridge._ensure_mediamtx()
        assert result is True
        assert bridge._mediamtx_proc is None

    @patch.object(RTSPBridge, "_is_port_in_use", return_value=False)
    @patch.object(RTSPBridge, "_find_mediamtx", return_value=None)
    def test_ensure_fails_if_binary_not_found(self, mock_find, mock_port, bridge):
        result = bridge._ensure_mediamtx()
        assert result is False

    @patch.object(RTSPBridge, "_is_port_in_use", return_value=False)
    def test_ensure_skips_if_auto_start_disabled(self, mock_port, mock_maxim):
        cfg = RTSPBridgeConfig(auto_start_mediamtx=False)
        b = RTSPBridge(mock_maxim, cfg)
        result = b._ensure_mediamtx()
        assert result is False

    def test_stop_mediamtx_terminates_process(self, bridge):
        proc = Mock()
        proc.pid = 12345
        proc.terminate = Mock()
        proc.wait = Mock()
        bridge._mediamtx_proc = proc

        bridge._stop_mediamtx()

        proc.terminate.assert_called()
        assert bridge._mediamtx_proc is None

    def test_stop_mediamtx_noop_if_not_started(self, bridge):
        bridge._stop_mediamtx()  # Should not raise

    def test_stop_calls_stop_mediamtx(self, bridge):
        """bridge.stop() should also stop MediaMTX."""
        proc = Mock()
        proc.pid = 12345
        proc.terminate = Mock()
        proc.wait = Mock()
        bridge._mediamtx_proc = proc

        bridge.stop()

        proc.terminate.assert_called()
        assert bridge._mediamtx_proc is None


class TestRTSPBridgeKillFFmpeg:
    def test_kill_ffmpeg_closes_stdin(self, bridge):
        proc = Mock()
        proc.stdin = Mock()
        proc.wait = Mock()
        bridge._proc = proc

        bridge._kill_ffmpeg()

        proc.stdin.close.assert_called()
        proc.wait.assert_called()
        assert bridge._proc is None

    def test_kill_ffmpeg_noop_when_no_proc(self, bridge):
        bridge._kill_ffmpeg()  # Should not raise

    def test_kill_ffmpeg_force_kills_on_timeout(self, bridge):
        proc = Mock()
        proc.stdin = Mock()
        proc.wait = Mock(side_effect=subprocess.TimeoutExpired(cmd="ffmpeg", timeout=5))
        bridge._proc = proc

        bridge._kill_ffmpeg()

        proc.kill.assert_called()
