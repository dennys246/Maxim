"""Media-surface capability truth (roadmap_1_1_to_1_3.md free findings, 2026-08-13).

Two silent capability lies, one contract:

1. ``ReachyMiniController.connect()`` used to wrap the SDK's video/audio
   devices unconditionally, so ``get_audio_stream()`` returned a live-looking
   wrapper under ``media_backend="no_media"`` — defeating the stream-surface
   rule ``derive_media_capabilities`` documents ("a connected controller whose
   getter returns None has positively declared the device absent").

2. ``CaptureManager`` was constructed only when ``has_vision``, so an
   audio-only robot got no audio capture thread at all; and once constructed
   it always spawned the frame/segmentation workers, which against a missing
   camera poll the SDK's absent device at loop rate (the 2026-08-01
   capability-truth log-flood lesson).
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

from maxim.hardware.reachy.controller import ReachyMiniController
from maxim.runtime.capture import CaptureManager


@pytest.fixture(autouse=True)
def _no_daemon_status_io(monkeypatch):
    monkeypatch.setattr(ReachyMiniController, "_daemon_status", staticmethod(lambda *a, **k: None))
    monkeypatch.setattr(ReachyMiniController, "_installed_sdk_version", staticmethod(lambda: (1, 8)))


def _mock_sdk():
    mod = MagicMock()
    return patch.dict(sys.modules, {"reachy_mini": mod}), mod


def _connect(mod_configure=None) -> ReachyMiniController:
    c = ReachyMiniController(host="10.42.0.1", connection_mode="network")
    ctx, mod = _mock_sdk()
    with ctx, patch.object(ReachyMiniController, "_port_open", return_value=True):
        if mod_configure is not None:
            mod_configure(mod)
        assert c.connect(timeout=2.0) is True
    return c


class TestStreamWrapperGating:
    def test_no_media_backend_yields_none_streams(self):
        """camera=None / audio=None on the SDK media manager (the no_media
        shape) must gate BOTH wrappers to None — the stream surface is the
        capability contract."""

        def configure(mod):
            mini = mod.ReachyMini.return_value
            mini.media.camera = None
            mini.media.audio = None

        c = _connect(configure)
        assert c.get_video_stream() is None
        assert c.get_audio_stream() is None

    def test_present_devices_keep_wrappers(self):
        """Default mock media (camera/audio auto-attributes, non-None) keeps
        the legacy behavior: both wrappers constructed."""
        c = _connect()
        assert c.get_video_stream() is not None
        assert c.get_audio_stream() is not None

    def test_audio_only_gates_video_not_audio(self):
        def configure(mod):
            mod.ReachyMini.return_value.media.camera = None

        c = _connect(configure)
        assert c.get_video_stream() is None
        assert c.get_audio_stream() is not None


class TestCaptureManagerVisionGate:
    def test_audio_only_runs_audio_thread_without_frame_workers(self):
        maxim = MagicMock()
        maxim.audio = True
        cm = CaptureManager(maxim=maxim, has_vision=False)
        cm.start()
        try:
            assert cm._frame_thread is None
            assert cm._segmentation_thread is None
            assert cm._audio_thread is not None
        finally:
            cm.stop(timeout=0.5)

    def test_has_vision_false_forces_segmentation_off(self):
        cm = CaptureManager(maxim=MagicMock(), has_vision=False, enable_segmentation=True)
        assert cm._enable_segmentation is False

    def test_default_keeps_legacy_frame_thread(self):
        maxim = MagicMock()
        maxim.audio = False
        cm = CaptureManager(maxim=maxim, enable_segmentation=False)
        cm.start()
        try:
            assert cm._frame_thread is not None
            assert cm._audio_thread is None
        finally:
            cm.stop(timeout=0.5)
