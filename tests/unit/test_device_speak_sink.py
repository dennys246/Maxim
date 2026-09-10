"""Tests for the device speak sink (utils.audio) — the single Reachy-speaker path.

The load-bearing property: the SDK's ``push_audio_sample`` is typed
``NDArray[float32]`` and does no conversion, while Piper returns int16. So the sink
MUST hand the device float32 in [-1, 1], not raw int16 (which reinterprets bytes as
noise at half duration). These tests pin that conversion for both the pairing path and
embodied speech, since both now go through one implementation.
"""

from __future__ import annotations

import numpy as np

from maxim.console import make_pairing_announcer
from maxim.utils.audio import (
    _to_device_float32,
    make_device_speak_sink,
    push_audio_to_device,
)


class _FakeMedia:
    def __init__(self) -> None:
        self.pushed: list[np.ndarray] = []

    def push_audio_sample(self, data: np.ndarray) -> None:
        self.pushed.append(data)


class _FakeMini:
    def __init__(self) -> None:
        self.media = _FakeMedia()


def test_to_device_float32_scales_int16_into_unit_range():
    out = _to_device_float32(np.array([0, 32767, -32767], dtype=np.int16))
    assert out.dtype == np.float32
    np.testing.assert_allclose(out, [0.0, 1.0, -1.0], atol=1e-4)


def test_to_device_float32_passes_float_through():
    out = _to_device_float32(np.array([0.5, -0.5], dtype=np.float32))
    assert out.dtype == np.float32
    np.testing.assert_allclose(out, [0.5, -0.5])


def test_device_receives_float32_not_int16():
    """The bug fix: int16 in, float32 out to the SDK."""
    mini = _FakeMini()
    ok = push_audio_to_device(mini, np.array([16384, -16384], dtype=np.int16))
    assert ok is True
    assert len(mini.media.pushed) == 1
    pushed = mini.media.pushed[0]
    assert pushed.dtype == np.float32, "SDK must receive float32, never int16 (silent-noise bug)"
    np.testing.assert_allclose(pushed, [0.5, -0.5], atol=1e-4)


def test_empty_or_none_returns_false_without_touching_device():
    mini = _FakeMini()
    assert push_audio_to_device(mini, None) is False
    assert push_audio_to_device(mini, np.array([], dtype=np.int16)) is False
    assert mini.media.pushed == []


def test_webrtc_error_falls_back_to_local(monkeypatch):
    calls = {}

    def _fake_local(samples, sample_rate=16000, blocking=False):
        calls["local"] = (samples, sample_rate)
        return True

    monkeypatch.setattr("maxim.utils.audio.play_audio_local", _fake_local)

    class _WebRTCMedia:
        def push_audio_sample(self, data):
            raise RuntimeError("Not implemented for WebRTC")

    class _WebRTCMini:
        media = _WebRTCMedia()

    ok = push_audio_to_device(_WebRTCMini(), np.array([1, 2], dtype=np.int16), sample_rate=22050)
    assert ok is True
    assert calls["local"][1] == 22050  # local fallback gets the sample_rate


def test_non_webrtc_error_calls_on_failure_then_falls_back(monkeypatch):
    monkeypatch.setattr("maxim.utils.audio.play_audio_local", lambda *a, **k: False)
    seen = []

    class _BadMedia:
        def push_audio_sample(self, data):
            raise RuntimeError("speaker on fire")

    class _BadMini:
        media = _BadMedia()

    ok = push_audio_to_device(_BadMini(), np.array([1], dtype=np.int16), on_failure=lambda e: seen.append(e))
    assert ok is False
    assert len(seen) == 1 and isinstance(seen[0], RuntimeError)


def test_prefer_local_env_skips_device(monkeypatch):
    monkeypatch.setenv("MAXIM_TTS_LOCAL", "1")
    monkeypatch.setattr("maxim.utils.audio.play_audio_local", lambda *a, **k: True)
    mini = _FakeMini()
    ok = push_audio_to_device(mini, np.array([1, 2], dtype=np.int16))
    assert ok is True
    assert mini.media.pushed == [], "prefer-local must not touch the device"


def test_make_device_speak_sink_is_inert_until_called():
    """Construction closes over mini only — no device access at build time."""
    touched = []

    class _WatchMini:
        @property
        def media(self):
            touched.append("media-accessed")
            return _FakeMedia()

    sink = make_device_speak_sink(_WatchMini())
    assert touched == [], "building the sink must not touch the audio device"
    sink(np.array([1], dtype=np.int16))
    assert touched == ["media-accessed"]


def test_pairing_announcer_over_device_sink_hands_float32_to_the_device():
    """End-to-end (A9.1): make_pairing_announcer + device sink -> SDK gets float32."""

    class _Int16TTS:
        sample_rate = 22050

        def synthesize(self, text: str):
            # Piper-shaped output: int16.
            return np.array([32767, 0, -32767], dtype=np.int16)

    mini = _FakeMini()
    announce = make_pairing_announcer(_Int16TTS(), make_device_speak_sink(mini))
    announce("424242")

    assert len(mini.media.pushed) == 1
    assert mini.media.pushed[0].dtype == np.float32
    np.testing.assert_allclose(mini.media.pushed[0], [1.0, 0.0, -1.0], atol=1e-4)
