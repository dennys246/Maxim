"""The Reachy speaker sink tells the truth (#1435).

Found by maxim-pulse on a real Reachy Mini: ``push_audio_to_device`` returned True while the SDK silently dropped
every sample (nothing in maxim called ``start_playing()``), and it never resampled Piper's 22,050 Hz to the device's
16,000 Hz. The design and the owner decisions: docs/plans/speaker_sink_1435.md.

The fake below is faithful to reachy-mini 1.8.3 where it matters here:
- ``MediaManager.push_audio_sample`` returns None ALWAYS; before ``start_playing()`` the LOCAL pipeline drops the
  buffer (only the SDK's own log says so), and with ``audio is None`` it drops after a warning;
- ``MediaManager.close()`` leaves the SAME manager in place with ``audio = None``;
- ``backend`` is a public attribute (``MediaBackend.LOCAL`` / ``WEBRTC``), and WebRTC's ``start_playing`` is a no-op;
- ``ReachyMini.disable_wobbling()`` turns off the SDK-side wobbler and tells the daemon; ``start_playing()`` starts
  an existing wobbler, which turns pushed audio into head offsets.
"""

from __future__ import annotations

import enum
import logging

import numpy as np
import pytest

from maxim.utils.audio import make_device_speak_sink, push_audio_to_device


class _Backend(enum.Enum):
    LOCAL = "local"
    WEBRTC = "webrtc"


class _FakeAudio:
    """The SDK's audio object: a playback pipeline that only accepts buffers once started."""

    def __init__(self, media: "_FakeMedia") -> None:
        self._media = media
        self.playing = False
        self.wobbler: str | None = None  # None, "idle" or "running"

    def start_playing(self) -> None:
        if self._media.backend is _Backend.WEBRTC:
            return  # GstWebRTCClient.start_playing is a no-op
        if self.wobbler is not None:
            self.wobbler = "running"
        self.playing = True

    def stop_playing(self) -> None:
        if self.wobbler is not None:
            self.wobbler = "idle"
        self.playing = False


class _FakeMedia:
    def __init__(self, backend: _Backend = _Backend.LOCAL, rate: int = 16000) -> None:
        self.backend = backend
        self.audio: _FakeAudio | None = _FakeAudio(self)
        self._rate = rate
        self.played: list[np.ndarray] = []  # what actually reached a started player
        self.dropped = 0
        self.head_offsets = 0  # wobbler-driven head motion
        self.start_calls = 0

    def get_output_audio_samplerate(self) -> int:
        return -1 if self.audio is None else self._rate

    def start_playing(self) -> None:
        self.start_calls += 1
        if self.audio is not None:
            self.audio.start_playing()

    def stop_playing(self) -> None:
        if self.audio is not None:
            self.audio.stop_playing()

    def push_audio_sample(self, data: np.ndarray) -> None:
        if self.audio is None or not self.audio.playing:
            self.dropped += 1
            return None
        self.played.append(data)
        if self.audio.wobbler == "running":
            self.head_offsets += 1
        return None

    def disable_wobbling(self) -> None:
        if self.audio is not None:
            self.audio.wobbler = None

    def close(self) -> None:
        if self.audio is not None:
            self.audio.stop_playing()
            self.audio = None


class _FakeMini:
    def __init__(self, media: _FakeMedia | None = None, *, wobbling: bool = False) -> None:
        self.media = media or _FakeMedia()
        self.daemon_wobbling = wobbling
        if wobbling and self.media.audio is not None:
            self.media.audio.wobbler = "idle"
        self.calls: list[str] = []

    def disable_wobbling(self) -> None:
        self.calls.append("disable_wobbling")
        self.media.disable_wobbling()
        self.daemon_wobbling = False


def _int16_sine(rate: int, seconds: float, amplitude: int = 20000, hz: float = 440.0) -> np.ndarray:
    t = np.arange(int(rate * seconds)) / rate
    return (amplitude * np.sin(2 * np.pi * hz * t)).astype(np.int16)


# ── Red gates (strict; flipped in place by the fix) ──────────────────────────────────────────────────────────────


@pytest.mark.xfail(strict=True, reason="#1435: the sink reports the SDK's silent drop as played")
def test_a_push_to_an_unstarted_player_returns_false_with_a_warning(caplog):
    mini = _FakeMini()
    with caplog.at_level(logging.WARNING, logger="maxim.utils.audio"):
        ok = push_audio_to_device(mini, _int16_sine(22050, 0.1), 22050)
    assert ok is False
    assert mini.media.played == []
    assert any("start_device_player" in r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING)


@pytest.mark.xfail(strict=True, reason="#1435: the device path never resamples to the device's rate")
def test_22050_hz_reaches_the_device_at_its_rate_with_duration_and_scale_kept():
    from maxim.utils.audio import start_device_player

    mini = _FakeMini()
    assert start_device_player(mini) is True
    src = _int16_sine(22050, 1.0)
    assert make_device_speak_sink(mini)(src, sample_rate=22050) is True

    (out,) = mini.media.played
    assert out.dtype == np.float32
    assert abs(len(out) - 16000) <= 1, "duration must be preserved at the device rate"
    assert float(np.max(np.abs(out))) <= 1.0, "R6: never the int16 scale on a float32 device"
    assert float(np.max(np.abs(out))) == pytest.approx(20000 / 32768, abs=0.02)


@pytest.mark.xfail(strict=True, reason="#1435: nothing disables the wobbler before the player starts")
def test_start_device_player_disables_the_wobbler_before_starting_and_speech_never_moves_the_head():
    from maxim.utils.audio import start_device_player

    mini = _FakeMini(wobbling=True)
    assert start_device_player(mini) is True
    assert mini.daemon_wobbling is False
    assert mini.media.audio is not None and mini.media.audio.wobbler is None
    assert make_device_speak_sink(mini)(_int16_sine(16000, 0.1), sample_rate=16000) is True
    assert mini.media.head_offsets == 0


@pytest.mark.xfail(strict=True, reason="#1435: a speak call must never start the player")
def test_speaking_never_starts_the_player():
    mini = _FakeMini()
    sink = make_device_speak_sink(mini)
    for _ in range(3):
        sink(_int16_sine(16000, 0.05), sample_rate=16000)
    assert mini.media.start_calls == 0
    assert mini.media.played == []
    assert push_audio_to_device(mini, _int16_sine(16000, 0.05), 16000) is False
