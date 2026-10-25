"""The Reachy speaker sink tells the truth (#1435).

Found by maxim-pulse on a real Reachy Mini: ``push_audio_to_device`` returned True while the SDK silently dropped
every sample (nothing in maxim called ``start_playing()``), and it never resampled Piper's 22,050 Hz to the device's
16,000 Hz. The design and the owner decisions: docs/plans/speaker_sink_1435.md.

The fake below is faithful to reachy-mini 1.8.3 where it matters here:
- ``MediaManager.push_audio_sample`` returns None ALWAYS; before ``start_playing()`` the LOCAL pipeline drops the
  buffer (only the SDK's own log says so), and with ``audio is None`` it drops after a warning;
- ``MediaManager.close()`` leaves the SAME manager in place with ``audio = None``;
- ``backend`` is a public attribute (``MediaBackend.LOCAL`` / ``WEBRTC``); WebRTC's ``start_playing`` is a no-op and its
  push drops silently until the peer connection sets ``audio._appsrc``;
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
        self._appsrc: object | None = None  # WebRTC: set only when the peer connection adds the send chain

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
        self.events: list[str] = []  # SDK calls in order, shared with the handle

    def get_output_audio_samplerate(self) -> int:
        return -1 if self.audio is None else self._rate

    def start_playing(self) -> None:
        self.start_calls += 1
        self.events.append("start_playing")
        if self.audio is not None:
            self.audio.start_playing()

    def stop_playing(self) -> None:
        self.events.append("stop_playing")
        if self.audio is not None:
            self.audio.stop_playing()

    def push_audio_sample(self, data: np.ndarray) -> None:
        accepts = self.audio is not None and (
            self.audio._appsrc is not None if self.backend is _Backend.WEBRTC else self.audio.playing
        )
        if not accepts:
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
        self.events.append("close")
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
        self.media.events.append("disable_wobbling")
        self.media.disable_wobbling()
        self.daemon_wobbling = False


def _int16_sine(rate: int, seconds: float, amplitude: int = 20000, hz: float = 440.0) -> np.ndarray:
    t = np.arange(int(rate * seconds)) / rate
    return (amplitude * np.sin(2 * np.pi * hz * t)).astype(np.int16)


# ── Red gates (strict; flipped in place by the fix) ──────────────────────────────────────────────────────────────


def test_a_push_to_an_unstarted_player_returns_false_with_a_warning(caplog):
    mini = _FakeMini()
    with caplog.at_level(logging.WARNING, logger="maxim.utils.audio"):
        ok = push_audio_to_device(mini, _int16_sine(22050, 0.1), 22050)
    assert ok is False
    assert mini.media.played == []
    assert any("start_device_player" in r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING)


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


def test_start_device_player_disables_the_wobbler_before_starting_and_speech_never_moves_the_head():
    from maxim.utils.audio import start_device_player

    mini = _FakeMini(wobbling=True)
    assert start_device_player(mini) is True
    assert mini.daemon_wobbling is False
    assert mini.media.audio is not None and mini.media.audio.wobbler is None
    assert mini.media.events == ["disable_wobbling", "start_playing"], "Q2: disable FIRST, so a wobbler never starts"
    assert make_device_speak_sink(mini)(_int16_sine(16000, 0.1), sample_rate=16000) is True
    assert mini.media.head_offsets == 0


def test_speaking_never_starts_the_player():
    mini = _FakeMini()
    sink = make_device_speak_sink(mini)
    for _ in range(3):
        sink(_int16_sine(16000, 0.05), sample_rate=16000)
    assert mini.media.start_calls == 0
    assert mini.media.played == []
    assert push_audio_to_device(mini, _int16_sine(16000, 0.05), 16000) is False


# ── The record follows the media manager (F1) ────────────────────────────────────────────────────────────────────


def test_a_closed_manager_refuses_even_with_a_record():
    """close()/release_media() keep the SAME manager with audio=None: the sink must not trust the record alone."""
    from maxim.utils.audio import start_device_player

    mini = _FakeMini()
    assert start_device_player(mini) is True
    mini.media.close()
    assert push_audio_to_device(mini, _int16_sine(16000, 0.05), 16000) is False


def test_stop_device_player_forgets_and_stops_and_a_new_manager_has_no_record():
    from maxim.utils.audio import start_device_player, stop_device_player

    mini = _FakeMini()
    assert start_device_player(mini) is True
    stop_device_player(mini)
    assert mini.media.audio is not None and mini.media.audio.playing is False
    assert push_audio_to_device(mini, _int16_sine(16000, 0.05), 16000) is False

    assert start_device_player(mini) is True
    mini.media = _FakeMedia()  # acquire_media() / a reconnect: a new manager, never started
    assert push_audio_to_device(mini, _int16_sine(16000, 0.05), 16000) is False


def test_the_release_media_path_drops_the_record_before_closing():
    from maxim.embodied_runtime.connection import ConnectionMixin
    from maxim.utils.audio import start_device_player

    class _Runtime(ConnectionMixin):
        def __init__(self, mini):
            self.mini = mini
            self.log = logging.getLogger("test")

        def _release_cv2(self):
            pass

    mini = _FakeMini()
    assert start_device_player(mini) is True
    media = mini.media
    _Runtime(mini)._release_media()
    assert media.audio is None
    media.audio = _FakeAudio(media)  # even a manager that comes back is not "started"
    media.audio.playing = True
    assert push_audio_to_device(mini, _int16_sine(16000, 0.05), 16000) is False


# ── The lifecycle owner's start (Q2: disable, then start) ────────────────────────────────────────────────────────


def test_a_repeat_start_reissues_the_disable_and_never_starts_twice():
    from maxim.utils.audio import start_device_player

    mini = _FakeMini()
    assert start_device_player(mini) is True
    mini.daemon_wobbling = True  # another daemon client turned it back on
    assert start_device_player(mini) is True
    assert mini.calls == ["disable_wobbling", "disable_wobbling"]
    assert mini.daemon_wobbling is False
    assert mini.media.start_calls == 1


def test_no_disable_or_a_failing_disable_never_starts_the_player(caplog):
    from maxim.utils.audio import start_device_player

    class _NoDisable:
        def __init__(self):
            self.media = _FakeMedia()

    class _RaisingDisable(_FakeMini):
        def disable_wobbling(self):
            raise ConnectionError("daemon gone")

    for mini in (_NoDisable(), _RaisingDisable(wobbling=True)):
        with caplog.at_level(logging.WARNING, logger="maxim.utils.audio"):
            assert start_device_player(mini) is False
        assert mini.media.start_calls == 0
        assert push_audio_to_device(mini, _int16_sine(16000, 0.05), 16000) is False
    assert any("could move the head" in r.getMessage() for r in caplog.records)


def test_webrtc_plays_on_the_robot_only_when_started_and_its_send_chain_is_connected(monkeypatch, caplog):
    """Owner decision 2026-10-24 (revised after the wire-integrity review): a connected WebRTC push reaches the robot."""
    from maxim.utils.audio import start_device_player

    local = []
    monkeypatch.setattr(
        "maxim.utils.audio.play_audio_local", lambda s, sample_rate, blocking: local.append(sample_rate) or True
    )
    mini = _FakeMini(_FakeMedia(_Backend.WEBRTC), wobbling=True)
    audio = mini.media.audio
    with caplog.at_level(logging.WARNING, logger="maxim.utils.audio"):
        # never started: local, with one WARNING
        assert push_audio_to_device(mini, _int16_sine(22050, 0.05), 22050) is True
        assert start_device_player(mini) is True
        assert mini.daemon_wobbling is False, "the daemon wobbler (driven by WebRTC audio) is off before any push"
        assert mini.media.events == ["disable_wobbling", "start_playing"]
        # started, send chain not connected yet: still local (the SDK would drop silently)
        assert push_audio_to_device(mini, _int16_sine(22050, 0.05), 22050) is True
        assert mini.media.played == [] and local == [22050, 22050]
        # connected: the robot, resampled to its rate
        audio._appsrc = object()
        assert push_audio_to_device(mini, _int16_sine(22050, 0.1), 22050) is True
    (out,) = mini.media.played
    assert abs(len(out) - 1600) <= 1 and local == [22050, 22050]
    reasons = [r.getMessage() for r in caplog.records if "unavailable over WebRTC" in r.getMessage()]
    assert len(reasons) == 2, "one WARNING per state (not started, then not connected), not one per utterance"
    assert "not started" in reasons[0] and "not built yet" in reasons[1]


def test_an_unknown_backend_refuses_instead_of_rerouting(monkeypatch):
    local = []
    monkeypatch.setattr("maxim.utils.audio.play_audio_local", lambda *a, **k: local.append(a) or True)

    mini = _FakeMini()
    mini.media.backend = None  # an SDK rename, or a handle with no backend
    assert push_audio_to_device(mini, _int16_sine(16000, 0.05), 16000) is False
    assert local == [], "an unknown backend must never hop to local speakers"


def test_no_media_backend_plays_locally(monkeypatch):
    monkeypatch.setattr("maxim.utils.audio.play_audio_local", lambda *a, **k: True)

    class _NoMedia:
        value = "no_media"

    media = _FakeMedia()
    media.backend = _NoMedia()
    media.audio = None
    assert push_audio_to_device(_FakeMini(media), _int16_sine(16000, 0.05), 16000) is True


def test_no_robot_handle_stays_on_local_speakers(monkeypatch):
    monkeypatch.setattr("maxim.utils.audio.play_audio_local", lambda *a, **k: True)
    assert push_audio_to_device(None, _int16_sine(16000, 0.05), 16000) is True


def test_a_full_scale_square_resampled_to_the_device_never_exceeds_unit_scale():
    from maxim.utils.audio import start_device_player

    mini = _FakeMini()
    assert start_device_player(mini) is True
    square = np.where(np.sin(2 * np.pi * 300 * np.arange(22050) / 22050) >= 0, 32767, -32768).astype(np.int16)
    assert push_audio_to_device(mini, square, 22050) is True
    (out,) = mini.media.played
    assert out.dtype == np.float32 and float(np.max(np.abs(out))) <= 1.0


# ── The runtime is the lifecycle owner (D5) ──────────────────────────────────────────────────────────────────────


def test_media_start_starts_the_player_once_on_output_capability_not_the_mic_flag():
    from types import SimpleNamespace

    from maxim.embodied_runtime.media_loop import MediaLoopMixin

    class _Runtime(MediaLoopMixin):
        def __init__(self, mini):
            self.mini = mini
            self.audio = False  # no mic capture; TTS can still speak
            self._capabilities = SimpleNamespace(has_audio=True)
            self.log = logging.getLogger("test")

    mini = _FakeMini(wobbling=True)
    runtime = _Runtime(mini)
    assert runtime._start_media_audio() == (None, None)
    assert mini.media.start_calls == 1 and mini.calls == ["disable_wobbling"]
    assert runtime.speak(_int16_sine(22050, 0.1), sample_rate=22050) is True
    assert runtime.speak(_int16_sine(22050, 0.1), sample_rate=22050) is True
    assert mini.media.start_calls == 1, "speaking never starts the player"
    assert mini.media.head_offsets == 0

    no_output = _Runtime(_FakeMini())
    no_output._capabilities = SimpleNamespace(has_audio=False)
    no_output._start_media_audio()
    assert no_output.mini.media.start_calls == 0


def test_media_start_runs_before_the_agentic_runtime_can_speak():
    import inspect

    from maxim.embodied_runtime.media_loop import MediaLoopMixin

    src = inspect.getsource(MediaLoopMixin.live)
    assert src.index("self._start_media_audio()") < src.index("self._start_agentic_runtime()")


def test_a_soft_reconnect_starts_the_new_handles_player():
    import threading
    from types import SimpleNamespace

    from maxim.embodied_runtime.connection import ConnectionMixin
    from maxim.utils.audio import start_device_player

    old, new = _FakeMini(), _FakeMini()

    class _Robot:
        def __init__(self):
            self.mini = old

        def reconnect(self, timeout, max_attempts):
            self.mini = new
            return True

        def start_recording(self):
            pass

    class _Runtime(ConnectionMixin):
        def __init__(self):
            self._robot = _Robot()
            self._simulation = False
            self._reconnect_lock = threading.Lock()
            self._capabilities = SimpleNamespace(has_audio=True)
            self._connection_failures = {}
            self.log = logging.getLogger("test")

        @property
        def mini(self):
            return self._robot.mini

        def _restore_capabilities(self):
            pass

    assert start_device_player(old) is True
    assert _Runtime()._soft_reconnect(reason="test") is True
    assert old.media.events[-2:] == ["stop_playing", "close"], (
        "the record is dropped and the player stopped BEFORE close"
    )
    assert new.media.start_calls == 1 and new.calls == ["disable_wobbling"]
    assert push_audio_to_device(new, _int16_sine(16000, 0.05), 16000) is True
    assert push_audio_to_device(old, _int16_sine(16000, 0.05), 16000) is False


# ── The other sinks and the speak tool tell the same truth (D3, D4) ─────────────────────────────────────────────


def test_the_reachy_audio_stream_pushes_only_to_a_started_player():
    from maxim.hardware.reachy.streams import ReachyAudioStream
    from maxim.utils.audio import start_device_player

    mini = _FakeMini()
    stream = ReachyAudioStream(mini)
    stream.start()
    pcm = np.zeros(160, dtype=np.float32)
    assert stream.push_sample(pcm) is False
    assert start_device_player(mini) is True
    assert stream.push_sample(pcm) is True
    assert stream.push_sample(pcm * 0 + 20000.0) is False, "out-of-scale float is refused, not clipped"


def test_the_speak_tool_reports_spoken_from_the_sink_and_hands_it_the_tts_rate(tmp_path):
    from maxim.tools.response import SpeakTool
    from maxim.utils.response_output import ResponseOutput

    class _Piper:
        sample_rate = 22050

        def synthesize(self, text):
            return _int16_sine(22050, 0.2)

    mini = _FakeMini()
    seen = []

    def _speaker(samples, sample_rate):
        seen.append((samples.dtype, sample_rate))
        return push_audio_to_device(mini, samples, sample_rate)

    out = ResponseOutput(sandbox_path=tmp_path, tts_engine=_Piper(), speaker_fn=_speaker)
    refused = SpeakTool(out).execute(text="hello", also_print=False)
    assert refused.output["spoken"] is False and "reason" in refused.output

    from maxim.utils.audio import start_device_player

    assert start_device_player(mini) is True
    spoken = SpeakTool(out).execute(text="hello", also_print=False)
    assert spoken.output == {"spoken": True, "printed": False}
    assert seen[-1] == (np.dtype(np.int16), 22050), "no pre-resample: the sink owns the device rate (R6)"
    assert float(np.max(np.abs(mini.media.played[-1]))) <= 1.0


def test_only_the_lifecycle_owner_starts_the_player_or_touches_the_wobbler():
    """The brief's rule, mechanized over src/: no direct start_playing()/enable_wobbling(), and no push that
    bypasses the record (the sink's refusal is only as strong as the absence of other pushes)."""
    import pathlib
    import re

    import maxim

    root = pathlib.Path(maxim.__file__).parent
    pattern = re.compile(r"\.(start_playing|enable_wobbling|push_audio_sample|cancel_move)\(")
    offenders = [
        f"{path.relative_to(root)}:{n}"
        for path in root.rglob("*.py")
        if path.relative_to(root).as_posix() != "utils/audio.py"
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if pattern.search(line)
    ]
    assert offenders == [], f"start or feed the Reachy player only through utils/audio.py: {offenders}"


# ── The record lock (F14, and the Executor lens's lock-order finding) ────────────────────────────────────────────


def test_on_failure_runs_with_the_record_lock_released():
    """on_failure may soft-reconnect, which takes the runtime's media lock; holding the record lock there could deadlock
    against a teardown that holds the media lock and calls stop_device_player."""
    import threading

    from maxim.utils.audio import start_device_player, stop_device_player

    mini = _FakeMini()
    assert start_device_player(mini) is True

    def _fails(data):
        raise ConnectionError("link down")

    mini.media.push_audio_sample = _fails
    lock_was_free = []

    def _on_failure(e):
        t = threading.Thread(target=stop_device_player, args=(mini,))
        t.start()
        t.join(timeout=2.0)
        lock_was_free.append(not t.is_alive())

    assert push_audio_to_device(mini, _int16_sine(16000, 0.05), 16000, on_failure=_on_failure) is False
    assert lock_was_free == [True]


def test_a_stop_cannot_slip_between_the_record_check_and_the_push():
    import threading

    from maxim.utils.audio import start_device_player, stop_device_player

    mini = _FakeMini()
    assert start_device_player(mini) is True
    entered, release = threading.Event(), threading.Event()
    real_push = mini.media.push_audio_sample
    order: list[str] = []

    def _slow_push(data):
        entered.set()
        release.wait(timeout=2.0)
        order.append("push")
        real_push(data)

    mini.media.push_audio_sample = _slow_push
    speaker = threading.Thread(target=push_audio_to_device, args=(mini, _int16_sine(16000, 0.05), 16000))
    speaker.start()
    assert entered.wait(timeout=2.0)
    stopper = threading.Thread(target=lambda: (stop_device_player(mini), order.append("stop")))
    stopper.start()
    stopper.join(timeout=0.2)
    assert stopper.is_alive(), "the stop must wait for the in-flight push"
    release.set()
    speaker.join(timeout=2.0)
    stopper.join(timeout=2.0)
    assert order == ["push", "stop"]


# ── Every refusal row of the lifecycle calls runs (coverage + truth) ─────────────────────────────────────────────


def test_start_device_player_refusal_table(caplog):
    from maxim.utils.audio import start_device_player

    class _Rate(_FakeMini):
        def __init__(self, rate=None, raises=False, start_raises=False):
            super().__init__()
            if rate is not None:
                self.media._rate = rate
            if raises:
                self.media.get_output_audio_samplerate = lambda: (_ for _ in ()).throw(RuntimeError("bus"))
            if start_raises:
                self.media.start_playing = lambda: (_ for _ in ()).throw(RuntimeError("gst"))

    closed = _FakeMini()
    closed.media.close()

    class _Unweakable:
        __slots__ = ("backend", "audio", "starts")

        def __init__(self):
            self.backend = _Backend.LOCAL
            self.audio = object()
            self.starts = 0

        def get_output_audio_samplerate(self):
            return 16000

        def start_playing(self):
            self.starts += 1

    class _UnweakMini(_FakeMini):
        def __init__(self):
            super().__init__()
            self.media = _Unweakable()

        def disable_wobbling(self):
            pass

    class _Unhashable(_FakeMedia):
        __hash__ = None  # weak-referenceable but unhashable (e.g. a class defining __eq__)

    unhashable = _FakeMini(_Unhashable())
    other = _FakeMini()
    other.media.backend = None
    cases = {
        "no handle": None,
        "audio closed": closed,
        "unknown backend": other,
        "rate unreadable": _Rate(raises=True),
        "no rate": _Rate(rate=-1),
        "start raises": _Rate(start_raises=True),
        "no record": _UnweakMini(),
        "unhashable": unhashable,
    }
    with caplog.at_level(logging.WARNING, logger="maxim.utils.audio"):
        for name, mini in cases.items():
            assert start_device_player(mini) is False, name
    assert cases["no record"].media.starts == 0, "refused BEFORE the player starts, never after"
    assert unhashable.media.start_calls == 0
    assert push_audio_to_device(cases["no record"], _int16_sine(16000, 0.05), 16000) is False


def test_stop_device_player_is_safe_everywhere(caplog):
    from maxim.utils.audio import start_device_player, stop_device_player

    stop_device_player(None)
    never = _FakeMini()
    stop_device_player(never)
    assert never.media.events == []

    class _NoMedia:
        media = None

    stop_device_player(_NoMedia())

    mini = _FakeMini()
    assert start_device_player(mini) is True

    def _raises():
        raise RuntimeError("gst")

    mini.media.stop_playing = _raises
    with caplog.at_level(logging.WARNING, logger="maxim.utils.audio"):
        stop_device_player(mini)
    assert any("Failed to stop the Reachy speaker" in r.getMessage() for r in caplog.records)


def test_a_restart_warns_again_after_the_next_refusal(caplog):
    from maxim.utils.audio import start_device_player, stop_device_player

    mini = _FakeMini()
    with caplog.at_level(logging.WARNING, logger="maxim.utils.audio"):
        push_audio_to_device(mini, _int16_sine(16000, 0.05), 16000)
        push_audio_to_device(mini, _int16_sine(16000, 0.05), 16000)
        assert start_device_player(mini) is True
        stop_device_player(mini)
        push_audio_to_device(mini, _int16_sine(16000, 0.05), 16000)
    refusals = [r for r in caplog.records if "player is not started" in r.getMessage()]
    assert len(refusals) == 2, "one per manager per started-state, not one per utterance and not one ever"


def test_bad_rates_and_shapes_are_refused_without_raising():
    from maxim.utils.audio import push_to_started_player, start_device_player

    mini = _FakeMini()
    assert start_device_player(mini) is True
    assert push_audio_to_device(mini, _int16_sine(16000, 0.05), 0) is False
    assert push_audio_to_device(mini, np.int16(5), 16000) is False
    assert push_to_started_player(None, _int16_sine(16000, 0.05), sample_rate=16000) is False
    assert push_to_started_player(mini, np.array([], dtype=np.int16), sample_rate=16000) is False
    assert push_to_started_player(mini, np.int16(5), sample_rate=16000) is False
    assert push_audio_to_device(mini, _int16_sine(16000, 0.05), None) is False
    assert push_audio_to_device(mini, _int16_sine(16000, 0.05), "fast") is False
    assert push_audio_to_device(mini, _int16_sine(16000, 0.05), float("inf")) is False
    assert push_audio_to_device(mini, _int16_sine(16000, 0.05), True) is False

    class _NoMedia:
        media = None

    assert push_to_started_player(_NoMedia(), _int16_sine(16000, 0.05), sample_rate=16000) is False
    assert mini.media.played == []


def test_the_stream_resamples_from_its_advertised_rate():
    from maxim.hardware.reachy.streams import ReachyAudioStream
    from maxim.utils.audio import start_device_player

    mini = _FakeMini()
    stream = ReachyAudioStream(mini)
    stream.start()
    assert start_device_player(mini) is True
    advertised = stream.output_sample_rate
    assert stream.push_sample(np.zeros(advertised // 10, dtype=np.float32)) is True
    assert abs(len(mini.media.played[-1]) - 1600) <= 1, "100 ms at the advertised rate is 100 ms on the device"


def test_the_selfy_fallback_pushes_through_the_started_player():
    from maxim.embodied_runtime.selfy import Maxim
    from maxim.utils.audio import start_device_player

    mini = _FakeMini()

    class _Robot:
        def get_audio_stream(self):
            return None

    runtime = Maxim.__new__(Maxim)
    runtime._robot = _Robot()
    runtime.log = logging.getLogger("test")
    import unittest.mock as mock

    # Maxim.mini is a property over the robot controller; pin it to the fake handle.
    with mock.patch.object(Maxim, "mini", new_callable=mock.PropertyMock, return_value=mini):
        pcm = np.zeros(160, dtype=np.float32)
        assert runtime.push_audio_sample(pcm) is False
        assert start_device_player(mini) is True
        assert runtime.push_audio_sample(pcm) is True
        mini.media.get_output_audio_samplerate = lambda: (_ for _ in ()).throw(RuntimeError("bus"))
        assert runtime.push_audio_sample(pcm) is False


def test_the_live_teardown_drops_the_record_and_the_rates_are_read_with_the_mic_on():
    import threading
    from types import SimpleNamespace

    from maxim.embodied_runtime.connection import ConnectionMixin
    from maxim.embodied_runtime.media_loop import MediaLoopMixin

    class _Runtime(MediaLoopMixin, ConnectionMixin):  # Maxim composes both; _release_media is ConnectionMixin's
        def __init__(self, mini):
            self.mini = mini
            self.audio = True
            self._capabilities = SimpleNamespace(has_audio=True)
            self.log = logging.getLogger("test")

        def _release_cv2(self):
            pass

    mini = _FakeMini()
    mini.stop_recording = lambda: None
    mini.media.get_input_audio_samplerate = lambda: 16000
    runtime = _Runtime(mini)
    assert runtime._start_media_audio() == (16000, 16000)
    runtime._stop_media_io(threading.Lock())
    assert mini.media.events[-2:] == ["stop_playing", "close"]
    assert push_audio_to_device(mini, _int16_sine(16000, 0.05), 16000) is False


def test_a_reconnect_without_audio_output_does_not_start_a_player():
    import threading
    from types import SimpleNamespace

    from maxim.embodied_runtime.connection import ConnectionMixin

    new = _FakeMini()

    class _Robot:
        mini = _FakeMini()

        def reconnect(self, timeout, max_attempts):
            self.mini = new
            return True

        def start_recording(self):
            pass

    class _Runtime(ConnectionMixin):
        def __init__(self):
            self._robot = _Robot()
            self._simulation = False
            self._reconnect_lock = threading.Lock()
            self._capabilities = SimpleNamespace(has_audio=False)
            self._connection_failures = {}
            self.log = logging.getLogger("test")

        @property
        def mini(self):
            return self._robot.mini

        def _restore_capabilities(self):
            pass

    assert _Runtime()._soft_reconnect(reason="test") is True
    assert new.media.start_calls == 0


def test_speak_response_reports_false_for_every_non_playing_path(tmp_path):
    from maxim.utils.audio import LocalSpeaker
    from maxim.utils.response_output import ResponseOutput

    assert ResponseOutput(sandbox_path=tmp_path)._speak_response("hi") is False

    class _Empty:
        sample_rate = 22050

        def synthesize(self, text):
            return np.array([], dtype=np.int16)

    class _Broken(_Empty):
        def synthesize(self, text):
            raise RuntimeError("voice gone")

    assert ResponseOutput(tmp_path, tts_engine=_Empty(), speaker_fn=lambda *a, **k: True)._speak_response("hi") is False
    assert (
        ResponseOutput(tmp_path, tts_engine=_Broken(), speaker_fn=lambda *a, **k: True)._speak_response("hi") is False
    )

    played = []
    speaker = LocalSpeaker(sample_rate=16000)
    import maxim.utils.audio as audio_mod

    real = audio_mod.play_audio_local
    audio_mod.play_audio_local = lambda s, rate, blocking=False: played.append(rate) or True
    try:
        assert speaker(np.zeros(4, dtype=np.int16), sample_rate=22050) is True
        assert speaker(np.zeros(4, dtype=np.int16)) is True
    finally:
        audio_mod.play_audio_local = real
    assert played == [22050, 16000]


# ── The fake is bound to the installed SDK (the brief calls it SDK-faithful) ─────────────────────────────────────


def test_the_fake_and_the_webrtc_probe_match_the_installed_sdk():
    """Pins what the fake and ``_webrtc_send_chain_ready`` assume. It skips in CI until the R2 reachy cell exists
    (mechanization backlog M69) and runs wherever reachy-mini is installed."""
    import inspect

    import importlib.util
    import pathlib

    reachy_mini = pytest.importorskip("reachy_mini")
    from reachy_mini.media import media_manager

    # Read, not imported: the WebRTC client needs a GStreamer runtime to import.
    webrtc_src = pathlib.Path(importlib.util.find_spec("reachy_mini.media").submodule_search_locations[0])
    webrtc_src = (webrtc_src / "webrtc_client_gstreamer.py").read_text(encoding="utf-8")

    assert media_manager.MediaBackend.LOCAL.value == "local"
    assert media_manager.MediaBackend.WEBRTC.value == "webrtc"
    assert media_manager.MediaBackend.NO_MEDIA.value == "no_media"
    init = inspect.getsource(media_manager.MediaManager.__init__)
    assert "self.backend" in init and "self.audio" in init
    for name in ("start_playing", "stop_playing", "push_audio_sample", "get_output_audio_samplerate", "close"):
        assert callable(getattr(media_manager.MediaManager, name)), name
    assert callable(getattr(reachy_mini.ReachyMini, "disable_wobbling", None)), "the wobbler can be turned off"
    assert "class GstWebRTCClient" in webrtc_src
    assert "self._appsrc = None" in webrtc_src, "the send-chain probe reads _appsrc: None until connected"
    assert "self._appsrc = appsrc" in webrtc_src, "set when the peer connection adds the send chain"
    assert "if self._appsrc is None:" in webrtc_src, "the push returns silently while unconnected"
    assert webrtc_src.count("self._appsrc = None") == 1, (
        "the SDK now clears _appsrc somewhere: revisit the 'built, not connected' known gap in the docs"
    )


def test_a_stop_during_the_conversion_refuses_the_push(monkeypatch):
    import maxim.utils.audio as audio_mod
    from maxim.utils.audio import start_device_player, stop_device_player

    mini = _FakeMini()
    assert start_device_player(mini) is True
    real = audio_mod._to_device_float32

    def _convert_then_stop(samples):
        out = real(samples)
        stop_device_player(mini)  # the runtime tears down while a long utterance converts
        return out

    monkeypatch.setattr(audio_mod, "_to_device_float32", _convert_then_stop)
    assert push_audio_to_device(mini, _int16_sine(22050, 0.1), 22050) is False
    assert mini.media.played == []


def test_a_close_during_the_conversion_refuses_the_push(monkeypatch):
    """An embedder that closes media without stop_device_player, mid-conversion: refused, not a silent drop."""
    import maxim.utils.audio as audio_mod
    from maxim.utils.audio import start_device_player

    mini = _FakeMini()
    assert start_device_player(mini) is True
    real = audio_mod._to_device_float32

    def _convert_then_close(samples):
        out = real(samples)
        mini.media.close()
        return out

    monkeypatch.setattr(audio_mod, "_to_device_float32", _convert_then_close)
    assert push_audio_to_device(mini, _int16_sine(22050, 0.1), 22050) is False


def test_webrtc_with_a_send_chain_but_never_started_plays_locally(monkeypatch):
    """The lifecycle owner never ran, so the daemon wobbler may be on: never push to the robot."""
    local = []
    monkeypatch.setattr("maxim.utils.audio.play_audio_local", lambda *a, **k: local.append(1) or True)
    mini = _FakeMini(_FakeMedia(_Backend.WEBRTC))
    mini.media.audio._appsrc = object()
    assert push_audio_to_device(mini, _int16_sine(16000, 0.05), 16000) is True
    assert mini.media.played == [] and local == [1]

    class _UnweakWebRTC:
        __slots__ = ("backend", "audio")

        def __init__(self):
            self.backend = _Backend.WEBRTC
            self.audio = None

    class _Handle:
        media = _UnweakWebRTC()

    assert push_audio_to_device(_Handle(), _int16_sine(16000, 0.05), 16000) is True
    assert local == [1, 1]


def test_a_restart_at_another_rate_during_the_conversion_refuses_the_push(monkeypatch):
    """The audio was resampled for the old rate: pushing it would play at the wrong pitch and report played."""
    import maxim.utils.audio as audio_mod
    from maxim.utils.audio import start_device_player, stop_device_player

    mini = _FakeMini()
    assert start_device_player(mini) is True
    real = audio_mod._to_device_float32

    def _convert_then_restart(samples):
        out = real(samples)
        stop_device_player(mini)
        mini.media._rate = 22050
        assert start_device_player(mini) is True
        return out

    monkeypatch.setattr(audio_mod, "_to_device_float32", _convert_then_restart)
    assert push_audio_to_device(mini, _int16_sine(22050, 0.1), 22050) is False
    assert mini.media.played == []
