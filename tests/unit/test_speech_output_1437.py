"""`maxim --tts` speaks, or refuses at start; it never logs "enabled" for a path that cannot play (#1437).

`cli.py` built a TTSEngine for `--tts`, logged "TTS enabled", and passed `speaker_fn=None`, so
`ResponseOutput.audio_enabled` was always False. The robot runtime had its own copy keyed on
`MAXIM_TTS_ENABLED`. Owner decisions 2026-10-26: one builder, `utils.response_output.build_speech_output`,
used by both; `--tts` and `MAXIM_TTS_ENABLED` enable it on either path; the CLI plays through local speakers
and the robot through its device sink; a request that cannot play refuses at start, naming the fix.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest


class _Voice:
    sample_rate = 22050
    is_available = True

    def __init__(self, model_name: str = "en_US-lessac-medium", **_: object) -> None:
        self.model_name = model_name

    def synthesize(self, text: str) -> np.ndarray:
        return (np.sin(np.arange(2205) / 5.0) * 8000).astype(np.int16)


class _NoVoice(_Voice):
    is_available = False


def _patch_voice(monkeypatch, voice_cls=_Voice):
    import maxim.models.audio.tts as tts

    monkeypatch.setattr(tts, "TTSEngine", voice_cls)
    monkeypatch.setattr("maxim.utils.response_output.require_optional_dependency", lambda *a, **k: None, raising=False)


def test_a_tts_request_on_the_cli_reaches_local_speakers(monkeypatch, tmp_path, caplog):
    import maxim.utils.audio as audio
    from maxim.utils.response_output import ResponseOutput, build_speech_output

    _patch_voice(monkeypatch)
    played = []
    monkeypatch.setattr(audio, "_check_local_audio", lambda: True)
    monkeypatch.setattr(audio, "play_audio_local", lambda s, rate, blocking=False: played.append(rate) or True)
    with caplog.at_level(logging.INFO):
        engine, speaker = build_speech_output(requested=True, model_name=None, sink=None, log=logging.getLogger("t"))
    out = ResponseOutput(sandbox_path=tmp_path, tts_engine=engine, speaker_fn=speaker)
    assert out.audio_enabled is True
    assert out._speak_response("hello") is True
    assert played == [22050], "the local speaker plays at the voice's own rate"
    assert any("TTS enabled" in r.getMessage() for r in caplog.records)


def test_a_tts_request_that_cannot_play_refuses_at_start_and_never_says_enabled(monkeypatch, caplog):
    import maxim.utils.audio as audio
    from maxim.utils.response_output import SpeechOutputUnavailable, build_speech_output

    _patch_voice(monkeypatch, _NoVoice)
    with caplog.at_level(logging.INFO), pytest.raises(SpeechOutputUnavailable) as no_voice:
        build_speech_output(requested=True, model_name="nope", sink=None, log=logging.getLogger("t"))
    assert "maxim.models.download --tts" in str(no_voice.value)

    _patch_voice(monkeypatch)
    monkeypatch.setattr(audio, "_check_local_audio", lambda: False)
    with caplog.at_level(logging.INFO), pytest.raises(SpeechOutputUnavailable) as no_speaker:
        build_speech_output(requested=True, model_name=None, sink=None, log=logging.getLogger("t"))
    assert "pymaxim[tts]" in str(no_speaker.value)
    assert not any("TTS enabled" in r.getMessage() for r in caplog.records)


def test_either_toggle_enables_speech_and_neither_means_silent(monkeypatch):
    from maxim.utils.response_output import build_speech_output

    _patch_voice(monkeypatch)
    log = logging.getLogger("t")
    robot_sink = lambda samples, *, sample_rate: True  # noqa: E731 - the robot's device sink shape
    monkeypatch.delenv("MAXIM_TTS_ENABLED", raising=False)
    assert build_speech_output(requested=False, model_name=None, sink=robot_sink, log=log) == (None, None)
    monkeypatch.setenv("MAXIM_TTS_ENABLED", "1")
    engine, speaker = build_speech_output(requested=False, model_name=None, sink=robot_sink, log=log)
    assert engine is not None and speaker is robot_sink, "on a robot the device sink speaks, not the laptop"


def test_local_speaker_availability_is_truthful_without_portaudio(monkeypatch):
    import builtins

    import maxim.utils.audio as audio

    real_import = builtins.__import__

    def _no_portaudio(name, *args, **kwargs):
        if name == "sounddevice":
            raise OSError("PortAudio library not found")  # what sounddevice raises on Linux without libportaudio
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_portaudio)
    monkeypatch.setattr(audio, "_local_audio_available", None)
    assert audio.LocalSpeaker().is_available is False


def test_both_entry_points_build_speech_through_the_one_builder():
    """SOURCE pins on the CLI and robot seams, the convention until Session A's slices extract cli._main_impl
    and _start_agentic_runtime (owner decision Q6, #963)."""
    import inspect

    from maxim import cli
    from maxim.embodied_runtime import agentic_runtime
    from maxim.embodied_runtime.selfy import Maxim

    cli_source = inspect.getsource(cli)
    assert "tts_engine, speaker_fn = build_speech_output(" in cli_source and "speaker_fn=None" not in cli_source
    assert 'tts=bool(getattr(args, "tts", False))' in cli_source  # the robot hears --tts
    assert "tts" in inspect.signature(Maxim.__init__).parameters
    agentic_source = inspect.getsource(agentic_runtime)
    assert "TTSEngine(" not in agentic_source
    assert 'tts_engine, speaker_fn = getattr(self, "_speech_output", (None, None))' in agentic_source


def _robot_runtime(**attrs):
    from types import SimpleNamespace

    from maxim.embodied_runtime.media_loop import MediaLoopMixin

    class _Runtime(MediaLoopMixin):
        def __init__(self):
            self.mini = None  # no SDK handle: the player start is skipped, speech is still resolved
            self.audio = False
            self._capabilities = SimpleNamespace(has_audio=True)
            self.log = logging.getLogger("t")
            self.__dict__.update(attrs)

    return _Runtime()


def test_the_robot_runtime_builds_speech_on_its_own_sink_at_media_start(monkeypatch):
    """The robot hears --tts: media start builds speech on the device sink (MediaLoopMixin.speak), before the
    agentic runtime, which only reads the result."""
    _patch_voice(monkeypatch)
    monkeypatch.delenv("MAXIM_TTS_ENABLED", raising=False)
    runtime = _robot_runtime(tts_requested=True, tts_model=None)
    runtime._start_media_audio()
    engine, speaker = runtime._speech_output
    assert engine is not None and speaker == runtime.speak
    quiet = _robot_runtime(tts_requested=False, tts_model=None)
    quiet._start_media_audio()
    assert quiet._speech_output == (None, None)


def test_a_robot_tts_request_that_cannot_play_stops_the_media_start(monkeypatch):
    """Raised from media start, outside the agentic runtime's warn_unless_refusal, so it is not downgraded."""
    from maxim.utils.response_output import SpeechOutputUnavailable

    _patch_voice(monkeypatch, _NoVoice)
    with pytest.raises(SpeechOutputUnavailable):
        _robot_runtime(tts_requested=True, tts_model="missing")._start_media_audio()
