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

    def load(self) -> None:
        return None


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


def test_the_robot_runtime_builds_speech_on_its_own_sink_before_it_moves(monkeypatch):
    """The robot hears --tts: live() builds speech on the device sink (MediaLoopMixin.speak) before awaken()
    and before the agentic runtime, which only reads the result."""
    import maxim.utils.audio as audio

    _patch_voice(monkeypatch)
    monkeypatch.setattr(audio, "_check_local_audio", lambda: True)  # mini is None: the sink plays locally
    monkeypatch.delenv("MAXIM_TTS_ENABLED", raising=False)
    runtime = _robot_runtime(tts_requested=True, tts_model=None)
    runtime._build_speech_output()
    engine, speaker = runtime._speech_output
    assert engine is not None and speaker == runtime.speak
    quiet = _robot_runtime(tts_requested=False, tts_model=None)
    quiet._build_speech_output()
    assert quiet._speech_output == (None, None)


def test_a_robot_tts_request_that_cannot_play_stops_the_media_start(monkeypatch):
    """Raised from media start, outside the agentic runtime's warn_unless_refusal, so it is not downgraded."""
    from maxim.utils.response_output import SpeechOutputUnavailable

    _patch_voice(monkeypatch, _NoVoice)
    with pytest.raises(SpeechOutputUnavailable):
        _robot_runtime(tts_requested=True, tts_model="missing")._build_speech_output()


def test_a_local_route_needs_a_local_speaker_even_with_a_robot_sink(monkeypatch):
    """The device sink plays on this machine with MAXIM_TTS_LOCAL or no SDK handle, so local speakers are
    checked then too, and a refusal names the fix."""
    import maxim.utils.audio as audio
    from maxim.utils.response_output import SpeechOutputUnavailable, build_speech_output

    _patch_voice(monkeypatch)
    monkeypatch.setattr(audio, "_check_local_audio", lambda: False)
    robot_sink = lambda samples, *, sample_rate: True  # noqa: E731
    log = logging.getLogger("t")
    engine, speaker = build_speech_output(requested=True, model_name=None, sink=robot_sink, log=log)
    assert speaker is robot_sink  # the robot's own speaker: no local check
    with pytest.raises(SpeechOutputUnavailable):
        build_speech_output(requested=True, model_name=None, sink=robot_sink, log=log, sink_plays_locally=True)
    monkeypatch.setenv("MAXIM_TTS_LOCAL", "1")
    with pytest.raises(SpeechOutputUnavailable):
        build_speech_output(requested=True, model_name=None, sink=robot_sink, log=log)


def test_a_voice_that_will_not_load_refuses_at_start(monkeypatch):
    from maxim.utils.response_output import SpeechOutputUnavailable, build_speech_output

    class InvalidProtobuf(Exception):  # onnxruntime's load errors derive from Exception, not RuntimeError
        pass

    class _Corrupt(_Voice):
        def load(self) -> None:
            raise InvalidProtobuf("Load model from voice.onnx failed: Protobuf parsing failed")

    _patch_voice(monkeypatch, _Corrupt)
    with pytest.raises(SpeechOutputUnavailable, match="could not be loaded"):
        build_speech_output(requested=True, model_name=None, sink=None, log=logging.getLogger("t"))


def test_no_output_device_means_no_local_speaker(monkeypatch):
    """sounddevice imports but the machine has no output device (a headless box): unavailable."""
    import sys
    import types

    import maxim.utils.audio as audio

    class _PortAudioError(Exception):
        pass

    def _no_device(kind=None):
        raise _PortAudioError("Error querying device -1")

    fake_sd = types.SimpleNamespace(PortAudioError=_PortAudioError, query_devices=_no_device)
    monkeypatch.setitem(sys.modules, "sounddevice", fake_sd)
    monkeypatch.setattr(audio, "_local_audio_available", None)
    assert audio._check_local_audio() is False


def test_live_builds_speech_before_the_robot_moves():
    """SOURCE pin: in MediaLoopMixin.live the speech build precedes awaken(), so a refusal moves nothing."""
    import inspect

    from maxim.embodied_runtime.media_loop import MediaLoopMixin

    src = inspect.getsource(MediaLoopMixin.live)
    assert src.index("self._build_speech_output()") < src.index("self.awaken(")


def test_a_speech_refusal_exits_2_from_the_cli_and_the_robot_run(monkeypatch, capsys, caplog):
    import maxim.cli as cli
    from maxim.utils.response_output import SpeechOutputUnavailable

    refusal = SpeechOutputUnavailable("no local speaker can play", "Install it with `pip install 'pymaxim[tts]'`.")

    def _refuse(argv):
        raise refusal

    monkeypatch.setattr(cli, "_main_impl", _refuse)
    with caplog.at_level(logging.WARNING):
        assert cli.main([]) == 2
    shown = capsys.readouterr().err + " ".join(r.getMessage() for r in caplog.records)
    assert "pymaxim[tts]" in shown  # the fix hint reaches the operator
    with pytest.raises(SpeechOutputUnavailable):  # the robot run loop hands it to main, never absorbs it
        cli._run_stopped_on(refusal, logging.getLogger("t"), verbosity=0)


def test_a_missing_tts_extra_is_the_same_refusal(monkeypatch):
    import maxim.utils.response_output as ro
    from maxim.utils.optional_deps import OptionalDependencyError

    def _missing(*a, **k):
        raise OptionalDependencyError("piper", feature="spoken responses")

    monkeypatch.setattr(ro, "require_optional_dependency", _missing)
    with pytest.raises(ro.SpeechOutputUnavailable, match="tts extra"):
        ro.build_speech_output(requested=True, model_name=None, sink=None, log=logging.getLogger("t"))


def test_a_no_media_robot_and_the_local_env_route_speech_locally(monkeypatch):
    """One rule for 'speech plays here': MAXIM_TTS_LOCAL (any truthy spelling), no SDK handle, or no_media."""
    from types import SimpleNamespace

    import maxim.utils.audio as audio

    monkeypatch.setattr(audio, "_backend_of", lambda media: getattr(media, "kind", None))
    local_backend = SimpleNamespace(media=SimpleNamespace(kind=audio._LOCAL))
    no_media = SimpleNamespace(media=SimpleNamespace(kind=audio._NO_MEDIA))
    assert audio.device_speech_plays_locally(None) is True
    assert audio.device_speech_plays_locally(no_media) is True
    assert audio.device_speech_plays_locally(local_backend) is False
    monkeypatch.setenv("MAXIM_TTS_LOCAL", " on ")
    assert audio.tts_local_requested() is True
    assert audio.device_speech_plays_locally(local_backend) is True


def test_a_no_media_robot_checks_local_speakers_at_start(monkeypatch):
    """The wiring: MediaLoopMixin._build_speech_output asks device_speech_plays_locally, so a no_media robot
    (which plays on this machine) refuses at start when no local speaker can play."""
    from types import SimpleNamespace

    import maxim.utils.audio as audio
    from maxim.utils.response_output import SpeechOutputUnavailable

    _patch_voice(monkeypatch)
    monkeypatch.setattr(audio, "_check_local_audio", lambda: False)
    runtime = _robot_runtime(tts_requested=True, tts_model=None)
    runtime.mini = SimpleNamespace(media=SimpleNamespace(backend="no_media"))
    monkeypatch.setattr(audio, "_backend_of", lambda media: audio._NO_MEDIA if media is runtime.mini.media else None)
    with pytest.raises(SpeechOutputUnavailable):
        runtime._build_speech_output()


def test_init_audio_parses_the_audio_settings_and_the_speech_request():
    runtime = _robot_runtime()
    runtime._init_audio(True, "bad", tts=True, tts_model="en_GB-voice")
    assert (runtime.audio, runtime.audio_len, runtime.tts_requested, runtime.tts_model) == (
        True,
        5.0,
        True,
        "en_GB-voice",
    )
    runtime._init_audio(False, -2, tts=False, tts_model=None)
    assert (runtime.audio, runtime.audio_len, runtime.tts_requested) == (False, 5.0, False)
