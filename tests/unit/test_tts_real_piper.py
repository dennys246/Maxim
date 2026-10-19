"""The ``tts`` extra against the REAL piper-tts and a real voice (#1332; standard R2, #1335).

Every other TTS test fakes the engine, which is how 1.3.2 shipped a ``PiperTTS`` written for the pre-1.3 piper API
while the extra's floor is ``piper-tts>=1.3.0``: synthesis raised ``wave.Error: # channels not specified`` on every
installable version, and spoken-code pairing played silence. These tests drive the library itself.

Where they run: wherever ``piper`` imports and ``MAXIM_TEST_TTS_VOICE_DIR`` names a directory holding the pinned
``en_US-lessac-medium`` voice (the voice pairing ships, sha256-verified through ``ensure_tts_voice``; owner decision
2026-10-18). CI's tts lane provisions it from a cache and runs with ``--require-extras=tts``, so a skip there for
either reason is a failure. Linux only in practice: piper-tts 1.8.0's macOS arm64 wheel cannot find its
espeak-ng-data, so a Mac runs only the signature pin.
"""

from __future__ import annotations

import inspect
import os
import wave
from pathlib import Path

import numpy as np
import pytest

piper = pytest.importorskip("piper", reason="requires the `tts` extra (pip install 'pymaxim[tts]')")

VOICE = "en_US-lessac-medium"
PHRASE = "Your pairing code is 4, 8, 2, 9, 1, 3."


@pytest.fixture(scope="module")
def voice_dir() -> Path:
    from maxim.models import download

    raw = os.environ.get("MAXIM_TEST_TTS_VOICE_DIR", "").strip()
    if not raw:
        pytest.skip(f"tts voice not provisioned: set MAXIM_TEST_TTS_VOICE_DIR to a directory holding {VOICE}")
    # auto=False + interactive=False: verify what is there, never download from a test.
    if not download.ensure_tts_voice(VOICE, auto=False, interactive=False, models_dir=raw):
        pytest.skip(f"tts voice not provisioned: {VOICE} is missing or fails its sha256 pin under {raw}")
    return Path(raw)


@pytest.fixture()
def engine(voice_dir: Path):
    from maxim.models.audio.tts import TTSEngine

    return TTSEngine(model_name=VOICE, models_dir=voice_dir, cache_phrases=False)


def _assert_speech(audio: np.ndarray, sample_rate: int) -> None:
    assert audio.dtype == np.int16 and audio.ndim == 1
    assert len(audio) > sample_rate // 2, f"{len(audio)} samples is under half a second of speech"
    assert np.isfinite(audio.astype(np.float64)).all()
    assert int(np.abs(audio.astype(np.int32)).max()) > 1000, "the audio is silence"


def test_the_piper_api_is_the_one_pipertts_calls():
    """A signature pin that needs no voice: an upstream API change fails here, on any box, not on a robot.

    ``PiperTTS`` iterates ``PiperVoice.synthesize(text)`` for ``AudioChunk``s and reads each chunk's
    ``audio_int16_array``; ``synthesize_wav`` is the WAV-writer entry the pre-fix code thought ``synthesize`` was."""
    from piper import AudioChunk, PiperVoice

    params = list(inspect.signature(PiperVoice.synthesize).parameters)
    assert params[:3] == ["self", "text", "syn_config"], params
    assert list(inspect.signature(PiperVoice.synthesize_wav).parameters)[:3] == ["self", "text", "wav_file"]
    assert isinstance(inspect.getattr_static(AudioChunk, "audio_int16_array"), property)
    assert "sample_rate" in AudioChunk.__dataclass_fields__


@pytest.mark.xfail(strict=True, raises=wave.Error, reason="#1332: PiperTTS calls the pre-1.3 piper API")
def test_synthesize_speaks_a_phrase(engine):
    audio = engine.synthesize(PHRASE)
    assert engine.sample_rate == 22050  # read from the voice's config on load
    _assert_speech(audio, engine.sample_rate)


@pytest.mark.xfail(strict=True, raises=wave.Error, reason="#1332: PiperTTS calls the pre-1.3 piper API")
def test_synthesize_to_file_writes_a_playable_wav(engine, tmp_path):
    out = engine.synthesize_to_file(PHRASE, tmp_path / "code.wav")
    with wave.open(str(out), "rb") as wav:
        assert (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) == (1, 2, engine.sample_rate)
        audio = np.frombuffer(wav.readframes(wav.getnframes()), dtype=np.int16)
    _assert_speech(audio, engine.sample_rate)


@pytest.mark.xfail(strict=True, raises=wave.Error, reason="#1332: PiperTTS calls the pre-1.3 piper API")
def test_synthesize_to_file_of_nothing_is_a_valid_empty_wav(engine, tmp_path):
    """No sentence means no chunk, so nothing sets the format: the WAV writer must not be left to raise on close."""
    out = engine.synthesize_to_file("   ", tmp_path / "empty.wav")
    with wave.open(str(out), "rb") as wav:
        assert (wav.getnchannels(), wav.getsampwidth(), wav.getnframes()) == (1, 2, 0)


@pytest.mark.xfail(strict=True, raises=AssertionError, reason="#1332: the announcer swallows the synthesis error")
def test_the_pairing_announcer_puts_real_speech_on_the_sink(engine):
    """The shipped composition (``console/announcer.py``): the announcer logs a failure by type and returns, so a
    broken engine shows up only as a sink that was never called."""
    from maxim.console.announcer import make_pairing_announcer

    played: list[tuple[np.ndarray, int]] = []
    make_pairing_announcer(engine, lambda samples, sample_rate: played.append((samples, sample_rate)))("482913")
    assert len(played) == 1, "the announcer never reached the speaker"
    samples, rate = played[0]
    assert rate == engine.sample_rate == 22050
    _assert_speech(samples, rate)
