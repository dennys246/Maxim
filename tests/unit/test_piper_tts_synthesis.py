"""``PiperTTS`` over a fake voice shaped like piper-tts >= 1.3 (#1332), for the fast suite, which has no piper.

The fake's shape (``synthesize(text)`` yields chunks carrying ``audio_int16_array``) is the one
``tests/unit/test_tts_real_piper.py::test_the_piper_api_is_the_one_pipertts_calls`` pins against the REAL library in
the tts lane (standard R2: a fake of a third-party class is checked against the real signature).
"""

from __future__ import annotations

import wave
from types import SimpleNamespace

import numpy as np
import pytest

from maxim.models.audio.tts import PiperTTS


class _FakeVoice:
    """``PiperVoice`` >= 1.3: one chunk per sentence; no ``synthesize_stream_raw``, no WAV-writer ``synthesize``."""

    def __init__(self, sentences: list[np.ndarray]) -> None:
        self._sentences = sentences
        self.calls: list[str] = []

    def synthesize(self, text, syn_config=None):
        self.calls.append(text)
        for audio in self._sentences:
            yield SimpleNamespace(
                audio_float_array=audio.astype(np.float32) / 32767,
                audio_int16_array=audio,
                sample_rate=16000,
                sample_width=2,
                sample_channels=1,
            )


@pytest.fixture()
def piper_tts(tmp_path):
    def make(sentences):
        tts = PiperTTS(model_path=tmp_path / "v.onnx", cache_phrases=False)
        tts._voice, tts.sample_rate = _FakeVoice(sentences), 16000  # loaded: _load_model returns at once
        return tts

    return make


def test_synthesize_joins_the_sentence_chunks_in_order(piper_tts):
    tts = piper_tts([np.array([1, 2], dtype=np.int16), np.array([3], dtype=np.int16)])
    audio = tts.synthesize("One. Two.")
    assert audio.dtype == np.int16 and audio.tolist() == [1, 2, 3]
    assert tts._voice.calls == ["One. Two."]


def test_text_with_no_sentence_synthesizes_to_empty_int16(piper_tts):
    audio = piper_tts([]).synthesize("...")
    assert audio.dtype == np.int16 and audio.size == 0


def test_synthesize_to_file_writes_its_own_format(piper_tts, tmp_path):
    out = piper_tts([np.array([5, -5, 7], dtype=np.int16)]).synthesize_to_file("Hi.", tmp_path / "a" / "hi.wav")
    with wave.open(str(out), "rb") as wav:
        assert (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) == (1, 2, 16000)
        assert np.frombuffer(wav.readframes(wav.getnframes()), dtype=np.int16).tolist() == [5, -5, 7]


def test_synthesize_to_file_of_no_sentence_is_a_valid_empty_wav(piper_tts, tmp_path):
    """The pre-fix writer left the format to the first chunk: no chunk meant ``wave.Error`` on close (#1332)."""
    out = piper_tts([]).synthesize_to_file("   ", tmp_path / "empty.wav")
    with wave.open(str(out), "rb") as wav:
        assert (wav.getnchannels(), wav.getsampwidth(), wav.getframerate(), wav.getnframes()) == (1, 2, 16000, 0)


def test_a_voice_producing_non_finite_samples_raises_instead_of_going_silent(piper_tts):
    """piper casts float samples to int16, and NaN casts to 0: without this check a broken voice speaks silence,
    and R1's readiness synthesis would arm over it (review fold)."""
    tts = piper_tts([np.array([1, 2], dtype=np.int16)])

    def nan_chunks(text, syn_config=None):
        yield SimpleNamespace(audio_float_array=np.array([0.1, np.nan], dtype=np.float32), audio_int16_array=None)

    tts._voice.synthesize = nan_chunks
    with pytest.raises(ValueError, match="non-finite"):
        tts.synthesize("Hi.")
