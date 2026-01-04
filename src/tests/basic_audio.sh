#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-}"
if [[ -z "$PYTHON" ]]; then
  if [[ -x ".venv/bin/python" ]]; then
    PYTHON=".venv/bin/python"
  else
    PYTHON="python3"
  fi
fi

TMP_DIR="$(mktemp -d)"
cleanup() { rm -rf "$TMP_DIR"; }
trap cleanup EXIT

echo "[basic_audio] tmp=$TMP_DIR"

export TMP_DIR
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

"$PYTHON" - <<'PY'
import json
import os
import wave

import numpy as np

from maxim.data.audio.sound import transcribe_audio


def write_wav(path: str, sample_rate: int, audio_i16: np.ndarray) -> None:
    arr = np.asarray(audio_i16, dtype=np.int16)
    channels = 1 if arr.ndim == 1 else int(arr.shape[1])
    wf = wave.open(path, "wb")
    try:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(np.ascontiguousarray(arr).tobytes())
    finally:
        wf.close()


class DummyTranscriber:
    def transcribe(self, audio, **kwargs):
        assert isinstance(audio, np.ndarray)
        assert audio.ndim == 1, f"expected mono 1D audio, got shape {audio.shape}"
        assert audio.dtype == np.float32, f"expected float32, got {audio.dtype}"
        assert audio.flags["C_CONTIGUOUS"], "audio not contiguous"
        return {"text": "dummy ok", "segments": []}


tmp_dir = os.environ.get("TMP_DIR", "")
if not tmp_dir:
    raise SystemExit("TMP_DIR not set by launcher.")

sr = 16000
t = np.linspace(0, 1.0, sr, endpoint=False, dtype=np.float32)
tone = 0.1 * np.sin(2 * np.pi * 440.0 * t)

stereo = np.stack([tone, tone], axis=1)
audio_i16 = (stereo * 32767.0).astype(np.int16)

wav_path = os.path.join(tmp_dir, "tone.wav")
write_wav(wav_path, sr, audio_i16)

dummy = DummyTranscriber()
out = transcribe_audio(dummy, audio_i16, language="en")
assert isinstance(out, dict) and out.get("text") == "dummy ok"

print("[basic_audio] OK: wrote wav:", wav_path)
print("[basic_audio] OK: transcribe_audio() mono conversion + dtype handling")

if os.environ.get("MAXIM_TEST_REAL_WHISPER", "").strip() not in ("1", "true", "yes", "on"):
    print("[basic_audio] SKIP: real whisper test (set MAXIM_TEST_REAL_WHISPER=1)")
    raise SystemExit(0)

try:
    from maxim.models.audio.transcription import WhisperTranscriber

    transcriber = WhisperTranscriber(model_size_or_path="tiny", device="cpu", compute_type="int8")
except Exception as e:
    print("[basic_audio] SKIP: faster-whisper unavailable:", e)
    raise SystemExit(0)

try:
    result = transcribe_audio(transcriber, wav_path, language="en", beam_size=1, vad_filter=True)
    print("[basic_audio] whisper text:", json.dumps(result.get("text", ""), ensure_ascii=False))
except Exception as e:
    print("[basic_audio] SKIP: whisper transcribe failed (model not cached?):", e)
PY

echo "[basic_audio] done"
