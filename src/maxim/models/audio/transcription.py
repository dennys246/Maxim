from __future__ import annotations

from typing import Any


class WhisperTranscriber:
    """
    Thin wrapper around `faster-whisper` so the rest of the codebase only needs a
    single, stable interface.
    """

    def __init__(
        self,
        *,
        model_size_or_path: str = "large-v3",
        device: str = "cpu",
        compute_type: str = "int8",
    ) -> None:
        try:
            from faster_whisper import WhisperModel
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "Missing dependency `faster-whisper`. Install it (and its backend) to enable transcription."
            ) from e

        self.model_size_or_path = str(model_size_or_path or "tiny")
        self.device = str(device or "cpu")
        self.compute_type = str(compute_type or "int8")
        self._model = WhisperModel(self.model_size_or_path, device=self.device, compute_type=self.compute_type)

    def transcribe(
        self,
        audio: Any,
        *,
        language: str = "en",
        beam_size: int = 1,
        vad_filter: bool = True,
    ) -> dict[str, Any]:
        segments, info = self._model.transcribe(
            audio,
            language=str(language or "en"),
            beam_size=int(beam_size or 1),
            vad_filter=bool(vad_filter),
        )

        seg_list: list[dict[str, Any]] = []
        text_parts: list[str] = []
        for seg in segments:
            seg_list.append(
                {
                    "start": float(getattr(seg, "start", 0.0) or 0.0),
                    "end": float(getattr(seg, "end", 0.0) or 0.0),
                    "text": str(getattr(seg, "text", "")),
                }
            )
            text_parts.append(str(getattr(seg, "text", "")))

        language_out = None
        duration_out = None
        try:
            language_out = getattr(info, "language", None)
            duration_out = getattr(info, "duration", None)
        except Exception:
            language_out = None
            duration_out = None

        return {
            "text": "".join(text_parts).strip(),
            "segments": seg_list,
            "language": language_out,
            "duration": duration_out,
        }
