"""Pairing-code announcer factory for the spoken-code pairing surface (A9.1).

``build_app(pairing_announcer=...)`` takes a ``(code: str) -> None`` callable that
makes the device speak the 6-digit pairing code aloud in the room. Composing a
text-to-speech engine with an audio sink is *library* logic, not presentation-layer
wiring — so pymaxim ships the factory here and an embedder (e.g. the Reachy bootstrap)
passes it to ``build_app`` in one call, keeping the composition out of its own
bootstrap. That is the ARCHITECTURE.md layer-ownership rule: the composition root may
*wire* pre-built pieces, it may not *build a new pipeline*.

Import-light on purpose: this module imports only the standard library, so
``import maxim.console`` stays free of FastAPI/pydantic (they live in the ``console``
extra). The TTS engine and the audio sink are *injected* (duck-typed), so this module
depends on neither ``models.audio`` nor ``embodied_runtime``.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

__all__ = ["make_pairing_announcer"]


def _spell_code(code: str) -> str:
    """Digit-by-digit phrase so a spoken code is unambiguous across a room.

    "482913" -> "Your pairing code is 4, 8, 2, 9, 1, 3." The commas give the TTS
    engine a beat between digits, which a run-together number does not.
    """
    spoken = ", ".join(code)
    return f"Your pairing code is {spoken}."


def make_pairing_announcer(
    tts: Any,
    speak: Callable[..., Any],
    *,
    phrase: Optional[Callable[[str], str]] = None,
    sample_rate: Optional[int] = None,
) -> Callable[[str], None]:
    """Build a ``pairing_announcer`` for ``build_app(pairing_announcer=...)``.

    Composes a text-to-speech engine with an audio sink into the
    ``(code: str) -> None`` callable the console's spoken-code pairing surface
    (A9.1) expects — so an embedder wires it in one call instead of composing
    TTS + ``speak`` inside its own bootstrap (presentation-layer logic the
    layer-ownership rule forbids).

    Args:
        tts: A text-to-speech engine exposing ``synthesize(text) -> samples``
            (e.g. :class:`maxim.models.audio.tts.TTSEngine`). If it also exposes a
            ``sample_rate`` attribute, that rate is passed to ``speak`` unless
            ``sample_rate`` overrides it.
        speak: An audio sink called as ``speak(samples, sample_rate=<int>)`` that
            plays the samples on the device (e.g. the media loop's ``speak``).
        phrase: Optional ``code -> text`` formatter; defaults to a digit-by-digit
            spelling for room clarity.
        sample_rate: Override the sample rate passed to ``speak``; defaults to
            ``tts.sample_rate`` when present, else ``16000``.

    Returns:
        A ``(code: str) -> None`` callable. The console runs it on a daemon
        thread, so it never propagates: a synth/playback failure is logged by
        exception **type only**, and the code is **never** logged (A7).
    """
    to_phrase = phrase if phrase is not None else _spell_code

    def announce(code: str) -> None:
        try:
            samples = tts.synthesize(to_phrase(code))
            sr = sample_rate if sample_rate is not None else getattr(tts, "sample_rate", 16000)
            speak(samples, sample_rate=sr)
        except Exception as exc:  # daemon-thread callable: handle-and-log, never propagate
            # A7: never log the code or the synth text — only the failure TYPE, which
            # cannot contain the code.
            logger.warning("pairing announcer failed to speak the code (%s)", type(exc).__name__)

    return announce
