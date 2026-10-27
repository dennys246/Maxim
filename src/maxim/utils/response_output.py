"""Response output management for CLI and file-based responses.

This module provides a unified interface for routing LLM responses to either:
- CLI (for simple, short responses)
- Sandbox files (for complex, lengthy responses)
- Audio output via TTS (optional)
"""

from __future__ import annotations

import logging
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from maxim.utils.optional_deps import OptionalDependencyError, require_optional_dependency

# Audio utilities deferred to call site to avoid eager numpy/scipy loading

if TYPE_CHECKING:
    pass

# ─────────────────────────────────────────────────────────────────────────────
# Sentence boundary detection for natural speech truncation
# ─────────────────────────────────────────────────────────────────────────────

# Matches sentence-ending punctuation followed by space or end of string
_SENTENCE_END_PATTERN = re.compile(r"[.!?]+[\s\n]+|[.!?]+$")


def _print_agent_response(text: str) -> None:
    """Route agent response text through the active display or raw print.

    When a MaximDisplay is active, uses ``display_response`` so the text
    appears in the log panel instead of corrupting the rich Live layout.
    When no display is active, prints with the original ``[Maxim]: ...``
    formatting and re-shows the ``maxim>`` prompt hint.
    """
    try:
        from maxim.simulation.sim_logger import get_active_display

        if get_active_display() is not None:
            from maxim.simulation.sim_logger import display_response

            display_response(text)
            return
    except ImportError:
        pass
    print(f"\n{text}\n", flush=True)
    sys.stdout.write("maxim> ")
    sys.stdout.flush()


def _truncate_at_sentence(text: str, max_length: int) -> str:
    """Truncate text at the nearest sentence boundary before max_length.

    Args:
        text: The text to truncate.
        max_length: Maximum length (will find sentence boundary before this).

    Returns:
        Truncated text ending at a natural sentence boundary.
    """
    if len(text) <= max_length:
        return text

    # Look for sentence endings within the allowed length
    search_text = text[:max_length]
    matches = list(_SENTENCE_END_PATTERN.finditer(search_text))

    if matches:
        # Use the last complete sentence
        last_match = matches[-1]
        return text[: last_match.end()].strip()

    # No sentence boundary found - look for other natural breaks
    # Try comma, semicolon, or dash
    for sep in [", ", "; ", " - ", " — "]:
        last_pos = search_text.rfind(sep)
        if last_pos > max_length // 2:  # Only if it's past halfway
            return text[: last_pos + len(sep)].strip() + "..."

    # Last resort: break at word boundary
    last_space = search_text.rfind(" ")
    if last_space > max_length // 2:
        return text[:last_space].strip() + "..."

    # Absolute fallback
    return text[:max_length].strip() + "..."


class ResponseOutput:
    """Manages LLM response output to CLI and files.

    Routes responses based on complexity:
    - Simple responses (< 500 chars, no code blocks) -> CLI
    - Complex responses -> Sandbox file with CLI notification

    Optionally supports TTS for audio output.
    """

    SIMPLE_THRESHOLD = 500  # Characters - responses under this go to CLI
    CODE_BLOCK_PATTERN = re.compile(r"```[\w]*\n[\s\S]*?```")
    TABLE_PATTERN = re.compile(r"\|.*\|.*\|")
    LIST_PATTERN = re.compile(r"^[\s]*[-*\d]+[.)]\s", re.MULTILINE)

    def __init__(
        self,
        sandbox_path: Path | str,
        logger: logging.Logger | None = None,
        tts_engine: Any | None = None,
        speaker_fn: Callable[..., bool] | None = None,
    ):
        """Initialize response output manager.

        Args:
            sandbox_path: Path to the sandbox directory.
            logger: Logger instance for debug output.
            tts_engine: Optional TTS engine for audio responses.
            speaker_fn: Optional audio sink, called as ``speaker_fn(samples, sample_rate=<int>)``
                and returning whether the audio played (the sink resamples to its device's
                rate; #1435).
        """
        self.sandbox_path = Path(sandbox_path)
        self.logger = logger or logging.getLogger(__name__)
        self.tts = tts_engine
        self.speaker = speaker_fn
        self.audio_enabled = tts_engine is not None and speaker_fn is not None

        # Ensure output directory exists
        self.response_dir = self.sandbox_path / "outputs" / "responses"
        self.response_dir.mkdir(parents=True, exist_ok=True)

    def output_response(
        self,
        text: str,
        response_type: str = "general",
        output_mode: str = "auto",
        speak: bool = False,
    ) -> str | None:
        """Route response to CLI or file based on complexity.

        Args:
            text: The response text to output.
            response_type: Type hint for content ("general", "code", "data").
            output_mode: "auto" (decide based on content), "cli" (force CLI),
                        or "file" (force file).
            speak: If True and TTS is enabled, also speak the response.

        Returns:
            File path if saved to file, None if output to CLI only.
        """
        text = text.strip()
        if not text:
            return None

        # Determine output destination
        if output_mode == "cli":
            use_cli = True
        elif output_mode == "file":
            use_cli = False
        else:  # auto
            use_cli = self._is_simple(text)

        # Output to appropriate destination
        if use_cli:
            self._output_to_cli(text)
            file_path = None
        else:
            file_path = self._output_to_file(text, response_type)

        # Optional TTS
        if speak and self.audio_enabled:
            self._speak_response(text)

        return file_path

    def _is_simple(self, text: str) -> bool:
        """Determine if response is simple enough for CLI.

        A response is considered simple if:
        - Length is under SIMPLE_THRESHOLD characters
        - Contains no code blocks
        - Contains no tables
        - Contains fewer than 5 list items
        """
        # Check length
        if len(text) > self.SIMPLE_THRESHOLD:
            return False

        # Check for code blocks
        if self.CODE_BLOCK_PATTERN.search(text):
            return False

        # Check for tables
        if self.TABLE_PATTERN.search(text):
            return False

        # Check for long lists (more than 5 items)
        list_matches = self.LIST_PATTERN.findall(text)
        if len(list_matches) > 5:
            return False

        return True

    def _output_to_cli(self, text: str) -> None:
        """Print response to CLI with formatting."""
        _print_agent_response(f"[Maxim]: {text}")

    def _output_to_file(self, text: str, response_type: str) -> str:
        """Save complex output to sandbox and notify user.

        Args:
            text: The response text to save.
            response_type: Type hint for determining file extension.

        Returns:
            Absolute path to the saved file.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Determine file extension based on content
        ext = self._detect_extension(text, response_type)
        filename = f"response_{timestamp}.{ext}"

        output_path = self.response_dir / filename
        output_path.write_text(text, encoding="utf-8")

        # Notify user via CLI
        try:
            relative_path = output_path.relative_to(self.sandbox_path.parent)
        except ValueError:
            relative_path = output_path

        _print_agent_response(f"[Maxim]: Response saved to: {relative_path}")

        self.logger.debug("Saved response to %s", output_path)
        return str(output_path)

    def _detect_extension(self, text: str, response_type: str) -> str:
        """Detect appropriate file extension based on content.

        Args:
            text: The response text to analyze.
            response_type: Type hint from caller.

        Returns:
            File extension without the dot (e.g., "py", "json", "md").
        """
        # Explicit type hints take precedence
        if response_type == "code":
            # Try to detect language from code block
            if "```python" in text:
                return "py"
            elif "```javascript" in text or "```js" in text:
                return "js"
            elif "```typescript" in text or "```ts" in text:
                return "ts"
            elif "```bash" in text or "```sh" in text:
                return "sh"
            elif "```rust" in text:
                return "rs"
            elif "```go" in text:
                return "go"
            return "py"  # Default to Python

        if response_type == "data":
            if "```json" in text or text.strip().startswith("{"):
                return "json"
            elif "```yaml" in text or "```yml" in text:
                return "yaml"
            elif "```csv" in text:
                return "csv"
            return "json"  # Default to JSON

        # Auto-detect from content
        if "```python" in text:
            return "py"
        elif "```json" in text:
            return "json"
        elif "```" in text:
            return "md"  # Markdown for mixed content with code
        elif self.TABLE_PATTERN.search(text):
            return "md"  # Markdown for tables

        return "txt"

    def _speak_response(self, text: str) -> bool:
        """Synthesize and play response via TTS.

        Args:
            text: The text to speak. Will be truncated at sentence boundary if too long.

        Returns:
            Whether the speaker reported that the audio played. False when TTS is off,
            synthesis is empty or fails, or the sink refuses (#1435: a speak tool must
            never report a silent drop as spoken).
        """
        if not self.audio_enabled:
            return False

        # Truncate at sentence boundary for natural speech
        max_speech_length = 1000
        if len(text) > max_speech_length:
            speech_text = _truncate_at_sentence(text, max_speech_length)
            self.logger.debug("Truncated speech from %d to %d chars at sentence boundary", len(text), len(speech_text))
        else:
            speech_text = text

        try:
            # Synthesize audio
            audio = self.tts.synthesize(speech_text)

            if len(audio) == 0:
                self.logger.debug("TTS returned empty audio for text: %s", speech_text[:50])
                return False

            # The sink resamples to its device's rate, so hand it the TTS's own rate. (A
            # pre-resample here to a guessed 16 kHz returned float64 on the int16 scale,
            # which hard-clipped on the device: pulse's R6, #1435.)
            played = bool(self.speaker(audio, sample_rate=int(self.tts.sample_rate)))
            if not played:
                self.logger.warning("Speech did not play: the speaker refused (see its warning)")
            return played

        except Exception as e:
            self.logger.warning("TTS failed: %s", e)
            return False


_TTS_TRUTHY = ("1", "true", "yes", "on")
_DEFAULT_VOICE = "en_US-lessac-medium"


class SpeechOutputUnavailable(RuntimeError):
    """Spoken responses were requested (``--tts`` or ``MAXIM_TTS_ENABLED``) but cannot play.

    Raised at startup by :func:`build_speech_output`, so a session never runs believing it speaks
    when it cannot (#1437). ``fix_hint`` names what to install or download.
    """

    def __init__(self, reason: str, fix_hint: str) -> None:
        self.fix_hint = fix_hint
        super().__init__(f"Spoken responses were requested but cannot play: {reason}. {fix_hint}")


def build_speech_output(
    *,
    requested: bool,
    model_name: str | None,
    sink: Callable[..., bool] | None,
    log: logging.Logger,
    sink_plays_locally: bool = False,
) -> tuple[Any | None, Callable[..., bool] | None]:
    """The one place spoken responses are set up, for every entry point (#1437).

    Speech is on when ``requested`` (the ``--tts`` flag) or ``MAXIM_TTS_ENABLED`` is truthy; otherwise
    this returns ``(None, None)`` and logs nothing. When on, it returns ``(tts_engine, speaker)``
    ready for :class:`ResponseOutput`:

    - the voice is ``model_name``, else ``MAXIM_TTS_MODEL``, else the default voice;
    - the speaker is ``sink`` when the caller has one (the robot's device sink,
      ``MediaLoopMixin.speak``), else this machine's local speakers (:class:`LocalSpeaker`);
    - local speakers are checked whenever speech will play on them: no sink, ``MAXIM_TTS_LOCAL``,
      or ``sink_plays_locally`` (``utils.audio.device_speech_plays_locally``: no SDK handle, or a
      ``no_media`` backend);
    - the voice is loaded here, so a corrupt model refuses at start, not at the first sentence.

    Raises:
        SpeechOutputUnavailable: the ``tts`` extra (piper) is not installed, the voice model is
            missing or will not load, or speech would play locally and no local speaker can. It
            never logs "enabled" for a path that cannot play. ``cli.main`` renders it with its
            ``fix_hint`` and exits 2.
    """
    import os

    enabled = requested or os.environ.get("MAXIM_TTS_ENABLED", "").strip().lower() in _TTS_TRUTHY
    if not enabled:
        return None, None
    try:
        require_optional_dependency("piper", feature="spoken responses (--tts / MAXIM_TTS_ENABLED)")
    except OptionalDependencyError as e:
        raise SpeechOutputUnavailable("the tts extra (piper) is not installed", e.fix_hint) from e
    from maxim.models.audio.tts import TTSEngine

    voice = model_name or os.environ.get("MAXIM_TTS_MODEL") or _DEFAULT_VOICE
    engine = TTSEngine(model_name=voice)
    if not engine.is_available:
        raise SpeechOutputUnavailable(
            f"the voice model {voice!r} is not downloaded",
            "Download it with `python -m maxim.models.download --tts`.",
        )
    try:
        engine.load()  # also sets the voice's real sample rate, which the local speaker needs
    except Exception as e:  # onnxruntime's load errors (InvalidProtobuf, ...) derive from Exception; re-raised typed
        raise SpeechOutputUnavailable(
            f"the voice model {voice!r} could not be loaded ({type(e).__name__}: {e})",
            "Re-download it with `python -m maxim.models.download --tts`.",
        ) from e
    from maxim.utils.audio import tts_local_requested

    speaker = sink
    prefer_local = tts_local_requested()
    where = "local speakers (MAXIM_TTS_LOCAL)" if prefer_local else "the robot's speaker"
    if speaker is None or prefer_local or sink_plays_locally:
        from maxim.utils.audio import LocalSpeaker

        local = LocalSpeaker(sample_rate=int(engine.sample_rate))
        if not local.is_available:
            raise SpeechOutputUnavailable(
                "no local speaker can play (sounddevice, the PortAudio library, or an output device is missing)",
                "Install it with `pip install 'pymaxim[tts]'`; on Linux also the PortAudio library "
                "(e.g. `apt install libportaudio2`).",
            )
        if speaker is None:
            speaker = local
            where = "local speakers"
        elif not prefer_local:
            where = "local speakers (no robot SDK handle, or a no_media backend)"
    log.info("TTS enabled with voice %s, speaking through %s", voice, where)
    return engine, speaker


# Convenience function for quick CLI output
def print_response(text: str) -> None:
    """Print a simple response to CLI without ResponseOutput instance."""
    _print_agent_response(f"[Maxim]: {text}")
