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

if TYPE_CHECKING:
    import numpy as np


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
        speaker_fn: Callable[[Any], None] | None = None,
    ):
        """Initialize response output manager.

        Args:
            sandbox_path: Path to the sandbox directory.
            logger: Logger instance for debug output.
            tts_engine: Optional TTS engine for audio responses.
            speaker_fn: Optional function to play audio samples.
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
        # Print with visual distinction
        print(f"\n[Maxim]: {text}\n", flush=True)

        # Re-display prompt hint
        sys.stdout.write("maxim> ")
        sys.stdout.flush()

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

        print(f"\n[Maxim]: Response saved to: {relative_path}\n", flush=True)
        sys.stdout.write("maxim> ")
        sys.stdout.flush()

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

    def _speak_response(self, text: str) -> None:
        """Synthesize and play response via TTS.

        Args:
            text: The text to speak. Will be truncated if too long.
        """
        if not self.audio_enabled:
            return

        # Truncate very long text for speech
        max_speech_length = 1000
        if len(text) > max_speech_length:
            speech_text = text[:max_speech_length] + "..."
            self.logger.debug("Truncated speech from %d to %d chars", len(text), max_speech_length)
        else:
            speech_text = text

        try:
            # Synthesize audio
            audio = self.tts.synthesize(speech_text)

            # Resample if needed (TTS typically outputs 22050Hz, Reachy needs 16000Hz)
            if hasattr(self.tts, "sample_rate") and self.tts.sample_rate != 16000:
                from maxim.utils.audio import resample_audio

                audio = resample_audio(audio, self.tts.sample_rate, 16000)

            # Play through speaker
            self.speaker(audio)

        except Exception as e:
            self.logger.warning("TTS failed: %s", e)


# Convenience function for quick CLI output
def print_response(text: str) -> None:
    """Print a simple response to CLI without ResponseOutput instance."""
    print(f"\n[Maxim]: {text}\n", flush=True)
    sys.stdout.write("maxim> ")
    sys.stdout.flush()
