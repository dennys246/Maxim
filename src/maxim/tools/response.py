"""Response tool for LLM to send responses to the user.

This tool allows the LLM to explicitly send responses through the
ResponseOutput system, choosing between CLI and file output.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from maxim.tools.base import Tool, ToolResult

if TYPE_CHECKING:
    from maxim.utils.response_output import ResponseOutput


class RespondTool(Tool):
    """Tool for LLM to send responses to the user.

    Use this tool to communicate information, answers, or explanations
    to the user. The system will automatically choose CLI or file output
    based on response complexity, or you can force a specific mode.

    Examples:
        - Short answer: respond(message="The time is 3:45 PM")
        - Code output: respond(message="```python\\ndef hello():\\n    pass\\n```", output_type="file")
        - Data: respond(message='{"key": "value"}', response_type="data")
    """

    name = "respond"
    description = (
        "Send a response to the user. Use for answering questions, providing information, "
        "or explaining concepts. Short responses appear in CLI; long or complex responses "
        "are saved to files."
    )
    input_schema = {
        "message": str,  # Required
        "response_type": (str, "general"),  # Optional, default "general"
        "output_type": (str, "auto"),  # Optional, default "auto"
        "speak": (bool, False),  # Optional, default False
    }

    def __init__(self, response_output: ResponseOutput) -> None:
        """Initialize with a ResponseOutput instance.

        Args:
            response_output: The ResponseOutput manager to use for output.
        """
        super().__init__()
        self._response_output = response_output

    def execute(
        self,
        message: str,
        response_type: str = "general",
        output_type: str = "auto",
        speak: bool = False,
        **kwargs: Any,
    ) -> ToolResult:
        """Send a response to the user.

        Args:
            message: The response text to send to the user.
            response_type: Type of content - "general", "code", or "data".
                Used for file extension detection.
            output_type: Where to send the response:
                - "auto": Decide based on content length and complexity
                - "cli": Force output to CLI (even if long)
                - "file": Force save to file (even if short)
            speak: If True and TTS is enabled, also speak the response aloud.

        Returns:
            ToolResult with success status and file path if saved to file.
        """
        if not message or not message.strip():
            return ToolResult(
                success=False,
                error="Message cannot be empty",
            )

        file_path = self._response_output.output_response(
            text=message,
            response_type=response_type,
            output_mode=output_type,
            speak=speak,
        )

        output = {
            "delivered": True,
            "mode": "file" if file_path else "cli",
        }
        if file_path:
            output["file_path"] = file_path

        return ToolResult(success=True, output=output)


class SpeakTool(Tool):
    """Tool to speak text aloud via TTS.

    This tool converts text to speech and plays it through the speaker.
    Requires TTS to be enabled in the ResponseOutput instance.
    """

    name = "speak"
    description = (
        "Speak text aloud using text-to-speech. Use this to verbally "
        "communicate with the user. Also prints the text to CLI."
    )
    input_schema = {
        "text": str,  # Required
        "also_print": (bool, True),  # Optional, default True
    }

    def __init__(self, response_output: ResponseOutput) -> None:
        """Initialize with a ResponseOutput instance.

        Args:
            response_output: The ResponseOutput manager with TTS configured.
        """
        super().__init__()
        self._response_output = response_output

    def execute(
        self,
        text: str = "",
        also_print: bool = True,
        **kwargs: Any,
    ) -> ToolResult:
        """Speak text aloud.

        Args:
            text: The text to speak.
            also_print: If True, also print the text to CLI.

        Returns:
            ToolResult with success status.
        """
        # Accept 'message' as alias for 'text' (LLMs often use 'message')
        if not text and kwargs.get("message"):
            text = kwargs["message"]
        if not text or not text.strip():
            return ToolResult(
                success=False,
                error="Text cannot be empty",
            )

        if not self._response_output.audio_enabled:
            # Fall back to just printing if TTS not available
            if also_print:
                self._response_output._output_to_cli(text)
            return ToolResult(
                success=True,
                output={"spoken": False, "printed": also_print, "reason": "TTS not enabled"},
            )

        # Speak (and optionally print)
        if also_print:
            self._response_output._output_to_cli(text)

        self._response_output._speak_response(text)

        return ToolResult(
            success=True,
            output={"spoken": True, "printed": also_print},
        )
