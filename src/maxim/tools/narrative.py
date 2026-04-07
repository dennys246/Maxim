# tools/narrative.py
"""Narrative action tools for simulation mode.

These tools let the AUT interact with the narrative environment
(speak in-world, reason explicitly) as opposed to robot-specific tools
like ``speak`` (TTS) or ``focus_interests`` (camera tracking).

Registered on the AUT's tool registry in sim mode only.
"""

from __future__ import annotations

from typing import Any

from maxim.tools.base import Tool, ToolResult


class SayTool(Tool):
    """Say something aloud in the current scene.

    Distinct from ``respond`` (talks to the CLI user) and ``speak`` (TTS).
    ``say`` is an in-world narrative action — speaking to NPCs, answering
    riddles, or saying passwords/names when prompted.

    The text becomes part of the AUT's action history, visible to the
    orchestrator via ``observe_actions``, and is captured by hippocampus
    through the normal episodic memory path.
    """

    name = "say"
    description = (
        "Say something aloud in the current scene. Use for speaking to "
        "NPCs, answering riddles, or saying passwords/names when prompted."
    )
    input_schema = {
        "text": str,
    }

    def execute(self, **kwargs: Any) -> ToolResult:
        # Accept common LLM param aliases for the text to say
        text = kwargs.get("text") or kwargs.get("message") or kwargs.get("phrase") or ""
        if not text:
            return ToolResult(success=False, error="Nothing to say")
        return ToolResult(
            success=True,
            output={"said": text, "mode": "narrative"},
        )


class ThinkTool(Tool):
    """Pause and reason about the current situation before acting.

    An explicit "think before acting" step that doesn't produce an
    external action. Useful for small models (7B) that tend to jump
    to action without reasoning.

    The thought is captured in hippocampus as episodic memory through
    the normal action-store path in the agent loop.
    """

    name = "think"
    description = (
        "Pause and reason about the current situation before acting. "
        "Use when you need to consider options, recall context, or plan "
        "your next move. This does not produce any visible action."
    )
    input_schema = {
        "thought": str,
    }

    def execute(self, **kwargs: Any) -> ToolResult:
        # Accept common LLM param aliases for the thought content
        thought = kwargs.get("thought") or kwargs.get("text") or kwargs.get("prompt") or ""
        if not thought:
            return ToolResult(success=False, error="Empty thought")
        return ToolResult(
            success=True,
            output={"thought": thought, "visible": False},
        )
