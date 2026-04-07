# tools/narrative.py
"""Narrative action tools for simulation mode.

These tools let the AUT interact with the narrative environment
(speak in-world, reason explicitly) as opposed to robot-specific tools
like ``speak`` (TTS) or ``focus_interests`` (camera tracking).

Registered on the AUT's tool registry in sim mode only.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from maxim.tools.base import Tool, ToolResult

if TYPE_CHECKING:
    from maxim.simulation.bridge import SimulationBridge


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


class ExamineTool(Tool):
    """Examine an object, person, or feature in the current scene.

    Queries the latest percept delivered through the SimulationBridge for
    mentions of the target, optionally enriched with hippocampal memories.
    This is the narrative counterpart to ``focus_interests`` (camera
    tracking) — the AUT looks closely at something in the story.

    Returns what the AUT observes about the target based on the current
    scene description and any relevant memories.
    """

    name = "examine"
    description = (
        "Examine an object, person, or feature in the current scene. "
        "Returns what you observe about it based on the scene and your memories."
    )
    input_schema = {
        "target": str,
    }

    def __init__(
        self,
        *,
        bridge: SimulationBridge | None = None,
        hippocampus: Any = None,
    ) -> None:
        super().__init__()
        self._bridge = bridge
        self._hippocampus = hippocampus

    def execute(self, **kwargs: Any) -> ToolResult:
        target = kwargs.get("target") or kwargs.get("object") or kwargs.get("text") or ""
        if not target:
            return ToolResult(success=False, error="No target specified to examine")

        observations: list[str] = []
        target_lower = target.lower()

        # Stage 1: scan latest bridge percept for target mentions
        if self._bridge is not None:
            transcript = self._bridge.percept_source._transcript_percepts
            if transcript:
                # Check last few percepts (most recent first)
                for entry in reversed(transcript[-3:]):
                    text = entry.get("cli_input", "") or entry.get("content", "")
                    if not text:
                        continue
                    # Find sentences mentioning the target
                    for sentence in text.replace("\n", " ").split(". "):
                        if target_lower in sentence.lower():
                            observations.append(sentence.strip().rstrip(".") + ".")

        # Stage 2: enrich from hippocampus if available
        if self._hippocampus is not None:
            try:
                memories = self._hippocampus.search_by_content(target, limit=3)
                for m in memories:
                    goal = getattr(getattr(m, "context", None), "goal", "")
                    if goal:
                        observations.append(f"You recall: {goal}")
            except Exception:
                pass

        if not observations:
            return ToolResult(
                success=True,
                output={
                    "target": target,
                    "observation": f"You don't see anything notable about {target}.",
                },
            )

        # Deduplicate while preserving order
        seen: set[str] = set()
        unique: list[str] = []
        for obs in observations:
            if obs not in seen:
                seen.add(obs)
                unique.append(obs)

        return ToolResult(
            success=True,
            output={
                "target": target,
                "observation": " ".join(unique[:5]),
            },
        )
