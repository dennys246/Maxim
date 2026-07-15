# tools/narrative.py
"""Narrative action tools for simulation mode.

These tools let the AUT interact with the narrative environment
(speak in-world, reason explicitly) as opposed to robot-specific tools
like ``speak`` (TTS) or ``focus_interests`` (camera tracking).

Registered on the AUT's tool registry in sim mode only.
"""

from __future__ import annotations

import threading
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

    When a ``BioEnrichmentPipeline`` is wired (L1), the thought is
    enriched with hippocampal memories, NAc predictions, ATL concepts,
    and SEM affordances.  The enriched response is returned as part of
    the tool output so the LLM's follow-up cycle can incorporate
    bio-system associations into its next action.

    **L2 — Active deliberation:**

    Tracks consecutive think calls and injects convergence/termination
    signals to prevent think loops.

    - **Convergence detection:** When keyword overlap between the current
      thought and recent thoughts exceeds 0.8, a nudge is injected.
    - **Hard cap:** After ``max_deliberation_hops`` (default 3) consecutive
      thinks without an action, a termination signal is injected.
    - **NAc gating:** Declining novelty in enrichment responses is exposed
      via declining valence, biasing the LLM toward action.

    Call ``reset_deliberation()`` when a non-think action fires (done by
    the agent loop's tool dispatch when any other tool executes).
    """

    name = "think"
    description = (
        "Pause and reason about the current situation before acting. "
        "Use when you need to consider options, recall context, or plan "
        "your next move. This does not produce any visible action — but "
        "your bio-systems will surface relevant memories and predictions."
    )
    # JSONSchema with empty ``required`` so the dispatch validator does not
    # block calls like ``think(text="...")`` or ``think({})``. ``execute()``
    # already accepts ``thought`` / ``text`` / ``prompt`` aliases and returns
    # a typed "Empty thought" error when none are supplied. Under the legacy
    # custom format (``{"thought": str}``) the validator raised
    # ``Missing required input: thought`` before the alias fallback ran,
    # producing the silent failures the cradle smoke surfaced.
    # See docs/plans/archive/cradle_activation_fixes.md (Finding B).
    input_schema = {
        "type": "object",
        "properties": {
            "thought": {
                "type": "string",
                "description": "Your reasoning about the current situation.",
            },
        },
        "required": [],
    }

    def __init__(
        self,
        *,
        pipeline: Any | None = None,
        max_deliberation_hops: int = 3,
    ) -> None:
        super().__init__()
        self._pipeline = pipeline  # BioEnrichmentPipeline (optional)
        self._max_hops = max_deliberation_hops
        # Deliberation state (L2) — reset when any non-think tool fires.
        # Lock protects against theoretical concurrent access (single-executor
        # semantics make this unlikely, but the lock is cheap insurance).
        self._lock = threading.Lock()
        self._consecutive_thinks: int = 0
        self._recent_thought_keywords: list[set[str]] = []

    def reset_deliberation(self) -> None:
        """Reset deliberation state. Called when a non-think action fires."""
        with self._lock:
            self._consecutive_thinks = 0
            self._recent_thought_keywords.clear()

    def execute(self, **kwargs: Any) -> ToolResult:
        # Accept common LLM param aliases for the thought content
        thought = kwargs.get("thought") or kwargs.get("text") or kwargs.get("prompt") or ""
        if not thought:
            return ToolResult(success=False, error="Empty thought")

        current_keywords = self._extract_thought_keywords(thought)
        output: dict[str, Any] = {"thought": thought, "visible": False}

        with self._lock:
            # Track deliberation state
            self._consecutive_thinks += 1
            hop = self._consecutive_thinks

            # Enrich via bio-systems (bypass novelty gate — explicit thoughts
            # always deserve enrichment).
            if self._pipeline is not None:
                try:
                    from maxim.integration.bio_enrichment import EnrichmentContext

                    context = EnrichmentContext(
                        recent_thoughts=tuple(" ".join(kw) for kw in self._recent_thought_keywords[-3:]),
                    )
                    result = self._pipeline.enrich(thought, context=context, bypass_gate=True)
                    if result is not None:
                        formatted = self._pipeline.format_thought_response(result)
                        if formatted:
                            output["bio_response"] = formatted
                            output["valence"] = result.valence
                except Exception:
                    pass  # Enrichment is best-effort; tool still succeeds

            # L2: Convergence detection — inject nudge if thinking in circles
            convergence_signal = self._check_convergence(current_keywords)
            if convergence_signal:
                output["deliberation_signal"] = convergence_signal

            # L2: Hard cap — inject termination signal after max hops
            if hop >= self._max_hops:
                output["deliberation_signal"] = (
                    "You've deliberated enough — time to act. Choose an action based on what you've learned."
                )

            # Track for future convergence detection
            self._recent_thought_keywords.append(current_keywords)
            # Keep only last 5 thought keyword sets
            if len(self._recent_thought_keywords) > 5:
                self._recent_thought_keywords = self._recent_thought_keywords[-5:]

        output["deliberation_hop"] = hop

        return ToolResult(
            success=True,
            output=output,
        )

    def _check_convergence(self, current_keywords: set[str]) -> str:
        """Detect if the agent is thinking about the same things repeatedly.

        Uses keyword overlap (Jaccard similarity) as a lightweight proxy
        for semantic similarity. Returns a nudge string or empty.

        Requires at least 3 keywords in the current thought to avoid
        false positives on short/generic thoughts like "run" or "yes".
        """
        if not self._recent_thought_keywords:
            return ""

        # Minimum keyword count to avoid false positives on short thoughts
        if len(current_keywords) < 3:
            return ""

        # Check overlap with each of the last 3 thoughts
        for prev_keywords in self._recent_thought_keywords[-3:]:
            if len(prev_keywords) < 3:
                continue
            overlap = len(current_keywords & prev_keywords)
            union = len(current_keywords | prev_keywords)
            if union > 0 and overlap / union >= 0.8:
                return (
                    "Your thoughts are converging — you're circling similar ideas. "
                    "You likely have enough context to act now."
                )
        return ""

    @staticmethod
    def _extract_thought_keywords(thought: str) -> set[str]:
        """Extract meaningful keywords from a thought for convergence detection."""
        stop_words = {
            "the",
            "a",
            "an",
            "is",
            "are",
            "was",
            "were",
            "to",
            "of",
            "in",
            "on",
            "at",
            "for",
            "and",
            "or",
            "but",
            "not",
            "it",
            "i",
            "my",
            "you",
            "this",
            "that",
            "with",
            "from",
            "by",
            "do",
            "does",
            "how",
            "what",
            "can",
            "could",
            "would",
            "should",
            "will",
            "about",
            "think",
            "need",
            "want",
            "know",
            "maybe",
            "perhaps",
            "might",
            "seems",
        }
        words = thought.lower().replace("_", " ").replace("-", " ").split()
        return {w for w in words if len(w) > 2 and w not in stop_words}


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
