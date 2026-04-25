"""Narrator — LLM-driven narrative generation for generative campaigns.

Provides the two-call approach (Option C):
1. Decision call (JSON): phase, scene_type, notes, done?
2. Generation call (plain text): narrative scene

Also provides the single-call fallback (Option B) for small models.
"""

from __future__ import annotations

import logging
from typing import Any

from maxim.simulation.arcs import NarrativeArc

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Fallback scenes — immersive alternatives when LLM generation fails.
# Phase names are free-form strings (not an enum), so we match on
# substrings with a robust default for unrecognized phases.
# ---------------------------------------------------------------------------

_FALLBACK_SCENES: dict[str, str] = {
    "intro": "A new scene begins to unfold before you...",
    "establish": "The world around you takes shape, details sharpening...",
    "seed": "Something stirs in the world around you, hinting at what's to come...",
    "conflict": "Tension hangs thick in the air...",
    "escalat": "The stakes rise around you...",
    "boundary": "You feel the weight of an important moment pressing in...",
    "trust": "A sense of quiet understanding settles between you and those nearby...",
    "reflect": "A quiet moment settles, inviting contemplation...",
    "recall": "Something stirs in the back of your mind...",
    "practice": "The path ahead demands careful attention...",
    "transfer": "What you've learned begins to take on new meaning...",
    "reinforc": "The familiar rhythm returns, stronger now...",
    "interfere": "An unexpected disruption cuts through the calm...",
}
_FALLBACK_DEFAULT = "The story continues to unfold around you..."


def _fallback_for_phase(phase: str) -> str:
    """Return immersive fallback text for a given phase name."""
    phase_lower = phase.lower()
    for key, text in _FALLBACK_SCENES.items():
        if key in phase_lower:
            return text
    return _FALLBACK_DEFAULT


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

NARRATOR_SYSTEM_DECISION = """\
You are the narrative director for a cognitive agent simulation. You control the \
pacing and structure of the story being told to an agent-under-test (AUT).

{arc_instructions}

RULES:
- You decide which phase the narrative is in and what type of scene to generate next.
- Track which phases are complete and which remain.
- Adapt pacing based on the AUT's responses — if the AUT engages deeply, \
  spend more turns; if it seems confused, move on.
- Set "done": true ONLY when all phases are complete and the simulation goal \
  is fully explored.

{memory_context}

Respond with ONLY valid JSON:
{{"phase": "<current_phase>", "scene_type": "<description>", "notes": "<brief_plan>", "done": false}}
"""

NARRATOR_SYSTEM_GENERATION = """\
You are a vivid, immersive narrator. Your job is to generate the next scene in \
an ongoing narrative for a cognitive agent called "{aut_name}".

STYLE:
- Write in second person ("You see...", "The merchant turns to you...")
- Be vivid but concise — 3-5 sentences per scene
- Include sensory details (sounds, smells, visual descriptions)
- End each scene with something that invites a response from the AUT
- Never break the fourth wall or reference the simulation

CURRENT PHASE: {phase}
SCENE DIRECTION: {scene_direction}
{story_context}

Output ONLY the narrative text. No JSON, no explanations, no meta-commentary.
"""

NARRATOR_SYSTEM_SINGLE = """\
You are a narrator running a cognitive simulation for an agent called "{aut_name}".
Your goal: {goal}

{arc_instructions}

For each turn, generate a vivid narrative scene (3-5 sentences, second person) \
that follows the current phase of the arc. Track your progress through the phases.

After generating the scene, decide:
- Which phase you're in
- Whether to advance to the next phase
- Whether the simulation goal is met (set done=true)

{memory_context}

Respond with ONLY valid JSON:
{{"phase": "<phase>", "narrative": "<your scene text>", "done": false}}
"""


# ---------------------------------------------------------------------------
# Narrator class
# ---------------------------------------------------------------------------


class Narrator:
    """Generates narrative scenes via LLM for generative campaigns.

    Supports two modes:
    - Two-call (Option C): separate decision + generation calls
    - Single-call (Option B): combined decision + narrative in one call

    Parameters
    ----------
    llm : LLMRouter or similar
        LLM for generating text and JSON.
    arc : NarrativeArc
        The narrative arc guiding the structure.
    aut_name : str
        Display name for the agent-under-test.
    goal : str
        The simulation goal.
    memory_context : str
        Optional context from AdaptivePlanner/PlanningContext.
    """

    def __init__(
        self,
        llm: Any,
        arc: NarrativeArc,
        aut_name: str = "AUT",
        goal: str = "",
        memory_context: str = "",
    ) -> None:
        self._llm = llm
        self._arc = arc
        self._aut_name = aut_name
        self._goal = goal
        self._memory_context = memory_context
        self._phase_idx: int = 0
        self._turns_in_phase: int = 0
        self._total_turns: int = 0
        self._story_context: str = ""
        self._done: bool = False
        self._decisions: list[dict[str, Any]] = []

    @property
    def current_phase(self) -> str:
        if self._phase_idx < len(self._arc.phases):
            return self._arc.phases[self._phase_idx].name
        return "complete"

    @property
    def is_done(self) -> bool:
        return self._done

    @property
    def total_turns(self) -> int:
        return self._total_turns

    # -- two-call approach (Option C) ---------------------------------------

    def decide(self, last_aut_response: str = "") -> dict[str, Any]:
        """Decision call — determine phase, scene type, and completion.

        Returns dict with: phase, scene_type, notes, done
        """
        if self._done:
            return {"phase": "complete", "scene_type": "none", "notes": "", "done": True}

        # Auto-advance phase if turn budget exceeded
        if self._phase_idx < len(self._arc.phases):
            phase = self._arc.phases[self._phase_idx]
            if self._turns_in_phase >= phase.turns_max:
                self._phase_idx += 1
                self._turns_in_phase = 0

        # Check if all phases exhausted
        if self._phase_idx >= len(self._arc.phases):
            self._done = True
            return {"phase": "complete", "scene_type": "none", "notes": "", "done": True}

        system = NARRATOR_SYSTEM_DECISION.format(
            arc_instructions=self._arc.to_narrator_instructions(),
            memory_context=self._memory_context,
        )
        user = self._build_decision_context(last_aut_response)

        try:
            result = self._llm.generate_json(
                f"{system}\n\n{user}",
                max_tokens=200,
            )
            if isinstance(result, dict):
                decision = {
                    "phase": result.get("phase", self.current_phase),
                    "scene_type": result.get("scene_type", "general"),
                    "notes": result.get("notes", ""),
                    "done": bool(result.get("done", False)),
                }
                self._decisions.append(decision)
                if decision["done"]:
                    self._done = True
                return decision
        except Exception as e:
            log.warning("Narrator decision call failed, using defaults: %s", e)

        # Fallback: advance through phases mechanically
        return {
            "phase": self.current_phase,
            "scene_type": "general",
            "notes": "",
            "done": False,
        }

    def generate(self, decision: dict[str, Any], last_aut_response: str = "") -> str:
        """Generation call — produce narrative text for the decided scene.

        Returns plain narrative text (no JSON).
        """
        phase = decision.get("phase", self.current_phase)
        scene_direction = decision.get("notes", decision.get("scene_type", ""))

        system = NARRATOR_SYSTEM_GENERATION.format(
            aut_name=self._aut_name,
            phase=phase,
            scene_direction=scene_direction,
            story_context=f"STORY SO FAR:\n{self._story_context}" if self._story_context else "",
        )
        user = ""
        if last_aut_response:
            user = f"The AUT's last response: {last_aut_response}\n\nGenerate the next scene."
        else:
            user = "Generate the opening scene."

        try:
            text = self._llm.generate_text(
                f"{system}\n\n{user}",
                max_tokens=500,
            )
            if text:
                self._turns_in_phase += 1
                self._total_turns += 1
                # Update compressed story context (keep last ~200 tokens worth)
                self._update_story_context(text, last_aut_response)
                return text.strip()
        except Exception as e:
            log.warning("Narrator generation failed (phase=%s): %s", phase, e)

        # Fallback: immersive phase-aware text (no brackets, no meta-commentary)
        self._turns_in_phase += 1
        self._total_turns += 1
        return _fallback_for_phase(phase)

    # -- single-call approach (Option B) ------------------------------------

    def generate_single(self, last_aut_response: str = "") -> tuple[str, dict[str, Any]]:
        """Single-call approach — decision + narrative in one LLM call.

        Returns (narrative_text, decision_dict).
        """
        if self._done:
            return ("The story has concluded.", {"phase": "complete", "done": True})

        # Auto-advance phase
        if self._phase_idx < len(self._arc.phases):
            phase = self._arc.phases[self._phase_idx]
            if self._turns_in_phase >= phase.turns_max:
                self._phase_idx += 1
                self._turns_in_phase = 0

        if self._phase_idx >= len(self._arc.phases):
            self._done = True
            return ("The story has concluded.", {"phase": "complete", "done": True})

        system = NARRATOR_SYSTEM_SINGLE.format(
            aut_name=self._aut_name,
            goal=self._goal,
            arc_instructions=self._arc.to_narrator_instructions(),
            memory_context=self._memory_context,
        )
        user = self._build_decision_context(last_aut_response)

        try:
            result = self._llm.generate_json(
                f"{system}\n\n{user}",
                max_tokens=600,
            )
            if isinstance(result, dict):
                narrative = result.get("narrative", "The story continues.")
                done = bool(result.get("done", False))
                if done:
                    self._done = True
                self._turns_in_phase += 1
                self._total_turns += 1
                self._update_story_context(narrative, last_aut_response)
                decision = {
                    "phase": result.get("phase", self.current_phase),
                    "done": done,
                }
                self._decisions.append(decision)
                return (narrative, decision)
        except Exception as e:
            log.warning("Narrator single-call failed (phase=%s): %s", self.current_phase, e)

        # Fallback: immersive phase-aware text (no brackets, no meta-commentary)
        self._turns_in_phase += 1
        self._total_turns += 1
        return (_fallback_for_phase(self.current_phase), {"phase": self.current_phase, "done": False})

    # -- internal -----------------------------------------------------------

    def _build_decision_context(self, last_aut_response: str) -> str:
        parts = [f"Goal: {self._goal}"]
        parts.append(f"Turn: {self._total_turns + 1}")
        parts.append(f"Current phase: {self.current_phase} (turn {self._turns_in_phase + 1} in phase)")

        phases_done = [self._arc.phases[i].name for i in range(self._phase_idx)]
        if phases_done:
            parts.append(f"Completed phases: {', '.join(phases_done)}")

        phases_remaining = [self._arc.phases[i].name for i in range(self._phase_idx, len(self._arc.phases))]
        if phases_remaining:
            parts.append(f"Remaining phases: {', '.join(phases_remaining)}")

        if last_aut_response:
            # Truncate to ~200 chars
            truncated = last_aut_response[:200] + "..." if len(last_aut_response) > 200 else last_aut_response
            parts.append(f"AUT's last response: {truncated}")

        if self._story_context:
            parts.append(f"Story so far: {self._story_context}")

        return "\n".join(parts)

    def _count_words(self, text: str) -> int:
        """Count words, using LLM tokenizer if available, else split()."""
        if hasattr(self._llm, "get_token_counter"):
            try:
                counter = self._llm.get_token_counter()
                return int(counter.count_tokens(text))
            except Exception:
                pass
        return len(text.split())

    def _update_story_context(self, narrative: str, aut_response: str) -> None:
        """Maintain compressed story context (~200 tokens / ~150 words)."""
        entry = f"[Turn {self._total_turns}] {narrative[:100]}"
        if aut_response:
            entry += f" AUT: {aut_response[:50]}"

        if self._story_context:
            self._story_context = f"{self._story_context}\n{entry}"
        else:
            self._story_context = entry

        # Cap at ~150 words (≈ 200 tokens). Word count tracks token count
        # within ~20% for English text, vs char count which diverges by
        # 2-4x on code/unicode.
        if self._count_words(self._story_context) > 150:
            lines = self._story_context.split("\n")
            while self._count_words("\n".join(lines)) > 150 and len(lines) > 2:
                lines.pop(0)
            self._story_context = "\n".join(lines)

    def export_decisions(self) -> list[dict[str, Any]]:
        """Return all decision records for YAML export."""
        return list(self._decisions)


# ---------------------------------------------------------------------------
# Scene manifest generation (pre-trigger)
# ---------------------------------------------------------------------------

_MANIFEST_SYSTEM = """\
You are a world builder for a cognitive agent simulation. Given a \
simulation goal, list the key physical entities that should populate \
the scene.

RULES:
- List 5-10 entities, one per line
- Include creatures, weapons, items, NPCs, and environmental features
- For each entity, write a short physical description (5-10 words)
- Focus on entities the agent can physically interact with or observe
- Do NOT list abstract concepts, emotions, or story beats
- Do NOT list the agent itself or its body

Example output for "explore a haunted dungeon":
a rusty iron gate blocking the entrance
a flickering torch mounted on the stone wall
a giant spider lurking in the shadows
a dusty wooden chest in the corner
a skeleton guard with a rusted sword
a deep pit trap hidden under loose stones
a carved stone altar glowing faintly
"""


def generate_scene_manifest(llm: Any, goal: str) -> str:
    """Generate a scene entity manifest from a simulation goal.

    Makes a single LLM call to produce a natural-language list of
    entities for the scenario. The output is intended for
    :meth:`ImaginationTrigger.process_manifest`.

    Returns raw text (one entity per line). Returns empty string on
    failure — the caller should proceed without pre-triggering.

    Args:
        llm: LLM router or similar (must have ``generate_json``).
        goal: The simulation goal string.
    """
    prompt = f"{_MANIFEST_SYSTEM}\n\nSimulation goal: {goal}\n\nList the entities:"

    try:
        # Use generate_json since LLMRouter doesn't have generate_text.
        # Ask for JSON with an "entities" key to get structured output,
        # then join into text for the imagination pipeline.
        result = llm.generate_json(
            f'{prompt}\n\nRespond with JSON: {{"entities": ["entity 1 description", "entity 2 description", ...]}}',
            max_tokens=300,
        )
        if isinstance(result, dict):
            entities = result.get("entities", [])
            if entities:
                text = "\n".join(str(e) for e in entities)
                log.info("Scene manifest generated: %d entities", len(entities))
                return text
        elif isinstance(result, str) and result.strip():
            log.info("Scene manifest generated: %d chars", len(result.strip()))
            return result.strip()
    except Exception as e:
        log.warning("Scene manifest generation failed: %s", e)

    return ""
