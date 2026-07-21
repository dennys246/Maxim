"""Plan-to-arc translation — bridges AdaptivePlanner output to narrative arcs.

Translates ``PlanCandidate`` decompositions into ``NarrativeArc`` format,
enriches narrator context with ``PlanningContext`` memory signals, and
manages bridge-and-compress for multi-arc continuation (large models).
"""

from __future__ import annotations

import logging
from typing import Any

from maxim.simulation.arcs import BUILTIN_ARCS, NarrativeArc, NarrativePhase

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Plan-to-arc translation
# ---------------------------------------------------------------------------


def translate_plan_to_arc(
    plan_actions: list[dict[str, Any]],
    goal: str,
    *,
    builtin_arcs: dict[str, NarrativeArc] | None = None,
    narrative_profile: str = "immersive",
) -> NarrativeArc:
    """Convert an AdaptivePlanner decomposition into a NarrativeArc.

    Each action in the plan becomes a narrative phase. If the plan
    structure matches a builtin arc, that template is used (enriched
    with planner context). Otherwise, a custom arc is generated from
    the decomposition.

    Parameters
    ----------
    plan_actions : list[dict]
        Actions from ``PlanCandidate.actions``. Each should have at least
        a ``"description"`` or ``"goal"`` key for the phase instruction.
    goal : str
        The simulation goal.
    builtin_arcs : dict, optional
        Available builtin arcs. Defaults to ``BUILTIN_ARCS``.
    narrative_profile : str
        Narrative style hint (``"immersive"``, ``"clinical"``, ``"terse"``).

    Returns
    -------
    NarrativeArc
        Arc with phases derived from the plan.
    """
    arcs = builtin_arcs or BUILTIN_ARCS

    # Try to match plan structure to a builtin arc
    matched = _match_plan_to_builtin(plan_actions, goal, arcs)
    if matched is not None:
        log.info("Plan matched builtin arc: %s", matched.name)
        return matched

    # Generate custom arc from plan decomposition
    phases: list[NarrativePhase] = []
    for i, action in enumerate(plan_actions):
        phase_name = action.get("phase", action.get("name", f"phase_{i + 1}"))
        description = action.get("description", action.get("goal", "Continue the narrative."))
        sub_goals = action.get("sub_goals", [])

        instruction = description
        if sub_goals:
            sub_goal_text = "; ".join(str(sg) for sg in sub_goals[:3])
            instruction += f" Include: {sub_goal_text}."

        turns_min = action.get("turns_min", 1)
        turns_max = action.get("turns_max", 3)

        phases.append(
            NarrativePhase(
                name=phase_name,
                instruction=instruction,
                turns_min=turns_min,
                turns_max=turns_max,
            )
        )

    if not phases:
        # Fallback: single exploration phase
        phases = [
            NarrativePhase(
                name="exploration",
                instruction=f"Explore the goal: {goal}",
                turns_min=3,
                turns_max=10,
            )
        ]

    return NarrativeArc(
        name=f"planner_{goal[:30].replace(' ', '_')}",
        description=f"Planner-generated arc for: {goal}",
        phases=phases,
        source="planner",
    )


def _match_plan_to_builtin(
    plan_actions: list[dict[str, Any]],
    goal: str,
    arcs: dict[str, NarrativeArc],
) -> NarrativeArc | None:
    """Try to match a plan's structure to a builtin arc.

    Matching criteria: if the plan's phase descriptions contain keywords
    that map to a specific builtin arc's phase names, and the phase count
    is compatible, use that builtin.
    """
    goal_lower = goal.lower()
    plan_text = " ".join(str(a.get("description", "")) + " " + str(a.get("phase", "")) for a in plan_actions).lower()

    best_arc = None
    best_score = 0

    # WORD-level matching (not substring): the old `word in goal_lower` /
    # `word in plan_text` matched a word as a *substring*, so short words ("a",
    # "co") false-matched inside longer words (e.g. "a" inside "scenario"),
    # letting any verbose arc description over-score an unrelated plan. Require a
    # full-word match of a word longer than 3 chars.
    goal_words = {w for w in goal_lower.split() if len(w) > 3}
    plan_words = {w for w in plan_text.split() if len(w) > 3}
    plan_phases = {str(a.get("phase", "")).lower() for a in plan_actions}

    for arc_name, arc in arcs.items():
        # goal keywords overlap arc description (word-level)
        desc_words = {w for w in arc.description.lower().split() if len(w) > 3}
        score = len(goal_words & desc_words)

        # plan phase names overlap arc phase names
        arc_phases = {p.name.lower() for p in arc.phases}
        score += len(plan_phases & arc_phases) * 2

        # plan text mentions arc phase-name words (word-level, meaningful terms)
        arc_phase_words = {w for p in arc.phases for w in p.name.split("_") if len(w) > 3}
        score += len(arc_phase_words & plan_words)

        if score > best_score and score >= 3:
            best_score = score
            best_arc = arc

    return best_arc


# ---------------------------------------------------------------------------
# Narrator context enrichment from PlanningContext
# ---------------------------------------------------------------------------


def enrich_narrator_context(planning_context: Any) -> str:
    """Format PlanningContext memory signals as narrator hints.

    Provides the narrator with memory-informed context about the AUT
    without requiring the narrator to understand the cognitive architecture.

    Parameters
    ----------
    planning_context : PlanningContext
        Memory signals from AdaptivePlanner.

    Returns
    -------
    str
        Formatted context for narrator system prompt.
    """
    if planning_context is None:
        return ""

    parts: list[str] = ["MEMORY CONTEXT FOR NARRATOR:"]

    # NAc prediction
    primary = getattr(planning_context, "primary_prediction", None)
    if primary is not None:
        valence = getattr(primary, "predicted_valence", "unknown")
        confidence = getattr(primary, "confidence", 0)
        parts.append(
            f"- NAc predicts {valence} outcome (confidence {confidence:.2f}). "
            f"{'The AUT may already know what to expect.' if confidence > 0.5 else 'This is relatively novel for the AUT.'}"
        )

    # Situation novelty
    novelty = getattr(planning_context, "situation_novelty", 1.0)
    if novelty < 0.5:
        parts.append(
            f"- The AUT has experienced similar scenarios before (novelty: {novelty:.2f}). "
            f"Make the opening feel familiar but introduce a twist by turn 2."
        )
    elif novelty > 0.8:
        parts.append(
            f"- This is highly novel for the AUT (novelty: {novelty:.2f}). "
            f"Introduce the scenario gradually — don't overwhelm."
        )

    # Ranked skills
    ranked_skills = getattr(planning_context, "ranked_skills", [])
    if ranked_skills:
        skill_names = [s.get("name", "?") for s in ranked_skills[:3]]
        parts.append(
            f"- The AUT has relevant skills: {', '.join(skill_names)}. Consider testing these at a higher difficulty."
        )

    # Reflections
    reflections = getattr(planning_context, "reflections", [])
    if reflections:
        parts.append(
            f"- The AUT has reflected on similar experiences before ({len(reflections)} reflections). "
            f"It may reference past events — acknowledge them if it does."
        )

    # Associated memories
    memories = getattr(planning_context, "associated_memories", [])
    if memories:
        parts.append(
            f"- Hippocampus associates this goal with {len(memories)} prior episodes. "
            f"The AUT's responses may be influenced by these memories."
        )

    # Successful strategies
    strategies = getattr(planning_context, "successful_strategies", [])
    if strategies:
        parts.append(
            f"- Past successful strategies exist ({len(strategies)}). "
            f"The AUT may try to reuse them — vary the scenario to test adaptability."
        )

    if len(parts) <= 1:
        return ""  # No useful context

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Bridge-and-compress (large model multi-arc continuation)
# ---------------------------------------------------------------------------


BRIDGE_PROMPT = """\
You are narrating a continuous story for a cognitive agent called "{aut_name}".

STORY SO FAR (compressed):
{compressed_context}

PREVIOUS ARC: {previous_arc_name}
  - What happened: {arc_summary}
  - AUT's last response: {last_response}

PLANNER CONTEXT FOR NEXT ARC:
{planner_context}

NEXT ARC PHASES:
{next_arc_phases}

Your task:
1. Write a SHORT bridge scene (2-3 sentences) that naturally transitions from \
the previous story to the next arc. Don't break immersion.
2. Update the compressed context to include the previous arc.
3. List which aspects of the goal have been tested so far.

Output JSON:
{{"bridge_narrative": "...", "compressed_context": "...", "tested_aspects": ["..."], "done": false}}
"""


def bridge_and_compress(
    llm: Any,
    *,
    aut_name: str,
    compressed_context: str,
    previous_arc: NarrativeArc,
    arc_summary: str,
    last_aut_response: str,
    planner_context: str,
    next_arc: NarrativeArc,
) -> dict[str, Any]:
    """Generate a bridge scene between two arcs and compress story context.

    Used by large models for multi-arc continuation. Returns dict with:
    - bridge_narrative: transition text to send to AUT
    - compressed_context: updated story summary
    - tested_aspects: what's been covered so far
    - done: whether the narrator thinks the goal is fully explored

    Parameters
    ----------
    llm : LLMRouter
        LLM for bridge generation.
    aut_name : str
        AUT display name.
    compressed_context : str
        Current story summary (~200 tokens).
    previous_arc : NarrativeArc
        The arc that just completed.
    arc_summary : str
        Brief summary of what happened in the previous arc.
    last_aut_response : str
        AUT's last response.
    planner_context : str
        Memory context from AdaptivePlanner for next arc.
    next_arc : NarrativeArc
        The next arc to transition into.

    Returns
    -------
    dict
        Bridge result with narrative, updated context, tested aspects.
    """
    next_phases = "\n".join(f"  {i + 1}. {p.name}: {p.instruction[:80]}..." for i, p in enumerate(next_arc.phases))

    prompt = BRIDGE_PROMPT.format(
        aut_name=aut_name,
        compressed_context=compressed_context or "(beginning of story)",
        previous_arc_name=previous_arc.name,
        arc_summary=arc_summary or "(no summary)",
        last_response=last_aut_response[:200] if last_aut_response else "(no response)",
        planner_context=planner_context or "(no memory context)",
        next_arc_phases=next_phases,
    )

    try:
        result = llm.generate_json(prompt, max_tokens=400)
        if isinstance(result, dict):
            return {
                "bridge_narrative": result.get("bridge_narrative", "The journey continues..."),
                "compressed_context": result.get("compressed_context", compressed_context),
                "tested_aspects": result.get("tested_aspects", []),
                "done": bool(result.get("done", False)),
            }
    except Exception as e:
        log.debug("Bridge-and-compress failed: %s", e)

    # Fallback
    return {
        "bridge_narrative": "The path leads onward, and a new chapter begins...",
        "compressed_context": compressed_context,
        "tested_aspects": [],
        "done": False,
    }
