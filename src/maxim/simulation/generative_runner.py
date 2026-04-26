"""Generative campaign runner — LLM-driven narrative simulation.

Entry point: ``run_generative_campaign()`` — generates narrative turns
dynamically following a narrative arc, sends them through the
SimulationBridge, and collects AUT responses.

Supports two approaches (auto-selected by model capability):
- Two-call (Option C): separate decision + generation LLM calls
- Single-call (Option B): combined in one call (small models)

Integrates with:
- SimulationBridge for AUT communication
- SEM entities for structured world objects (swords, NPCs)
- NarrativeArc for phase structure
- YAML export for replay
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from maxim.simulation.arcs import (
    BUILTIN_ARCS,
    NarrativeArc,
    load_arc_yaml,
    select_arc_for_goal,
)
from maxim.simulation.narrator import Narrator

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass
class GenerativeCampaignResult:
    """Result of a generative campaign run."""

    goal: str
    arc_name: str
    total_turns: int
    turns: list[dict[str, Any]] = field(default_factory=list)
    decisions: list[dict[str, Any]] = field(default_factory=list)
    aut_responses: list[str] = field(default_factory=list)
    finish_reason: str = "completed"  # "completed", "max_turns", "error"
    duration_s: float = 0.0
    entity_tools_generated: int = 0
    exported_yaml_path: str | None = None


# ---------------------------------------------------------------------------
# SEM entity loading
# ---------------------------------------------------------------------------


def _load_world_entities(
    arc: NarrativeArc,
    tool_registry: Any = None,
) -> int:
    """Load SEM world_entities from arc metadata if available.

    Returns the number of tools generated.
    """
    world_entities_data = arc.metadata.get("world_entities")
    if not world_entities_data or tool_registry is None:
        return 0

    try:
        from maxim.embodiment.spec import _parse_entity
        from maxim.embodiment.tool_bridge import generate_tools_for_entity

        tools_generated = 0
        for ent_data in world_entities_data:
            entity = _parse_entity(ent_data)
            tools = generate_tools_for_entity(entity, tool_registry)
            tools_generated += len(tools)
        return tools_generated
    except ImportError:
        log.debug("Embodiment module not available — skipping world entities")
        return 0


def _activate_phase_entities(
    phase_entities: tuple[str, ...],
    tool_registry: Any,
    activated: set[str],
    *,
    embodiment: Any = None,
    entity_map: Any = None,
) -> int:
    """Activate entities for a narrative phase via component refs.

    Resolves refs from ComponentRegistry. When a ref is not found,
    falls back to the imagination system to design an entity on the fly.
    Already-activated entities (tracked in *activated* set) are skipped.

    Returns the number of new tools generated.
    """
    if not phase_entities or tool_registry is None:
        return 0

    new_refs = [ref for ref in phase_entities if ref not in activated]
    if not new_refs:
        return 0

    tools_generated = 0
    try:
        from maxim.embodiment.component_registry import ComponentRegistry
        from maxim.embodiment.tool_bridge import generate_tools_for_entity
        from maxim.exceptions import ComponentNotFoundError

        registry = ComponentRegistry()

        for ref in new_refs:
            entity = None
            try:
                entity = registry.instantiate(ref)
                log.info("Phase entity activated from registry: %s", ref)
            except (KeyError, ValueError, ComponentNotFoundError):
                # Not in registry — fall back to imagination
                log.warning(
                    "World entity %r not found in ComponentRegistry — "
                    "imagining one from the name. Create a YAML spec in "
                    "_data/components/ for precise control.",
                    ref,
                )
                entity = _imagine_entity_from_ref(ref)

            if entity is not None:
                tools = generate_tools_for_entity(
                    entity,
                    tool_registry,
                    embodiment=embodiment,
                    entity_map=entity_map,
                )
                tools_generated += len(tools)
                activated.add(ref)
                log.info(
                    "Phase entity %s: %d tools registered",
                    ref,
                    len(tools),
                )
    except ImportError:
        log.debug("Embodiment module not available — skipping phase entities")

    return tools_generated


def _imagine_entity_from_ref(ref: str) -> Any:
    """Design an entity from a component ref string via imagination.

    Extracts a human-readable name from the ref path (e.g.,
    ``"items/cradle_sticky_lever"`` → ``"sticky lever"``) and uses
    the EntityDesigner to create a minimal entity spec.

    Returns an Entity or None if imagination is unavailable.
    """
    # Extract readable name: "items/cradle_sticky_lever" → "sticky lever"
    name = ref.rsplit("/", 1)[-1]
    name = name.replace("cradle_", "").replace("_", " ")

    try:
        from maxim.embodiment.sem import Entity
        from maxim.embodiment.spec import SpecSensor

        # Minimal entity with an examine affordance — imagination would
        # produce richer specs but we keep this as a lightweight fallback
        # that doesn't require an LLM call.
        entity = Entity(
            name=name.replace(" ", "_"),
            entity_type="item",
            metadata={"imagined": True, "source_ref": ref},
        )
        entity.sensors["presence"] = SpecSensor(
            _name="presence",
            _entity_name=entity.name,
            _unit="ratio",
            _schema={"type": "float", "range": [0, 1]},
            _initial=1.0,
            _entity_ref=entity,
        )
        entity.vital_metrics["presence"] = 1.0
        log.info("Imagined minimal entity %r from ref %r", entity.name, ref)
        return entity
    except Exception as exc:
        log.warning("Failed to imagine entity from ref %r: %s", ref, exc)
        return None


# ---------------------------------------------------------------------------
# YAML export
# ---------------------------------------------------------------------------


def export_campaign_yaml(
    session_dir: str | Path,
    result: GenerativeCampaignResult,
) -> Path:
    """Export generated turns + arc metadata as a replayable campaign YAML.

    Saved to ``{session_dir}/generated_campaign.yaml``.
    The format matches the existing campaign YAML schema so ``--campaign``
    can load it directly.
    """
    session_dir = Path(session_dir)
    session_dir.mkdir(parents=True, exist_ok=True)
    path = session_dir / "generated_campaign.yaml"

    # Build campaign YAML structure
    campaign: dict[str, Any] = {
        "name": f"generated_{result.arc_name}",
        "description": f"Auto-generated campaign for: {result.goal}",
        "generated": True,
        "goal": result.goal,
        "arc": result.arc_name,
        "total_turns": result.total_turns,
        "finish_reason": result.finish_reason,
        "duration_s": round(result.duration_s, 2),
    }

    # Convert turns to campaign percept format
    percepts: list[dict[str, Any]] = []
    for turn in result.turns:
        percept: dict[str, Any] = {
            "at": turn.get("turn", 0),
            "text": turn.get("narrative", ""),
            "phase": turn.get("phase", "unknown"),
            "salience": turn.get("salience", 0.8),
            "novelty": turn.get("novelty", 0.7),
        }
        if turn.get("aut_response"):
            percept["aut_response"] = turn["aut_response"]
        percepts.append(percept)

    campaign["percepts"] = percepts

    # Include decisions for analysis
    if result.decisions:
        campaign["decisions"] = result.decisions

    with open(path, "w") as f:
        yaml.dump(campaign, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    log.info("Exported generative campaign to %s", path)
    return path


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------


def run_generative_campaign(
    goal: str,
    bridge: Any,
    llm: Any,
    *,
    arc: NarrativeArc | None = None,
    arc_yaml: str | None = None,
    aut_name: str = "AUT",
    max_turns: int = 50,
    use_two_call: bool = True,
    memory_context: str = "",
    tool_registry: Any = None,
    session_dir: str | None = None,
    planner: Any = None,
    planner_state: Any = None,
) -> GenerativeCampaignResult:
    """Run a generative campaign.

    Parameters
    ----------
    goal : str
        The simulation goal.
    bridge : SimulationBridge
        Bridge for sending narrative to AUT and receiving responses.
    llm : LLMRouter or similar
        LLM for narrator generation.
    arc : NarrativeArc, optional
        Pre-built arc. If None, selected from builtins or YAML.
    arc_yaml : str, optional
        Path to custom arc YAML file.
    aut_name : str
        Display name for the AUT.
    max_turns : int
        Hard cap on total turns.
    use_two_call : bool
        If True, use two-call approach (Option C).
        If False, use single-call (Option B).
    memory_context : str
        Context from AdaptivePlanner/PlanningContext for narrator.
    tool_registry : ToolRegistry, optional
        If provided, SEM world entities generate tools here.
    session_dir : str, optional
        Directory for YAML export.

    Returns
    -------
    GenerativeCampaignResult
        Complete result including turns, decisions, and export path.
    """
    start_time = time.time()

    # Resolve arc — planner-driven if available, else keyword match / YAML
    planning_context_str = memory_context
    if arc is None:
        if arc_yaml:
            arc = load_arc_yaml(arc_yaml)
        elif planner is not None:
            arc, planning_context_str = _resolve_arc_via_planner(
                goal,
                planner,
                planner_state,
                memory_context,
            )
        else:
            arc = select_arc_for_goal(goal)
        if arc is None:
            arc = BUILTIN_ARCS["memory_recall"]

    log.info(
        "Starting generative campaign: goal=%r, arc=%s (%s), turns=%d-%d",
        goal,
        arc.name,
        arc.source,
        arc.total_turns_min,
        arc.total_turns_max,
    )

    # Load SEM world entities if available
    entity_tools = _load_world_entities(arc, tool_registry)

    # Create narrator with enriched memory context
    narrator = Narrator(
        llm=llm,
        arc=arc,
        aut_name=aut_name,
        goal=goal,
        memory_context=planning_context_str,
    )

    # Run turn loop
    turns: list[dict[str, Any]] = []
    aut_responses: list[str] = []
    last_aut_response = ""
    finish_reason = "completed"
    # Track per-phase entity activation
    _activated_entities: set[str] = set()
    _last_phase_idx: int = -1

    for turn_idx in range(max_turns):
        if narrator.is_done:
            break

        # Activate world_entities when entering a new phase
        if narrator._phase_idx != _last_phase_idx:
            _last_phase_idx = narrator._phase_idx
            if narrator._phase_idx < len(arc.phases):
                phase = arc.phases[narrator._phase_idx]
                if phase.world_entities:
                    n_tools = _activate_phase_entities(
                        phase.world_entities,
                        tool_registry,
                        _activated_entities,
                    )
                    if n_tools > 0:
                        log.info(
                            "Phase %r (act=%s): activated %d entity tools",
                            phase.name,
                            phase.act or "?",
                            n_tools,
                        )

        try:
            if use_two_call:
                narrative, decision = _two_call_turn(narrator, last_aut_response)
            else:
                narrative, decision = _single_call_turn(narrator, last_aut_response)
        except Exception as e:
            log.error("Narrator failed at turn %d: %s", turn_idx + 1, e)
            finish_reason = "error"
            break

        if narrator.is_done and not narrative:
            break

        # Send to AUT via bridge
        try:
            bridge_result = bridge.send_and_wait(
                narrative,
                salience=0.8,
                novelty=0.7,
            )
            last_aut_response = bridge_result.get("response", "") or ""
        except Exception as e:
            log.error("Bridge failed at turn %d: %s", turn_idx + 1, e)
            last_aut_response = ""

        # Record turn
        turn_record = {
            "turn": turn_idx + 1,
            "phase": decision.get("phase", narrator.current_phase),
            "narrative": narrative,
            "aut_response": last_aut_response,
            "decision": decision,
            "salience": 0.8,
            "novelty": 0.7,
        }
        turns.append(turn_record)
        aut_responses.append(last_aut_response)

        log.info(
            "Turn %d [%s]: %s",
            turn_idx + 1,
            decision.get("phase", "?"),
            narrative[:80] + "..." if len(narrative) > 80 else narrative,
        )
    else:
        if not narrator.is_done:
            finish_reason = "max_turns"

    duration = time.time() - start_time

    result = GenerativeCampaignResult(
        goal=goal,
        arc_name=arc.name,
        total_turns=len(turns),
        turns=turns,
        decisions=narrator.export_decisions(),
        aut_responses=aut_responses,
        finish_reason=finish_reason,
        duration_s=duration,
        entity_tools_generated=entity_tools,
    )

    # Export to YAML
    if session_dir:
        export_path = export_campaign_yaml(session_dir, result)
        result.exported_yaml_path = str(export_path)

    log.info(
        "Generative campaign complete: %d turns, %s, %.1fs",
        len(turns),
        finish_reason,
        duration,
    )

    return result


# ---------------------------------------------------------------------------
# Turn execution helpers
# ---------------------------------------------------------------------------


def _two_call_turn(
    narrator: Narrator,
    last_aut_response: str,
) -> tuple[str, dict[str, Any]]:
    """Execute one turn using two-call approach (Option C)."""
    decision = narrator.decide(last_aut_response)
    if decision.get("done"):
        return ("", decision)
    narrative = narrator.generate(decision, last_aut_response)
    return (narrative, decision)


def _single_call_turn(
    narrator: Narrator,
    last_aut_response: str,
) -> tuple[str, dict[str, Any]]:
    """Execute one turn using single-call approach (Option B)."""
    return narrator.generate_single(last_aut_response)


# ---------------------------------------------------------------------------
# Planner-driven arc resolution
# ---------------------------------------------------------------------------


def _resolve_arc_via_planner(
    goal: str,
    planner: Any,
    planner_state: Any,
    memory_context: str,
) -> tuple[NarrativeArc | None, str]:
    """Use AdaptivePlanner to decompose goal into a NarrativeArc.

    Returns (arc, enriched_memory_context). If planner fails,
    returns (None, original_memory_context).
    """
    try:
        from maxim.simulation.plan_arc_bridge import (
            enrich_narrator_context,
            translate_plan_to_arc,
        )

        # Ask planner to decompose the goal
        candidates = planner.propose_plans(
            goal={"description": goal, "goal": goal},
            state=planner_state,
            memory=None,
        )

        if not candidates:
            return None, memory_context

        best = candidates[0]
        actions = best.actions if hasattr(best, "actions") else []

        if not actions:
            return None, memory_context

        # Translate plan to arc
        arc = translate_plan_to_arc(actions, goal)

        # Enrich narrator context from planning memory signals
        pctx = getattr(best, "planning_context", None)
        enriched = enrich_narrator_context(pctx)
        if enriched:
            if memory_context:
                enriched = f"{memory_context}\n\n{enriched}"
        else:
            enriched = memory_context

        log.info("Planner decomposed goal into arc: %s (%d phases)", arc.name, len(arc.phases))
        return arc, enriched

    except Exception as e:
        log.debug("Planner arc resolution failed: %s", e)
        return None, memory_context
