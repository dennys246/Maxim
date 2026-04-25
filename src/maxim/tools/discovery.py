"""SEM Tool Discovery — semantic tool discovery + universal sensing.

Provides two tools that replace the flat per-entity tool explosion:

- ``UniversalSenseTool`` — single ``sense(entity_name)`` tool that reads
  all sensors on any named entity, replacing N×M individual read/sense tools.
- ``SenseToolsTool`` — ``sense_tools(query)`` that matches intent
  against entity modulators/affordances and activates the relevant tools
  via I3's scene-scoped mechanism.  Results are ranked by NAc valence
  when available (S2).

Also provides:
- Goal-based top-k selection algorithm for turn-1 affordance visibility.
- LRU eviction for discovered tools that haven't been called recently (S2).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from maxim.tools.base import Tool, ToolOutput

if TYPE_CHECKING:
    from maxim.decisions.nac import NAc
    from maxim.embodiment.component_index import ComponentIndex
    from maxim.embodiment.entity_map import EntityMap
    from maxim.tools.registry import ToolRegistry

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LRU eviction for discovered tools (S2)
# ---------------------------------------------------------------------------

DISCOVERY_LRU_TURNS = 5  # Deactivate after 5 turns of non-use

# Module-level state for LRU tracking.  Keyed by tool name.
_tool_last_used: dict[str, int] = {}
_goal_selected: set[str] = set()  # Exempt from LRU eviction


def mark_tool_used(tool_name: str, turn: int) -> None:
    """Record that a tool was called on this turn.  Called from executor."""
    _tool_last_used[tool_name] = turn


def mark_goal_selected(tool_names: list[str]) -> None:
    """Mark tools as goal-selected (exempt from LRU eviction)."""
    _goal_selected.update(tool_names)


def evict_stale_discoveries(current_turn: int, registry: ToolRegistry) -> list[str]:
    """Deactivate discovered tools not called in DISCOVERY_LRU_TURNS.

    Called at prompt build time before the tool list is assembled.
    Only affects scene-scoped tools (core tools are exempt).
    Goal-selected tools are exempt from eviction.

    Returns list of evicted tool names.
    """
    evicted: list[str] = []
    for tool_name, last_turn in list(_tool_last_used.items()):
        if tool_name in _goal_selected:
            continue
        if current_turn - last_turn > DISCOVERY_LRU_TURNS:
            if registry.get_tool_scene(tool_name) is not None:
                registry.deactivate_tool(tool_name)
                evicted.append(tool_name)
                del _tool_last_used[tool_name]
    if evicted:
        log.info("LRU eviction: deactivated %d tools: %s", len(evicted), evicted)
    return evicted


def reset_discovery_state() -> None:
    """Reset module-level discovery state between sim sessions."""
    _tool_last_used.clear()
    _goal_selected.clear()


# ---------------------------------------------------------------------------
# UniversalSenseTool
# ---------------------------------------------------------------------------


class UniversalSenseTool(Tool):
    """Read all sensors on a named entity.

    Replaces the per-entity ``sense_<name>`` and ``read_<name>_<sensor>``
    tools with a single tool that takes an entity name parameter.
    """

    name = "sense"
    description = "Sense the state of an entity — health, durability, position, etc. Name the entity you want to sense."
    input_schema: dict[str, Any] = {"entity_name": str}

    def __init__(self, *, entity_map: EntityMap) -> None:
        self._entity_map = entity_map
        super().__init__()

    def execute(self, **kwargs: Any) -> Any:
        entity_name = kwargs.get("entity_name", "")
        if not entity_name:
            names = self._entity_map.list_names()
            return ToolOutput(
                success=False,
                error=f"Specify an entity name. Known entities: {', '.join(names)}",
            )
        entity = self._entity_map.resolve(entity_name)
        if entity is None:
            names = self._entity_map.list_names()
            return ToolOutput(
                success=False,
                error=f"Unknown entity: {entity_name}. Known entities: {', '.join(names)}",
            )
        readings = entity.read_all_sensors()
        result = {}
        for sensor_name, reading in readings.items():
            result[sensor_name] = {
                "value": reading.value,
                "unit": reading.unit,
            }
        return ToolOutput(success=True, output={"entity": entity.name, "sensors": result})


# ---------------------------------------------------------------------------
# SensePresenceTool
# ---------------------------------------------------------------------------


class SensePresenceTool(Tool):
    """Scan the environment for entities the agent can interact with.

    Returns all live entities in the scene — their name, type, and a
    summary of their capabilities (modulators + affordances).  Also
    triggers the imagination pipeline for any scene description phrases
    that haven't been processed yet, bringing seed components to life.
    """

    name = "sense_presence"
    description = (
        "Scan your surroundings for interactive entities — creatures, objects, "
        "NPCs, environmental features. Shows what exists around you and what "
        "each entity can do. Use this when you arrive somewhere new or want "
        "to know what you can interact with."
    )
    input_schema: dict[str, Any] = {
        "context": "Optional description of what you're looking for (e.g., 'weapons', 'enemies', 'allies')"
    }

    def __init__(
        self,
        *,
        entity_map: EntityMap,
        imagination_trigger: Any | None = None,
        nac: NAc | None = None,
        atl: Any | None = None,
        agent_id: str = "",
    ) -> None:
        self._entity_map = entity_map
        self._imagination_trigger = imagination_trigger
        self._nac = nac
        self._atl = atl
        self._agent_id = agent_id
        super().__init__()

    def execute(self, **kwargs: Any) -> Any:
        context_query = kwargs.get("context", "")

        # If there's a context query, run it through imagination to potentially
        # instantiate entities the agent is looking for
        if context_query and self._imagination_trigger is not None:
            try:
                self._imagination_trigger.process_percept(
                    context_query,
                    scene_context={"source": "sense_presence"},
                    scene_id="sense_presence_scan",
                )
            except Exception:
                pass  # Best-effort

        entities = self._entity_map.list_entities()
        if not entities:
            return ToolOutput(
                success=True,
                output="No interactive entities detected in your surroundings.",
            )

        lines = [f"You sense {len(entities)} entity/entities nearby:"]
        for entity in entities:
            is_self = self._entity_map.is_self(entity)
            ownership = "YOU" if is_self else "SCENE"
            etype = f" ({entity.entity_type})" if entity.entity_type else ""
            lines.append(f"\n  [{ownership}] {entity.name}{etype}")
            if is_self:
                cap_label = "Your capabilities"
            else:
                cap_label = "Observed capabilities (not callable — use your own tools to interact)"
            for mod_name, mod in entity.modulators.items():
                aff_names = list(mod.affordances.keys())
                if aff_names:
                    annotated = [self._annotate_aff(n) for n in aff_names]
                    lines.append(f"    {cap_label} — {mod_name}: {', '.join(annotated)}")
            # Show sensor summary
            try:
                readings = entity.read_all_sensors()
                if readings:
                    sensor_parts = []
                    for s_name, s_reading in readings.items():
                        sensor_parts.append(f"{s_name}={s_reading.value:.1f}")
                    lines.append(f"    state: {', '.join(sensor_parts)}")
            except Exception:
                pass

        lines.append("")
        lines.append(
            "Use sense_tools to find YOUR specific actions, "
            "or sense(entity_name) to read any entity's full sensor state."
        )
        return ToolOutput(success=True, output="\n".join(lines))

    def _annotate_aff(self, aff_name: str) -> str:
        """Annotate an affordance name with valence from substrate concepts."""
        if self._nac is None or self._atl is None:
            return aff_name
        try:
            from maxim.similarity.decomposer import AFFORDANCE_STRATEGY

            chunks = AFFORDANCE_STRATEGY.extract(aff_name)
            for chunk in chunks:
                concepts = self._atl.recall(name=chunk.text, category="substrate", limit=1)
                for concept in concepts:
                    bias = self._nac.reward_bias(self._agent_id, concept.id)
                    if bias < -0.01:
                        return f"{aff_name} [DANGEROUS]"
                    elif bias > 0.01:
                        return f"{aff_name} [effective]"
        except Exception:
            pass
        return aff_name


# ---------------------------------------------------------------------------
# SenseToolsTool
# ---------------------------------------------------------------------------


def _keyword_overlap(query_words: set[str], text: str) -> float:
    """Score keyword overlap between query words and a text string."""
    text_words = set(text.lower().replace("_", " ").split())
    if not text_words or not query_words:
        return 0.0
    return len(query_words & text_words) / max(len(query_words), 1)


class SenseToolsTool(Tool):
    """Discover physical capabilities by intent.

    Takes a natural language query describing what the agent wants to do,
    matches it against entity modulators and affordances, and activates
    the relevant tools.  Results appear in the next prompt turn.
    """

    name = "sense_tools"
    description = (
        "Discover what physical actions you can perform. Describe what "
        "you want to do — e.g., 'attack with sword', 'repair equipment', "
        "'move quietly'."
    )
    input_schema: dict[str, Any] = {"query": str}

    def __init__(
        self,
        *,
        entity_map: EntityMap,
        tool_registry: ToolRegistry,
        component_index: ComponentIndex | None = None,
        nac: NAc | None = None,
        atl: Any | None = None,
        agent_id: str = "",
    ) -> None:
        self._entity_map = entity_map
        self._registry = tool_registry
        self._component_index = component_index
        self._nac = nac
        self._atl = atl
        self._agent_id = agent_id
        super().__init__()

    def execute(self, **kwargs: Any) -> Any:
        query = kwargs.get("query", "").strip()
        if not query:
            return ToolOutput(
                success=False,
                error="Describe what you want to do. Example: 'attack with sword'",
            )

        query_words = set(query.lower().replace("_", " ").split())
        # Only search the agent's OWN tools (self entities), not scene entities.
        entities = self._entity_map.list_self_entities()

        if not entities:
            return ToolOutput(
                success=True,
                output="No entities available. You have no physical capabilities to discover.",
            )

        # Check if query matches a scene entity — provide a hint
        scene_entities = self._entity_map.list_scene_entities()
        _scene_hint = ""
        for se in scene_entities:
            se_score = _keyword_overlap(query_words, f"{se.name} {se.entity_type}")
            if se_score > 0.3:
                _scene_hint = (
                    f"\nNote: '{se.name}' is a nearby entity you can observe with "
                    f'sense("{se.name}"), but you cannot control it. '
                    f"Use YOUR tools (move, use, etc.) to interact with it."
                )
                break

        # Score each affordance tool against the query
        scored: list[tuple[str, float, str]] = []  # (tool_name, score, description)
        for entity in entities:
            # Entity relevance: name/type keyword match
            entity_score = _keyword_overlap(
                query_words,
                f"{entity.name} {entity.entity_type}",
            )
            # Also check ComponentIndex for semantic similarity
            if self._component_index is not None and entity_score < 0.3:
                match = self._component_index.find(query)
                if match is not None:
                    ref_name = match.ref.rsplit("/", 1)[-1] if "/" in match.ref else match.ref
                    if ref_name.lower() == entity.name.lower():
                        entity_score = max(entity_score, match.score * 0.8)

            for modulator in entity.modulators.values():
                mod_score = _keyword_overlap(query_words, modulator.name)
                for aff_name, aff_schema in modulator.affordances.items():
                    aff_desc = aff_schema.description or aff_name
                    aff_score = _keyword_overlap(query_words, f"{aff_name} {aff_desc}")
                    combined = max(entity_score, mod_score, aff_score)
                    if combined > 0:
                        tool_name = f"{entity.name}_{aff_name}"
                        scored.append((tool_name, combined, aff_desc))

        # Apply NAc valence boost/penalty (S2)
        if self._nac is not None:
            scored = self._apply_nac_ranking(scored)

        # Sort by score descending, take top 8
        scored.sort(key=lambda x: x[1], reverse=True)
        top_matches = scored[:8]

        # If no specific matches, return modulator category summary
        if not top_matches:
            try:
                from maxim.simulation.sim_logger import sim_log

                _ent_names = [e.name for e in entities]
                sim_log(
                    "SEM_TRACE",
                    f"sense_tools('{query}'): NO matches — entities={_ent_names}, "
                    f"scored={len(scored)} candidates but all score=0",
                )
            except Exception:
                pass
            summary = self._modulator_summary(entities)
            if _scene_hint:
                summary = ToolOutput(success=summary.success, output=str(summary.output) + _scene_hint)
            return summary

        # Activate matched tools and build annotated descriptions
        activated: list[str] = []
        descriptions: list[str] = []
        for tool_name, score, desc in top_matches:
            # Check if the tool exists in the registry (may be under a
            # slightly different resolved name due to collision prefixing)
            resolved = self._find_tool_in_registry(tool_name)
            if resolved is not None:
                # Activate if currently inactive
                scene_id = self._registry.get_tool_scene(resolved)
                if scene_id is not None and not self._registry.is_tool_active(resolved):
                    self._registry.activate_scene(scene_id)
                activated.append(resolved)
                tool_obj = self._registry.get(resolved)
                annotation = self._nac_annotation(resolved)
                desc_line = f"- {resolved}: {tool_obj.description}"
                if annotation:
                    desc_line += f" [{annotation}]"
                descriptions.append(desc_line)

        if not activated:
            summary = self._modulator_summary(entities)
            if _scene_hint:
                summary = ToolOutput(success=summary.success, output=str(summary.output) + _scene_hint)
            return summary

        result_lines = [f"Found {len(activated)} capabilities:"]
        result_lines.extend(descriptions)
        if _scene_hint:
            result_lines.append(_scene_hint)
        result_lines.append("")
        result_lines.append(
            "These tools are now available. USE one of them as your next action — "
            "pick the most relevant tool above and call it."
        )

        log.info("sense_tools: activated %d tools for query %r", len(activated), query)
        try:
            from maxim.simulation.sim_logger import sim_discovery, sim_log

            sim_discovery(query, matched=len(top_matches), activated=len(activated), source="SenseToolsTool")
            sim_log(
                "SEM_TRACE",
                f"sense_tools('{query}'): {len(activated)} activated — {', '.join(activated)}",
            )
        except Exception:
            pass
        return ToolOutput(success=True, output="\n".join(result_lines))

    def _find_tool_in_registry(self, expected_name: str) -> str | None:
        """Find a tool by expected name, falling back to prefix-aware match.

        Tool names may have been prefixed for collision resolution
        (e.g., ``parent_entity_affordance`` instead of ``entity_affordance``).
        Only matches on underscore-delimited suffix to avoid false positives
        like ``crystal_dragon_slash`` matching ``rusty_sword_slash``.
        """
        all_tools = self._registry.list_all()
        if expected_name in all_tools:
            return expected_name
        # Try prefix-aware match: tool may have parent prefix prepended
        suffix = f"_{expected_name}"
        for name in all_tools:
            if name.endswith(suffix):
                return name
        return None

    def _modulator_summary(self, entities: list[Any]) -> ToolOutput:
        """Return a modulator category summary for vague queries."""
        lines = ["Your capabilities by category:"]
        for entity in entities:
            for mod_name, mod in entity.modulators.items():
                aff_names = list(mod.affordances.keys())
                if aff_names:
                    lines.append(f"- {mod_name.capitalize()} ({entity.name}): {', '.join(aff_names)}")
        lines.append("")
        lines.append(
            "Try a more specific query like 'attack with sword' or 'repair equipment' to activate those tools."
        )
        return ToolOutput(success=True, output="\n".join(lines))

    # -- NAc valence integration (S2) ------------------------------------------

    def _apply_nac_ranking(self, scored: list[tuple[str, float, str]]) -> list[tuple[str, float, str]]:
        """Boost/penalize scores based on NAc causal link valence."""
        if self._nac is None:
            return scored
        result: list[tuple[str, float, str]] = []
        for tool_name, score, desc in scored:
            links = self._nac.get_links_for_event(tool_name)
            if links:
                # Use the most confident link
                best = max(links, key=lambda lk: lk.confidence)
                if best.confidence >= 0.3:
                    from maxim.decisions.causal_link import Valence

                    if best.outcome_valence == Valence.POSITIVE:
                        score *= 1.2  # boost
                    elif best.outcome_valence == Valence.NEGATIVE:
                        score *= 0.8  # penalize (but don't suppress)
            result.append((tool_name, score, desc))
        return result

    def _nac_annotation(self, tool_name: str) -> str:
        """Generate a short annotation from NAc valence for a tool.

        Two-level lookup:
        1. Entity-specific causal link (existing, e.g., "dragon_fire_breath")
        2. Substrate concept bias fallback (new, e.g., "fire" node reward_bias)
        """
        if self._nac is None:
            return ""

        # 1. Entity-specific causal link (existing path)
        links = self._nac.get_links_for_event(tool_name)
        if links:
            best = max(links, key=lambda lk: lk.confidence)
            if best.confidence >= 0.3:
                from maxim.decisions.causal_link import Valence

                if best.outcome_valence == Valence.POSITIVE:
                    return "worked well before"
                elif best.outcome_valence == Valence.NEGATIVE:
                    return f"caution: {best.outcome_signature}"

        # 2. Substrate concept bias fallback (affordance transfer)
        if self._atl is not None:
            try:
                from maxim.similarity.decomposer import AFFORDANCE_STRATEGY

                # Extract the bare affordance name from the tool when possible.
                # Tool names are "{entity_name}_{affordance_name}" — decomposing
                # the full name wastes cycles on entity-prefix words ("rusty",
                # "sword") that won't match substrate concepts.
                decompose_target = tool_name
                try:
                    tool_obj = self._registry.get(tool_name)
                    aff_name = getattr(tool_obj, "_affordance_name", None)
                    if aff_name:
                        decompose_target = aff_name
                except (KeyError, Exception):
                    pass  # Fall back to full tool name

                chunks = AFFORDANCE_STRATEGY.extract(decompose_target)
                for chunk in chunks:
                    concepts = self._atl.recall(name=chunk.text, category="substrate", limit=1)
                    for concept in concepts:
                        bias = self._nac.reward_bias(self._agent_id, concept.id)
                        if bias < -0.01:
                            return "caution: similar affordance was dangerous"
                        elif bias > 0.01:
                            return "similar affordance worked well"
            except Exception:
                pass

        return ""


# ---------------------------------------------------------------------------
# Goal-based top-k selection
# ---------------------------------------------------------------------------


def select_goal_relevant_tools(
    goal: str,
    entity_map: EntityMap,
    registry: ToolRegistry,
    *,
    max_tools: int = 5,
    min_tools: int = 3,
) -> list[str]:
    """Score affordance tools against sim goal keywords, return top-k to keep active.

    When fewer than *min_tools* match the goal keywords, falls back to
    selecting one affordance per entity (the first affordance of the
    modulator with the most affordances) to ensure physical tools are
    visible even for vague goals like "explore freely".

    Returns tool names that should stay active.  All other affordance
    tools should be scene-deactivated.
    """
    from maxim.embodiment.tool_bridge import ModulatorAffordanceTool

    goal_keywords = set(goal.lower().split())
    scores: list[tuple[str, float]] = []

    for tool_name in registry.list_all():
        try:
            tool = registry.get(tool_name)
        except KeyError:
            continue
        if not isinstance(tool, ModulatorAffordanceTool):
            continue
        tool_words = set(tool.description.lower().split())
        tool_words.add(tool._modulator.name.lower())
        tool_words.add(tool._entity.name.lower())
        overlap = len(goal_keywords & tool_words)
        if overlap > 0:
            scores.append((tool_name, float(overlap)))

    scores.sort(key=lambda x: x[1], reverse=True)
    top_k = [name for name, _ in scores[:max_tools]]

    # Vague goal fallback: one affordance per self entity if top-k is thin
    if len(top_k) < min_tools:
        for entity in entity_map.list_self_entities():
            if not entity.modulators:
                continue
            best_mod = max(
                entity.modulators.values(),
                key=lambda m: len(m.affordances),
            )
            if not best_mod.affordances:
                continue
            first_aff = next(iter(best_mod.affordances))
            candidate = f"{entity.name}_{first_aff}"
            # The tool might have been prefix-resolved; scan the registry
            if candidate not in top_k:
                for rname in registry.list_all():
                    if rname.endswith(f"_{first_aff}") and entity.name in rname:
                        if rname not in top_k:
                            top_k.append(rname)
                            break
                else:
                    if candidate in set(registry.list_all()) and candidate not in top_k:
                        top_k.append(candidate)

    return top_k
