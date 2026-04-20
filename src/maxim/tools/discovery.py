"""SEM Tool Discovery — semantic tool discovery + universal sensing.

Provides two tools that replace the flat per-entity tool explosion:

- ``UniversalSenseTool`` — single ``sense(entity_name)`` tool that reads
  all sensors on any named entity, replacing N×M individual read/sense tools.
- ``DiscoverToolsTool`` — ``discover_tools(query)`` that matches intent
  against entity modulators/affordances and activates the relevant tools
  via I3's scene-scoped mechanism.

Also provides the goal-based top-k selection algorithm used at prompt
build time to keep a small set of goal-relevant affordance tools visible
from turn 1.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from maxim.tools.base import Tool, ToolOutput

if TYPE_CHECKING:
    from maxim.embodiment.component_index import ComponentIndex
    from maxim.embodiment.entity_map import EntityMap
    from maxim.tools.registry import ToolRegistry

log = logging.getLogger(__name__)


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
# DiscoverToolsTool
# ---------------------------------------------------------------------------


def _keyword_overlap(query_words: set[str], text: str) -> float:
    """Score keyword overlap between query words and a text string."""
    text_words = set(text.lower().replace("_", " ").split())
    if not text_words or not query_words:
        return 0.0
    return len(query_words & text_words) / max(len(query_words), 1)


class DiscoverToolsTool(Tool):
    """Discover physical capabilities by intent.

    Takes a natural language query describing what the agent wants to do,
    matches it against entity modulators and affordances, and activates
    the relevant tools.  Results appear in the next prompt turn.
    """

    name = "discover_tools"
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
    ) -> None:
        self._entity_map = entity_map
        self._registry = tool_registry
        self._component_index = component_index
        super().__init__()

    def execute(self, **kwargs: Any) -> Any:
        query = kwargs.get("query", "").strip()
        if not query:
            return ToolOutput(
                success=False,
                error="Describe what you want to do. Example: 'attack with sword'",
            )

        query_words = set(query.lower().replace("_", " ").split())
        entities = self._entity_map.list_entities()

        if not entities:
            return ToolOutput(
                success=True,
                output="No entities available. You have no physical capabilities to discover.",
            )

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

        # Sort by score descending, take top 8
        scored.sort(key=lambda x: x[1], reverse=True)
        top_matches = scored[:8]

        # If no specific matches, return modulator category summary
        if not top_matches:
            return self._modulator_summary(entities)

        # Activate matched tools
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
                descriptions.append(f"- {resolved}: {tool_obj.description}")

        if not activated:
            return self._modulator_summary(entities)

        result_lines = [f"Found {len(activated)} capabilities:"]
        result_lines.extend(descriptions)
        result_lines.append("")
        result_lines.append("These tools are now available for your next action.")

        log.info("discover_tools: activated %d tools for query %r", len(activated), query)
        return ToolOutput(success=True, output="\n".join(result_lines))

    def _find_tool_in_registry(self, expected_name: str) -> str | None:
        """Find a tool by expected name, falling back to fuzzy match."""
        all_tools = self._registry.list_all()
        if expected_name in all_tools:
            return expected_name
        # Try fuzzy: the tool may have been prefixed for collision resolution
        for name in all_tools:
            if name.endswith(expected_name) or expected_name.endswith(name):
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

    # Vague goal fallback: one affordance per entity if top-k is thin
    if len(top_k) < min_tools:
        for entity in entity_map.list_entities():
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
