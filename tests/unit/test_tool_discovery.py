"""Tests for SEM Tool Discovery (S1) + Entity Ownership.

Covers:
- EntityMap registration, resolution, collision disambiguation
- EntityMap ownership: register_self vs register_scene
- UniversalSenseTool entity resolution and sensor reading
- SenseToolsTool query matching, activation, self-entity filter, scene hints
- Goal-based top-k selection with vague-goal fallback
- SensePresenceTool ownership labels
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from maxim.embodiment.entity_map import EntityMap
from maxim.embodiment.sem import (
    AffordanceSchema,
    Entity,
)
from maxim.embodiment.spec import SpecModulator, SpecSensor
from maxim.embodiment.tool_bridge import (
    describe_entity_capabilities,
    generate_tools_for_entity,
)
from maxim.tools.discovery import (
    SenseToolsTool,
    SensePresenceTool,
    UniversalSenseTool,
    select_goal_relevant_tools,
)
from maxim.tools.registry import ToolRegistry


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_entity(
    name: str,
    entity_type: str = "weapon",
    *,
    sensors: dict[str, dict] | None = None,
    modulators: dict[str, dict[str, str]] | None = None,
) -> Entity:
    """Build a test entity with SpecSensor/SpecModulator stubs."""
    ent = Entity(name=name, entity_type=entity_type)

    if sensors:
        for sname, spec in sensors.items():
            ent.sensors[sname] = SpecSensor(
                _name=sname,
                _entity_name=name,
                _unit=spec.get("unit", "ratio"),
                _schema={"type": "float", "range": [0, 1]},
                _initial=spec.get("initial", 0.5),
                _entity_ref=ent,
            )
            ent.vital_metrics[sname] = spec.get("initial", 0.5)

    if modulators:
        for mname, affordances in modulators.items():
            affs = {}
            for aff_name, desc in affordances.items():
                affs[aff_name] = AffordanceSchema(description=desc)
            ent.modulators[mname] = SpecModulator(
                _name=mname,
                _entity_name=name,
                _affordances=affs,
            )
    return ent


def _sword() -> Entity:
    return _make_entity(
        "rusty_sword",
        "weapon",
        sensors={"durability": {"initial": 0.3}, "sharpness": {"initial": 0.5}},
        modulators={
            "combat": {
                "slash": "Slash at a target with the sword",
                "parry": "Parry an incoming attack",
                "throw": "Throw the weapon at a target",
            },
            "maintenance": {
                "sharpen": "Sharpen the blade",
                "repair": "Repair the sword with materials",
            },
        },
    )


def _humanoid() -> Entity:
    return _make_entity(
        "base_humanoid",
        "body",
        sensors={"health": {"initial": 1.0}, "stamina": {"initial": 0.8}},
        modulators={
            "locomotion": {
                "move": "Move to a location",
                "rest": "Rest to recover stamina",
            },
            "manipulation": {
                "pick_up": "Pick up an object",
                "drop": "Drop a held object",
            },
        },
    )


def _dragon() -> Entity:
    return _make_entity(
        "dragon",
        "creature",
        sensors={"health": {"initial": 1.0}, "fire_charge": {"initial": 0.8}},
        modulators={
            "combat": {
                "fire_breath": "Breathe fire at a target",
                "tail_sweep": "Sweep tail to knock back nearby enemies",
            },
            "flight": {
                "circle": "Circle overhead to reposition",
            },
        },
    )


# ---------------------------------------------------------------------------
# EntityMap tests
# ---------------------------------------------------------------------------


class TestEntityMap:
    def test_register_and_resolve(self):
        emap = EntityMap()
        sword = _sword()
        emap.register(sword)
        assert emap.resolve("rusty_sword") is sword

    def test_resolve_case_insensitive(self):
        emap = EntityMap()
        sword = _sword()
        emap.register(sword)
        assert emap.resolve("Rusty_Sword") is sword
        assert emap.resolve("rusty sword") is sword

    def test_resolve_unknown_returns_none(self):
        emap = EntityMap()
        assert emap.resolve("nonexistent") is None

    def test_list_names(self):
        emap = EntityMap()
        emap.register(_sword())
        emap.register(_humanoid())
        names = emap.list_names()
        assert "rusty_sword" in names
        assert "base_humanoid" in names

    def test_list_entities_deduplicates(self):
        emap = EntityMap()
        sword = _sword()
        emap.register(sword)
        entities = emap.list_entities()
        assert len(entities) == 1
        assert entities[0] is sword

    def test_collision_uses_full_path(self):
        emap = EntityMap()
        _make_entity("guard", "npc")
        _make_entity("guard", "npc")
        # Attach to different parents so full_path differs
        town = Entity("town", "location")
        dungeon = Entity("dungeon", "location")
        guard1_child = _make_entity("guard", "npc")
        guard1_child.parent = town
        town.children.append(guard1_child)
        guard2_child = _make_entity("guard", "npc")
        guard2_child.parent = dungeon
        dungeon.children.append(guard2_child)

        emap.register(guard1_child)
        emap.register(guard2_child)
        # Both should be resolvable by full_path
        assert emap.resolve("town.guard") is guard1_child
        assert emap.resolve("dungeon.guard") is guard2_child
        # Simple "guard" should be gone (ambiguous)
        assert emap.resolve("guard") is None

    def test_len(self):
        emap = EntityMap()
        emap.register(_sword())
        emap.register(_humanoid())
        assert len(emap) == 2

    def test_contains(self):
        emap = EntityMap()
        emap.register(_sword())
        assert "rusty_sword" in emap
        assert "nonexistent" not in emap


# ---------------------------------------------------------------------------
# EntityMap ownership tests
# ---------------------------------------------------------------------------


class TestEntityMapOwnership:
    def test_register_self_marks_as_self(self):
        emap = EntityMap()
        sword = _sword()
        emap.register_self(sword)
        assert emap.is_self(sword)
        assert emap.resolve("rusty_sword") is sword

    def test_register_scene_not_self(self):
        emap = EntityMap()
        dragon = _dragon()
        emap.register_scene(dragon)
        assert not emap.is_self(dragon)
        assert emap.resolve("dragon") is dragon

    def test_register_defaults_to_scene(self):
        emap = EntityMap()
        dragon = _dragon()
        emap.register(dragon)
        assert not emap.is_self(dragon)

    def test_list_self_entities(self):
        emap = EntityMap()
        sword = _sword()
        dragon = _dragon()
        emap.register_self(sword)
        emap.register_scene(dragon)
        self_ents = emap.list_self_entities()
        assert len(self_ents) == 1
        assert self_ents[0] is sword

    def test_list_scene_entities(self):
        emap = EntityMap()
        sword = _sword()
        dragon = _dragon()
        emap.register_self(sword)
        emap.register_scene(dragon)
        scene_ents = emap.list_scene_entities()
        assert len(scene_ents) == 1
        assert scene_ents[0] is dragon

    def test_list_entities_returns_all(self):
        emap = EntityMap()
        sword = _sword()
        dragon = _dragon()
        emap.register_self(sword)
        emap.register_scene(dragon)
        all_ents = emap.list_entities()
        assert len(all_ents) == 2

    def test_mixed_self_and_scene(self):
        emap = EntityMap()
        humanoid = _humanoid()
        sword = _sword()
        dragon = _dragon()
        emap.register_self(humanoid)
        emap.register_self(sword)
        emap.register_scene(dragon)
        assert len(emap.list_self_entities()) == 2
        assert len(emap.list_scene_entities()) == 1
        assert len(emap.list_entities()) == 3


# ---------------------------------------------------------------------------
# describe_entity_capabilities tests
# ---------------------------------------------------------------------------


class TestDescribeEntityCapabilities:
    def test_describes_modulators(self):
        dragon = _dragon()
        desc = describe_entity_capabilities(dragon)
        assert "fire_breath" in desc
        assert "tail_sweep" in desc
        assert "circle" in desc

    def test_no_modulators(self):
        ent = _make_entity("rock", "object", sensors={"weight": {"initial": 1.0}})
        desc = describe_entity_capabilities(ent)
        assert desc == "No observable capabilities."


# ---------------------------------------------------------------------------
# UniversalSenseTool tests
# ---------------------------------------------------------------------------


class TestUniversalSenseTool:
    def test_sense_reads_all_sensors(self):
        emap = EntityMap()
        sword = _sword()
        emap.register(sword)
        tool = UniversalSenseTool(entity_map=emap)
        result = tool.execute(entity_name="rusty_sword")
        assert result.success
        assert "durability" in result.output["sensors"]
        assert "sharpness" in result.output["sensors"]
        assert result.output["sensors"]["durability"]["value"] == pytest.approx(0.3)

    def test_sense_unknown_entity(self):
        emap = EntityMap()
        tool = UniversalSenseTool(entity_map=emap)
        result = tool.execute(entity_name="nonexistent")
        assert not result.success
        assert "Unknown entity" in result.error

    def test_sense_empty_name(self):
        emap = EntityMap()
        emap.register(_sword())
        tool = UniversalSenseTool(entity_map=emap)
        result = tool.execute(entity_name="")
        assert not result.success
        assert "Specify an entity" in result.error

    def test_sense_lists_known_entities_on_error(self):
        emap = EntityMap()
        emap.register(_sword())
        tool = UniversalSenseTool(entity_map=emap)
        result = tool.execute(entity_name="nonexistent")
        assert "rusty_sword" in result.error

    def test_sense_works_on_scene_entities(self):
        """sense() can read any entity — self or scene."""
        emap = EntityMap()
        dragon = _dragon()
        emap.register_scene(dragon)
        tool = UniversalSenseTool(entity_map=emap)
        result = tool.execute(entity_name="dragon")
        assert result.success
        assert "health" in result.output["sensors"]


# ---------------------------------------------------------------------------
# SenseToolsTool tests
# ---------------------------------------------------------------------------


class TestSenseToolsTool:
    def _build_discovery_env(self):
        """Build a registry + entity_map with sword + humanoid as self entities."""
        registry = ToolRegistry()
        emap = EntityMap()
        sword = _sword()
        humanoid = _humanoid()
        emap.register_self(sword)
        emap.register_self(humanoid)
        generate_tools_for_entity(sword, registry, entity_map=emap)
        generate_tools_for_entity(humanoid, registry, entity_map=emap)
        return registry, emap

    def test_discover_combat_tools(self):
        registry, emap = self._build_discovery_env()
        tool = SenseToolsTool(entity_map=emap, tool_registry=registry)
        result = tool.execute(query="attack sword combat slash")
        assert result.success
        assert "slash" in result.output.lower()

    def test_discover_empty_query(self):
        registry, emap = self._build_discovery_env()
        tool = SenseToolsTool(entity_map=emap, tool_registry=registry)
        result = tool.execute(query="")
        assert not result.success

    def test_discover_vague_query_returns_summary(self):
        registry, emap = self._build_discovery_env()
        tool = SenseToolsTool(entity_map=emap, tool_registry=registry)
        result = tool.execute(query="xyzzy nothing matches")
        assert result.success
        # Should get modulator category summary
        assert "capabilities by category" in result.output.lower()

    def test_discover_no_entities(self):
        registry = ToolRegistry()
        emap = EntityMap()
        tool = SenseToolsTool(entity_map=emap, tool_registry=registry)
        result = tool.execute(query="attack")
        assert result.success
        assert "no entities" in result.output.lower()

    def test_discover_activates_deactivated_tools(self):
        """Simulate hybrid mode: tools registered as scene-scoped, then deactivated."""
        registry = ToolRegistry()
        emap = EntityMap()
        sword = _sword()
        emap.register_self(sword)

        # Generate tools then re-register as scene-scoped (mirroring orchestrator flow)
        tools = generate_tools_for_entity(sword, registry, entity_map=emap)
        registry.register_scene_tools(tools, "sword_scene")

        # Deactivate the scene — simulating hybrid mode
        registry.deactivate_scene("sword_scene")
        active = registry.list()
        assert not any("slash" in t for t in active)

        tool = SenseToolsTool(entity_map=emap, tool_registry=registry)
        result = tool.execute(query="slash sword combat")
        assert result.success
        # After discovery, the scene should be reactivated
        active_after = registry.list()
        assert any("slash" in t for t in active_after)

    def test_discover_excludes_scene_entity_tools(self):
        """Scene entity affordances must NOT appear in sense_tools results."""
        registry = ToolRegistry()
        emap = EntityMap()
        sword = _sword()
        dragon = _dragon()
        emap.register_self(sword)
        emap.register_scene(dragon)
        # Only generate tools for self entity
        generate_tools_for_entity(sword, registry, entity_map=emap)

        tool = SenseToolsTool(entity_map=emap, tool_registry=registry)
        result = tool.execute(query="fire breath dragon combat")
        assert result.success
        # Dragon tools should NOT appear — only sword tools or a summary
        assert "fire_breath" not in result.output
        assert "tail_sweep" not in result.output

    def test_discover_scene_entity_hint(self):
        """When query matches a scene entity, provide a hint."""
        registry = ToolRegistry()
        emap = EntityMap()
        sword = _sword()
        dragon = _dragon()
        emap.register_self(sword)
        emap.register_scene(dragon)
        generate_tools_for_entity(sword, registry, entity_map=emap)

        tool = SenseToolsTool(entity_map=emap, tool_registry=registry)
        result = tool.execute(query="dragon creature")
        assert result.success
        assert "sense" in result.output.lower()
        assert "observe" in result.output.lower() or "cannot control" in result.output.lower()


# ---------------------------------------------------------------------------
# Goal-based top-k selection tests
# ---------------------------------------------------------------------------


class TestGoalSelection:
    def _build_env(self):
        registry = ToolRegistry()
        emap = EntityMap()
        sword = _sword()
        humanoid = _humanoid()
        emap.register_self(sword)
        emap.register_self(humanoid)
        generate_tools_for_entity(sword, registry, entity_map=emap)
        generate_tools_for_entity(humanoid, registry, entity_map=emap)
        return registry, emap

    def test_combat_goal_selects_combat_tools(self):
        registry, emap = self._build_env()
        selected = select_goal_relevant_tools("test sword combat", emap, registry)
        # Should include combat-related tools
        tool_names_str = " ".join(selected)
        assert "slash" in tool_names_str or "combat" in tool_names_str or "sword" in tool_names_str

    def test_vague_goal_fallback(self):
        registry, emap = self._build_env()
        selected = select_goal_relevant_tools("explore freely", emap, registry)
        # Should still return at least min_tools (3) via fallback
        assert len(selected) >= 1  # at least one per entity

    def test_empty_goal(self):
        registry, emap = self._build_env()
        selected = select_goal_relevant_tools("", emap, registry)
        # Should get fallback tools
        assert len(selected) >= 1

    def test_max_tools_respected(self):
        registry, emap = self._build_env()
        selected = select_goal_relevant_tools(
            "test sword combat slash parry throw move rest pick_up drop sharpen repair",
            emap,
            registry,
            max_tools=3,
        )
        assert len(selected) <= 3

    def test_scene_entities_excluded_from_fallback(self):
        """Scene entities should not contribute tools in the vague-goal fallback."""
        registry = ToolRegistry()
        emap = EntityMap()
        sword = _sword()
        dragon = _dragon()
        emap.register_self(sword)
        emap.register_scene(dragon)
        generate_tools_for_entity(sword, registry, entity_map=emap)

        selected = select_goal_relevant_tools("explore", emap, registry)
        for tool_name in selected:
            assert "dragon" not in tool_name


# ---------------------------------------------------------------------------
# Integration: generate_tools_for_entity with entity_map
# ---------------------------------------------------------------------------


class TestEntityMapIntegration:
    def test_generate_tools_populates_entity_map(self):
        registry = ToolRegistry()
        emap = EntityMap()
        sword = _sword()
        generate_tools_for_entity(sword, registry, entity_map=emap)
        assert emap.resolve("rusty_sword") is sword

    def test_generate_tools_without_entity_map(self):
        """Existing callers without entity_map still work."""
        registry = ToolRegistry()
        sword = _sword()
        tools = generate_tools_for_entity(sword, registry)
        assert len(tools) > 0


# ---------------------------------------------------------------------------
# S2: Per-tool deactivation tests
# ---------------------------------------------------------------------------


class TestDeactivateTool:
    def test_deactivate_single_tool(self):
        registry = ToolRegistry()
        sword = _sword()
        tools = generate_tools_for_entity(sword, registry)
        registry.register_scene_tools(tools, "sword_scene")
        # All tools active
        assert registry.is_tool_active("rusty_sword_slash")
        # Deactivate one tool
        assert registry.deactivate_tool("rusty_sword_slash")
        assert not registry.is_tool_active("rusty_sword_slash")
        # Other tools in same scene still active
        assert registry.is_tool_active("rusty_sword_parry")

    def test_deactivate_nonexistent_returns_false(self):
        registry = ToolRegistry()
        assert not registry.deactivate_tool("nonexistent")

    def test_deactivate_core_tool_returns_false(self):
        """Core tools (no scene) can't be deactivated via deactivate_tool."""
        registry = ToolRegistry()
        sword = _sword()
        generate_tools_for_entity(sword, registry)
        # Core-registered tools have no _scene_meta entry
        assert not registry.deactivate_tool("rusty_sword_slash")


# ---------------------------------------------------------------------------
# S2: LRU eviction tests
# ---------------------------------------------------------------------------


class TestLRUEviction:
    def test_evict_stale_tools(self):
        from maxim.tools.discovery import (
            DISCOVERY_LRU_TURNS,
            evict_stale_discoveries,
            mark_tool_used,
            reset_discovery_state,
        )

        reset_discovery_state()
        registry = ToolRegistry()
        sword = _sword()
        tools = generate_tools_for_entity(sword, registry)
        registry.register_scene_tools(tools, "sword_scene")

        # Mark slash as used on turn 1
        mark_tool_used("rusty_sword_slash", 1)
        mark_tool_used("rusty_sword_parry", 1)

        # Evict at turn 1 + LRU_TURNS + 1 — both should be evicted
        evicted = evict_stale_discoveries(1 + DISCOVERY_LRU_TURNS + 1, registry)
        assert "rusty_sword_slash" in evicted
        assert "rusty_sword_parry" in evicted
        assert not registry.is_tool_active("rusty_sword_slash")

    def test_recently_used_not_evicted(self):
        from maxim.tools.discovery import (
            evict_stale_discoveries,
            mark_tool_used,
            reset_discovery_state,
        )

        reset_discovery_state()
        registry = ToolRegistry()
        sword = _sword()
        tools = generate_tools_for_entity(sword, registry)
        registry.register_scene_tools(tools, "sword_scene")

        # Mark slash as used on turn 5
        mark_tool_used("rusty_sword_slash", 5)

        # Evict at turn 7 — still within LRU window
        evicted = evict_stale_discoveries(7, registry)
        assert "rusty_sword_slash" not in evicted
        assert registry.is_tool_active("rusty_sword_slash")

    def test_goal_selected_exempt(self):
        from maxim.tools.discovery import (
            DISCOVERY_LRU_TURNS,
            evict_stale_discoveries,
            mark_goal_selected,
            mark_tool_used,
            reset_discovery_state,
        )

        reset_discovery_state()
        registry = ToolRegistry()
        sword = _sword()
        tools = generate_tools_for_entity(sword, registry)
        registry.register_scene_tools(tools, "sword_scene")

        mark_tool_used("rusty_sword_slash", 1)
        mark_goal_selected(["rusty_sword_slash"])

        # Even past LRU window, goal-selected tools survive
        evicted = evict_stale_discoveries(1 + DISCOVERY_LRU_TURNS + 1, registry)
        assert "rusty_sword_slash" not in evicted
        assert registry.is_tool_active("rusty_sword_slash")


# ---------------------------------------------------------------------------
# Scene entity ownership: no tools generated, observe-only
# ---------------------------------------------------------------------------


class TestSceneEntityOwnership:
    def test_scene_entity_has_no_tools_in_registry(self):
        """Scene entities registered via register_scene have no affordance tools."""
        registry = ToolRegistry()
        emap = EntityMap()
        dragon = _dragon()
        emap.register_scene(dragon)
        # No generate_tools_for_entity called for scene entity
        all_tools = registry.list_all()
        assert not any("dragon" in t for t in all_tools)

    def test_scene_entity_sensable(self):
        """Scene entities can be sensed via UniversalSenseTool."""
        emap = EntityMap()
        dragon = _dragon()
        emap.register_scene(dragon)
        tool = UniversalSenseTool(entity_map=emap)
        result = tool.execute(entity_name="dragon")
        assert result.success
        assert result.output["entity"] == "dragon"

    def test_sense_presence_labels_ownership(self):
        """sense_presence shows [YOU] for self, [SCENE] for scene entities."""
        emap = EntityMap()
        humanoid = _humanoid()
        dragon = _dragon()
        emap.register_self(humanoid)
        emap.register_scene(dragon)
        tool = SensePresenceTool(entity_map=emap)
        result = tool.execute()
        output = result.output
        assert "[YOU]" in output
        assert "[SCENE]" in output
        assert "base_humanoid" in output
        assert "dragon" in output
        assert "not callable" in output.lower()

    def test_sense_presence_all_self(self):
        """When only self entities exist, no [SCENE] labels."""
        emap = EntityMap()
        humanoid = _humanoid()
        emap.register_self(humanoid)
        tool = SensePresenceTool(entity_map=emap)
        result = tool.execute()
        assert "[YOU]" in result.output
        assert "[SCENE]" not in result.output

    def test_sense_presence_context_optional_in_jsonschema(self):
        """``context`` must export as optional, not required.

        Regression for the legacy description-as-value pattern: when
        ``input_schema`` was ``{"context": "Optional description..."}``
        the JSONSchema export emitted ``required: ["context"]`` despite
        the prose saying "Optional", so strict MCP / Anthropic clients
        rejected calls that omitted it. ``execute()`` already accepts a
        missing context (and the auto-sense caller in agent_loop never
        passes one), so the fix is in the schema.
        """
        emap = EntityMap()
        tool = SensePresenceTool(entity_map=emap)
        schema = tool.to_json_schema()
        assert schema["type"] == "object"
        assert "context" in schema["properties"]
        # The critical assertion: not required.
        assert "context" not in schema.get("required", [])
        # Description preserved so the LLM still sees the usage hint.
        assert "description" in schema["properties"]["context"]


# ---------------------------------------------------------------------------
# S2: NAc valence ranking test
# ---------------------------------------------------------------------------


class TestNAcRanking:
    def test_nac_boosts_positive_valence(self):
        """NAc positive valence boosts discovery ranking."""
        from unittest.mock import MagicMock

        from maxim.decisions.causal_link import CausalLink, Valence

        registry = ToolRegistry()
        emap = EntityMap()
        sword = _sword()
        emap.register_self(sword)
        generate_tools_for_entity(sword, registry, entity_map=emap)

        # Mock NAc with a positive link for slash
        nac = MagicMock()
        positive_link = MagicMock(spec=CausalLink)
        positive_link.confidence = 0.8
        positive_link.outcome_valence = Valence.POSITIVE
        positive_link.outcome_signature = "success"
        nac.get_links_for_event.return_value = [positive_link]

        tool = SenseToolsTool(entity_map=emap, tool_registry=registry, nac=nac)
        result = tool.execute(query="slash sword combat")
        assert result.success
        assert "worked well before" in result.output

    def test_nac_annotates_negative_valence(self):
        """NAc negative valence adds caution annotation."""
        from unittest.mock import MagicMock

        from maxim.decisions.causal_link import CausalLink, Valence

        registry = ToolRegistry()
        emap = EntityMap()
        sword = _sword()
        emap.register_self(sword)
        generate_tools_for_entity(sword, registry, entity_map=emap)

        nac = MagicMock()
        negative_link = MagicMock(spec=CausalLink)
        negative_link.confidence = 0.7
        negative_link.outcome_valence = Valence.NEGATIVE
        negative_link.outcome_signature = "blade_shattered"
        nac.get_links_for_event.return_value = [negative_link]

        tool = SenseToolsTool(entity_map=emap, tool_registry=registry, nac=nac)
        result = tool.execute(query="slash sword combat")
        assert result.success
        assert "caution" in result.output


# ---------------------------------------------------------------------------
# Affordance concept transfer annotations (Stage 3)
# ---------------------------------------------------------------------------


class TestAffordanceTransferAnnotations:
    """Tests for substrate-level affordance valence annotations."""

    def _make_nac_with_bias(self, agent_id: str, node_id: str, bias: float):
        """Create a NAc with a pre-set reward bias on a node."""
        from maxim.decisions.nac import NAc

        nac = NAc()
        nac._reward_bias[(agent_id, node_id)] = bias
        return nac

    def _make_atl_with_concept(self, name: str, concept_id: str):
        """Create an ATL with a pre-set substrate concept."""
        from maxim.memory.atl import ATL
        from maxim.memory.semantic_types import Concept, ConceptProvenance

        atl = ATL()
        concept = Concept(
            id=concept_id,
            timestamp=0.0,
            name=name,
            category="substrate",
            provenance=ConceptProvenance.AGENT_INFERENCE,
        )
        atl.store(concept)
        return atl

    def test_sense_presence_annotates_dangerous(self):
        """SensePresenceTool adds [DANGEROUS] for negative bias affordances."""
        entity_map = MagicMock()
        entity = _make_entity(
            "dragon",
            "creature",
            modulators={"combat": {"fire_breath": "breathe fire"}},
        )
        entity_map.list_entities.return_value = [entity]
        entity_map.is_self.return_value = False

        nac = self._make_nac_with_bias("aut-1", "node-fire", -0.1)
        atl = self._make_atl_with_concept("fire breath", "node-fire")

        tool = SensePresenceTool(
            entity_map=entity_map,
            nac=nac,
            atl=atl,
            agent_id="aut-1",
        )
        result = tool.execute()
        assert result.success
        assert "DANGEROUS" in str(result.output)

    def test_sense_presence_no_annotation_without_bias(self):
        """SensePresenceTool shows plain names when no bias exists."""
        entity_map = MagicMock()
        entity = _make_entity(
            "dragon",
            "creature",
            modulators={"combat": {"fire_breath": "breathe fire"}},
        )
        entity_map.list_entities.return_value = [entity]
        entity_map.is_self.return_value = False

        from maxim.decisions.nac import NAc
        from maxim.memory.atl import ATL

        tool = SensePresenceTool(
            entity_map=entity_map,
            nac=NAc(),
            atl=ATL(),
            agent_id="aut-1",
        )
        result = tool.execute()
        assert result.success
        assert "DANGEROUS" not in str(result.output)

    def test_sense_presence_degrades_without_nac(self):
        """SensePresenceTool works without NAc — no annotations, no crash."""
        entity_map = MagicMock()
        entity = _make_entity(
            "dragon",
            "creature",
            modulators={"combat": {"fire_breath": "breathe fire"}},
        )
        entity_map.list_entities.return_value = [entity]
        entity_map.is_self.return_value = False

        tool = SensePresenceTool(entity_map=entity_map)
        result = tool.execute()
        assert result.success
        assert "fire_breath" in str(result.output)

    def test_sense_tools_substrate_fallback_annotation(self):
        """SenseToolsTool falls back to substrate concept bias."""
        # Store concepts for individual words — the ATL lookup is by exact name
        nac = self._make_nac_with_bias("aut-1", "node-fire", -0.1)
        atl = self._make_atl_with_concept("fire", "node-fire")

        entity_map = MagicMock()
        sword = _make_entity(
            "rusty_sword",
            "weapon",
            modulators={"combat": {"fire_slash": "flaming slash attack"}},
        )
        entity_map.list_self_entities.return_value = [sword]
        entity_map.list_scene_entities.return_value = []

        registry = MagicMock()
        registry.list_all.return_value = ["rusty_sword_fire_slash"]
        mock_tool = MagicMock(description="Flaming slash attack")
        mock_tool._affordance_name = "fire_slash"
        registry.get.return_value = mock_tool
        registry.get_tool_scene.return_value = "scene-1"
        registry.is_tool_active.return_value = True

        tool = SenseToolsTool(
            entity_map=entity_map,
            tool_registry=registry,
            nac=nac,
            atl=atl,
            agent_id="aut-1",
        )
        # The tool name "rusty_sword_fire_slash" decomposes, and "fire"
        # matches the ATL concept with negative bias → annotation
        annotation = tool._nac_annotation("rusty_sword_fire_slash")
        assert "caution" in annotation or "similar" in annotation
