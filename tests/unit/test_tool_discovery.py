"""Tests for SEM Tool Discovery (S1).

Covers:
- EntityMap registration, resolution, collision disambiguation
- UniversalSenseTool entity resolution and sensor reading
- DiscoverToolsTool query matching, activation, and fallback
- Goal-based top-k selection with vague-goal fallback
"""

from __future__ import annotations

import time
from typing import Any

import pytest

from maxim.embodiment.entity_map import EntityMap
from maxim.embodiment.sem import (
    AffordanceSchema,
    Entity,
    SensorReading,
)
from maxim.embodiment.spec import SpecModulator, SpecSensor
from maxim.embodiment.tool_bridge import generate_tools_for_entity
from maxim.tools.discovery import (
    DiscoverToolsTool,
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
        guard1 = _make_entity("guard", "npc")
        guard2 = _make_entity("guard", "npc")
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


# ---------------------------------------------------------------------------
# DiscoverToolsTool tests
# ---------------------------------------------------------------------------


class TestDiscoverToolsTool:
    def _build_discovery_env(self):
        """Build a registry + entity_map with sword + humanoid, tools registered."""
        registry = ToolRegistry()
        emap = EntityMap()
        sword = _sword()
        humanoid = _humanoid()
        emap.register(sword)
        emap.register(humanoid)
        generate_tools_for_entity(sword, registry, entity_map=emap)
        generate_tools_for_entity(humanoid, registry, entity_map=emap)
        return registry, emap

    def test_discover_combat_tools(self):
        registry, emap = self._build_discovery_env()
        tool = DiscoverToolsTool(entity_map=emap, tool_registry=registry)
        result = tool.execute(query="attack sword combat slash")
        assert result.success
        assert "slash" in result.output.lower()

    def test_discover_empty_query(self):
        registry, emap = self._build_discovery_env()
        tool = DiscoverToolsTool(entity_map=emap, tool_registry=registry)
        result = tool.execute(query="")
        assert not result.success

    def test_discover_vague_query_returns_summary(self):
        registry, emap = self._build_discovery_env()
        tool = DiscoverToolsTool(entity_map=emap, tool_registry=registry)
        result = tool.execute(query="xyzzy nothing matches")
        assert result.success
        # Should get modulator category summary
        assert "capabilities by category" in result.output.lower()

    def test_discover_no_entities(self):
        registry = ToolRegistry()
        emap = EntityMap()
        tool = DiscoverToolsTool(entity_map=emap, tool_registry=registry)
        result = tool.execute(query="attack")
        assert result.success
        assert "no entities" in result.output.lower()

    def test_discover_activates_deactivated_tools(self):
        """Simulate hybrid mode: tools registered as scene-scoped, then deactivated."""
        registry = ToolRegistry()
        emap = EntityMap()
        sword = _sword()
        emap.register(sword)

        # Generate tools then re-register as scene-scoped (mirroring orchestrator flow)
        tools = generate_tools_for_entity(sword, registry, entity_map=emap)
        registry.register_scene_tools(tools, "sword_scene")

        # Deactivate the scene — simulating hybrid mode
        registry.deactivate_scene("sword_scene")
        active = registry.list()
        assert not any("slash" in t for t in active)

        tool = DiscoverToolsTool(entity_map=emap, tool_registry=registry)
        result = tool.execute(query="slash sword combat")
        assert result.success
        # After discovery, the scene should be reactivated
        active_after = registry.list()
        assert any("slash" in t for t in active_after)


# ---------------------------------------------------------------------------
# Goal-based top-k selection tests
# ---------------------------------------------------------------------------


class TestGoalSelection:
    def _build_env(self):
        registry = ToolRegistry()
        emap = EntityMap()
        sword = _sword()
        humanoid = _humanoid()
        emap.register(sword)
        emap.register(humanoid)
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
# S2: NAc valence ranking test
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# S3: Imagination deferred registration test
# ---------------------------------------------------------------------------


class TestImaginationDeferred:
    def test_imagination_registers_tools_deactivated(self):
        """Simulates the imagination flow: tools registered but scene-deactivated."""
        registry = ToolRegistry()
        emap = EntityMap()
        dragon = _make_entity(
            "crystal_dragon",
            "creature",
            sensors={"health": {"initial": 1.0}},
            modulators={"combat": {"bite": "Bite the target", "claw": "Claw attack"}},
        )
        emap.register(dragon)
        tools = generate_tools_for_entity(dragon, registry, entity_map=emap)
        scene_id = "imagination_dragon"
        registry.register_scene_tools(tools, scene_id)
        # Immediately deactivate (as imagination trigger does)
        registry.deactivate_scene(scene_id)

        # Tools exist but are not active
        assert "crystal_dragon_bite" not in registry.list()
        assert "crystal_dragon_bite" in registry.list_all()

        # Discovery can reactivate
        tool = DiscoverToolsTool(entity_map=emap, tool_registry=registry)
        result = tool.execute(query="dragon bite combat")
        assert result.success
        # Tools should now be active
        assert "crystal_dragon_bite" in registry.list()


# ---------------------------------------------------------------------------
# S2: NAc valence ranking test
# ---------------------------------------------------------------------------


class TestNAcRanking:
    def test_nac_boosts_positive_valence(self):
        """NAc positive valence boosts discovery ranking."""
        from unittest.mock import MagicMock

        from maxim.decisions.causal_link import CausalLink, TemporalDelta, Valence

        registry = ToolRegistry()
        emap = EntityMap()
        sword = _sword()
        emap.register(sword)
        generate_tools_for_entity(sword, registry, entity_map=emap)

        # Mock NAc with a positive link for slash
        nac = MagicMock()
        positive_link = MagicMock(spec=CausalLink)
        positive_link.confidence = 0.8
        positive_link.outcome_valence = Valence.POSITIVE
        positive_link.outcome_signature = "success"
        nac.get_links_for_event.return_value = [positive_link]

        tool = DiscoverToolsTool(entity_map=emap, tool_registry=registry, nac=nac)
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
        emap.register(sword)
        generate_tools_for_entity(sword, registry, entity_map=emap)

        nac = MagicMock()
        negative_link = MagicMock(spec=CausalLink)
        negative_link.confidence = 0.7
        negative_link.outcome_valence = Valence.NEGATIVE
        negative_link.outcome_signature = "blade_shattered"
        nac.get_links_for_event.return_value = [negative_link]

        tool = DiscoverToolsTool(entity_map=emap, tool_registry=registry, nac=nac)
        result = tool.execute(query="slash sword combat")
        assert result.success
        assert "caution" in result.output
