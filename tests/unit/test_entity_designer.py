"""Tests for Phase 7 — Entity Designer, character templates, and architect tools."""

from __future__ import annotations

import textwrap


from maxim.simulation.entity_designer import EntityDesigner, validate_entity_spec
from maxim.simulation.character_templates import (
    PC_ARCHETYPES,
    NPC_ROLES,
    get_archetype,
    get_npc_role,
    list_archetypes,
    list_npc_roles,
)


# ---------------------------------------------------------------------------
# EntityDesigner — fallback mode (no LLM)
# ---------------------------------------------------------------------------


class TestEntityDesignerFallback:
    def test_design_npc_basic(self):
        designer = EntityDesigner()
        spec = designer.design("a town guard", entity_type="npc")
        assert spec["entity_type"] == "npc"
        assert "hp" in spec["sensors"]
        assert "social" in spec["modulators"]

    def test_design_weapon(self):
        designer = EntityDesigner()
        spec = designer.design("a rusty sword", entity_type="weapon")
        assert spec["entity_type"] == "weapon"
        assert "durability" in spec["sensors"]

    def test_design_creature(self):
        designer = EntityDesigner()
        spec = designer.design("a fierce wolf", entity_type="creature")
        assert spec["entity_type"] == "creature"
        assert "hp" in spec["sensors"]
        assert "aggression" in spec["sensors"]

    def test_design_environment(self):
        designer = EntityDesigner()
        spec = designer.design("a dark cave", entity_type="environment")
        assert spec["entity_type"] == "environment"
        assert "visibility" in spec["sensors"]

    def test_suspicious_keyword_sets_low_trust(self):
        designer = EntityDesigner()
        spec = designer.design("a suspicious stranger", entity_type="npc")
        assert "trust" in spec["sensors"]
        assert spec["sensors"]["trust"]["initial"] <= 0.3

    def test_friendly_keyword_sets_high_trust(self):
        designer = EntityDesigner()
        spec = designer.design("a friendly innkeeper", entity_type="npc")
        assert "trust" in spec["sensors"]
        assert spec["sensors"]["trust"]["initial"] >= 0.7

    def test_battle_hardened_boosts_hp(self):
        designer = EntityDesigner()
        spec = designer.design("a battle-hardened veteran", entity_type="npc")
        assert spec["sensors"]["hp"]["initial"] > 20
        assert spec["sensors"]["hp"]["range"][1] > 20

    def test_design_with_base_ref(self, tmp_path):
        """Designer uses ComponentRegistry base when available."""
        from maxim.embodiment.component_registry import ComponentRegistry

        # Create a test component
        comp_dir = tmp_path / "npcs"
        comp_dir.mkdir()
        (comp_dir / "test_base.yaml").write_text(textwrap.dedent("""\
            component:
              name: test_base
              category: npcs
            entity:
              name: test_base
              entity_type: npc
              sensors:
                hp: { unit: points, range: [0, 20], initial: 20 }
                trust: { unit: ratio, range: [0, 1], initial: 0.5 }
        """))

        registry = ComponentRegistry(search_paths=[tmp_path], include_defaults=False)
        designer = EntityDesigner(component_registry=registry)
        spec = designer.design("a suspicious version", base_ref="npcs/test_base", entity_type="npc")

        # Should have base sensors
        assert "hp" in spec["sensors"]
        # Suspicious keyword should lower trust
        assert "trust" in spec["sensors"]

    def test_design_npc_convenience(self):
        designer = EntityDesigner()
        spec = designer.design_npc(
            name="Captain Aldric",
            personality="Stern and duty-bound",
            race="human",
            archetype="warrior",
            base_ref=None,
        )
        assert spec["name"] == "Captain Aldric"
        assert spec["metadata"]["race"] == "human"
        assert spec["metadata"]["class"] == "warrior"
        assert "Stern" in spec["metadata"]["persona_prompt"]

    def test_design_item_convenience(self):
        designer = EntityDesigner()
        spec = designer.design_item(
            name="Healing Potion",
            description="Restores 10 HP when consumed",
        )
        assert spec["name"] == "Healing Potion"


# ---------------------------------------------------------------------------
# validate_entity_spec
# ---------------------------------------------------------------------------


class TestValidateEntitySpec:
    def test_valid_spec(self):
        spec = {
            "name": "test",
            "entity_type": "npc",
            "sensors": {
                "hp": {"unit": "points", "range": [0, 20], "initial": 20},
            },
            "modulators": {
                "social": {
                    "affordances": {
                        "speak": {"params": {"message": "str"}, "description": "Say something"},
                    },
                },
            },
        }
        assert validate_entity_spec(spec) == []

    def test_missing_name(self):
        errors = validate_entity_spec({"sensors": {}})
        assert any("name" in e.lower() for e in errors)

    def test_sensor_missing_unit(self):
        errors = validate_entity_spec({
            "name": "test",
            "sensors": {"hp": {"range": [0, 20]}},
        })
        assert any("unit" in e for e in errors)

    def test_sensor_missing_range(self):
        errors = validate_entity_spec({
            "name": "test",
            "sensors": {"hp": {"unit": "points"}},
        })
        assert any("range" in e for e in errors)

    def test_sensor_invalid_range(self):
        errors = validate_entity_spec({
            "name": "test",
            "sensors": {"hp": {"unit": "points", "range": [20, 0]}},
        })
        assert any("min" in e.lower() for e in errors)

    def test_sensor_initial_out_of_range(self):
        errors = validate_entity_spec({
            "name": "test",
            "sensors": {"hp": {"unit": "points", "range": [0, 20], "initial": 30}},
        })
        assert any("outside range" in e for e in errors)

    def test_affordance_missing_description(self):
        errors = validate_entity_spec({
            "name": "test",
            "modulators": {
                "combat": {
                    "affordances": {
                        "slash": {"params": {"target": "str"}},
                    },
                },
            },
        })
        assert any("description" in e for e in errors)


# ---------------------------------------------------------------------------
# Character Templates
# ---------------------------------------------------------------------------


class TestCharacterTemplates:
    def test_all_archetypes_exist(self):
        assert len(PC_ARCHETYPES) >= 5
        for name in ("warrior", "mage", "rogue", "diplomat", "healer"):
            assert name in PC_ARCHETYPES

    def test_archetype_has_attributes(self):
        warrior = get_archetype("warrior")
        assert warrior is not None
        assert "str" in warrior["attributes"]
        assert "dex" in warrior["attributes"]

    def test_archetype_has_abilities(self):
        mage = get_archetype("mage")
        assert mage is not None
        assert len(mage["abilities"]) >= 3

    def test_archetype_has_sensors(self):
        rogue = get_archetype("rogue")
        assert rogue is not None
        assert "hp" in rogue["suggested_sensors"]

    def test_all_npc_roles_exist(self):
        assert len(NPC_ROLES) >= 6
        for name in ("guard", "merchant", "innkeeper", "villain", "companion"):
            assert name in NPC_ROLES

    def test_npc_role_has_personality(self):
        guard = get_npc_role("guard")
        assert guard is not None
        assert guard["default_personality"]

    def test_list_functions(self):
        assert "warrior" in list_archetypes()
        assert "guard" in list_npc_roles()

    def test_case_insensitive(self):
        assert get_archetype("WARRIOR") is not None
        assert get_npc_role("GUARD") is not None


# ---------------------------------------------------------------------------
# Architect Persona
# ---------------------------------------------------------------------------


class TestArchitectPersona:
    def test_persona_registered(self):
        from maxim.simulation.personas import SIMULATION_PERSONAS
        assert "adventure_architect" in SIMULATION_PERSONAS

    def test_persona_has_tools_listed(self):
        from maxim.simulation.personas import SIMULATION_PERSONAS
        architect = SIMULATION_PERSONAS["adventure_architect"]
        assert "browse_components" in architect.context_prompt
        assert "design_entity" in architect.context_prompt
        assert "emit_campaign" in architect.context_prompt

    def test_persona_count_updated(self):
        from maxim.simulation.personas import SIMULATION_PERSONAS
        assert len(SIMULATION_PERSONAS) == 10  # 9 + architect


# ---------------------------------------------------------------------------
# Architect Tools
# ---------------------------------------------------------------------------


class TestArchitectTools:
    def test_browse_components_tool(self, tmp_path):
        from maxim.simulation.tools_dm import BrowseComponentsTool
        from maxim.embodiment.component_registry import ComponentRegistry

        # Create test component
        d = tmp_path / "weapons"
        d.mkdir()
        (d / "sword.yaml").write_text(textwrap.dedent("""\
            component:
              name: sword
              category: weapons
              tags: [weapon, melee]
            entity:
              name: sword
              entity_type: weapon
        """))

        registry = ComponentRegistry(search_paths=[tmp_path], include_defaults=False)
        tool = BrowseComponentsTool(component_registry=registry)
        result = tool.execute(category="weapons")
        assert result["count"] >= 1
        assert result["components"][0]["ref"] == "weapons/sword"

    def test_browse_encounters_tool(self, tmp_path):
        from maxim.simulation.tools_dm import BrowseEncountersTool
        from maxim.simulation.encounter_library import EncounterLibrary

        d = tmp_path / "combat"
        d.mkdir()
        (d / "fight.yaml").write_text(textwrap.dedent("""\
            encounter:
              name: fight
              tags: [combat]
              scene: "Enemies attack!"
              choices: [fight, flee]
        """))

        library = EncounterLibrary(search_paths=[tmp_path], include_defaults=False)
        tool = BrowseEncountersTool(encounter_library=library)
        result = tool.execute(category="combat")
        assert result["count"] >= 1

    def test_design_entity_tool(self):
        from maxim.simulation.tools_dm import DesignEntityTool

        designer = EntityDesigner()
        tool = DesignEntityTool(entity_designer=designer)
        result = tool.execute(description="a fierce dragon", entity_type="creature")
        assert result["valid"] is True
        assert "spec" in result

    def test_emit_campaign_tool(self, tmp_path):
        from maxim.simulation.tools_dm import EmitCampaignTool

        tool = EmitCampaignTool(output_dir=str(tmp_path))
        result = tool.execute(campaign_yaml="campaign:\n  name: test\n")
        assert result["success"] is True
        assert tool.last_emitted == "campaign:\n  name: test\n"

    def test_emit_campaign_no_content(self):
        from maxim.simulation.tools_dm import EmitCampaignTool

        tool = EmitCampaignTool()
        result = tool.execute()
        assert not result.success
