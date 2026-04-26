"""Tests for body archetype system (Phase 3 of SEM World Enrichment).

Covers:
- Archetype YAML loading and parsing
- Archetype validation against component specs
- Archetype scaffold generation
- LLM prompt formatting
- ImaginationDesigner archetype inference and scaffolding
- Seed component archetype field validation (parametrized)
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from maxim.embodiment.archetype import (
    archetype_scaffold,
    format_archetype_prompt,
    get_archetype,
    load_archetypes,
    validate_component_archetype,
)

# Clear module-level cache before each test module run
import maxim.embodiment.archetype as _arch_mod


@pytest.fixture(autouse=True)
def _clear_archetype_cache():
    """Clear the module-level archetype cache between tests."""
    _arch_mod._cache = None
    yield
    _arch_mod._cache = None


# ---------------------------------------------------------------------------
# Archetype loading
# ---------------------------------------------------------------------------


class TestArchetypeLoading:
    def test_load_archetypes_from_default_dir(self):
        """All 7 archetypes load from _data/components/archetypes/."""
        archetypes = load_archetypes()
        assert len(archetypes) >= 7
        expected = {"humanoid", "quadruped", "serpentine", "avian", "vehicle", "machine", "environmental"}
        assert expected.issubset(set(archetypes.keys()))

    def test_archetype_has_required_fields(self):
        archetypes = load_archetypes()
        for name, arch in archetypes.items():
            assert arch.name == name
            assert arch.description
            assert len(arch.body_parts) > 0, f"{name} has no body_parts"

    def test_get_archetype(self):
        arch = get_archetype("humanoid")
        assert arch is not None
        assert arch.name == "humanoid"
        assert "head" in arch.body_parts
        assert "torso" in arch.body_parts

    def test_get_archetype_unknown(self):
        assert get_archetype("nonexistent") is None

    def test_humanoid_has_expected_parts(self):
        arch = get_archetype("humanoid")
        assert arch is not None
        assert "head" in arch.body_parts
        assert "arms" in arch.body_parts
        assert "legs" in arch.body_parts
        assert "communication" in arch.body_parts

    def test_quadruped_has_optional_wings(self):
        arch = get_archetype("quadruped")
        assert arch is not None
        assert "wings" in arch.optional_parts

    def test_all_part_names_includes_optional(self):
        arch = get_archetype("quadruped")
        assert arch is not None
        all_names = arch.all_part_names
        assert "head" in all_names  # required
        assert "wings" in all_names  # optional

    def test_all_affordance_names(self):
        arch = get_archetype("humanoid")
        assert arch is not None
        affs = arch.all_affordance_names
        assert "look" in affs
        assert "move" in affs
        assert "speak" in affs

    def test_caching(self):
        a1 = load_archetypes()
        a2 = load_archetypes()
        assert a1 is a2  # same object (cached)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestArchetypeValidation:
    def test_valid_humanoid_component(self):
        comp = {"archetype": "humanoid"}
        entity = {
            "modulators": {
                "head": {},
                "torso": {},
                "arms": {},
                "legs": {},
                "communication": {},
            }
        }
        warnings = validate_component_archetype(comp, entity)
        assert warnings == []

    def test_extra_modulator_warns(self):
        comp = {"archetype": "humanoid"}
        entity = {
            "modulators": {
                "head": {},
                "tentacles": {},  # Not in humanoid vocabulary
            }
        }
        warnings = validate_component_archetype(comp, entity)
        assert len(warnings) == 1
        assert "tentacles" in warnings[0]

    def test_optional_part_is_valid(self):
        comp = {"archetype": "quadruped"}
        entity = {
            "modulators": {
                "head": {},
                "torso": {},
                "wings": {},  # optional for quadruped
            }
        }
        warnings = validate_component_archetype(comp, entity)
        assert warnings == []

    def test_no_archetype_always_passes(self):
        comp = {}
        entity = {"modulators": {"anything": {}}}
        warnings = validate_component_archetype(comp, entity)
        assert warnings == []

    def test_unknown_archetype_warns(self):
        comp = {"archetype": "nonexistent"}
        entity = {"modulators": {}}
        warnings = validate_component_archetype(comp, entity)
        assert len(warnings) == 1
        assert "Unknown" in warnings[0]


# ---------------------------------------------------------------------------
# Scaffold generation
# ---------------------------------------------------------------------------


class TestArchetypeScaffold:
    def test_humanoid_scaffold(self):
        scaffold = archetype_scaffold("humanoid")
        assert scaffold is not None
        mods = scaffold["modulators"]
        assert "head" in mods
        assert "torso" in mods
        assert "arms" in mods
        assert "legs" in mods

    def test_scaffold_has_sensors_and_affordances(self):
        scaffold = archetype_scaffold("humanoid")
        assert scaffold is not None
        head = scaffold["modulators"]["head"]
        assert "sensors" in head
        assert "affordances" in head
        assert "awareness" in head["sensors"]
        assert "look" in head["affordances"]

    def test_scaffold_unknown_archetype(self):
        assert archetype_scaffold("nonexistent") is None

    def test_scaffold_does_not_include_optional_parts(self):
        scaffold = archetype_scaffold("quadruped")
        assert scaffold is not None
        assert "wings" not in scaffold["modulators"]  # optional, not scaffolded


# ---------------------------------------------------------------------------
# LLM prompt formatting
# ---------------------------------------------------------------------------


class TestFormatArchetypePrompt:
    def test_prompt_contains_all_archetypes(self):
        prompt = format_archetype_prompt()
        assert "humanoid" in prompt
        assert "quadruped" in prompt
        assert "serpentine" in prompt
        assert "avian" in prompt
        assert "vehicle" in prompt
        assert "machine" in prompt
        assert "environmental" in prompt

    def test_prompt_contains_required_parts(self):
        prompt = format_archetype_prompt()
        assert "Required parts:" in prompt

    def test_prompt_contains_optional_parts(self):
        prompt = format_archetype_prompt()
        assert "Optional parts:" in prompt


# ---------------------------------------------------------------------------
# ImaginationDesigner archetype inference
# ---------------------------------------------------------------------------


class TestDesignerArchetypeInference:
    def test_dragon_infers_quadruped(self):
        from maxim.imagination.designer import ImaginationDesigner

        arch = ImaginationDesigner._infer_archetype("fire-breathing dragon", "creatures")
        assert arch == "quadruped"

    def test_snake_infers_serpentine(self):
        from maxim.imagination.designer import ImaginationDesigner

        arch = ImaginationDesigner._infer_archetype("venomous snake", "creatures")
        assert arch == "serpentine"

    def test_robot_infers_machine(self):
        from maxim.imagination.designer import ImaginationDesigner

        arch = ImaginationDesigner._infer_archetype("patrol robot", "creatures")
        assert arch == "machine"

    def test_npc_defaults_to_humanoid(self):
        from maxim.imagination.designer import ImaginationDesigner

        arch = ImaginationDesigner._infer_archetype("mysterious stranger", "npcs")
        assert arch == "humanoid"

    def test_keyword_takes_priority_over_type(self):
        from maxim.imagination.designer import ImaginationDesigner

        # "drone" keyword → machine, even though type is "creatures"
        arch = ImaginationDesigner._infer_archetype("flying drone", "creatures")
        assert arch == "machine"

    def test_unknown_description_unknown_type(self):
        from maxim.imagination.designer import ImaginationDesigner

        arch = ImaginationDesigner._infer_archetype("mysterious glowing orb", None)
        assert arch is None


# ---------------------------------------------------------------------------
# ImaginationDesigner scaffold application
# ---------------------------------------------------------------------------


class TestDesignerScaffoldApplication:
    def test_scaffold_fills_empty_modulators(self):
        from maxim.imagination.designer import ImaginationDesigner

        entity_spec = {
            "name": "test",
            "sensors": {},
            "modulators": {},
        }
        result = ImaginationDesigner._apply_archetype_scaffold(entity_spec, "humanoid")
        assert "head" in result["modulators"]
        assert "arms" in result["modulators"]

    def test_scaffold_preserves_existing_modulators(self):
        from maxim.imagination.designer import ImaginationDesigner

        entity_spec = {
            "name": "test",
            "sensors": {},
            "modulators": {
                "head": {"affordances": {"laser_eyes": {"params": {}}}},
            },
        }
        result = ImaginationDesigner._apply_archetype_scaffold(entity_spec, "humanoid")
        # Custom head preserved
        assert "laser_eyes" in result["modulators"]["head"]["affordances"]
        # Other parts added from scaffold
        assert "torso" in result["modulators"]

    def test_scaffold_skipped_when_enough_modulators(self):
        from maxim.imagination.designer import ImaginationDesigner

        entity_spec = {
            "name": "test",
            "sensors": {},
            "modulators": {
                "head": {},
                "torso": {},
                "arms": {},
                "legs": {},
                "communication": {},
                "extra": {},
            },
        }
        result = ImaginationDesigner._apply_archetype_scaffold(entity_spec, "humanoid")
        # Already has 6 modulators >= 5 (humanoid required), scaffold skipped
        assert "extra" in result["modulators"]

    def test_scaffold_unknown_archetype_returns_unchanged(self):
        from maxim.imagination.designer import ImaginationDesigner

        entity_spec = {"name": "test", "sensors": {}, "modulators": {}}
        result = ImaginationDesigner._apply_archetype_scaffold(entity_spec, "nonexistent")
        assert result == entity_spec


# ---------------------------------------------------------------------------
# Seed component archetype validation (parametrized)
# ---------------------------------------------------------------------------


_SEED_COMPONENTS_DIR = Path(__file__).resolve().parent.parent.parent / "src" / "maxim" / "_data" / "components"


def _collect_seed_yamls():
    """Collect all seed component YAMLs (excluding archetypes)."""
    yamls = []
    for cat_dir in sorted(_SEED_COMPONENTS_DIR.iterdir()):
        if not cat_dir.is_dir() or cat_dir.name == "archetypes":
            continue
        for yaml_file in sorted(cat_dir.glob("*.yaml")):
            yamls.append(yaml_file)
    return yamls


@pytest.mark.parametrize(
    "yaml_path",
    _collect_seed_yamls(),
    ids=lambda p: f"{p.parent.name}/{p.stem}",
)
def test_seed_component_has_valid_archetype(yaml_path: Path):
    """Every seed component with an archetype field names a valid archetype."""
    with open(yaml_path) as f:
        raw = yaml.safe_load(f)
    if raw is None:
        pytest.skip("Empty YAML file")

    component = raw.get("component", {})
    archetype_name = component.get("archetype")
    if archetype_name is None:
        pytest.skip("No archetype field")

    arch = get_archetype(archetype_name)
    assert arch is not None, f"Unknown archetype '{archetype_name}' in {yaml_path.name}"
