"""Tests for maxim.embodiment.component_registry — SEM component catalog."""

from __future__ import annotations

import textwrap

import pytest

from maxim.embodiment.component_registry import (
    ComponentRegistry,
    deep_merge,
    _read_component_header,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def components_dir(tmp_path):
    """Create a temporary component directory with test YAML files."""
    weapons = tmp_path / "weapons"
    weapons.mkdir()

    npcs = tmp_path / "npcs"
    npcs.mkdir()

    # Simple weapon
    (weapons / "iron_sword.yaml").write_text(textwrap.dedent("""\
        component:
          name: iron_sword
          tags: [weapon, melee]
          category: weapons
        entity:
          name: iron_sword
          entity_type: weapon
          sensors:
            durability:
              unit: points
              range: [0, 100]
              initial: 80
            sharpness:
              unit: ratio
              range: [0, 1]
              initial: 0.7
          modulators:
            combat:
              affordances:
                slash:
                  params: {target: str}
                  description: "Slash with sword"
          failure_modes:
            - name: broken
              trigger: {field: durability, op: "<=", value: 0, pain: 0.8}
    """))

    # Base NPC template
    (npcs / "base_npc.yaml").write_text(textwrap.dedent("""\
        component:
          name: base_npc
          tags: [npc, humanoid]
          category: npcs
        entity:
          name: base_npc
          entity_type: npc
          sensors:
            hp:
              unit: points
              range: [0, 20]
              initial: 20
            mood:
              unit: ratio
              range: [-1, 1]
              initial: 0.0
          modulators:
            social:
              affordances:
                speak:
                  params: {message: str}
                  description: "Say something"
    """))

    # Guard extends base_npc
    (npcs / "guard.yaml").write_text(textwrap.dedent("""\
        component:
          name: guard
          tags: [npc, humanoid, military]
          category: npcs
          extends: npcs/base_npc
        entity:
          name: guard
          entity_type: npc
          sensors:
            hp:
              initial: 15
            suspicion:
              unit: ratio
              range: [0, 1]
              initial: 0.3
          modulators:
            combat:
              affordances:
                attack:
                  params: {target: str}
                  description: "Attack a target"
    """))

    return tmp_path


@pytest.fixture
def registry(components_dir):
    """Create a registry using ONLY the test components directory."""
    return ComponentRegistry(search_paths=[components_dir], include_defaults=False)


# ---------------------------------------------------------------------------
# deep_merge
# ---------------------------------------------------------------------------

class TestDeepMerge:
    def test_shallow_override(self):
        assert deep_merge({"a": 1}, {"a": 2}) == {"a": 2}

    def test_add_new_key(self):
        assert deep_merge({"a": 1}, {"b": 2}) == {"a": 1, "b": 2}

    def test_nested_merge(self):
        base = {"sensors": {"hp": {"range": [0, 20], "initial": 20}}}
        override = {"sensors": {"hp": {"initial": 15}}}
        result = deep_merge(base, override)
        assert result == {"sensors": {"hp": {"range": [0, 20], "initial": 15}}}

    def test_nested_add_sensor(self):
        base = {"sensors": {"hp": {"range": [0, 20]}}}
        override = {"sensors": {"trust": {"range": [0, 1]}}}
        result = deep_merge(base, override)
        assert "hp" in result["sensors"]
        assert "trust" in result["sensors"]

    def test_list_replaced_not_merged(self):
        base = {"tags": ["a", "b"]}
        override = {"tags": ["c"]}
        assert deep_merge(base, override) == {"tags": ["c"]}

    def test_does_not_mutate_inputs(self):
        base = {"a": {"b": 1}}
        override = {"a": {"c": 2}}
        base_copy = {"a": {"b": 1}}
        deep_merge(base, override)
        assert base == base_copy


# ---------------------------------------------------------------------------
# _read_component_header
# ---------------------------------------------------------------------------

class TestReadComponentHeader:
    def test_reads_explicit_header(self, tmp_path):
        p = tmp_path / "test.yaml"
        p.write_text("component:\n  name: test\n  tags: [a]\nentity:\n  name: test\n")
        header = _read_component_header(p)
        assert header is not None
        assert header["name"] == "test"

    def test_legacy_body_format(self, tmp_path):
        p = tmp_path / "robot_arm.yaml"
        p.write_text("name: robot\nbody:\n  name: arm\n  entity_type: arm\n")
        header = _read_component_header(p)
        assert header is not None
        assert header["name"] == "robot"
        assert header.get("_legacy") is True

    def test_legacy_world_entities(self, tmp_path):
        p = tmp_path / "demo.yaml"
        p.write_text("world_entities:\n  - name: sword\n")
        header = _read_component_header(p)
        assert header is not None
        assert header.get("_legacy") is True

    def test_returns_none_for_unrelated_yaml(self, tmp_path):
        p = tmp_path / "other.yaml"
        p.write_text("key: value\nother: stuff\n")
        assert _read_component_header(p) is None

    def test_returns_none_for_invalid_yaml(self, tmp_path):
        p = tmp_path / "bad.yaml"
        p.write_text(":::invalid:::\n  - [broken")
        assert _read_component_header(p) is None


# ---------------------------------------------------------------------------
# ComponentRegistry — discovery & indexing
# ---------------------------------------------------------------------------

class TestRegistryDiscovery:
    def test_discovers_components(self, registry):
        assert registry.has("weapons/iron_sword")
        assert registry.has("npcs/base_npc")
        assert registry.has("npcs/guard")

    def test_list_categories(self, registry):
        cats = registry.list_categories()
        assert "weapons" in cats
        assert "npcs" in cats

    def test_list_refs(self, registry):
        refs = registry.list_refs()
        assert "weapons/iron_sword" in refs
        assert "npcs/guard" in refs

    def test_list_refs_filtered(self, registry):
        weapon_refs = registry.list_refs(category="weapons")
        assert "weapons/iron_sword" in weapon_refs
        assert "npcs/guard" not in weapon_refs

    def test_has_returns_false_for_missing(self, registry):
        assert not registry.has("weapons/nonexistent")


# ---------------------------------------------------------------------------
# ComponentRegistry — get & resolve
# ---------------------------------------------------------------------------

class TestRegistryGet:
    def test_get_simple_component(self, registry):
        spec = registry.get("weapons/iron_sword")
        assert "entity" in spec
        assert spec["entity"]["name"] == "iron_sword"
        assert "durability" in spec["entity"]["sensors"]

    def test_get_returns_deep_copy(self, registry):
        spec1 = registry.get("weapons/iron_sword")
        spec2 = registry.get("weapons/iron_sword")
        # Mutating one should not affect the other
        spec1["entity"]["name"] = "modified"
        assert spec2["entity"]["name"] == "iron_sword"

    def test_get_with_extends(self, registry):
        spec = registry.get("npcs/guard")
        entity = spec["entity"]
        # Should have parent's sensors (hp with overridden initial, mood from parent)
        assert "hp" in entity["sensors"]
        assert entity["sensors"]["hp"]["initial"] == 15  # Overridden
        assert entity["sensors"]["hp"]["range"] == [0, 20]  # From parent
        assert "mood" in entity["sensors"]  # From parent
        # Should have child's new sensor
        assert "suspicion" in entity["sensors"]
        # Should have both parent's and child's modulators
        assert "social" in entity["modulators"]  # From parent
        assert "combat" in entity["modulators"]  # From child

    def test_get_missing_raises(self, registry):
        with pytest.raises(KeyError, match="not found"):
            registry.get("weapons/nonexistent")

    def test_circular_extends_raises(self, tmp_path):
        d = tmp_path / "loop"
        d.mkdir()
        (d / "a.yaml").write_text(textwrap.dedent("""\
            component:
              name: a
              category: loop
              extends: loop/b
            entity:
              name: a
              entity_type: test
        """))
        (d / "b.yaml").write_text(textwrap.dedent("""\
            component:
              name: b
              category: loop
              extends: loop/a
            entity:
              name: b
              entity_type: test
        """))
        reg = ComponentRegistry(search_paths=[tmp_path])
        with pytest.raises(ValueError, match="Circular inheritance"):
            reg.get("loop/a")

    def test_missing_parent_raises(self, tmp_path):
        d = tmp_path / "orphan"
        d.mkdir()
        (d / "child.yaml").write_text(textwrap.dedent("""\
            component:
              name: child
              category: orphan
              extends: orphan/missing_parent
            entity:
              name: child
              entity_type: test
        """))
        reg = ComponentRegistry(search_paths=[tmp_path])
        with pytest.raises(KeyError, match="missing_parent"):
            reg.get("orphan/child")


# ---------------------------------------------------------------------------
# ComponentRegistry — instantiate
# ---------------------------------------------------------------------------

class TestRegistryInstantiate:
    def test_instantiate_returns_entity(self, registry):
        from maxim.embodiment.sem import Entity
        entity = registry.instantiate("weapons/iron_sword")
        assert isinstance(entity, Entity)
        assert entity.name == "iron_sword"
        assert entity.entity_type == "weapon"

    def test_instantiate_creates_sensors(self, registry):
        entity = registry.instantiate("weapons/iron_sword")
        assert "durability" in entity.sensors
        assert "sharpness" in entity.sensors

    def test_instantiate_independent_instances(self, registry):
        e1 = registry.instantiate("weapons/iron_sword")
        e2 = registry.instantiate("weapons/iron_sword")
        # They should be separate objects
        assert e1 is not e2
        # Mutating one should not affect the other
        e1.vital_metrics["durability"] = 0.0
        assert e2.vital_metrics["durability"] != 0.0

    def test_instantiate_with_overrides(self, registry):
        entity = registry.instantiate("npcs/guard", name="captain")
        assert entity.name == "captain"

    def test_instantiate_with_extends(self, registry):
        entity = registry.instantiate("npcs/guard")
        # Should have parent + child sensors
        assert "hp" in entity.sensors
        assert "mood" in entity.sensors  # From base_npc
        assert "suspicion" in entity.sensors  # Guard-specific
        # Should have both modulators
        assert "social" in entity.modulators
        assert "combat" in entity.modulators

    def test_instantiate_vital_metrics_set(self, registry):
        entity = registry.instantiate("weapons/iron_sword")
        assert entity.vital_metrics["durability"] == 80.0
        assert entity.vital_metrics["sharpness"] == 0.7


# ---------------------------------------------------------------------------
# ComponentRegistry — query
# ---------------------------------------------------------------------------

class TestRegistryQuery:
    def test_query_by_category(self, registry):
        results = registry.query(category="weapons")
        assert len(results) == 1
        assert results[0].ref == "weapons/iron_sword"

    def test_query_by_tags(self, registry):
        results = registry.query(tags=["military"])
        assert len(results) == 1
        assert results[0].ref == "npcs/guard"

    def test_query_by_multiple_tags(self, registry):
        results = registry.query(tags=["npc", "humanoid"])
        refs = [r.ref for r in results]
        assert "npcs/base_npc" in refs
        assert "npcs/guard" in refs

    def test_query_by_sensor(self, registry):
        results = registry.query(has_sensor="durability")
        assert len(results) == 1
        assert results[0].ref == "weapons/iron_sword"

    def test_query_by_affordance(self, registry):
        results = registry.query(has_affordance="slash")
        assert len(results) == 1
        assert results[0].ref == "weapons/iron_sword"

    def test_query_no_matches(self, registry):
        assert registry.query(tags=["nonexistent"]) == []


# ---------------------------------------------------------------------------
# ComponentRegistry — register (manual)
# ---------------------------------------------------------------------------

class TestRegistryRegister:
    def test_register_inline(self, registry):
        registry.register("misc/test_item", {
            "component": {"name": "test_item", "tags": ["test"]},
            "entity": {"name": "test_item", "entity_type": "item"},
        })
        assert registry.has("misc/test_item")
        spec = registry.get("misc/test_item")
        assert spec["entity"]["name"] == "test_item"


# ---------------------------------------------------------------------------
# ComponentRegistry — search path priority
# ---------------------------------------------------------------------------

class TestSearchPathPriority:
    def test_higher_priority_shadows(self, tmp_path):
        low = tmp_path / "low" / "items"
        low.mkdir(parents=True)
        high = tmp_path / "high" / "items"
        high.mkdir(parents=True)

        (low / "potion.yaml").write_text(textwrap.dedent("""\
            component:
              name: potion
              category: items
            entity:
              name: potion
              entity_type: item
              sensors:
                charges: { unit: count, range: [0, 3], initial: 1 }
        """))
        (high / "potion.yaml").write_text(textwrap.dedent("""\
            component:
              name: potion
              category: items
            entity:
              name: potion
              entity_type: item
              sensors:
                charges: { unit: count, range: [0, 3], initial: 3 }
        """))

        # high-priority path listed first
        reg = ComponentRegistry(search_paths=[high.parent, low.parent], include_defaults=False)
        spec = reg.get("items/potion")
        assert spec["entity"]["sensors"]["charges"]["initial"] == 3


# ---------------------------------------------------------------------------
# Integration: bundled components load correctly
# ---------------------------------------------------------------------------

class TestBundledComponents:
    def test_bundled_components_exist(self):
        """Verify the seed components shipped in _data/ are discoverable."""
        from maxim.utils.paths import bundled_data
        components_dir = bundled_data() / "components"
        if not components_dir.is_dir():
            pytest.skip("Bundled components not found (not installed?)")

        reg = ComponentRegistry(search_paths=[components_dir])
        assert reg.has("weapons/rusty_sword")
        assert reg.has("npcs/guard")
        assert reg.has("npcs/base_humanoid")
        assert reg.has("creatures/wolf")

    def test_bundled_guard_extends_base(self):
        """Verify inheritance works on real bundled components."""
        from maxim.utils.paths import bundled_data
        components_dir = bundled_data() / "components"
        if not components_dir.is_dir():
            pytest.skip("Bundled components not found")

        reg = ComponentRegistry(search_paths=[components_dir])
        spec = reg.get("npcs/guard")
        entity = spec["entity"]
        # Guard extends base_humanoid — should have base sensors + guard sensors
        assert "hp" in entity["sensors"]
        assert "mood" in entity["sensors"]  # From base_humanoid
        assert "suspicion" in entity["sensors"]  # Guard-specific
