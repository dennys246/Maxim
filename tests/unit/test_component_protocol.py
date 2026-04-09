"""SEM Protocol Tests — validate every bundled component structurally.

Runs the 8 foundry protocol tests (from asset_foundry_plan.md) against
all bundled components. These are fast (no LLM, no simulation) and catch
spec issues before publication.

Tests per component:
1. Instantiation — _parse_entity() succeeds, valid entity_type
2. Sensor R/W — every sensor initializes within range, responds to writes
3. Affordance enumeration — all affordances have descriptions + typed params
4. Tool generation — tool_bridge generates tools without collisions
5. Failure mode triggers — trigger fields reference real sensors, valid ops
6. Composition — body_part/weapon types can reparent/detach
7. Cascade compatibility — CascadeResolver can read all sensors
8. Inheritance — children of extends parents have all parent sensors
"""

from __future__ import annotations

import pytest

from maxim.embodiment.component_registry import ComponentRegistry


@pytest.fixture(scope="module")
def registry():
    return ComponentRegistry()


@pytest.fixture(scope="module")
def all_components(registry):
    # Exclude embodiment/ category — those are multi-entity test scenes,
    # not standalone SEM components (missing entity_type, different structure)
    return [c for c in registry.query() if c.category != "embodiment"]


@pytest.fixture(scope="module")
def component_refs(all_components):
    return [c.ref for c in all_components]


# ---------------------------------------------------------------------------
# Test 1 — Instantiation
# ---------------------------------------------------------------------------


class TestInstantiation:
    def test_all_components_instantiate(self, registry, component_refs):
        """Every bundled component instantiates without error."""
        for ref in component_refs:
            entity = registry.instantiate(ref)
            assert entity is not None, f"{ref}: instantiate returned None"
            assert entity.name, f"{ref}: entity has no name"
            assert entity.entity_type, f"{ref}: entity has no entity_type"

    def test_entity_types_are_valid(self, registry, component_refs):
        valid_types = {"npc", "creature", "weapon", "body_part", "environment", "character", "robot", "item", "vehicle"}
        for ref in component_refs:
            entity = registry.instantiate(ref)
            assert entity.entity_type in valid_types, (
                f"{ref}: entity_type '{entity.entity_type}' not in {valid_types}"
            )


# ---------------------------------------------------------------------------
# Test 2 — Sensor Read/Write Cycle
# ---------------------------------------------------------------------------


class TestSensorReadWrite:
    def test_all_sensors_readable(self, registry, component_refs):
        """Every sensor reads without error and returns a numeric value."""
        for ref in component_refs:
            entity = registry.instantiate(ref)
            for sensor_name, sensor in entity.sensors.items():
                reading = sensor.read()
                assert reading is not None, f"{ref}.{sensor_name}: read() returned None"
                val = reading.value if hasattr(reading, "value") else reading
                assert isinstance(val, (int, float)), (
                    f"{ref}.{sensor_name}: read value is {type(val).__name__}, expected numeric"
                )

    def test_sensors_initialize_within_range(self, registry, all_components):
        """Sensor initial values are within their declared ranges."""
        for info in all_components:
            spec = registry.get(info.ref)
            entity_spec = spec.get("entity", spec)
            sensors_spec = entity_spec.get("sensors", {})
            entity = registry.instantiate(info.ref)

            for sensor_name, sensor in entity.sensors.items():
                sensor_def = sensors_spec.get(sensor_name, {})
                if isinstance(sensor_def, dict) and "range" in sensor_def:
                    range_min, range_max = sensor_def["range"]
                    reading = sensor.read()
                    val = reading.value if hasattr(reading, "value") else reading
                    assert range_min <= val <= range_max, (
                        f"{info.ref}.{sensor_name}: initial value {val} outside "
                        f"range [{range_min}, {range_max}]"
                    )

    def test_sensors_respond_to_writes(self, registry, component_refs):
        """Writing to vital_metrics updates sensor readings."""
        for ref in component_refs:
            entity = registry.instantiate(ref)
            for sensor_name, sensor in entity.sensors.items():
                entity.vital_metrics[sensor_name] = 42.0
                reading = sensor.read()
                val = reading.value if hasattr(reading, "value") else reading
                assert val == pytest.approx(42.0), (
                    f"{ref}.{sensor_name}: wrote 42.0 but read {val}"
                )


# ---------------------------------------------------------------------------
# Test 3 — Affordance Enumeration
# ---------------------------------------------------------------------------


class TestAffordances:
    def test_modulators_have_affordances(self, registry, all_components):
        """Every modulator has at least one affordance."""
        for info in all_components:
            spec = registry.get(info.ref)
            entity_spec = spec.get("entity", spec)
            modulators = entity_spec.get("modulators", {})
            for mod_name, mod in modulators.items():
                affordances = mod.get("affordances", {})
                assert len(affordances) >= 1, (
                    f"{info.ref}: modulator '{mod_name}' has no affordances"
                )

    def test_affordances_have_descriptions(self, registry, all_components):
        """Every affordance has a non-empty description."""
        for info in all_components:
            spec = registry.get(info.ref)
            entity_spec = spec.get("entity", spec)
            for mod_name, mod in entity_spec.get("modulators", {}).items():
                for aff_name, aff in mod.get("affordances", {}).items():
                    desc = aff.get("description", "")
                    assert desc.strip(), (
                        f"{info.ref}: {mod_name}.{aff_name} has no description"
                    )

    def test_affordance_params_are_typed(self, registry, all_components):
        """Affordance params have string type annotations."""
        valid_types = {"str", "float", "int", "bool"}
        for info in all_components:
            spec = registry.get(info.ref)
            entity_spec = spec.get("entity", spec)
            for mod_name, mod in entity_spec.get("modulators", {}).items():
                for aff_name, aff in mod.get("affordances", {}).items():
                    for param_name, param_type in aff.get("params", {}).items():
                        assert param_type in valid_types, (
                            f"{info.ref}: {aff_name}.{param_name} has type "
                            f"'{param_type}', expected one of {valid_types}"
                        )


# ---------------------------------------------------------------------------
# Test 4 — Tool Generation
# ---------------------------------------------------------------------------


class TestToolGeneration:
    @pytest.fixture()
    def tool_registry(self):
        from maxim.tools.registry import ToolRegistry
        return ToolRegistry()

    def test_all_components_generate_tools(self, registry, component_refs, tool_registry):
        """tool_bridge generates at least one tool per component."""
        from maxim.embodiment.tool_bridge import generate_tools_for_entity

        for ref in component_refs:
            entity = registry.instantiate(ref)
            tools = generate_tools_for_entity(entity, registry=tool_registry)
            assert len(tools) >= 1, f"{ref}: generated 0 tools"

    def test_no_tool_name_collisions_within_entity(self, registry, component_refs):
        """No tool name collisions within a single entity."""
        from maxim.embodiment.tool_bridge import generate_tools_for_entity
        from maxim.tools.registry import ToolRegistry

        for ref in component_refs:
            entity = registry.instantiate(ref)
            tr = ToolRegistry()
            tools = generate_tools_for_entity(entity, registry=tr)
            names = [t.name for t in tools]
            assert len(names) == len(set(names)), (
                f"{ref}: tool name collision: {[n for n in names if names.count(n) > 1]}"
            )


# ---------------------------------------------------------------------------
# Test 5 — Failure Mode Triggers
# ---------------------------------------------------------------------------


class TestFailureModes:
    def test_failure_triggers_reference_valid_sensors(self, registry, all_components):
        """Failure mode trigger fields reference sensors that exist."""
        valid_ops = {"<", "<=", ">", ">=", "==", "!="}
        for info in all_components:
            spec = registry.get(info.ref)
            entity_spec = spec.get("entity", spec)
            sensors = entity_spec.get("sensors", {})
            for failure in entity_spec.get("failure_modes", []):
                trigger = failure.get("trigger", {})
                field = trigger.get("field", "")
                assert field in sensors, (
                    f"{info.ref}: failure trigger references missing sensor '{field}'. "
                    f"Available: {list(sensors.keys())}"
                )
                op = trigger.get("op", "")
                assert op in valid_ops, (
                    f"{info.ref}: failure trigger op '{op}' not in {valid_ops}"
                )
                pain = trigger.get("pain", 0)
                assert 0 <= pain <= 1, (
                    f"{info.ref}: failure pain {pain} outside [0, 1]"
                )


# ---------------------------------------------------------------------------
# Test 6 — Composition (parent-child)
# ---------------------------------------------------------------------------


class TestComposition:
    def test_attachable_types_can_reparent(self, registry, all_components):
        """body_part and weapon entities can attach to a parent."""
        from maxim.embodiment.sem import Entity

        attachable_types = {"body_part", "weapon"}
        for info in all_components:
            entity = registry.instantiate(info.ref)
            if entity.entity_type not in attachable_types:
                continue

            parent = Entity(name="test_host", entity_type="character")
            entity.reparent(parent)
            assert entity.parent is parent, f"{info.ref}: reparent failed"
            assert entity in parent.children, f"{info.ref}: not in parent's children"

            entity.detach()
            assert entity.parent is None, f"{info.ref}: detach failed"
            assert entity not in parent.children, f"{info.ref}: still in parent's children"


# ---------------------------------------------------------------------------
# Test 7 — Cascade Compatibility
# ---------------------------------------------------------------------------


class TestCascadeCompatibility:
    def test_all_sensors_readable_via_cascade_resolver(self, registry, component_refs):
        """CascadeResolver can read every sensor on every component."""
        from maxim.simulation.dm_runtime import CascadeResolver

        for ref in component_refs:
            entity = registry.instantiate(ref)
            resolver = CascadeResolver({entity.name: entity})
            for sensor_name in entity.sensors:
                val = resolver._read_sensor(f"{entity.name}.{sensor_name}", {})
                assert val is not None, (
                    f"{ref}: CascadeResolver can't read {entity.name}.{sensor_name}"
                )


# ---------------------------------------------------------------------------
# Test 8 — Inheritance
# ---------------------------------------------------------------------------


class TestInheritance:
    def test_children_have_all_parent_sensors(self, registry, all_components):
        """Components with extends have all parent sensors."""
        for info in all_components:
            if not info.extends:
                continue

            try:
                parent_spec = registry.get(info.extends)
            except KeyError:
                pytest.fail(f"{info.ref}: extends '{info.extends}' not found in registry")

            parent_entity_spec = parent_spec.get("entity", parent_spec)
            parent_sensors = set(parent_entity_spec.get("sensors", {}).keys())

            child_spec = registry.get(info.ref)
            child_entity_spec = child_spec.get("entity", child_spec)
            child_sensors = set(child_entity_spec.get("sensors", {}).keys())

            missing = parent_sensors - child_sensors
            assert not missing, (
                f"{info.ref} extends {info.extends} but is missing sensors: {missing}"
            )
