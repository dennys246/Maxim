"""Auto-tool generation from SEM entity trees.

Generates three types of tools from Entity trees and registers them
in the ToolRegistry:
- ``SensorReadTool`` — read one sensor on an entity
- ``ModulatorAffordanceTool`` — execute one affordance on a modulator
- ``EntitySenseTool`` — read ALL sensors on an entity at once

Tool names use progressive prefixing to avoid collisions:
1. ``{entity.name}_{affordance}`` (default, LLM-friendly)
2. ``{parent}_{entity}_{affordance}`` (on collision)
3. Full path (last resort)
"""

from __future__ import annotations

import logging
from typing import Any

from maxim.embodiment.sem import (
    AffordanceSchema,
    Entity,
    Modulator,
    Sensor,
)
from maxim.tools.base import Tool, ToolOutput
from maxim.tools.registry import ToolRegistry

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Name resolution
# ---------------------------------------------------------------------------


def _resolve_tool_name(
    base_name: str,
    entity: Entity,
    existing_names: set[str],
) -> str:
    """Find a unique tool name by progressively prepending parent names.

    Tries: base_name -> parent_base_name -> ... -> full_path_base_name.
    """
    candidate = base_name
    ancestor = entity.parent
    while candidate in existing_names:
        if ancestor is None:
            raise ValueError(f"Cannot resolve unique tool name for {base_name!r} on {entity.full_path!r}")
        candidate = f"{ancestor.name}_{candidate}"
        ancestor = ancestor.parent
    return candidate


# ---------------------------------------------------------------------------
# Tool classes
# ---------------------------------------------------------------------------


class SensorReadTool(Tool):
    """Auto-generated tool: read a sensor on an entity."""

    def __init__(self, entity: Entity, sensor: Sensor, tool_name: str) -> None:
        self.name = tool_name
        self.description = (
            f"Read the {sensor.name} sensor on {entity.name} ({entity.entity_type}). Returns value in {sensor.unit}."
        )
        self.input_schema: dict[str, Any] = {}
        self._entity = entity
        self._sensor = sensor
        super().__init__()

    def execute(self, **kwargs: Any) -> Any:
        reading = self._sensor.read()
        return {
            "entity": reading.entity_name,
            "sensor": reading.sensor_name,
            "value": reading.value,
            "unit": reading.unit,
        }


class ModulatorAffordanceTool(Tool):
    """Auto-generated tool: execute one affordance on a modulator.

    After execution, reads back entity sensor state, evaluates failure
    modes immediately (don't wait for 1Hz poll), and feeds the actual
    sensor deltas to Cerebellum for forward model training.
    """

    def __init__(
        self,
        entity: Entity,
        modulator: Modulator,
        affordance_name: str,
        schema: AffordanceSchema,
        tool_name: str,
        embodiment: Any = None,
        cerebellum: Any = None,
    ) -> None:
        self.name = tool_name
        self.description = (
            schema.description or f"Execute {affordance_name} on {entity.name} via {modulator.name} modulator."
        )
        self.input_schema: dict[str, Any] = dict(schema.params)
        self.timeout = schema.timeout
        self._entity = entity
        self._modulator = modulator
        self._affordance_name = affordance_name
        self._embodiment = embodiment
        self._cerebellum = cerebellum
        super().__init__()

    def execute(self, **kwargs: Any) -> Any:
        result = self._modulator.execute(self._affordance_name, kwargs)
        if not result.success:
            return ToolOutput(success=False, error=result.error)

        # Read back entity sensor state after action (cascade effects)
        entity_state = {}
        try:
            for sensor_name, sensor in self._entity.sensors.items():
                reading = sensor.read()
                if isinstance(reading.value, (int, float)):
                    entity_state[sensor_name] = reading.value
        except Exception:
            pass

        # Immediate failure evaluation — don't wait for 1Hz poll
        active_failures: list[dict[str, Any]] = []
        if self._embodiment is not None:
            try:
                failure_events = self._embodiment.evaluate_failures()
                active_failures = [
                    {"name": ev.failure_name, "entity": ev.entity_path, "pain": ev.pain_intensity}
                    for ev in failure_events
                ]
            except Exception:
                pass

        # Cerebellum observes cascade outcome for forward model training
        if self._cerebellum is not None and entity_state:
            try:
                sensor_ranges = {}
                for sname, sensor in self._entity.sensors.items():
                    schema = getattr(sensor, "reading_schema", None)
                    if schema and isinstance(schema, dict):
                        r = schema.get("range")
                        if r and len(r) == 2:
                            sensor_ranges[sname] = r
                self._cerebellum.observe_from_action(
                    entity_path=self._entity.full_path,
                    modulator=self._modulator.name,
                    affordance=self._affordance_name,
                    params=kwargs,
                    actual_sensors=entity_state,
                    sensor_ranges=sensor_ranges,
                )
                try:
                    from maxim.simulation.sim_logger import sim_cerebellum

                    conf = self._cerebellum.get_confidence(
                        self._entity.full_path,
                        self._modulator.name,
                        self._affordance_name,
                        kwargs,
                        sensor_ranges,
                    )
                    sim_cerebellum(self._entity.full_path, self._affordance_name, conf)
                except Exception:
                    pass
            except Exception:
                pass

        output_dict: dict[str, Any] = {
            "entity": result.entity_name,
            "affordance": result.affordance,
            "success": True,
            "entity_state": entity_state,
            # NOTE: `active_failures` stays in `output` for callers
            # (LLMs, sim display) that want to reason about failures
            # in the prompt. The bio-pipeline signal for NAc learning
            # travels through `side_effects["embodiment_failures"]`
            # and is consumed by runtime/executor.py. Two audiences,
            # two channels, same data.
            "active_failures": active_failures,
            **result.metadata,
        }
        return ToolOutput(
            success=True,
            output=output_dict,
            side_effects=({"embodiment_failures": active_failures} if active_failures else None),
        )


class EntitySenseTool(Tool):
    """Auto-generated tool: read ALL sensors on an entity at once."""

    def __init__(self, entity: Entity, tool_name: str) -> None:
        self.name = tool_name
        sensor_names = ", ".join(entity.sensors.keys())
        self.description = f"Read all sensors on {entity.name} ({entity.entity_type}). Sensors: {sensor_names}."
        self.input_schema: dict[str, Any] = {}
        self._entity = entity
        super().__init__()

    def execute(self, **kwargs: Any) -> Any:
        readings = self._entity.read_all_sensors()
        return {name: {"value": r.value, "unit": r.unit} for name, r in readings.items()}


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


def generate_tools_for_entity(
    entity: Entity,
    registry: ToolRegistry,
    embodiment: Any = None,
    cerebellum: Any = None,
    entity_map: Any = None,
) -> list[Tool]:
    """Generate all tools for an entity tree and register them.

    Names are resolved against the registry to avoid collisions — if two
    robots both have a ``shoulder``, the second gets prefixed automatically.

    When *embodiment* is provided, ModulatorAffordanceTools run immediate
    failure evaluation after execution (no 1Hz poll delay). When *cerebellum*
    is provided, they feed actual sensor deltas for forward model training.

    When *entity_map* is provided (an ``EntityMap`` instance), the entity
    tree is registered in it for name-based resolution by
    ``UniversalSenseTool`` and ``SenseToolsTool``.

    Parameters
    ----------
    entity : Entity
        Root of the entity tree (walks descendants).
    registry : ToolRegistry
        Tools are registered here as they are created.
    entity_map : EntityMap, optional
        If provided, entity tree is registered for name-based lookup.

    Returns
    -------
    list[Tool]
        All generated tools (also registered in *registry*).
    """
    # Populate EntityMap if provided
    if entity_map is not None:
        entity_map.register(entity)
    existing: set[str] = set(registry.list_all())
    tools: list[Tool] = []

    for ent in entity.walk():
        # Bulk sensor read
        if ent.sensors:
            tname = _resolve_tool_name(f"sense_{ent.name}", ent, existing)
            tool = EntitySenseTool(ent, tname)
            registry.register(tool)
            existing.add(tname)
            tools.append(tool)

        # Individual sensor reads
        for sensor in ent.sensors.values():
            tname = _resolve_tool_name(
                f"read_{ent.name}_{sensor.name}",
                ent,
                existing,
            )
            tool = SensorReadTool(ent, sensor, tname)
            registry.register(tool)
            existing.add(tname)
            tools.append(tool)

        # Modulator affordances
        for modulator in ent.modulators.values():
            for aff_name, aff_schema in modulator.affordances.items():
                tname = _resolve_tool_name(
                    f"{ent.name}_{aff_name}",
                    ent,
                    existing,
                )
                tool = ModulatorAffordanceTool(
                    ent,
                    modulator,
                    aff_name,
                    aff_schema,
                    tname,
                    embodiment=embodiment,
                    cerebellum=cerebellum,
                )
                registry.register(tool)
                existing.add(tname)
                tools.append(tool)

    log.info(
        "Generated %d tools for entity tree '%s'",
        len(tools),
        entity.name,
    )
    return tools


def describe_entity_capabilities(entity: Entity) -> str:
    """Describe an entity's capabilities as text for observation.

    Returns a structured description of modulators and affordances
    suitable for ``sense_presence`` output or percept text.  Does NOT
    generate callable tools — use ``generate_tools_for_entity`` for that.
    """
    lines: list[str] = []
    for ent in entity.walk():
        for mod_name, mod in ent.modulators.items():
            aff_parts: list[str] = []
            for aff_name, aff_schema in mod.affordances.items():
                desc = aff_schema.description or aff_name
                aff_parts.append(f"{aff_name} ({desc})")
            if aff_parts:
                lines.append(f"  {mod_name}: {', '.join(aff_parts)}")
    return "\n".join(lines) if lines else "No observable capabilities."


def deregister_entity_tools(
    entity: Entity,
    registry: ToolRegistry,
) -> int:
    """Remove all tools generated for an entity tree from the registry.

    Returns the number of tools removed.
    """
    count = 0
    known = set(registry.list_all())
    for ent in entity.walk():
        prefixes = [
            f"sense_{ent.name}",
            f"read_{ent.name}_",
            f"{ent.name}_",
        ]
        for tname in list(known):
            if any(tname.startswith(p) or tname == p for p in prefixes):
                if registry.deregister(tname):
                    known.discard(tname)
                    count += 1
    return count
