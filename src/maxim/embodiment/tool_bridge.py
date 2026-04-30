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
# Sensor delta application
# ---------------------------------------------------------------------------


def _apply_sensor_deltas(
    body: Entity,
    deltas: dict[str, float],
    *,
    delta_kind: str,
) -> None:
    """Apply a {sensor_name: delta} map to a body's sensors.

    Shared by ``self_effect`` (writes to the executor's body) and
    ``target_effect`` (writes to the resolved target body).  Handles
    entity-level sensors (``"hunger"``) and qualified modulator
    sub-sensors (``"arms.thermal"``), range clamping (sensor schema
    range or ``[0, 1]`` fallback), and ``sim_sensor`` logging.

    Missing sensors emit a warning rather than raising so a partially-
    valid delta map applies what it can.

    Parameters
    ----------
    body : Entity
        Root entity of the body that receives the deltas.
    deltas : dict[str, float]
        Sensor-name → delta map (negative deltas decrease the value).
    delta_kind : str
        ``"self_effect"`` or ``"target_effect"`` — used in the
        missing-sensor warning so logs disambiguate the two paths.
    """
    if not deltas:
        return

    for sensor_name, delta in deltas.items():
        old_val: float | None = None
        target_metrics: dict[str, float] | None = None
        target_key: str = sensor_name

        if "." in sensor_name:
            # Qualified modulator sub-sensor: "arms.thermal"
            mod_name, sub_name = sensor_name.split(".", 1)
            mod = body.modulators.get(mod_name)
            if mod is not None and hasattr(mod, "vital_metrics"):
                target_metrics = mod.vital_metrics
                target_key = sub_name
                old_val = target_metrics.get(sub_name)
        else:
            target_metrics = body.vital_metrics
            old_val = target_metrics.get(sensor_name)

        if old_val is None or target_metrics is None:
            log.warning(
                "%s target %r not found on body %s",
                delta_kind,
                sensor_name,
                body.name,
            )
            continue

        # Clamp to sensor range if available, else [0, 1]
        lo, hi = 0.0, 1.0
        if "." in sensor_name:
            mod_name, sub_name = sensor_name.split(".", 1)
            mod = body.modulators.get(mod_name)
            if mod is not None and hasattr(mod, "_sensors"):
                sub_spec = mod._sensors.get(sub_name, {})
                if isinstance(sub_spec, dict) and "range" in sub_spec:
                    lo, hi = sub_spec["range"]
        else:
            sensor = body.sensors.get(sensor_name)
            if sensor is not None:
                rng = sensor.reading_schema.get("range")
                if rng and len(rng) == 2:
                    lo, hi = rng

        new_val = max(lo, min(hi, old_val + delta))
        target_metrics[target_key] = new_val
        try:
            from maxim.simulation.sim_logger import sim_sensor

            sim_sensor(
                body.full_path,
                sensor_name,
                new_val,
                baseline=old_val,
            )
        except Exception:
            pass


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
        entity_map: Any = None,
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
        self._affordance_schema = schema
        self._embodiment = embodiment
        self._cerebellum = cerebellum
        self._entity_map = entity_map
        super().__init__()

    def _resolve_target_body(self, target_arg: Any) -> Entity | None:
        """Resolve a ``target=`` kwarg to a body :class:`Entity`.

        ``"self"`` / ``"aut"`` / ``"player"`` (case-insensitive) map to
        the executor's own body (``self._embodiment.root``) when an
        embodiment is wired.  Anything else is looked up in the
        entity_map; ``Entity`` instances passed directly are returned
        as-is.  Returns ``None`` if no target was provided or the target
        cannot be resolved — caller treats that as a silent no-op
        (Stage 1 invariant: ``target_effect`` is silent without a
        target).

        Alias dual-semantics warning: the alias path is *AUT-context-
        relative* — "self" means whatever body the executor is bound
        to.  When :class:`maxim.simulation.tools.OrchestratorActorTool`
        constructs an ephemeral ``ModulatorAffordanceTool`` bound to a
        SCENE entity's body, "self" inside this resolver would point
        at the scene entity, not the AUT.  ``OrchestratorActorTool``
        therefore pre-resolves its target via the AUT-side entity_map
        and passes the resolved :class:`Entity` instance directly via
        ``target=<Entity>`` so this method's first branch (``isinstance
        (target_arg, Entity)``) returns it without touching the alias
        path.  Do NOT add a string-alias path that bypasses this
        invariant — re-introduces the dragon-vs-AUT routing footgun.
        """
        if target_arg is None or target_arg == "":
            return None
        if isinstance(target_arg, Entity):
            return target_arg
        if isinstance(target_arg, str):
            lowered = target_arg.strip().lower()
            if lowered in ("self", "aut", "player") and self._embodiment is not None:
                return self._embodiment.root
            if self._entity_map is not None:
                resolved = self._entity_map.resolve(target_arg)
                if resolved is not None:
                    return resolved
        return None

    def execute(self, **kwargs: Any) -> Any:
        # Check affordance preconditions (component integrity gating).
        # When a body part is too damaged, its affordances fail with an
        # explanatory message — producing the natural tool-failure → pain
        # → learning chain.  The agent learns "can't fly with broken wing."
        if hasattr(self._modulator, "check_affordance_requires"):
            allowed, reason = self._modulator.check_affordance_requires(self._affordance_name)
            if not allowed:
                return ToolOutput(
                    success=False,
                    error=reason,
                    side_effects={
                        "affordance_blocked": {
                            "affordance": self._affordance_name,
                            "modulator": self._modulator.name,
                            "entity": self._entity.name,
                            "reason": reason,
                        }
                    },
                )

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
        # Self-effect: voluntary affordance execution writes back to the
        # executor's own body.  Fires when the agent explicitly called the
        # tool (not reflex/orchestrator).  Supports entity-level sensors
        # ("hunger") and qualified modulator sub-sensors ("arms.thermal").
        if self._affordance_schema.self_effect and self._embodiment is not None:
            _apply_sensor_deltas(
                self._embodiment.root,
                self._affordance_schema.self_effect,
                delta_kind="self_effect",
            )

        # Target-effect: when the affordance fires WITH a target parameter,
        # write deltas to the resolved target's body sensors.  Silent
        # no-op if no target is provided (preserves backward compatibility
        # with self-targeted-only affordances).  Target resolution uses
        # the entity_map; "self"/"aut"/"player" map to the executor's body.
        if self._affordance_schema.target_effect:
            target_arg = kwargs.get("target")
            target_body = self._resolve_target_body(target_arg)
            if target_body is not None:
                _apply_sensor_deltas(
                    target_body,
                    self._affordance_schema.target_effect,
                    delta_kind="target_effect",
                )

        # Build side_effects dict
        side_effects: dict[str, Any] | None = None
        if active_failures:
            side_effects = {"embodiment_failures": active_failures}

        # Entity acquisition: if this is a pick_up affordance and the target
        # is acquirable, signal the executor to reparent + register tools.
        target_name = kwargs.get("object") or kwargs.get("target")
        is_pickup = self._affordance_name == "pick_up" or (
            self._affordance_name == "use"
            and str(kwargs.get("action", "")).lower() in ("pick_up", "pickup", "grab", "take")
        )
        is_drop = self._affordance_name == "drop" or (
            self._affordance_name == "use" and str(kwargs.get("action", "")).lower() in ("drop", "release", "put_down")
        )
        if is_pickup and target_name:
            # Check if target entity is acquirable via entity_map
            if self._entity_map is not None:
                target_entity = self._entity_map.resolve(target_name)
                if target_entity is not None and target_entity.metadata.get("acquirable"):
                    if side_effects is None:
                        side_effects = {}
                    side_effects["entity_acquired"] = target_name
        elif is_drop and target_name:
            if side_effects is None:
                side_effects = {}
            side_effects["entity_released"] = target_name

        return ToolOutput(
            success=True,
            output=output_dict,
            side_effects=side_effects,
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
                    entity_map=entity_map,
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
