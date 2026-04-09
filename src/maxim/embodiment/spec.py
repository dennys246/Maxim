"""YAML spec loader — parses body/world entity definitions into Entity trees.

Loads ``EmbodimentSpec`` from YAML files and builds a live Entity tree
with the appropriate sensor/modulator backends attached.

Example YAML::

    body:
      name: robot_arm
      entity_type: arm
      children:
        - name: shoulder
          entity_type: joint
          sensors:
            angle: {unit: degrees, range: [-180, 180]}
          modulators:
            motor:
              affordances:
                rotate_angle:
                  params: {degrees: float, speed: float}
                  description: "Rotate shoulder joint"
          failure_modes:
            - name: overextension
              trigger: {field: angle, op: ">", value: 175, pain: 0.8}
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from maxim.embodiment.sem import (
    AffordanceSchema,
    Entity,
    FailureMode,
    FailureTrigger,
)

log = logging.getLogger(__name__)

# Map YAML type names to Python types for AffordanceSchema params
_TYPE_MAP: dict[str, type] = {
    "float": float,
    "int": int,
    "str": str,
    "bool": bool,
}


@dataclass
class EmbodimentSpec:
    """Parsed body/world specification from YAML."""

    name: str
    root_entity: Entity
    source_path: str | None = None
    world_entities: list[Entity] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict, repr=False)


def resolve_entity_spec(
    spec: dict[str, Any] | str,
    registry: Any | None = None,
) -> dict[str, Any]:
    """Resolve an entity spec, looking up registry refs as needed.

    Parameters
    ----------
    spec : dict or str
        If a dict, returned as-is (inline entity).
        If a string, treated as a registry ref (e.g. ``"npcs/guard"``).
        If a dict with a ``ref`` key, the ref is resolved and any
        ``overrides`` are deep-merged on top.
    registry : ComponentRegistry or None
        Component registry for ref resolution.  Required if *spec*
        contains a ref.

    Returns
    -------
    dict
        Resolved entity spec dict ready for ``_parse_entity()``.
    """
    # String ref: "npcs/guard"
    if isinstance(spec, str):
        if registry is None:
            raise ValueError(f"Entity ref '{spec}' requires a ComponentRegistry")
        try:
            resolved = registry.get(spec)
        except KeyError as e:
            raise KeyError(f"Failed to resolve entity ref '{spec}': {e}") from e
        return resolved.get("entity", resolved)

    # Dict with ref key: { ref: "npcs/guard", overrides: { name: "captain" } }
    if isinstance(spec, dict) and "ref" in spec:
        ref_str = spec["ref"]
        if registry is None:
            raise ValueError(f"Entity ref '{ref_str}' requires a ComponentRegistry")
        try:
            resolved = registry.get(ref_str)
        except KeyError as e:
            raise KeyError(f"Failed to resolve entity ref '{ref_str}': {e}") from e
        entity_spec = resolved.get("entity", resolved)
        overrides = spec.get("overrides")
        if overrides:
            from maxim.embodiment.component_registry import deep_merge

            entity_spec = deep_merge(entity_spec, overrides)
        return entity_spec

    # Plain inline dict
    return spec


def load_spec(path: str | Path, registry: Any | None = None) -> EmbodimentSpec:
    """Load an EmbodimentSpec from a YAML file.

    The YAML must have a top-level ``body`` key and/or ``world_entities`` key.

    Parameters
    ----------
    path : str or Path
        Path to the YAML file.
    registry : ComponentRegistry or None
        Optional component registry for resolving entity refs in
        ``world_entities`` entries.

    Returns
    -------
    EmbodimentSpec
        Parsed specification with live Entity tree(s).

    Raises
    ------
    FileNotFoundError
        If the YAML file does not exist.
    ValueError
        If the YAML is malformed or missing required fields.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Embodiment spec not found: {path}")

    with open(path) as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError(f"Expected YAML dict, got {type(raw).__name__}")

    body_data = raw.get("body")
    world_data = raw.get("world_entities", [])

    if body_data is None and not world_data:
        raise ValueError("YAML must have a 'body' key and/or 'world_entities' key")

    root: Entity | None = None
    if body_data is not None:
        root = _parse_entity(resolve_entity_spec(body_data, registry))

    world_entities: list[Entity] = []
    for ent_data in world_data:
        world_entities.append(_parse_entity(resolve_entity_spec(ent_data, registry)))

    if root is None and world_entities:
        root = world_entities[0]

    spec_name = raw.get("name", root.name if root else "unnamed")

    return EmbodimentSpec(
        name=spec_name,
        root_entity=root,  # type: ignore[arg-type]
        source_path=str(path),
        world_entities=world_entities,
        raw=raw,
    )


def _parse_entity(
    data: dict[str, Any],
    parent: Entity | None = None,
) -> Entity:
    """Recursively parse an entity dict from YAML into an Entity tree.

    Sensors and modulators are created as stubs (``SpecSensor`` /
    ``SpecModulator``) that hold the YAML metadata. The caller is
    responsible for attaching live backends (LLM, hardware, etc.)
    via ``attach_backends()``.
    """
    name = data.get("name")
    if not name:
        raise ValueError("Entity must have a 'name' field")

    entity_type = data.get("entity_type", "generic")
    metadata = {
        k: v
        for k, v in data.items()
        if k not in ("name", "entity_type", "sensors", "modulators", "failure_modes", "children")
    }

    entity = Entity(
        name=name,
        entity_type=entity_type,
        parent=parent,
        metadata=metadata,
    )

    # -- sensors ------------------------------------------------------------
    for sensor_name, sensor_spec in data.get("sensors", {}).items():
        entity.sensors[sensor_name] = SpecSensor(
            _name=sensor_name,
            _entity_name=name,
            _unit=sensor_spec.get("unit", "unknown"),
            _schema=_build_reading_schema(sensor_spec),
            _initial=sensor_spec.get("initial"),
            _entity_ref=entity,
        )

    # -- modulators ---------------------------------------------------------
    for mod_name, mod_spec in data.get("modulators", {}).items():
        affordances: dict[str, AffordanceSchema] = {}
        for aff_name, aff_spec in mod_spec.get("affordances", {}).items():
            params = _parse_params(aff_spec.get("params", {}))
            affordances[aff_name] = AffordanceSchema(
                params=params,
                description=aff_spec.get("description", ""),
                timeout=aff_spec.get("timeout", 30.0),
            )
        entity.modulators[mod_name] = SpecModulator(
            _name=mod_name,
            _entity_name=name,
            _affordances=affordances,
        )

    # -- failure modes ------------------------------------------------------
    for fm_data in data.get("failure_modes", []):
        entity.failure_modes.append(_parse_failure_mode(fm_data))

    # -- children -----------------------------------------------------------
    for child_data in data.get("children", []):
        _parse_entity(child_data, parent=entity)

    # -- initial vital metrics from sensor ranges ---------------------------
    for sensor_name, sensor in entity.sensors.items():
        schema = sensor.reading_schema
        if schema.get("type") in ("float", "int") and "range" in schema:
            lo, hi = schema["range"]
            initial = schema.get("initial")
            if initial is not None:
                entity.vital_metrics[sensor_name] = float(initial)
            else:
                entity.vital_metrics[sensor_name] = float((lo + hi) / 2)

    return entity


def _build_reading_schema(spec: dict[str, Any]) -> dict[str, Any]:
    """Build a reading_schema dict from a YAML sensor spec."""
    schema: dict[str, Any] = {}

    if "range" in spec:
        schema["type"] = "float"
        schema["range"] = list(spec["range"])
    elif "shape" in spec:
        schema["type"] = "ndarray"
        schema["shape"] = list(spec["shape"])
        if "dtype" in spec:
            schema["dtype"] = spec["dtype"]
    else:
        schema["type"] = spec.get("type", "float")

    if "initial" in spec:
        schema["initial"] = spec["initial"]

    return schema


def _parse_params(
    params_spec: dict[str, Any],
) -> dict[str, type | tuple[type, Any]]:
    """Parse affordance parameter spec from YAML.

    Supports:
    - ``{degrees: float}`` → required float
    - ``{degrees: {type: float, default: 0}}`` → optional float
    - ``{target: str}`` → required str
    """
    result: dict[str, type | tuple[type, Any]] = {}
    for pname, pspec in params_spec.items():
        if isinstance(pspec, str):
            result[pname] = _TYPE_MAP.get(pspec, str)
        elif isinstance(pspec, dict):
            ptype = _TYPE_MAP.get(pspec.get("type", "str"), str)
            if "default" in pspec:
                result[pname] = (ptype, pspec["default"])
            else:
                result[pname] = ptype
        elif isinstance(pspec, type):
            result[pname] = pspec
        else:
            result[pname] = type(pspec)
    return result


def _parse_failure_mode(data: dict[str, Any]) -> FailureMode:
    """Parse a failure mode from YAML."""
    name = data.get("name", "unnamed")

    # Single trigger shorthand
    trigger_data = data.get("trigger")
    triggers_data = data.get("triggers", [])

    triggers: list[FailureTrigger] = []
    if trigger_data is not None:
        triggers.append(
            FailureTrigger(
                field=trigger_data["field"],
                op=trigger_data["op"],
                value=float(trigger_data["value"]),
                pain=float(trigger_data.get("pain", 0.5)),
            )
        )
    for td in triggers_data:
        triggers.append(
            FailureTrigger(
                field=td["field"],
                op=td["op"],
                value=float(td["value"]),
                pain=float(td.get("pain", 0.5)),
            )
        )

    # Compound trigger (all: [...])
    all_data = (
        data.get("trigger", {}).get("all")
        if isinstance(data.get("trigger"), dict) and "all" in data.get("trigger", {})
        else None
    )
    if all_data is not None:
        triggers = []
        for td in all_data:
            triggers.append(
                FailureTrigger(
                    field=td["field"],
                    op=td["op"],
                    value=float(td["value"]),
                    pain=float(td.get("pain", 0.5)),
                )
            )

    trigger_mode = "all" if all_data is not None else data.get("trigger_mode", "any")

    recovery = None
    rc_data = data.get("recovery_condition")
    if rc_data is not None:
        recovery = FailureTrigger(
            field=rc_data["field"],
            op=rc_data["op"],
            value=float(rc_data["value"]),
        )

    return FailureMode(
        name=name,
        composes=data.get("composes", []),
        triggers=triggers,
        trigger_mode=trigger_mode,
        pain_intensity=float(data.get("pain_intensity", triggers[0].pain if triggers else 0.5)),
        persistent=data.get("persistent", False),
        recovery_condition=recovery,
    )


# ---------------------------------------------------------------------------
# Spec-level sensor/modulator stubs (hold YAML metadata, delegate to backend)
# ---------------------------------------------------------------------------


class SpecSensor:
    """Sensor stub created from YAML spec.

    Holds metadata from the YAML definition.  Reads from the entity's
    ``vital_metrics`` dict by default (populated during YAML loading).
    A real backend (LLM, hardware) can be attached later.
    """

    __slots__ = ("_name", "_entity_name", "_unit", "_schema", "_initial", "_backend", "_entity_ref")

    def __init__(
        self,
        _name: str,
        _entity_name: str,
        _unit: str,
        _schema: dict[str, Any],
        _initial: float | None = None,
        _backend: Any | None = None,
        _entity_ref: Any | None = None,
    ) -> None:
        self._name = _name
        self._entity_name = _entity_name
        self._unit = _unit
        self._schema = _schema
        self._initial = _initial
        self._backend = _backend
        self._entity_ref = _entity_ref

    @property
    def name(self) -> str:
        return self._name

    @property
    def unit(self) -> str:
        return self._unit

    @property
    def reading_schema(self) -> dict[str, Any]:
        return self._schema

    def read(self) -> Any:
        """Read from backend if attached, else from entity vital_metrics."""
        if self._backend is not None:
            return self._backend.read()
        import time as _time

        from maxim.embodiment.sem import SensorReading

        # Read from entity's vital_metrics (live state) if available
        val = self._initial if self._initial is not None else 0.0
        if self._entity_ref is not None:
            val = self._entity_ref.vital_metrics.get(self._name, val)
        return SensorReading(
            sensor_name=self._name,
            entity_name=self._entity_name,
            value=val,
            unit=self._unit,
            timestamp=_time.time(),
        )


class SpecModulator:
    """Modulator stub created from YAML spec.

    Holds affordance schemas from the YAML definition.  Execution is
    a no-op by default; a real backend (LLM, hardware) can be attached.
    """

    __slots__ = ("_name", "_entity_name", "_affordances", "_backend")

    def __init__(
        self,
        _name: str,
        _entity_name: str,
        _affordances: dict[str, AffordanceSchema],
        _backend: Any | None = None,
    ) -> None:
        self._name = _name
        self._entity_name = _entity_name
        self._affordances = _affordances
        self._backend = _backend

    @property
    def name(self) -> str:
        return self._name

    @property
    def affordances(self) -> dict[str, AffordanceSchema]:
        return self._affordances

    def execute(self, affordance: str, params: dict[str, Any]) -> Any:
        """Execute via backend if attached, else return stub success."""
        if self._backend is not None:
            return self._backend.execute(affordance, params)
        from maxim.embodiment.sem import ModulatorResult

        if affordance not in self._affordances:
            return ModulatorResult(
                success=False,
                modulator_name=self._name,
                entity_name=self._entity_name,
                affordance=affordance,
                params=params,
                error=f"Unknown affordance: {affordance}",
            )
        return ModulatorResult(
            success=True,
            modulator_name=self._name,
            entity_name=self._entity_name,
            affordance=affordance,
            params=params,
        )


def attach_backends(
    entity: Entity,
    sensor_factory: Any = None,
    modulator_factory: Any = None,
) -> None:
    """Attach live backends to all SpecSensor/SpecModulator stubs in a tree.

    Parameters
    ----------
    entity : Entity
        Root of the entity tree.
    sensor_factory : callable, optional
        ``(entity, sensor_name, spec_sensor) -> Sensor`` or None to keep stubs.
    modulator_factory : callable, optional
        ``(entity, mod_name, spec_modulator) -> Modulator`` or None to keep stubs.
    """
    for ent in entity.walk():
        if sensor_factory is not None:
            for sname, sensor in list(ent.sensors.items()):
                if isinstance(sensor, SpecSensor):
                    backend = sensor_factory(ent, sname, sensor)
                    if backend is not None:
                        sensor._backend = backend

        if modulator_factory is not None:
            for mname, mod in list(ent.modulators.items()):
                if isinstance(mod, SpecModulator):
                    backend = modulator_factory(ent, mname, mod)
                    if backend is not None:
                        mod._backend = backend
