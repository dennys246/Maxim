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
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from maxim.embodiment.sem import (
    AffordanceSchema,
    CouplingSpec,
    DriveSpec,
    Entity,
    EntropicDriveSpec,
    FailureMode,
    FailureTrigger,
    HomeostaticDriveSpec,
    ModulationSpec,
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


@dataclass(frozen=True, slots=True)
class LatentAffordance:
    """A latent motor program that surfaces when contextually relevant.

    Unlike regular affordances (always-visible tools), latent affordances
    only appear in the prompt when a reflex fires — they represent
    body responses the agent *could* take but wouldn't think of unprompted.

    Integrity gating uses the same ``requires`` mechanism as regular
    affordances: can't dodge with broken legs.
    """

    name: str
    description: str = ""
    requires: dict[str, float] = field(default_factory=dict)


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


def _parse_drive_spec(drive_data: dict[str, Any]) -> DriveSpec:
    """Parse a ``drive:`` YAML block into a HomeostaticDriveSpec or EntropicDriveSpec."""
    mode = drive_data.get("drift_mode", "entropic")

    if mode == "homeostatic":
        modulated_by = None
        if "modulated_by" in drive_data:
            modulated_by = tuple(
                ModulationSpec(
                    source=m["source"],
                    target_field=m["target_field"],
                    modulation_range=tuple(m["range"]),
                )
                for m in drive_data["modulated_by"]
            )
        return HomeostaticDriveSpec(
            set_point=float(drive_data.get("set_point", 0.0)),
            drift_rate=float(drive_data.get("drift_rate", 0.001)),
            comfort_band=float(drive_data.get("comfort_band", 0.0)),
            pain_scale=float(drive_data.get("pain_scale", 0.5)),
            pain_model=str(drive_data.get("pain_model", "linear")),
            modulated_by=modulated_by,
        )
    elif mode == "entropic":
        coupled_to = None
        if "coupled_to" in drive_data:
            coupled_to = tuple(
                CouplingSpec(
                    sensor=c["sensor"],
                    below=float(c["below"]),
                    multiplier=float(c["multiplier"]),
                )
                for c in drive_data["coupled_to"]
            )
        return EntropicDriveSpec(
            drift_direction=str(drive_data.get("drift_direction", "up")),
            drift_rate=float(drive_data.get("drift_rate", 0.001)),
            deprivation_threshold=float(drive_data.get("deprivation_threshold", 0.7)),
            deprivation_pain=float(drive_data.get("deprivation_pain", 0.3)),
            satisfaction_threshold=float(drive_data.get("satisfaction_threshold", 0.3)),
            coupled_to=coupled_to,
        )
    else:
        raise ValueError(f"Unknown drive drift_mode: {mode!r}. Expected 'homeostatic' or 'entropic'.")


def _check_c5_direct_health_deprecation(data: dict[str, Any]) -> None:
    """Emit a DeprecationWarning when an entity declares both a direct
    ``health`` sensor and modulators that carry sensors (component-damage
    model), without opting into derived health.

    The canonical 1.0 shape is one source of truth for entity health:
    when an entity has modulators with sub-sensors, ``Body.evaluate_failures``
    derives ``vital_metrics["health"]`` from modulator integrities.
    A direct ``health`` sensor duplicates that state and goes stale —
    the agent reads ``health = 1.0`` while modulators sit at 0.3 integrity.

    To opt in to derived health, declare ``health: derived`` at the entity
    root (alongside ``sensors``/``modulators``).  Becomes a hard error in 1.0.

    Mirrors the deprecation pattern in ``cli_utils._resolve_persona_mode``:
    ``DeprecationWarning`` is silenced by Python's default warning filter
    outside ``__main__``, and ``_parse_entity`` is reached via deep call
    chains from the orchestrator / foundry / imagination paths — so we
    also print a stderr line for human visibility.
    """
    sensors = data.get("sensors") or {}
    if not isinstance(sensors, dict) or "health" not in sensors:
        return

    # Single canonical form: ``health: derived`` (string) at the entity root.
    # Boolean ``True`` is intentionally NOT accepted — shipping one form
    # into 1.0 keeps the contract narrow.
    if data.get("health") == "derived":
        return

    modulators = data.get("modulators") or {}
    if not isinstance(modulators, dict):
        return
    has_modulator_sensors = any(isinstance(m, dict) and m.get("sensors") for m in modulators.values())
    if not has_modulator_sensors:
        return

    msg = (
        f"Entity {data['name']!r} declares a direct 'health' sensor while "
        f"also having modulators with sub-sensors (component-damage model). "
        f"The direct sensor duplicates state computed from modulator "
        f"integrities and will go stale. Either (a) declare "
        f"'health: derived' at the entity root so health is computed "
        f"from modulator integrities, or (b) remove the direct 'health' "
        f"sensor if the modulators already model the relevant integrity. "
        f"This becomes a hard error in 1.0. (C5 deprecation)"
    )
    print(f"DeprecationWarning: {msg}", file=sys.stderr)
    warnings.warn(msg, DeprecationWarning, stacklevel=2)


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

    _check_c5_direct_health_deprecation(data)

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
        # Parse drive spec from entity-level sensors
        if isinstance(sensor_spec, dict) and "drive" in sensor_spec:
            entity.drive_specs[sensor_name] = _parse_drive_spec(sensor_spec["drive"])

    # -- modulators ---------------------------------------------------------
    for mod_name, mod_spec in data.get("modulators", {}).items():
        affordances: dict[str, AffordanceSchema] = {}
        for aff_name, aff_spec in mod_spec.get("affordances", {}).items():
            params = _parse_params(aff_spec.get("params", {}))
            requires_raw = aff_spec.get("requires", {})
            requires = {k: float(v) for k, v in requires_raw.items()} if requires_raw else {}
            self_effect_raw = aff_spec.get("self_effect", {})
            self_effect = {k: float(v) for k, v in self_effect_raw.items()} if self_effect_raw else {}
            target_effect_raw = aff_spec.get("target_effect", {})
            target_effect = {k: float(v) for k, v in target_effect_raw.items()} if target_effect_raw else {}
            affordances[aff_name] = AffordanceSchema(
                params=params,
                description=aff_spec.get("description", ""),
                timeout=aff_spec.get("timeout", 30.0),
                requires=requires,
                self_effect=self_effect,
                target_effect=target_effect,
            )

        # Latent motor programs (surfaces when reflexes fire)
        latent_affs: list[LatentAffordance] = []
        for la_name, la_spec in mod_spec.get("latent_affordances", {}).items():
            if isinstance(la_spec, dict):
                la_requires_raw = la_spec.get("requires", {})
                la_requires = {k: float(v) for k, v in la_requires_raw.items()} if la_requires_raw else {}
                latent_affs.append(
                    LatentAffordance(
                        name=str(la_name),
                        description=la_spec.get("description", ""),
                        requires=la_requires,
                    )
                )

        # Per-modulator sensors (component-level damage model). Normalise
        # sensors:null and sensors:[] (legal YAML, semantically "no sensors")
        # so the C4 abstract check below and the .items() loop both behave
        # consistently — the pre-fix `.get("sensors", {})` returned the bare
        # None/list and crashed downstream.
        mod_sensors_raw = mod_spec.get("sensors")
        mod_sensors = mod_sensors_raw if isinstance(mod_sensors_raw, dict) else {}
        mod_integrity_fn = mod_spec.get("integrity", "weighted_mean")
        mod_damage_affinities = mod_spec.get("damage_affinities", {})
        mod_abstract = bool(mod_spec.get("abstract", False))

        # The C4 deprecation warning fires inside SpecModulator.__init__ —
        # see that constructor for why the structural enforcement lives at
        # the type, not here.
        modulator = SpecModulator(
            _name=mod_name,
            _entity_name=name,
            _affordances=affordances,
            _sensors=mod_sensors,
            _integrity_fn=mod_integrity_fn,
            _damage_affinities=mod_damage_affinities,
            _latent_affordances=tuple(latent_affs),
            _abstract=mod_abstract,
        )

        # Initialize modulator vital_metrics from sensor specs
        for ms_name, ms_spec in mod_sensors.items():
            if isinstance(ms_spec, dict) and "range" in ms_spec:
                initial = ms_spec.get("initial")
                if initial is not None:
                    modulator.vital_metrics[ms_name] = float(initial)
                else:
                    lo, hi = ms_spec["range"]
                    modulator.vital_metrics[ms_name] = float((lo + hi) / 2)
            # Parse drive spec from modulator sub-sensors
            if isinstance(ms_spec, dict) and "drive" in ms_spec:
                # Use qualified name so evaluate_failures can look it up
                qualified = f"{mod_name}.{ms_name}"
                entity.drive_specs[qualified] = _parse_drive_spec(ms_spec["drive"])

        entity.modulators[mod_name] = modulator

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

    Optionally holds per-modulator **sensors** and an **integrity**
    aggregation function (component-level damage model).  When sensors
    are present, ``compute_integrity()`` derives a single ``integrity``
    value from the sub-sensor readings.  This enables damage to target
    body-part modulators instead of a flat entity-level health sensor.
    """

    __slots__ = (
        "_name",
        "_entity_name",
        "_affordances",
        "_latent_affordances",
        "_backend",
        "_sensors",
        "_integrity_fn",
        "_damage_affinities",
        "_abstract",
        "vital_metrics",
    )

    def __init__(
        self,
        _name: str,
        _entity_name: str,
        _affordances: dict[str, AffordanceSchema],
        _backend: Any | None = None,
        _sensors: dict[str, Any] | None = None,
        _integrity_fn: str = "weighted_mean",
        _damage_affinities: dict[str, dict[str, float]] | None = None,
        _latent_affordances: tuple[LatentAffordance, ...] = (),
        _abstract: bool = False,
    ) -> None:
        self._name = _name
        self._entity_name = _entity_name
        self._affordances = _affordances
        self._latent_affordances = _latent_affordances
        self._backend = _backend
        self._sensors = _sensors or {}
        self._integrity_fn = _integrity_fn
        self._damage_affinities = _damage_affinities or {}
        self._abstract = _abstract
        # Mutable state for per-modulator sensor values (like entity.vital_metrics)
        self.vital_metrics: dict[str, float] = {}

        # C4 (v0.9 deprecation, v1.x hard ConfigurationError): a modulator
        # with no sensors is silent dead weight for the component-damage
        # model unless it's explicitly capability-only. Per CLAUDE.md "push
        # silent-no-op invariants into types, not helpers" — enforcing here
        # covers _parse_entity, Entity.from_dict, foundry-generated specs,
        # and any future programmatic builder in one place. The 1.x flip
        # is `warnings.warn(...)` → `raise ConfigurationError(...)`.
        if not self._sensors and not self._abstract:
            warnings.warn(
                (
                    f"Modulator {self._entity_name!r}.{self._name!r} declares no sensors. "
                    "Capability-only modulators must declare `abstract: true` "
                    "in their YAML to opt out; otherwise declare at least one "
                    "sensor. This will become a hard ConfigurationError in 1.x. "
                    "See docs/plans/v1_refinement.md §C4."
                ),
                DeprecationWarning,
                stacklevel=2,
            )

    @property
    def name(self) -> str:
        return self._name

    @property
    def affordances(self) -> dict[str, AffordanceSchema]:
        return self._affordances

    @property
    def sensors(self) -> dict[str, Any]:
        """Per-modulator sensors (may be empty for capability-only modulators)."""
        return self._sensors

    @property
    def integrity_fn(self) -> str:
        """Aggregation function name: 'weighted_mean', 'min', 'max'."""
        return self._integrity_fn

    @property
    def damage_affinities(self) -> dict[str, dict[str, float]]:
        """Damage type → sub-sensor weight map (level 3 only)."""
        return self._damage_affinities

    @property
    def latent_affordances(self) -> tuple[LatentAffordance, ...]:
        """Latent motor programs declared on this modulator."""
        return self._latent_affordances

    @property
    def abstract(self) -> bool:
        """Capability-only marker (C4): modulator intentionally has no sensors.

        Set in YAML via ``abstract: true``. Suppresses the v0.9 deprecation
        warning for modulators that genuinely expose only affordances and have
        no observable state of their own (e.g. ``weapon.combat``, ``npc.social``).
        """
        return self._abstract

    def available_latent_affordances(self) -> list[LatentAffordance]:
        """Return latent affordances that pass integrity gating.

        Same threshold mechanism as regular affordance ``requires``:
        if the modulator's integrity is below the threshold, the
        latent affordance is not available (can't dodge with broken legs).
        """
        if not self._latent_affordances:
            return []
        integrity = self.compute_integrity()
        result: list[LatentAffordance] = []
        for la in self._latent_affordances:
            if not la.requires:
                result.append(la)
                continue
            threshold = la.requires.get("integrity", 0.0)
            if integrity >= threshold:
                result.append(la)
        return result

    def compute_integrity(self) -> float:
        """Derive modulator integrity from sub-sensor vital_metrics.

        Returns 1.0 if no sub-sensors are defined (backward compat —
        capability-only modulators are always at full integrity).
        """
        if not self.vital_metrics:
            return 1.0

        values = list(self.vital_metrics.values())
        if not values:
            return 1.0

        if self._integrity_fn == "min":
            return min(values)
        elif self._integrity_fn == "max":
            return max(values)
        else:
            # weighted_mean (default) — uses 'weight' from sensor spec
            # if available, else equal weights
            total_weight = 0.0
            weighted_sum = 0.0
            for sname, sval in self.vital_metrics.items():
                sensor_spec = self._sensors.get(sname, {})
                weight = sensor_spec.get("weight", 1.0) if isinstance(sensor_spec, dict) else 1.0
                weighted_sum += sval * weight
                total_weight += weight
            if total_weight == 0:
                return 1.0
            return weighted_sum / total_weight

    def apply_damage(self, amount: float, damage_type: str | None = None) -> float:
        """Apply damage to this modulator's sub-sensors.

        If *damage_type* is given and ``damage_affinities`` exist,
        distributes damage according to the affinity weights.
        Otherwise distributes proportionally across all sub-sensors.

        Returns the new integrity value after damage.
        """
        if not self.vital_metrics:
            # No sub-sensors — can't take component damage
            return 1.0

        if damage_type and damage_type in self._damage_affinities:
            # Level 3: route through affinities.
            # Only count affinities for sensors that actually exist in
            # vital_metrics — prevents silent damage loss on typos
            # (executor review finding #1).
            affinities = self._damage_affinities[damage_type]
            valid = {k: v for k, v in affinities.items() if k in self.vital_metrics}
            if not valid and affinities:
                import logging

                logging.getLogger(__name__).warning(
                    "%s: damage_affinities[%s] references no valid sensors "
                    "(keys: %s, available: %s) — falling back to proportional",
                    self._name,
                    damage_type,
                    list(affinities.keys()),
                    list(self.vital_metrics.keys()),
                )
                # Fall through to proportional distribution
                valid = {}
            if valid:
                total_affinity = sum(valid.values())
                for sensor_name, affinity in valid.items():
                    self.vital_metrics[sensor_name] = max(
                        0.0,
                        self.vital_metrics[sensor_name] - amount * (affinity / total_affinity),
                    )
            # else: fall through to proportional below

        if not (damage_type and damage_type in self._damage_affinities and valid):
            # Level 2 (no damage_type) or affinity fallback: distribute
            # proportionally across all sub-sensors
            n = len(self.vital_metrics)
            per_sensor = amount / n if n > 0 else 0.0
            for sensor_name in self.vital_metrics:
                self.vital_metrics[sensor_name] = max(
                    0.0,
                    self.vital_metrics[sensor_name] - per_sensor,
                )

        return self.compute_integrity()

    def check_affordance_requires(self, affordance_name: str) -> tuple[bool, str]:
        """Check if an affordance's preconditions are met.

        Returns ``(allowed, reason)``. If allowed is False, reason
        describes which requirement failed (e.g., "Wing too damaged to
        take flight (integrity: 0.15, requires: 0.30)").
        """
        schema = self._affordances.get(affordance_name)
        if schema is None or not schema.requires:
            return True, ""

        integrity = self.compute_integrity()

        for req_name, req_threshold in schema.requires.items():
            if req_name == "integrity":
                if integrity < req_threshold:
                    return False, (
                        f"{self._name.replace('_', ' ').title()} too damaged for "
                        f"{affordance_name.replace('_', ' ')} "
                        f"(integrity: {integrity:.2f}, requires: {req_threshold:.2f})"
                    )
            elif req_name in self.vital_metrics:
                val = self.vital_metrics[req_name]
                if val < req_threshold:
                    return False, (
                        f"{self._name.replace('_', ' ').title()} {req_name.replace('_', ' ')} "
                        f"too low for {affordance_name.replace('_', ' ')} "
                        f"({val:.2f}, requires: {req_threshold:.2f})"
                    )

        return True, ""

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
