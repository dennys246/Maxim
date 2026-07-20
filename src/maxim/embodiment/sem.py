"""Sensor-Entity-Modulator (SEM) protocol — composability foundation.

Every piece of hardware or virtual entity is described as a triple:
- Entity: the physical/virtual thing (joint, camera, sword, NPC)
- Sensor: reads state from the entity (angle, durability, trust)
- Modulator: changes state of the entity (rotate, slash, speak)

Each is a small protocol class. Entities compose into trees
(arm -> elbow -> wrist -> gripper). The system auto-generates agent
tools, Cerebellum model keys, ATL concepts, and pain triggers from
the registered SEM graph.
"""

from __future__ import annotations

import operator
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


# ---------------------------------------------------------------------------
# Data carriers
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SensorReading:
    """One reading from a sensor.

    Forward-compat (CC3, 1.0): ``extra`` is the post-1.0 escape hatch
    for additive metadata (provenance tags, calibration hints, etc.)
    so new sensor types can carry context without bumping a major
    version. Producers should prefer declared fields when the data is
    part of the SEM contract; ``extra`` is for genuinely additive
    observability metadata only.

    ``extra`` values MUST be JSON-serializable if the reading is going
    to be persisted (``atomic_write_json`` raises on ndarray/datetime/
    arbitrary objects). The ``value`` field is intentionally typed
    ``Any`` because some sensors yield ndarray/audio frames; ``extra``
    is the strict-JSON sibling.

    ``extra`` is excluded from ``__hash__`` and ``__eq__`` (``hash=False,
    compare=False``) so the dataclass stays hashable when ``extra``
    contains a mutable dict.
    """

    sensor_name: str
    entity_name: str
    value: Any  # float, dict, ndarray — depends on sensor
    unit: str
    timestamp: float
    extra: dict[str, Any] = field(default_factory=dict, hash=False, compare=False)


@dataclass(frozen=True, slots=True)
class ModulatorResult:
    """Outcome of a modulator action."""

    success: bool
    modulator_name: str
    entity_name: str
    affordance: str
    params: dict[str, Any]
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class AffordanceSchema:
    """Describes one named action a modulator can perform.

    ``params`` uses the same format as ``Tool.input_schema``:
    ``{"name": type}`` for required, ``{"name": (type, default)}`` for optional.

    ``requires`` is an optional dict of preconditions that must be met
    for the affordance to execute.  Keys are sensor/integrity names,
    values are minimum thresholds.  When the parent modulator's
    integrity (or a specific sub-sensor) drops below the threshold,
    the affordance is **blocked** — the tool returns a failure result
    explaining why, producing the natural tool-failure → pain → learning
    chain.  Example: ``requires={"integrity": 0.3}`` means the affordance
    needs at least 30% component integrity to execute.

    ``self_effect`` is the per-affordance sensor-delta map applied to
    the *executor's own body* on voluntary use (e.g. eating food drops
    the eater's hunger).  ``target_effect`` is the parallel map applied
    to a *resolved target body* when the affordance fires with a
    ``target`` parameter (e.g. ``breathe_fire`` on a dragon writes
    thermal deltas onto the target).  When no target is provided,
    ``target_effect`` is silently skipped — the affordance still works
    for self-targeted use (this preserves backward compatibility with
    every existing affordance that has no ``target_effect`` field).

    SHAPE-FROZEN at 1.0 (CC3). The dict-typed fields (``params``,
    ``requires``, ``self_effect``, ``target_effect``) absorb most
    extension needs at the YAML layer, so a free-form ``extra: dict``
    hatch is deliberately rejected — it would re-open the silent-typo
    class for sensor-name keys (``arms.thermall`` instead of
    ``arms.thermal``) that ``_apply_sensor_deltas`` already fail-loud
    warns on for declared keys. All fields have defaults, so additive
    fields appended at the end stay non-breaking. Adding a *required*
    field post-1.0 is a major-version-bump change. Per CLAUDE.md:
    "Do NOT add fields to these frozen dataclasses post-1.0 without a
    breaking-change plan."
    """

    params: dict[str, type | tuple[type, Any]] = field(default_factory=dict)
    description: str = ""
    timeout: float = 30.0
    requires: dict[str, float] = field(default_factory=dict)
    self_effect: dict[str, float] = field(default_factory=dict)  # agent sensor deltas on voluntary use
    target_effect: dict[str, float] = field(default_factory=dict)  # sensor deltas applied to resolved target
    # When True, the goal-relevance top-k (``select_goal_relevant_tools``) keeps
    # this affordance active regardless of goal keyword overlap — for body
    # actions that are always available (you can always turn your head / attend
    # to a sound), not gated on the static goal mentioning them. Prevents the
    # orient affordances (listen/turn) from being deactivated turn-1 and then
    # failing "not active" when a sound actually arrives.
    always_active: bool = False


# ---------------------------------------------------------------------------
# Drive specs — homeostatic vs entropic interoceptive drives
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CouplingSpec:
    """How one drive's state modulates another drive's drift rate.

    Example: hunger drifts 2x faster when stamina drops below 0.4.
    Ships as 1.0 interface — implementation deferred post-cradle.

    SHAPE-FROZEN at 1.0 (CC3). All three fields are load-bearing
    interface reservations parsed from YAML; an ``extra`` dict would
    invite YAML authors to stash unstructured logic that the
    deferred-implementation evaluator could not honour. Adding any new
    field post-1.0 is a major-version-bump change. Per CLAUDE.md:
    "Do NOT add fields to these frozen dataclasses post-1.0 without a
    breaking-change plan."
    """

    sensor: str  # source sensor name (e.g., "stamina")
    below: float  # threshold on source that activates coupling
    multiplier: float  # drift_rate multiplier when active (e.g., 2.0)


@dataclass(frozen=True, slots=True)
class ModulationSpec:
    """How an external system modulates a homeostatic drive's parameters.

    Example: SCN circadian signal adjusts core_temperature set_point ±0.1.
    Ships as 1.0 interface — implementation deferred post-cradle.

    SHAPE-FROZEN at 1.0 (CC3). Same rationale as :class:`CouplingSpec`
    — adding any new field post-1.0 is a major-version-bump change.
    """

    source: str  # modulating system (e.g., "scn", "nac")
    target_field: str  # which drive field to modulate ("set_point", "drift_rate")
    modulation_range: tuple[float, float]  # bounds (e.g., (-0.1, 0.1))


@dataclass(frozen=True, slots=True)
class HomeostaticDriveSpec:
    """Body self-regulates toward set_point. Discomfort proportional to deviation.

    Homeostatic drives model thermoregulation, pressure recovery, and similar
    systems where the body has an equilibrium point it returns to when
    external forces are removed.  Pain intensity is proportional to how far
    the current value deviates beyond the comfort band from the set point.

    Environmental forces (fire, sun, contact) push the sensor value away
    from set_point.  The body's homeostatic drift pulls it back at
    ``drift_rate`` per second.  When environmental push > body drift,
    pain accumulates.  When the force is removed, homeostasis restores
    the sensor and pain subsides.

    SHAPE-FROZEN at 1.0 (CC3). YAML-parsed drive contract; ``pain_model``
    is the future-proofing knob for non-linear pain formulas. Adding
    any new field post-1.0 is a major-version-bump change. Per CLAUDE.md:
    "Do NOT add fields to these frozen dataclasses post-1.0 without a
    breaking-change plan."
    """

    set_point: float  # body's equilibrium target
    drift_rate: float  # body's self-regulation rate per second
    comfort_band: float = 0.0  # no discomfort within ±band of set_point
    pain_scale: float = 0.5  # intensity per unit outside comfort band
    pain_model: str = "linear"  # "linear" (v1); future: "exponential", "asymmetric"
    modulated_by: tuple[ModulationSpec, ...] | None = None  # 1.0 interface, deferred


@dataclass(frozen=True, slots=True)
class EntropicDriveSpec:
    """Drifts away from equilibrium. Requires external action to reset.

    Entropic drives model hunger, thirst, fatigue, and similar systems
    where the state degrades over time and only external action (eating,
    drinking, resting) reverses the drift.

    ``drift_direction`` is ``"up"`` (toward 1.0) or ``"down"`` (toward 0.0).
    Pain fires when the value crosses ``deprivation_threshold``.  A positive
    Reaction fires when the value crosses back below
    ``satisfaction_threshold`` after being deprived.

    SHAPE-FROZEN at 1.0 (CC3). YAML-parsed drive contract. Adding any
    new field post-1.0 is a major-version-bump change. Per CLAUDE.md:
    "Do NOT add fields to these frozen dataclasses post-1.0 without a
    breaking-change plan."
    """

    drift_direction: str  # "up" or "down"
    drift_rate: float  # per-second drift rate
    deprivation_threshold: float  # PainSignal fires beyond this
    deprivation_pain: float  # pain intensity at deprivation
    satisfaction_threshold: float  # positive Reaction fires when crossing back
    coupled_to: tuple[CouplingSpec, ...] | None = None  # 1.0 interface, deferred


# Union type for convenience
DriveSpec = HomeostaticDriveSpec | EntropicDriveSpec


def drive_pain_for_value(spec: DriveSpec, value: float) -> float:
    """Pain intensity a drive spec produces at a given sensor value, in [0, 1].

    The drive-pain formula for BOTH spec kinds, used by:

    - ``Embodiment.evaluate_failures`` — the **homeostatic** branch is routed
      through this helper (behaviour-preserving: the ``FailureEvent`` and the
      published ``PainSignal`` already clamped to ``[0, 1]``, which is what this
      returns). The **entropic** branch keeps its threshold check inline to
      preserve exact fire-on-threshold semantics for the degenerate
      ``deprivation_pain == 0`` config, so this helper is *not* the single call
      site for entropic pain there — the two are pinned equal by
      ``tests/unit/test_drive_pain_helper.py`` (entropic parametrizations);
    - the motor-credit ``drive_potential_diff`` (orient reward) — for BOTH kinds,
      the *reduction* in this value from before to after an action IS the relief
      that action produced, the state-conditioned POSITIVE reward
      substrate-primary selection needs (drive-pain reduction is bio-faithful
      negative reinforcement AND mechanically selectable — see the June orient
      study, ``reference_recommend_action_reward_driven``).

    Homeostatic: ``min(1, (|value - set_point| - comfort_band) * pain_scale)``
    when outside the comfort band, else ``0``. (The firing guard the failure
    path uses is ``pain > 0``, equivalent to the pre-refactor ``excess > 0`` for
    the ``pain_scale > 0`` of every shipped config.)
    Entropic: ``deprivation_pain`` once ``value`` is past
    ``deprivation_threshold`` in the drift direction, else ``0``.
    """
    if isinstance(spec, HomeostaticDriveSpec):
        excess = abs(value - spec.set_point) - spec.comfort_band
        return min(1.0, excess * spec.pain_scale) if excess > 0 else 0.0
    if isinstance(spec, EntropicDriveSpec):
        if spec.drift_direction == "up" and value >= spec.deprivation_threshold:
            return spec.deprivation_pain
        if spec.drift_direction == "down" and value <= spec.deprivation_threshold:
            return spec.deprivation_pain
        return 0.0
    return 0.0


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------


@runtime_checkable
class Sensor(Protocol):
    """Reads state from an entity.  One sensor = one readable quantity."""

    @property
    def name(self) -> str:
        """Sensor identifier, unique within its entity.

        Examples: ``'angle'``, ``'frame'``, ``'temperature'``, ``'durability'``.
        """
        ...

    @property
    def unit(self) -> str:
        """Human-readable unit.

        Examples: ``'degrees'``, ``'celsius'``, ``'rgb_frame'``, ``'ratio'``.
        """
        ...

    @property
    def reading_schema(self) -> dict[str, Any]:
        """Describes the value shape for tool generation and similarity.

        Examples::

            {"type": "float", "range": [0, 360]}
            {"type": "ndarray", "shape": [480, 640, 3], "dtype": "uint8"}
        """
        ...

    def read(self) -> SensorReading:
        """Take a reading.

        Non-blocking for most sensors; may block briefly for frame capture.
        """
        ...


@runtime_checkable
class Modulator(Protocol):
    """Changes state of an entity.  One modulator = one controllable axis."""

    @property
    def name(self) -> str:
        """Modulator identifier, unique within its entity.

        Examples: ``'motor'``, ``'lifecycle'``, ``'combat'``, ``'social'``.
        """
        ...

    @property
    def affordances(self) -> dict[str, AffordanceSchema]:
        """Named actions this modulator can perform.

        Example::

            {"rotate_angle": AffordanceSchema(
                params={"degrees": float, "speed": float},
                description="Rotate the joint",
            )}
        """
        ...

    def execute(self, affordance: str, params: dict[str, Any]) -> ModulatorResult:
        """Execute an affordance.  Returns structured result."""
        ...


# ---------------------------------------------------------------------------
# Entity — composable tree node
# ---------------------------------------------------------------------------


class Entity:
    """A physical or virtual thing with sensors and modulators.

    Entities compose into trees: ``arm -> elbow -> wrist -> gripper``.
    Each entity is self-describing — its sensors, modulators, vital
    metrics, and failure modes are introspectable at runtime.
    """

    __slots__ = (
        "name",
        "entity_type",
        "sensors",
        "modulators",
        "parent",
        "children",
        "metadata",
        "vital_metrics",
        "failure_modes",
        "drive_specs",
    )

    def __init__(
        self,
        name: str,
        entity_type: str,
        *,
        sensors: dict[str, Sensor] | None = None,
        modulators: dict[str, Modulator] | None = None,
        parent: Entity | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.name = name
        self.entity_type = entity_type
        self.sensors: dict[str, Sensor] = sensors or {}
        self.modulators: dict[str, Modulator] = modulators or {}
        self.parent: Entity | None = parent
        self.children: list[Entity] = []
        self.metadata: dict[str, Any] = metadata or {}
        self.vital_metrics: dict[str, float] = {}
        self.failure_modes: list[FailureMode] = []
        self.drive_specs: dict[str, DriveSpec] = {}  # sensor_name → DriveSpec

        if parent is not None:
            parent.children.append(self)

    # -- tree navigation ----------------------------------------------------

    @property
    def full_path(self) -> str:
        """Dot-separated path from root.  e.g. ``'left_arm.elbow'``."""
        if self.parent is None:
            return self.name
        return f"{self.parent.full_path}.{self.name}"

    def walk(self) -> Iterator[Entity]:
        """Depth-first traversal of this entity and all descendants."""
        yield self
        for child in self.children:
            yield from child.walk()

    def find(self, path: str) -> Entity | None:
        """Find a descendant by dot-path relative to this entity."""
        parts = path.split(".", 1)
        for child in self.children:
            if child.name == parts[0]:
                return child.find(parts[1]) if len(parts) > 1 else child
        return None

    # -- sensor convenience -------------------------------------------------

    def read_all_sensors(self) -> dict[str, SensorReading]:
        """Read every sensor on this entity.  Returns ``{sensor_name: reading}``."""
        return {name: sensor.read() for name, sensor in self.sensors.items()}

    def read_scalar_sensors(self) -> dict[str, SensorReading]:
        """Read only scalar-valued sensors (skip frames, audio, etc.).

        A sensor is considered scalar if its ``reading_schema["type"]``
        is ``"float"`` or ``"int"``, or if ``reading_schema`` is absent
        (assume scalar for backward compat).
        """
        result: dict[str, SensorReading] = {}
        for name, sensor in self.sensors.items():
            schema = sensor.reading_schema
            stype = schema.get("type", "float")
            if stype in ("float", "int"):
                result[name] = sensor.read()
        return result

    # -- component-level damage -----------------------------------------------

    def derive_health(self) -> float | None:
        """Derive entity health from modulator component integrities.

        Returns a weighted mean of modulator integrities using
        ``metadata["health_weights"]`` if present.  Returns None if
        no modulators have component sensors (backward compat — entity
        uses direct ``vital_metrics["health"]`` instead).

        Called by ``Body.evaluate_failures()`` to update
        ``vital_metrics["health"]`` when ``metadata.get("health") == "derived"``.
        """
        # Collect modulator integrities
        integrities: dict[str, float] = {}
        for mod_name, mod in self.modulators.items():
            if hasattr(mod, "compute_integrity"):
                integrity = mod.compute_integrity()
                if hasattr(mod, "vital_metrics") and mod.vital_metrics:
                    integrities[mod_name] = integrity

        if not integrities:
            return None  # No component sensors → not using derived health

        weights = self.metadata.get("health_weights", {})
        total_weight = 0.0
        weighted_sum = 0.0
        for mod_name, integrity in integrities.items():
            w = weights.get(mod_name, 1.0)
            weighted_sum += integrity * w
            total_weight += w

        if total_weight == 0:
            return None
        return weighted_sum / total_weight

    def get_component(self, name: str) -> Any | None:
        """Get a modulator by name (for component-level damage targeting).

        Returns the modulator if found, None otherwise.
        """
        return self.modulators.get(name)

    # -- tree mutation (DM entity transfer) ------------------------------------

    def reparent(self, new_parent: Entity) -> None:
        """Move this entity to a new parent. Updates both parent references."""
        if self.parent is not None:
            self.parent.children.remove(self)
        self.parent = new_parent
        new_parent.children.append(self)

    def detach(self) -> None:
        """Remove this entity from its parent (drop/destroy)."""
        if self.parent is not None:
            self.parent.children.remove(self)
            self.parent = None

    # -- visibility (DM scene management) -------------------------------------

    def reveal(self, name: str) -> None:
        """Change a sensor or affordance visibility to 'visible'.

        Used by DM runtime when conditions trigger disclosure
        (insight check, examination, trust threshold).
        """
        self.metadata.setdefault("visibility", {})[name] = "visible"

    def hide(self, name: str) -> None:
        """Change a sensor or affordance visibility to 'hidden'."""
        self.metadata.setdefault("visibility", {})[name] = "hidden"

    def get_visibility(self, name: str) -> str:
        """Get visibility for a sensor or affordance. Default: 'visible'."""
        return self.metadata.get("visibility", {}).get(name, "visible")

    # -- serialization ------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize entity tree to a dict suitable for YAML/JSON persistence.

        Captures the full entity tree including sensors (metadata only,
        not live backends), modulators (affordance schemas), children,
        vital metrics, failure modes, and metadata.

        Round-trips with ``Entity.from_dict()``.
        """

        def _sensor_dict(s: Any) -> dict[str, Any]:
            d: dict[str, Any] = {"name": s.name}
            if hasattr(s, "unit"):
                d["unit"] = s.unit
            if hasattr(s, "reading_schema"):
                d["reading_schema"] = s.reading_schema
            if hasattr(s, "_initial"):
                d["initial"] = s._initial
            return d

        def _modulator_dict(m: Any) -> dict[str, Any]:
            d: dict[str, Any] = {"name": m.name}
            if hasattr(m, "affordances"):
                affs = {}
                for aff_name, schema in m.affordances.items():
                    affs[aff_name] = {
                        "description": getattr(schema, "description", ""),
                        "timeout": getattr(schema, "timeout", 30.0),
                    }
                d["affordances"] = affs
            # Always emit the C4 abstract marker (symmetric serialization).
            # Asymmetric "only-when-true" emission silently loses the explicit
            # opt-in/opt-out signal once 1.x flips the warning to a hard
            # error — the same trap CC1's `_format_version` discipline
            # exists to close. False is the documented default; emitting it
            # explicitly keeps reloaded entities indistinguishable from the
            # source they were saved from.
            d["abstract"] = bool(getattr(m, "abstract", False))
            # Preserve per-modulator sensor metadata so reloaded modulators
            # reconstruct as the same shape that was saved. Without this,
            # `Entity.from_dict` returns no-sensor stubs for modulators that
            # originally had component-damage sensors, which (after C4) would
            # spuriously trip the C4 ConfigurationError on every load.
            mod_sensors = getattr(m, "sensors", None)
            if mod_sensors:
                d["sensors"] = dict(mod_sensors)
            return d

        def _trigger_dict(t: Any) -> dict[str, Any]:
            return {
                "field": t.field,
                "op": t.op,
                "value": t.value,
                "pain": t.pain,
            }

        def _failure_dict(fm: Any) -> dict[str, Any]:
            d: dict[str, Any] = {"name": fm.name}
            if fm.composes:
                d["composes"] = list(fm.composes)
            if fm.triggers:
                d["triggers"] = [_trigger_dict(t) for t in fm.triggers]
            d["trigger_mode"] = fm.trigger_mode
            d["pain_intensity"] = fm.pain_intensity
            d["persistent"] = fm.persistent
            if fm.recovery_condition:
                d["recovery_condition"] = _trigger_dict(fm.recovery_condition)
            return d

        result: dict[str, Any] = {
            "name": self.name,
            "entity_type": self.entity_type,
        }
        if self.sensors:
            result["sensors"] = {k: _sensor_dict(v) for k, v in self.sensors.items()}
        if self.modulators:
            result["modulators"] = {k: _modulator_dict(v) for k, v in self.modulators.items()}
        if self.metadata:
            result["metadata"] = dict(self.metadata)
        if self.vital_metrics:
            result["vital_metrics"] = dict(self.vital_metrics)
        if self.failure_modes:
            result["failure_modes"] = [_failure_dict(fm) for fm in self.failure_modes]
        if self.drive_specs:
            drive_dict: dict[str, Any] = {}
            for ds_name, ds in self.drive_specs.items():
                if isinstance(ds, HomeostaticDriveSpec):
                    drive_dict[ds_name] = {
                        "drift_mode": "homeostatic",
                        "set_point": ds.set_point,
                        "drift_rate": ds.drift_rate,
                        "comfort_band": ds.comfort_band,
                        "pain_scale": ds.pain_scale,
                        "pain_model": ds.pain_model,
                    }
                elif isinstance(ds, EntropicDriveSpec):
                    drive_dict[ds_name] = {
                        "drift_mode": "entropic",
                        "drift_direction": ds.drift_direction,
                        "drift_rate": ds.drift_rate,
                        "deprivation_threshold": ds.deprivation_threshold,
                        "deprivation_pain": ds.deprivation_pain,
                        "satisfaction_threshold": ds.satisfaction_threshold,
                    }
            result["drive_specs"] = drive_dict
        if self.children:
            result["children"] = [child.to_dict() for child in self.children]
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any], parent: "Entity | None" = None) -> "Entity":
        """Reconstruct an Entity tree from a dict (reverse of ``to_dict()``).

        Sensors and modulators are restored as ``SpecSensor``/``SpecModulator``
        stubs from ``maxim.embodiment.spec``.  These hold metadata and can
        read from ``vital_metrics``.  Attach live backends (LLM, hardware)
        with ``attach_backends()`` if needed.
        """
        entity = cls(
            name=data["name"],
            entity_type=data["entity_type"],
            parent=parent,
            metadata=data.get("metadata"),
        )
        if "vital_metrics" in data:
            entity.vital_metrics = dict(data["vital_metrics"])

        # Reconstruct drive specs
        if "drive_specs" in data:
            for ds_name, ds_data in data["drive_specs"].items():
                mode = ds_data.get("drift_mode")
                if mode == "homeostatic":
                    entity.drive_specs[ds_name] = HomeostaticDriveSpec(
                        set_point=ds_data["set_point"],
                        drift_rate=ds_data["drift_rate"],
                        comfort_band=ds_data.get("comfort_band", 0.0),
                        pain_scale=ds_data.get("pain_scale", 0.5),
                        pain_model=ds_data.get("pain_model", "linear"),
                    )
                elif mode == "entropic":
                    entity.drive_specs[ds_name] = EntropicDriveSpec(
                        drift_direction=ds_data["drift_direction"],
                        drift_rate=ds_data["drift_rate"],
                        deprivation_threshold=ds_data["deprivation_threshold"],
                        deprivation_pain=ds_data["deprivation_pain"],
                        satisfaction_threshold=ds_data["satisfaction_threshold"],
                    )

        # Reconstruct sensors as SpecSensor stubs
        if "sensors" in data:
            try:
                from maxim.embodiment.spec import SpecSensor

                for sname, sdata in data["sensors"].items():
                    entity.sensors[sname] = SpecSensor(
                        _name=sdata.get("name", sname),
                        _entity_name=data["name"],
                        _unit=sdata.get("unit", ""),
                        _schema=sdata.get("reading_schema", {"type": "float"}),
                        _initial=sdata.get("initial"),
                        _entity_ref=entity,
                    )
            except ImportError:
                pass  # spec module not available — skip sensor reconstruction

        # Reconstruct modulators as SpecModulator stubs
        if "modulators" in data:
            try:
                from maxim.embodiment.spec import SpecModulator

                for mname, mdata in data["modulators"].items():
                    affs = {}
                    for aff_name, aff_data in mdata.get("affordances", {}).items():
                        affs[aff_name] = AffordanceSchema(
                            description=aff_data.get("description", ""),
                            timeout=aff_data.get("timeout", 30.0),
                        )
                    # Legacy compat: pre-C4 snapshots have neither
                    # `sensors` nor `abstract` on the modulator dict. Treat
                    # them as abstract=True so they don't spuriously warn
                    # on load — the saved-entity contract didn't carry the
                    # discriminator so we can't re-derive intent. Snapshots
                    # written by 0.9+ always carry an explicit `abstract`
                    # boolean (see _modulator_dict).
                    has_explicit_abstract = "abstract" in mdata
                    has_sensors = bool(mdata.get("sensors"))
                    if has_explicit_abstract:
                        mod_abstract = bool(mdata["abstract"])
                    else:
                        mod_abstract = not has_sensors
                    mod_sensors = mdata.get("sensors") or {}
                    if not isinstance(mod_sensors, dict):
                        mod_sensors = {}
                    entity.modulators[mname] = SpecModulator(
                        _name=mdata.get("name", mname),
                        _entity_name=data["name"],
                        _affordances=affs,
                        _sensors=mod_sensors,
                        _abstract=mod_abstract,
                    )
            except ImportError:
                pass  # spec module not available — skip modulator reconstruction

        # Reconstruct failure modes
        if "failure_modes" in data:
            for fm_data in data["failure_modes"]:
                triggers = []
                for t in fm_data.get("triggers", []):
                    triggers.append(
                        FailureTrigger(
                            field=t["field"],
                            op=t["op"],
                            value=t["value"],
                            pain=t.get("pain", 0.5),
                        )
                    )
                recovery = None
                if "recovery_condition" in fm_data:
                    rc = fm_data["recovery_condition"]
                    recovery = FailureTrigger(
                        field=rc["field"],
                        op=rc["op"],
                        value=rc["value"],
                        pain=rc.get("pain", 0.5),
                    )
                entity.failure_modes.append(
                    FailureMode(
                        name=fm_data.get("name", ""),
                        composes=fm_data.get("composes", []),
                        triggers=triggers,
                        trigger_mode=fm_data.get("trigger_mode", "any"),
                        pain_intensity=fm_data.get("pain_intensity", 0.5),
                        persistent=fm_data.get("persistent", False),
                        recovery_condition=recovery,
                    )
                )

        # Reconstruct children recursively
        for child_data in data.get("children", []):
            cls.from_dict(child_data, parent=entity)
        return entity

    def save(self, path: str) -> None:
        """Save entity tree to a JSON file."""
        from maxim.utils.atomic_io import atomic_write_json
        from maxim.utils.format_version import with_format_version

        atomic_write_json(path, with_format_version(self.to_dict()))

    @classmethod
    def load(cls, path: str) -> "Entity":
        """Load entity tree from a JSON file."""
        import json
        import logging

        from maxim.utils.format_version import check_format_version

        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        check_format_version(data, "sem_entity", log=logging.getLogger(__name__))
        return cls.from_dict(data)

    # -- repr ---------------------------------------------------------------

    def __repr__(self) -> str:
        sens = list(self.sensors.keys())
        mods = list(self.modulators.keys())
        kids = len(self.children)
        return f"Entity({self.name!r}, type={self.entity_type!r}, sensors={sens}, modulators={mods}, children={kids})"


# ---------------------------------------------------------------------------
# Failure mode spec
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class FailureTrigger:
    """Structured trigger condition — no eval, no arbitrary code.

    Evaluated as: ``sensor_reading <op> value``.
    """

    field: str  # sensor name
    op: str  # ">", "<", ">=", "<=", "=="
    value: float
    pain: float = 0.5  # pain intensity when triggered

    def evaluate(self, sensor_value: float) -> bool:
        """Return True if the trigger condition is met."""
        _OPS = {
            ">": operator.gt,
            "<": operator.lt,
            ">=": operator.ge,
            "<=": operator.le,
            "==": operator.eq,
        }
        op_fn = _OPS.get(self.op)
        if op_fn is None:
            return False
        return bool(op_fn(sensor_value, self.value))


@dataclass(slots=True)
class FailureMode:
    """Declarative failure mode attached to an entity.

    Composed from the fixed vocabulary:
    ``overextension``, ``overheating``, ``strain``,
    ``fatigue``, ``impact``, ``exhaustion``.
    """

    name: str
    composes: list[str] = field(default_factory=list)
    triggers: list[FailureTrigger] = field(default_factory=list)
    trigger_mode: str = "any"  # "any" or "all"
    pain_intensity: float = 0.5
    persistent: bool = False
    recovery_condition: FailureTrigger | None = None
    active: bool = False
    last_fired: float = 0.0

    def evaluate(self, sensor_readings: dict[str, float]) -> bool:
        """Check if this failure mode should fire given current readings."""
        if self.persistent and self.active:
            # check recovery
            if self.recovery_condition is not None:
                val = sensor_readings.get(self.recovery_condition.field, 0.0)
                if self.recovery_condition.evaluate(val):
                    self.active = False
                    return False
            return True  # still active, no recovery yet

        results = []
        for trigger in self.triggers:
            val = sensor_readings.get(trigger.field, 0.0)
            results.append(trigger.evaluate(val))

        if self.trigger_mode == "all":
            fired = all(results) if results else False
        else:
            fired = any(results) if results else False

        if fired:
            self.active = True
            self.last_fired = time.time()
        return fired


# Fixed failure mode vocabulary (6 base modes)
BASE_FAILURE_MODES: frozenset[str] = frozenset(
    {
        "overextension",
        "overheating",
        "strain",
        "fatigue",
        "impact",
        "exhaustion",
    }
)
