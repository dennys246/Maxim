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
    """One reading from a sensor."""

    sensor_name: str
    entity_name: str
    value: Any  # float, dict, ndarray — depends on sensor
    unit: str
    timestamp: float


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
    """

    params: dict[str, type | tuple[type, Any]] = field(default_factory=dict)
    description: str = ""
    timeout: float = 30.0


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
