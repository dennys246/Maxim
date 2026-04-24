"""Embodiment runtime — holds entity tree, evaluates failures, emits pain.

The ``Embodiment`` class is the runtime owner of an SEM entity tree.
It evaluates failure triggers against sensor readings, publishes
``PainSignal`` through the existing ``PainBus``, and manages linear
vital-metric drift.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from maxim.embodiment.sem import Entity, FailureMode, SensorReading

log = logging.getLogger(__name__)


@dataclass
class EmbodimentConfig:
    """Configuration for the Embodiment runtime."""

    poll_hz: float = 1.0
    pain_proximity_threshold: float = 0.2
    vital_drift_rate: float = 0.001
    enable_pain: bool = True


@dataclass
class FailureEvent:
    """Record of a failure mode firing."""

    entity_path: str
    failure_name: str
    pain_intensity: float
    sensor_readings: dict[str, float]
    timestamp: float = field(default_factory=time.time)


class Embodiment:
    """Runtime manager for an SEM entity tree.

    Responsibilities:
    - Holds the root entity and provides tree-level operations
    - Evaluates failure triggers against sensor readings
    - Publishes PainSignals through the PainBus
    - Tracks vital-metric drift (linear, no homeostasis in Phase 0)
    - Provides snapshot of body state for prompt injection
    """

    def __init__(
        self,
        root: Entity,
        *,
        config: EmbodimentConfig | None = None,
        pain_bus: Any | None = None,
    ) -> None:
        self.root = root
        self.config = config or EmbodimentConfig()
        self._pain_bus = pain_bus
        self._failure_history: list[FailureEvent] = []
        self._last_poll: float = 0.0
        self._tick_count: int = 0

    # -- entity access ------------------------------------------------------

    def find(self, path: str) -> Entity | None:
        """Find an entity by dot-path from root."""
        if path == self.root.name:
            return self.root
        return self.root.find(path)

    def all_entities(self) -> list[Entity]:
        """Return flat list of all entities in the tree."""
        return list(self.root.walk())

    # -- sensor reading -----------------------------------------------------

    def read_all(self) -> dict[str, dict[str, SensorReading]]:
        """Read all sensors on all entities.

        Returns ``{entity_path: {sensor_name: reading}}``.
        """
        result: dict[str, dict[str, SensorReading]] = {}
        for ent in self.root.walk():
            readings = ent.read_all_sensors()
            if readings:
                result[ent.full_path] = readings
        return result

    def read_scalars(self) -> dict[str, dict[str, float]]:
        """Read scalar sensors only — for similarity, pain gates, prompts.

        Returns ``{entity_path: {sensor_name: value}}``.
        """
        result: dict[str, dict[str, float]] = {}
        for ent in self.root.walk():
            readings = ent.read_scalar_sensors()
            if readings:
                result[ent.full_path] = {name: r.value for name, r in readings.items()}
        return result

    # -- failure evaluation -------------------------------------------------

    def evaluate_failures(self) -> list[FailureEvent]:
        """Evaluate all failure triggers on all entities.

        Returns list of newly-fired failure events.  Also publishes
        PainSignals through the PainBus if configured.
        """
        events: list[FailureEvent] = []

        for ent in self.root.walk():
            if not ent.failure_modes:
                continue

            # collect scalar sensor values for trigger evaluation
            readings: dict[str, float] = {}
            for sname, sensor in ent.sensors.items():
                schema = sensor.reading_schema
                if schema.get("type") in ("float", "int"):
                    try:
                        r = sensor.read()
                        readings[sname] = float(r.value)
                    except Exception:
                        pass

            # also include vital_metrics (may have drifted values)
            for vname, vval in ent.vital_metrics.items():
                if vname not in readings:
                    readings[vname] = vval

            # Log sensor readings for display/JSONL (Track 5: SEM observability)
            try:
                from maxim.simulation.sim_logger import sim_sensor

                for sname, sval in readings.items():
                    baseline = ent.vital_metrics.get(sname)
                    sim_sensor(ent.full_path, sname, sval, baseline=baseline)
            except Exception:
                pass

            for fm in ent.failure_modes:
                if fm.evaluate(readings):
                    event = FailureEvent(
                        entity_path=ent.full_path,
                        failure_name=fm.name,
                        pain_intensity=fm.pain_intensity,
                        sensor_readings=dict(readings),
                    )
                    events.append(event)
                    self._failure_history.append(event)
                    self._publish_pain(ent, fm, readings)

        return events

    def _publish_pain(
        self,
        entity: Entity,
        failure: FailureMode,
        readings: dict[str, float],
    ) -> None:
        """Publish a rich-context ``PainSignal`` for an embodiment failure.

        Publishing through ``PainBus.publish`` (rather than constructing
        a ``Reaction`` directly on ``reaction_bus``) lets downstream
        bio-pipeline consumers — ``ToolPainBridge._on_embodiment_pain``,
        ``create_pain_nac_subscriber``, hippocampus episodic capture —
        see the full cause-description metadata: ``source``, ``entity``,
        ``entity_type``, ``failure_mode``, ``composes``,
        ``sensor_readings``. The downstream Reaction published to
        ``reaction_bus`` still carries the typed surface for subscribers
        that want the strict view.
        """
        if not self.config.enable_pain or self._pain_bus is None:
            return

        try:
            from maxim.proprioception.pain import PainSignal, PainType

            signal = PainSignal(
                pain_type=PainType.EXTERNAL_SIGNAL,
                intensity=failure.pain_intensity,
                timestamp=time.time(),
                context={
                    "source": "embodiment",
                    "entity": entity.full_path,
                    "entity_type": entity.entity_type,
                    "failure_mode": failure.name,
                    "composes": list(failure.composes or []),
                    "sensor_readings": dict(readings),
                    # Retained for legacy consumers that read entity_path.
                    "entity_path": entity.full_path,
                },
            )
            self._pain_bus.publish(signal)
            log.debug(
                "Pain published: %s on %s (%.2f)",
                failure.name,
                entity.full_path,
                failure.pain_intensity,
            )
        except Exception:
            log.exception("Failed to publish embodiment pain signal")

    # -- vital metric drift -------------------------------------------------

    def tick_vital_drift(self, dt: float = 1.0) -> None:
        """Apply linear drift to vital metrics.

        Called once per poll cycle.  Drift rate is configured in
        ``EmbodimentConfig.vital_drift_rate``.
        """
        rate = self.config.vital_drift_rate
        for ent in self.root.walk():
            for vname in list(ent.vital_metrics.keys()):
                # Drift toward degradation (e.g., fatigue increases)
                if vname in ("fatigue", "strain", "exhaustion"):
                    ent.vital_metrics[vname] = min(
                        1.0,
                        ent.vital_metrics[vname] + rate * dt,
                    )
                elif vname in ("durability", "sharpness"):
                    ent.vital_metrics[vname] = max(
                        0.0,
                        ent.vital_metrics[vname] - rate * dt,
                    )
                # Other metrics don't drift by default

    # -- body state snapshot for prompts ------------------------------------

    def body_state_summary(self) -> list[dict[str, Any]]:
        """Build a summary of body state for prompt injection.

        Returns a list of entity state dicts, with pain-proximity
        warnings where sensor values are near failure thresholds.
        """
        summary: list[dict[str, Any]] = []

        for ent in self.root.walk():
            if not ent.sensors:
                continue

            ent_state: dict[str, Any] = {
                "entity": ent.full_path,
                "type": ent.entity_type,
                "sensors": {},
            }

            for sname, sensor in ent.sensors.items():
                schema = sensor.reading_schema
                if schema.get("type") not in ("float", "int"):
                    continue

                try:
                    reading = sensor.read()
                    val = float(reading.value)
                except Exception:
                    continue

                sensor_info: dict[str, Any] = {
                    "value": val,
                    "unit": reading.unit,
                }

                # Check pain proximity
                rng = schema.get("range")
                if rng:
                    lo, hi = rng
                    range_size = hi - lo
                    if range_size > 0:
                        # Check against failure triggers
                        for fm in ent.failure_modes:
                            for trigger in fm.triggers:
                                if trigger.field == sname:
                                    dist = abs(val - trigger.value) / range_size
                                    if dist < self.config.pain_proximity_threshold:
                                        sensor_info["warning"] = (
                                            f"{fm.name} threshold at {trigger.value}{reading.unit}, pain {trigger.pain}"
                                        )
                                        sensor_info["pain_proximity"] = 1.0 - dist

                ent_state["sensors"][sname] = sensor_info

            if ent_state["sensors"]:
                summary.append(ent_state)

        return summary

    def format_body_state_for_prompt(self) -> str:
        """Format body state as a string for LLM prompt injection."""
        summary = self.body_state_summary()
        if not summary:
            return ""

        lines = ["=== Body State ==="]
        has_warnings = False
        for ent_state in summary:
            for sname, sinfo in ent_state["sensors"].items():
                unit = sinfo.get("unit", "")
                val = sinfo["value"]
                warning = sinfo.get("warning")
                if warning:
                    lines.append(f"- {ent_state['entity']}.{sname}: {val}{unit} (WARN: {warning})")
                    has_warnings = True
                else:
                    lines.append(f"- {ent_state['entity']}.{sname}: {val}{unit}")

        if has_warnings:
            lines[0] = "=== Body State (pain-relevant) ==="

        return "\n".join(lines)

    # -- stats --------------------------------------------------------------

    @property
    def failure_history(self) -> list[FailureEvent]:
        return list(self._failure_history)

    # -- failure persistence -------------------------------------------------

    def export_failure_state(self) -> dict[str, Any]:
        """Export active failure states for persistence."""
        active_failures: list[dict[str, Any]] = []
        for ent in self.root.walk():
            for fm in ent.failure_modes:
                if fm.active:
                    active_failures.append(
                        {
                            "entity_path": ent.full_path,
                            "failure_name": fm.name,
                            "composes": fm.composes,
                            "pain_intensity": fm.pain_intensity,
                            "last_fired": fm.last_fired,
                            "persistent": fm.persistent,
                        }
                    )

        return {
            "active_failures": active_failures,
            "failure_history": [
                {
                    "entity_path": e.entity_path,
                    "failure_name": e.failure_name,
                    "pain_intensity": e.pain_intensity,
                    "timestamp": e.timestamp,
                    "sensor_readings": e.sensor_readings,
                }
                for e in self._failure_history[-100:]  # cap at last 100
            ],
        }

    def import_failure_state(self, data: dict[str, Any]) -> None:
        """Restore active failure states from persistence."""
        for af in data.get("active_failures", []):
            entity = self.find(af.get("entity_path", ""))
            if entity is None:
                continue
            for fm in entity.failure_modes:
                if fm.name == af["failure_name"]:
                    fm.active = af.get("persistent", False) and True
                    fm.last_fired = af.get("last_fired", 0.0)
                    break

    def save_failures(self, path: str) -> None:
        """Save failure state to JSON file."""
        import json
        from pathlib import Path

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        state = self.export_failure_state()
        try:
            from maxim.utils.atomic_io import atomic_write_json

            atomic_write_json(path, state)
        except ImportError:
            with open(path, "w") as f:
                json.dump(state, f, indent=2)

    def load_failures(self, path: str) -> bool:
        """Load failure state from JSON file. Returns True if loaded."""
        import json
        from pathlib import Path

        if not Path(path).exists():
            return False
        with open(path) as f:
            data = json.load(f)
        self.import_failure_state(data)
        return True

    def stats(self) -> dict[str, Any]:
        """Return runtime statistics."""
        entity_count = sum(1 for _ in self.root.walk())
        sensor_count = sum(len(ent.sensors) for ent in self.root.walk())
        modulator_count = sum(len(ent.modulators) for ent in self.root.walk())
        affordance_count = sum(sum(len(m.affordances) for m in ent.modulators.values()) for ent in self.root.walk())
        failure_mode_count = sum(len(ent.failure_modes) for ent in self.root.walk())
        active_failures = sum(1 for ent in self.root.walk() for fm in ent.failure_modes if fm.active)
        return {
            "entities": entity_count,
            "sensors": sensor_count,
            "modulators": modulator_count,
            "affordances": affordance_count,
            "failure_modes": failure_mode_count,
            "active_failures": active_failures,
            "failure_events": len(self._failure_history),
            "tick_count": self._tick_count,
        }
