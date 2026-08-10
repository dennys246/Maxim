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
from collections.abc import Iterable
from typing import Any

from maxim.embodiment.sem import Entity, FailureMode, SensorReading
from maxim.utils.logging import log_swallowed_exception

log = logging.getLogger(__name__)

# Drive-pain breach latch tuning (channel 2 / PainBus only — see
# evaluate_failures). A breach re-publishes only when severity grows by more
# than `_BREACH_DEEPEN_FRACTION` of the spec's own band/threshold gap (floored
# at `_BREACH_MIN_EPS`), so per-tick drift creep cannot re-fire while genuine
# re-injury does. `_BREACH_HYSTERESIS` pulls the homeostatic *recovery* point
# strictly inside the firing point so a noisy world-set sensor sitting on the
# band edge cannot chatter one "onset" per jitter. Entropic drives declare
# their own recovery point (`satisfaction_threshold`) so they do not use it.
_BREACH_DEEPEN_FRACTION = 0.05
_BREACH_MIN_EPS = 1e-3
_BREACH_HYSTERESIS = 0.2


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
        distributor: Any | None = None,
        agent_id: str = "",
    ) -> None:
        self.root = root
        self.config = config or EmbodimentConfig()
        self._pain_bus = pain_bus
        self._distributor = distributor  # TemporalCreditDistributor for SCN drive events
        # agent_id flows into every PainSignal.context this body publishes
        # so reaction_bus subscribers (notably _distribute_reward_from_reaction
        # in bio_stack.py) can credit the right agent's eligibility traces.
        # Empty string means "agent_id-unaware" and reactions will be silently
        # skipped by the reward distributor — fine for foundry test embodiments
        # and scene-entity Embodiment wrappers that aren't a learning subject.
        self.agent_id = agent_id
        self._failure_history: list[FailureEvent] = []
        self._last_poll: float = 0.0
        self._tick_count: int = 0
        # Sensors owned by a LIVE exteroceptive writer (e.g. the DoA feed
        # world-setting ``azimuth`` from real measurements). While a sensor
        # is in this set, ``ModulatorAffordanceTool.execute`` excludes it
        # from MODELED self_effect writes AND from drive credit/blame: a
        # modeled shift on a world-measured sensor is a fabrication the
        # next reading reverts, and crediting it books relief for actuation
        # that never happened — repeatably (the pre-merge review's phantom
        # credit mill, live_audio_orient_wiring.md). Runtime-only state;
        # never serialized. Empty (the default) = sim semantics, unchanged.
        self.live_world_set_sensors: set[str] = set()

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

        Automatically applies drive drift before evaluation: when
        ``evaluate_failures`` is called from tool_bridge or simulation
        tools (paths that don't use EmbodimentPerceptSource), drive
        drift would otherwise never execute.  The dt is computed from
        wall-clock time since the last evaluation.

        When an entity has ``metadata["health"] == "derived"``, entity
        health is computed from modulator component integrities before
        failure mode evaluation.  This enables component-level damage
        to cascade upward to entity-level failure modes (e.g., wing
        integrity < 0.3 → entity health drops → death failure mode).

        Drive-spec evaluation:
        - Homeostatic drives emit pain proportional to deviation from
          set_point beyond comfort_band.
        - Entropic drives emit pain when crossing deprivation_threshold
          (handled via auto-generated failure modes at parse time, or
          checked here for entities loaded without failure mode generation).
        """
        from maxim.embodiment.sem import (
            EntropicDriveSpec,
            HomeostaticDriveSpec,
            drive_pain_for_value,
        )

        # Apply drive drift before failure evaluation.  This ensures
        # drives advance even in code paths that don't use
        # EmbodimentPerceptSource (generative runner, tool_bridge
        # immediate evaluation, sim tools).
        now = time.time()
        if self._last_poll > 0:
            drift_dt = now - self._last_poll
            if drift_dt > 0:
                self.tick_vital_drift(drift_dt)
        self._last_poll = now

        events: list[FailureEvent] = []

        for ent in self.root.walk():
            # Derive entity health from component integrities if configured
            if ent.metadata.get("health") == "derived":
                derived = ent.derive_health()
                if derived is not None:
                    ent.vital_metrics["health"] = derived

            # Include per-modulator integrity readings so failure modes
            # can trigger on component state (e.g., trigger on wing.integrity)
            for mod_name, mod in ent.modulators.items():
                if hasattr(mod, "compute_integrity") and hasattr(mod, "vital_metrics") and mod.vital_metrics:
                    ent.vital_metrics[f"{mod_name}.integrity"] = mod.compute_integrity()

            # collect scalar sensor values for trigger evaluation
            readings: dict[str, float] = {}
            for sname, sensor in ent.sensors.items():
                schema = sensor.reading_schema
                if schema.get("type") in ("float", "int"):
                    try:
                        r = sensor.read()
                        readings[sname] = float(r.value)
                    except Exception:
                        log_swallowed_exception()

            # also include vital_metrics (may have drifted values)
            for vname, vval in ent.vital_metrics.items():
                if vname not in readings:
                    readings[vname] = vval

            # Include modulator sub-sensor vital_metrics for drive evaluation
            for mod_name, mod in ent.modulators.items():
                if hasattr(mod, "vital_metrics"):
                    for ms_name, ms_val in mod.vital_metrics.items():
                        qualified = f"{mod_name}.{ms_name}"
                        if qualified not in readings:
                            readings[qualified] = ms_val

            # Log sensor readings for display/JSONL (Track 5: SEM observability)
            try:
                from maxim.simulation.sim_logger import sim_sensor

                for sname, sval in readings.items():
                    baseline = ent.vital_metrics.get(sname)
                    sim_sensor(ent.full_path, sname, sval, baseline=baseline)
            except Exception:
                log_swallowed_exception()

            # -- Standard failure mode evaluation --
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

            # -- Drive-spec pain evaluation --
            for ds_name, ds in ent.drive_specs.items():
                current = readings.get(ds_name)
                if current is None:
                    # Unreadable sensor (backend error, modulator removed,
                    # entity detached mid-session). Treat "unknown" as "not
                    # breached" and drop any latched severity — retaining it
                    # would silently swallow the next genuine breach when the
                    # sensor comes back out of band.
                    ent.drive_breach_severity.pop(ds_name, None)
                    continue

                # Drive state trace (for --display debug / JSONL)
                try:
                    from maxim.simulation.sim_logger import sim_log

                    if isinstance(ds, HomeostaticDriveSpec):
                        deviation = abs(current - ds.set_point)
                        sim_log(
                            f"drive:{ds_name}",
                            f"val={current:.3f} set={ds.set_point} dev={deviation:.3f} band={ds.comfort_band}",
                        )
                    elif isinstance(ds, EntropicDriveSpec):
                        sim_log(
                            f"drive:{ds_name}",
                            f"val={current:.3f} threshold={ds.deprivation_threshold} dir={ds.drift_direction}",
                        )
                except Exception:
                    log_swallowed_exception()

                # Transition latch: pain fires on band ENTRY only, so the
                # crossing lands inside the CAUSING action's execute and a
                # bystander evaluating during a lingering breach emits
                # nothing — on both attribution channels. Motivation is
                # CHANNEL SPLIT (pre-merge two-lens fold — see the
                # transition-based drive-pain invariant in CLAUDE.md):
                #
                #  * The returned FailureEvent list (channel 1, direct
                #    attribution via ToolOutput.side_effects) stays
                #    STATE-BASED — one event per call while breached.
                #    tool_bridge's B8 delta filter already attributes this
                #    channel correctly, and it is state-INDEPENDENT, so it
                #    stays right even when a sensor saturates (where a
                #    level latch goes blind). Latching here silently
                #    starved that filter and flipped a repeat harmful
                #    affordance to POSITIVE credit.
                #  * _publish_drive_pain (channel 2, the unfiltered PainBus
                #    path this plan was written to fix) is latched on
                #    SEVERITY: it fires on band entry and again only when
                #    the breach materially DEEPENS (re-injury), which is
                #    the bio-faithful shape — repeated noxious stimulus of
                #    already-injured tissue sensitizes, it does not go
                #    silent. Steady-state and recovery are silent, which
                #    kills the per-tick flood.
                #
                # Severity is in SENSOR units for both drive kinds (excess
                # past the comfort band / excursion past the deprivation
                # threshold) so the epsilon is comparable to the spec's own
                # thresholds. Clearing uses HYSTERESIS (a recovery point
                # strictly inside the firing point) so a noisy world-set
                # sensor hovering at the boundary — live DoA azimuth is
                # exactly this — cannot chatter one "onset" per jitter.
                # The latch lives on the ENTITY, not this wrapper, so it
                # survives reparenting (entity acquisition) and ephemeral
                # per-invocation Embodiment wrappers, and cannot collide
                # between same-named siblings.
                breach_latch = ent.drive_breach_severity

                if isinstance(ds, HomeostaticDriveSpec):
                    # drive_pain_for_value is the single source of truth for the
                    # pain formula (shared with the motor-credit potential_diff).
                    # Behaviour-preserving: the FailureEvent and _publish_drive_pain
                    # both already clamped to [0, 1], which is exactly what the
                    # helper returns.
                    pain = drive_pain_for_value(ds, current)
                    deviation = abs(current - ds.set_point)
                    severity = max(0.0, deviation - ds.comfort_band)
                    # Clear only once comfortably back inside the band.
                    cleared = deviation <= ds.comfort_band * (1.0 - _BREACH_HYSTERESIS)
                    eps = max(_BREACH_MIN_EPS, _BREACH_DEEPEN_FRACTION * ds.comfort_band)
                    if pain > 0:
                        event = FailureEvent(
                            entity_path=ent.full_path,
                            failure_name=f"drive:{ds_name}:discomfort",
                            pain_intensity=pain,
                            sensor_readings=dict(readings),
                        )
                        events.append(event)
                        self._failure_history.append(event)
                        latched = breach_latch.get(ds_name)
                        if latched is None or severity > latched + eps:
                            breach_latch[ds_name] = severity
                            self._publish_drive_pain(ent, ds_name, pain, readings)
                    elif cleared:
                        breach_latch.pop(ds_name, None)

                elif isinstance(ds, EntropicDriveSpec):
                    # NB: this inline threshold check mirrors the entropic branch
                    # of drive_pain_for_value; kept explicit here to preserve the
                    # exact fire-on-threshold semantics regardless of the
                    # (degenerate) deprivation_pain == 0 config.
                    if ds.drift_direction == "up":
                        deprived = current >= ds.deprivation_threshold
                        severity = max(0.0, current - ds.deprivation_threshold)
                        # satisfaction_threshold is the schema's own recovery
                        # point — strictly inside deprivation_threshold, i.e.
                        # the hysteresis is already declared on the spec.
                        cleared = current <= ds.satisfaction_threshold
                    elif ds.drift_direction == "down":
                        deprived = current <= ds.deprivation_threshold
                        severity = max(0.0, ds.deprivation_threshold - current)
                        cleared = current >= ds.satisfaction_threshold
                    else:
                        deprived, severity, cleared = False, 0.0, False
                    eps = max(
                        _BREACH_MIN_EPS,
                        _BREACH_DEEPEN_FRACTION * abs(ds.deprivation_threshold - ds.satisfaction_threshold),
                    )
                    if deprived:
                        event = FailureEvent(
                            entity_path=ent.full_path,
                            failure_name=f"drive:{ds_name}:deprived",
                            pain_intensity=ds.deprivation_pain,
                            sensor_readings=dict(readings),
                        )
                        events.append(event)
                        self._failure_history.append(event)
                        latched = breach_latch.get(ds_name)
                        if latched is None or severity > latched + eps:
                            breach_latch[ds_name] = severity
                            self._publish_drive_pain(
                                ent, ds_name, ds.deprivation_pain, readings, event_suffix="deprived"
                            )
                    elif cleared:
                        breach_latch.pop(ds_name, None)

        return events

    def _publish_drive_pain(
        self,
        entity: Entity,
        drive_name: str,
        intensity: float,
        readings: dict[str, float],
        *,
        event_suffix: str = "discomfort",
    ) -> None:
        """Publish a PainSignal for a drive-spec threshold crossing."""
        if not self.config.enable_pain or self._pain_bus is None:
            return

        try:
            from maxim.proprioception.pain import PainSignal, PainType

            signal = PainSignal(
                pain_type=PainType.EXTERNAL_SIGNAL,
                intensity=min(1.0, max(0.0, intensity)),
                timestamp=time.time(),
                context={
                    "source": f"drive:{drive_name}",
                    "entity": entity.full_path,
                    # See _publish_pain for the entity_type vs
                    # entity_name distinction — drive pain ships both
                    # for parity with embodiment-failure pain.
                    "entity_type": entity.entity_type,
                    "entity_name": entity.name,
                    "failure_mode": f"drive:{drive_name}",
                    "sensor_readings": {k: v for k, v in readings.items() if isinstance(v, (int, float))},
                    "entity_path": entity.full_path,
                    "agent_id": self.agent_id,
                },
            )
            self._pain_bus.publish(signal)
        except Exception as exc:
            log.debug("Drive pain publish failed for %s: %s", drive_name, exc)

        # Emit TemporalEvent for SCN oscillator learning (best-effort)
        self._emit_drive_temporal_event(f"drive:{drive_name}:{event_suffix}", entity.name)

    def _emit_drive_temporal_event(self, event_type: str, agent_id: str) -> None:
        """Emit a TemporalEvent for a drive state transition (best-effort)."""
        if self._distributor is None:
            return
        try:
            from maxim.time.temporal_event import TemporalEvent
            from maxim.time.temporal_signature import TemporalSignature

            event = TemporalEvent(
                event_type=event_type,
                agent_id=agent_id,
                temporal_signature=TemporalSignature.now(),
                metadata={"source": "drive_protocol"},
            )
            self._distributor.record_event(event)
        except Exception as exc:
            log.debug("Drive temporal event failed for %s: %s", event_type, exc)

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
                    # entity_type is the YAML CATEGORY ("creature",
                    # "weapon", "body") — coarser than needed for
                    # Pavlovian percept aversion (Wire 2 needs noun-
                    # level discrimination so dragon-pain doesn't
                    # generalize to wolf-pain).  entity_name is the
                    # YAML noun ("dragon", "rusty_sword") and is the
                    # canonical Wire 2 percept-aversion key.  Both
                    # fields ship; consumers pick based on the
                    # abstraction level they need.
                    "entity_type": entity.entity_type,
                    "entity_name": entity.name,
                    "failure_mode": failure.name,
                    "composes": list(failure.composes or []),
                    "sensor_readings": dict(readings),
                    # Retained for legacy consumers that read entity_path.
                    "entity_path": entity.full_path,
                    "agent_id": self.agent_id,
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
        """Apply drift to vital metrics via drive specs or legacy hardcoded names.

        Called once per poll cycle.  Drive specs on the entity take
        precedence over the legacy hardcoded metric-name dispatch.
        Homeostatic drives drift toward ``set_point``; entropic drives
        drift in ``drift_direction``.
        """
        from maxim.embodiment.sem import EntropicDriveSpec, HomeostaticDriveSpec

        rate = self.config.vital_drift_rate
        for ent in self.root.walk():
            # --- Drive-spec-based drift (preferred) ---
            for ds_name, ds in ent.drive_specs.items():
                # Resolve the sensor value — could be entity-level or modulator-level
                if "." in ds_name:
                    mod_name, sensor_name = ds_name.split(".", 1)
                    mod = ent.modulators.get(mod_name)
                    if mod is None or not hasattr(mod, "vital_metrics"):
                        continue
                    current = mod.vital_metrics.get(sensor_name)
                    if current is None:
                        continue
                    if isinstance(ds, HomeostaticDriveSpec):
                        delta = ds.set_point - current
                        step = min(abs(delta), ds.drift_rate * dt)
                        mod.vital_metrics[sensor_name] = current + (step if delta > 0 else -step)
                    elif isinstance(ds, EntropicDriveSpec):
                        if ds.drift_direction == "up":
                            mod.vital_metrics[sensor_name] = min(1.0, current + ds.drift_rate * dt)
                        else:
                            mod.vital_metrics[sensor_name] = max(0.0, current - ds.drift_rate * dt)
                else:
                    current = ent.vital_metrics.get(ds_name)
                    if current is None:
                        continue
                    if isinstance(ds, HomeostaticDriveSpec):
                        delta = ds.set_point - current
                        step = min(abs(delta), ds.drift_rate * dt)
                        ent.vital_metrics[ds_name] = current + (step if delta > 0 else -step)
                    elif isinstance(ds, EntropicDriveSpec):
                        if ds.drift_direction == "up":
                            ent.vital_metrics[ds_name] = min(1.0, current + ds.drift_rate * dt)
                        else:
                            ent.vital_metrics[ds_name] = max(0.0, current - ds.drift_rate * dt)

            # --- Legacy hardcoded drift (for entities without drive specs) ---
            driven_sensors = set(ent.drive_specs.keys())
            for vname in list(ent.vital_metrics.keys()):
                if vname in driven_sensors:
                    continue  # already handled by drive spec
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
                    log_swallowed_exception()
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

            # Include drive state from vital_metrics for sensors with DriveSpecs
            # that may not be captured by sensor.read() (vital_metrics path)
            from maxim.embodiment.sem import EntropicDriveSpec, HomeostaticDriveSpec

            for ds_name, ds in ent.drive_specs.items():
                if ds_name in ent_state["sensors"]:
                    # Already captured — add drive annotation
                    val = ent_state["sensors"][ds_name]["value"]
                elif "." not in ds_name and ds_name in ent.vital_metrics:
                    # Entity-level vital metric not captured by sensor read
                    val = ent.vital_metrics[ds_name]
                    ent_state["sensors"][ds_name] = {"value": val, "unit": "ratio"}
                else:
                    continue

                # Annotate with drive state
                if isinstance(ds, HomeostaticDriveSpec):
                    deviation = abs(val - ds.set_point)
                    if deviation > ds.comfort_band:
                        excess = deviation - ds.comfort_band
                        ent_state["sensors"][ds_name]["drive"] = (
                            f"outside comfort band, discomfort {excess * ds.pain_scale:.2f}"
                        )
                    else:
                        ent_state["sensors"][ds_name]["drive"] = "comfortable"
                elif isinstance(ds, EntropicDriveSpec):
                    if ds.drift_direction == "up" and val >= ds.deprivation_threshold:
                        ent_state["sensors"][ds_name]["drive"] = f"deprived, intensity {ds.deprivation_pain:.2f}"
                    elif ds.drift_direction == "down" and val <= ds.deprivation_threshold:
                        ent_state["sensors"][ds_name]["drive"] = f"deprived, intensity {ds.deprivation_pain:.2f}"
                    elif ds.drift_direction == "up" and val > ds.satisfaction_threshold:
                        ent_state["sensors"][ds_name]["drive"] = "rising"
                    else:
                        ent_state["sensors"][ds_name]["drive"] = "satisfied"

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
                drive = sinfo.get("drive")
                parts = [f"- {ent_state['entity']}.{sname}: {val}{unit}"]
                if drive:
                    parts.append(f"(DRIVE: {drive})")
                if warning:
                    parts.append(f"(WARN: {warning})")
                    has_warnings = True
                lines.append(" ".join(parts))

        if has_warnings:
            lines[0] = "=== Body State (pain-relevant) ==="

        return "\n".join(lines)

    # -- stats --------------------------------------------------------------

    @property
    def failure_history(self) -> list[FailureEvent]:
        return list(self._failure_history)

    # -- Wire 3: embodiment-state → action filter (release_0_9_1.md Stage 1) -
    #
    # **I/O-layer boundary, not substrate contamination.** The thresholds
    # gate the LLM proposer's tool surface, NOT substrate encoding.
    # EC clusters, NAc reward_bias, and the natural failure → pain →
    # NAc learning chain are untouched. This is the same downstream-of-
    # encoding exemption Wire-A's bias-band labels operate under (per
    # bio-fidelity pre-merge review).
    #
    # Default thresholds (also documented in
    # docs/plans/deferred/bio_emergent_persona_foundations.md § Wire 3).
    # The agent_loop hook reads these via the method signature so a
    # non-default threshold pair can ship in a future tuning experiment
    # without touching call sites.
    #
    # **Band semantics (pinned in tests at the strict-vs-inclusive split):**
    # - ``integrity < 0.3``         → disabled (filtered from prompt)
    # - ``0.3 <= integrity < 0.6``  → degraded (annotated in prompt)
    # - ``integrity >= 0.6``        → healthy (no annotation)
    # The bands partition [0, 1] cleanly — no overlap, no gap.

    _WIRE_3_DISABLE_THRESHOLD: float = 0.3
    _WIRE_3_DEGRADE_THRESHOLD: float = 0.6

    def _iter_modulator_affordance_pairs(
        self,
    ) -> Iterable[tuple[str, float]]:
        """Yield ``(base_tool_name, integrity)`` for every modulator
        affordance on the entity tree.

        ``base_tool_name`` is the ``{entity.name}_{affordance_name}``
        form ``tool_bridge.generate_tools_for_entity`` uses BEFORE
        ``_resolve_tool_name`` disambiguates against the registry.
        On the common single-body topology (Roy / cradle / Reachy)
        there are no name collisions, so the registered tool name
        equals the base name and Wire 3 matches cleanly. The
        agent_loop hook compares against the live tool list — if a
        collision did rename a tool to ``{ancestor}_{base_name}``,
        the integrity filter fails open (the tool stays available)
        rather than silently mis-gating.

        Modulators without a ``compute_integrity`` method (older
        SpecModulator-shaped types, capability-only modulators) yield
        ``1.0`` — equivalent to "not damaged", per the same
        backward-compat convention ``SpecModulator.compute_integrity``
        uses when ``vital_metrics`` is empty.
        """
        for ent in self.root.walk():
            for mod in ent.modulators.values():
                if hasattr(mod, "compute_integrity"):
                    try:
                        integrity = float(mod.compute_integrity())
                    except Exception as e:
                        # Bio-fidelity fold (Wire 3 review): a broken
                        # integrity calc is itself a signal the body's
                        # self-monitoring is failing. Currently fail-open
                        # to integrity=1.0 (preserves loop stability)
                        # but surface as WARNING so the broken modulator
                        # is visible in operator review / Roy-3 logs.
                        # Treat-as-disabled (more cautious) is the bio-
                        # faithful alternative, deferred to a future
                        # tuning experiment.
                        log.warning(
                            "Wire 3: compute_integrity() raised on %s/%s — treating as healthy (1.0): %s",
                            ent.name,
                            getattr(mod, "name", "?"),
                            e,
                        )
                        integrity = 1.0
                else:
                    integrity = 1.0
                if not hasattr(mod, "affordances"):
                    continue
                for aff_name in mod.affordances.keys():
                    yield f"{ent.name}_{aff_name}", integrity

    def get_disabled_affordances(self, *, threshold: float | None = None) -> set[str]:
        """Affordances routed through critically-damaged components.

        Returns the set of base tool names (``{entity.name}_{affordance_name}``)
        whose owning modulator's ``compute_integrity()`` is **strictly
        below** the disable threshold (default ``0.3``). The agent_loop
        hook filters these from the per-tick available-tools list
        BEFORE the LLM prompt sees them — a damaged-arm agent stops
        attempting arm-routed affordances without any prompt-injection
        scaffolding, the cleanest emergent "trait" demonstration in
        bio_emergent_persona_foundations.md § Wire 3.

        See ``_iter_modulator_affordance_pairs`` for the base-name
        derivation contract; failures match cleanly on Roy's
        single-body topology and fail-open under name collisions.

        Args:
            threshold: Override ``_WIRE_3_DISABLE_THRESHOLD`` (0.3).
                Below this integrity, the affordance is disabled.
        """
        cutoff = float(threshold) if threshold is not None else self._WIRE_3_DISABLE_THRESHOLD
        return {name for name, integrity in self._iter_modulator_affordance_pairs() if integrity < cutoff}

    @staticmethod
    def integrity_to_felt_phrase(integrity: float) -> str:
        """Map a degraded-band integrity value to a felt-sensation phrase.

        Per bio-fidelity pre-merge review (Wire 3 fold), the prompt-
        visible annotation reads as proprioceptive percept ("feels
        strained", "feels weakened") rather than as a system advisor
        ("DAMAGED: integrity 0.4"). The numeric integrity stays in the
        ``sim_log("WIRE_3_FILTER", ...)`` JSONL event for post-hoc
        Roy-3 analysis; the LLM sees the qualitative phrase only.

        Mirrors Wire-A's ``bias_to_band`` 5-band approach but with
        2 bands inside the narrower degraded range [0.3, 0.6):

        - ``0.45 <= integrity < 0.6``  → ``"feels strained"``
        - ``0.3 <= integrity < 0.45``  → ``"feels weakened, prone to failing"``

        Values outside the degraded range return ``""`` (the caller
        is the agent_loop hook, which only invokes this method on
        values it knows are in the degraded band; the empty-string
        case is defensive — never happens via the documented flow).
        """
        if integrity >= 0.45 and integrity < 0.6:
            return "feels strained"
        if integrity >= 0.3 and integrity < 0.45:
            return "feels weakened, prone to failing"
        return ""

    def get_degraded_affordances(
        self,
        *,
        disable_threshold: float | None = None,
        degrade_threshold: float | None = None,
    ) -> dict[str, float]:
        """Affordances on partially-damaged components.

        Returns ``{base_tool_name: integrity}`` for every modulator
        affordance whose owning modulator's integrity is in the
        ``[disable_threshold, degrade_threshold)`` range — damaged
        but not disabled. The agent_loop hook annotates these tools'
        descriptions with ``[DAMAGED: integrity 0.X]`` so the LLM
        proposer sees the cost of using them; learning is post-hoc
        via the standard reward path (damaged-tool use → likelier
        failure → NAc credit).

        Args:
            disable_threshold: Below this integrity, the affordance
                is in ``get_disabled_affordances`` instead and is NOT
                in this map. Default 0.3.
            degrade_threshold: At or above this integrity, the
                affordance is healthy and NOT in this map. Default 0.6.
        """
        lo = float(disable_threshold) if disable_threshold is not None else self._WIRE_3_DISABLE_THRESHOLD
        hi = float(degrade_threshold) if degrade_threshold is not None else self._WIRE_3_DEGRADE_THRESHOLD
        return {name: integrity for name, integrity in self._iter_modulator_affordance_pairs() if lo <= integrity < hi}

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
            from maxim.utils.format_version import with_format_version

            atomic_write_json(path, with_format_version(state))
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
        from maxim.utils.format_version import check_format_version

        check_format_version(data, "body_failures", log=log)
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
