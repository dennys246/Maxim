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
    EntropicDriveSpec,
    HomeostaticDriveSpec,
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


def _drive_potential_diff(
    body: Entity,
    effect: dict[str, float],
    pre_values: dict[str, float],
) -> float:
    """Net progress an affordance's ``effect`` moved the body's drives TOWARD comfort.

    For each sensor the ``effect`` touched that carries an entity-level drive
    spec, returns the SUM of ``drive_comfort_progress(spec, before, after)`` —
    POSITIVE when the action moved drives toward comfort, NEGATIVE when away. This
    is the state-conditioned reward substrate-primary selection needs so it can
    learn "turn TOWARD the sound" rather than merely "turning succeeds" (the June
    orient study — pain alone is positive-gated out of ``recommend_action``).

    **Value-based, not pain-based** (the #405 fix): ``drive_pain_for_value`` is a
    STEP function for entropic drives, so a warm/feed that reduces cold/hunger
    without crossing the deprivation threshold registered ZERO relief and starved
    warmth/feeding of credit (Exp 42 floored 60→8 contacts). ``drive_comfort_progress``
    is graded and nonzero for any real movement. The consumer signs the net so the
    cluster reward is ``±1`` — comparable to the tool-success signal non-drive
    actions get (a small magnitude would still lose the argmax to a flat ``+1``).

    Only entity-level drive sensors (``body.drive_specs``) are scored — the
    centeredness (azimuth) and hunger/thirst/energy drives all live there.
    Qualified modulator sub-sensors (``arms.thermal``) carry no drive spec today
    and are skipped; those affordances fall back to the ±1 tool-success signal.

    Scores the acting body's own ``self_effect`` only; a ``target_effect`` (a
    caregiver acting on another body, e.g. a mother feeding an infant) is
    intentionally NOT scored here — that relief is the other body's state, not
    the actor's learned policy. The caller passes ``self_effect`` as ``effect``.
    """
    from maxim.embodiment.sem import drive_comfort_progress

    drive_specs = getattr(body, "drive_specs", {}) or {}
    metrics = getattr(body, "vital_metrics", {}) or {}
    total = 0.0
    for name in effect:
        spec = drive_specs.get(name)
        if spec is None:
            continue
        before = pre_values.get(name)
        after = metrics.get(name)
        if before is None or after is None:
            continue
        total += drive_comfort_progress(spec, before, after)
    return total


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

    # Generated per-entity by ``generate_tools_for_entity``; lives in the
    # scene-scoped roster and grayscale-eligible when inactive. See
    # [docs/plans/deferred/sense_tool_registry.md] § "Tool metadata".
    kind = "sem-modulator-derived"

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


def _drive_failure_sensor(failure_name: str) -> str | None:
    """Sensor name for a drive-spec failure (``drive:<sensor>:discomfort`` /
    ``drive:<sensor>:deprived``), else ``None`` for a standard failure_mode.

    The sensor is everything between the ``drive:`` prefix and the final
    ``:<suffix>`` — so qualified sub-sensors like ``arms.thermal`` survive.
    """
    if not failure_name.startswith("drive:"):
        return None
    body = failure_name[len("drive:") :]
    idx = body.rfind(":")
    if idx <= 0:
        return None
    return body[:idx]


def _intrinsically_harmful_sensors(root: Entity, *effect_dicts: dict[str, float] | None) -> set[str]:
    """Sensors whose effect delta would breach a *healthy* sensor by itself.

    Delta-attribution (Exp 42): a drive failure on a sensor is attributed to
    the affordance only when the affordance's own delta is large enough to
    cause that failure on its own — i.e. a band-exceeding thermal shock, not a
    gentle touch. This prevents a bystander affordance (the safe warm,
    ``arms.thermal +0.05``) from inheriting blame for a lingering breach a
    HARMFUL affordance (``+0.6``) caused — the mis-attribution that collapsed
    Exp 42's safe-vs-harm discrimination once dense warming kept the breach
    omnipresent. The rule depends only on the affordance's delta and the
    sensor's drive spec, NOT on the current (possibly polluted) sensor state,
    so it attributes repeated harm correctly (every harmful warm is blamed,
    not just the first).

    LIMITATION (deliberate trade-off, review fold): this is a delta-ONLY
    heuristic — an affordance whose delta is *below* the band but that pushes an
    already-near-breach sensor over the line is a genuine partial cause yet is
    spared (false-negative attribution → its negative signal is lost). Exp 42
    avoids this because every harmful delta (+0.6) is well over the band (0.5).

    LOAD-BEARING, DO NOT REMOVE (2026-07-28 transition-drive-pain fold): an
    earlier draft of that plan proposed retiring this filter once drive pain
    became transition-based, on the theory that a latched emitter would fire
    only inside the causing action. The pre-merge two-lens round disproved it.
    The ``Body.evaluate_failures`` **direct** channel deliberately stayed
    state-based precisely because this filter is state-INDEPENDENT and so
    remains correct when a sensor **saturates** (``arms.thermal`` clamps at
    1.0, where no emitter-side severity latch can detect further deepening).
    Only the PainBus channel is latched. See the channel-split invariant in
    CLAUDE.md and docs/plans/deferred/transition_based_drive_pain.md.
    """
    specs: dict[str, Any] = {}
    for ent in root.walk():
        for ds_name, ds in getattr(ent, "drive_specs", {}).items():
            specs[ds_name] = ds

    harmful: set[str] = set()
    for effects in effect_dicts:
        if not effects:
            continue
        for sensor, delta in effects.items():
            ds = specs.get(sensor)
            if ds is None:
                continue
            try:
                d = float(delta)
            except (TypeError, ValueError):
                continue
            if isinstance(ds, HomeostaticDriveSpec):
                # Applying this delta to a sensor at set_point yields a
                # deviation of |delta|; harmful iff that alone exceeds the band.
                if abs(d) > ds.comfort_band:
                    harmful.add(sensor)
            elif isinstance(ds, EntropicDriveSpec):
                # Harmful iff the delta alone drives a healthy (0) sensor past
                # the deprivation threshold in the deprivation direction. Relief
                # deltas (warming reduces cold) never qualify.
                if ds.drift_direction == "up" and d >= ds.deprivation_threshold:
                    harmful.add(sensor)
                elif ds.drift_direction == "down" and d <= -ds.deprivation_threshold:
                    harmful.add(sensor)
    return harmful


class ModulatorAffordanceTool(Tool):
    """Auto-generated tool: execute one affordance on a modulator.

    After execution, reads back entity sensor state, evaluates failure
    modes immediately (don't wait for 1Hz poll), and feeds the actual
    sensor deltas to Cerebellum for forward model training.
    """

    # See SensorReadTool.kind — same provenance rationale.
    kind = "sem-modulator-derived"

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

        # --- Self-effect / target-effect apply BEFORE evaluate_failures ---
        # Order-of-operations invariant (Exp 37 root-cause fix, 2026-06-01):
        # body-sensor writes from self_effect / target_effect MUST happen
        # before the failure check so evaluate_failures sees the post-effect
        # state. Pre-fix the order was inverted, which made every
        # self_effect-based pain affordance (fire_pit_touch arms.thermal,
        # sharp_rock_pick_up laceration via acquired sharpness sensor)
        # silently produce empty active_failures. Pinned by
        # tests/unit/test_self_effect.py::TestSelfEffectFailureCascade.

        # Self-effect: voluntary affordance execution writes back to the
        # executor's own body.  Fires when the agent explicitly called the
        # tool (not reflex/orchestrator).  Supports entity-level sensors
        # ("hunger") and qualified modulator sub-sensors ("arms.thermal").
        # Motor-credit (GAP 1): score the drive relief this affordance's own
        # self_effect produces. Snapshot the pre-effect values of the drive
        # sensors it will touch, apply, then diff — positive = relief, so
        # substrate-primary selection learns "turn toward the sound," not just
        # "turning succeeds." Emitted on side_effects["drive_potential_diff"].
        # ``None`` = no drive sensor touched (or collateral harm — see the harm
        # gate after failure evaluation); a float (incl 0.0 / negatives) = the
        # measured net relief. ``accounted_sensors`` are the drive sensors this
        # diff represents, used by the collateral-harm gate below.
        drive_potential_diff: float | None = None
        accounted_sensors: set[str] = set()
        # Live-world-owned sensor filter (live_audio_orient_wiring.md
        # pre-merge fold): a sensor a LIVE exteroceptive writer owns (the
        # DoA feed world-setting ``azimuth``) is excluded from the modeled
        # self_effect ENTIRELY — no write (the next real measurement would
        # revert a fabricated shift), no credit (relief for actuation that
        # never happened books a repeatable phantom +1 — the credit mill),
        # and no B8 blame (the declared delta was never applied, so the
        # affordance is a bystander to any lingering breach). Empty set
        # (sim, scripted substrate, bodies without a live feed) leaves the
        # declared self_effect untouched. The eventual live credit design
        # is Stage 5's act-now-credit-later pending map.
        _self_effect = self._affordance_schema.self_effect
        _live_owned = getattr(self._embodiment, "live_world_set_sensors", None) or ()
        # Drive-touched-but-unmeasured marker (sem_motor_binding.md Phase 1):
        # when the affordance's DECLARED effect targets a drive sensor a
        # live measurement stream owns, the modeled credit is filtered and
        # no measured credit exists yet (Phase 2) — the consumer must NOT
        # fall through to the flat +1 tool-success floor. A real motor-bound
        # turn in a silent room would otherwise mint direction-blind +1s
        # into the cluster surface (the probe-3 floor-drowning failure).
        drive_credit_withheld = False
        if _self_effect and _live_owned:
            _body_for_drives = getattr(self._embodiment, "root", None)
            _drive_names = set(getattr(_body_for_drives, "drive_specs", {}) or {})
            drive_credit_withheld = bool(set(_self_effect) & set(_live_owned) & _drive_names)
            _self_effect = {k: v for k, v in _self_effect.items() if k not in _live_owned}
        if _self_effect and self._embodiment is not None:
            _body = self._embodiment.root
            _drive_specs = getattr(_body, "drive_specs", {}) or {}
            _metrics = getattr(_body, "vital_metrics", {}) or {}
            pre_values = {name: _metrics[name] for name in _self_effect if name in _drive_specs and name in _metrics}
            accounted_sensors = set(pre_values)
            _apply_sensor_deltas(
                self._embodiment.root,
                _self_effect,
                delta_kind="self_effect",
            )
            if pre_values:
                drive_potential_diff = _drive_potential_diff(
                    self._embodiment.root,
                    _self_effect,
                    pre_values,
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

        # Immediate failure evaluation — don't wait for 1Hz poll.
        # Now that self_effect / target_effect have applied above, the
        # body sensors reflect the post-effect state and any drive-band
        # breach (e.g., arms.thermal > comfort_band) is detected here.
        active_failures: list[dict[str, Any]] = []
        if self._embodiment is not None:
            failure_events: list[Any] = []
            try:
                failure_events = self._embodiment.evaluate_failures()
            except Exception:
                log.debug("evaluate_failures raised during affordance execute", exc_info=True)

            def _as_dict(ev: Any) -> dict[str, Any]:
                return {"name": ev.failure_name, "entity": ev.entity_path, "pain": ev.pain_intensity}

            # Delta-attribution (Exp 42): a drive-spec failure is reported as
            # caused by THIS affordance only if its own self/target_effect delta
            # is intrinsically harmful to that sensor (would breach a healthy
            # sensor). A gentle bystander affordance executing while a HARMFUL
            # affordance's breach lingers is NOT blamed — that bystander
            # mis-attribution polluted the safe source's learning and collapsed
            # the safe-vs-harm discrimination. Standard (non-drive) failure_modes
            # pass through unchanged so the SEM pain cascade and Exp 37/38 are
            # preserved.
            #
            # SCOPE (review fold): this filters the side_effects/direct-attribution
            # channel only. The PARALLEL channel — evaluate_failures() →
            # _publish_drive_pain() → PainBus → create_pain_nac_subscriber
            # (context-similarity) — is NOT filtered here; it does not re-pollute
            # in Exp 42 (safe net stays +0.99), but the proper root-cause fix is
            # transition-based drive-pain in body.py (fire on band ENTRY, not
            # every tick) — tracked in docs/plans/deferred/transition_based_drive_pain.md.
            # Do NOT assume both channels are delta-attributed.
            #
            # On ANY filter error, fall back to the UNFILTERED events rather than
            # silently dropping the causing tool's failure (the unfiltered list is
            # the pre-B8 behavior) — a filter bug must not become a silent no-op
            # on the SEM-cascade / Exp 37/38 path.
            try:
                # NOTE: _self_effect (live-world-owned sensors filtered), not
                # the raw declaration — an unapplied delta must not blame the
                # affordance for a breach it did not touch.
                harmful_sensors = _intrinsically_harmful_sensors(
                    self._embodiment.root,
                    _self_effect,
                    self._affordance_schema.target_effect,
                )
                active_failures = [
                    _as_dict(ev)
                    for ev in failure_events
                    if (s := _drive_failure_sensor(ev.failure_name)) is None or s in harmful_sensors
                ]
            except Exception:
                log.debug(
                    "delta-attribution filter raised; falling back to unfiltered failures",
                    exc_info=True,
                )
                active_failures = [_as_dict(ev) for ev in failure_events]

        # Motor-credit harm gate (GAP 1, pre-merge review fold): a positive
        # relief signal is trustworthy only if the action caused no COLLATERAL
        # harm — a failure on a sensor its potential_diff did NOT account for.
        #   - SAME-sensor drive discomfort (e.g. azimuth still off-center after a
        #     relieving turn) is NOT collateral: potential_diff already reflects
        #     that sensor's net change (relief or worsening). Nulling on it would
        #     make EVERY orient turn — which leaves the body still off-center and
        #     so always trips drive:azimuth:discomfort — lose its relief and be
        #     mis-credited as harm. That would silently defeat GAP 1.
        #   - A non-drive failure_mode (thermal shock, laceration) or a failure on
        #     an UNTOUCHED drive sensor IS collateral — not captured by this
        #     action's relief — so drop the signal (None) and let the consumer's
        #     harm fallback (learn_success False -> -1) dominate. This is the
        #     attractive-but-harmful (deceptive-hearth) case the review flagged.
        if drive_potential_diff is not None and active_failures:
            for _f in active_failures:
                if _drive_failure_sensor(_f.get("name", "")) not in accounted_sensors:
                    drive_potential_diff = None
                    break

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

        # Build side_effects dict
        side_effects: dict[str, Any] | None = None
        if active_failures:
            side_effects = {"embodiment_failures": active_failures}

        # Motor-credit signal: the net value-progress toward comfort this action
        # produced (drive_comfort_progress). Emitted (incl. 0.0 and negatives)
        # whenever the self_effect touched an entity-level drive sensor AND caused
        # no collateral harm; ``None`` (key absent) means "no drive sensor touched,
        # or collateral harm". The consumer (runtime/tool_dispatch.py) takes the
        # SIGN (±1); a present 0.0 (no net progress) falls back to the tool-success
        # signal. See docs/user/tool_side_effects.md.
        if drive_potential_diff is not None:
            if side_effects is None:
                side_effects = {}
            side_effects["drive_potential_diff"] = drive_potential_diff
        elif drive_credit_withheld:
            # See the marker above: drive-touched-but-unmeasured. The
            # consumer treats this like drive_relief_only for the floor
            # decision — no cluster credit either way, tool-success ±1
            # still flows to the causal-link surface as usual.
            if side_effects is None:
                side_effects = {}
            side_effects["drive_credit_withheld"] = True

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

    # See SensorReadTool.kind — same provenance rationale.
    kind = "sem-modulator-derived"

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
