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

import logging
import math
import operator
import time
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)

# The Entity JSON format (``Entity.save`` / ``Entity.load``). 1.1 (#1124): each modulator carries its
# sub-sensor ``values``, ``integrity`` function and ``damage_affinities``; the entity's ``vital_metrics``
# carries no dotted keys. A 1.0 file still loads: its modulators start from their sensor spec and its
# dotted keys are dropped with a WARNING.
ENTITY_FORMAT_VERSION: str = "1.1"


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
    Pain fires when the value crosses ``deprivation_threshold``.  No positive
    Reaction fires when the drive crosses back past ``satisfaction_threshold``;
    the crossing only clears the breach latch
    (``embodiment/body.py::Embodiment.evaluate_failures``) and sets the
    "rising"/"satisfied" label (``Embodiment.body_state_summary``).  The
    threshold also bounds the relief/pressure spans
    (``corrective_need_intensity``, ``drive_span``,
    ``relief_fraction_from_progress``, ``drive_pressure``), and per-action
    relief IS credited (``drive_comfort_progress`` -> +/-1 motor credit in
    ``runtime/tool_dispatch.py``).  What was never built is a Reaction on the
    crossing (the grounding line's GL2c, ``docs/plans/grounding.md``).

    SHAPE-FROZEN at 1.0 (CC3). YAML-parsed drive contract. Adding any
    new field post-1.0 is a major-version-bump change. Per CLAUDE.md:
    "Do NOT add fields to these frozen dataclasses post-1.0 without a
    breaking-change plan."
    """

    drift_direction: str  # "up" or "down"
    drift_rate: float  # per-second drift rate
    deprivation_threshold: float  # PainSignal fires beyond this
    deprivation_pain: float  # pain intensity at deprivation
    satisfaction_threshold: float  # crossing back clears the breach latch; bounds relief spans; no Reaction (GL2c)
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


def corrective_need_intensity(spec: DriveSpec, value: float) -> float | None:
    """Cold-start action-selection PRIOR: the deficit intensity in [0, 1] a drive
    emits as a *corrective need*, or None when the drive is not in deficit.

    Deliberately NOT ``drive_pain_for_value``: this feeds the substrate-primary
    action-selection prior (``_read_drive_states`` → ``NAc.recommend_action``), which
    wants a simple monotone deficit magnitude AND must keep the pre-existing "cold"
    derivation BYTE-IDENTICAL — ``min(1, |value - set_point|)`` — whereas
    ``drive_pain_for_value`` weights by ``comfort_band``/``pain_scale``, which would move
    every thermal body's cold intensity. Kept here (the embodiment layer owns drive
    semantics), dispatched by ``isinstance`` beside the pain/comfort helpers, rather
    than re-derived in the runtime.

    Homeostatic: below set_point past the comfort band → ``min(1, |value - set_point|)``.
    Entropic draining ("down"): graded from the satisfaction threshold down to the
    deprivation threshold. Entropic "up" drives and above-set_point deficits have no
    corrective direction here (return None) — the 1.3 survival-loop scope (break 1).
    """
    if isinstance(spec, HomeostaticDriveSpec):
        deviation = value - spec.set_point
        if deviation < -spec.comfort_band:
            return min(1.0, abs(deviation))
        return None
    if isinstance(spec, EntropicDriveSpec):
        if spec.drift_direction == "down" and value < spec.satisfaction_threshold:
            span = spec.satisfaction_threshold - spec.deprivation_threshold
            if span <= 0:
                return 1.0
            return max(0.0, min(1.0, (spec.satisfaction_threshold - value) / span))
    return None


def drive_comfort_progress(spec: DriveSpec, before: float, after: float) -> float:
    """How far an action moved a drive TOWARD comfort (before → after).

    Positive = toward comfort, negative = away. This is the motor-credit signal,
    and it is deliberately **value-based, not pain-based** — ``drive_pain_for_value``
    is a STEP function for entropic drives (``deprivation_pain`` past the threshold,
    ``0`` below), so a ``warm_self`` / ``feed`` that reduces cold/hunger but stays
    past the threshold registers ZERO pain-reduction even though it made real
    progress. Rewarding on pain-reduction therefore starves entropic-relief
    actions (warmth, feeding) of credit and the substrate-primary agent abandons
    them (the #405 Exp-42 floor). Value-progress is graded and nonzero for any
    real movement toward comfort:

    - **Homeostatic** (regulate to ``set_point``): reduction in absolute deviation,
      ``|before - set_point| - |after - set_point|`` (positive = moved toward the
      set point; this is what orient/azimuth centeredness needs).
    - **Entropic** ``drift_direction == "up"`` (high is bad, e.g. cold/hunger):
      ``before - after`` (positive = value decreased toward comfort).
    - **Entropic** ``drift_direction == "down"`` (low is bad): ``after - before``.

    The consumer takes the SIGN of the net progress across the touched drives so
    the cluster reward is ``±1`` — the same scale as the tool-success signal
    non-drive actions get (a graded magnitude would still lose the argmax to a
    flat ``+1``). Direction (orient toward vs away) and the collateral-harm gate
    are preserved; only the entropic starvation is fixed.

    Note: this is a REWARD gradient, not the pain formula — it deliberately
    ignores ``comfort_band``/``pain_scale`` and credits movement toward the set
    point even *inside* the comfort band (where ``drive_pain_for_value`` is 0).
    That is desirable for orient (keep centering when already "centered enough")
    and is why reward and pain can diverge for homeostatic drives too, not only
    for the entropic step function.
    """
    if isinstance(spec, HomeostaticDriveSpec):
        return abs(before - spec.set_point) - abs(after - spec.set_point)
    if isinstance(spec, EntropicDriveSpec):
        return (before - after) if spec.drift_direction == "up" else (after - before)
    return 0.0


def drive_span(spec: DriveSpec, lo: float, hi: float) -> float | None:
    """The largest movement this drive allows -- the denominator for a relief fraction.

    memory-strength Phase 2b-ii (owner decision 2026-09-22): ``1.0`` means "the biggest relief this
    drive can give".

    - **Homeostatic**: the farthest the value can sit from its set point, from the declared range
      (the half-span on every shipped body, whose set points are range midpoints). ``None`` without
      a usable range.
    - **Entropic**: its OWN deprivation-to-satisfaction band -- not the declared range. A body's
      range is deliberately wider than the world's observable span (``minecraft_player`` declares
      health ``[0, 40]`` for 20 max hp so the encoder's neutral lands at the midpoint), and
      inheriting that here made a full satisfaction of a starving drive read 0.25 while ``1.0`` was
      unreachable (found in the Phase 2b-ii review). The band is also exactly the scale
      ``drive_pressure`` uses, so pressure 1.0 -> 0.0 and relief 1.0 describe one event.
    """
    if isinstance(spec, HomeostaticDriveSpec):
        if not (math.isfinite(lo) and math.isfinite(hi)) or hi <= lo:
            return None
        span = max(hi - spec.set_point, spec.set_point - lo)
    else:
        span = abs(spec.deprivation_threshold - spec.satisfaction_threshold)
    return span if span > 0 else None


def relief_fraction_from_progress(spec: DriveSpec, progress: float, lo: float, hi: float) -> float | None:
    """How much of the relief this drive COULD give, a raw signed progress actually gave: ``[0, 1]``.

    Positive part only -- movement away from comfort is harm, which the pain channel carries --
    over ``drive_span``; ``None`` when the drive has no usable denominator. The producers
    (``tool_bridge``) difference before/after themselves and emit the per-drive terms, so the
    executor normalises through HERE: one definition of "the most this drive can give".
    ``drive_comfort_progress`` itself is untouched -- it is a credit signal with its own experiment
    triggers, and this is a record.
    """
    span = drive_span(spec, lo, hi)
    if span is None or not math.isfinite(progress):
        return None
    return max(0.0, min(1.0, progress / span))


def drive_pressure(spec: DriveSpec, value: float, lo: float, hi: float) -> float | None:
    """How hard this drive is pushing right now, in ``[0, 1]`` (memory-strength Phase 2b-ii).

    ``0.0`` inside the comfort band is a MEASUREMENT ("this drive is not pushing"); ``None`` means
    the drive could not be read at all. Unlike ``corrective_need_intensity`` -- which answers a
    different question (which corrective affordance to pick), is raw-unit and returns ``None`` for
    entropic "up" drives and above-set-point deficits -- this covers every drive kind and both
    directions, and is normalised by the drive's own declared range. That function is left exactly
    as it is: it feeds the interoception channel the survival experiments' fingerprints cover.

    - Homeostatic: deviation past the comfort band, over the widest deviation the range allows.
    - Entropic: from the satisfaction threshold toward the deprivation threshold, either direction.
    """
    if not math.isfinite(value):
        return None
    if isinstance(spec, HomeostaticDriveSpec):
        span = drive_span(spec, lo, hi)
        if span is None:
            return None
        deviation = abs(value - spec.set_point) - spec.comfort_band
        headroom = span - spec.comfort_band
        if headroom <= 0:
            return 1.0 if deviation > 0 else 0.0
        return max(0.0, min(1.0, deviation / headroom))
    if isinstance(spec, EntropicDriveSpec):
        span = spec.deprivation_threshold - spec.satisfaction_threshold  # signed by drift direction
        if span == 0:  # a degenerate spec: satisfied is deprived, so only equality reads as comfort
            return 0.0 if value == spec.satisfaction_threshold else 1.0
        return max(0.0, min(1.0, (value - spec.satisfaction_threshold) / span))
    return None


def drift_step(spec: DriveSpec, value: float, dt: float) -> float:
    """Where a drive's own drift moves ``value`` over ``dt`` seconds: the arithmetic of
    ``Embodiment.tick_vital_drift``, which runs on this (grounding GL2a, owner decision G14).

    Homeostatic: toward ``set_point`` at ``drift_rate``, never past it. Entropic: in ``drift_direction``
    at ``drift_rate``, clamped to ``[0, 1]`` ("up" rises; any other direction falls). Pure, so the
    applied drift an evaluation netted can be recomputed and pinned.
    """
    if isinstance(spec, HomeostaticDriveSpec):
        delta = spec.set_point - value
        step = min(abs(delta), spec.drift_rate * dt)
        return value + (step if delta > 0 else -step)
    if isinstance(spec, EntropicDriveSpec):
        if spec.drift_direction == "up":
            return min(1.0, value + spec.drift_rate * dt)
        return max(0.0, value - spec.drift_rate * dt)
    return value


# ---------------------------------------------------------------------------
# The body-consequence record (grounding GL2a, docs/plans/autonomic_layer.md §3.1)
# ---------------------------------------------------------------------------

#: The provenance a body-consequence record may carry (owner decisions G6, G16). ``narrated`` is
#: discounted and ``apparatus`` excluded by the record's later consumers; GL2a produces ``experienced``
#: only (the tool path, G9). There is no default: a record without provenance cannot be built.
OUTCOME_PROVENANCE: frozenset[str] = frozenset({"experienced", "narrated", "imagined", "apparatus"})


def _check_extra(owner: Any, extra: Any) -> None:
    """CC3 path (a): ``extra`` holds JSON values only and never shadows a declared field."""
    import json

    if not isinstance(extra, dict):
        raise ValueError(f"{type(owner).__name__}.extra must be a dict, got {type(extra).__name__}")
    declared = {f for f in owner.__dataclass_fields__ if f != "extra"}
    collisions = set(extra) & declared
    if collisions:
        raise ValueError(f"{type(owner).__name__}.extra keys collide with declared fields: {sorted(collisions)}")
    try:
        json.dumps(extra)
    except (TypeError, ValueError) as e:
        raise ValueError(f"{type(owner).__name__}.extra must hold JSON values only: {e}") from None
    object.__setattr__(owner, "extra", dict(extra))


@dataclass(frozen=True, slots=True)
class CauseRef:
    """Who or what caused a body consequence (grounding GL2a; ``None`` on the record = unknown / world).

    CC3 path (a): defaults on every field plus ``extra`` (JSON values only; ``__post_init__`` rejects a
    key that collides with a declared field). The post-fence resume stage adds ``cause_pid`` (the
    physical event that caused it) as a defaulted field (owner decision G17).
    """

    entity: str = ""  # YAML noun of the causing entity ("fire_pit"); never the sufferer
    affordance: str = ""  # "touch", "warm_self"
    tool: str = ""  # the tool signature, when a tool call caused it
    extra: dict[str, Any] = field(default_factory=dict, hash=False, compare=False)

    def __post_init__(self) -> None:
        _check_extra(self, self.extra)

    def to_dict(self) -> dict[str, Any]:
        return {"entity": self.entity, "affordance": self.affordance, "tool": self.tool, "extra": dict(self.extra)}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CauseRef:
        return cls(
            entity=str(data.get("entity", "")),
            affordance=str(data.get("affordance", "")),
            tool=str(data.get("tool", "")),
            extra=dict(data.get("extra") or {}),
        )


_DriveBlock = tuple[tuple[str, float], ...]
_OUTCOME_CORE = ("nociception", "drive_pain", "relief", "harm", "urgency")
_OUTCOME_BLOCKS = ("pressure_before", "pressure_after", "drive_delta", "deviation_after")


@dataclass(frozen=True, slots=True)
class InteroceptiveOutcome:
    """One body-consequence event: what happened to the body, and was it good (grounding GL2a).

    The insular record of ``docs/plans/autonomic_layer.md`` §3.1, record-only in GL2a: produced on the
    tool path by ``Executor._stamp_invocation`` (owner decisions G9, G14) and written into the loop
    capture's ``EncodingSignals.extra["interoception"]``; no reader acts on it. Built ONLY through
    :func:`interoceptive_outcome`, whose ``cause=`` and ``provenance=`` are required keywords.

    CC3 path (a): defaults on every field plus ``extra`` (JSON values only; ``__post_init__`` rejects a
    colliding key). The defaults exist only to satisfy path (a): ``__post_init__`` REJECTS the
    sentinel ``provenance=""`` (and any kind outside ``OUTCOME_PROVENANCE``), so no record exists
    without provenance. GL2a's record carries NO event id (owner decision G17): the post-fence resume
    stage adds ``pid`` as a defaulted field and from then rejects ``pid=None``. ``invocation_id`` is the
    executor's in-process uuid, a diagnostic only: it is not persisted (``to_dict``), not compared, and
    never a join key.

    Per-drive blocks are sorted ``(drive, value)`` pairs over the drives with a declared range only
    (never imputed); on the tool path, only the invoked affordance's own declared drives (G14), net of
    the drift the body applied during the invocation. ``pressure_*`` is ``drive_pressure`` (unsigned,
    ``[0, 1]``); ``drive_delta`` is the signed progress toward comfort over the drive's span
    (``[-1, 1]``, the physical description, not the valence); ``deviation_after`` is signed
    ``(v - set_point) / span`` for a homeostatic drive and the pressure for an entropic one.

    The core is body-agnostic, each in ``[0, 1]``: ``relief`` / ``harm`` are the largest DROP / RISE in
    drive pressure (first-order alliesthesia: the same physical change scores by need), ``harm``
    excluding the tissue-damage drives, whose loss is ``nociception`` instead (health counts once, G11);
    ``nociception`` is the action's nociceptive pain (caused, else felt: ``extra["nociception_basis"]``)
    or this event's injury, whichever is larger, and never anticipatory; ``drive_pain`` is the largest
    drive-pain level after the action (``drive_pain_for_value``: a homeostatic breach or an entropic
    deprivation, the tissue-damage drives excluded); ``urgency`` (v1, G12) is the largest pressure after.

    Scope, stated (GL2a): ``experienced`` means "minted by the agent's own executor", not "free of other
    writers". A narrator write landing on the body during ``tool.run`` shows in the after-read, unnetted,
    until the post-fence lock and write epoch land (``autonomic_layer.md`` §3.1.4).
    The valence ``relief - harm - nociception`` is the projection's (GL4), not a field.
    """

    invocation_id: str = field(default="", compare=False)
    agent_id: str = ""
    body_path: str = ""  # whose body: drive names collide across bodies
    provenance: str = ""  # REQUIRED: one of OUTCOME_PROVENANCE
    sufferer: str = ""  # entity path whose body changed
    cause: CauseRef | None = None
    pressure_before: _DriveBlock = ()
    pressure_after: _DriveBlock = ()
    drive_delta: _DriveBlock = ()
    deviation_after: _DriveBlock = ()
    caused: tuple[tuple[str, bool], ...] = ()  # True = the action's declared effect (every tool-path entry)
    satiated: tuple[str, ...] = ()  # declared drives whose breach latch cleared in this invocation
    nociception: float = 0.0
    drive_pain: float = 0.0
    relief: float = 0.0
    harm: float = 0.0
    urgency: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict, hash=False, compare=False)

    def __post_init__(self) -> None:
        if self.provenance not in OUTCOME_PROVENANCE:
            raise ValueError(
                f"InteroceptiveOutcome.provenance must be one of {sorted(OUTCOME_PROVENANCE)}, got {self.provenance!r}"
            )
        for name in _OUTCOME_CORE:
            value = getattr(self, name)
            if not (isinstance(value, (int, float)) and math.isfinite(value) and 0.0 <= value <= 1.0):
                raise ValueError(f"InteroceptiveOutcome.{name} must be a number in [0, 1], got {value!r}")
        _check_extra(self, self.extra)

    def to_dict(self) -> dict[str, Any]:
        """The persisted form (JSON values only; ``invocation_id`` stays in-process)."""
        out: dict[str, Any] = {
            "agent_id": self.agent_id,
            "body_path": self.body_path,
            "provenance": self.provenance,
            "sufferer": self.sufferer,
            "cause": self.cause.to_dict() if self.cause is not None else None,
        }
        for name in _OUTCOME_BLOCKS:
            out[name] = [[k, v] for k, v in getattr(self, name)]
        out["caused"] = [[k, v] for k, v in self.caused]
        out["satiated"] = list(self.satiated)
        for name in _OUTCOME_CORE:
            out[name] = getattr(self, name)
        out["extra"] = dict(self.extra)
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> InteroceptiveOutcome:
        cause = data.get("cause")
        kwargs: dict[str, Any] = {
            name: tuple((str(k), float(v)) for k, v in data.get(name) or ()) for name in _OUTCOME_BLOCKS
        }
        return cls(
            agent_id=str(data.get("agent_id", "")),
            body_path=str(data.get("body_path", "")),
            provenance=str(data.get("provenance", "")),
            sufferer=str(data.get("sufferer", "")),
            cause=CauseRef.from_dict(cause) if cause is not None else None,
            caused=tuple((str(k), bool(v)) for k, v in data.get("caused") or ()),
            satiated=tuple(str(d) for d in data.get("satiated") or ()),
            extra=dict(data.get("extra") or {}),
            **kwargs,
            **{name: float(data.get(name, 0.0)) for name in _OUTCOME_CORE},
        )


def affordance_declared_drives(
    self_effect: Mapping[str, float] | None, drive_specs: Mapping[str, Any], live_owned: Iterable[str]
) -> tuple[str, ...]:
    """The drives an affordance's own ``self_effect`` declares it moves: the record's drive set (G14).

    The keys ``tool_bridge`` applies and scores (``_self_effect`` after the live-owned filter) that
    carry a drive spec on the body, qualified modulator sub-sensors (``arms.thermal``) included: the
    record reads them through the one resolver (#1125), where the credit reads are still blind (#1161).
    """
    live = set(live_owned)
    return tuple(sorted(name for name in (self_effect or {}) if name in drive_specs and name not in live))


def interoceptive_outcome(
    specs: Mapping[str, DriveSpec],
    ranges: Mapping[str, tuple[float, float]],
    before: Mapping[str, float],
    after: Mapping[str, float],
    drift: Mapping[str, float],
    nociception: float | None,
    satiated: Iterable[str],
    *,
    cause: CauseRef | None,
    provenance: str,
    agent_id: str = "",
    body_path: str = "",
    sufferer: str = "",
    invocation_id: str = "",
    drift_dt_s: float = 0.0,
) -> InteroceptiveOutcome:
    """The one way to build an :class:`InteroceptiveOutcome` (grounding GL2a, §3.1).

    ``specs`` are the drives the record covers (on the tool path, the invoked affordance's own declared
    drives, G14), ``before`` / ``after`` their values around the action, and ``drift`` the drift the
    body APPLIED to each during it (clamped, on the post-delta value; ``Embodiment.outcome_window``):
    the record reports values net of it and keeps the observed change in ``extra``. ``nociception`` is
    the action's nociceptive pain (``ToolPainBridge.pop_invocation_pain``: caused, else felt), ``None``
    when unmeasured. ``satiated`` are the drives whose breach latch the body cleared during it.

    ``cause=`` and ``provenance=`` are REQUIRED keywords, so forgetting either is a ``TypeError``.
    """
    from maxim.proprioception.pain import TISSUE_DAMAGE_DRIVES

    blocks: dict[str, dict[str, float]] = {name: {} for name in _OUTCOME_BLOCKS}
    drift_by_drive: dict[str, float] = {}
    observed: dict[str, float] = {}
    relief = harm = urgency = drive_pain = injury = 0.0
    for name in sorted(specs):
        spec = specs[name]
        if name not in before or name not in after:
            continue
        lo, hi = ranges.get(name, (float("nan"), float("nan")))
        span = drive_span(spec, lo, hi)
        v0 = float(before[name])
        applied = float(drift.get(name, 0.0))
        v1 = float(after[name]) - applied
        p0, p1 = drive_pressure(spec, v0, lo, hi), drive_pressure(spec, v1, lo, hi)
        if span is None or p0 is None or p1 is None:
            continue  # no declared range: the drive is absent, never imputed
        delta = max(-1.0, min(1.0, drive_comfort_progress(spec, v0, v1) / span))
        if isinstance(spec, HomeostaticDriveSpec):
            deviation = max(-1.0, min(1.0, (v1 - spec.set_point) / span))
        else:
            deviation = p1
        blocks["pressure_before"][name] = p0
        blocks["pressure_after"][name] = p1
        blocks["drive_delta"][name] = delta
        blocks["deviation_after"][name] = deviation
        drift_by_drive[name] = applied
        observed[name] = float(after[name]) - v0
        relief = max(relief, p0 - p1)
        urgency = max(urgency, p1)
        if f"drive:{name}" in TISSUE_DAMAGE_DRIVES:
            injury = max(injury, -delta)  # this event's normalised loss, not the deficit's level (G11)
        else:
            harm = max(harm, p1 - p0)
            drive_pain = max(drive_pain, drive_pain_for_value(spec, v1))
    covered = tuple(sorted(blocks["drive_delta"]))
    extra: dict[str, Any] = {
        "drift_dt_s": float(drift_dt_s),
        "drift": drift_by_drive,
        "observed_change": observed,
        # Caused and felt pain are one value until ToolPainBridge splits them (after the fence).
        "nociception_basis": "caused_or_felt",
    }
    return InteroceptiveOutcome(
        invocation_id=invocation_id,
        agent_id=agent_id,
        body_path=body_path,
        provenance=provenance,
        sufferer=sufferer,
        cause=cause,
        **{name: tuple(sorted(values.items())) for name, values in blocks.items()},
        caused=tuple((name, True) for name in covered),
        satiated=tuple(sorted(set(satiated) & set(specs))),
        nociception=max(float(nociception or 0.0), injury),
        drive_pain=drive_pain,
        relief=relief,
        harm=harm,
        urgency=urgency,
        extra=extra,
    )


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
        "drive_breach_severity",
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
        # Drive-pain breach latch: sensor_name → latched breach severity (in
        # sensor units). Owned by the ENTITY, not the Embodiment wrapper, so a
        # body cannot "forget" it is injured because a different observer
        # looked at it (simulation tools construct ephemeral per-invocation
        # Embodiments) or because it was reparented by entity acquisition.
        # Written only by Embodiment.evaluate_failures; gates the PainBus
        # channel so re-injury re-publishes but a standing breach does not
        # re-fire per tick. Session-runtime state — deliberately NOT part of
        # to_dict/from_dict, mirroring FailureMode.active's non-persistence.
        # Contrast the two latch polarities: FailureMode.persistent means
        # "keep firing until recovery"; this one means "stay quiet unless the
        # injury deepens". Declaring `persistent: true` on a drive does
        # nothing — drives are not FailureModes.
        self.drive_breach_severity: dict[str, float] = {}

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

    def component_integrities(self) -> dict[str, float]:
        """Each component modulator's integrity, by modulator name, derived from its sub-sensors now.

        The one producer of the ``<mod>.integrity`` readings that failure triggers, telemetry and the
        visible-sensor view use (#1124). Never stored on ``vital_metrics``: a stored copy shadowed the real
        sub-sensors, went stale, and was persisted. A modulator with no sub-sensor values (capability-only)
        has no integrity reading.
        """
        return {
            mod_name: mod.compute_integrity()
            for mod_name, mod in self.modulators.items()
            if hasattr(mod, "compute_integrity") and getattr(mod, "vital_metrics", None)
        }

    def derive_health(self) -> float | None:
        """Derive entity health from modulator component integrities.

        Returns a weighted mean of modulator integrities using
        ``metadata["health_weights"]`` if present.  Returns None if
        no modulators have component sensors (backward compat — entity
        uses direct ``vital_metrics["health"]`` instead).

        Called by ``Body.evaluate_failures()`` to update
        ``vital_metrics["health"]`` when ``metadata.get("health") == "derived"``.
        """
        integrities = self.component_integrities()
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

        Captures the entity tree: sensors (metadata only, not live backends),
        modulators (sub-sensor specs and values, integrity function, damage
        affinities, affordance descriptions), children, vital metrics, failure
        modes, drive specs and metadata. ``Entity.from_dict()`` restores those.

        NOT captured, so NOT restored (#1159): an affordance's ``params``,
        ``requires``, ``self_effect``, ``target_effect`` and ``always_active``;
        latent affordances; a failure mode's runtime ``active``/``last_fired``.
        A dotted ``vital_metrics`` key is not saved either: component state lives
        on its modulator (#1124), so one is dropped here with a WARNING.
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
            # Component state and its spec (format 1.1, #1124): without them a reloaded modulator was empty,
            # so its integrity and the entity's derived health stayed at their saved values forever.
            values = getattr(m, "vital_metrics", None)
            if values:
                d["values"] = dict(values)
            if hasattr(m, "integrity_fn"):
                d["integrity"] = m.integrity_fn
            affinities = getattr(m, "damage_affinities", None)
            if affinities:
                d["damage_affinities"] = {k: dict(v) for k, v in affinities.items()}
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
            dotted = sorted(k for k in self.vital_metrics if "." in k)
            if dotted:
                # Owner decision 2026-10-07 (#1124): the writer enforces what the loader assumes, so a save and a
                # load agree. ``vital_metrics`` is a public dict; a dotted key there is never a real sensor.
                logger.warning(
                    "Entity %r: not saving dotted vital_metrics keys %s; component state lives on its modulator "
                    "(#1124)",
                    self.name,
                    dotted,
                )
            result["vital_metrics"] = {k: v for k, v in self.vital_metrics.items() if "." not in k}
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
            # A dotted key here is a stored ``<mod>.integrity`` (written before format 1.1) or an orphan
            # sub-sensor value (pre-#874 ``set_entity_sensor``); either shadowed the modulator's real value.
            # Dropped (owner decision 2026-10-07, #1124): integrity is re-derived, an orphan is stale.
            metrics = dict(data["vital_metrics"])
            entity.vital_metrics = {k: v for k, v in metrics.items() if "." not in k}

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
                    modulator = SpecModulator(
                        _name=mdata.get("name", mname),
                        _entity_name=data["name"],
                        _affordances=affs,
                        _sensors=mod_sensors,
                        _integrity_fn=mdata.get("integrity", "weighted_mean"),
                        _damage_affinities=mdata.get("damage_affinities") or {},
                        _abstract=mod_abstract,
                        _entity_ref=entity,
                    )
                    # Format 1.1 carries the sub-sensor values; an older file starts them from the spec,
                    # as a freshly parsed body does.
                    saved_values = mdata.get("values")
                    if isinstance(saved_values, dict):
                        modulator.vital_metrics.update({k: float(v) for k, v in saved_values.items()})
                    else:
                        modulator.vital_metrics.update(modulator.initial_values())
                    entity.modulators[mname] = modulator
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

        _migrate_dotted_vital_metrics(entity, data)

        # Reconstruct children recursively
        for child_data in data.get("children", []):
            cls.from_dict(child_data, parent=entity)
        return entity

    def save(self, path: str) -> None:
        """Save entity tree to a JSON file."""
        from maxim.utils.atomic_io import atomic_write_json
        from maxim.utils.format_version import with_format_version

        atomic_write_json(path, with_format_version(self.to_dict(), version=ENTITY_FORMAT_VERSION))

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


def _migrate_dotted_vital_metrics(entity: Entity, data: dict[str, Any]) -> None:
    """Carry a saved file's dotted ``vital_metrics`` keys onto the modulators, or drop them, with a WARNING.

    A file written before format 1.1 holds a component's last integrity as ``<mod>.integrity`` and no sub-sensor
    values. Dropping it would heal every saved body, so when that modulator has no saved ``values`` and the saved
    integrity differs from its start, its weighted (``weight`` > 0) non-drive sub-sensors are set to the saved
    integrity, so integrity and derived health come back as saved (owner decision 2026-10-07, #1124). A drive
    sub-sensor (``arms.thermal``) is left at its start: it is not damage, and setting it to an integrity would be
    a burn. A 1.0 file never saved the integrity function either, so the modulator loads as ``weighted_mean``: a
    ``min``/``max`` component (the dragon's torso) is then rewritten to reproduce its saved integrity even when
    unhurt, and later damage on it is averaged. A result that does not reproduce the saved value (a weighted
    drive sub-sensor) is reported; a value that is not a finite number is dropped and reported. Every other dotted key (a pre-#874
    orphan, or an integrity a 1.1 file's ``values`` already supersede) is dropped.
    """
    saved = data.get("vital_metrics") or {}
    dotted = sorted(k for k in saved if "." in k)
    if not dotted:
        return
    migrated: list[str] = []
    inexact: list[str] = []
    unusable: list[str] = []
    for key in dotted:
        mod_name, _, sub = key.partition(".")
        mod = entity.modulators.get(mod_name)
        mod_data = (data.get("modulators") or {}).get(mod_name) or {}
        values = getattr(mod, "vital_metrics", None)
        compute_integrity = getattr(mod, "compute_integrity", None)
        if sub != "integrity" or not values or compute_integrity is None or isinstance(mod_data.get("values"), dict):
            continue
        try:
            target = float(saved[key])
        except (TypeError, ValueError):
            target = math.nan
        if not math.isfinite(target):  # not a number, NaN or inf: would poison every sub-sensor it touched
            unusable.append(f"{key}={saved[key]!r}")
            continue
        if abs(compute_integrity() - target) <= 1e-9:
            migrated.append(key)  # undamaged: the spec start already is the saved state
            continue
        specs = getattr(mod, "sensors", {}) or {}
        for ms_name in values:
            spec = specs.get(ms_name)
            weight = spec.get("weight", 1.0) if isinstance(spec, dict) else 1.0
            if weight > 0 and f"{mod_name}.{ms_name}" not in entity.drive_specs:
                values[ms_name] = target
        migrated.append(key)
        if abs(compute_integrity() - target) > 1e-6:
            inexact.append(f"{key}={target:.3f}->{compute_integrity():.3f}")
    dropped = [k for k in dotted if k not in migrated]
    logger.warning(
        "Entity %r (format %s): component state lives on its modulator (#1124); migrated %s into their "
        "modulators' sub-sensors%s, dropped %s%s",
        data.get("name"),
        data.get("_format_version", "0.x"),
        migrated or "nothing",
        f" (not reproduced exactly: {inexact})" if inexact else "",
        dropped or "nothing",
        f" (not a finite number: {unusable})" if unusable else "",
    )


# ---------------------------------------------------------------------------
# Sensor resolution (one rule for every reader and writer that resolves a name)
# ---------------------------------------------------------------------------


def _sensor_location(body: Entity, sensor_name: str) -> tuple[dict[str, float], str] | None:
    """Where a (possibly qualified) sensor's value lives on ``body``: ``(metrics, key)``, or ``None``.

    ``"arms.thermal"`` is the ``thermal`` sub-sensor of the ``arms`` modulator, so its value is
    ``body.modulators["arms"].vital_metrics["thermal"]``; a bare name is an entity-level sensor in
    ``body.vital_metrics``. ``None`` when the body has no such sensor, or its value is ``None``.
    Reads no range, so a malformed declared range can never cost a value read (#1125 review).
    """
    if "." in sensor_name:
        mod_name, sub_name = sensor_name.split(".", 1)
        mod = body.modulators.get(mod_name)
        metrics = getattr(mod, "vital_metrics", None) if mod is not None else None
        if metrics is None or metrics.get(sub_name) is None:
            return None
        return metrics, sub_name
    if body.vital_metrics.get(sensor_name) is None:
        return None
    return body.vital_metrics, sensor_name


def _resolve_sensor_slot(body: Entity, sensor_name: str) -> tuple[dict[str, float], str, float, float] | None:
    """Where a (possibly qualified) sensor lives on ``body``, and its declared range.

    The embodiment's ONE resolution rule, for writes and reads alike. Tool and affordance writes use
    this (#874: ``set_entity_sensor`` in both modes, ``self_effect``/``target_effect``); drive-value
    READS use its location half through ``_read_sensor_value`` (#1125: the executor's
    ``drive_pressure`` record, ``Embodiment.body_state_summary``). Other writers (DM cascade,
    cerebellum predictions, vital drift) do not go through it (one shared resolver: #1156), and neither do the credit reads in ``tool_bridge``
    (#1161). The location is ``_sensor_location``; this adds the range on top. Returns
    ``(metrics, key, lo, hi)``, the range being the sensor's schema range or ``[0, 1]``, or
    ``None`` when the body has no such sensor (a caller must not write it: a qualified name written
    to the root is an orphan key, which ``evaluate_failures`` overrides with the real sub-sensor
    and reports, and which ``Entity.to_dict`` does not save, #1124).
    """
    location = _sensor_location(body, sensor_name)
    if location is None:
        return None
    metrics, key = location
    lo, hi = 0.0, 1.0
    if "." in sensor_name:
        mod = body.modulators[sensor_name.split(".", 1)[0]]
        sub_spec = getattr(mod, "_sensors", {}).get(key, {})
        if isinstance(sub_spec, dict) and "range" in sub_spec:
            lo, hi = sub_spec["range"]
        return metrics, key, lo, hi
    sensor = body.sensors.get(sensor_name)
    if sensor is not None:
        rng = sensor.reading_schema.get("range")
        if rng and len(rng) == 2:
            lo, hi = rng
    return metrics, key, lo, hi


def _read_sensor_value(body: Entity, sensor_name: str) -> float | None:
    """The current value of a (possibly qualified) sensor, by the one resolution rule.

    The read half of ``_resolve_sensor_slot`` (#1125): a drive declared as ``arms.thermal`` on the
    root lives on the ``arms`` modulator, so a reader that looks it up as
    ``root.vital_metrics["arms.thermal"]`` never finds it. Uses only the LOCATION, never the range.
    ``None`` when the body has no such sensor or its value is not a number.
    """
    location = _sensor_location(body, sensor_name)
    if location is None:
        return None
    metrics, key = location
    try:
        return float(metrics[key])
    except (TypeError, ValueError):
        return None  # a non-numeric value is "not readable", like a missing one


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
    # Runtime only (never serialized): trigger fields already reported as having no reading. Per instance, so
    # each entity's failure modes report their own missing fields.
    _warned_missing: set[str] = field(default_factory=set, init=False, repr=False, compare=False)

    def _reading(self, sensor_readings: dict[str, float], name: str, source: str) -> float | None:
        """The reading a trigger tests, or None, reported once per field, when nothing produces it.

        A missing reading used to evaluate as 0.0, so a ``<`` trigger fired on a key nothing wrote: a reloaded
        humanoid with no ``head.integrity`` was concussed unhurt (#1124). Unknown is not a breach, as an
        unreadable drive sensor already is in ``evaluate_failures``.

        Behaviour tier: invariant (fail-closed: unknown is not a breach). Owner decision on #1124, 2026-10-07.
        A trigger naming a field nothing can produce is also warned about at parse (``spec._parse_entity``).
        """
        val = sensor_readings.get(name)
        if val is None and name not in self._warned_missing:
            self._warned_missing.add(name)
            logger.warning(
                "Failure mode %r%s: trigger field %r has no reading; it does not fire (#1124)",
                self.name,
                f" on {source}" if source else "",
                name,
            )
        return val

    def evaluate(self, sensor_readings: dict[str, float], *, source: str = "") -> bool:
        """Check if this failure mode should fire given current readings.

        A trigger whose field has no reading does not fire, and a recovery condition with no reading does
        not clear; each such field is reported once. *source* (the entity path) only labels that report.
        """
        if self.persistent and self.active:
            # check recovery
            if self.recovery_condition is not None:
                val = self._reading(sensor_readings, self.recovery_condition.field, source)
                if val is not None and self.recovery_condition.evaluate(val):
                    self.active = False
                    return False
            return True  # still active, no recovery yet

        results = []
        for trigger in self.triggers:
            val = self._reading(sensor_readings, trigger.field, source)
            results.append(val is not None and trigger.evaluate(val))

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
