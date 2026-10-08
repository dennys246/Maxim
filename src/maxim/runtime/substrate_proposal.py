"""Substrate-primary action generation: ``propose_via_substrate`` and the sensor reads it encodes.

Moved verbatim out of ``runtime/agent_loop.py`` (1.3.2 decomposition, slice 3; owner decision S1,
2026-10-07): the whole substrate-proposer family, from the drive-name corrective-need table through
``_attach_live_situation``. It is a LEAF: it imports nothing from the agent loop, so the loop
(``agent_loop``, ``loop_substrate``), the executor (``_read_drive_ranges``), the setup
(``NO_SITUATION_CUE``) and the survival/experiment harnesses under ``scripts/`` (which call
``propose_via_substrate`` directly, bypassing the loop) all import it from here. There are no
re-exports from ``agent_loop``.

A test that replaces the proposer patches ``substrate_proposal.propose_via_substrate``:
``loop_substrate.substrate_tick`` reads it through this module at call time. A test that replaces a
reader the proposer or ``_attach_live_situation`` calls (``_encode_current_clusters``,
``_read_world_ranges``, ...) patches it here too, since these functions read this module's globals.
The module LOGS on the ``maxim.runtime.agent_loop`` logger (``logger`` below is that same object), so
its records keep the agent loop's logger name.
"""

from __future__ import annotations

import dataclasses
import logging
import math
import os
import time
from typing import TYPE_CHECKING, Any

from maxim.agents.llm_worker import LLMProposal
from maxim.embodiment.sensory_streams import AUDIO_TAG, INTEROCEPTION_TAG, WORLD_TAG, ModalityChannel
from maxim.utils.logging import log_swallowed_exception

if TYPE_CHECKING:
    from collections.abc import Callable

# The SAME logger object as ``agent_loop.logger`` (logging returns one logger per name), so records keep
# the ``maxim.runtime.agent_loop`` name.
logger = logging.getLogger("maxim.runtime.agent_loop")


# ─────────────────────────────────────────────────────────────────────────────
# Substrate-primary action generation (Phase -1 of grounded_language_acquisition.md)
# ─────────────────────────────────────────────────────────────────────────────


# Drive-name substring -> the corrective NEED emitted when that drive is in DEFICIT
# (first match wins). This generalizes the former hardcoded "cold" name-sniff (the fix
# `_read_drive_states`'s own NOTE anticipated) WITHOUT adding a field to the CC3-frozen
# DriveSpec. The emitted need name is what `NAc._DRIVE_TOOL_AFFINITIES` keys on, so a
# deficit lands on a corrective affordance instead of the polarity-inverted raw sensor
# value R2 found (the intrinsic survival loop, break 1 — docs/experiments/r2_drive_premise_check.md).
# Brittleness, stated: this is a drive-NAME substring map (first match wins), the same
# name-convention the code's old "cold" sniff used and that CC3 forbids replacing with a
# DriveSpec field (both specs SHAPE-FROZEN at 1.0). It mis-fires on other bodies' drive
# names (a health drive named `hp`/`vitality` gets nothing; `food_temperature` -> `cold`
# by order). Scoped to the 1.3 minecraft_player survival body (break 1); a body with
# differently-named drives must extend this table. Entropic "up" drives derive no need
# (corrective_need_intensity returns None) — deliberately out of scope for break 1.
_DRIVE_CORRECTIVE_NEEDS: tuple[tuple[str, str], ...] = (
    ("temp", "cold"),
    ("thermal", "cold"),
    ("food", "hunger"),  # entropic drain: low food -> "hunger" -> eat (existing affinity)
    ("health", "threat"),  # homeostatic deficit: low health -> "threat" (flight/freeze/recover)
)


def _corrective_need_for(ds_name: str) -> str | None:
    """The corrective-need name a drive emits on deficit, or None (substring, first match)."""
    low = ds_name.lower()
    for needle, need in _DRIVE_CORRECTIVE_NEEDS:
        if needle in low:
            return need
    return None


def _read_drive_states(executor: Any) -> dict[str, float]:
    """Extract current drive values from the executor's embodiment.

    Returns ``{drive_name: value in [0, 1]}`` for every drive declared on
    every entity walked from the embodiment root. Empty dict when no
    embodiment is wired.

    Used by substrate-primary AUT mode to feed ``NAc.recommend_action``
    without going through the LLM. Reads the current values directly from
    ``Entity.vital_metrics`` (entity-level drives) and modulator
    ``vital_metrics`` (sub-sensor drives like ``arms.thermal``).
    """
    embodiment = getattr(executor, "embodiment", None)
    if embodiment is None or getattr(embodiment, "root", None) is None:
        return {}

    drives: dict[str, float] = {}
    # Derived corrective NEEDS (see below), keyed by need name, accumulated as the
    # max breach across all drives that map to that need, emitted once at the end.
    derived_needs: dict[str, float] = {}
    for ent in embodiment.root.walk():
        specs = getattr(ent, "drive_specs", {})
        for ds_name, spec in specs.items():
            if "." in ds_name:
                mod_name, sensor_name = ds_name.split(".", 1)
                mod = ent.modulators.get(mod_name)
                if mod is None or not hasattr(mod, "vital_metrics"):
                    continue
                value = mod.vital_metrics.get(sensor_name)
            else:
                value = ent.vital_metrics.get(ds_name)
            if value is None:
                continue
            try:
                fval = float(value)
            except (TypeError, ValueError):
                continue
            drives[ds_name] = fval

            # Derive a positive corrective NEED from this drive's DEFICIT. The
            # drive-affinity heuristic in NAc.recommend_action only fires on positive
            # [0,1] need intensities, so a raw sensor value (largest when SATIATED) is
            # invisible-or-inverted for action selection — R2's finding: behaviour
            # "moves backwards". A drive that maps to a corrective need
            # (_DRIVE_CORRECTIVE_NEEDS: food->hunger, health->threat, thermal->cold)
            # emits that need at its deficit intensity, which the affinity table lands
            # on the corrective affordance (eat / defensive repertoire / warm-seek).
            # LLM-free path only (propose_via_substrate); LLM-AUT reads body_state
            # directly, so Exp 37/38 are unaffected.
            need = _corrective_need_for(ds_name)
            if need is not None:
                # Intensity math lives in the embodiment layer (isinstance-dispatched
                # beside drive_pain_for_value), not re-derived here.
                from maxim.embodiment.sem import corrective_need_intensity

                intensity = corrective_need_intensity(spec, fval)
                if intensity is not None and intensity > 0.0:
                    derived_needs[need] = max(derived_needs.get(need, 0.0), intensity)

    for need, intensity in derived_needs.items():
        # setdefault: never clobber a real drive literally named e.g. "cold"/"hunger".
        # NB: the derived need (normalized [0,1]) is ALSO encoded into the interoception
        # ModalityChannel below, not only the action prior — intended, and it takes the
        # legacy [0,1] range map (see _read_drive_ranges) exactly like the "cold" need.
        drives.setdefault(need, intensity)
    return drives


def _read_drive_ranges(executor: Any) -> "dict[str, tuple[float, float]]":
    """Per-sensor ``(lo, hi)`` range for the drive sensors ``_read_drive_states``
    reads, so ``SensorEncoder.encode_sensors`` normalizes SIGNED sensors
    (azimuth / thermal on ``[-1, 1]``) MONOTONICALLY instead of folding (P1 —
    the range-blind map aliases center with hard-left and collides opposite-sign
    values near center, so a left sound and a right sound could share one EC
    cluster and the orient policy couldn't condition on direction).

    Mirrors ``_read_drive_states``' walk (they iterate the same ``drive_specs``);
    the ``test_read_drive_ranges_covers_every_signed_drive`` guard pins that they
    agree so a future signed drive can't silently re-fold. A drive sensor with no
    declared range is omitted → the encoder falls back to the legacy ``[0, 1]``-ish
    map, which is correct for ``[0, 1]`` drives (hunger/thirst/energy) and the
    derived ``"cold"`` need (also ``[0, 1]``). Only signed sensors need a range.

    UNITS INVARIANT: ``reading_schema["range"]`` MUST be in the same units as the
    values ``_read_drive_states`` reads from ``vital_metrics`` (which ``spec.py``
    initializes from the declared range, so this holds for every YAML drive). A
    drive that declared a raw-unit range (e.g. ``[0, 360]``) but wrote normalized
    values would map every value near 0 — worse than the fold. Do not mix units.
    """
    embodiment = getattr(executor, "embodiment", None)
    if embodiment is None or getattr(embodiment, "root", None) is None:
        return {}
    ranges: dict[str, tuple[float, float]] = {}
    for ent in embodiment.root.walk():
        for ds_name in getattr(ent, "drive_specs", {}):
            rng = None
            if "." in ds_name:
                mod_name, sub_name = ds_name.split(".", 1)
                mod = ent.modulators.get(mod_name)
                if mod is not None and hasattr(mod, "_sensors"):
                    sub = mod._sensors.get(sub_name, {})
                    if isinstance(sub, dict):
                        rng = sub.get("range")
            else:
                sensor = ent.sensors.get(ds_name)
                if sensor is not None:
                    rng = sensor.reading_schema.get("range")
            # Per-sensor guard: a malformed range (non-iterable scalar, wrong
            # length, non-numeric bounds) must NOT bubble — this function is
            # evaluated as an argument inside the encode_sensors try/except, so a
            # raise here would silently disable ALL substrate encoding for the
            # agent, every tick. Skip just the bad sensor → it falls back to the
            # legacy [0,1] map instead.
            try:
                if rng is not None and len(rng) == 2:
                    ranges[ds_name] = (float(rng[0]), float(rng[1]))
            except (TypeError, ValueError):
                logger.debug("drive %r has a malformed range %r; skipping (legacy map)", ds_name, rng)
    return ranges


# Exteroceptive world-set sensors the substrate encodes for PERCEPTION (not as
# drives/needs). ``azimuth`` = head-relative sound direction (base_humanoid's
# capability-driven orient sensor). LEGACY membership set: since 1.1.4 PR 2 a
# sensor can DECLARE its channel (`modality: audio` in the body YAML →
# `_read_declared_modality_states`), which is what this tuple's original
# comment asked for ("a future exteroceptive sensor is one entry, not a code
# change at the read site" — it is now zero code changes). The tuple stays for
# every existing body, which declares nothing; do not grow it — declare.
_EXTEROCEPTIVE_ROOT_SENSORS: tuple[str, ...] = ("azimuth",)


def _read_declared_modality_states(executor: Any, modality: str) -> dict[str, float]:
    """Entity-level sensors DECLARING ``modality: <tag>`` in their body YAML.

    The declaration-driven half of channel membership (1.1.4 PR 2): walks
    every entity from the embodiment root (same walk as ``_read_drive_states``
    — sensor names are flat-keyed across entities, last writer wins, matching
    the drive-read convention) and returns ``{name: value}`` for sensors whose
    ``reading_schema["modality"]`` equals ``modality``. Values come from
    ``Entity.vital_metrics`` exactly like every other read. Empty dict = the
    body declares nothing for this channel (every pre-PR-2 body).
    """
    embodiment = getattr(executor, "embodiment", None)
    root = getattr(embodiment, "root", None)
    if root is None:
        return {}
    walk = getattr(root, "walk", None)
    entities = walk() if callable(walk) else (root,)
    out: dict[str, float] = {}
    for ent in entities:
        sensors = getattr(ent, "sensors", {}) or {}
        vm = getattr(ent, "vital_metrics", {}) or {}
        for name, sensor in sensors.items():
            schema = getattr(sensor, "reading_schema", {}) or {}
            if schema.get("modality") != modality:
                continue
            value = vm.get(name)
            if value is None:
                continue
            try:
                out[name] = float(value)
            except (TypeError, ValueError):
                continue
    return out


def _read_declared_modality_ranges(executor: Any, modality: str) -> "dict[str, tuple[float, float]]":
    """Declared ``(lo, hi)`` for the sensors ``_read_declared_modality_states``
    reads. LOCKSTEP INVARIANT (same class as ``_read_drive_ranges``): the two
    walks must emit the same sensor set; a malformed or wrong-arity range is
    skipped per-sensor (that sensor re-folds through the legacy map — never
    raised: a raise here would silently disable ALL substrate encoding; only
    the type-error shape logs, at debug, matching the legacy path)."""
    embodiment = getattr(executor, "embodiment", None)
    root = getattr(embodiment, "root", None)
    if root is None:
        return {}
    # Duck-typed like every other reader: a fake/minimal root without walk()
    # is treated as a single entity (the place-code wiring tests' fakes).
    walk = getattr(root, "walk", None)
    entities = walk() if callable(walk) else (root,)
    ranges: dict[str, tuple[float, float]] = {}
    for ent in entities:
        sensors = getattr(ent, "sensors", {}) or {}
        for name, sensor in sensors.items():
            schema = getattr(sensor, "reading_schema", {}) or {}
            if schema.get("modality") != modality:
                continue
            rng = schema.get("range")
            try:
                if rng is not None and len(rng) == 2:
                    ranges[name] = (float(rng[0]), float(rng[1]))
            except (TypeError, ValueError):
                logger.debug("declared %s sensor %r has a malformed range %r; skipping", modality, name, rng)
    return ranges


def _read_world_states(executor: Any) -> dict[str, float]:
    """Value source for the ``"world"`` ModalityChannel — purely
    declaration-driven (``modality: world`` on the sensor), no hardcoded
    membership, no drive coupling. A world sensor that ALSO carries a drive
    appears in both encodes, the same documented pattern as a drive-bearing
    azimuth. The channel is A4-GAINED and frozen-centroid (plan D3/D6)."""
    return _read_declared_modality_states(executor, WORLD_TAG)


def _read_world_ranges(executor: Any) -> "dict[str, tuple[float, float]]":
    return _read_declared_modality_ranges(executor, WORLD_TAG)


# Place-code opt-in (modality_resolution_and_alignment.md; Exp 46 validated).
# Default OFF: turning it on changes EC cluster identity for the audio channel,
# which is a re-validation trigger for Exp 48 (and Exp 46's own numbers). Same
# default-OFF-pending-ablation shape as MAXIM_ENABLE_BODY_STATE_PROMPT.
_PLACE_CODE_ENV = "MAXIM_PLACE_CODE_EXTEROCEPTION"
_PLACE_CODE_PREFIX = "azdir"


def place_code_exteroception_enabled() -> bool:
    """True when the exteroceptive channel should emit a population code.

    Read per call (not cached): the autouse conftest scrub flips it between
    tests, and a cached read would leak one test's arm into the next.
    """
    from maxim.prompts.cluster_bias_annotation import annotation_disabled_via_env

    return annotation_disabled_via_env(os.environ.get(_PLACE_CODE_ENV))


def _read_exteroceptive_states(executor: Any) -> dict[str, float]:
    """Read world-set EXTEROCEPTIVE root sensors (``azimuth``) — the value
    source for the ``"audio"`` ModalityChannel, encoded in its OWN
    ``encode_sensors(modality="audio")`` call so an agent can condition its
    action on WHERE a stimulus is, even when it carries no drive/need about it.

    Distinct from ``_read_drive_states``: those are interoceptive needs that
    also drive the affinity heuristic; these are pure perception and NEVER
    enter ``current_drives`` — nor the interoception encode (the pre-seam
    ``{**drives, **extero}`` merge diluted direction among the drives and
    collapsed left/right onto one cluster; see
    docs/plans/archive/exteroception_interoception_seam.md). Load-bearing for
    ``bodies/infant_operant`` (cradle_mother operant experiment), whose azimuth
    sensor has ``drive: null``. A body whose azimuth ALSO carries a drive gets
    the value in BOTH encodes. The plan's intent is two representations of two
    DIFFERENT things — location (audio cluster, this read) vs discomfort
    (interoception cluster, the drive read) — but that split is only
    STRUCTURALLY REACHABLE today, not enforced: ``_read_drive_states`` reads
    the raw signed value (not a comfort-distance fold), so for such a body the
    interoception encode carries the same signed azimuth as the audio encode,
    and drive-relief credit (interoception) plus operant credit (audio) can
    reinforce the same directional contingency on two stacking clusters.
    ``bodies/reachy_mini`` IS such a body (innate azimuth centeredness drive +
    this sensor, since live_audio_orient_wiring Stage 0b) — Exp 54 Phase C reads
    the nursery-taught audio bias out under it, with nothing crediting, so the
    stacking interaction is out of that measurement's scope; folding drive-
    bearing signed sensors to discomfort magnitude in the intero read stays a
    named deferred item in docs/plans/archive/exteroception_interoception_seam.md
    (its trigger has fired for the user path; it bites only with credit ON).
    """
    embodiment = getattr(executor, "embodiment", None)
    root = getattr(embodiment, "root", None)
    if root is None:
        return {}
    sensors = getattr(root, "sensors", {}) or {}
    vm = getattr(root, "vital_metrics", {}) or {}
    out: dict[str, float] = {}
    for name in _EXTEROCEPTIVE_ROOT_SENSORS:
        if name in sensors and name in vm:
            try:
                out[name] = float(vm[name])
            except (TypeError, ValueError):
                continue
    legacy_emitted = set(out)  # names the legacy walk ACTUALLY read this call
    if out and place_code_exteroception_enabled():
        # Population code REPLACES the raw scalar — emitting both would hand the
        # encoder a redundant basis pair whose constant-ish contribution dilutes
        # the very dimension the code exists to resolve (the extero/intero
        # dilution failure, one level down).
        from maxim.similarity.place_code import place_code

        coded: dict[str, float] = {}
        for name, value in out.items():
            coded.update(place_code(value, prefix=f"{_PLACE_CODE_PREFIX}_{name}_"))
        out = coded
    # Declared `modality: audio` sensors join the channel RAW (1.1.4 PR 2) —
    # the place code stays scoped to the legacy tuple, its validated domain
    # (azimuth-shaped [-1,1] scalars; Exp 46's centers assume it). Dedupe is
    # against what the legacy walk ACTUALLY EMITTED this call — not the tuple
    # by name — so a CHILD entity's sensor that happens to share a tuple name
    # still joins when the root has no such sensor (executor-lens review,
    # PR 2 round: name-global exclusion silently dropped it).
    declared = _read_declared_modality_states(executor, AUDIO_TAG)
    for name, value in declared.items():
        if name not in legacy_emitted:
            out.setdefault(name, value)
    return out


def _read_exteroceptive_ranges(executor: Any) -> "dict[str, tuple[float, float]]":
    """Declared ``(lo, hi)`` for the exteroceptive sensors ``_read_exteroceptive_
    states`` reads, so signed sensors (azimuth on ``[-1, 1]``) fold MONOTONICALLY
    (P1) — a left sound and a right sound must not collapse into one cluster."""
    embodiment = getattr(executor, "embodiment", None)
    root = getattr(embodiment, "root", None)
    if root is None:
        return {}
    sensors = getattr(root, "sensors", {}) or {}
    vm = getattr(root, "vital_metrics", {}) or {}
    # The same predicate the STATES walk's legacy loop emits under — the
    # declared-audio dedupe below must mirror it exactly (dedupe by actual
    # emission, never by tuple name; executor-lens review, PR 2 round).
    legacy_names = {n for n in _EXTEROCEPTIVE_ROOT_SENSORS if n in sensors and n in vm}
    # LOCKSTEP INVARIANT (same class as _read_drive_ranges): this walk and
    # _read_exteroceptive_states must emit the same sensor SET. A value with no
    # declared range silently re-folds through the legacy range-blind map (P1),
    # so a place-coded value walk with a raw range walk would encode seven
    # activations under the wrong normalisation. Guarded by
    # test_place_code_wiring.py::test_value_and_range_walks_stay_in_lockstep.
    if place_code_exteroception_enabled():
        from maxim.similarity.place_code import place_code_ranges

        coded_ranges: dict[str, tuple[float, float]] = {}
        for name in _EXTEROCEPTIVE_ROOT_SENSORS:
            if sensors.get(name) is None:
                continue
            coded_ranges.update(place_code_ranges(prefix=f"{_PLACE_CODE_PREFIX}_{name}_"))
        # LOCKSTEP with the declared-audio merge in _read_exteroceptive_states:
        # declared sensors join RAW on both walks even when the legacy tuple is
        # place-coded, or a declared value would silently re-fold rangeless.
        for name, rng_pair in _read_declared_modality_ranges(executor, AUDIO_TAG).items():
            if name not in legacy_names:
                coded_ranges.setdefault(name, rng_pair)
        return coded_ranges

    ranges: dict[str, tuple[float, float]] = {}
    for name in _EXTEROCEPTIVE_ROOT_SENSORS:
        sensor = sensors.get(name)
        if sensor is None:
            continue
        rng = sensor.reading_schema.get("range")
        try:
            if rng is not None and len(rng) == 2:
                ranges[name] = (float(rng[0]), float(rng[1]))
        except (TypeError, ValueError):
            logger.debug("exteroceptive %r has a malformed range %r; skipping (legacy map)", name, rng)
    # LOCKSTEP with the declared-audio merge in _read_exteroceptive_states.
    for name, rng_pair in _read_declared_modality_ranges(executor, AUDIO_TAG).items():
        if name not in legacy_names:
            ranges.setdefault(name, rng_pair)
    return ranges


# ── Substrate modality channels (extero/intero seam) ─────────────────────
#
# Declarative registry: one entry per sensory stream, one
# ``encode_sensors(modality=tag)`` call per non-empty channel — NEVER merged
# into a single encode (docs/plans/archive/exteroception_interoception_seam.md: the
# pre-seam ``{**drives, **extero}`` merge diluted exteroceptive direction
# among the interoceptive drives in one text-embed cluster, collapsing
# left/right onto the same EC node → the embodied orient sim at chance).
# Adding a future modality (vision, touch) is one tuple entry here.
# EC scans within-modality only and "audio" is already frozen-centroid, so
# each channel gets its own cluster space with the right centroid policy.
# NOTE (selection dynamics): ``max_cluster_reward_bias`` caps PER cluster, so
# the summed cluster term in ``recommend_action`` scales with the number of
# active channels (±N for N modalities) — adding a channel here is a
# selection-dynamics change; re-check gate calibration (min_confidence)
# when you add one. The world channel's addition was RE-BASELINED, not
# assumed: scripts/selection_dynamics_rebaseline.py + the committed record
# (docs/plans/archive/world_seam_1_1_4.md §PR 2). The channel is inert (empty read →
# no encode) for every body that declares no `modality: world` sensor.
_SUBSTRATE_CHANNELS: "tuple[ModalityChannel, ...]" = (
    ModalityChannel(INTEROCEPTION_TAG, _read_drive_states, _read_drive_ranges),
    ModalityChannel(AUDIO_TAG, _read_exteroceptive_states, _read_exteroceptive_ranges),
    ModalityChannel(WORLD_TAG, _read_world_states, _read_world_ranges),
)


def _encode_current_clusters(sensor_encoder: Any, agent_id: str, executor: Any) -> dict[str, str]:
    """Encode the CURRENT sensor state into ``{modality: cluster_id}``.

    The same per-channel encode ``propose_via_substrate`` does, but callable at
    outcome time so an llm-primary / real-hardware action (where the LLM, not the
    substrate, chose the action) can still key its real drive-relief outcome onto
    the interoception (and audio) cluster — closing the substrate WRITE path in
    those modes (Phase 1, substrate_learns_from_experience.md). Returns ``{}`` on
    no encoder / no sensors / encode failure (never raises into the loop).
    """
    clusters: dict[str, str] = {}
    if sensor_encoder is None:
        return clusters
    for ch in _SUBSTRATE_CHANNELS:
        try:
            vals = ch.read_values(executor)
            if not vals:
                continue
            node_id = sensor_encoder.encode_sensors(
                agent_id=agent_id,
                sensors=vals,
                modality=ch.tag,
                ranges=ch.read_ranges(executor) or None,
            )
        except Exception:
            # Same policy as propose_via_substrate: a channel that fails to encode
            # is "sensors but no cluster" — surface it, don't crash.
            logger.warning(
                "substrate channel %r encoding raised at outcome time — cluster absent",
                ch.tag,
                exc_info=True,
            )
            continue
        if node_id:
            clusters[ch.tag] = node_id
    return clusters


def _encode_was_designed_rest(sensor_encoder: Any, agent_id: str, modality: str) -> bool:
    """Duck-typed probe of ``SensorEncoder.last_encode_was_designed_rest``
    (fakes without it: never designed rest, the WARNING stays)."""
    probe = getattr(sensor_encoder, "last_encode_was_designed_rest", None)
    if not callable(probe):
        return False
    try:
        return bool(probe(agent_id=agent_id, modality=modality))
    except (TypeError, AttributeError, KeyError):
        # Narrow + logged (this file is on the swallow lint's zero list; the
        # safe direction is keeping the WARNING, so a broken probe must not
        # hide silently either).
        logger.debug("designed-rest probe raised; treating as not-designed-rest", exc_info=True)
        return False


_DEFAULT_SUBSTRATE_MIN_CONFIDENCE = 0.3


def _resolve_min_confidence(explicit: float | None) -> float:
    """Resolve ``min_confidence`` for ``propose_via_substrate``.

    Precedence: explicit caller argument > ``MAXIM_NAC_MIN_CONFIDENCE`` env
    var > ``_DEFAULT_SUBSTRATE_MIN_CONFIDENCE`` (0.3). The env var exists for
    Roy-2c (H1 vs H2 disambiguator) and the Wire-A ablation surface in
    [docs/plans/archive/release_0_9_1.md](../../docs/plans/archive/release_0_9_1.md). Invalid
    env values fall back to the default with a warning, not a crash.
    """
    if explicit is not None:
        return explicit
    raw = os.environ.get("MAXIM_NAC_MIN_CONFIDENCE")
    if raw is None or raw == "":
        return _DEFAULT_SUBSTRATE_MIN_CONFIDENCE
    try:
        return float(raw)
    except ValueError:
        logger.warning(
            "MAXIM_NAC_MIN_CONFIDENCE=%r is not a float; using default %.2f",
            raw,
            _DEFAULT_SUBSTRATE_MIN_CONFIDENCE,
        )
        return _DEFAULT_SUBSTRATE_MIN_CONFIDENCE


class _NoSituationCue:
    """The explicit opt-out for ``propose_via_substrate(situation_cue=...)`` (memory 2S-d)."""

    def __repr__(self) -> str:
        return "NO_SITUATION_CUE"


# A caller with no episodic memory (e.g. the Exp 53 readout: an NAc and an EC, no Hippocampus) says
# so with this, never with ``None`` -- ``None`` is what a hub with no ATL would hand over by accident.
NO_SITUATION_CUE = _NoSituationCue()


def propose_via_substrate(
    *,
    nac: Any,
    agent_id: str,
    executor: Any,
    situation_cue: "Callable[[str, dict[str, str] | None], Any] | _NoSituationCue",
    min_confidence: float | None = None,
    sensor_encoder: Any | None = None,
) -> LLMProposal | None:
    """Build an ``LLMProposal`` from ``NAc.recommend_action`` — no LLM call.

    Called from the agent loop when ``aut_mode == "substrate-primary"`` in
    place of ``llm_worker.submit_context``. Returns ``None`` when the
    substrate has no opinion (no learned bias, no active drive) — the loop
    treats this as IDLE for that tick rather than proposing randomly.

    The returned proposal carries ``strategy_used="substrate-primary"`` so
    downstream tracing can distinguish substrate-proposed actions from
    LLM-proposed ones.

    Args:
        situation_cue: REQUIRED (memory 2S-d): ``MemoryHub.situation_cue``, called with this tick's
            clusters so a situation CHANGE recalls the memories formed in it, or ``NO_SITUATION_CUE``
            for a caller with no episodic memory. Required because the survival harnesses call this
            function directly, bypassing the loop -- an optional hook would silently never fire
            there. Recall only until 2S-e consumes it; a failing cue never costs the tick.
        sensor_encoder: Optional :class:`SensorEncoder` (Phase 0 of
            grounded_language_acquisition.md). When wired, the current
            drive snapshot is hashed into the substrate via
            ``encode_sensors`` once per tick *before* reading drives for
            ``recommend_action``. This lets EC accumulate sensor-pattern
            nodes during substrate-primary runs — without it the
            text-only ``LinguisticEncoder`` path is the substrate's only
            front door, so substrate-primary mode never produces EC nodes.
    """
    if situation_cue is None or not (situation_cue is NO_SITUATION_CUE or callable(situation_cue)):
        raise TypeError(
            "propose_via_substrate(situation_cue=...) takes MemoryHub.situation_cue or NO_SITUATION_CUE, "
            f"got {situation_cue!r}"
        )
    if nac is None or executor is None:
        return None

    registry = getattr(executor, "registry", None)
    if registry is None or not hasattr(registry, "list"):
        return None

    available_tools = list(registry.list())
    if not available_tools:
        return None

    # Substrate-primary is an EMBODIED action test: exclude read-only cognitive
    # introspection tools (memory_recall, temporal_patterns, system_stats, …).
    # They always succeed, so their causal confidence snowballs toward the cap
    # and dominates recommend_action — starving the embodied affordances the
    # mode exists to measure (the meta-tool fixation that VOID'd the Exp 42
    # triage: the agent fidgeted with temporal_patterns/system_stats instead of
    # warming). LLM-AUT is unaffected — it never calls this path. Set lives in
    # tools/introspection.py so it can't drift from the registered tool names.
    from maxim.tools.introspection import INTROSPECTION_TOOL_NAMES

    available_tools = [t for t in available_tools if t not in INTROSPECTION_TOOL_NAMES]
    if not available_tools:
        return None

    # Optional experiment-scoped whitelist: restrict substrate-primary action
    # selection to a MINIMAL affordance repertoire. The introspection filter above
    # only removes read-only cognitive tools; non-introspection tools that also
    # "always succeed" (sense_presence, sense, examine, say, …) still snowball
    # causal confidence and out-compete the affordances under test — the cradle
    # orient infant kept choosing sense_presence (causal_pos 0.99) over turn_left/
    # turn_right. A newborn's motor repertoire is small; the 22 generic tools are
    # the artificial part. Substring match (tools are body-prefixed, e.g.
    # infant_operant_turn_left). Experiment/harness toggle (env, not config).
    # Autouse scrub: tests/conftest.py.
    #
    # BAND-AID (tracked): this masks the ROOT cause rather than fixing it — a tool
    # that merely EXECUTES accrues causal credit as if it made goal/drive progress,
    # so mechanically-successful tools drown a specific operant/drive signal. The
    # real fix (credit-on-progress-not-execution) is
    # docs/plans/deferred/credit_on_progress_not_execution.md; this whitelist is a
    # scoped work-around for the dormant cradle_mother demo until that lands.
    _tool_whitelist = os.environ.get("MAXIM_SUBSTRATE_TOOL_WHITELIST", "").strip()
    if _tool_whitelist:
        _wl_terms = [w.strip() for w in _tool_whitelist.split(",") if w.strip()]
        if _wl_terms:
            available_tools = [t for t in available_tools if any(term in t for term in _wl_terms)]
            if not available_tools:
                return None

    # Per-modality channel reads (extero/intero seam). Each channel is read
    # once here for the ENCODE; interoception is re-read after the pain
    # tick below so selection sees post-drift drives. EVERY non-empty
    # channel gets its OWN encode below.
    #
    # ORDER (Wire 4, Exp 58 wiring W-4): the encode runs BEFORE the
    # evaluate_failures pain tick so pain published this tick books fear
    # on THIS tick's situation clusters — pre-fix, damage on the lit→dark
    # transition tick saw the PREVIOUS tick's clusters and wrote fear on
    # the LIT cluster (aimed straight at the specificity gate). Cluster
    # identity from the pre-drift snapshot is equivalent for this purpose:
    # world-owned sensors do not drift, and one tick of drift cannot move
    # a cluster id. ACCEPTED LAG (architecture-lens): bodies with
    # metadata["health"] == "derived" refresh health from modulator
    # integrities INSIDE evaluate_failures, so their interoception encode
    # now lags integrity damage by one tick (self-correcting next tick);
    # the minecraft body is bridge-written, not derived, so Exp 58 is
    # unaffected — revisit if a derived-health body joins a fear line.
    channel_values: dict[str, dict[str, float]] = {ch.tag: ch.read_values(executor) for ch in _SUBSTRATE_CHANNELS}

    # Phase 0 sensor encoding — feed the current sensor snapshot to EC so
    # substrate-primary mode produces nodes the way the LLM-primary
    # text-percept path does. ONE ``encode_sensors(modality=tag)`` call per
    # non-empty channel — NEVER merged: the pre-seam ``{**drives, **extero}``
    # merge encoded exteroceptive direction as one term in a text-embed sum
    # dominated by the drives, so left/right collapsed onto one EC cluster
    # and the agent was blind to direction (the dilution root cause,
    # docs/plans/archive/exteroception_interoception_seam.md). Fail-soft per channel:
    # an encoding error must not block the action proposal or the other
    # channels. The resulting ``{modality: cluster_id}`` set flows into
    # recommend_action (additive cluster_reward_bias sum) and onto the
    # proposal for the outcome path's credit routing.
    clusters: dict[str, str] = {}
    if sensor_encoder is not None:
        for ch in _SUBSTRATE_CHANNELS:
            vals = channel_values.get(ch.tag)
            if not vals:
                continue
            try:
                node_id = sensor_encoder.encode_sensors(
                    agent_id=agent_id,
                    sensors=vals,
                    modality=ch.tag,
                    ranges=ch.read_ranges(executor) or None,
                )
            except Exception:
                # WARNING, not debug: a channel with sensors that failed to
                # encode IS "sensors but no cluster" — downstream, the credit
                # router silently falls back (operant pending keys on
                # interoception when audio is missing), so a quiet failure
                # here becomes invisible mis-routed credit (pre-merge review,
                # both lenses).
                logger.warning(
                    "substrate channel %r encoding raised — its cluster is absent this tick",
                    ch.tag,
                    exc_info=True,
                )
                continue
            if node_id:
                clusters[ch.tag] = node_id
            elif _encode_was_designed_rest(sensor_encoder, agent_id, ch.tag):
                # A gained body resting at neutral encodes nothing BY DESIGN
                # (D2) — per-tick WARNING here would make designed rest
                # indistinguishable from failure (plan §PR 3 seam note).
                logger.debug("substrate channel %r rests at neutral — no cluster by design", ch.tag)
            else:
                # A channel with sensors that yields no cluster is the
                # dilution failure mode's silent sibling — surface it.
                logger.warning(
                    "substrate channel %r has %d sensor(s) but yielded no cluster",
                    ch.tag,
                    len(vals),
                )
    cluster_id = clusters.get(INTEROCEPTION_TAG)

    # Wire 4 (Exp 58): stash THIS tick's clusters on NAc so the
    # pain→cluster-fear subscriber keys fear to the current situation.
    # Noted even when empty (clears the stash — pain with no situation
    # books nothing rather than a stale one).
    try:
        nac.note_active_clusters(agent_id, clusters or None)
    except Exception:
        logger.warning("note_active_clusters raised — pain this tick cannot key to a situation", exc_info=True)

    # Memory 2S-d: a situation CHANGE recalls the memories formed in it (recall only until 2S-e).
    # isinstance, not identity, so mypy narrows: the guard at the top admits only the
    # NO_SITUATION_CUE sentinel or a callable, so the two tests select the same calls.
    if not isinstance(situation_cue, _NoSituationCue):
        try:
            situation_cue(agent_id, clusters or None)
        except Exception:
            log_swallowed_exception()

    # Substrate-primary mode owns its own clock — without an LLM submit
    # path there's no other code that calls into the embodiment, so
    # drive drift would never advance. Ticking evaluate_failures() here
    # mirrors the llm-primary path, where the tick is event-driven via
    # tool execution (tool_bridge / sim tools calling evaluate_failures):
    # applies wall-clock drift via tick_vital_drift, then evaluates
    # failures (which publish pain signals — now keyed to the clusters
    # noted above). See the CLAUDE.md embodiment-tick invariant.
    embodiment = getattr(executor, "embodiment", None)
    if embodiment is not None:
        try:
            embodiment.evaluate_failures()
        except Exception:
            logger.debug("substrate-primary tick: evaluate_failures raised", exc_info=True)

    # Post-drift interoception re-read: selection must see the drives the
    # pain tick just advanced (the pre-hoist behaviour, preserved).
    drives = dict(_read_drive_states(executor))

    # Wire 4 READ: learned fear of the ACTIVE situation surfaces as an
    # anticipatory threat need, combined with the innate reactive
    # ``health→threat`` need by MAX, never sum (Exp 58 bio SF-6 — a sum
    # can exceed 1.0 and be dropped by recommend_action's raw-sensor
    # guard). Zero when no active cluster clears the fear threshold.
    try:
        fear_need = float(nac.anticipatory_threat_need(agent_id, clusters or None))
    except Exception:
        logger.warning("anticipatory_threat_need raised — learned fear is silent this tick", exc_info=True)
        fear_need = 0.0
    if fear_need > 0.0:
        drives["threat"] = max(float(drives.get("threat", 0.0) or 0.0), fear_need)

    resolved_min_confidence = _resolve_min_confidence(min_confidence)
    recommendation = nac.recommend_action(
        agent_id=agent_id,
        available_tools=available_tools,
        current_drives=drives or None,
        current_cluster_id=cluster_id,
        current_clusters=clusters or None,
        min_confidence=resolved_min_confidence,
    )
    if recommendation is None:
        return None

    action = {
        "tool_name": recommendation["tool_name"],
        "params": recommendation.get("params", {}) or {},
    }
    return LLMProposal(
        request_id=f"substrate-{int(time.time() * 1000)}",
        action=action,
        reasoning=recommendation.get("reasoning", ""),
        strategy_used="substrate-primary",
        confidence=float(recommendation.get("confidence", resolved_min_confidence)),
        mode_goal_achieved=False,
        triggering_input="",
        # G4 + seam: stash the active EC cluster set on the proposal so the
        # outcome path can route credit per modality into
        # ``NAc._cluster_reward_bias[(agent, cluster, tool)]`` — see
        # record_outcome in tool_dispatch.py. ``cluster_id`` is the legacy
        # interoception alias; ``clusters`` is the full per-modality set.
        # Both ``None``/empty when no sensor encoder was wired or no channel
        # produced a cluster.
        cluster_id=cluster_id,
        clusters=clusters or None,
        cluster_margins=_situation_margins(sensor_encoder, agent_id, clusters),
    )


def _situation_margins(sensor_encoder: Any, agent_id: str, clusters: dict[str, str] | None) -> dict[str, float] | None:
    """The EC match margin each situation cluster was just encoded with (memory-strength Phase 2S-c).

    Read IMMEDIATELY after the encodes that produced ``clusters`` -- ``last_encode_margin`` is the
    encoder's most recent encode per (agent, modality), so a later tick would read a different one.
    A modality whose encode ran no scan (the min-delta gate) has no margin and is left out.
    """
    reader = getattr(sensor_encoder, "last_encode_margin", None)
    if not clusters or not callable(reader):
        return None
    margins: dict[str, float] = {}
    for modality in clusters:
        margin = reader(agent_id=agent_id, modality=modality)
        if isinstance(margin, (int, float)) and not isinstance(margin, bool) and math.isfinite(margin):
            margins[modality] = float(margin)
    return margins or None


def _attach_live_situation(proposal: Any, *, aut_mode: str, sensor_encoder: Any, agent_id: str, executor: Any) -> Any:
    """In llm-primary the LLM chose the action, so ``propose_via_substrate`` never ran and no
    substrate cluster was captured: encode the current interoception (+audio) state HERE -- the
    PRE-action drive state, the correct credit key -- so the real drive-relief outcome reinforces the
    cluster-reward substrate via record_outcome (drive_relief_only, no tool-success floor), and the
    capture records where it happened and how novel that was (Phase 1 of
    substrate_learns_from_experience.md; Phase 2S-b/c). No-op in substrate-primary (clusters already
    captured) and when unembodied.
    """
    if (
        aut_mode == "substrate-primary"
        or sensor_encoder is None
        or getattr(proposal, "clusters", None) is not None
        or getattr(executor, "embodiment", None) is None
    ):
        return proposal
    live = _encode_current_clusters(sensor_encoder, agent_id, executor)
    if not live:
        return proposal
    return dataclasses.replace(
        proposal,
        cluster_id=live.get(INTEROCEPTION_TAG),
        clusters=live,
        cluster_margins=_situation_margins(sensor_encoder, agent_id, live),
    )
