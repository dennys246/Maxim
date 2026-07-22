"""Reactive mother — the fading caregiver scaffold for the cradle orient sim.

The mother is a per-turn, world-driven effect on the PASSIVE infant body (the
opposite of an AUT-initiated affordance): each turn she (1) guides the infant's
head toward her by the current act's *fade* strength, (2) feeds it (hunger/thirst
relief) *iff* it is oriented toward her, and (3) speaks motherese the infant
hears. Across the 4-act cradle arc the guidance fades (full → partway → none), so
the infant must increasingly orient itself to be fed — and the fraction it orients
itself per act is the measured learning curve.

This is the **reactive-script v1** from the cradle plan, NOT the deferred
generative Mother NPC (no separate LLM, deterministic, a scripted stimulus). Every
primitive it uses already exists; this module only sequences them:

- head guidance → ``world_set_azimuth`` (azimuth is a WORLD-SET root sensor, not
  delta-applied). The mother can center the head **beyond the infant's own motor
  reach** (``reflex_oriented_azimuth``'s reach clamp is the infant's *own* limit;
  a caregiver physically turning the head is not bound by it) — that extra reach
  is part of what makes the scaffold a scaffold.
- feeding → ``_apply_sensor_deltas(root, {"hunger": -amount}, ...)`` (additive,
  clamped) — a caregiver acting on the infant, the ``target_effect`` kind.
- motherese → a text percept injected through a substrate-safe path (the caller
  supplies ``inject``; ``send_and_wait`` is suppressed in substrate-primary).

Wiring (deferred to the integration step): call ``reactive_mother_tick`` once per
turn from the generative-runner turn loop (``embodiment`` is already in scope),
with the ``MotherScaffold`` for the current ``NarrativePhase``.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Motherese — infant-directed speech (audio-visual + language grounding). Rotated
# deterministically per turn (no LLM; a scripted stimulus).
DEFAULT_MOTHERESE: tuple[str, ...] = (
    "here comes the choo-choo train",
    "who's a hungry baby",
    "look at mama, sweet one",
    "mmm, yummy yummy",
    "over here, little one",
)


@dataclass(frozen=True)
class MotherScaffold:
    """Per-phase reactive-mother config — the fade schedule for one cradle act.

    Runtime-ephemeral sim config (not persisted / wire-crossing), so it is out of
    scope for the CC3 frozen-dataclass audit; add fields with defaults freely.

    guide_strength: fraction the mother turns the infant's head toward her this
        turn. ``1.0`` = fully center (Act 1, passive infant); ``0.5`` = halfway
        (Act 2, co-active); ``0.0`` = no guidance (Act 3+, autonomous). This is
        the primary fade knob.
    feed_amount: hunger relief applied when the infant is oriented (thirst gets
        ``feed_amount * thirst_ratio``). ``0.0`` disables feeding.
    oriented_threshold: ``|azimuth|`` below which the infant counts as facing the
        mother — the contingency that unlocks feeding (default matches the
        base_humanoid centeredness comfort_band, 0.1).
    thirst_ratio: thirst relief as a fraction of ``feed_amount``.
    stimulus_azimuths: the sequence of directions the mother "calls" from,
        rotated per turn (world-set onto the infant's azimuth each turn as the
        thing to orient toward). Empty = no stimulus (the tick only feeds/guides —
        e.g. a non-orient act). Values in ``[-1, 1]``.
    speech: motherese lines rotated per turn; empty = silent.
    """

    guide_strength: float = 0.0
    feed_amount: float = 0.5
    oriented_threshold: float = 0.1
    thirst_ratio: float = 0.6
    stimulus_azimuths: tuple[float, ...] = ()
    speech: tuple[str, ...] = field(default_factory=lambda: DEFAULT_MOTHERESE)


def reactive_mother_tick(
    embodiment: Any,
    *,
    scaffold: MotherScaffold,
    turn_idx: int = 0,
    inject: Callable[[str], None] | None = None,
    nac: Any | None = None,
    agent_id: str = "",
    feed_reward: float = 1.0,
    prev_stimulus: float | None = None,
) -> dict[str, Any]:
    """Apply one turn of the reactive mother's OPERANT caregiving to the infant.

    The mother is an operant SHAPER (2026-07-21 redesign): she places a sound and
    rewards the infant when its *own* turn moved TOWARD that sound. The reward is
    a feed (hunger/thirst relief) AND — the load-bearing part — a call to
    ``nac.credit_operant_reward`` that reinforces the infant's own recent action
    on the action-selection surface (validated by
    ``scripts/orient_substrate/3_operant_feed_probe.py``). With the infant body's
    intrinsic centeredness drive removed (``bodies/infant_operant``), this operant
    credit is the SOLE teacher of orienting — remove the mother (``no_feed`` arm)
    and the infant never learns. Run with ``MAXIM_OPERANT_ONLY_CREDIT=1`` so the
    tool-success floor doesn't drown the signal (probe 3 ``tool_floor`` arm).

    Order is LOAD-BEARING: **reward-for-prior-progress (on the PRIOR azimuth) →
    place the new stimulus → (optional guide) → speak.** The reward must read the
    azimuth the infant's *previous* turn left, BEFORE the new stimulus overwrites
    it — that is what rewards the infant's own orient and gives the temporal
    structure "orient this turn → fed + credited next turn". The pending operant
    action (set in ``tool_dispatch`` at the infant's action) is likewise from the
    previous turn, so the credit lands on the action that produced the progress.

    Shaping contingency: when ``prev_stimulus`` (the sound placed last turn) is
    known, the infant is rewarded iff ``|prev_stimulus| - |az_prior| > 0`` — it
    turned toward the sound. Without ``prev_stimulus`` (legacy / non-operant use)
    it falls back to the oriented-threshold contingency.

    ``guide_strength`` (the old fading-scaffold knob) is retained for back-compat
    but the operant arc sets it to 0: physically turning the head and then
    crediting the infant's own action for the mother's guide would be dishonest,
    so the honest operant curriculum is pure shaping and the "fade" is the
    EMERGENT learning curve (directedness rising across the session).

    Returns telemetry: ``fed``/``credited``/``guided``/``spoke`` plus ``az_prior``
    (what the infant left), ``az_stimulus`` (mother's direction this turn — the
    caller feeds it back as next turn's ``prev_stimulus``), ``az_guided``,
    ``progress``. Fail-soft: missing body / azimuth sensor is a no-op.
    """
    out: dict[str, Any] = {
        "fed": False,
        "credited": False,
        "guided": False,
        "spoke": None,
        "az_prior": None,
        "az_stimulus": None,
        "az_guided": None,
        "progress": None,
    }
    root = getattr(embodiment, "root", None)
    if root is None:
        return out
    vm = getattr(root, "vital_metrics", None)
    if vm is None:
        return out

    from maxim.embodiment.audio_localization import world_set_azimuth

    # 1. REWARD the infant's PRIOR turn — operant shaping toward the sound.
    az_prior = vm.get("azimuth")
    out["az_prior"] = az_prior
    # Compute directedness (turned toward last turn's sound) UNCONDITIONALLY when
    # we can — the ``no_feed`` control arm (feed_amount 0) must still log
    # ``progress`` so the analyzer can compare directedness across arms
    # independent of whether the infant was fed.
    progress: float | None = None
    if prev_stimulus is not None and az_prior is not None:
        progress = abs(float(prev_stimulus)) - abs(float(az_prior))
        out["progress"] = progress
    should_feed = False
    if scaffold.feed_amount > 0.0 and az_prior is not None:
        if progress is not None:
            should_feed = progress > 1e-6  # turned TOWARD the sound (shaping)
        else:
            # Legacy / non-operant fallback: reward being oriented.
            should_feed = abs(float(az_prior)) <= scaffold.oriented_threshold
    if should_feed:
        deltas = {"hunger": -scaffold.feed_amount}
        if scaffold.thirst_ratio > 0.0:
            deltas["thirst"] = -scaffold.feed_amount * scaffold.thirst_ratio
        try:
            from maxim.embodiment.tool_bridge import _apply_sensor_deltas

            _apply_sensor_deltas(root, deltas, delta_kind="target_effect")
            out["fed"] = True
        except Exception:
            logger.debug("reactive_mother_tick: feed (_apply_sensor_deltas) failed", exc_info=True)
        # Operant credit: the relief reinforces the infant's OWN recent action on
        # the action-selection surface. This is the teacher (drive removed).
        if nac is not None and agent_id:
            try:
                credited = nac.credit_operant_reward(agent_id, feed_reward)
                out["credited"] = credited is not None
            except Exception:
                logger.debug("reactive_mother_tick: credit_operant_reward failed", exc_info=True)

    # 2. STIMULUS — the mother calls from a direction this turn (world-set azimuth).
    if scaffold.stimulus_azimuths:
        stim = scaffold.stimulus_azimuths[turn_idx % len(scaffold.stimulus_azimuths)]
        try:
            world_set_azimuth(embodiment, float(stim))
            out["az_stimulus"] = stim
        except Exception:
            logger.debug("reactive_mother_tick: stimulus world_set_azimuth failed", exc_info=True)

    # 3. GUIDE (fading scaffold) — world-SET azimuth toward center by guide_strength.
    #    NOT clamped to the infant's own motor reach: the caregiver turns the head.
    az_now = vm.get("azimuth")
    if scaffold.guide_strength > 0.0 and az_now is not None and abs(float(az_now)) > scaffold.oriented_threshold:
        target = float(az_now) * (1.0 - min(1.0, scaffold.guide_strength))
        try:
            world_set_azimuth(embodiment, target)
            out["guided"] = True
        except Exception:
            logger.debug("reactive_mother_tick: guide world_set_azimuth failed", exc_info=True)
    out["az_guided"] = vm.get("azimuth")

    # 4. SPEAK — motherese as a text percept (caller supplies a substrate-safe inject).
    if scaffold.speech and inject is not None:
        line = scaffold.speech[turn_idx % len(scaffold.speech)]
        try:
            inject(line)
            out["spoke"] = line
        except Exception:
            logger.debug("reactive_mother_tick: motherese inject failed", exc_info=True)

    return out
