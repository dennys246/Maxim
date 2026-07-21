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
) -> dict[str, Any]:
    """Apply one turn of the reactive mother's caregiving to the infant body.

    Order is LOAD-BEARING: **feed-if-oriented (on the PRIOR azimuth) → place the
    stimulus → guide (fading) → speak.** Feeding must read the azimuth the
    infant's *previous* turn left, BEFORE the new stimulus overwrites it — that is
    what rewards the infant's own orient (and gives the temporal structure
    "orient this turn → fed next turn"). Feeding on the post-stimulus/post-guide
    azimuth would let each new stimulus erase the infant's orient before it could
    be rewarded, and the infant would never learn.

    - **Feed** relieves hunger/thirst when the prior azimuth is within
      ``oriented_threshold`` (the infant faced the mother last turn).
    - **Stimulus** world-sets ``azimuth`` to the mother's direction this turn
      (``stimulus_azimuths`` rotated by ``turn_idx``) — the thing to orient
      toward. Substrate-primary's §1.16 audio world-set is gated out, so the
      mother sets it directly (world-driven, not the AUT path).
    - **Guide** world-sets ``azimuth`` toward center by ``guide_strength`` (the
      fading scaffold; the caregiver can center beyond the infant's own reach).
    - **Speak** injects motherese via the caller's substrate-safe ``inject``.

    Returns telemetry for the fade-curve analyzer: ``fed``/``guided``/``spoke``
    plus ``az_prior`` (what the infant left), ``az_stimulus`` (mother's direction),
    ``az_guided`` (after the scaffold). Fail-soft: missing body / azimuth sensor
    is a no-op.
    """
    out: dict[str, Any] = {
        "fed": False,
        "guided": False,
        "spoke": None,
        "az_prior": None,
        "az_stimulus": None,
        "az_guided": None,
    }
    root = getattr(embodiment, "root", None)
    if root is None:
        return out
    vm = getattr(root, "vital_metrics", None)
    if vm is None:
        return out

    from maxim.embodiment.audio_localization import world_set_azimuth

    # 1. FEED if the infant's PRIOR turn left it oriented — rewards its own orient.
    az_prior = vm.get("azimuth")
    out["az_prior"] = az_prior
    if scaffold.feed_amount > 0.0 and az_prior is not None and abs(float(az_prior)) <= scaffold.oriented_threshold:
        deltas = {"hunger": -scaffold.feed_amount}
        if scaffold.thirst_ratio > 0.0:
            deltas["thirst"] = -scaffold.feed_amount * scaffold.thirst_ratio
        try:
            from maxim.embodiment.tool_bridge import _apply_sensor_deltas

            _apply_sensor_deltas(root, deltas, delta_kind="target_effect")
            out["fed"] = True
        except Exception:
            logger.debug("reactive_mother_tick: feed (_apply_sensor_deltas) failed", exc_info=True)

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
