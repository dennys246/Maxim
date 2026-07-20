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
    speech: motherese lines rotated per turn; empty = silent.
    """

    guide_strength: float = 0.0
    feed_amount: float = 0.5
    oriented_threshold: float = 0.1
    thirst_ratio: float = 0.6
    speech: tuple[str, ...] = field(default_factory=lambda: DEFAULT_MOTHERESE)


def reactive_mother_tick(
    embodiment: Any,
    *,
    scaffold: MotherScaffold,
    turn_idx: int = 0,
    inject: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Apply one turn of the reactive mother's caregiving to the infant body.

    Order is load-bearing: **guide, then check oriented, then feed** — so in Act 1
    the full guide centers the head (oriented) and the infant experiences
    oriented-paired-with-feeding before it can orient itself; in Act 3 (no guide)
    it is fed only if its *own* prior turns left it oriented.

    Returns a telemetry dict for the fade-curve analyzer: whether it guided/fed/
    spoke, and the azimuth before/after (so the harness can attribute the orient
    to the infant vs. the mother). Fail-soft: a missing body / azimuth sensor is a
    no-op (unembodied or non-orienting bodies are unaffected).
    """
    out: dict[str, Any] = {"guided": False, "fed": False, "spoke": None, "az_before": None, "az_after": None}
    root = getattr(embodiment, "root", None)
    if root is None:
        return out
    vm = getattr(root, "vital_metrics", None)
    if vm is None:
        return out

    az_before = vm.get("azimuth")
    out["az_before"] = az_before

    # 1. Fading guidance — world-SET azimuth toward center by guide_strength.
    #    NOT clamped to the infant's own motor reach: the caregiver turns the head.
    if scaffold.guide_strength > 0.0 and az_before is not None and abs(az_before) > scaffold.oriented_threshold:
        target = float(az_before) * (1.0 - min(1.0, scaffold.guide_strength))
        try:
            from maxim.embodiment.audio_localization import world_set_azimuth

            world_set_azimuth(embodiment, target)
            out["guided"] = True
        except Exception:
            logger.debug("reactive_mother_tick: world_set_azimuth failed", exc_info=True)

    az_after = vm.get("azimuth")
    out["az_after"] = az_after

    # 2. Feed — CONTINGENT on the infant being oriented toward the mother. This
    #    contingency is what makes orienting worth learning: in Act 1 the guide
    #    delivers oriented→fed; in Act 3 the infant must orient itself to be fed.
    if scaffold.feed_amount > 0.0 and az_after is not None and abs(float(az_after)) <= scaffold.oriented_threshold:
        deltas = {"hunger": -scaffold.feed_amount}
        if scaffold.thirst_ratio > 0.0:
            deltas["thirst"] = -scaffold.feed_amount * scaffold.thirst_ratio
        try:
            from maxim.embodiment.tool_bridge import _apply_sensor_deltas

            _apply_sensor_deltas(root, deltas, delta_kind="target_effect")
            out["fed"] = True
        except Exception:
            logger.debug("reactive_mother_tick: feed (_apply_sensor_deltas) failed", exc_info=True)

    # 3. Speak — motherese as a text percept (caller supplies a substrate-safe inject).
    if scaffold.speech and inject is not None:
        line = scaffold.speech[turn_idx % len(scaffold.speech)]
        try:
            inject(line)
            out["spoke"] = line
        except Exception:
            logger.debug("reactive_mother_tick: motherese inject failed", exc_info=True)

    return out
