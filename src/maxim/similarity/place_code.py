"""Place-cell population coding for scalar sensors (direction, initially).

**Why.** ``_sensor_embed`` represents one scalar as a two-basis interpolation —
``(1-v)*basis_low + v*basis_high`` — a 1-D arc between two near-orthogonal hash
vectors. Cosine along that arc falls off slowly, so at the production sensor
threshold a full ``[-1, 1]`` azimuth sweep resolves into only **~2-3 EC nodes**
(left / centre / right). Exp 46 measured exactly this and named it a
*perceptual*, not a learning, limit: a graded orienting task (six directions,
six turns) plateaus at ~0.30 because the agent cannot perceive −0.9 from −0.6.

**The fix, already validated.** Tile the range with overlapping Gaussian
tuning curves — one "cell" per direction — and hand the encoder N activations
instead of one scalar. Each cell contributes its OWN basis pair in
``_sensor_embed`` (which sums per sensor NAME), so resolution becomes a design
parameter instead of a threshold side-effect. Exp 46:
**6/6 distinct clusters at width ≤ 0.15**, and with it the graded task went
**taught 0.19 → 0.82 (LEARNED + MOTHER-TAUGHT PASS)**, versus yoked 0.03 /
none 0.17. Its write-up recommends this as "the standard direction encoding for
spatial tasks going forward"; this module is that promotion, from
``scripts/orient_substrate/6_graded_orient_curve.py`` into production.

**Bio-mapping: FUNCTIONAL.** Overlapping tuning curves tiling a variable is
genuinely how head-direction cells represent heading (Taube 1990) — but a
tuning-curve *encoding* is not the *mechanism*. The head-direction system is a
**ring attractor**: recurrent excitation plus global inhibition producing a bump
that persists without input and integrates angular velocity. None of that comes
free here. MECHANISM-tier would require bump persistence, path integration from
commanded yaw, and landmark correction, each measured. Do not upgrade this tag
without them.

**NOT a ring code.** ``azimuth`` is *not* circular: ``doa_to_azimuth`` maps
−1 = hard left, 0 = centred (front), +1 = hard right — the endpoints are 180°
apart. Wrapping them onto each other would destroy the left/right
discrimination Exp 45's orient policy and Exp 48's operant result depend on.
The cell centres below therefore tile a **line segment**, not a circle. (The
sensor's real degeneracy is front/back at az≈0 — a linear-mic-array hardware
limit that no encoding can fix.)
"""

from __future__ import annotations

import math

# Exp 46's validated parameters. Width 0.12 sits inside the ``≤ 0.15`` band that
# gave 6/6 separation; centres tile [-1, 1] at 0.3 spacing (2.5 widths apart, so
# adjacent cells overlap enough to interpolate but not enough to merge).
DEFAULT_TUNING_WIDTH: float = 0.12
DEFAULT_CELL_CENTERS: tuple[float, ...] = (-0.9, -0.6, -0.3, 0.0, 0.3, 0.6, 0.9)

# Activations below this contribute nothing an EC cosine can see but do add a
# basis vector each; dropping them keeps the embedding sparse and makes the
# encoder's ``sensor_names`` provenance stamp meaningful (see module docstring
# of similarity/encoder.py on encoder_provenance).
_ACTIVATION_FLOOR: float = 1e-3


def place_code(
    value: float,
    *,
    prefix: str,
    centers: tuple[float, ...] = DEFAULT_CELL_CENTERS,
    width: float = DEFAULT_TUNING_WIDTH,
) -> dict[str, float]:
    """Gaussian population code for one scalar. ``{f"{prefix}{i}": activation}``.

    Pure function, no EC/encoder coupling — the caller decides where it goes.
    Activations are the unnormalised Gaussian ``exp(-(v-c)^2 / 2w^2)`` in
    ``[0, 1]``, so every cell's declared range is ``(0.0, 1.0)`` (see
    :func:`place_code_ranges`) and ``_normalize_value`` is the identity on them.
    """
    if width <= 0:
        raise ValueError(f"place_code width must be > 0, got {width!r}")
    out: dict[str, float] = {}
    for i, c in enumerate(centers):
        a = math.exp(-((float(value) - c) ** 2) / (2.0 * width * width))
        if a >= _ACTIVATION_FLOOR:
            out[f"{prefix}{i}"] = a
    return out


def place_code_ranges(
    *,
    prefix: str,
    centers: tuple[float, ...] = DEFAULT_CELL_CENTERS,
) -> dict[str, tuple[float, float]]:
    """Declared ranges for :func:`place_code` outputs — every cell is ``(0, 1)``.

    Emitted for EVERY cell, including ones :func:`place_code` drops below the
    activation floor: the ranges map is a static declaration of the channel's
    shape, while the values map is per-reading. (The encoder tolerates ranges
    for absent sensors; the reverse — a value with no range — is what silently
    re-folds a signed sensor, per the P1 range-aware invariant.)
    """
    return {f"{prefix}{i}": (0.0, 1.0) for i in range(len(centers))}
