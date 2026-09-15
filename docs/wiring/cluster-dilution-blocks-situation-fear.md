# Situation clusters dilute below separability on a many-sensor world channel (L11, live)

**Established:** 2026-09-14, Exp 58 (learned dark-fear) — the danger and safe situations
would not encode to distinct world clusters live, blocking situation-keyed fear at the
instrument check. The canonical live realization of L11
([docs/limits/l11_sensor_dilution.md](../limits/l11_sensor_dilution.md)).

## The fact

The substrate encodes a modality channel by summing its N sensors into one vector and
comparing by cosine; two situations become the SAME cluster when their cosine exceeds the
`0.85` pattern-completion threshold. On the Minecraft body the `world` channel has **17
sensors**, so each contributes ~1/17 — and **no partial-axis contrast between two
situations clears 0.85.** For clusters to separate you need either a FULL-range swing on a
sensor or several axes swinging large at once; a handful of partial swings sums to a cosine
still well above 0.85 and the situations merge.

## Why it bites (the survival-fear case, concretely)

Exp 58 needed a "danger" situation distinct from a "safe" one so pain could book fear onto
the danger cluster and a probe could read it. Three discriminators were tried; all merged:

- **light_level** — unreliable/non-physical in this world (see
  [world-light-sensing.md](world-light-sensing.md) + the day/night probe): patchy,
  skylight-contaminated underground. Dead as a discriminator here regardless of dilution.
- **depth (`y_altitude`)** alone — a 12-block difference is only 12/128 ≈ 0.09 of the
  sensor's range; far too small a fraction to move the summed cosine below 0.85. Merged,
  and *unstably* (jitter flipped borderline cases run to run).
- **depth + adjacent hostile + ~21-block position** — three axes at once, still merged.
  `hostile_count` was even identical at both situations (the bridge counts ALL loaded
  hostiles, so a forceloaded pit mob shows at safe too); only `nearest_hostile_dist`,
  `y_altitude`, and the offsets swung, all partially.

Phase-0's `survival_phase0` gate reported 1.0/1.0 separability — but only because its box
swung light AND altitude across their FULL ranges together. **The offline gate validated an
easier problem than the live classroom.** A separability instrument must exercise the
*actual* contrast the experiment relies on, at its real magnitude, or it gives false
confidence (the verify-the-instrument lesson, one level deeper).

## What this means for design

- **A situation-keyed contingency on the world channel needs a cluster that actually
  separates.** Before building such an experiment, MEASURE the live cosine geometry of the
  real situations (not a scripted big-swing proxy); if they sit above 0.85, no amount of
  world-building downstream will make the fear/credit readable.
- **The remedies are substrate-level, not apparatus-level** (L11 ledger): channel-split the
  world modality into per-type sub-channels (each back in the small-N regime), a
  `1−k/N` scaled threshold, and/or the A4 nonlinear gain (already default for `world`, and
  insufficient alone at N=17). Pick from a live diagnostic, not from the synthetic law.
- **Don't force separation by tuning a sensor's range to the apparatus** (e.g. narrowing
  `y_altitude` so 12 blocks reads as a big swing) without a confounding-lens check — it is
  D1-adjacent (tuning the instrument to make one experiment pass).

## See also

`docs/limits/l11_sensor_dilution.md` (the measured 1/N law, A4 bake-off, grouping + scaled
threshold); `docs/experiments/exp58_survival_wants_prereg.md` §Outcome (the block);
[world-light-sensing.md](world-light-sensing.md) + [sensor-range-clamps.md](sensor-range-clamps.md)
(the other two survival-world sensor lessons).
