# Cluster separation is a DIRECTION problem, not a magnitude problem (the deep form of L11)

**Established:** 2026-09-15, across the Exp 58 → L11 diagnostic → Slice-2-rejection → Exp 60 arc.
This is the generalization that L11 (`docs/limits/l11_sensor_dilution.md`,
[cluster-dilution-blocks-situation-fear.md](cluster-dilution-blocks-situation-fear.md)) was really
about. Read this before designing ANY situation-keyed contingency (fear, reward, place) on a summed
sensor channel.

## The fact

The substrate encodes a modality by summing per-sensor SHA-basis vectors and comparing situations by
**cosine**. Two situations merge into one cluster when their cosine exceeds the `0.85`
pattern-completion threshold. The A4 gain weights each sensor `w = (|v − 0.5|·2)^p` (p=3.0 for
`world`).

**Cosine is scale-invariant — it sees only DIRECTION, not magnitude. The A4 gain is a MAGNITUDE
weight. So the gain cannot, by itself, make two situations separate.** What separates two situations
is one (or several) sensors whose normalized value **rotates the summed vector** between them — i.e.
moves across a large *directional* arc. What fails to separate them is a sensor that moves only a
little, or moves while staying on the same side of its neutral point.

## What this means concretely (measured)

- A sensor resting at neutral (0.5) contributes the **zero vector** (weight 0). A sensor at an
  **extreme** (0.0 or 1.0) contributes full weight along its low/high basis. So the big separating
  moves are **neutral → extreme** or **extreme → opposite extreme**.
- A sensor that moves a *little on one side of neutral* barely rotates the sum, no matter how much
  gain mass it carries. Exp 58's `nearest_hostile_dist` moved 0.179 → 0.087 (both below 0.5): it
  carried real gain mass (0.27 → 0.56) yet the pair stayed at cos ≈ 0.99. Isolating it into its own
  channel made separation **worse** (0.977 → 0.991), because a channel of one small-arc mover is
  *more* collinear, not less. Re-massing (dropping the constant sensors) also failed (0.991). Verified
  offline on the real encoder bases: `docs/experiments/data/l11_slice2_cosine_check.py`.
- Contrast `oxygen` in Exp 60: rest 20 → normalized 0.5 (silent), drowning → 0.0 (full weight, a new
  basis direction that was *absent* at rest). That neutral→extreme swing separates cleanly
  (cos ≈ 0.78). This is why drowning is a viable cue where dark=danger was not.

## Corollaries for design

1. **The gain-weighted per-sensor "mass" metric predicts CONTRIBUTION, not SEPARABILITY.** The L11
   geometry probe's gain-weight table (`scripts/survival_world/l11_geometry_probe.py`) told us
   `nearest_hostile_dist` was the "live contributor" — true by mass, false as a predictor of
   separation. A separability diagnostic must compute the actual **cosine** between the situations
   (and, if per-sensor attribution is wanted, a *directional* decomposition), never gain-weight mass
   alone. The cosine number is the truth; the mass table is colour.
2. **For a contingency that must key on a situation through a whole pre-event WINDOW, use a binary
   in-state sensor, not the gradient that only crosses threshold at event onset.** Exp 60: `oxygen`
   alone only separates the underwater cluster at oxygen≈0 (= drowning-damage onset), so the ~15s
   pre-damage window read as the shore cluster — the anticipation window was empty. A binary
   `is_in_water` (range `[-1,1]`, rest 0 = neutral/silent, submerged 1 = full-weight) makes the
   underwater cluster distinct from dive-second-0 (measured cos ≈ 0.79 at full air). A binary
   state-flag at rest-neutral is the clean way to get a *stable* situation cluster.
3. **Before building any substrate remedy, REPLAY it offline on real captured vectors through the
   real encoder.** Both the Slice-2 rejection and the Exp 60 go-ahead were decided by a ~30-line
   offline cosine computation on vectors already captured (`docs/experiments/data/*_cosine_check.py`),
   for near-zero cost, before any substrate code. A remedy that can't clear the threshold on the real
   captured vectors will not clear it live — falsify it cheaply first.
4. **Adding a rest-at-neutral sensor is SAFE; re-tagging or changing an existing sensor is
   DESTRUCTIVE.** The embedding dimension is fixed (it does not grow with sensor count), and a sensor
   resting at neutral adds the zero vector — so adding `is_in_water` left every dry-scenario encoding
   byte-identical and invalidated no persisted substrate. By contrast a modality re-tag (Slice-2)
   changes the geometry/roster and **orphans persisted EC nodes** — and `maxim substrate invalidate
   --drop-geometry` only migrates by geometry, NOT by modality, so re-tagged nodes (and their NAc
   biases) are destroyed, not migrated. Prefer additive, rest-neutral sensors over re-tagging.
5. **If a real contrast still won't separate, the block is APPARATUS + REPRESENTATION, not the
   channel grouping.** No amount of channel-splitting or thresholding rescues a near-collinear pair.
   The honest fixes are upstream: an apparatus whose cue swings a sensor full-range across neutral, or
   the deferred **set-point-aware neutral** substrate primitive (`docs/plans/setpoint_aware_neutral.md`,
   `_sensor_embed` decision D1) that makes contribution relative to a sensor's rest value so small
   off-baseline moves rotate the embedding.

## See also

`docs/limits/l11_sensor_dilution.md` (the 1/N mass law — necessary background, but this direction
framing supersedes "it's dilution" as the operative mental model);
[cluster-dilution-blocks-situation-fear.md](cluster-dilution-blocks-situation-fear.md) (the live
Exp 58 realization); `docs/plans/l11_slice2_channel_split.md` §Decision (the rejected remedy + the
verified table); `docs/experiments/exp60_drowning_avoidance_prereg.md` (the viable cue).

**Embodiment gotcha noted here so it isn't lost:** the mineflayer pathfinder is DEAD in water — its
move generators hard-return on liquid nodes, so `flee`/`goto` throws `NoPath` from any submerged
start. Water actuation must bypass the pathfinder (`bot.setControlState("jump", true)` = swim up),
as `escape_water` does. (Exp 60 environment lens, verified in the vendored pathfinder source.)
