# L11 — Sensor-count dilution and the discrimination ceiling (tracking doc)

**Ledger entry:** [README.md](README.md) §L11 · **Disposition:** MITIGATED (mitigation SELECTED by bake-off
2026-09-01 — the nonlinear gain, arm A4; **not yet shipped, not yet re-measured on a real body**)
**Instrument:** `similarity/encoder.py::_sensor_embed` → EC
`pattern_complete_or_separate` at `SensorEncoderConfig.pattern_threshold = 0.85`.
Applies to every substrate modality channel.

> Graduated to its own doc under README rule 6 (≥2 measurements **or** active work):
> it has seven distinct measurements and is on the 1.1.4 critical path.

## The limit, precisely

Two limits, and the second is the one that matters.

**Detection** — *can the substrate see that the state changed?* Each sensor contributes
`(1-v)·basis_low + v·basis_high` to a **sum**, so one sensor is 1/N of the result.
Cosine already normalises length, so the ratio is what degrades:

> **cos ≈ 1 − 0.57 / N**

Clean 1/N from N=1 to N=200. At N ≥ 15 a full single-sensor swing no longer clears the
fixed 0.85 threshold.

**Discrimination** — *can it tell **which** sensor changed?* At N=100, two entirely
different sensors going to extremes read **cos 0.990**. This is the deeper limit:
detection is recoverable by moving the threshold, discrimination is not. A substrate
that cannot tell *which* dimension moved cannot learn "turn_left helps when the sound is
left" — it can only learn "something is happening."

**Why it matters beyond the number.** `cluster_reward_bias` is keyed on the cluster id.
If two meaningfully different body states complete onto one cluster, they share a bias —
the agent cannot hold different policies for them, no matter how much it learns.

## Measurement history

| date | what | result | source |
|---|---|---|---|
| ~2026-07 | extero/intero dilution, ~6 drives | single-sensor swing leaves cos ≈ 0.83 vs threshold 0.85 | [exteroception_interoception_seam.md](../plans/archive/exteroception_interoception_seam.md); guarded by `tests/unit/test_modality_seam.py::TestDilutionRegression` |
| 2026-09-01 | scaling law, N=1…200 | **cos ≈ 1 − 0.57/N**, confirmed 1/N | synthetic sweep over the shipped encoder |
| 2026-09-01 | signal vs noise | signal 0.119 → 0.006 (N=4→100); all-sensor 2% jitter **flat at ~0.0008**; SNR 185:1 → 7.5:1 | ” |
| 2026-09-01 | **discrimination** | cos(sensor A spikes, sensor B spikes) = 0.845 (N=6) → **0.990 (N=100)** | ” |
| 2026-09-01 | scaled threshold | `1 − 0.30/N` → **100% signal separation AND 100% noise rejection, N=6…80** | ” |
| 2026-09-01 | grouping into G channels (50 sensors) | discrimination 0.980 (G=1) → 0.897 (G=5) → **0.831 (G=10)** — helps, does **not** clear 0.85 alone | ” |
| 2026-09-01 | embedding dimension | 384 → 3072 moves cosine by **< 0.001**, marginally worse | ” |
| 2026-09-01 | sparse/hashed bases (k=16 dims/sensor) | **identical to the plain sum** | ” |
| 2026-09-01 | distributional moments | detection 0.63 (N=6) → **0.27 (N=100)** — N-independent — but discrimination **0.999**, and worse at every mixing weight | ” |
| 2026-09-03 | bake-off at N=6/8 (infant-scale channels; same frozen arms) | A4 stability COLLAPSES at N=6 (0.62; primary 0.62 vs A1's 0.88) and is not the best arm below N=12 — the noise-floor cost the pairwise data predicted. A4 is a many-sensor tool | `scripts/encoding_bakeoff.py --sensors 6,8`; data `docs/experiments/data/encoding_bakeoff_n6_n8_2026-09-03.json` |
| 2026-09-03 | gain over the audio place code (7 directions, real EC) | separation stays 7/7 but jitter-stability degrades 1.000 → 0.957 (gain crushes the interpolating intermediate activations); a gained RAW azimuth zero-vectors the CENTERED reading | `scripts/place_code_gain_check.py`; data `docs/experiments/data/place_code_gain_check_2026-09-03.json` |
| 2026-09-03 | scan cost at A4 allocation (mixed-modality store, real EC) | p95 ≈291 ms at the projected 4h-session store (8,375 nodes); 5 ms crossing ≈238 nodes → **index-prerequisite**; vectorized exact scan 0.89 ms @ 20k-node store | `scripts/ec_scan_cost.py`; data `docs/experiments/data/ec_scan_cost_2026-09-03.json` |

All 2026-09-01 rows are synthetic sweeps over the **shipped** encoder, recorded in
[minecraft_benchmark.md](../plans/minecraft_benchmark.md) §"The sensor ceiling is a
THRESHOLD artifact".

## What raises the ceiling (mitigation lineage)

**Ruled out, measured — recorded so they are not re-proposed:**

| candidate | verdict | why |
|---|---|---|
| more embedding dimensions | **no** | dilution is an *averaging* problem, not a *capacity* one. 384 is already ample; `dim` matters only as a floor (must stay ≫ N for near-orthogonal bases) |
| sparse / hashed / randomised bases | **no** | identical to the plain sum. No basis trick escapes summing N terms and comparing the sum by cosine |
| distributional moments (mean/sd/skew/kurtosis/max-dev) | **detection only** | N-independent for detection, but **permutation-invariant by construction** — a moment cannot say *which* sensor moved, and adding it makes discrimination worse at every weight. No choice of moments fixes this; it is what a moment *is* |

**SELECTED (1.1.4), by bake-off — see §Bake-off below:** the **nonlinear gain** (arm A4),
weighting each sensor's contribution by its distance from set point, at the **unchanged**
0.85 threshold. Perfect on all three criteria from N=30 to N=100. The principle is that a
sensor resting at its set point should not be shouting — which is what the comfort-band
drive design already encodes.

**Membership is per-modality and measured, not global (PR 1, 2026-09-03):**
`SensorEncoderConfig.gain_modalities = {"world"}`. The N=6/8 rows above show A4's
stability collapsing 0.97 → 0.62 at N=6 and losing to A1 (primary 0.62 vs 0.88) — though
still above the control A0's 0.37 on the frozen primary; the grounds are the stability
collapse and the re-stale cost, not "worse than shipping" — and the place-code row shows it
degrades audio. So interoception and audio stay in the pre-A4 space, byte-identical (their
geometry tags do not move; test-pinned), and no EARNED row re-stales. "world" is additionally
FROZEN-CENTROID from birth (plan decision D6): the membership evidence was measured under
frozen semantics, and an unfrozen first-running-mean sensor modality at ~120× allocation is
an unmeasured drift configuration. Flipping a modality in later is a geometry change + graduation-trigger event.

**Superseded, and recorded because it was the standing recommendation until measured:**

1. ~~Sensor-count-scaled threshold `1 − k/N`~~ — scores 0.70–0.84, degrading with N.
2. ~~Per-type modality channels~~ — near-useless alone (0.00 at N=100).
3. ~~Both together~~ — **measured WORSE than the threshold alone** (0.62 vs 0.76 at N=50),
   because grouping shrinks per-channel N, which loosens `1 − k/N` and lets noise
   separate. This was the pre-bake-off recommendation and it was wrong.

Per-type channels remain worth doing for a **separate** reason — a sensor should be able to
declare its own modality rather than channel membership living in hardcoded name tuples
(`_EXTEROCEPTIVE_ROOT_SENSORS = ("azimuth",)`, whose own comment calls itself "kept a named
set so a future exteroceptive sensor is one entry, not a code change at the read site"), and
1.1.4 has to generalise that tuple anyway. Just not as this limit's mitigation.

**Known cost of the mitigation:** A4 allocates ~120× the control's clusters against an EC
scan that is exact `O(N_nodes · d)` with no cap or pruning — **a scan-cost prerequisite,
initially mis-filed as D51** (corrected 2026-09-03: `pattern_complete_or_separate` never
consults `LSHIndex`; the cost lands on the exact `_substrate_nodes` scan, which has no index
of any kind, and production shares ONE EC across channels so per-encode cost scales with the
TOTAL store. Measurement + frozen decision rule: `scripts/ec_scan_cost.py`,
[docs/plans/world_seam_1_1_4.md](../plans/world_seam_1_1_4.md) decision D4). Separately, if per-type channels ship for their own reasons,
`recommend_action` sums `cluster_reward_bias` additively across the active channel set, so
the term's range grows with channel count (±2 today, ±5 at G=5) while `min_confidence`
stays 0.3 — every added channel is a selection-dynamics recalibration and nothing in CI
catches it.

## Bake-off (2026-09-01) — the mitigation choice, measured

Six arms against the **real** `EntorhinalCortex`, metric frozen before the runs
(`scripts/encoding_bakeoff.py`; data `docs/experiments/data/encoding_bakeoff_2026-09-01.json`).
PRIMARY = `min(separation, stability, discrimination)` — the weakest link, so no arm can
buy one criterion by sacrificing another. Economy (clusters per 100 states) is reported
alongside as a **cost**, never folded in.

| arm | N=12 | N=30 | N=50 | N=100 | economy @N=50 |
|---|---|---|---|---|---|
| A0 current (control) | 0.20 | **0.00** | **0.00** | **0.00** | 0.5 |
| A1 scaled threshold | 0.82 | 0.84 | 0.76 | 0.70 | 63 |
| A2 grouping only | 0.63 | 0.30 | 0.04 | **0.00** | 3 |
| A3 threshold + grouping | 0.63 | 0.56 | 0.62 | 0.70 | 82 (160 @N=100) |
| **A4 nonlinear gain** | **0.94** | **1.00** | **1.00** | **1.00** | 58 |
| A5 gain + threshold | 0.28 | **0.00** | **0.00** | **0.00** | 100 |

**A4 — the nonlinear gain alone, at the UNCHANGED 0.85 threshold — wins at every N**, and
is perfect (1.00/1.00/1.00) from N=30 up.

**Three results that overturn the pre-bake-off recommendation, which was A3.**

1. **A3 is worse than A1 alone** (0.62 vs 0.76 at N=50) and has the worst economy of any
   arm at N=100 (160 clusters per 100 states). Combining the two mitigations *hurts*:
   grouping shrinks per-channel N, which loosens `1 − k/N`, which lets noise separate —
   stability drops to 0.56–0.62. The pre-registered recommendation to ship both was
   **wrong**, and pairwise synthetic measurement did not reveal it. Only the full metric
   on the real EC did.
2. **A5 (gain + threshold) is actively harmful** — stability collapses to 0.00. The gain
   already separates on its own; tightening the threshold on top makes noise separate too.
3. **A2 (grouping alone) is close to useless at scale** — 0.00 at N=100, confirming the
   earlier finding that grouping does not clear the fixed bar by itself.

**The cost is real and it promotes a dormant defect.** A4 allocates ~58 clusters per 100
states against the control's 0.5 — roughly **120×**. The EC scan is exact
`O(N_nodes · d)` with no cap and no pruning, so **the scan itself becomes the prerequisite
for a large-sensor body** (initially mis-filed as D51 — see the correction under §"What
raises the ceiling"; `LSHIndex` is not on this path). That trade was the single most
important open item below, now MEASURED — see open question 5.

**Scope, unchanged and binding:** synthetic bodies with uncorrelated SHA bases and iid
noise, because no shipped body exceeds ~12 sensors — the regime does not exist yet. Real
drives correlate. This says *which candidate is worth building*, not *which is validated*.

**A harness note worth keeping.** The first run reported every arm at stability 0.00 with
an empty node store. The cause was that `pattern_complete_or_separate` deliberately
allocates an id **without registering it** — its own comment says the caller registers via
`register_substrate_node` "after ATL activation succeeds", keeping EC stateless on the
separation path. Using the real component is not the same as using the real *protocol*; a
harness that skips the second half measures an EC that never remembers anything.

## Claim linkage

Bounds the representation behind **Exp 42** (interoception clusters), **Exp 48**
(extero/intero seam), and **Exp 53b** — whose `Re-run on:` trigger states outright
*"the representation is what transfers"*.

**It does not retract any of them.** All three ran at ≈6 drives, inside the safe band.
The limit bounds any *future* body that grows past it — which is exactly what a
Minecraft or microduck body would do.

**Consequence for the mitigation's own schedule (NARROWED 2026-09-03 by the world-only
membership):** the shipped A4 does NOT re-stale Exp 53b — its channels (interoception +
audio) are byte-identical, decision-equivalence-guarded, and the fired triggers are
discharged with a dated annotation on the ledger row. What re-stales 53b is FLIPPING an
existing modality into `gain_modalities` later; that re-run would then join the hardware
block. The block that remains: this limit's post-mitigation re-measure on a real body +
the owed roll/pitch recalibration + 1.2's n=12 two-robot replication — same scarce
resource, plan as one.

## Open questions

1. **~~Is `k = 0.30` right?~~ Superseded by the bake-off** — the scaled threshold is no
   longer the selected mitigation. The tuning question survives only if A1/A3 are
   revisited; note the bake-off showed `k = 0.30` is clearly mis-tuned once grouping
   shrinks per-channel N (A3 stability 0.56–0.62).
1a. **What is the right gain exponent?** A4 used `p = 3.0`, one value, unswept.
2. **Do real sensors behave like the synthetic ones?** These measurements use
   uncorrelated SHA bases and iid noise. Real drives correlate — hunger and fatigue
   drift together — which changes the geometry. Confirm on a real body before relying
   on the numbers.
3. **What is the right grouping?** Per-modality is the obvious cut; whether interoceptive
   drives should further split (thermal / nutritive / nociceptive) is unmeasured.
4. **Does the nonlinear-gain variant beat grouping?** A gain rising with distance from
   set point held signal essentially flat across a 16× sensor increase (0.79 → 0.73) but
   with a higher noise floor, and it is *worse* than the plain sum below N≈30. Not
   scheduled; recorded as a live alternative.
5. **Where does cluster-count growth bite? — now the top open item.** A4 costs ~120× the
   control's cluster allocation. The EC scan is exact `O(N_nodes · d)`, uncapped and
   unpruned (measured elsewhere: 2.7 ms @ 100 nodes, 136 ms @ 5,000 — per encode, per
   channel, per tick). ~~D51 is therefore a prerequisite for A4~~ **Corrected 2026-09-03:
   the prerequisite is the `_substrate_nodes` scan itself — `LSHIndex` (D51) is not on
   this path and fixing it would not help. Measured at A4's allocation rate by
   `scripts/ec_scan_cost.py` (frozen decision rule in its docstring; verdict recorded in
   [docs/plans/world_seam_1_1_4.md](../plans/world_seam_1_1_4.md)).** MEASURED 2026-09-03,
   verdict **index-prerequisite**: the Python scan crosses 5 ms at ≈238 nodes vs a projected
   8,375-node 4-hour-session store (p95 there ≈291 ms); the chosen remedy is a vectorized
   EXACT scan (p95 0.89 ms @ a 20k-node store), shipping with A4. Data:
   `docs/experiments/data/ec_scan_cost_2026-09-03.json`.

## Re-measure on

- `similarity/encoder.py::_sensor_embed` change (the encoding equation)
- `SensorEncoderConfig.pattern_threshold` change — **including shipping the scaled
  threshold**, which is what moves this entry toward `RETIRED`
- `_SUBSTRATE_CHANNELS` count change, or a new `ModalityChannel`
- any body whose **per-channel** sensor count exceeds ~12
- EC `pattern_complete_or_separate` change

> **Retirement condition.** This entry moves to `RETIRED` only when the mitigation is
> **shipped AND re-measured on a real body** at a per-channel count above the current
> safe band. Shipping alone is not sufficient — the 2026-09-01 numbers are synthetic.
