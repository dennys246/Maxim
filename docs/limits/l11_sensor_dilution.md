# L11 — Sensor-count dilution and the discrimination ceiling (tracking doc)

**Ledger entry:** [README.md](README.md) §L11 · **Disposition:** MITIGATED (mitigation
designed and scheduled for 1.1.4; **not yet shipped, not yet re-measured**)
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

**Scheduled (1.1.4), and it takes both halves:**

1. **Sensor-count-scaled threshold** — `pattern_threshold = 1 − k/N` rather than a
   constant. Measured at `k = 0.30`: 100% signal separation and 100% noise rejection
   from N=6 to N=80.
2. **Per-type modality channels**, declared on the sensor schema. A sensor currently
   cannot declare its own modality — the YAML accepts `unit`, `range`, `initial`,
   `drive` and nothing else — so channel membership lives in hardcoded name tuples
   (`_EXTEROCEPTIVE_ROOT_SENSORS = ("azimuth",)`, whose own comment says it is "kept a
   named set so a future exteroceptive sensor is one entry, not a code change at the
   read site"). Declaring it groups sensors, which recovers discrimination.

**Neither is sufficient alone.** The threshold fixes detection but not the fact that two
sensors in one 50-sensor channel are confusable; grouping improves discrimination but
does not clear the fixed 0.85 bar.

**Known cost of the mitigation:** `recommend_action` sums `cluster_reward_bias`
additively across the active channel set, so the term's range grows with channel count
(±2 today, ±5 at G=5) while `min_confidence` stays 0.3. **Every added channel is a
selection-dynamics recalibration**, and nothing in CI catches it. And higher separation
allocates more clusters against an EC scan that is exact `O(N_nodes · d)` with no cap or
pruning — which makes **D51** (the degenerate LSH) load-bearing rather than dormant.

## Claim linkage

Bounds the representation behind **Exp 42** (interoception clusters), **Exp 48**
(extero/intero seam), and **Exp 53b** — whose `Re-run on:` trigger states outright
*"the representation is what transfers"*.

**It does not retract any of them.** All three ran at ≈6 drives, inside the safe band.
The limit bounds any *future* body that grows past it — which is exactly what a
Minecraft or microduck body would do.

**Consequence for the mitigation's own schedule:** shipping the threshold or the
grouping change re-stales Exp 53b **on hardware**. That re-run and 1.2's n=12 two-robot
replication are the same scarce resource and should be planned as one hardware block.

## Open questions

1. **Is `k = 0.30` right, and is a linear `1 − k/N` the right family?** One synthetic
   sweep, one value. Needs a sensitivity check before it becomes a constant.
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
5. **Where does cluster-count growth bite?** Higher separation means more clusters
   against an unbounded linear scan. Unmeasured, and it couples this limit to D51.

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
