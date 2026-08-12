# Modality resolution & alignment — the substrate's discriminability audit

**Status:** DRAFT (2026-08-11). Prerequisite for
[deferred/retrosplenial_spatial_frames.md](deferred/retrosplenial_spatial_frames.md)
(§6 step 2) and for [cross_modal_perception_fabric.md](cross_modal_perception_fabric.md).
**Origin:** the RSC pre-check measured the azimuth channel at **2 EC nodes across its
entire range** and the follow-up analysis showed that number is **mathematically forced**,
not a tuning accident.

## 1. The finding, and why it generalises beyond azimuth

`_sensor_embed` builds every sensor reading as a **two-basis linear interpolation**:

```python
(1 - v) * basis_low(name) + v * basis_high(name)
```

The two bases are uncorrelated hash-derived vectors, so a sensor's whole value range
traces a **1-D arc** between two near-orthogonal points on the sphere. Cosine similarity
along that arc falls off slowly and then collapses (measured with the real function,
`azimuth` over `[-1, 1]`):

| t (normalized) | 0.1 | 0.3 | 0.5 | 0.6 | 0.7 | 0.8 | 1.0 |
|---|---|---|---|---|---|---|---|
| cos(v₀, vₜ) | 0.994 | 0.928 | 0.744 | 0.609 | 0.462 | 0.318 | 0.075 |

Against `ECConfig.pattern_complete_threshold = 0.44`, everything from t=0 out to t≈0.7 is
"the same thing". Buckets over a full sensor range, by threshold:

| threshold | 0.44 (prod) | 0.60 | 0.70 | 0.80 | 0.90 | 0.95 | 0.98 | 0.99 |
|---|---|---|---|---|---|---|---|---|
| distinct nodes | **2** | 2 | 2 | 3 | 4 | 5 | 8 | 11 |

**This applies to every scalar sensor, not just azimuth** — the same code path encodes
cold, hunger, thermal, pressure, stamina, and every drive. The substrate can represent
roughly **one bit per sensor dimension**. Multi-sensor snapshots do better (each sensor
contributes its own basis pair, so the space is richer), but any claim that rests on
resolving a *single* continuous dimension — direction, temperature gradient, distance —
is running at 2 buckets.

## 2. The alignment problem (the deeper half)

One global threshold serves modalities with **incompatible embedding geometries**:

| modality | geometry | what 0.44 means there |
|---|---|---|
| `text` (LinguisticEncoder) | sentence-transformer embeddings, spread over a high-dim sphere | the tuned paraphrase-vs-distinct boundary — **0.40→0.44 was earned** by the EC drift work and is regression-guarded |
| `sensor` / drive (`_sensor_embed`) | 1-D arc between two hash bases | ~2 buckets over the whole range |
| `interoception`, `audio` | same arc geometry, but **frozen centroids** | no drift, still ~2 buckets |

So the threshold is simultaneously **correct for text and far too loose for sensors** —
and it cannot be fixed by moving it: reaching 5 sensor buckets needs 0.95, which would
shatter text paraphrase collapse (the exact regression the drift lesson exists to
prevent). **Resolution and alignment are one problem, and the current architecture forces
a single answer to two different questions.**

Corollaries worth stating explicitly:

- **Circular quantities are encoded linearly.** Azimuth −1.0 and +1.0 are the same
  physical direction; the arc encoding places them at maximum distance. Any angular
  channel needs a circular representation.
- **Frozen-centroid modalities freeze a coarse representation.** Freezing prevents drift
  (correct, earned) but does nothing about discriminability — an `audio` node covers half
  the range and always will.
- **Cross-modal comparison is unprincipled today.** A cosine of 0.6 in text and 0.6 in
  sensor mean different things; nothing in the code says so.

## 3. Audit protocol — run per modality, record the numbers

For each modality below, measure and record (script: extend
[`scripts/rsc_precheck.py`](../../scripts/rsc_precheck.py) into a general
`scripts/modality_resolution_audit.py`):

- **R1 Resolution** — distinct EC nodes over the input range at the production threshold.
- **R2 Threshold curve** — nodes vs threshold (the table shape above), so the
  resolution/threshold trade-off is explicit per modality.
- **R3 Boundary placement** — WHERE the bucket edges fall. Coarse-but-well-placed can be
  fine (a boundary at "centred vs off-centre" is meaningful); coarse-and-arbitrary is not.
- **R4 Alignment** — the same-similarity-means-the-same-thing check: sample within-class
  and between-class pairs and report the cosine distributions. Modalities whose
  distributions overlap the threshold differently are misaligned.
- **R5 Consumer sensitivity** — which downstream claim actually depends on this
  modality's resolution? (Exp 45/49 orient rides the drive VALUE, not cluster identity —
  so its resolution requirement is zero. Cluster-keyed *learning* about direction needs
  many buckets. Record per consumer, not per modality.)

Modalities to audit: `text`, `sensor`, `interoception`, `audio`, `drive`, plus any
`affordance`-name encoding (which routes through the LinguisticEncoder with the
affordance decomposition strategy — different geometry again).

## 4. Fix options (decide AFTER the audit, not now)

**F-A Per-modality thresholds.** Cheapest. `ECConfig` grows a per-modality override map.
Fixes *alignment* (each geometry gets its own boundary) but not *resolution* — sensors
would need ≈0.99, which is knife-edge: at that threshold, encoder noise and float drift
start creating spurious nodes, and the EC drift lesson's marginal-admission failure mode
reappears from the other direction.

**F-B Population coding for scalar/angular sensors.** Replace the two-basis interpolation
with **N overlapping tuned bases** (RBF tiling for bounded scalars; a ring code —
sin/cos or N-around-the-circle — for angular ones). Resolution becomes a *design
parameter* (N) instead of a threshold side-effect, and circular topology is handled
natively.

**This is also the bio-faithful answer, which is worth noting given the roadmap's
standards:** the brain does not encode direction as an interpolation between two
vectors. Head-direction cells, place cells and grid cells are all **population codes with
overlapping tuning curves** — precisely F-B. So the engineering fix and the bio-fidelity
fix coincide, which is rare and a good sign. It would enter the roadmap as a
**MECHANISM**-tier claim (population coding genuinely implemented) rather than a
FUNCTIONAL analogy.

**F-C Do nothing, and bound the claims instead.** Legitimate: state in the paper and the
roadmap that the substrate resolves ~1 bit per scalar dimension, and confine claims to
what that supports (binary preference — which *is* what Exp 42/44/48 rest on). Cheapest
and most honest short-term move; blocks the spatial work permanently.

Recommendation to evaluate at audit time: **F-B for angular/spatial channels only**,
leaving text and interoception untouched, plus F-A if the audit shows a second modality
misaligned. Scope discipline: do not re-encode every modality because one is broken.

## 5. Blast radius

Changing an encoding or a threshold changes **EC cluster assignment**, which is upstream
of cluster-keyed reward bias — so it touches every clustering-dependent result. The
re-validation table in
[deferred/retrosplenial_spatial_frames.md](deferred/retrosplenial_spatial_frames.md) §5
(Exp 42/42b, 45/45e, 46, 48, 49/50, 44b/44c) applies **in full** to this plan, and this
plan is the one that actually triggers it. Additional risks specific here:

- **The EC drift lesson is directly in scope.** Thresholds interact with centroid drift;
  the isolated-vs-sequential measurement discipline is mandatory for any threshold change
  (measure both; sharp disagreement = drift).
- **Frozen-modality contract.** `DEFAULT_FROZEN_CENTROID_MODALITIES` must stay in sync
  with `ECConfig` (they silently diverged once when `audio` was added — pinned by
  `test_hivemind_frozen_modalities_match_ec_default`). Any modality list change touches
  the hivemind merge path too.
- **Persisted substrate compatibility.** A changed encoder invalidates every persisted EC
  node: old `ec.json` centroids are in the old geometry. Needs an explicit
  migration-or-invalidate decision + the `encoder_provenance` stamp (already shipped) to
  detect mixed-geometry substrates.

## 6. Acceptance criteria

1. Audit table (R1–R5) filled for every modality, committed as an experiment artifact.
2. A written per-modality statement: "this modality resolves N buckets; consumers X, Y
   depend on it; that is / is not sufficient, because …".
3. Any fix ships behind a flag, default OFF, with pre-registered thresholds (never tuned
   on an outcome) and the isolated-vs-sequential drift check.
4. `scripts/rsc_precheck.py` re-run passes (H-C ≫ 2, H-B separates places) before the
   spatial channel is built.
5. Re-validation table executed or each row explicitly waived in writing.

## 7. Sequencing

1. Build `scripts/modality_resolution_audit.py` (generalise the RSC pre-check) — cheap,
   offline, no runs.
2. Fill the audit table; publish as `docs/experiments/` artifact.
3. Decide F-A / F-B / F-C **from the numbers**, in a written decision with the front-gate
   question answered.
4. Implement behind a flag; drift check; re-validation table.
5. Only then: RSC spatial frames, then cross-modal fabric.
