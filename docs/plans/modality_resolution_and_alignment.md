# Modality resolution & alignment — the substrate's discriminability audit

**Status:** DRAFT (2026-08-11). Prerequisite for
[deferred/retrosplenial_spatial_frames.md](deferred/retrosplenial_spatial_frames.md)
(§6 step 2) and for [cross_modal_perception_fabric.md](cross_modal_perception_fabric.md).
**Origin:** the RSC pre-check measured coarse azimuth resolution; a four-lens review
(2026-08-11) then corrected the number, the threshold, and the framing. **Read §0 first —
several claims in the first draft were wrong and the corrections change the conclusion.**

## 0. Corrections from the four-lens review (READ FIRST)

| first draft | corrected |
|---|---|
| azimuth resolves to **2 nodes** | **3 nodes**, boundaries at **left / centre / right** (~1.6 bits) |
| production threshold **0.44** | **0.85** — `SensorEncoderConfig.pattern_threshold`; `ECConfig`'s 0.44 is the *text* path only |
| "one global threshold serves incompatible geometries" | **FALSE** — sensor and text thresholds are already decoupled by construction |
| "multi-sensor snapshots do better" | **BACKWARDS** — static co-sensors *reduce* single-dimension resolution (3 → 1-2) |
| F-A per-modality thresholds = cheapest new option | **already shipped** (per-encoder); residual gap is that `interoception` and `audio` share one value |
| F-B population coding = option pending audit | **already prototyped and validated in-repo** — `scripts/orient_substrate/6_graded_orient_curve.py` (2026-07-22), 6/7 buckets |
| blast radius = the full 7-experiment table | **only if the shared encoder/threshold changes.** A place-coded `_read_exteroceptive_states` touches the audio channel alone |
| `encoder_provenance` detects mixed geometry | **it does not** — nothing compares it at load; only the hivemind export reads it |
| orient needs zero cluster resolution | **wrong** — orient and Exp 48 are both cluster-keyed; they need ~1 bit and get it |

**And the decisive one — this question is already SETTLED, with a behavioural result.**
[Exp 46](../experiments/46_operant_orient_creche.md) (2026-07-22) measured *"a single
azimuth scalar folds to just **2 EC clusters at every threshold (0.44→0.93)**… a
perceptual, not a learning, limit"*, applied the Gaussian place code (**6/6 distinct
clusters**), and got **taught 0.19 → 0.82, LEARNED + MOTHER-TAUGHT PASS** — then wrote
the recommendation this plan was re-deriving: *"it should be the standard direction
encoding for spatial tasks going forward."*

**Net effect: this plan should shrink to a wiring PR.** The finding is real but weaker
than drafted (~3 well-placed states, sufficient for every current consumer), the fix is
validated, and the open work is *promoting `_place_code` from script into
`_read_exteroceptive_states`* plus an Exp 48 re-run — **not** an audit programme. Two
further conflicts to settle before any of it: this plan's threshold table (4 nodes @0.90,
5 @0.95) **contradicts** Exp 46's flat-2 measurement over the same range and must be
reconciled; and [cross_modal_perception_fabric.md](cross_modal_perception_fabric.md)
Stage 0b **already owns** cluster-resolution work (`docs/plans/README.md` registers
population coding under the 1.3 fabric), so the ownership overlap must be resolved rather
than duplicated.

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

| threshold | 0.44 (TEXT path) | 0.60 | 0.70 | 0.80 | 0.90 | 0.95 | 0.98 | 0.99 |
|---|---|---|---|---|---|---|---|---|
| distinct nodes | **2** | 2 | 2 | 3 | 4 | 5 | 8 | 11 |

*(Table method: greedy first-match, ≥201 samples, frozen-centroid policy. At the
**production** sensor threshold of 0.85 the answer is **3 buckets**; the count is also
order-dependent, 2-3 across shuffles, so any re-run must report mean ± range.)*

**This applies to every scalar sensor** — the same path encodes cold, hunger, thermal,
pressure, stamina, and every drive. But two corrections to the first draft:
**multi-sensor snapshots make each individual dimension WORSE, not better** (measured:
adding 4 static co-sensors lifts `cos(az=-1, az=+1)` from 0.075 to 0.751 and drops the
sweep from 3 buckets to 2) — which is precisely the extero/intero dilution the
`ModalityChannel` split already fixed. And the isolation is production-faithful *only*
for `audio` (`_EXTEROCEPTIVE_ROOT_SENSORS = ("azimuth",)`); on the multi-sensor
interoception channel, per-dimension resolution is *lower* than the table shows.

## 2. The alignment problem — real, but far narrower than the first draft claimed

The first draft argued that one global threshold serves incompatible geometries and
cannot be moved. **That is false.** The paths are already decoupled:

| path | threshold | source |
|---|---|---|
| `text` / `vision` / affordance names | **0.44** | `ECConfig.pattern_complete_threshold` (the earned paraphrase boundary) |
| `interoception` / `audio` (all sensors, drives) | **0.85** | `SensorEncoderConfig.pattern_threshold`, passed explicitly |

`EntorhinalCortex` scans **within-modality only**, so raising the sensor threshold
provably cannot touch text paraphrase collapse. Going 3 → 5 sensor buckets is a one-field
change today.

**The genuine residual gaps, stated honestly:**

1. **Nobody chose 0.85 against measured discriminability.** Its own docstring says so:
   *"Phase 0+ work will retune this once we have cluster-purity data."* This audit is that
   data.
2. **The threshold is per-ENCODER, not per-MODALITY** — `interoception` and `audio` share
   0.85 despite different consumers and different sensor cardinality.
3. **A second, undiscussed limiter: `min_delta = 0.05`.** `SensorEncoder` short-circuits
   the EC scan entirely when no sensor moved ≥0.05, returning the cached node. On a
   `[-1,1]` azimuth that is a hard ~4.5° dead zone **regardless of encoding** — any
   population code past ~40 buckets is capped by this gate, not by geometry.
4. **Frozen-centroid modalities resolve BETTER, not worse.** Running-mean drift *costs*
   1-2 buckets at every threshold (the matched centroid migrates toward each admission and
   swallows the next reading). The first draft treated freezing as resolution-neutral.
5. **Affordance names are `text`, not a separate geometry** — only the decomposition
   strategy differs, so affordance nodes share one cluster space with text percepts at
   0.44. `"sensor"` and `"drive"` (listed in the first draft) **do not exist** in
   production; `"vision"` (text geometry) was omitted.

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

**RECOMMENDATION (post-review): F-C today.** Measured resolution is ~3 well-placed
states (left/centre/right), which is *sufficient for every current consumer* — orient
needs one bit and gets it with sign purity 1.00; Exp 48's operant discrimination is
left-vs-right; Exp 42/44 rest on categorical preference. No trigger from the RSC plan's
T1-T4 has fired, and its explicit non-trigger is "better orienting performance." Building
F-B now is elegance, which the front-gate rule forbids. **The paper-facing sentence:**
*"the substrate partitions each scalar sensor into ~3 categorical states with meaningful
boundaries; all shipped claims rest on categorical preference, which this supports."*

F-B is pre-validated (probe 6, 6/7 buckets) and should be promoted **only when a graded
consumer exists** — and the honest test of that is
`scripts/orient_substrate/6_graded_orient_curve.py` run as a pre-registered two-arm
ablation (raw scalar vs `_place_code`) on the 6-alternative graded orient where chance is
0.167. That converts "~1.6 bits" from an encoder statistic into a *behavioural* claim,
which is what the graduation ledger requires.

**On the bio-tier claim: population coding would be FUNCTIONAL, not MECHANISM.** Head-
direction cells genuinely are a cosine-tuned ring population — but place cells are sparse
and multi-field (not an RBF tiling) and **grid cells are a periodic multi-scale modular
code, not RBF at all**, so citing them inverts the mechanism. What makes the HD system a
*mechanism* is the **ring attractor** — recurrent excitation + global inhibition, bump
persistence without input, angular-velocity integration, landmark correction — none of
which comes free with an RBF encoding. Claiming MECHANISM for a tuning-curve encoding
would also contradict the roadmap's existing EC row, which holds attractor dynamics as
deliberate ANALOG.

## 5. Blast radius — CORRECTED, and the first draft's coverage was close to inverted

The two candidate changes have **almost disjoint** blast radii, and the first draft
applied the spatial table to both.

**A change to the AUDIO/spatial channel** (place-coding `_read_exteroceptive_states`)
touches only:
- **Exp 48** — its entire claim is *which* EC cluster operant credit lands in; its ledger
  trigger names the ModalityChannel registry verbatim. Re-run required.
- **Exp 46** — is itself a cluster-count measurement; its *finding*, not just its number,
  changes.
- **Exp 49 H3 + arm C only** — H1/H2 are EC-free.

**FALSE POSITIVES in the first draft — drop them.** Exp 45/45e and Exp 50 **never call
EC**: the orient backbone builds state from `az_bin(...)`, a hand-written bin string
passed as `current_cluster_id` (`scripts/orient_backbone/live_3_learn.py`), and imports
NAc only. Exp 42/42b runs `infant_humanoid_chilled`, which declares **no azimuth sensor**,
so the audio channel is inactive; it is at risk only from an *interoception-side* change.
Scheduling hardware sessions for 45/50 on this basis would have been wasted.

**Sharper consequence of the same fact:** `az_bin` is a *hand-curated discretisation
upstream of the substrate* — the interim-contamination pattern. So **Exp 45's Earned
status does not transfer to any EC-clustered orient policy**, which is exactly what the
RSC/fabric work would build. Resolution work cannot be validated by re-running Exp 45; it
needs its own experiment.

**A change to the TEXT threshold (F-A)** hits an entirely different and much larger set
that the first draft listed **none** of: Exp 24, 25, 26 (*this doc IS the 0.44 tuning*),
27, 28, 20 (Roy-2c), 22 (Roy-5a), 35/36, plus the P1/P2 sweeps — and
`test_ec_centroid_drift_fix.py` asserts the 0.44 default directly.

**MISSING row 1, in both drafts:** the Earned Tier-1 ledger row **"EC pattern completion /
separation"** — the one Earned row whose subject *is* the mechanism being changed, and the
only one with **no `Re-run on:` and no `Regression guard:` field**. Give it both, then
list it first. Also missing: **Exp 37/38** (trigger names Wire-A/NAc-bias refactor, and
Wire-A *is* the cluster-bias annotation), **affordance concept transfer** (an EC-node-
identity claim in text space), and **Exp 43**; **Exp 47** flagged re-check-on-adoption;
**Exp 10** waived in writing.

**Structural consumers the first draft missed entirely:**

- **`min_confidence` needs RE-CALIBRATION, not just a re-run.** `recommend_action` sums
  `cluster_reward_bias` **additively across active channels**; the code carries an
  in-line warning that adding a channel is a selection-dynamics change. A `space` channel
  is a third ±1 term against a fixed reward_bias ≤0.20.
- **A place code DEFEATS the hivemind merge dimension-guard.** `_cosine` returns 0.0 on
  dim mismatch — but a place code keeps `dim=384` and the same `"audio"` tag, so
  old-geometry and new-geometry nodes merge whenever partial cosine ≥ 0.44. Because
  `audio` is frozen, the corruption is **invisible**: the centroid never moves, only
  counts and contributors inflate.
- **`ec_merge` cannot express per-modality thresholds** (`cosine_threshold: float = 0.44`
  hardcoded, pinned by no test, and the layer refuses internal imports) — F-A would
  silently misalign local clustering from merge clustering.
- **`nac_merge` never folds cluster biases across agents** (keys are
  `agent\x1fcluster\x1ftool`; UUIDs never collide). Session-relative bearings therefore
  make the RSC plan's T2 (multi-vantage identity across agents) **unreachable without a
  world anchor** — the mechanical answer to that plan's open question.
- **`bio_enrichment` mutates text centroids on a READ path** (deliberate reconsolidation,
  ~1/(n+1) per query), so querying degrades text resolution over time.
- **Hebbian binding is already inert**, independent of resolution: `memory_hub` stashes a
  **1-tuple** of substrate nodes and `apply_hebbian_on_close` returns early on `< 2`.

**Persisted-substrate reality (corrected):**
- `encoder_provenance` **detects nothing** — write-only and export-only; its sole readers
  are the hivemind bundle/CLI export, and merge explicitly *severs* it. It would
  discriminate a caller-side place code (`sensor_names` changes) but never a core
  `_sensor_embed` geometry change.
- Node ids are `uuid4()`, content-independent, and `EC.save()` records **no threshold
  regime** — so F-A is entirely invisible in on-disk state.
- The decision must be **invalidate-both-in-lockstep**, not "migrate or invalidate":
  old-geometry centroids never match new embeddings, so NAc's persisted
  `cluster_reward_bias` triples dangle until wall-clock decay while learning silently
  restarts at zero. The NAc/EC pair invariant guarantees both are *present*, not that they
  are *geometrically coherent* — this is a new failure mode it does not cover.
- **There is no working invalidate command:** `MEMORY_PATHS` has no `ec` key, so `maxim`
  cannot delete `ec.json` at all (not even under `all`). Shipping one is a prerequisite.

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

**REVISED after the review — the audit programme is mostly unnecessary.**

1. **Reconcile with Exp 46 and CMPF Stage 0b first** (ownership + the contradictory
   threshold table). This may reduce the whole plan to step 3.
2. **Re-run the pre-check against production** (`modality="audio"`, frozen, 0.85,
   isolated *and* sequential, shuffled orders) so the recorded number is production-true.
3. **The actual open work:** promote `_place_code` from `scripts/orient_substrate/6` into
   `_read_exteroceptive_states` behind a flag — caller-side only, so the blast radius is
   the audio channel — plus `min_confidence` recalibration, the merge dim-guard question,
   an `ec` invalidate command, and an Exp 48 re-run.
4. Register this plan in `docs/plans/README.md` and the roadmap (it is currently in
   neither), and refresh the roadmap's stale *"a cheap pre-check may close it outright"*
   line.
5. F-A (text threshold) is a **separate** decision with a **different, larger** blast
   radius — do not bundle it.
