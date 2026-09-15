# L11 world-channel diagnostic — measure the live cluster geometry before any substrate fix

**Status: PLAN DRAFT v2 (2026-09-14) — three-lens concept review FOLDED
(`docs/plans/rationale/l11-diagnostic/{measurement-validity,substrate-faithful,wiring}.md`).**
Motivated by Exp 58's block: the survival danger/safe situations do not form distinct world
clusters on the ~17-sensor `world` channel (L11 dilution, live —
`docs/experiments/exp58_survival_wants_prereg.md` §Outcome;
`docs/wiring/cluster-dilution-blocks-situation-fear.md`). This is **Step B** of the L11 approach
(A = bank the finding, done; B = this diagnostic; C = build the remedy the diagnostic picks).
**B GATES C**: no substrate change is built until the diagnostic proves, on a *live re-encode*, that
a remedy separates the real situations — and its effect on Exp 56/57 is established, not assumed.

## Slice-1 RESULT (2026-09-15) — measured, `diluted_present`

Ran on the live classroom (16 world sensors, A4 p=3.0, 30 samples/situation;
record `docs/experiments/data/l11_geometry_2026-09-15.json`). **cos(safe,dark) = 0.977 (A4) /
0.996 (A0)**, fresh-EC ids not distinct → verdict **`diluted_present`**. The measurement
corrected the pre-registration's leading hypothesis:

- The discriminator is **not** gain-silenced. `nearest_hostile_dist` is the lone live
  contributor (gain weight 0.27→0.56; the clustermob ~22 blocks → adjacent). The gain passes
  it through (A0 0.996 → A4 0.977) but can't clear the threshold.
- The dilution is **constant-sensor mass**: `light_level` (weight 1.0 both, Δ0) and
  `time_of_day` (0.77 both, Δ0) carry maximal mass, zero contrast — they out-vote the one
  working sensor and pin the cosine near 1.
- **Depth is nearly static** (`y_altitude` Δnorm 0.037, below move threshold) — the Exp 58
  discriminator barely moves; hostile-adjacency carries the contrast.

**Refined Slice-2 aim (was "isolate depth/threat"):** a per-type channel-split that puts the
*discriminating* sensors (`nearest_hostile_dist`, `hostile_count`, `y_altitude`,
`distance_from_spawn`) in a small-N threat/spatial channel, away from the constant
environmental sensors that dominate the full-channel sum. `1−k/N` scaled threshold is the
secondary arm. Both replay through production code and must pass a LIVE re-encode past the
Exp 58 cluster-distinct preflight (§5) — Slice 1 only nominated.

## Why a diagnostic first (the lesson)

The Exp 58 apparatus saga was *reasoning about* the fix (light → depth → depth+hostile) and building
each, when we should have measured the cluster geometry first. Worse, the offline gates
(`survival_phase0`, exp58 gates) reported 1.0/1.0 separability because they swung light AND altitude
full-range together — they **validated an easier problem than the live classroom**. This diagnostic
measures the REAL contrast at its REAL magnitude through the REAL encoder, and picks the remedy from
that. **The review caught that the first draft would repeat that trap one level up** — the four
corrections below are load-bearing.

## What the review corrected (fold summary)

1. **The `world` modality is FROZEN-CENTROID / first-touch, not running-mean drift** (all three
   lenses). Live clustering is first-touch prototype allocation scanned against *all* stored world
   nodes; centroids do not move. So (a) a pairwise `cos(safe,dark) < 0.85` is *necessary but not
   sufficient* and errs both ways (dark can complete onto an intermediate/cross-traffic cluster;
   one low-contrast sample can read merged while id-sets are disjoint), and (b) the sequential
   replay must stay frozen — an unfrozen replay measures a world that doesn't exist.
2. **The replay must run through PRODUCTION encoder code, never a mirror** (all three lenses). The
   bake-off tool I first cited (`scripts/encoding_bakeoff.py`) hand-copies `_sensor_embed` and does
   round-robin **stride** grouping, not the **per-type** split — it would measure a different remedy
   than C ships. Build instead on `scripts/l11_real_trace_remeasure.py::analyze`, which already
   replays live traces through the shipped `SensorEncoder.encode_sensors(modality="world")` against a
   real EC ("never a mirror").
3. **The sole BUILD verdict is a LIVE re-encode past the exact cluster-distinct preflight that
   blocked Exp 58** (measurement + wiring). Offline replay only *nominates* a candidate.
4. **The A4 gain is the prime suspect, and only a gain-weighted metric can see it** (substrate).
   `world ∈ gain_modalities` (cubic gain p=3.0) already, so the Exp 58 merge happened *with* the
   mitigation active; the discriminating sensors (`y_altitude` Δ≈0.09, `nearest_hostile_dist`) swing
   partially and sit near the gain's neutral 0.5, exactly where `(|v−0.5|·2)³` silences them.

## The question it answers

> On the live survival classroom, what is the actual frozen-centroid cluster-ID assignment for the
> situations the experiment keys on; which sensors carry (or, under the A4 gain, fail to carry) the
> contrast; and **which L11 remedy — if any — makes the danger/safe situations land in distinct
> clusters on a LIVE re-encode, scored by the bake-off's composite `min(separation, stability,
> discrimination)`, without a knob tuned to this one sample?**

Output is a **provenance-stamped decision record**, not a behavioural claim: a nominated remedy that
PASSED a live re-encode, or "none separates these → the situations are genuinely too similar / the
contingency is infeasible on this substrate." A separation-only pass never authorizes a build.

## Pre-registration (frozen BEFORE any capture — the measurement-validity gate)

To stop the table being read post-hoc to bless the pre-named favourite (the D43/D44 "measured a
possibility, presented as proof" shape), freeze before capturing a single vector:

- **Situations** the experiment keys on: `safe` (upper chamber), `dark` (deep pit), gradient
  (cave-mouth / mid-stairs / deep), each × {hostile present, absent} × {health full, hurt}.
- **Sample size** per situation (≥ enough to bound the live jitter; Exp 58 Addendum 4 already saw one
  spot split 6/4 across two cluster ids — so N and a boundary-crossing CI are pre-set, not chosen
  after).
- **The bar, verbatim from the bake-off:** composite `min(separation, stability, discrimination)`
  with numeric thresholds — NOT separation alone. `docs/limits/l11_sensor_dilution.md` already scored
  grouping+threshold *worse and unstable* (small per-channel N lets noise separate); a
  separation-only diagnostic would re-select exactly that noise.
- **D1 guard:** no sensor-range narrowing or gain/threshold value chosen *because* it passes this
  sample. The stability + discrimination legs of the composite are the mechanical guard against
  tuning-to-apparatus.
- **The apparatus is a stop-rule**, pinned to the Exp 58 classroom fingerprint (the remedy → apparatus
  → contrast circularity means C must not silently invalidate what B measured).

## Components

### 1. Read-only telemetry — use the seam that already exists
Do **not** hand-roll a tap. `SensorEncoder`/EC already expose `pattern_complete_readonly` (D8),
structurally non-mutating (the naive `pattern_complete_or_separate` mutates even for frozen `world`:
node-count increment + first-touch stamp). Add, if needed, one pure read-only method **on
`SensorEncoder`** that reads `self.config` (so it can't drift from production), returns the
gain-applied embedding + the frozen-centroid cosines + the pattern-complete *decision* and cluster
id, and touches **no** side-effect state: not the `min_delta` delta-gate stash (it can return a cached
node or gate out a live encode), not the NAc eligibility trace, not `register_substrate_node`.
Snapshot the sensor dict once (the sync pump writes on a background thread).

### 2. Live vector capture — at the encoder INPUT, not the raw bridge dict
Capture through `runtime/agent_loop.py::_read_world_states` + `_read_world_ranges` — the exact
range-normalized/clamped ~17-sensor `world` vector production encodes — **not** raw `latest_state()`
(the bridge emits ~18 keys incl. drive keys that pollute the channel, unclamped). Enumerate the
`modality: world` sensor set **at runtime** (docs disagree 16 vs 17; derive it, don't hard-code).
Capture the pre-registered situation grid, multiple samples each → distributions, not points.

### 3. Geometry metrics — frozen-centroid, and gain-weighted
- **Cluster-ID assignment** (the real DV), not just pairwise cosine: replay the captured vectors
  through the frozen-centroid scan against a real EC seeded with cross-traffic, and report whether
  safe/dark land in **distinct ids** — plus the cosine margin vs `0.85` as a secondary readout.
- **Per-sensor contribution, GAIN-WEIGHTED** `(|v−0.5|·2)³` (not raw-normalized delta): this is where
  the answer likely hides — a sensor that moved but sits near neutral contributes ≈nothing under A4.
  This metric is what lets the diagnostic *see* whether the signal is diluted, gain-silenced, or
  genuinely absent.
- **Isolated vs sequential.** Isolated = the pure tap's direct embedding cosine with **no EC**
  (`make_fresh_encoder` reuses `aut.bio.ec`, so situation A's nodes would leak into B — don't use it
  for the isolated arm). Sequential = real EC, kept **frozen**. The real sequential hazard here is
  first-touch prototype + separation-cascade (arrival order), not drift.

### 4. Remedy replay — through production code, each arm carrying its prior verdict
Replay the SAME captured vectors on `l11_real_trace_remeasure.py::analyze` (real `SensorEncoder`):
- **A4 gain-exponent** sweep and **`1−k/N` scaled threshold** — both have real params; replay live via
  `_encode_current_clusters`. **Cite each arm's prior bake-off rejection beside it** (grouping 0.00 @
  N=100; A3 grouping+threshold WORSE than baseline; A5 gain+threshold collapses to 0.00). The plan's
  first draft read the bake-off backwards — these remedies interact *negatively*.
- **Per-type channel-split** — has **no production seam yet**, so it is inherently a "worth-building"
  possibility, NOT a live-confirmable arm in B. Its exact sensor→sub-channel mapping + per-channel
  gain/threshold must be **pinned into the decision record and shipped verbatim by C**, and its number
  labeled **offline-only** — never presented as equivalent to a live-confirmed one.

### 5. The BUILD gate — a live re-encode, not the offline table
The offline table only **nominates**. The nominated remedy (for gain/threshold — the arms with a live
seam) must then pass a **live re-encode through the exact cluster-distinct preflight that refused
Exp 58**: safe and dark encode to distinct clusters live, at the composite bar. Only that issues a
BUILD verdict. Channel-split, having no B seam, ships as a C hypothesis with an offline nomination,
and C's first act is to build the seam and run this same live gate before anything downstream.

### 6. Exp 56/57 non-regression — honest scoping (resolves the plan's own Q4)
The review found this is **not proxy-able in B**: Exp 56/57 use different bodies
(`minecraft_bench`/`minecraft_bench57`) with different sensor sets, and channel-split is a
geometry-tag change that **orphans persisted earned substrate** (so "preserves separations" is
necessary, not sufficient — a migration/re-baseline is also owed). Therefore: B does **not** claim to
clear Exp 56/57. Either (a) B captures real Exp 56/57 vectors under the *standing* apparatus, or (b)
this becomes the **first gate of C's re-baseline** — and in neither case does survival separation alone
authorize a channel-split ship. The C-plan owns the persisted-substrate migration explicitly.

## Deliverable

A provenance-stamped **decision record** (gated-evidence path) stamping the sensor set, ranges,
encoder config fingerprint (à la exp58's `FROZEN`), classroom anchor, and code hash — so "from real
live capture" is verifiable, not asserted. Contents: the live cluster-ID geometry (frozen-centroid
assignments + gain-weighted per-sensor + isolated/sequential), the remedy table (each arm with its
prior verdict and a live-vs-offline label), the composite-metric scores, and a recommendation. This
record is what the Step-C substrate-change plan cites and is designed against; C then gets its own
four-lens design review + two-lens code review + the Exp 56/57 re-baseline.

## Discipline

- **Diagnostic only.** Read-only via `pattern_complete_readonly`/a pure `SensorEncoder` method (no EC
  mutation, no delta-gate/NAc/register side effects). No behavioural claim, no substrate change here.
- **Production code, never a mirror.** Replay through the shipped encoder (`l11_real_trace_remeasure`),
  not `encoding_bakeoff`'s hand-copy/stride.
- **Real contrast, real magnitude, real geometry.** Live classroom vectors at the encoder input;
  frozen-centroid cluster-ID assignment; gain-weighted contribution.
- **Pre-register the composite bar + sample + situations before capture.** Separation-only never
  passes; no knob tuned to this sample (D1).
- **A live re-encode past the Exp 58 cluster-distinct preflight is the sole BUILD gate.** Offline
  nominates; channel-split's offline number is labeled offline-only.
- **B does not claim to clear Exp 56/57** — that check needs the real apparatus and owns a persisted-
  substrate migration; it lands in B-under-standing-apparatus or as C's first gate.
- Provenance-stamped record; the C-plan is gated on this record's recommendation.

## Resolved review questions (were open in v1)

- *Read-only & non-drifting?* Yes — `pattern_complete_readonly` (D8) exists; the pure method reads
  `self.config`. (substrate SF-2, wiring W3)
- *Replay predicts live?* No, not on its own — hence the mandatory live re-encode as the sole gate.
  (measurement DNB-1, wiring W6)
- *Situation sample sufficient?* Only once pre-registered with N + boundary CI + apparatus stop-rule.
  (measurement SF-1)
- *Exp 56/57 capturable in B?* Not by proxy — real apparatus or C's first gate; B doesn't claim it.
  (wiring W5, measurement DNB-3)

## Open (small) items for the build

- Confirm the runtime `modality: world` sensor count (16 vs 17 doc drift) and enumerate dynamically.
- Confirm `l11_real_trace_remeasure.py::analyze` accepts the captured-vector format, or add a thin
  adapter that does not re-implement the encoder.
- Fix `hostile_count` counts-all-loaded-mobs (identical at safe/dark) at capture time, or exclude it
  from the contrast and note the apparatus artefact in the record.
