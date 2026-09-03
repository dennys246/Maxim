# 1.1.4 "The world seam" — implementation plan

**Status:** ACTIVE (kickoff 2026-09-03).
**Roadmap row:** [roadmap_1_1_to_1_3.md](roadmap_1_1_to_1_3.md) §1.1.4 (as corrected 2026-09-03 — the
encoding change is **A4, the nonlinear gain, at the unchanged 0.85 threshold**).
**Design source:** [minecraft_benchmark.md](minecraft_benchmark.md) §"What to build (1.1.4)".
**Evidence:** [../limits/l11_sensor_dilution.md](../limits/l11_sensor_dilution.md) §Bake-off.
**Ship gate (roadmap):** infrastructure only, **no claim**. Smoke benchmark green;
`is_sim_mode=False` verified to consolidate.

This is the implementation-level companion to the roadmap row: the PR ladder, the design
decisions taken at kickoff (with the code-survey findings that forced them), and the gates in
executable form. Scope: the Minecraft bridge, `bodies/minecraft_player.yaml`, the world modality
channel plus its selection-dynamics re-baseline, the two-AUT-one-world harness, and the A4
encoding change with its prerequisite.

---

## Survey findings that shape the plan (2026-09-03, code survey against main @ 928f381d)

### F1. The A4 prerequisite is the substrate-node scan, not `LSHIndex` (D51 row correction)

The roadmap row says "D51 (the degenerate LSH) is a PREREQUISITE." Mechanically that names the
wrong structure. `pattern_complete_or_separate` never touches `LSHIndex` — it is a raw Python
loop over `EC._substrate_nodes` (all nodes of ALL modalities, modality filtered in Python),
with no index of any kind. The degenerate `LSHIndex` (D51 proper) is reached from
`NAc._predict_impl` via `find_similar` (gated on `config.use_ec_similarity`) and from the
live-registered `SimilaritySearchTool` introspection tool via `find_similar_by_memory`
(ungated) — prediction and introspection paths, never the encode path. Fixing `LSHIndex` would not accelerate the scan A4 inflates ~120×.

**What the prerequisite actually is:** a cost measurement of the scan at A4's allocation rate,
and — if the numbers demand it — a cap/prune policy or an index on `_substrate_nodes`, which
exists in no form today. D51 stays open as its own defect with its own (prediction-path) blast
radius. The ledger row and L11 are corrected in PR 0 so the next reader does not fix the wrong
structure and declare the prerequisite met.

A production detail that shapes the measurement: `build_bio_stack` shares ONE EC across all
modality channels, and the scan iterates the whole store — so per-encode cost scales with the
TOTAL node count across channels, not the per-channel count. The harness measures with a
mixed-modality store for exactly this reason.

### F2. The measured A4 arm diverges from the shipped path in three ways

The bake-off's gain (`scripts/encoding_bakeoff.py::_embed`, `GAIN_EXPONENT = 3.0`) is
`w = (|v − 0.5| · 2)^3` applied to each sensor's contribution. Porting it into
`similarity/encoder.py::_sensor_embed` crosses three divergences, each a recorded decision
(D1–D3 below), not an accident:

1. **The neutral point is the literal 0.5, not a set point.** The encoder has no set-point
   concept — set points live in `embodiment/sem.py` drive specs and nothing plumbs them
   through (`EntropicDriveSpec` has no set point at all). The harness's synthetic sensors rest
   at the [0,1] midpoint by construction, so 0.5 *is* their neutral. On a real body,
   `_normalize_value` maps a sensor's range to [0,1], so 0.5 = range midpoint — which agrees
   with the set point when the set point is centered (Reachy azimuth: set_point 0.0 on
   [−1, 1] → 0.5) and disagrees when it is not (a nutritive drive resting at 0.2 would rest
   "loud").
2. **The measured arm composed the gain with an identity normalization, not with
   `_normalize_value`.** In particular gain-over-the-legacy-folding-map (range-blind signed
   sensors) was never exercised.
3. **A body resting exactly at neutral embeds to the ZERO VECTOR** (every weight 0), the
   degenerate-cosine case the bake-off's own "measurement caution" paragraph warned about. The
   harness never hits it (`uniform(0.30, 0.70)` almost surely avoids exact 0.5); a real body
   whose sensors initialize AT their set points — which is what `initial:` usually declares —
   hits it on tick one.

### F3. Other load-bearing seams, confirmed in code

- **Geometry tag:** `encode_sensors` already computes `encoding_geometry_tag(...)` and threads
  it through `pattern_complete_or_separate` / `register_substrate_node` (gates 1+2, #596/#597).
  A4 MUST move the tag — a same-dimension encoding-space change is exactly the D4 hole the tag
  exists to catch.
- **Consolidation seam:** the lightweight-vs-full close is decided by
  `run_agentic_loop(consolidation=...)` (default: lightweight iff `sim.is_sim_mode`), executed
  in `runtime/bio_integration.py::end_bio_session`. `console/handle.py` passes
  `consolidation="full"` explicitly — the pattern the benchmark harness copies.
- **Percepts must be text-shaped:** `MemoryHub.on_percept_received` returns early unless
  `transcript_chunk` or `content` is non-empty text.
- **The injected-reader pattern to copy** is `embodiment/audio_localization.py::AzimuthDoASource`
  (+ `DoAFeed` for the threaded variant) — NOT `embodiment/backends/`, which holds only
  `cerebellum_modulator.py`.
- **`ModalityChannel` is runtime-ephemeral** (its docstring exempts it from CC3), so widening
  it for modality-derived channels is a plain change.
- **Latent bug found:** `drive: null` on a *modulator sub-sensor* crashes
  (`"drive" in ms_spec` reaches `_parse_drive_spec(None)`); the entity-level sensor guard is
  null-safe (`is not None`). Fixed in PR 2 where the schema is touched.

---

## Decisions taken at kickoff

- **D1 (neutral point): ship the measured equation literally** — gain about 0.5 in normalized
  space, `p = 3.0`. A set-point-aware gain and an exponent sweep are unmeasured variants
  (L11 open questions 1a/2); they wait for the L11 re-measure on a real body
  (literal-vs-structural pre-registration discipline). Recorded consequence: a body whose set
  point is far from its range midpoint rests loud; the body author's lever today is declaring
  the range around the set point.
- **D2 (zero vector): `encode_sensors` returns `None` on a degenerate (zero) embedding** — no
  cluster write. "A body at rest encodes nothing" is the same principle as the gain itself.
  Guarded by a unit test pinning the fresh-body-at-`initial:` case.
- **D3 (audio place-code channel): measure before enabling. RESOLVED 2026-09-03 by two
  measurements, and the answer went FURTHER than this bullet anticipated — the gain is
  per-modality with `gain_modalities = {"world"}`, world ONLY:**
  - *Audio ungained* (as this bullet contemplated): gain over the place code degrades
    jitter-stability 1.000 → 0.957 while separation stays 7/7 — the gain crushes the
    intermediate activations that carry the interpolation (`p=3`: activation 0.7 → weight
    0.064). And a gained RAW-scalar azimuth zero-vectors the CENTERED reading, deleting the
    "sound dead ahead" cluster the operant results key on. Data:
    `docs/experiments/data/place_code_gain_check_2026-09-03.json`.
  - *Interoception ungained* (this bullet assumed ON; the bake-off at the infant's actual
    scale says otherwise): at N=6, A4's stability collapses to **0.62** — noise separates,
    the noise-floor cost the pairwise data predicted — and A4 is not even the best arm
    (A1 scores 0.88). A4's dominance begins at N=12 (0.94) and is perfect from N=30. Gaining
    interoception would ship a measured-WORSE representation for every existing body. Data:
    `docs/experiments/data/encoding_bakeoff_n6_n8_2026-09-03.json` (the committed bake-off
    instrument, arms frozen 2026-09-01, run at N∈{6,8}).
  - Consequence, load-bearing: **every existing modality's encoding space is byte-identical
    after PR 1** (gain `None` reproduces the pre-A4 sum exactly; ungained geometry tags do
    not move — both test-pinned). **Exp 42 / 52 / 53b do NOT re-stale**, and the hardware
    block shrinks to the already-owed items (L11 re-measure + roll/pitch recalibration).
    Flipping a modality into `gain_modalities` later is a geometry change and a
    graduation-trigger event — measure first, as these were.
- **D4 (scan-cost decision rule): frozen in `scripts/ec_scan_cost.py`'s docstring before the
  run** (PR 0; the script is authoritative). Summary: projected p95 per-encode scan cost at
  the 4-hour/2 Hz horizon store — ≤ 5 ms ships A4 bare with the number recorded; otherwise a
  store bound is prerequisite, capped at the largest store size whose p95 ≤ 5 ms, shipping in
  the same PR as A4; if that cap falls below 1,000 nodes (≈ one 30-min A4 session's
  allocation) a structural replacement for the exact Python scan is the prerequisite instead,
  and 1.1.4 re-sequences. The harness also times a numpy-vectorized EXACT scan as
  explicitly-exploratory remedy sizing — it informs the remedy, never the verdict.
- **D5 (D67, the finite-resource decision):** the *mechanic* lands in 1.1.4 where scarcity is
  the point — Minecraft food items ship `target_effect: {portions: -1}` (or the world-truth
  readback equivalent through the backend seam). `items/cradle_food.yaml` takes the ledger's
  option (b): its description is corrected to match the code and cradle feeding stays
  unlimited by design — option (a) would re-stale Exp 52 (re-validated twice 2026-09-02) for
  zero 1.1.4 benefit.

## PR 0 result (2026-09-03) — the D4 verdict

Data: `docs/experiments/data/ec_scan_cost_2026-09-03.json` (clean-tree, provenance-stamped,
`ts`-stamped). **Verdict: index-prerequisite.** The pure-Python exact scan crosses the 5 ms
bar at ≈ **238 nodes** — far below the 1,000-node capacity floor (a single ~30-min A4
session's allocation), so a cap cannot carry A4 at any store size a real session reaches.
Projected horizon store 8,375 nodes (organic allocation 0.145 nodes/state, two channels,
4 h @ 2 Hz) → **p95 ≈ 291 ms per encode**, read off the measured piecewise curve (p50s are
cleanly linear, 1.9 ms @ 100 → 405 ms @ 20k; the large-store p95 rows are noisier, and the
verdict is insensitive — every reading sits two orders of magnitude over the bar). Two
earlier runs, both superseded and not citable as evidence (one pre-preflight from a dirty
scripts/ tree, one with the modality-mix defect below), reached the same verdict under the
corrected decision implementation.

**Remedy chosen (design decision, informed by the exploratory phase — the verdict itself is
the frozen rule's):** a **vectorized exact scan** — the same cosine as one numpy
matrix–vector product over a per-modality matrix (filtered once at registration, the way the
remedy would actually be built, vs the Python loop's filter-per-call over the whole store) —
measured at **p95 0.89 ms at a 20k-node store** (12k world nodes scanned), ~2.4× the horizon
store, comfortably under the 5 ms bar at every measured size. Semantics-preserving, so no
ANN approximation risk enters the substrate. "1.1.4 re-sequences" therefore resolves mildly:
the scan replacement ships in **PR 1, before the A4 equation change in the same PR**, guarded
by an old-vs-new equivalence test (identical match/threshold decisions on random stores,
including tie and just-at-threshold cases — float-summation order differs between the loop
and the BLAS path, so the test must assert decision equivalence, not bit equality).

**Harness notes, recorded like the bake-off's.** (1) The first full run's `decide()` computed
N_cap as `max(measured rows, a two-point-fit extrapolation)`; the fit's −40.5 ms intercept
manufactured N_cap = 3,094 and flipped the verdict to cap-prerequisite. (2) The correction
left verdict branch 1 on a least-squares fit that published a horizon p95 BELOW a directly
measured smaller store; both branches now read the piecewise curve. (3) The controlled
store's modality mix decayed from 60% to 24% world across rows (the fill fraction tracked the
moving target size), understating large-store cost ~2×; now an exact interleave with per-row
composition recorded in the evidence. (1) was caught in-session against the rule's own words;
(2) and (3) were caught by the two-lens review round. The rule never changed; the
implementation was brought back to it three times, and each time the verdict survived.

**Process deviation, acknowledged:** the frozen rule and the data share one PR (the bake-off
precedent), rather than the prereg merging as its own PR first per the gated-record rule (2).
Mitigations: merge-commit (no squash) preserves the intra-PR time axis; the record carries
`ts` + provenance (`working_tree_dirty_src_scripts: false`); the prereg lint passes. Future
measurement harnesses of this kind should split prereg PR from data PR.

## The PR ladder

Every PR: two-lens pre-merge review round, fold commits verified on the target; new symbols
caller-grepped (non-test) before any "fixed/shipped" claim; red gates `xfail(strict=True)`;
before merge, confirm `unit-tests` + `lint` + `release-build` AND a CodeQL row are PRESENT in
the check list (D63 — after any retarget, push an empty commit and verify the branch head
moved). Changes accumulate under `[Unreleased]`; the version bumps only in the release
transaction.

### PR 0 — scan-cost measurement + ledger corrections (the prerequisite)

- `scripts/ec_scan_cost.py`: the real `EntorhinalCortex`, real
  `pattern_complete_or_separate` + `register_substrate_node` protocol, A4 embedding, states
  presented at A4's organic allocation rate, mixed-modality store (F1). Two measurements:
  (a) organic growth to a session-length horizon, latency at node-count checkpoints;
  (b) controlled scan timing at fixed store sizes to pin the ms/node coefficient.
  Metric, horizon and the D4 decision rule frozen in the docstring before the run.
- Data → `docs/experiments/data/ec_scan_cost_2026-09-03.json` (bake-off precedent: script +
  frozen metric committed before the data commit, same PR, merge commit — no squash).
- Doc corrections per F1: bugs ledger D51 row, L11 §Known-cost + open question 5, roadmap
  1.1.4 row (one clause).
- This plan doc.

### PR 1 — A4 in `_sensor_embed`, unchanged 0.85 threshold

- The gain in `similarity/encoder.py::_sensor_embed` (decisions D1/D2), exponent as a
  `SensorEncoderConfig` field (default 3.0), threshold untouched (A5 collapsed to 0.00 —
  never combine).
- Geometry tag + `record_encoder_provenance` payload carry the gain mode/exponent (F3);
  existing installs take gate 1's skip-and-warn; the owed `ec invalidate`/re-encode path from
  1.1.3 is this PR's migration story if it fits, else stays explicitly owed.
- D3's synthetic place-code check, then the audio decision recorded here.
- **The vectorized exact scan ships HERE, before the A4 equation change in the same PR**
  (PR 0 verdict: index-prerequisite; see §PR 0 result). Equivalence-guarded: identical
  match/threshold decisions old-vs-new on random stores, ties and at-threshold cases
  included.
- Graduation-trigger sweep, **revised by D3's resolution**: with `gain_modalities =
  {"world"}` no existing modality's space changes (gain `None` is byte-identical, ungained
  tags pinned), so **no EARNED row re-stales** — Exp 53b's trigger names `_sensor_embed` /
  `pattern_complete_or_separate`, and both changes are decision-equivalent for its
  channels (the equivalence guard is the evidence). L11's re-measure trigger DOES fire (the
  equation gained a parameter) and is discharged by the hardware block's re-measure.
- **Honest reachability note (the #590 lesson, stated rather than discovered):** the gained
  path has ZERO production callers until PR 2 wires the world channel — nothing passes
  `modality="world"` today. PR 1 ships the capability + its guards; the CALLER is PR 2's
  channel, and the caller-proof is PR 4's non-vacuous smoke gate asserting world-channel
  encode activity > 0. Until PR 2 merges, "A4 shipped" may not be claimed.

### PR 2 — `modality:` on the sensor schema, the world channel, the re-baseline

- `modality:` field parsed in `embodiment/spec.py` (own reason: a sensor declares its
  modality; membership stops living in `_EXTEROCEPTIVE_ROOT_SENSORS = ("azimuth",)` — the
  tuple its own comment wants generalized). Channels derived from declarations; the two
  existing channels keep byte-identical behavior for existing bodies (regression-pinned).
- The `world` channel entry.
- **Selection-dynamics re-baseline:** `recommend_action` sums `cluster_reward_bias` across
  channels (±1 per cluster, `min_confidence` fixed at 0.3) — measure score distributions at
  2 vs 3 channels, recalibrate explicitly, record here. Assumed-not-measured is the failure
  mode the roadmap row names.
- Fix the modulator-sub-sensor `drive: null` crash (F3).

### PR 3 — the bridge, the body, scarcity

- Mineflayer (JS) process + WS/JSON-RPC; `MinecraftPerceptSource` implementing the frozen
  4-member `PerceptSource` protocol, percepts text-shaped (F3); world-truth affordance
  backend via the injected-reader pattern (F3) — a Minecraft action calls the game and reads
  back truth into `Entity.vital_metrics`, never trusts the declarative delta.
- `bodies/minecraft_player.yaml` — sensors declare `modality: world`; every world-owned
  sensor has `drive: null` or `drift_rate: 0.0` (the drift loop must not fight the writer).
- D5: finite food items; `cradle_food.yaml` description corrected (option b).

### PR 4 — two-AUT-one-world harness + the ship gate, non-vacuous

- Two `run_agentic_loop` threads, separate `percept_source`/`action_sink` pairs (the
  `console/handle.py` bridge pattern), isolated agent homes (shared `~/.maxim` collides),
  `consolidation="full"` passed explicitly.
- Gate, in executable form:
  - smoke benchmark green **with an in-harness apparatus assertion that the substrate path
    ran** (encode activity > 0, `ec.json` written) — D64 is the proof that "clean" can mean
    "never reached"; this gate must not have that shape;
  - a `strict=True` red gate asserting a harness session takes the FULL close
    (consolidation verified) — written failing first, flipped by the wiring.

### The hardware block (operator; scheduled after PR 1 merges — not gating the sim-side PRs)

**Shrunk by D3's resolution (world-only gain):** Exp 53b's re-run is NO LONGER triggered —
its channels (interoception + audio) are byte-identical after PR 1, decision-equivalence-
guarded. What remains, one session: L11 post-mitigation re-measure on a real body above the
safe band (still owed — the N=6/8 and bake-off numbers are synthetic, and RETIRED requires a
real body) → the owed roll/pitch recalibration. If a future decision flips interoception or
audio into `gain_modalities`, the 53b re-run comes back and joins that block.

## Out of scope, recorded

- Any behavioral claim (the four-arm benchmark and the dose–response ladder are 1.2).
- The perception-sharing vs memory-sharing follow-on (in-process percept fan-out) — 1.2.
- Voyager/GITM/SPRING comparison harness, live demo UX, MCP surface — original-stub items,
  all deferred.
- D65 (offline-skipped affordance-transfer tests) — not on this path.
- A real ANN index on `_substrate_nodes` — only if PR 0's numbers force it (D4 rule).
