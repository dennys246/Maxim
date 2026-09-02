# Measurement limits — the instrument ledger

**What this is.** The fifth ledger. The repo tracks *behavioral* claims
([behavioral_graduation_candidates.md](../plans/behavioral_graduation_candidates.md)),
*algorithmic* claims ([bio_faithful_roadmap.md](../plans/bio_faithful_roadmap.md)),
*engineering rules* (the repository's agent-guidance invariants), and *defects*
([bugs/README.md](../bugs/README.md)). This tracks **what each measurement
instrument can and cannot resolve** — the characterized limits of the apparatus
itself. Not defects (nothing is broken), not process rules
([simulation_apparatus_standards.md](../plans/simulation_apparatus_standards.md)
S1–S8 govern *how* runs are conducted): these are physical properties of the
instruments that every pre-registration must design around.

**Why it exists.** Every entry below was discovered *expensively* — several of
them twice — because it lived only in the experiment doc that first hit it.
The 2026-08-18 phase-locking finding (L2) burned three campaigns and a full
sweep before it was recognized as the same class of limit as the visibility
floor (L1) found a week earlier. An experiment designer reads THIS FILE at
design time; the S-standards tell you how to run, this file tells you what
your metric can actually see.

## Rules (mirroring the bugs ledger)

1. **Measured only.** Every limit cites the experiment/investigation that
   measured it and a magnitude. Suspected limits belong in a plan's
   open-questions section until measured.
2. **Every entry has a disposition:** `BINDING` (no mitigation exists — design
   around it), `MITIGATED` (a tool exists — name it and when it must be used),
   or `RETIRED` (the apparatus changed such that the limit no longer applies;
   kept one release for reference, then pruned).
3. **Design consequence is mandatory** — the one sentence a pre-registration
   author must act on. A limit without a consequence is trivia.
4. **Claim linkage** when a limit bounds a graduated claim: name the row.
5. **Every entry carries a `Re-measure on:` trigger** (mirroring the
   graduation rows' `Re-run on:`) — limits move when the apparatus does, and
   an unmeasured drift is how a MITIGATED entry silently rots.
6. **A limit graduates to its own tracking doc** (`docs/limits/<slug>.md`:
   measurement-history table, mitigation lineage, open questions, triggers)
   when it has **≥2 measurements or active work** — never before (the
   "living doc that never lived" lesson: empty satellites get archived, not
   filled). The README entry stays as the compressed index + link.

## The limits

### L1 — Argmax novelty-visibility floor (~0.11) · MITIGATED

- **Re-measure on:** exploration-policy change; `recommend_action` scoring
  change; place-code default-ON (the split-bias interaction, see L3).

- **Instrument:** substrate-primary action selection (`NAc.recommend_action`
  argmax) under an exploration bonus.
- **Limit:** a learned bias below **~0.11** cannot move the argmax against the
  novelty/exploration term — sub-threshold learning is invisible in behavior,
  and crossing the threshold flips selection discontinuously.
- **Measured:** Exp 48 investigation, 2026-08-11
  ([48_cradle_mother_seam.md](../experiments/48_cradle_mother_seam.md) §heartbeat re-run).
- **Design consequence:** never infer "no learning" from "no behavior change"
  near the floor; instrument the margin instead.
- **Mitigation:** decision provenance (#504) — `explore_decisive` /
  `learned_margin` measure the gap directly. Any sub-threshold-learning claim
  MUST cite them.

### L2 — Deterministic-apparatus phase-locking · MITIGATED · [tracking doc](l2_phase_locking.md)

- **Instrument:** any cyclic scripted stimulus × deterministic (greedy,
  temperature-0) agent — canonically the `cradle_mother` directedness metric.
- **Limit:** the closed loop falls into phase-locked attractors; the metric
  quantizes to a few exact fractions (Exp 48 sweep: every cell a
  seed-invariant twelfth — taught 8/12 → 8/12 → 4/12, control 2/12 → 2/12 →
  6/12 across explore weights 1.5/1.0/0.75, arms INVERTING at 0.75 and the
  control moving with zero teaching). Learned-bias changes jump the system
  between attractors instead of shifting the metric gradually — L1 at
  whole-apparatus scale.
- **Measured:** Exp 48 explore-weight sweep, 2026-08-18
  ([48_cradle_mother_seam.md](../experiments/48_cradle_mother_seam.md) §sweep ew=0.75).
- **Design consequence:** a graded-learning claim on a deterministic apparatus
  requires a **dither source**; without one the metric measures phase
  geometry, not skill. Corollary: seed-invariant results on a "seeded"
  apparatus are a red flag that the seed isn't reaching the dynamics.
- **Mitigation:** seeded stimulus-order shuffle (#514,
  `MotherScaffold.stimulus_order="shuffled"`) — exposure-balanced per block,
  order unpredictable, deterministic per seed. Apparatus-v3 runs MUST use it.
  **Measured 2026-08-25 (Exp 52 Phase B, 12 seeds/arm):** per-seed late-bin SD
  0.130 / 0.079 / 0.082 (taught / satiated / no_feed) with 6–8 distinct values per
  arm — no seed-invariant fractions; the agent side needed no extra dither.
  Directedness is a graded measure again ([52_nurture.md](../experiments/52_nurture.md)).
- **Bounds:** the Exp 48 apparatus-v2 MOTHER-TAUGHT re-earn (+0.482) — real
  and causal, but interpreted as credit-tipped attractor selection, not
  graded orienting (ledger row qualifier, 2026-08-18).

### L3 — Azimuth representational resolution (~3 nodes) · MITIGATED · [tracking doc](l3_azimuth_resolution.md)

> **Third measurement, on live hardware (2026-08-26, [Exp 53/53b](../experiments/53_cross_context_readout.md)):** the nursery's three `audio` clusters partition the axis FAR-LEFT ≤ −0.5 / CENTRE −0.4…+0.3 / RIGHT ≥ +0.4 (identical across seeds 42/43/44); live percepts complete into them 120/120, and the centre bin's right half (+0.2) turns the *wrong* way 18/18 — predicted pre-data. Design consequence: a taught policy is only as fine as this partition; the place code (default-ON, 1.1.x item 5) is the mitigation. **Re-measure on:** `_sensor_embed` / range normalisation change, place-code default flip, a nursery re-run on a new body.

- **Instrument:** exteroceptive azimuth encoding → EC clustering (the raw
  scalar path).
- **Limit:** the raw scalar resolves ~**3** distinct place-like nodes
  (left/centre/right); policies cannot be more graded than the representation.
  Distinct from L2: this bounds what the *substrate* can represent, L2 bounds
  what the *metric* can see.
- **Measured:** RSC pre-check 2026-08-11 (`scripts/rsc_precheck.py`;
  [deferred/retrosplenial_spatial_frames.md](../plans/deferred/retrosplenial_spatial_frames.md) §2);
  production measurement 3 → 7 nodes with the place code
  ([modality_resolution_and_alignment.md](../plans/modality_resolution_and_alignment.md) §7).
- **Design consequence:** graded-orient claims need the place code ON
  (`MAXIM_PLACE_CODE_EXTEROCEPTION`, default OFF; its default-ON gates are the
  1.1.x roadmap item) or an explicit hand-binned readout; and splitting one
  cluster into N divides per-node bias — check the result stays above L1.

### L4 — `safe_pref` saturation (SD 0.000) · BINDING

- **Instrument:** Exp 42/42b substrate-primary discrimination metric.
- **Limit:** `safe_pref` sits at 0.98–1.00 with **SD 0.000** across all
  arms/configurations — the metric detects *breakage* but cannot detect a
  moderate regression.
- **Measured:** Exp 42b re-validation, 2026-07-29
  ([42b_drive_pain_fold_revalidation.md](../experiments/42b_drive_pain_fold_revalidation.md)).
- **Design consequence:** a green Exp 42 heartbeat means "not broken", never
  "no degradation"; a sensitivity-graded degradation arm is the tracked
  follow-up. Claim linkage: the Tier-1 Maintained row carries this caveat.
- **Re-measure on:** the degradation arm landing; any change to the safe/harm
  warmth sources or the B8 delta-attribution path.

### L5 — Actions/turn as a stopwatch (wall-clock ÷ 0.5s) · MITIGATED

- **Instrument:** substrate-primary AUT action counts per narrator turn.
- **Limit:** unbounded, actions/turn tracked narrator wall-clock latency —
  machine- and load-dependent, never reproducible across hosts (the Exp 48
  magnitude non-reproduction's mechanical core).
- **Measured:** Exp 48 investigation, 2026-08-11.
- **Mitigation:** the turn-scoped action budget (#505,
  `MAXIM_SUBSTRATE_ACTIONS_PER_TURN`, stamped per record). Unset = the
  stopwatch regime; any cross-machine magnitude comparison without the budget
  declared is invalid (S6).
- **Re-measure on:** agent-loop pacing change (the 0.5s idle constant);
  bridge `send_and_wait` window semantics change.

### L6 — Prior-agreement ceiling voids · MITIGATED

- **Instrument:** LLM-AUT behavioral-delta gates (Exp 37 class) and any gate
  whose baseline can saturate.
- **Limit:** when the fresh-agent baseline is already at ceiling (Exp 37
  Mistral24B Arm A = 1.000, SD 0.000), rise/delta criteria are structurally
  unattainable — a FAIL is a void, not a negative. The Goldilocks framing is
  the positive form: substrate signal is only visible where priors leave
  headroom.
- **Measured:** Exp 37 Mistral24B fire, 2026-06-11; generalized as apparatus
  standard S7 after being blown through again in Exp 48's v1 gate.
- **Mitigation:** S7 ceiling clauses in gates (gate v2's LEARNED-AT-CEILING is
  the canonical instance): a ceiling is *reported*, never silently
  passed/failed.
- **Re-measure on:** every new model fire (each model's headroom is its own
  measurement — the Goldilocks map grows per fire).

### L7 — Act-granularity vs fast convergence · BINDING (v3 design driver)

- **Instrument:** act-binned learning curves (4 × 12-turn bins) scoring
  within-session learning.
- **Limit:** under the action budget the operant policy converges **within
  act1** (taught act1 0.545 vs control 0.18 — most learning is over before
  the first bin closes), so act-level rise criteria measure the tail, not the
  learning. Combined with L2's plateau, a +0.15 act-level rise from a
  post-convergence baseline can be arithmetically impossible.
- **Measured:** Exp 48 apparatus-v2 re-baseline + ew1.0 sweep, 2026-08-17.
- **Design consequence:** the v3 gate's baseline window must be turn-level
  (first-K-turns), designed from the descriptive mother-log curves
  (`data/48_*_mother_log.jsonl`) and frozen pre-data.
- **Re-measure on:** action-budget value change (convergence speed scales
  with actions/turn); any arc whose act length changes.

### L8 — Exp 37 fires are not reproducible across time, code held fixed · BINDING

- **Instrument:** the Exp 37 LLM-AUT delta gate, and by extension any gate
  whose verdict is compared against a number recorded months earlier.
- **Limit (measured, not inferred):** re-running the **identical commit** on
  the **identical seeds** today does not reproduce its own earlier result. The
  serving environment moves the baseline more than the mechanism does, and the
  run records capture nothing that would let anyone detect or reconstruct it.
- **Measured 2026-08-22**, Arm A / `fire_pit` / Qwen2.5-32B, seeds 43–47, all
  runs valid (`_llm_unavailable = 0`, durations 1528–1567 s against June's
  1492–1641 s):

  | fire | code | n | mean | SD |
  |---|---|---|---|---|
  | June 2026-06-13 | `54d60226` | 5 | **0.420** | 0.27 |
  | today | `54d60226` (same commit) | 5 | **0.710** | 0.119 |
  | today | current main | 5 | **0.750** | — |

  Old code today (0.710) is indistinguishable from current code today (0.750);
  both sit far above the same commit's own June result (0.420). **The ~145 PRs
  between fires are exonerated.** The distribution also tightened (SD 0.27 →
  0.119), so this is a relocation of the whole distribution, not drift in a
  summary statistic.
- **Consequence for the evidence base:** the Qwen32B **+1.43 SD PASS** — the
  anchor of the four-model Goldilocks map — **cannot currently be re-derived**.
  That does not make it false; it makes it unverifiable. Any citation of the
  cross-model map must carry that caveat until the serving environment is
  pinned. It also reframes the 1.1 heartbeat FAIL: current code scoring −0.56
  SD is not a regression, it is Qwen32B leaving the Goldilocks zone because its
  baseline rose to ~0.71–0.75 under today's serving stack, leaving no headroom.
- **Root cause NOT established.** Suspects, none provable from the records:
  serving topology (June's harness targeted a remote leader over the tunnel;
  today reuses a local `Q4_K_M` on :8100), quantization, llama-cpp-server build,
  or sampling defaults. `n_ctx` is partially excluded — 4096 overflows the
  prompt (10 dead calls) under the same prompt-construction code, so June
  cannot have been running there.
- **Why it was undetectable:** the run record stamps `model` (the REQUESTED
  profile name) and nothing else about what served it. No `n_ctx`, no
  quantization, no endpoint, no server build. Every hour of the 2026-08-22
  archaeology traces to that gap, and the same gap means we still cannot say
  what changed.
- **Cost paid:** one ½-day 32B heartbeat that produced an unusable verdict,
  plus a day of reconstruction to find out why.
- **Mitigation (required before the next Exp 37 fire):** stamp
  `resolved_model`, `endpoint`, `n_ctx`, quantization, and server build into
  every run record; and score fires by position (`A`, `B`, `C` recorded every
  time, gate on `B − C`) rather than against a remembered number. Without the
  first, a future re-fire will land in exactly this position again.
- **Amends L6:** L6 says "each model's headroom is its own measurement." Too
  weak in two directions — headroom belongs to the (model, task, **serving
  environment**) triple, and it is not stable across time even with the model
  and code held fixed.
- **Open confound for any successor experiment:** Δ is mechanically
  anti-correlated with A, so the monotone decline across the five Exp 37 fires
  sorted by baseline is also what a do-nothing null predicts. Gate on `B − C`,
  not `B − A`.
- **Raw data:** `docs/experiments/data/37_oldhash_qwen32b_A_2026-08-22.jsonl`
  (S4; the only evidence that the June environment is gone).

### L9 — DoA sweep gain: score the full-range fit under an R² gate, not the central fit · MITIGATED

- **Instrument:** `scripts/orient_backbone/doa_sweep.py` gain, and the H2
  heartbeat gate that scores it against `[0.52, 0.62]`.
- **Where the band actually comes from — NOT a fit statistic.** The H1
  pre-registration derives it from the big-step decision boundary: `|az| ≈ 0.33`
  was set from a gain of **0.55–0.57**, and H2 fires when the measured gain
  implies a boundary moving more than **0.03 az**, i.e. gain outside
  `[0.52, 0.62]` via `boundary ∝ 1/g`. Its midpoint is 0.57. An earlier draft of
  this entry claimed the band "was derived from full-range fits: 0.578 is
  exactly the `post-headfix` ascending full-range value" — **that was wrong**.
  0.578 is an H1 *result*, produced by a three-parameter fold model
  (`az = g·fold(psi − psi0)`), and its coincidence with one OLS full-range fit
  from a different session a month earlier is just that: a coincidence.
- **So the statistic is a decision made HERE, not a recovered one.** The
  historical 0.55–0.57 corpus does not record which fit produced it, and the
  surviving citations are mixed (`45c_flip_bins.md` quotes "sweep 0.546–0.578",
  where 0.546 is a central value and 0.578 a full-range one). This entry
  *chooses* the full-range fit, on the evidence below, and that choice is the
  contribution — it is not an archaeology finding.
- **Why full-range:** across every committed sweep, admitting fits at
  **R² ≥ 0.99, n ≥ 25, non-dry-run**, grouped **per `sweep_start`** (see the
  grouping note below):

  | sweep | pass | full-range | R² | central |
  |---|---|---|---|---|
  | `mag1-setup` | desc | +0.5768 | 0.9971 | +0.605 |
  | `post-headfix` | asc | +0.5776 | 0.9982 | +0.549 |
  | `post-headfix` | desc | +0.5705 | 0.9982 | +0.546 |
  | `h1-geom1-front` | asc | +0.5735 | 0.9959 | +0.519 |
  | `h1-geom1-front` | desc | +0.5835 | 0.9954 | +0.556 |
  | `h1-geom2` run 2 | asc | +0.5758 | 0.9958 | +0.543 |
  | `h1-geom2` run 2 | desc | +0.5746 | 0.9980 | +0.555 |
  | heartbeat 2026-08-23 run 1 | asc | +0.5725 | 0.9978 | +0.519 |

  **Full-range spread 0.0130** (mean +0.5756, all inside the band). The *same
  eight curves* scored centrally span **0.0859** — 86% of the band width. The
  central fit uses ~11 points over a short lever arm and is the noisy one.
- **H2 verdict for the 1.1 heartbeat: PASS.** The admitted curve is **+0.5725 at
  R² 0.9978**, within 0.002 of H1's front-geometry OLS (+0.5735).
- **Group by `run_id`, never by `--label`.** The first draft of this table
  merged same-label sweeps and consequently reported
  `h1-geom2-displaced` as a *failing* curve at "R² 0.60/0.63". Those were the R²
  of two different geometries merged into one fit. Split properly, that label's
  **run 2 ADMITS** (+0.5758 / +0.5746) — and H1's own results section already
  records why: *"The first 'geom2-displaced'-labelled run was a placement
  misunderstanding — the source stayed in front… the label in the JSONL is
  wrong."* The lesson generalises: **this file's own analyses are subject to the
  discipline the tooling enforces.**
- **What the R² gate is, precisely: a LINEARITY detector — not a
  contamination detector.** An earlier draft claimed it "separates real
  measurements from contaminated ones without any appeal to judgement." It does
  not. It rejects the genuinely displaced geometry (`h1-geom2` run 3, R²
  0.61/0.53) because the ±π/2 array fold falls inside the swept range and no
  straight line fits — that is a real measurement of a real geometry, correctly
  refused rather than mis-measured. **Consequence: the full-range statistic
  cannot MEASURE off-axis placements at all, only decline them.** The central
  fit can — run 3's central value (+0.650/+0.644) reproduces H1's displaced
  0.645. Use full-range for the H2 gate; keep the central fit for geometry work.
- **The gate is a cliff — that is why ≥ 4 passes matter.** At R² ≥ 0.98 three
  more curves enter (+0.5693/+0.5708/+0.5983), all still in band. At R² ≥ 0.97
  `mag1-setup` ascending enters at **+0.4178** and the picture breaks. There is
  roughly 0.005 of margin on the admit side and 0.010 on the reject side.
- **Committed sweep data can be SYNTHETIC.** The first `sweep_start` in
  `45_doa_sweep_baseline.jsonl` / `45_doa_sweep_post_headfix.jsonl` carries
  `"dry_run": true` — re-running `--dry-run` on this build reproduces its
  numbers exactly (+0.569/+0.571). That file is cited as a regression guard in
  the graduation ledger and is half simulated. `dry_run` now rides on every
  record, not just `sweep_start`, so a point can no longer be mistaken for a
  measurement.
- **Design consequence:** score the **full-range** fit under **R² ≥ 0.99, n ≥ 25,
  `dry_run == false`**, grouped by `run_id`; report every pass with its
  admission status and the admitted spread. The script now scores this itself
  (`H2_BAND`, `ADMIT_R2`, `ADMIT_N`) and stamps `h2`, `admitted_gains`, and the
  thresholds into the `sweep_done` record, so no consumer has to re-fit — the
  re-fitting is what produced the grouping error above.
- **Claim linkage:** clears the H2 branch of the 1.1 heartbeat walk. Does not
  disturb the orient Layer 1 EARNED row (scored on orient outcome, not gain).
- **Relation to L8:** the instructive contrast. L8 is a gate that genuinely
  cannot be scored because the environment moved. L9 *looked* like L8 for a
  session and was not: the apparatus was fine and the metric definition was
  wrong. Check that the statistic you are comparing is the statistic the
  baseline was derived from — and, if the provenance is unrecorded, say you are
  choosing rather than recovering.
- **Re-measure on:** shell/pinnae acoustic mods, XVF3800 firmware or Reachy
  SDK/daemon version change, source hardware or placement protocol change, mic
  sample-rate change.
- **Raw data:** `docs/experiments/data/doa_sweep_heartbeat_1.1_2026-08-23.jsonl`
  (S4; three sweeps — `sweep_start` #2 is an aborted attempt, exclude it),
  compared against `h1_doa_sweep.jsonl` and `45_doa_sweep_post_headfix.jsonl`.

### L10 — DoA sign-flips reject roughly half of all sweep passes · BINDING

Split out of L9 (review fold): it has its own design consequence and its own
triggers, and it was less discoverable nested inside a MITIGATED entry.

- **Instrument:** the same speech-gated DoA read, at `|psi| ≳ 1.0` in the sweeps —
  and, observed 2026-08-24, at `|body| = 0.84` rad with the head riding along in the
  production orient path (the onset is lower than the sweep-only figure).
- **Limit (measured 2026-08-23):** 3 of 4 passes in one session failed the L9
  admission criterion because of sporadic mirror-image reads at large `|psi|`.
  Run 1 descending fell to R² 0.861, run 2 ascending to 0.901.
- **Limit, production credit path (measured 2026-08-24):** during the H1 `_big`
  delivered-shift block, `ReachyOrientMotorBackend`'s first post-settle azimuth
  window returned +0.289 where an independent later median-of-5 read −0.289 (an
  exact mirror) on a right turn — one folded reading in 18 turns, landing in
  `measured_drive_transitions` and therefore in the credit SIGN of a graduated
  mechanism (Exp 45 row). 0/16 in the admitted run.
- **Not a weak source.** Attempts-per-accepted-read for clean vs
  outlier-bearing points: run 1 descending **1.77 vs 2.31**, run 3 ascending
  **1.59 vs 1.50**. The second pair is flat; the first shows a ~30% gap. On
  n = 2 passes that is **within noise, not identical** — an earlier draft called
  it "the same", which overstated one of the two data points. Supporting
  evidence that it is estimator behaviour rather than signal strength: run 3's
  flip at `psi = +1.00` occurred at 9 attempts, an entirely normal yield.
- **A genuinely dead run is unmistakable** and should be triaged out first:
  19% yield, 59% outlier samples, 6.0 attempts/accepted, gain +0.104 at R² 0.10
  — near-constant azimuth regardless of head angle.
- **Design consequence:** **budget ≥ 4 passes per sweep** so at least 2 admit.
  A 2-pass sweep has a substantial chance of yielding one admitted curve, which
  the script now reports as `PROVISIONAL` rather than scoring. **Credit path:** a
  single post-settle window cannot detect a fold; a sign-consistency check across
  two windows before a transition is credited is the mitigation — tracked as
  bugs-ledger D31 (not yet built).
- **Re-measure on:** XVF3800 firmware, mic geometry or shell change, DoA
  estimator replacement.
- **Raw data:** as L9; production-path observation in
  `docs/experiments/data/h1_partc_big_block.jsonl`, run `20260824T213320Z-76884`,
  record `i=1` (0-based turn index).

### L11 — Sensor-count dilution, and the discrimination ceiling behind it · MITIGATED · [tracking doc](l11_sensor_dilution.md)

- **Instrument:** `similarity/encoder.py::_sensor_embed` → EC
  `pattern_complete_or_separate` at `SensorEncoderConfig.pattern_threshold = 0.85`.
  Applies to every substrate modality channel.
- **Limit, in two parts.** *Detection* (can the substrate see that a state changed?)
  follows **`cos ≈ 1 − 0.57/N`** in the sensor count — clean 1/N from N=1 to N=200,
  so at N ≥ 15 a full single-sensor swing no longer clears 0.85. That is the known
  extero/intero dilution finding, quantified. **The deeper limit is
  *discrimination*** (can it tell *which* sensor changed?): at N=100 two entirely
  different sensors going to extremes read **cos 0.990** — 99% alike. Detection is
  recoverable; discrimination is the real ceiling.
- **Measured:** 2026-09-01, synthetic sweep over the shipped encoder
  ([minecraft_benchmark.md](../plans/minecraft_benchmark.md) §"The sensor ceiling is
  a THRESHOLD artifact"). Signal falls 0.119 → 0.006 across N=4→100 while an
  all-sensor 2% jitter stays flat at ~0.0008, so SNR degrades 185:1 → 7.5:1.
- **Three non-levers, measured so they are not re-proposed.** *Dimension:* 8× more
  embedding dimensions (384 → 3072) moves cosine by **< 0.001** — dilution is an
  averaging problem, not a capacity problem. *Sparse/hashed bases:* **identical** to
  the plain sum; no basis trick escapes summing N terms and comparing by cosine.
  *Distributional moments* (mean/sd/skew/kurtosis/max-dev): give an N-*independent*
  detection signal (cos 0.27 at N=100) but are **permutation-invariant by
  construction**, so they read **cos 0.999** between two different sensors spiking
  and make discrimination *worse at every weight*.
- **Design consequence:** a modality channel carrying more than ~12 scalars at the
  fixed threshold is measuring almost nothing per sensor, and one carrying enough
  sensors for two different excursions to be confusable is measuring *the wrong
  thing*. **Budget sensors per channel, not per body**, and state the per-channel
  count in the pre-registration.
- **Mitigation SELECTED by bake-off 2026-09-01 — the NONLINEAR GAIN (arm A4), at the unchanged 0.85 threshold**, which scores a perfect 1.00 on all three criteria from N=30 to N=100 (tracking doc §Bake-off). **This overturned the pre-bake-off recommendation:** threshold + grouping (A3) measured *worse* than the threshold alone, because grouping shrinks per-channel N, which loosens `1 − k/N` and lets noise separate. Gain + threshold (A5) is actively harmful — stability 0.00. **Cost:** ~120× the control's cluster allocation, which makes **D51** a prerequisite rather than a dormancy candidate. The superseded design was: a sensor-count-scaled threshold
  — `1 − 0.30/N` gives **100% signal separation and 100% noise rejection from N=6 to
  N=80** — *plus* per-type modality channels declared on the sensor schema, since
  grouping alone (discrimination 0.980 at G=1 → 0.831 at G=10) does **not** clear the
  fixed 0.85 bar. Neither is sufficient alone.
- **Claim linkage:** bounds the representation behind Exp 42 (interoception
  clusters), Exp 48 (extero/intero seam) and Exp 53b (whose trigger states *"the
  representation is what transfers"*). All three ran at N ≈ 6 drives, inside the safe
  band — **the limit does not retract them**; it bounds any future body that grows
  past it.
- **Re-measure on:** `_sensor_embed` change, `pattern_threshold` change,
  `_SUBSTRATE_CHANNELS` count change, any body whose per-channel sensor count exceeds
  ~12.

### L12 — A hand-written English prior sits inside action selection · MITIGATED (twins) / BINDING (otherwise)

- **Instrument:** `decisions/nac.py::_DRIVE_TOOL_AFFINITIES`, consumed by
  `NAc.recommend_action` Component 3. A drive whose value exceeds 0.5 pays
  `drive_value` to any tool whose name *contains the drive name*, or
  `0.7 × drive_value` to any tool matching a hand-authored keyword list
  (`cold`/`thermal` → `warm`, `fire`, `blanket`, `huddle`; `hunger` → `eat`,
  `pick_up`, `food`, `consume`, `feed`; and five more).
- **Limit:** tool *names* carry semantics into the substrate. This is a
  cold-start heuristic standing in for EC integration, and it means a
  substrate-primary agent is not prior-free — the priors are in a Python dict
  rather than in an LLM.
- **Measured 2026-09-01, on Exp 42's real tool set at drive 0.9:** every warmth tool
  receives **+0.630 — safe and harm alike — so Δ = 0.000 across all four matched
  safe/harm pairs**, against +0.000 for `sense_presence`/`examine`/`move`.
- **Claim linkage — and it clears the row.** Exp 42's §Results attributes its
  GRADUATE to "B8 delta-attribution + the pre-existing drive-affinity heuristic",
  which read as though the word list carried the discrimination. It does not: the
  term is **symmetric between twins sharing a keyword** and cannot express a
  safe-vs-harm preference. It decides warming-vs-not-warming only. **Exp 42's
  `safe_pref` result does not rest on it.**
- **Design consequence:** the mitigation is *twin naming*, and it is fragile. A body
  whose tools are **not** twins — Minecraft, where `eat`, `pick_up`, `drink`,
  `sleep`, `flee`, `hide`, `look`, `heal` and `fire` are all in the table — has the
  answer pre-installed, **in substrate-primary mode**, before any learning. For any
  such body: use opaque tool and drive names (`aff_07`, `d1`), and **assert
  `score_components["drive"] == 0.0`** from the decision-provenance event rather than
  assuming it. Do not simply delete the table — that changes the mechanism Exp 42
  graduated on.
- **Re-measure on:** `_DRIVE_TOOL_AFFINITIES` edit, `recommend_action` Component-3
  change, any new body whose tool names are not twins.

## Repository capability assessment

[score_cards/](score_cards/) records the repository grades (one card per assessor, `YYYY-MM-DD-<assessor>.md`; 2026-08-19 has independent Codex and Claude cards; 2026-08-27 is the Claude 1.1.0 re-score) for
research integrity, runtime correctness, maintainability, tests/CI, documentation,
and release governance. It also records the evidence required to improve each
grade and assigns the corrective work to 1.1 or 1.1.x.

The scorecard is intentionally **not** an `L*` entry: these `L*` rows describe
measured properties of experimental instruments, while the scorecard evaluates the
engineering system and its development process.

**Cadence status (2026-08-27):** the cards say re-score at each release cut; 1.1.0 was
cut 2026-08-26 and the Claude re-score is issued at the `v1.1.0` commit as
[score_cards/2026-08-27-claude.md](score_cards/2026-08-27-claude.md) (blind to the
reconciliation; its independence disclosure records where the two diverge). The Codex
1.1.0 card is still owed. The 08-19 findings were reconciled against post-1.1.0 `main`
on 2026-08-27 and placed in the roadmap
([§Scorecard → roadmap reconciliation](../plans/roadmap_1_1_to_1_3.md#scorecard--roadmap-reconciliation-2026-08-27):
1.1.x item 16 + 1.2 gate 8); item 16's effect becomes visible as a delta at the 1.2 cut.

---

**Adding an entry:** measured magnitude + source + disposition + design
consequence, in that order. If you can't fill "design consequence", it isn't
a limit yet — it's an observation.
