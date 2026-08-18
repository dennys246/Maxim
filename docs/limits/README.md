# Measurement limits — the instrument ledger

**What this is.** The fifth ledger. The repo tracks *behavioral* claims
([behavioral_graduation_candidates.md](../plans/behavioral_graduation_candidates.md)),
*algorithmic* claims ([bio_faithful_roadmap.md](../plans/bio_faithful_roadmap.md)),
*engineering rules* (CLAUDE.md invariants), and *defects*
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

## The limits

### L1 — Argmax novelty-visibility floor (~0.11) · MITIGATED

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

### L2 — Deterministic-apparatus phase-locking · MITIGATED

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
- **Bounds:** the Exp 48 apparatus-v2 MOTHER-TAUGHT re-earn (+0.482) — real
  and causal, but interpreted as credit-tipped attractor selection, not
  graded orienting (ledger row qualifier, 2026-08-18).

### L3 — Azimuth representational resolution (~3 nodes) · MITIGATED

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

---

**Adding an entry:** measured magnitude + source + disposition + design
consequence, in that order. If you can't fill "design consequence", it isn't
a limit yet — it's an observation.
