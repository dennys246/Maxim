# R2 learned-bias pre-registration — does measured relief TEACH a substrate-primary agent to eat when hungry?

> **Status (audit 2026-09-13): SUPERSEDED** — this v1 prereg never took data (the Outcome
> placeholder below is empty by design); its raw-flip design was superseded by the marginal-cluster
> design, and the whole R2 learned-bias question was then resolved offline (commit 96509352: the
> cluster credit is a behavioural MESSENGER, not a cause — see
> [r2_learned_bias_v2_prereg.md](r2_learned_bias_v2_prereg.md) §Outcome and
> [../wiring/substrate-learning-channels.md](../wiring/substrate-learning-channels.md)).

> **STATUS: FROZEN PRE-REGISTRATION (2026-09-12) — NO data taken.** All parameters are settled
> (§ Open decisions → ALL RESOLVED). This document must be merged to `main` with a clean tree
> BEFORE the first data timestamp; the data PR references the frozen commit (research-claim
> non-negotiables). Any change after the first data point is a dated addendum, never an edit to the
> frozen sections.
>
> **PRE-DATA AMENDMENT (2026-09-12, before any data — legitimate, nothing observed).** The
> harness two-lens review found the original primary metric (a raw `None→eat` flip) CONFOUNDED: a
> successful eat books a cluster-INDEPENDENT `tool:eat` causal link (`nac.observe`) that the
> ablation does NOT suppress and that alone flips the probe in BOTH arms — so the raw flip measures
> generic repetition, not the drive-relief CLUSTER credit the claim is about (length-matched →
> forced null; length-mismatched → false held). **The primary metric is amended to the MARGINAL
> cluster probe** (§ Metrics): the difference between `recommend_action(current_clusters=encoded)`
> and `current_clusters=None` on the SAME trained NAc, which cancels the prior and the causal link
> and isolates the learned cluster bias. Caught pre-data; the raw-flip design below is superseded by
> the marginal design.

## Question

R2 (`r2_drive_premise_check.md`) was PREMISE-NULL: the `minecraft_player` world-owned drives did
not measurably move behaviour toward corrective affordances. The 1.3 survival-loop build closed
the three structural breaks, and the live smoke showed breaks 1+2+3 **compose** — but that
validated the credit path's INPUT (the relief signal), not learning. This measures the deeper
premise, the one Oasis's substrate-learning thesis rests on:

**Does break-2's measured-relief credit, accrued over repeated hungry→eat→relief episodes through
the REAL `record_outcome` path, produce a LEARNED bias that raises corrective (`eat`) selection
BEYOND what the cold-start drive prior alone produces — i.e., does the game's own reward TEACH
the want?**

The distinction from break 1 is load-bearing. Break 1 (the cold prior) already selects `eat`
under a STRONG deficit. Learning is a *different* claim: that experience shifts the policy where
the prior alone does not. So the primary probe is a state where the cold prior does **not**
select eat.

## Apparatus (substrate-primary, no LLM in the action path)

- The shipped break-3 world: `scripts/survival_world/setup_world.py` (Paper 1.16.5 survival,
  hunger drains — game-native pressure, D1-clean — food seeded so `eat` is executable).
- Body `bodies/minecraft_player`; AUT via `build_minecraft_aut` (the canonical builder — no
  hand-composed agent).
- **Credit through the REAL path.** Each executed action runs the production
  `execute → read_learning_side_effects → record_outcome` sequence (the Exp 56
  `execute_and_record` precedent — the loop's own intake, not a hand-rolled credit). This is
  non-negotiable: the D43 lesson is that a composition defect hides when you measure hand-composed
  pieces; the credit MUST book through `record_outcome` into the NAc cluster.
- Runs **on big-mac-mini** (the bridge binds `127.0.0.1`, so the AUT must be co-located). This is
  safe: the action path is substrate-primary (no LLM), so it is **not** a second consumer of the
  qwen32b `:8100` server — the co-location trap does not apply.
- `MAXIM_OPERANT_ONLY_CREDIT` is NOT set (this is not the Exp 56 frozen apparatus); the survival
  loop's own credit is the object of study.

## Arms (confound isolation)

Three arms, same seeds across arms, same world, same training length:

1. **LEARNING** — the full loop with measured-relief credit ON. Repeated episodes: drain to a
   strong deficit (game-native hunger), agent selects substrate-primary, executes, relief is
   measured and booked via `record_outcome`.
2. **NO-CREDIT ablation** — identical loop, but the measured-relief credit is WITHHELD by
   **harness-level suppression**: the harness zeroes the break-2 side-effects (forces
   `drive_credit_withheld`, drops `drive_potential_diff`) between `read_learning_side_effects` and
   `record_outcome`, in the HARNESS ONLY — no `src/` change, no new env var in a hot path. Isolates
   the *credit* as the cause: if learning still appears here, it is the prior or mere repetition,
   not the game's reward. **This ablation gets its own deep parallel review before freeze** (two
   readers verify it FULLY ablates — see Validity gates — and it is documented for users in
   `docs/agents/` + memory), because a mislabelled ablation that secretly still credits would make
   the entire claim vacuous.
3. **SATIATED control** — the agent is kept satiated during "training" (no deficit → no relief →
   no credit accrues). Rules out drift / repetition / encoder artifacts that would move the probe
   without any relief signal at all.

## Metrics (primary defined on the LEARNING arm; computed identically on all arms)

- **Probe states (multi-deficit, fixed, frozen):** **food ∈ {11, 12, 13}** — the `None`-band just
  above the cold prior's eat-transition (see the pre-freeze disclosure below: the cold prior is
  `None` for food ≥11 and flips to `eat` at food ≤10). These are the states where a *flip* to eat
  is detectable AND where training must GENERALIZE from the deep-deficit training cluster (food
  ~3–6) to a milder probe cluster — the cluster-generalization test, graded across three distances.
  (food ≤10 already eats cold → no flip headroom; food ≥16 needs an implausibly large bias.)
- **`recommend_action` is DETERMINISTIC** given (frozen NAc, drives) — verified pre-freeze (5/5
  identical, stable across fresh instances). So `K = 1`: `P(select eat | F)` is binary per seed.
- **Primary statistic (MARGINAL cluster probe — amended):** at each `F`, per seed, on the SAME
  post-training frozen NAc, `marginal_flip(F) ∈ {0,1}` = (`recommend_action(current_clusters=encoded)`
  selects `eat`) AND NOT (`recommend_action(current_clusters=None)` selects `eat`). The `None`-clusters
  call scores prior + causal-link only; the encoded-clusters call adds the learned cluster bias — so
  the difference is **purely the learned cluster bias's marginal behavioural effect**, with the
  cluster-independent causal link (which is in BOTH calls) cancelled. The arm's value at `F` is the
  **marginal-flip-fraction** across its `N` seeds. By construction NO-CREDIT/SATIATED have
  cluster-bias 0 → `current_clusters=encoded` == `None` → marginal_flip ≡ 0.
- **Structural pre-registration:** the metric IS "the fraction of seeds whose post-training NAc
  selects `eat` WITH its encoded clusters but NOT without them, at each fixed probe state, compared
  across arms." No post-hoc substitution of a different probe set, statistic, or window; all three
  `F` are pre-registered as primary (a single `F` passing is the multi-deficit robustness, not
  cherry-picking). Probing uses the same `recommend_action` the smoke/R2 probe use (same real
  consumer), `min_confidence = 0.0`. (The raw `None→eat` flip is retained as a REPORTED diagnostic —
  it exposes the causal-link contribution — but is NOT the claim.)
- **Secondary (mechanism, supporting only):** the learned reward-bias magnitude on the eat-associated
  interoception cluster (`cluster_reward_bias(agent_id, cid, "tool:eat")`), which the ablation zeros
  by construction. Should co-move with the marginal-flip primary in LEARNING and stay 0 in the
  controls. Reported; the claim rests on the behavioural primary.

## Pre-freeze apparatus disclosure (measured 2026-09-12, stated here; frozen sections above unchanged)

Measured once on a FRESH substrate, before any training data (the L11 precedent — disclose the
pre-freeze instrument state). `recommend_action` verified deterministic. Cold-prior curve
(health=20), which fixes the probe band:

| food | derived hunger | cold prior pick | conf |
|---|---|---|---|
| ≥11 | ≤0.5 | **None** (no eat) | — |
| 10 | 0.6 | eat | 0.42 |
| 8 | 0.8 | eat | 0.56 |
| ≤6 | 1.0 | eat | 0.70 (saturated) |

So the `None`-band is food ≥11; the eat-transition is at food 10; the probe set {11,12,13} sits
in the `None`-band where a learned cluster bias can produce a marginal flip.

## Decision rule (frozen — no post-hoc motion)

`N = 20` seeds/arm; `M = 0.20` (the minimum meaningful marginal-flip-fraction). `mflipfrac_ARM(F)` =
the fraction of the arm's `N` seeds with a MARGINAL cluster flip at `F` (eat selected WITH encoded
clusters but NOT without — § Metrics). Arms run the SAME episode count (cycle-matched; the controls
do not stop early on a flat-zero bias trace — review fix).

**PREMISE-HELD (learning demonstrated) iff, for at least one pre-registered probe state `F`, ALL of:**
1. `mflipfrac_LEARNING(F) ≥ M` (the learned cluster bias moves that probe), AND
2. `mflipfrac_LEARNING(F) > mflipfrac_NO-CREDIT(F)` by a one-sided permutation test on the per-seed
   marginal flips, `p < 0.05`, AND
3. `mflipfrac_LEARNING(F) > mflipfrac_SATIATED(F)` by the same test, `p < 0.05`.

(Reporting all three `F`; a single `F` satisfying the rule is a pass because each was pre-registered
as primary — this is the multi-deficit robustness, not post-hoc selection.) Seeds flagged
`credit_did_not_book` (LEARNING), `ablation_leaked` (NO-CREDIT), or `did_not_plateau` are REFUSED —
the run exits non-zero on the gated write rather than letting a vacuous/under-trained seed sit in the
denominator (review fix; the prereg's own anti-vacuity rule, now enforced not just recorded).

Otherwise **PREMISE-NULL stands** (a null ships as a null — the Exp 53 shape). In particular:
- `mflipfrac_LEARNING ≈ mflipfrac_NO-CREDIT` at every `F` → the cluster credit did not change
  behaviour beyond the prior/causal baseline → NULL.
- `mflipfrac_LEARNING ≈ 0` at every `F` even with credit that booked → the learned cluster bias did
  not transfer to any probe (see cluster-generalization limit) → NULL/INCONCLUSIVE, reported with the
  mechanism reason,
  NEVER reinterpreted into a pass.

The gated `data/r2_drive_premise.json` record and the R2 doc's PREMISE-NULL status flip to
PREMISE-HELD ONLY on rule satisfaction, in the same PR as the data.

## Validity gates (refusals — exit non-zero)

- **Clean tree + prereg-on-main-before-data.** The harness stamps `executed_code_provenance`
  (in-process; not a sub-sim spawner, so `assert_repo_interpreter` is N/A) and refuses
  `--write-experiment-results` on a dirty tree without `--allow-dirty` (which is disallowed for
  the gated record).
- **Verify-the-instrument.** Before any measurement seed, `setup_world.py verify` must pass (the
  world affords `eat`, exit 4 otherwise) and a one-shot break-3 smoke must show the loop composes.
- **The credit must actually book (anti-D64 / anti-vacuity).** In the LEARNING arm the harness
  asserts `record_outcome` was invoked AND the eat-cluster reward-bias actually moved off zero
  across training; if it did not, the run measured nothing and refuses (a harness that credits a
  no-op is the D43/vacuous-guard family).
- **The ablation must actually ablate.** In the NO-CREDIT arm the harness asserts the eat-cluster
  bias stayed flat (credit genuinely withheld); otherwise the "ablation" is a mislabelled copy of
  the learning arm and refuses.
- **The refusals are ENFORCED, not merely recorded (review fix).** `credit_did_not_book`,
  `ablation_leaked`, and `did_not_plateau` cause a non-zero exit on the gated write — a flagged seed
  is never allowed to sit in the metric denominator (the original harness computed the flags and
  ignored them — the exact vacuous-guard failure the prereg cites).
- **Instrument still holds at run time (review fix).** The harness asserts `pre_flip` (the cold-NAc
  pick) is non-eat at every probe `F` before counting a seed — apparatus drift that moved the cold
  prior would otherwise make a probe silently un-flippable (a false null). It also asserts each
  training episode actually reached the deep-deficit band (food ≤ 4) — a drain that stalls high
  trains the wrong cluster.
- **Arms are cycle-matched and interleaved (review fix).** All arms run the same episode budget (the
  controls do not stop early on a flat-zero bias), and arms are interleaved within each seed
  (`for seed: for arm`) so live-world/time drift cannot alias onto arm.
- **Broadened frozen-apparatus assertion (review fix).** Beyond `substrate_explore_bonus_weight`, the
  harness pins the config surface that governs clustering + bias magnitude (`max_cluster_reward_bias`,
  reward-bias decay, and the EC similarity/centroid thresholds that decide whether food-4 and food-11
  land in the same cluster — the generalization result hinges on these), refusing on a drifted
  `~/.maxim` config.
- **Per-episode credit-source recorded (review fix).** Each LEARNING episode records whether the
  credit was `drive_relief`-sourced vs the generic tool-success floor, so a PREMISE-HELD earned
  partly by the floor (not the game-native relief the claim names) is auditable, not hidden.

## Data + provenance

- Gated record: `docs/experiments/data/r2_learned_bias.jsonl` (per-seed per-arm Δ + the mechanism
  read + the provenance block). Merge-commit (NO squash) for the data PR; tag waits a day.
- Prereg PR (this doc, frozen) merges to `main` FIRST; the data PR references the frozen commit.

## Known-limit acknowledgments

- **The cluster-independent causal link is why the metric is MARGINAL (review-caught).** A
  successful eat books a `tool:eat` causal link (`nac.observe`) that is not cluster-keyed and is not
  suppressed by the ablation; it alone can flip the raw probe in every arm. The marginal metric
  (with-clusters minus without-clusters) cancels it. Residual risk: if the learned cluster bias is
  small relative to the prior+causal baseline, the marginal flip is ~0 even when the bias moved off
  zero — a real NULL (the behavioural effect was below the argmax-flip threshold). The secondary
  cluster-bias-magnitude read distinguishes "bias didn't form" from "bias formed but didn't flip the
  choice"; both are honest nulls of the behavioural claim, reported as such.
- **Cluster generalization is the other live risk.** Credit books on the cluster encoded at credit
  time (strong deficit, food ~4); the probe encodes a milder cluster (food 11–13). If the substrate's
  clusters are too fine to transfer, the LEARNING arm's marginal flip is ~0 even though credit
  booked — a genuine NULL, not a bug to tune away. Mitigated by the multi-deficit probe {11,12,13}
  (graded transfer distances) and the magnitude secondary.
- **The LLM-knows-what-food-is confound does NOT apply** — the action path is substrate-primary,
  no LLM. (It would apply to an LLM-primary arm; out of scope here.)
- **Single body, single world.** This is not a cross-world generalization claim.
- **Repetition without relief** is controlled by the NO-CREDIT arm (eats, builds the same causal
  link, but the cluster credit is suppressed); time/drift by the SATIATED arm (no eat episodes).

## The ladder (this is rung 1)

This prereg is **rung 1: game-native, no injection** — the R2 flip. Later rungs (separate preregs):
- **Rung 2 — homeostatic pain integration:** uses the harness-injected-signal LANE (see the
  2026-09-12 DECISIONS.md record) to study whether the substrate integrates an interoceptive pain
  valence for an out-of-bounds state the game under-models. A **substrate-mechanism** claim (the
  injected signal is the independent variable), NEVER an environment-driven one. Does not touch
  rung 1.
- **Rung 3 — salience / novelty / spatial:** the "vein" mining-classroom extension (material veins
  → spatial-relationship learning). Deferred; needs a probe a cached association can't produce (R1
  representation-vs-association).

## Open decisions (settle before freeze)

ALL RESOLVED (frozen): multi-deficit probe `food ∈ {11, 12, 13}` (None-band, per the disclosure);
`N=20`/arm; `M=0.20`; harness-level ablation; `min_confidence=0.0`; **`K=1`** (`recommend_action`
verified deterministic); pre-freeze cold-prior disclosure DONE.

- **Training = train-to-plateau.** Train until the eat-cluster reward-bias change per episode falls
  below **1%** of the running bias over a **5-episode** window, cap **`C = 60`** episodes. If the
  plateau is not reached by `C`, the run reports **"did not plateau"** (a validity flag) — NOT a
  null; under-training must never manufacture a false null.
- **Training deficit target:** drain to **food ≤ 4** each episode (deep in the already-eat band, as
  the smoke did) so credit books on the strong-deficit cluster whose transfer to the milder probe
  band we measure.

## Outcome

_(dated addendum after the run — the frozen sections above unchanged)._
