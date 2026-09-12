# R2 learned-bias pre-registration — does measured relief TEACH a substrate-primary agent to eat when hungry?

> **STATUS: FROZEN PRE-REGISTRATION (2026-09-12) — NO data taken.** All parameters are settled
> (§ Open decisions → ALL RESOLVED). This document must be merged to `main` with a clean tree
> BEFORE the first data timestamp; the data PR references the frozen commit (research-claim
> non-negotiables). Any change after the first data point is a dated addendum, never an edit to the
> frozen sections.

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
- **Primary statistic (literal):** at each `F`, per seed, `flip(F) ∈ {0,1}` = (post-training frozen
  NAc selects `eat` at `F`) AND (cold NAc did not) — a **None→eat flip**. The arm's value at `F` is
  the **flip-fraction** across its `N` seeds. `Δ` is unnecessary — the cold pick at each `F` is a
  fixed `None` (disclosure below), so the post-training pick IS the signal.
- **Structural pre-registration:** the metric IS "the fraction of seeds whose post-training frozen
  NAc flips `None→eat` at each fixed probe state, compared across arms." No post-hoc substitution of
  a different probe set, statistic, or window; all three `F` are pre-registered as primary (a single
  `F` passing is the multi-deficit robustness, not cherry-picking). Probing uses the same
  `recommend_action` the smoke/R2 probe use (same real consumer), `min_confidence = 0.0`.
- **Secondary (graded, supporting):** at an already-eat state (food = 8, cold conf ≈0.56, below the
  0.70 saturation), does post-training eat-confidence rise? Supports the mechanism; the claim rests
  on the flip primary.

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
in the `None`-band where a learned bias can produce a flip.
- **Secondary (mechanism, supporting only):** the learned reward-bias magnitude on the
  eat-associated cluster, before vs after — a direct substrate read that should co-move with the
  behavioural Δ in the LEARNING arm and stay flat in the ablation/satiated arms. Reported, but the
  claim is defined on the behavioural primary.

## Decision rule (frozen — no post-hoc motion)

`N = 20` seeds/arm; `M = 0.20` (the minimum meaningful flip-fraction). `flipfrac_ARM(F)` = the
fraction of the arm's `N` seeds whose post-training frozen NAc flips `None→eat` at `F`.

**PREMISE-HELD (learning demonstrated) iff, for at least one pre-registered probe state `F`, ALL of:**
1. `flipfrac_LEARNING(F) ≥ M` (the learning arm moves that probe off the cold `None`), AND
2. `flipfrac_LEARNING(F) > flipfrac_NO-CREDIT(F)` by a one-sided permutation test on the per-seed
   binary flips, `p < 0.05`, AND
3. `flipfrac_LEARNING(F) > flipfrac_SATIATED(F)` by the same test, `p < 0.05`.

(Reporting all three `F`; a single `F` satisfying the rule is a pass because each was pre-registered
as primary — this is the multi-deficit robustness, not post-hoc selection.)

Otherwise **PREMISE-NULL stands** (a null ships as a null — the Exp 53 shape). In particular:
- `flipfrac_LEARNING ≈ flipfrac_NO-CREDIT` at every `F` → the movement is the prior/repetition, not
  credit → NULL.
- `flipfrac_LEARNING ≈ 0` at every `F` even with credit that booked → no learning transferred to any
  probe (see cluster-generalization limit) → NULL/INCONCLUSIVE, reported with the mechanism reason,
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

## Data + provenance

- Gated record: `docs/experiments/data/r2_learned_bias.jsonl` (per-seed per-arm Δ + the mechanism
  read + the provenance block). Merge-commit (NO squash) for the data PR; tag waits a day.
- Prereg PR (this doc, frozen) merges to `main` FIRST; the data PR references the frozen commit.

## Known-limit acknowledgments

- **Cluster generalization is the live risk.** Credit books on the cluster encoded at credit time
  (strong deficit); the probe encodes a different cluster (mild deficit). If the substrate's
  clusters are too fine to transfer, the LEARNING arm's Δ is ~0 even though credit booked — a
  genuine NULL, not a bug to tune away. ⟨DECIDE⟩ mitigation: probe at multiple deficits (incl.
  within the trained range) and pre-register each, OR accept single-`F_probe` and report the
  transfer limit honestly.
- **The LLM-knows-what-food-is confound does NOT apply** — the action path is substrate-primary,
  no LLM. (It would apply to an LLM-primary arm; out of scope here.)
- **Single body, single world.** This is not a cross-world generalization claim.
- **Repetition without relief** is controlled by the satiated arm, not merely asserted.

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
