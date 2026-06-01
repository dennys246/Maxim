# Experiment 37: Cross-Session Behavioral Delta — Cradle Graduation

**Date pre-registered:** 2026-05-30
**Status:** PRE-REGISTERED — implementation not started. Falsification conditions and acceptance thresholds locked here BEFORE any trial runs.
**Worktree:** `feat/1-0-graduation-cross-session` at `/Users/dennyschaedig/Scripts/Maxim-wt-cross-session`
**Authorization:** [kickoff_1_0_graduation_cross_session.md](../../.claude/projects/-Users-dennyschaedig-Scripts-Maxim/memory/kickoff_1_0_graduation_cross_session.md); scenario + metric + ablation + backend choices selected by user via AskUserQuestion during kickoff session.

## Purpose — dual-gate evidence

This single experiment serves both 1.0 gates:

1. **Tier 1 Behavioral Graduation** ([behavioral_graduation_candidates.md](../plans/behavioral_graduation_candidates.md) row 1): "Cross-session learning without fine-tuning" — current status PARTIAL (3 memories/turn on resume per [Exp 10](10_cross_session_enrichment.md); predictions + concepts pending; **no behavioral measurement of downstream effect**). This experiment is the EARNED-or-reframed graduation evidence.
2. **1.0 Benchmarking Gate** ([benchmarking_1_0.md](../plans/benchmarking_1_0.md)): the "paired fresh-vs-resume Cradle behavioral measurement" that the doc explicitly names as the headline missing artifact. This experiment is the gate's primary-criterion run.

Path C from the kickoff: one artifact, two views. The graduation answers "do the bio-mechanisms carry behavioral weight"; the benchmark answers "does the agent's end-to-end performance reflect that." The same paired-trials measurement satisfies both.

## Pre-registered acceptance criteria

All thresholds and rules below are locked here BEFORE any sim runs. The pre-registration discipline (per [feedback_invariant_two_tier_tracking.md](../../.claude/projects/-Users-dennyschaedig-Scripts-Maxim/memory/feedback_invariant_two_tier_tracking.md)) means a passing result is evidence the gate works; a failing result is honest evidence the gate doesn't work — both are valid outcomes. Post-hoc threshold relaxation is forbidden.

### Primary criterion (must pass — non-negotiable for graduation EARNED)

**The fresh-vs-resume behavioral delta is non-zero, in the expected direction (Arm B avoids the failed action class more than Arm A), and isolable to the bio-substrate via Arm C.**

Operationalized:

- **Metric:** **repeat-failure-action rate** — fraction of turns in which the agent selects an action belonging to the "failed action class" defined per scenario (below).
- **Direction predicted:** Arm B rate < Arm A rate.
- **Variance-survival rule (verbatim from benchmarking_1_0.md):** Arm B's mean must lie outside Arm A's 95th-percentile band (computed across the same N=5 paired trials). Eyeballing is not acceptable.
- **Isolation rule:** Arm C's mean must fall WITHIN Arm A's 95th-percentile band. If Arm C also shows shrinkage, the result is the "general caution" confound — primary FAILS and the graduation reframes.

#### Failed action class — Fire pit scenario

- Action class members: `touch(fire_pit)`, `pick_up(fire_pit)` (the latter fails because fire_pit is non-acquirable; both fall into the "physically contact the fire" class).
- Failure trigger: `arms.thermal: 0.6` self_effect exceeds the body's `comfort_band: 0.5` → drive pain emits → PainBus dispatches → NAc records aversive bias against the `fire_pit` substrate cluster. AND/OR the `thermal_contact` reflex fires from narrator description.
- Per-turn membership: count 1 if any tool call in the turn matches the action class; 0 otherwise.

#### Failed action class — Sharp rock scenario

- Action class members: `pick_up(sharp_rock)`, `touch(sharp_rock)`.
- Failure trigger: `pick_up` reparents sharp_rock to body → its `sharpness: 0.8` sensor joins the body's damage model → `laceration` failure_mode (`sharpness > 0.5 → pain 0.4`) fires. `touch` self_effect writes `arms.pressure: 0.4`.
- Per-turn membership: count 1 if any tool call in the turn matches the action class; 0 otherwise.

### Corroborating criteria (≥1 of 3 must hit for EARNED)

All three are pre-registered. The gate passes when the primary clears AND at least one corroborating metric shows the predicted shift with ≥1 SD effect on the Arm A baseline. If ALL THREE diverge from the prediction while the primary still hits, that's evidence the primary is a measurement artifact — investigate before claiming EARNED rather than ignore.

1. **Affordance-selection preference shift** (most mechanism-bridging). Among all actions targeting the failure-scenario entity (`fire_pit` or `sharp_rock`), fraction that are `observe` / `examine` (safe) vs `touch` / `pick_up` (failed-class). Predicted: Arm B safe-fraction > Arm A safe-fraction by ≥1 SD.
2. **Tool-class diversity shift.** Count of distinct tool classes used per session. Predicted: Arm B uses fewer distinct failure-class tools (i.e., faster exclusion of dead-ends) than Arm A, by ≥1 SD.
3. **Time-to-safe-steady-state** (narrow interpretation). Number of turns from session start until the agent enters a "safe steady-state" — defined as 3 consecutive turns containing zero failure-class actions. Predicted: Arm B reaches safe steady-state in fewer turns than Arm A, by ≥1 SD.

**Note on time-to-goal scope:** the benchmarking_1_0.md corroborating-metric list named "time-to-goal on success scenarios," which would require adding a separate SUCCESS scenario (e.g., cradle_food hunger satisfaction). To keep scope tight, this pre-registration uses the **narrow form**: turns-to-safe-steady-state within the existing failure scenarios. The broad form (separate success scenario) is a follow-up if 0/3 corroborating metrics hit.

### Isolation arm (Arm C — required for primary)

- **Construction:** Arm C runs the SAME failure scenario as Arm B, but the resumed session loaded from a DIFFERENT prior session — one where the agent encountered cradle_blanket and cradle_button (peaceful entities, no failure-class actions, no NAc aversive bias on fire_pit / sharp_rock).
- **Acceptance:** Arm C's primary-metric mean must fall within Arm A's 95th-percentile band. If Arm C ALSO shows the Arm B delta, the result is "agent gets generally cautious after any prior session" — primary FAILS and the graduation does NOT EARN.
- **Why this matters:** without Arm C, the result is observed-but-not-isolated. The Roy-5b-confound-isolation lesson (exp 36) is the standing reminder: substrate-attributable behavioral claims require an isolation arm whose only difference is the SPECIFIC substrate content, not the presence-of-substrate-at-all.

### Secondary criterion (should pass — couples to behavioral_graduation_candidates Tier 3)

**The cross-session delta is mechanism-attributable. At least one of three ablation arms must shrink Arm B's delta toward Arm A's baseline.**

Ablation arms (Arm B-only — fresh agents have no substrate to ablate, so Arms A and C are not re-run per ablation):

1. **Wire-A annotation off** — `MAXIM_DISABLE_CLUSTER_BIAS_ANNOTATION=1`. Cost: env var only, no code change. Tests whether cluster reward bias substrate-voice in LLM prompts is load-bearing.
2. **Wire 1 variance annotation off** — `MAXIM_DISABLE_VARIANCE_ANNOTATION=1`. Cost: env var only. Tests whether tool-description variance bands carry behavioral weight.
3. **NAc reward bias zeroed** — `MAXIM_NAC_REWARD_BIAS_DISABLED=1`. Cost: new env var to be added in implementation PR. When set, `NAc.distribute_reward` becomes a no-op AND `decay_reward_biases` is skipped AND `get_agent_tool_biases` returns empty. Tests whether the bio-learning substrate is load-bearing vs LLM in-context recall doing the work.

Pass rule: ≥1 ablation shrinks Arm B's primary-metric mean toward Arm A's mean by ≥1 SD (of Arm A baseline).

**If the secondary FAILS** (Arm B delta survives ALL three ablations): the primary may still pass, but the bio-attribution is exposed. 1.0 release notes must retract the "mechanism-driven cross-session learning" framing and reframe as "cross-session memory surfacing produces behavioral change via the LLM's in-context reasoning, not via the bio-substrate's reward learning." Graduation entry downgrades or moves to Dropped per [behavioral_graduation_candidates.md](../plans/behavioral_graduation_candidates.md) lifecycle.

### Tertiary criterion (informational only — NOT gating)

Replication on a second model backend. Local qwen2.5-14b-instruct re-run of Arm A vs Arm B baseline (no ablations). If the delta replicates on local, the result is more robust; if it doesn't, the delta may be Sonnet-prompt-driven (worth flagging but doesn't fail the primary).

## Trial structure

Per scenario (fire_pit, sharp_rock):

| Condition | Arms | Trials | Runs |
|---|---|---|---|
| Baseline | A, B, C | 5 | 15 |
| Wire-A off | B only | 5 | 5 |
| Wire 1 off | B only | 5 | 5 |
| NAc bias zeroed | B only | 5 | 5 |
| **Subtotal per scenario (target runs)** | | | **30** |

× 2 scenarios = **60 target runs**.

Plus Arm C peaceful priors (shared per trial-pair across both scenarios, since the peaceful prior is scenario-agnostic): 5 trials × 1 prior = **+5 prior runs** total. **Grand total: 65 runs on Claude Sonnet.**

Cost projection at Cradle Exp 11's $0.21/run × 65 runs ≈ **$13.65**. (Arm A target runs are reused as the substrate-prior for Arm B and the three B-family ablation arms via `shutil.copytree`; this sharing is what keeps the count near the per-scenario target budget. Arm C requires its own peaceful prior since the failure scenario's substrate is what we're isolating against.)

Wall time: ~10 min/run; 65 runs × 10 min ≈ 11 hours of LLM time, parallelizable in batches of 3-5 concurrent runs depending on rate limits. Probably 2-3 calendar days for execution + ~1-2 weeks for harness implementation and pre-flight smoke tests.

The harness implementation pins the per-arm orchestration + sandbox layout at [scripts/benchmark_cross_session.py](../../scripts/benchmark_cross_session.py); the reproducibility-pin protocol at [docs/experiments/protocols/37_cross_session_graduation_reproduction.md](protocols/37_cross_session_graduation_reproduction.md) §A-E carries the runbook + per-decision rationale (turn-binning operationalization, pair-boundary cost-cap, JSONL append idempotency, FAILURE_CLASS YAML cross-check).

Add tertiary replication on local qwen2.5-14b for Arm A vs Arm B baseline only: 5 trials × 2 arms × 2 scenarios = 20 more runs, ~free per-run cost but slow wall time.

## Falsification conditions

The graduation does NOT EARN if any of the following:

1. **Primary FAIL:** Arm B's mean falls WITHIN Arm A's 95th-percentile band on at least one scenario.
2. **Isolation FAIL:** Arm C's mean falls OUTSIDE Arm A's 95th-percentile band on at least one scenario (i.e., the resumed-from-different-scenario agent ALSO shows the avoidance).
3. **Secondary catastrophic FAIL:** primary passes AND all three ablations leave Arm B's delta intact AND 0/3 corroborating metrics hit. In this case the bio-attribution is unsupported even if "cross-session memory" surfaces behavioral change.

In any failure case: pause; surface to user; the decision is "reframe (drop bio-attribution from 1.0 release notes for cross-session learning) OR fix the underlying issue (substrate-pipeline gap, prompt construction, dose-too-low) OR delay 1.0 ship." Per the kickoff escalation rule: reframe text in 1.0 release notes requires explicit user authorization before landing.

## Graduation paths from this experiment

| Outcome | Tier 1 entry status | 1.0 Benchmark gate | Action |
|---|---|---|---|
| All 3 criteria pass | **EARNED 2026-MM-DD** | PRIMARY PASSES | Update `behavioral_graduation_candidates.md` row 1; flip CLAUDE.md `[behavioral]` tag on the relevant Hippocampus persistence + recall invariants; reference Exp 37 + protocol in `Regression guard:` line |
| Primary pass + Isolation pass + Secondary pass + 0/3 corroborating | **EARNED with footnote** | PRIMARY PASSES | Earn with caveat that corroborating metrics need refinement; the primary repeat-failure-action rate is the load-bearing claim per benchmarking_1_0.md |
| Primary pass + Isolation pass + Secondary FAIL + ≥1 corroborating | **PARTIAL → reframed** | PRIMARY PASSES, secondary FAILS | Surface to user: pull bio-attribution from 1.0 release notes; cross-session memory surfaces ship as engineering feature without bio-claim |
| Primary FAIL or Isolation FAIL | **PARTIAL — investigation gate** | FAILS | Surface to user; root-cause; either fix or reframe; potentially delays 1.0 ship |

## Implementation work — out of pre-reg scope, tracked here for visibility

The pre-registration ships in this PR. The execution work follows as separate PRs:

1. **`MAXIM_NAC_REWARD_BIAS_DISABLED` env var** (~30 LOC src + ~3 unit tests). Touches `decisions/nac.py` only. Same shape as existing `MAXIM_NAC_MIN_CONFIDENCE` env var. Lives in env-var registry section of CLAUDE.md.
2. **Benchmark harness** at `scripts/benchmark_cross_session.py` (~250 LOC). Drives paired-trials bookkeeping; logs per-run JSONL to `~/.maxim/sessions/exp37_<run_id>/` then aggregates into `docs/experiments/data/37_results.jsonl` for analysis.
3. **Fixture preparation.** Define two stable cradle scenario fixtures (fire pit scene, sharp rock scene) with controlled phase progression. Likely re-uses existing cradle arc with explicit scenario-specific entity manifests. Validates fresh-start trials are reproducible across seeds before paired trials run.
4. **Analysis script** at `scripts/analyze_exp37.py` — computes the variance-survival rule + corroborating shifts + ablation deltas, emits a single `docs/experiments/37_cross_session_graduation.md` result-section append.
5. **Execute.** Run the 60 + 20 tertiary trials. Probably batched over 2-3 calendar days.
6. **Two-lens pre-merge review** on the experiment doc + analysis BEFORE updating `behavioral_graduation_candidates.md` status. The Roy-3c-bisect lesson is the standing reminder that mechanism-attribution conclusions need two-reviewer cross-confirmation.
7. **Update `behavioral_graduation_candidates.md`** row 1 → EARNED (or reframed). Update CLAUDE.md tag if Earned.

Estimated wall: 1-2 weeks total for steps 1-5, +1 week for review fold + status updates. Total for graduation: 2-3 weeks from this pre-registration to a green-or-red verdict.

## Reproduction (forward-looking — fill on execution)

Will be drafted at `docs/experiments/protocols/37_cross_session_graduation_reproduction.md` as part of step 2 above. The protocol pre-registers the seed range, model version, prompt version, and encoder version so future re-runs (on the minor-version heartbeat trigger per [behavioral_graduation_candidates.md](../plans/behavioral_graduation_candidates.md)) reproduce this baseline exactly.

## Cross-references

- [docs/plans/benchmarking_1_0.md](../plans/benchmarking_1_0.md) — 1.0 benchmarking gate (this is its primary experiment).
- [docs/plans/behavioral_graduation_candidates.md](../plans/behavioral_graduation_candidates.md) — Tier 1 entry row 1 (this is its EARNED-or-reframe gate).
- [docs/plans/v1_refinement.md](../plans/v1_refinement.md) §1.6 — sequenced 1.0 plan placing this experiment in Phase 2.
- [docs/experiments/10_cross_session_enrichment.md](10_cross_session_enrichment.md) — the PARTIAL prior evidence this experiment converts to EARNED.
- [docs/experiments/11_cradle_sensorimotor_poc.md](11_cradle_sensorimotor_poc.md) — Cradle infrastructure validated; substrate for this experiment.
- [docs/experiments/36_roy_5b_confound_isolation.md](36_roy_5b_confound_isolation.md) — confound-isolation discipline; Arm C exists to honor this lesson.
- [feedback_invariant_two_tier_tracking.md](../../.claude/projects/-Users-dennyschaedig-Scripts-Maxim/memory/feedback_invariant_two_tier_tracking.md) — graduation tag discipline.
- [feedback_confound_isolation_discipline.md](../../.claude/projects/-Users-dennyschaedig-Scripts-Maxim/memory/feedback_confound_isolation_discipline.md) — confound-isolation pre-registration.
- [project_research_claim_non_negotiables.md](../../.claude/projects/-Users-dennyschaedig-Scripts-Maxim/memory/project_research_claim_non_negotiables.md) — the load-bearing 1.0 claim.
- [project_v1_0_sequenced_plan.md](../../.claude/projects/-Users-dennyschaedig-Scripts-Maxim/memory/project_v1_0_sequenced_plan.md) — Phase 2 graduation cadence.
- [kickoff_1_0_graduation_cross_session.md](../../.claude/projects/-Users-dennyschaedig-Scripts-Maxim/memory/kickoff_1_0_graduation_cross_session.md) — authorization + user-decision audit trail.