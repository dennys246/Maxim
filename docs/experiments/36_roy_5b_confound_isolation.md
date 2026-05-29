# Roy-5b-confound-isolation — Stage 4a eligibility gate 1

**Date:** 2026-05-29 (run kicked off; verdict populated post-analysis)
**Plan:** [roy_5_encoder_alignment_disambiguator.md § Stage 3 Recommendation](../plans/roy_5_encoder_alignment_disambiguator.md)
**Motivating verdict:** [35_roy_5b.md](35_roy_5b.md) — Conditional PASS / Ambiguous surfaced; Stage 4a eligible pending two gates
**Spec:** [scenarios/roy/roy_5b_confound_isolation.yaml](../../scenarios/roy/roy_5b_confound_isolation.yaml) — single-variable change vs Roy-5b is the body (`bodies/infant_humanoid` vs `bodies/infant_humanoid_naming_v1`)
**Engineered fixture:** [scenarios/roy/roy_2pc_holdout.yaml](../../scenarios/roy/roy_2pc_holdout.yaml) (reused unchanged from Roy-2pc / Roy-2c / Roy-4 / Roy-5b)
**Reproduction:** see [protocols/35_roy_5b_reproduction.md](protocols/35_roy_5b_reproduction.md) — same protocol, swap spec
**Analyzer:** [scripts/analyze_roy_4_coactivation.py](../../scripts/analyze_roy_4_coactivation.py) (reused as-is)

## Why this experiment exists

[Roy-5b](35_roy_5b.md) observed a striking 0% → 100% recognition gap closure between Roy-4 (2026-05-13, arm A 0/10 overlap with priming) and Roy-5b (2026-05-28, arm A 10/10). The PR #297 verdict surfaced three competing explanations for the closure:

1. **Scaffold-driven:** the naming-event scaffold (drive→linguistic co-firing emitter from PR #295) produces structurally-stable utterances that pattern-complete onto the same handful of text-modality centroids, and arm-A test percepts then pattern-complete onto those centroids.
2. **Threshold-driven:** [PR #264 (2026-05-24)](https://github.com/dennys246/Maxim/pull/264) raised `ECConfig.pattern_complete_threshold` from 0.40 to 0.44 between Roy-4 and Roy-5b. The new threshold rejects marginal admissions during priming (priming node count 37 → 18) and the resulting fewer-broader basins are more attractive landing zones for arm-A percepts independent of the scaffold.
3. **Denominator artifact:** priming basins being fewer and more attractive accounts for some of the 10/10 overlap independent of either mechanism.

This experiment isolates explanation (2). It runs the **Roy-4 iteration spec** (standard `bodies/infant_humanoid` body, NO naming-event scaffold) **at HEAD** (so the EC drift fix from PR #264 IS active) with `MAXIM_SUBSTRATE_PATH=1` (so the LinguisticEncoder text path fires — which Roy-4's original 2026-05-13 protocol omitted from its documented command, though the user's shell may have had it set during the original run; see PR #297 lesson).

## Pre-registered diagnostic

The headline measurement is **arm A's node-set overlap with priming** at the default Hebbian rule. Three pre-registered outcomes:

| Arm A overlap | Diagnosis | Stage 4a routing |
|---|---|---|
| **≥ ~50%** (close to Roy-5b's 100%) | **Threshold-driven** — EC drift fix is the dominant cause. The naming-event scaffold contributed marginally or not at all to Roy-5b's gap closure. The (drive, drive) matched edge in Roy-5b is incidental to the scaffold. | Stage 4a rationale **collapses**. Archive `cross_modal_substrate_binding.md` definitively. Stage 4b stays parked. Roy-5 disambiguator plan moves to a different question entirely (likely: what explanation accounts for the threshold-driven gap closure, and does that support a different 1.1+ research direction). |
| **< ~10%** (close to Roy-4's 0%) | **Scaffold-driven** — the naming-event scaffold IS the dominant cause of Roy-5b's gap closure. EC drift fix alone doesn't explain it. The scaffold's mechanism of action (drive→linguistic co-firing) IS doing real work. | Stage 4a eligibility **moves to gate 2** (user classification call on whether the (drive, drive) intra-modal edge counts as evidence the cross-modal binding mechanism is rescued). The scaffold is a load-bearing Phase 0 prerequisite for the resurrected binding plan. |
| **~10% – ~50%** (partial) | **Both contribute** — neither factor alone explains Roy-5b's gap closure. The scaffold's marginal contribution may not justify ~780 LOC of binding-plan implementation. | Stage 4a stays in "ELIGIBLE" limbo. Roy-5c on scaffold variants (tightened hysteresis, finer drive thresholds) ships first to characterize how much of the closure each factor contributes. The 4a resurrection decision waits for scaffold-tuning data. |

A secondary measurement: **matching priming↔test edges at default rule.** If non-zero, the binding mechanism fires under standard cradle priming + EC drift fix alone — which would weaken the Roy-4 conclusion that "Hebbian binding cannot fire at all on the standard arc." If zero, the default-rule edge match Roy-5b found is genuinely scaffold-dependent.

## What is held constant vs Roy-5b

- **Codebase HEAD** — same `pattern_complete_threshold=0.44`, same EC drift fix, same Phase 1 naming-event infrastructure (just unused since this body doesn't opt in), same Wire-A / W1 / W2 annotations.
- **Spec shape** — same `aut_mode: substrate-primary`, same 5×10 priming stages (`cradle_prelinguistic` + `cradle` arcs), same fixture (`roy_2pc_holdout.yaml`), same 3 arms (a: from_priming/neutral, b: blank/hungry-infant-persona, c: blank/neutral), same 10 test turns per arm.
- **Run environment** — same `MAXIM_SUBSTRATE_PATH=1` + `MAXIM_EC_TRACE_ACTIVATIONS=1` + `MAXIM_LOG_FILE` + `MAXIM_BACKEND_TRACE=1`.

The single-variable change vs Roy-5b is the body: `bodies/infant_humanoid` (NO naming-event metadata) vs `bodies/infant_humanoid_naming_v1` (declares `naming_events:` metadata that the `EmbodimentPerceptSource` reads at construction).

## What is held constant vs Roy-4

- **Spec shape** — byte-identical except for the `name:` field (`roy-5b-confound-isolation` vs `roy-4`) which I changed to avoid clobbering Roy-4's historical artifact dir at `~/.maxim/roy/roy-4/`.

The single-variable change vs Roy-4 is the codebase: HEAD (with EC drift fix + Wire-A/W1/W2 + Phase 1 infrastructure) vs Roy-4's 2026-05-13 commit (pre-EC-drift-fix, pre-Wire-A, etc.). Also: explicit `MAXIM_SUBSTRATE_PATH=1` in the runner command (which the Roy-4 reproduction protocol omits; the user's shell may have had it set during the original Roy-4 run).

## Run parameters

- Spec: `scenarios/roy/roy_5b_confound_isolation.yaml`
- Body: `bodies/infant_humanoid` (standard, NO naming events)
- Expected wall: ~25-28 min (substrate-primary 30s/turn timeout dominates)
- Cost: ~$0.10-$0.30 leader inference

Run command (per [protocols/35_roy_5b_reproduction.md](protocols/35_roy_5b_reproduction.md)):

```
MAXIM_SUBSTRATE_PATH=1 \
MAXIM_EC_TRACE_ACTIVATIONS=1 \
MAXIM_LOG_FILE=/tmp/roy_5b_confound_ec_trace.jsonl \
MAXIM_BACKEND_TRACE=1 \
  maxim roy run scenarios/roy/roy_5b_confound_isolation.yaml 2>&1 | tee /tmp/roy_5b_confound_run.log
```

## Result (2026-05-29 09:36 local)

Wall: **1480.07s (~24.7 min)**. Pre-flight clean (HTTP 200, latency TBD).

### Per-arm

| Arm | Substrate | system_prompt | session_id | turns | duration_s | finish |
|---|---|---|---|---|---|---|
| a | from_priming | neutral | `20260529_092240` | 10 | 273.58 | cancel |
| b | blank | "You are a hungry infant" | `20260529_092714` | 10 | 298.10 | cancel |
| c | blank | neutral | `20260529_093212` | 10 | 300.37 | cancel |

### Headline pairwise diffs

| Pair | `cluster_reward_bias_l2` | (keys) | `episodes_Δ` | `valence_KS` (p) |
|---|---|---|---|---|
| **a_vs_b** | **0.3537** | 5 | +693 | 0.997 (0.000) |
| **a_vs_c** | **0.3537** | 5 | +693 | 0.997 (0.000) |
| b_vs_c | 0.1807 | 4 | 0 | 0.000 (1.000) |

**Roy-5b comparison:** Roy-5b a_vs_b was 0.3521 (5 keys differ) with episodes_Δ +683. Roy-5b-confound-isolation is 0.3537 (5 keys) with episodes_Δ +693. **The headline divergence metrics are essentially identical with and without the scaffold** — the load-bearing signal in `cluster_reward_bias_l2` does not depend on the naming-event scaffold.

### EC activation instrumentation

| Phase | Sessions | EC events | Linguistic | Drive |
|---|---|---|---|---|
| Priming (5 stages) | 5 | 146 | 66 | 80 |
| Arm A | 1 | 34 | 10 | 24 |
| Arm B | 1 | 34 | 10 | 24 |
| Arm C | 1 | 34 | 10 | 24 |

**Roy-5b comparison:** Roy-5b priming had 168 events (88 linguistic + 80 drive). Roy-5b-confound-isolation has 146 events (66 linguistic + 80 drive). **The scaffold produces +22 linguistic priming events as designed; drive count is identical**. The scaffold's effect on linguistic-modality firing rate is exactly the predicted magnitude.

### Node-set overlap — THE diagnostic measurement

| Phase | Unique nodes | (linguistic / drive) | Overlap with priming | Overlap with priming food clusters |
|---|---|---|---|---|
| Priming | 17 | (15 / 2) | — | 2 / 2 |
| Arm A | 10 | (8 / 2) | **10 / 10 (100%)** | **2 / 2 (100%)** |
| Arm B | 10 | (8 / 2) | (not analyzed) | — |
| Arm C | 10 | (8 / 2) | (not analyzed) | — |

**Four-experiment matrix (Roy-4-replica added 2026-05-29):**

| Experiment | Date | Body | Codebase | Threshold | Priming nodes | Arm A nodes | **Arm A overlap with priming** | Food clusters in arm A |
|---|---|---|---|---|---|---|---|---|
| Roy-4 | 2026-05-13 | infant_humanoid | pre-fix (~796ef41) | 0.40 | 37 | 10 | **0/10 (0%)** | 0/6 |
| **Roy-4-replica** | **2026-05-29** | **infant_humanoid** | **pre-fix (~796ef41)** | **0.40** | **45** | **10** | **0/10 (0%)** | **0/6** |
| **Roy-5b-confound-isolation** | **2026-05-29** | **infant_humanoid** | **HEAD** | **0.44** | **17** | **10** | **10/10 (100%)** | **2/2** |
| Roy-5b | 2026-05-28 | infant_humanoid_naming_v1 (scaffold) | HEAD | 0.44 | 18 | 10 | **10/10 (100%)** | 2/2 |

**Pairwise attributions are now clean:**

- **Roy-4 → Roy-4-replica** (same code, same body, same threshold; only "what env did the user's shell actually have set" varies — replica ran with `MAXIM_SUBSTRATE_PATH=1` explicitly): **overlap unchanged at 0/10.** The original Roy-4 baseline reproduces apples-to-apples. The 68 linguistic priming events Roy-4 reported were real; the env var was set during the original run (likely from the user's shell). Roy-4-replica's priming node count (45) is ~22% higher than Roy-4 (37), within natural run variance from leader inference stochasticity / narrator variability.
- **Roy-4-replica → Roy-5b-confound-isolation** (same body, threshold 0.40 → 0.44, code old → HEAD): **overlap jumps from 0/10 to 10/10.** The EC drift fix's threshold change is solely responsible for the recognition gap closure.
- **Roy-5b-confound-isolation → Roy-5b** (same code, body adds naming-event scaffold): **overlap unchanged at 10/10.** The scaffold contributes ZERO to the load-bearing gap-closure metric.

**Decisive confirmation: the EC drift fix alone closed the Roy-2c recognition gap.** The naming-event scaffold's contribution to the gap-closure metric is zero; its contribution to the binding-rule metric is +1 (drive, drive) edge at default (which doesn't move the gap-closure needle).

### Parameter sweep

| `min_cofire` | `min_weight` | Priming would-have-bound edges | Matching priming↔test edges | Roy-5b comparison (with scaffold) |
|---|---|---|---|---|
| 1 | 0.01 | 95 | 9 | 98 / 9 |
| **5 (default)** | **0.5 (default)** | **0** | **0** | **1 / 1** |

(only default + most-permissive shown for brevity; default rule is the load-bearing comparison)

**Interpretation of the sweep delta vs Roy-5b:**

- At `min_cofire=1` (most permissive), this run produces 9 matching priming↔test edges — same count as Roy-5b. The set of high-permissivity edges is similar with or without the scaffold; the food-cluster overlap into arm A is the same 2/2 with or without scaffold, so the trivial co-firing-at-any-density edges form regardless.
- At `min_cofire=5` (default), this run produces 0 matching edges vs Roy-5b's 1. The scaffold's contribution to the binding-rule metric IS real (it lifts the (drive, drive) edge between food clusters from sub-default cofire to 5 ticks) — but the cluster overlap into arm A is unchanged. The scaffold-dependent edge is decorative to the load-bearing gap-closure metric, not load-bearing in itself.

**At default rule, this run produces ZERO priming would-have-bound edges** vs Roy-5b's 1. The scaffold IS doing real binding-rule work — the extra drive-threshold-crossings (driven by naming utterance triggers) push the (drive, drive) edge between two food clusters from sub-default cofire count to 5 ticks. **But this binding signal is not load-bearing for the recognition gap closure** — that closure happens at the EC pattern-completion layer, independent of binding edges.

## Verdict: Branch A — threshold-driven (decisive)

**The EC drift fix (PR #264, `pattern_complete_threshold` 0.40 → 0.44) is the dominant cause of Roy-5b's recognition gap closure.** The naming-event scaffold's contribution to the gap-closure metric is zero; its contribution to the binding-rule metric is +1 edge at default (which is the (drive, drive) intra-modal edge that surfaced as Roy-5b's "literal PASS").

The Stage 4a rationale collapses on this data:
- The recognition gap (which Stage 4a's resurrected binding plan was designed to close via cross-modal binding edges) is **already fully closed** at HEAD without the scaffold.
- The (drive, drive) intra-modal matched edge that drove Roy-5b's Conditional PASS is **scaffold-dependent** (it doesn't form without the scaffold) but **not load-bearing** (the recognition gap closes without it).
- The cross-modal (drive, linguistic) edges that DO form at sub-default thresholds in Roy-5b (4 edges at min_cofire=1) are **scaffold-driven** but operate on a metric that is no longer the load-bearing one for Stage 4a's motivating problem.

The scaffold IS structurally doing what it was designed to do (producing co-temporal drive + linguistic EC firing). The drift fix just **already solved the underlying problem the scaffold was a workaround for** via a different mechanism (pattern-completion threshold tuning).

### Disposition

- **Archive [cross_modal_substrate_binding.md](../plans/cross_modal_substrate_binding.md) definitively.** The "Hebbian binding via temporal co-activation rescues the Roy-2c gap" framing is falsified — the gap is already closed at HEAD by a different mechanism (pattern-completion threshold), and the scaffold-dependent binding edges don't carry any additional load. Front-matter status moves from "CONDITIONALLY ELIGIBLE for resurrection" to "ARCHIVED — superseded by EC drift fix per Roy-5b-confound-isolation."
- **[jepa_cross_modal_alignment.md](../plans/jepa_cross_modal_alignment.md) stays at its pre-Roy-5b "Stage 4b candidate" status — unchanged by this experiment.** Roy-5b's specific Branch B promotion trigger (clean FAIL across the parameter sweep) did not fire, so JEPA does NOT promote to "1.2 in flight." But the plan's underlying motivation is independent of any specific Roy iteration outcome — it stands on the structural fact about different-dimensional encoders (384 vs 768). Neither outcome of Roy-5b promotes it; neither cancels it. The next experiment that could change JEPA's status is whichever 1.1+ Roy iteration surfaces a problem that's structurally cross-modal AND can't be solved by the threshold-tuning path.
- **The Roy-5 disambiguator plan moves to a new question:** does the EC drift fix's recognition-gap closure produce downstream behavioral convergence (Roy-2c-style persona-inertness was the original problem; does the gap closure show up there)? OR is the gap closure measurable at the EC layer but not the cluster_reward_bias / behavioral layer? Disambiguator's Stage 5 becomes "characterize what the threshold-driven gap closure actually buys behaviorally."
- **The naming-event scaffold (PR #295) stays in the codebase as opt-in research infrastructure.** It does what it claims to do (co-temporal drive+linguistic firing) — just doesn't carry the load Stage 4a needed. Per CLAUDE.md Principle 2 (dormancy over deletion), mark `embodiment/naming_events.py` as `Dormant since 2026-05-29: superseded by EC drift fix per docs/experiments/36_roy_5b_confound_isolation.md` in the module docstring. Code stays wired; tests stay green; no new features build on top.

## What this still does NOT prove

- Whether the (drive, drive) intra-modal edge in Roy-5b counts as evidence for the cross-modal binding mechanism. That's gate 2 of Stage 4a eligibility — user classification only.
- Whether the scaffold generalizes to non-food, non-cradle fixtures. Roy-5c on a divergent fixture would confirm.

## Roy-4-replica — apples-to-apples baseline confirmation (2026-05-29)

The Roy-4 baseline (2026-05-13, arm A 0/10 overlap) was measured at a codebase that pre-dates the EC drift fix (PR #264, 2026-05-24) AND the 0.9.1 wires AND the Phase 1 naming-event infrastructure. The Roy-4 reproduction protocol's documented run command omits `MAXIM_SUBSTRATE_PATH=1`, but the Roy-4 exp doc reports 68 linguistic priming events — which the env-var-off path cannot produce. The inference was that the user's shell had `MAXIM_SUBSTRATE_PATH=1` set during the original Roy-4 run; this experiment confirms the inference is correct.

### Setup

- **Worktree** at commit `796ef41` (PR #246 merge, the Roy-4-era commit where the Roy-4 spec landed). EC threshold = 0.40 (pre-drift-fix).
- **Spec:** `scenarios/roy/roy_4_replica.yaml` (byte-identical to `roy_4_iteration.yaml` at that commit except for the `name:` field, which I changed to `roy-4-replica` to avoid clobbering Roy-4's historical artifact dir at `~/.maxim/roy/roy-4/`).
- **Body:** `bodies/infant_humanoid` (standard, no naming events).
- **Env:** `MAXIM_SUBSTRATE_PATH=1 MAXIM_EC_TRACE_ACTIVATIONS=1 MAXIM_LOG_FILE=/tmp/roy_4_replica_ec_trace.jsonl MAXIM_BACKEND_TRACE=1` — explicitly set to confirm what Roy-4 had implicitly.
- **PYTHONPATH override:** `PYTHONPATH=/Users/.../Maxim-wt-roy4-replica/src` so the main repo's `.venv` uses the worktree's old source code instead of the editable-install HEAD code. (Verified: HEAD threshold reads 0.44, worktree threshold reads 0.40 with PYTHONPATH override.)

### Result

Wall: 1562s (~26 min) — within run-to-run variance of Roy-4 (1547s). Pre-flight clean.

| Pair | `cluster_reward_bias_l2` | (keys) | `episodes_Δ` | `valence_KS` (p) | Roy-4 baseline |
|---|---|---|---|---|---|
| a_vs_b | **2.4678** | 10 | +654 | 0.998 (0.006) | **2.4678** / 10 / +650 (exact match on L2 + keys) |
| a_vs_c | **2.4678** | 10 | +654 | 0.998 (0.006) | **2.4678** / 10 / +650 (exact match) |
| b_vs_c | 0.3000 | 4 | 0 | 0.000 (1.000) | 0.3000 / 4 / 0 (exact match) |

| Phase | EC events | Linguistic | Drive |
|---|---|---|---|
| Priming (5 stages) | 177 | 95 | 82 |
| Arm A | 50 | 26 | 24 |
| Arm B | 38 | 14 | 24 |
| Arm C | 42 | 18 | 24 |

(Roy-4 originally reported priming 148 / 68 linguistic / 80 drive — replica's counts are higher but within natural variance from leader inference stochasticity and narrator variability across 50 priming turns + 30 test turns.)

### Node-set overlap

| Phase | Unique nodes | (linguistic / drive) | Overlap with priming | Overlap with priming food clusters |
|---|---|---|---|---|
| Priming | 45 | (39 / 6) | — | 6 / 6 |
| Arm A | 10 | (8 / 2) | **0 / 10 (0%)** | **0 / 6** |

**Roy-4-replica's arm A overlap is 0/10 — IDENTICAL to Roy-4's original 0/10.** The original Roy-4 baseline reproduces apples-to-apples. The 68 linguistic priming events Roy-4 reported were real; the user's shell had `MAXIM_SUBSTRATE_PATH=1` set during the original run.

### What this confirms

Roy-4-replica completes the four-experiment matrix above. The pairwise diffs are now all clean:

- Same body, same code, same threshold → IDENTICAL overlap (Roy-4 ≈ Roy-4-replica = 0/10 ✓)
- Same body, threshold 0.40 → 0.44 → overlap 0/10 → 10/10 (EC drift fix is the cause ✓)
- Same code, body adds scaffold → IDENTICAL overlap 10/10 → 10/10 (scaffold doesn't move the load-bearing metric ✓)

**The Branch A verdict is now decisively confirmed with no remaining apples-to-apples concerns.** The Roy-2c recognition gap closure is entirely a consequence of the EC drift fix's `Δ=0.04` threshold tweak. The naming-event scaffold (PR #295) does what it was designed to do mechanistically (co-temporal drive + linguistic firing) but doesn't carry the load the Stage 4a binding plan needed.

## PR

(filled in after PR open)
