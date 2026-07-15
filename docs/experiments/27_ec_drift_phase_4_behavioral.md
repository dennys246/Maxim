# 27 — EC centroid drift fix, Phase 4 (Roy-2c behavioral validation)

**Date:** 2026-05-24
**Branch:** [`feat/0-9-1-ec-drift-phase-4-behavioral`](https://github.com/dennys246/Maxim/tree/feat/0-9-1-ec-drift-phase-4-behavioral)
**Plan:** [docs/plans/archive/ec_centroid_drift_fix.md § Phase 4](../plans/archive/ec_centroid_drift_fix.md)
**Companion:** [20_roy_2c.md](20_roy_2c.md) (pre-fix baseline), [25_ec_centroid_drift_fix_phase_1.md](25_ec_centroid_drift_fix_phase_1.md), [26_ec_drift_phase_2_regression.md](26_ec_drift_phase_2_regression.md), [28_ec_drift_phase_3_5_nac_parameterization.md](28_ec_drift_phase_3_5_nac_parameterization.md)
**Iteration spec:** [scenarios/roy/roy_2c_iteration.yaml](../../scenarios/roy/roy_2c_iteration.yaml) (unchanged)
**Engineered fixture:** [scenarios/roy/roy_2pc_holdout.yaml](../../scenarios/roy/roy_2pc_holdout.yaml) (unchanged)
**Reproduction:** [protocols/20_roy_2c_reproduction.md](protocols/20_roy_2c_reproduction.md) — same as Roy-2c, with `MAXIM_SUBSTRATE_PATH=1` added
**Pre-fix baseline preserved:** `~/.maxim/roy/roy-2c-pre-fix-backup-20260523/`

## Status

**VERDICT: UNCHANGED on behavioral signal; SIGNIFICANT structural improvement.**

The pre-registered "sharpens" criterion (Arm A `sense_food_source` count ≥ 3 calls more than Arm C) **did not hit**. All three arms produced **0× `sense_food_source` and 8× FAILED `infant_humanoid_pick_up`** — byte-identical action distributions, same as pre-fix.

**However:** the substrate-level structure changed substantially. The pre-fix `a_vs_b.cluster_reward_bias_l2 = 2.566 with 10 differing keys` (six +1.0 priming `sense_food_source` UUIDs + four ±0.30-0.45 test-phase pick_up UUIDs) collapsed post-fix to **`a_vs_b.cluster_reward_bias_l2 = 0.535 with 4 differing keys`** (one near-zero `sense_food_source` + three pick_up). The priming substrate now produces fewer, more semantically-coherent EC clusters — exactly what the fix was designed to do. The fact that this structural change does NOT translate to a behavioral signal on this fixture confirms the Roy-2c H1 diagnosis (LinguisticEncoder → EC alignment between priming-substrate text and CLI-fixture text) was structurally upstream of centroid drift, not caused by it.

Per the kickoff routing: **fix ships as substrate hygiene + V1 prerequisite; JEPA / cross-modal binding stays a 1.0-or-1.1 gate.**

## Pre-registered routing logic

From [kickoff_ec_drift_phase_4.md](../../.claude/projects/-Users-dennyschaedig-Scripts-Maxim/memory/kickoff_ec_drift_phase_4.md):

| Outcome | Diagnosis | Action |
|---|---|---|
| **Sharpens** — Arm A `sense_food_source` count ≥ 3 more than Arm C | Centroid drift WAS load-bearing for Roy persona inertness | JEPA / cross-modal binding moves to 1.1+ |
| **Unchanged** — A ≈ B ≈ C reproduces | Drift was real but NOT the dominant Roy failure mode | Fix is substrate hygiene + V1 prereq; JEPA stays a 1.0-or-1.1 gate |
| **Regresses** — Arm A < Arm C on sense_food_source | Fix surfaced an interaction we missed | Halt + return to Phase 1 |

**Observed: unchanged.** Routing: ship as substrate hygiene + V1 prereq.

## Per-arm action distribution

```
Arm A (substrate-primed, neutral):       1× null + 8× FAILED infant_humanoid_pick_up (0× sense_food_source)
Arm B (blank, "hungry infant"):          1× null + 8× FAILED infant_humanoid_pick_up (0× sense_food_source)
Arm C (blank, neutral):                  1× null + 8× FAILED infant_humanoid_pick_up (0× sense_food_source)
```

Per-arm action count rose from pre-fix 5 to post-fix 9 (1 null + 8 pick_ups). The null first-turn is structural (cold-start substrate has no biases to score). The eight subsequent pick_up proposals are byte-identical to each other and across arms — same H1-confirming pattern as the pre-fix run.

## Headline pairwise structural diffs

| Metric | Pre-fix Roy-2c (EC=0.40) | Post-fix Roy-2c (EC=0.44) | Δ |
|---|---:|---:|---:|
| Wall time (s) | 1284.2 | **1204.6** | -79.6 (-6.2%) |
| `a_vs_b` cluster_reward_bias L2 | 2.566 | **0.535** | -79.1% (4.8× smaller) |
| `a_vs_b` cluster keys differing | 10 | **4** | -6 |
| `a_vs_b` causal_link Δ | +147 | +305 | +158 (more learning events captured) |
| `a_vs_b` ATL concepts Δ | (unrecorded) | +296 | n/a |
| `a_vs_b` Hippocampus episodes Δ | +664 | +665 | +1 |
| `a_vs_b` valence KS | 0.994 | 0.990 | -0.004 |
| `b_vs_c` cluster_reward_bias L2 | 0.765 | **0.516** | -32.6% |
| **Arm A `sense_food_source` count** | **0** | **0** | **unchanged** |
| **Arm-A vs Arm-C `sense_food_source` gap** | **0** | **0** | **unchanged** |

## Top cluster deltas — a_vs_b

### Pre-fix (10 keys, L2=2.566)

```
6× tool:sense_food_source  delta=+1.0  (priming-acquired UUIDs, never touched at test)
4× tool:infant_humanoid_pick_up  delta=±0.30 / ±0.45  (test-phase updates, NEW UUIDs disjoint from priming)
```

### Post-fix (4 keys, L2=0.535)

```
3× tool:infant_humanoid_pick_up  delta=−0.390 / +0.366 / +0.021  (test-phase updates)
1× tool:sense_food_source  delta=+0.0015  (near-zero — priming bias collapsed to a single shared cluster)
```

The mechanism: pre-fix EC threshold 0.40 admitted marginal paraphrase variants of priming text (cosines 0.42-0.48) into the same priming cluster but drifted the centroid; later marginal variants then ALSO admitted by the drifted centroid created spurious sibling clusters. Each priming reward landed on a slightly different drifted centroid → six distinct `sense_food_source` UUIDs accumulated +1.0 reward bias each. Post-fix threshold 0.44 rejects those marginal admissions; priming text concentrates reward on one or two stable clusters. The arm-A vs arm-B diff over `sense_food_source` keys collapses from six +1.0 entries to one near-zero entry.

## Why behavioral signal didn't move

The Roy-2c H1 diagnosis names the failure mode: **the CLI test-percept text ("you sense food nearby", "the smell of food fills the air") embeds into a DIFFERENT EC region than the priming-substrate text** (cradle-stage narrator output + sensor/drive state strings). The two embedding regions are non-overlapping in cosine space, even though humans read the semantic content as obviously related.

The centroid drift fix tightens clustering WITHIN each region (fewer spurious siblings per concept). It does NOT bridge regions. Arm A's priming reward bias accumulates on tighter, more semantically-coherent clusters — but those clusters are STILL not the clusters the test percepts activate. `recommend_action`'s consultation of the priming clusters' bias is structurally correct, but the active cluster ID on test percepts doesn't match any of those keys.

**This is exactly the scenario the H1 confirmation predicted:** drift was real (post-fix shows clear hygiene improvement), but drift was downstream of the deeper alignment problem, not its cause. Fixing drift gives a cleaner substrate but doesn't bridge the cross-source alignment gap. The cross-source gap is what Wire-A (Stage 2 of [release_0_9_1.md](../plans/archive/release_0_9_1.md)) routes around by surfacing tool-level cluster_reward_bias at the prompt regardless of active cluster, and what JEPA / cross-modal binding work in the [grounded_language_acquisition.md](../plans/grounded_language_acquisition.md) (Phase 2 deferred) and [jepa_cross_modal_alignment.md](../plans/deferred/jepa_cross_modal_alignment.md) plans is positioned to address structurally.

## What this proves

1. **Centroid drift was real.** Post-fix structural diffs confirm the pre-fix substrate carried six spurious sibling clusters per concept that the fix collapses to one. The Phase 1 diagnostic was not a synthetic artifact.
2. **Centroid drift is NOT the dominant Roy persona-inertness mechanism.** The behavioral signal (sense_food_source count on the engineered overlap fixture) is unchanged. The cross-source alignment gap H1 named is structurally upstream of the drift fix.
3. **The fix still belongs in 1.0.** V1 cross-session validation silently depends on the same machinery — cross-session recall of a previously-encoded concept drifts toward "anything second-person-sensory" the more text the substrate has seen. Phase A of V2 (substrate-only baseline) is the next test that depends on this fix being shipped.
4. **JEPA / cross-modal binding priority unchanged.** Stays a 1.0-or-1.1 gate. The behavioral failure mode that JEPA targets is independent of centroid drift.

## What this still does NOT prove

- Whether Wire-A's annotation pattern, once shipped (0.9.1 Stage 5 / Roy-3), recovers behavioral signal on this fixture. Post-fix clusters being more semantically-coherent COULD make Wire-A's bias-surfacing more effective; Roy-3 measures it.
- Whether a more aligned encoder (CLIP-style or JEPA-style cross-modal trained on (sensor_pattern, narrator_text, CLI_percept) triples) would close the gap. Untested; out of 0.9.1 scope.
- Cross-session persistence (single-session as before).

## What the fix DID materially improve

Beyond the headline 79% cluster L2 reduction:

- **Wall time -6.2%** (1284s → 1205s). Tighter clustering produces faster recommend_action lookups (fewer cluster keys to score) and fewer null-tool turns (the substrate has biases on more representative clusters from earlier in the conversation).
- **Causal link delta +158** (147 → 305 per arm-pair). More learning events captured — the post-fix substrate is recording 2× more causal links per turn, likely because tighter cluster identity lets the SCN-coupled eligibility traces fire more often.
- **b_vs_c stochastic-cluster floor down 32.6%** (0.765 → 0.516). The blank-arm vs blank-arm noise floor tightens too — the post-fix substrate has less stochasticity-driven cluster fragmentation overall.

These are substrate-quality improvements that benefit V1 cross-session validation directly, even though they don't move the Roy-2c headline.

## Reproduction

```bash
# Pre-flight (same as Roy-2c reproduction):
PYTHONPATH=src python -c "
import os
os.environ['MAXIM_NAC_MIN_CONFIDENCE'] = '0.0'
from maxim.runtime.agent_loop import _resolve_min_confidence
assert _resolve_min_confidence(None) == 0.0
print('env-var resolver OK')
"

# Verify EC default is 0.44 (Phase 3 shipped this):
PYTHONPATH=src python -c "from maxim.similarity.ec import ECConfig; assert ECConfig().pattern_complete_threshold == 0.44"

# Backup the pre-fix Roy-2c baseline before re-running:
cp -r ~/.maxim/roy/roy-2c ~/.maxim/roy/roy-2c-pre-fix-backup-$(date +%Y%m%d)

# Re-run with MAXIM_SUBSTRATE_PATH=1 + the standard Roy-2c env:
pkill -f "maxim.*sim" 2>/dev/null; sleep 2
PYTHONPATH=src \
MAXIM_SUBSTRATE_PATH=1 \
MAXIM_NAC_MIN_CONFIDENCE=0.0 \
MAXIM_LOG_FILE=/tmp/roy_2c_postfix.jsonl \
MAXIM_BACKEND_TRACE=1 \
  maxim roy run scenarios/roy/roy_2c_iteration.yaml 2>&1 | tee /tmp/roy_2c_postfix_run.log

# Compare structural numbers:
for arm in a b c; do
  sid=$(jq -r ".arms.$arm.session_id" ~/.maxim/roy/roy-2c/result.json)
  echo "=== arm $arm ==="
  jq -c '{tool, success, error}' ~/.maxim/sim_reports/$sid/actions.jsonl
done

# Cluster L2 comparison:
jq '.pairwise_diffs.a_vs_b.nac.cluster_reward_bias.l2' ~/.maxim/roy/roy-2c-pre-fix-backup-*/result.json
jq '.pairwise_diffs.a_vs_b.nac.cluster_reward_bias.l2' ~/.maxim/roy/roy-2c/result.json
```

Total cost: ~20 min wall, $0 (local qwen2.5-14b via cloudflared peer).

## What's next

**Phase 5** (doc-only thread-through) unblocked. Verdict for the operator-facing decision:

- **0.9.1 ship?** The fix is small (~10 src LOC + 80 test LOC + Phase 3.5 parameterization). Regression guards are pinned. P1 and P2 sweeps both improved. The structural improvement is real even if the Roy-2c behavioral signal is unchanged. **Argument for 0.9.1: downstream V1 cross-session runs benefit immediately; downstream Roy-3 measurement happens on cleaner clusters.** Argument for 1.0: thematically the EC fix is structurally adjacent to 0.9.1's "annotation patterns" theme but not aligned. Operator decides in Phase 5 PR.
- **JEPA / cross-modal binding priority:** unchanged. Stays a 1.0-or-1.1 gate per pre-registered routing.
- **Roy-5 / roy_5_encoder_alignment_disambiguator.md:** unchanged in scope. The cross-source alignment gap H1 named is a different mechanism from drift.

## PR

[TBD — will link after PR opens]
