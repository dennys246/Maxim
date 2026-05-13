# Roy-2c — `min_confidence=0.0` probe (H1 vs H2 disambiguator)

**Date:** 2026-05-13 (run completed 2026-05-13 12:43 local)
**Plan:** [release_0_9_1.md § Stage 0a](../plans/release_0_9_1.md) · [persona_convergence_crucible.md § "Iteration log"](../plans/persona_convergence_crucible.md)
**Companion:** [19_roy_2pc.md](19_roy_2pc.md) (Roy-2pc shipped the A ≈ B ≈ C outcome Roy-2c disambiguates) · [18_roy_2.md](18_roy_2.md) · [17_roy_1b.md](17_roy_1b.md)
**Spec:** [scenarios/roy/roy_2c_iteration.yaml](../../scenarios/roy/roy_2c_iteration.yaml)
**Engineered fixture:** [scenarios/roy/roy_2pc_holdout.yaml](../../scenarios/roy/roy_2pc_holdout.yaml) (reused unchanged from Roy-2pc)
**Reproduction:** [protocols/20_roy_2c_reproduction.md](protocols/20_roy_2c_reproduction.md)

## Status

H1-vs-H2 disambiguator. Single-variable change vs Roy-2pc: `MAXIM_NAC_MIN_CONFIDENCE=0.0` set in the runner environment (new env var introduced in [release_0_9_1.md Stage 0a](../plans/release_0_9_1.md)). Same priming, same fixture, same arms.

**Pre-registered diagnostic logic:**

| Outcome | Diagnosis |
|---|---|
| A > B > C on `sense_food_source` counts | **H2 confirmed** — gate was the block. Lower threshold rescues the wire. Wire-A still ships (gate-tuning is interim only). |
| **A ≈ B ≈ C reproduces** | **H1 confirmed** — LinguisticEncoder → EC alignment is the block. Wire-A is the only architectural fix; no gate-tuning rescues a wire that's never consulted. |
| A < C | Unexpected (priming somehow suppressed `sense_food_source`); investigate before Wire-A design. |

## What shipped

- [`scenarios/roy/roy_2c_iteration.yaml`](../../scenarios/roy/roy_2c_iteration.yaml) — single-env-var variant of `roy_2pc_iteration.yaml`.
- `MAXIM_NAC_MIN_CONFIDENCE` env-var override at `agent_loop._resolve_min_confidence` ([release_0_9_1.md Stage 0a](../plans/release_0_9_1.md)).
- `conftest.py` autouse scrub fixture (`_isolate_maxim_nac_min_confidence`) per [feedback_opt_in_env_in_hot_paths.md](../../.claude/projects/-Users-dennyschaedig-Scripts-Maxim/memory/feedback_opt_in_env_in_hot_paths.md).
- 8 new unit tests in [`tests/unit/test_substrate_min_confidence_env.py`](../../tests/unit/test_substrate_min_confidence_env.py) covering precedence, explicit-zero honoring, invalid fallback, empty-string semantics.

## Result

Wall: **1284.2s (~21.4 min)** — faster than Roy-2pc's 1502s (~25.0 min). Lower gate accepts more proposals per turn → less wall-clock burned on the 30s timeout. Pre-flight clean (`outcome: ok`, `latency_ms: 228.4`).

### Per-arm

| Arm | Substrate | system_prompt | session_id | turns | duration_s | finish |
|---|---|---|---|---|---|---|
| a | from_priming | neutral | `20260513_123436` | 10 | 224.93 | cancel |
| b | blank | "You are a hungry infant" | `20260513_123821` | 10 | 241.82 | cancel |
| c | blank | neutral | `20260513_124223` | 10 | 238.57 | cancel |

Arm durations dropped 25-30% vs Roy-2pc (225-242s vs 295-336s) — lower gate accepts proposals faster.

### Headline pairwise diffs

| Pair | `reward_bias_l2` | **`cluster_reward_bias_l2`** | (keys) | `causal_link_Δ` | `episodes_Δ` | `valence_KS` (p) | `salience_KS` (p) |
|---|---|---|---|---|---|---|---|
| **a_vs_b** | 0.0 | **2.5661** | 10 | +147 | +664 | 0.994 (1.7e-8)* | 0.826 (6.3e-5) |
| **a_vs_c** | 0.0 | **2.5661** | 10 | +147 | +664 | 0.994 (1.7e-8)* | (similar) |
| b_vs_c | 0.0 | 0.7649 | 4 | 0 | 0 | 0.000 (1.000) | (similar) |

\* Sample-asymmetry caveat from Roy-2pc partially relaxed: arms B and C each captured **4 hippocampus episodes** (vs Roy-2pc's 1), but all 4 are FAILED pick_ups with valence -1.0 — KS still detects "distributions differ" but the blank-arm distribution is a 4-point spike at -1.0. Not a clean persona-convergence signal; arm B's episode pipeline is producing replicated single-failure-mode events, not a varied test-percept response.

### Cluster-reward top deltas (a_vs_b)

```
6× tool:sense_food_source        delta=+1.0    (priming carryover, UNCHANGED from Roy-2pc — priming clusters never updated during test)
4× tool:infant_humanoid_pick_up  deltas=±0.30, ±0.45  (test-phase updates on FOUR NEW EC clusters, disjoint from priming's six)
```

**The structural finding.** The 4 test-phase cluster updates land on **entirely new** EC cluster UUIDs — disjoint from the 6 priming-acquired UUIDs. If the engineered percepts had pattern-completed onto priming clusters, we'd see the +1.0 entries shift to ±0.85 / ±1.15 (priming bias + test-phase positive/negative outcome). Instead the priming clusters sit at exactly +1.0 (no test-phase activity touched them) AND new pick_up clusters appear at modest ±0.30-0.45. **The cluster sets are disjoint.**

### Test-phase tool distribution (the load-bearing Roy-2c signal)

```
Arm A (substrate-primed, neutral):       5× infant_humanoid_pick_up (all FAILED — Missing required input: object)
Arm B (blank, "hungry infant"):          5× infant_humanoid_pick_up (all FAILED — Missing required input: object)
Arm C (blank, neutral):                  5× infant_humanoid_pick_up (all FAILED — Missing required input: object)
```

**All three arms produced the BYTE-IDENTICAL action distribution: 5× FAILED `infant_humanoid_pick_up`.** Per-arm action count increased from Roy-2pc's 2 to Roy-2c's 5 — the gate WAS active in Roy-2pc; it filtered 3 sub-threshold proposals per arm. But **zero of those newly-accepted proposals are `sense_food_source`** despite arm A carrying +1.0 cluster_reward_bias on six EC clusters keyed to that tool. The 5 of 10 turns that still produced no action are pure-zero proposals (recommend_action returned None with literally no candidate above 0.0).

**This is the pre-registered "A ≈ B ≈ C reproduces with gate dropped" outcome → H1 confirmed.**

### Roy-2pc → Roy-2c direct A/B (the disambiguator)

Single-variable change vs Roy-2pc: env var `MAXIM_NAC_MIN_CONFIDENCE=0.0`. Priming, fixture, arms byte-identical.

| Metric | Roy-2pc (gate=0.3, default) | Roy-2c (gate=0.0) | What the diff means |
|---|---|---|---|
| Wall time | 1502s | 1284s | Lower gate → less wall burned on sub-threshold timeouts |
| Per-arm action count | 2 | **5** | **Gate WAS active.** 3 sub-threshold proposals per arm now accepted under gate=0.0. |
| Tool family of accepted proposals | `infant_humanoid_pick_up` | `infant_humanoid_pick_up` | **Tool selection unchanged.** Newly-accepted proposals are the same tool, not `sense_food_source`. |
| Arm A `sense_food_source` count | 0 | **0** | **The H1-confirming result.** Lower gate did not surface `sense_food_source` despite arm A's +1.0 cluster bias on six EC clusters. |
| `cluster_reward_bias_l2` (a_vs_b) | 2.4678 | 2.5661 | +4%; 4 test-phase pick_up keys grew from ±0.15 to ±0.30/±0.45 (more accepted proposals → larger cluster updates) |
| `b_vs_c.cluster_reward_bias_l2` (noise) | 0.30 | 0.76 | +154%; stochastic-cluster floor scales with accepted-proposal count |
| Test-phase EC clusters activated | 4 new pick_up clusters | 4 new pick_up clusters | **Same disjoint cluster shape across both gate settings.** Engineered percepts activate a different EC region than priming substrate, gate-independent. |
| Arm B test-phase episodes | 1 | **4** | More accepted proposals → more captured episodes; sample asymmetry relaxed but blank-arm episode distribution is still a single-failure-mode spike |
| `valence_KS` p-value | 0.006 | **1.7e-8** | Sample size grew (1→4 in blank arm); KS gets stronger but blank-arm distribution is still pathological (4 identical -1.0 episodes) |

## What this proves

**H1 confirmed cleanly: LinguisticEncoder → EC alignment is the block.**

The disambiguation comes from two independent observables in Roy-2c, not just from the headline tool count:

1. **Per-arm action count rose 2 → 5.** Roy-2c's gate=0.0 demonstrated the threshold gate WAS filtering proposals in Roy-2pc — three additional `infant_humanoid_pick_up` proposals per arm crossed the threshold. The mechanism is real and observable.
2. **None of the newly-accepted proposals are `sense_food_source`.** If H2 (gate filtering primed-cluster-matched proposals) were the true block, dropping the gate to 0.0 would have surfaced at least one `sense_food_source` call in arm A — arm A's +1.0 cluster bias on six EC clusters is the strongest possible positive signal in the priming substrate. The fact that zero `sense_food_source` proposals crossed the threshold even at gate=0.0 means **`recommend_action` is never seeing those priming clusters as the active cluster on these engineered percepts**.
3. **The four test-phase EC cluster updates are disjoint from the six priming clusters.** If the engineered percepts had pattern-completed onto priming clusters, the +1.0 entries would have shifted to ±0.85 / ±1.15 reflecting test-phase positive/negative outcome contributions. Instead the priming clusters sit at unchanged +1.0 (no test-phase activity touched them) AND four new pick_up clusters appear at modest ±0.30-0.45. The cluster sets are structurally disjoint.

**The structural diagnosis:** LinguisticEncoder embeds the priming substrate's WMS contents (sensor/drive state + cradle-stage narrator output — "the infant is in a room with a fire pit nearby" / drive-value text) into one EC region. The engineered test percepts ("you sense food nearby" / "the smell of food fills the air") embed into a *different* region, even though humans read the semantic overlap as obvious. The cluster_reward_bias map has the right *tool* keys (`sense_food_source` is the tool the priming arc reinforced) but the wrong *cluster* keys (the priming UUIDs are never the active cluster on test percepts).

This is **not** a wire bug — the cluster wire is structurally healthy and consumed correctly. It is an **embedding-space alignment problem** between two different sources of substrate input (sensor/drive vs CLI-percept text) that share semantic content but produce different embeddings.

### Definitively proved
- The `min_confidence=0.3` gate WAS filtering proposals in Roy-2pc (Roy-2c's per-arm action count rising 2 → 5 demonstrates this).
- Lowering the gate does NOT rescue the cluster wire on engineered-overlap test percepts (zero `sense_food_source` calls despite +1.0 cluster bias).
- The priming-acquired EC clusters and the test-phase EC clusters are structurally disjoint under LinguisticEncoder embedding.
- `MAXIM_NAC_MIN_CONFIDENCE` env-var override works end-to-end (the experimental knob this disambiguator required).
- The cluster wire reproduces SIXTH iteration in a row (Roy-0 → Roy-1a → Roy-1b → Roy-2 → Roy-2pc → Roy-2c), all priming-side `cluster_reward_bias_l2` values within 5%.

### Cleanly refuted
- **H2 (gate filtering primed-cluster-matched proposals).** Refuted by zero `sense_food_source` calls under gate=0.0.

### Still unfalsified (but H1 is the dominant explanation)
- **Whether a NEW LinguisticEncoder, or sensor-encoding entry point for engineered fixture percepts, would rescue the cluster wire.** Unmeasured; not on 0.9.1's critical path.

## What this means for 0.9.1

H1 confirmation **strengthens** the Wire-A (cluster-bias annotation) case. The cluster wire writes the substrate correctly but the consumer (`recommend_action`) cannot read the bias on percepts that activate different EC clusters than priming did. Wire-A bypasses the EC retrieval path entirely by surfacing the cluster_reward_bias map at the LLM prompt — the LLM proposer sees "sense_food_source [strongly rewarding from prior experience]" as a tool-level hint regardless of which EC clusters the current percept activated.

No 0.9.1 plan changes required. Roy-2c confirms the architectural fix the plan already specifies.

**Secondary observations for the 0.9.1 plan:**

- **Wire-A's bias rendering should NOT depend on `current_cluster_id` matching.** Stage 2 of [release_0_9_1.md](../plans/release_0_9_1.md) currently spec'd `_collect_active_clusters` + `get_active_cluster_biases` which keyed on active cluster IDs. H1 confirmation suggests the activated-cluster intersection with priming clusters may be empty often. **Revise: Wire-A should aggregate `_cluster_reward_bias` across ALL priming-acquired clusters for the agent**, not just clusters matching the current percept. The tool-name aggregation is the right granularity. Active-cluster restriction is the bug that motivated the wire's existence.
- **min_confidence default stays at 0.3.** Roy-2c's gate=0.0 produced *more accepted-but-still-wrong* proposals. The gate is doing real work blocking sub-threshold noise; lowering it is not a winning move on its own.

## What this still does NOT prove

- **Whether Wire-A's annotation, once shipped, reaches the LLM proposer's decision pathway under prompt budget pressure.** Roy-3 (Stage 5 of 0.9.1) is the test.
- **Whether a more aligned encoder (e.g., a NEW sensor-encoding entry point for CLI fixtures) would rescue the cluster wire.** Untested; out of 0.9.1 scope.
- **Cross-session persistence** (single-session as before).

## Reproduction

See [protocols/20_roy_2c_reproduction.md](protocols/20_roy_2c_reproduction.md).

## Recommendation

**0.9.1 plan proceeds unchanged on critical path.** Wire-A's design needs one revision: aggregate `_cluster_reward_bias` across all priming-acquired clusters per `agent_id`, not just clusters matching the current percept's active set. The Stage 2 sizing estimate (~150 LOC) is unaffected.

**Defer to post-0.9.1:** the question of whether a sensor-encoding entry point for CLI fixtures (or a more aligned LinguisticEncoder configuration) would rescue the cluster wire is a research question, not a 0.9.1 blocker. Wire-A's annotation pattern routes around the encoder alignment problem regardless.

**No further Roy-2 sub-iterations planned.** Roy-2c definitively narrows the hypothesis space; Roy-3 (post-wires) is the next iteration on the harness.

## PR

https://github.com/dennys246/Maxim/pull/244 — bundles Roy-2pc commits because PR #243's stacked-merge target was the Roy-2 branch, not main.
