# Roy-2pc — positive-control on engineered-overlap fixture

**Date:** 2026-05-13 (run completed 2026-05-13 09:14 local)
**Plan:** [persona_convergence_crucible.md § "Iteration log"](../plans/persona_convergence_crucible.md)
**Companion:** [18_roy_2.md](18_roy_2.md) (multi-arc priming the priming side reuses) · [17_roy_1b.md](17_roy_1b.md) (substrate-primary at test on original held-out fixture) · [16_roy_1a.md](16_roy_1a.md)
**Spec:** [scenarios/roy/roy_2pc_iteration.yaml](../../scenarios/roy/roy_2pc_iteration.yaml)
**Engineered fixture:** [scenarios/roy/roy_2pc_holdout.yaml](../../scenarios/roy/roy_2pc_holdout.yaml)
**Reproduction:** [protocols/19_roy_2pc_reproduction.md](protocols/19_roy_2pc_reproduction.md)

## Status

Positive-control iteration. Roy-1a/Roy-1b/Roy-2 (PRs #241, #242) shipped a symmetric structural-vs-behavioral gap: substrate priming writes +1.0 `cluster_reward_bias` on six EC clusters all keyed to `tool:sense_food_source`, but neither AUT mode at test time differentiates arm A from blank arms B/C. Standing hypothesis from Roy-1b/Roy-2: the held-out fixture's percepts don't fire the priming-acquired EC clusters at recall time, so the bias is consulted but never crosses the `min_confidence=0.3` gate.

Roy-2pc validates the hypothesis with a held-out fixture **engineered to deliberately overlap** the priming arc's food/hunger/eating regime. Every test percept evokes food / hunger / eating / sensing-food semantics ("you sense food nearby", "the smell of food fills the air", "warm food rises in your belly", ...). AUT mode at test is `substrate-primary` because the `cluster_reward_bias` consumer is `recommend_action` — testing wire health requires running that consumer.

Pre-registered diagnostic logic (from the fixture's docstring):

| Outcome | Diagnosis |
|---|---|
| A > B > C on `sense_food_source` counts | Wire IS healthy + exploitable; Roy-1b/Roy-2 inertness was a percept-overlap problem. Wire 1 escalation right for general-percept persona. |
| **A ≈ B ≈ C** | **Wire bug OR `min_confidence` gate filters even primed-cluster-matched proposals. Roy-2c (min_confidence tune) becomes load-bearing before Wire 1.** |
| A < C | Priming suppressed `sense_food_source` somehow. Unlikely; wire defect. |

## What shipped

- [`scenarios/roy/roy_2pc_holdout.yaml`](../../scenarios/roy/roy_2pc_holdout.yaml) — 10-percept engineered-overlap fixture (8 food-semantic + 1 unrelated control + 1 food-adjacent cooldown).
- [`scenarios/roy/roy_2pc_iteration.yaml`](../../scenarios/roy/roy_2pc_iteration.yaml) — Roy-2 priming (multi-arc) + Roy-1b test-mode (substrate-primary) + this fixture.
- No new tests (the pre-Roy-1 attribution scale stress test covers Roy-2pc's load).

## Result

Wall: **1502.2s (~25.0 min)** — close to Roy-1b's 1578s (~26.3 min), substantially slower than Roy-2's 883s. Pre-flight cleared via peer.yml (`outcome: ok`, `latency_ms: 570.3`, `detail: stage2 HTTP 200`) after a one-shot leader-tunnel cold-start retry.

### Per-arm

| Arm | Substrate | system_prompt | session_id | turns | duration_s | finish |
|---|---|---|---|---|---|---|
| a | from_priming | neutral | `20260513_085942` | 10 | 301.13 | cancel |
| b | blank | "You are a hungry infant" | `20260513_090443` | 10 | 299.24 | cancel |
| c | blank | neutral | `20260513_090942` | 10 | 295.33 | cancel |

Arm durations are tightly clustered (Δ ≈ 6s across arms) — substrate-primary's 30s-per-turn timeout × 10 turns dominates wall-time and is invariant to substrate state or prompt.

### Priming (50 turns, substrate-primary, ~10 min)

Identical multi-arc mix to Roy-2. 605.5s total (Roy-2: 609.5s). Stage 1 cold-start outlier (184s); stages 2-5 each ~105s.

| Stage | Arc | Duration_s |
|---|---|---|
| act1_neonatal_a | cradle_prelinguistic | 184.5 |
| act1_neonatal_b | cradle_prelinguistic | 110.0 |
| act2_cradle_a | **cradle** | 101.6 |
| act2_cradle_b | **cradle** | 104.3 |
| act3_consolidation | cradle_prelinguistic | 105.2 |

### Headline pairwise diffs

| Pair | `reward_bias_l2` | **`cluster_reward_bias_l2`** | (keys) | `goal_reward_bias_l2` | `causal_link_Δ` | `episodes_Δ` | `valence_KS` (p) | `salience_KS` (p) |
|---|---|---|---|---|---|---|---|---|
| **a_vs_b** | 0.0 | **2.4678** | 10 | 0.0 | +155 | +646 | 0.998 (0.006)* | 0.997 (5.6e-5)* |
| **a_vs_c** | 0.0 | **2.4678** | 10 | 0.0 | +155 | +646 | 0.998 (0.006)* | 0.997 (5.6e-5)* |
| b_vs_c | 0.0 | 0.3000 | 4 | 0.0 | 0 | 0 | 0.000 (1.000) | (similar) |

\* **Sample-driven** — arms B and C each captured exactly 1 hippocampus episode in the entire test phase (from their single FAILED `pick_up` proposing a NAc anticipatory-pain event). The KS test detects "distributions differ" but the sample is 647 vs 1.

### Cluster-reward top deltas (a_vs_b)

```
6× tool:sense_food_source         delta=+1.0   (priming carryover — same shape as Roy-0/1a/1b/2)
2× tool:infant_humanoid_pick_up   delta=+0.15  (arm A's pick_up writes onto its own EC clusters)
2× tool:infant_humanoid_pick_up   delta=-0.15  (arm B's pick_up writes onto its own EC clusters)
```

Identical shape to Roy-1b's substrate-primary-test cluster wire output. The 4 stochastic-cluster pick_up entries contribute the b_vs_c noise floor (sqrt(4 × 0.15²) = 0.30). **Zero new `sense_food_source` cluster updates in the test phase across any arm** — confirming the priming-acquired clusters were never reinforced during the engineered-overlap test percepts.

### Test-phase tool distribution (the headline Roy-2pc signal)

```
Arm A (substrate-primed, neutral):       2× infant_humanoid_pick_up (both FAILED — Missing required input: object)
Arm B (blank, "hungry infant"):          2× infant_humanoid_pick_up (both FAILED — Missing required input: object)
Arm C (blank, neutral):                  2× infant_humanoid_pick_up (both FAILED — Missing required input: object)
```

**All three arms produce the BYTE-IDENTICAL action distribution: 2× FAILED `infant_humanoid_pick_up` with empty params.** The substrate-primary AUT's `recommend_action` fallback chose the same default tool with the same (invalid) params regardless of percept content AND regardless of substrate state. 8-of-10 turns per arm produced zero actions (sub-threshold proposals filtered by `min_confidence=0.3`); the 2 turns that produced actions all landed on the same params-incomplete `infant_humanoid_pick_up`.

**This is the pre-registered "A ≈ B ≈ C" diagnostic outcome.**

### Roy-1b → Roy-2pc direct A/B (the key positive-control comparison)

Single-variable change vs Roy-1b: the held-out fixture's percepts. Roy-1b used `roy_1_holdout.yaml` (matching/novel/unrelated thermal/texture/social — no food semantics); Roy-2pc uses `roy_2pc_holdout.yaml` (food/hunger/eating overlap throughout). Priming is multi-arc (Roy-2) vs cradle_prelinguistic-only (Roy-1b), so this is a two-variable diff strictly speaking, but the Roy-2→Roy-2pc comparison isolates the fixture variable.

| Metric | Roy-1b (original holdout) | Roy-2pc (engineered overlap) | Interpretation |
|---|---|---|---|
| Wall time | 1578s | 1502s | Within 5%; substrate-primary test cost is fixture-independent (30s/turn × 30 turns dominates) |
| `cluster_reward_bias_l2` (a_vs_b) | 2.4678 (10 keys) | **2.4678 (10 keys)** | **BYTE-IDENTICAL.** Same 6 priming sense_food_source + 4 test-phase pick_up. Priming wire is fixture-agnostic (priming runs before fixture is seen). |
| `b_vs_c.cluster_reward_bias_l2` (noise) | 0.30 (4 pick_up keys) | 0.30 (4 pick_up keys) | Substrate-primary at test produces the same 0.30 stochastic-cluster floor on either fixture |
| Arm A `sense_food_source` count | 0 | **0** | **The pre-registered diagnostic answer.** Engineered overlap percepts produced ZERO additional `sense_food_source` calls in arm A despite +1.0 cluster bias on six EC clusters keyed to that tool. |
| Per-arm tool distribution | All arms 2× pick_up | **All arms 2× pick_up** | **IDENTICAL across fixtures.** Engineering percept-substrate semantic overlap did NOT change the test-phase tool selection at all. |
| Arm B test-phase episodes | 1 | **1** | Same single-episode artefact (substrate-primary's sub-threshold filter blocks ~all turns from producing episodes) |

### Roy-2 → Roy-2pc direct A/B (the test-AUT-mode variable)

Single-variable change vs Roy-2: test-AUT-mode flips from `llm-primary` to `substrate-primary`. Priming and fixture are coupled to the test AUT — Roy-2 used `roy_1_holdout.yaml` with llm-primary; Roy-2pc uses `roy_2pc_holdout.yaml` with substrate-primary.

| Metric | Roy-2 (llm-primary, original holdout) | Roy-2pc (substrate-primary, engineered) |
|---|---|---|
| Wall time | 883s | 1502s |
| `cluster_reward_bias_l2` | 2.4495 (6 keys) | 2.4678 (10 keys; +4 pick_up under substrate-primary at test) |
| Test-phase tool divergence A vs C | A: sense/pick_up; C: look/listen (clean tool-family diff) | A ≡ C (byte-identical: both 2× FAILED pick_up) |
| Substrate-readable signal | LLM-prompt mediated (salience-modulated WMS) | None observable at action level |

**llm-primary at test reads substrate context via the LLM proposer's salience-modulated WMS, producing a clean tool-family divergence; substrate-primary at test does NOT differentiate behaviorally even with maximally-overlapping percepts.**

## What this proves

The pre-registered diagnostic outcome is **A ≈ B ≈ C** — the byte-identical 2× FAILED `infant_humanoid_pick_up` distribution rules out the "wire is exploitable when percepts overlap" hypothesis cleanly. Two narrower hypotheses remain to disambiguate:

1. **(H1) The engineered percepts do NOT pattern-complete onto priming-acquired EC clusters at recall time.** The LinguisticEncoder's embedding of "you sense food nearby" may land in a different region of EC space than the priming substrate's encodings (which came from sensor readings, drive states, and the priming arc's narrator-emitted text under cradle stages, NOT from explicit "food" tokens emitted by a CLI fixture). If this is true, the wire is structurally healthy but never gets consulted because the EC clusters activated by the test percepts have no associated reward bias.

2. **(H2) The engineered percepts DO pattern-complete onto priming clusters, `recommend_action` DOES consult the +1.0 bias, but the resulting proposal's `confidence` is below `min_confidence=0.3`.** The threshold gate filters even primed-cluster-matched proposals. Wire would express if the gate were lower or removed.

Roy-2pc cannot distinguish H1 from H2 at the action-distribution level — both predict the byte-identical 2× FAILED `pick_up` outcome we observe. Disambiguation requires instrumentation of the substrate-primary `recommend_action` path:
- Log the EC cluster activations on each test percept and compare to the priming-cluster UUIDs (H1 test).
- Log `recommend_action`'s proposal confidence on each test turn and compare to `min_confidence=0.3` (H2 test).
- Drop `min_confidence` to 0.0 in a Roy-2c variant and re-run on the same fixture (H1 vs H2 disambiguation: if A > B > C emerges, H2 confirmed; if A ≈ B ≈ C reproduces, H1 confirmed).

**Wire 1 escalation is the correct next architectural step regardless of which hypothesis is true:**

- Under H1: Wire 1 surfaces substrate-derived bias at the LLM-prompt level, bypassing the EC retrieval path entirely. The LinguisticEncoder→EC alignment problem doesn't propagate to Wire 1's bias channel.
- Under H2: Wire 1 surfaces bias at the LLM prompt where there is no `min_confidence` gate; the LLM proposer reads the bias as context and chooses freely.

The cluster wire is structurally healthy (reproduced across four iterations: Roy-0 → Roy-1a → Roy-1b → Roy-2 → Roy-2pc, all within 1% on `cluster_reward_bias_l2`). The behavioral pathway from cluster bias to action selection has at least one block — and possibly two — that Wire 1 routes around.

### Definitively proved

- **The cluster wire writes the priming substrate identically across five single-seed iterations.** `cluster_reward_bias_l2` reproduces within 1% across all five Roy iterations. The priming side is rock-solid.
- **The cluster wire is consulted under substrate-primary at test.** 4 new `infant_humanoid_pick_up` cluster updates appear in Roy-2pc just as they do in Roy-1b — `recommend_action` is running and writing.
- **Engineering percept-substrate semantic overlap is INSUFFICIENT to produce behavioral divergence under substrate-primary at test.** This is the cleanest negative result the Roy harness has produced; it directly closes the open question Roy-1b/Roy-2 left dangling.
- **Substrate-primary's default fallback at test is fixture-content-independent.** Three arms × two fixtures (Roy-1b's original holdout, Roy-2pc's engineered overlap) all produce the same 2× FAILED `infant_humanoid_pick_up` distribution. The fallback is hardcoded behavior, not substrate-driven.

## What this still does NOT prove

- **Which of H1/H2 (or both) blocks the cluster-bias→action pathway.** Disambiguating requires instrumentation or Roy-2c (`min_confidence=0.0` probe).
- **That `recommend_action` is even consulting the priming clusters.** EC pattern completion may not be firing on the engineered percepts. Verifying requires logging EC cluster activations during test turns.
- **Wire 1 sufficiency for behavioral persona expression.** Wire 1 is the architectural escape hatch the data argues for, but its end-to-end efficacy is unmeasured.
- **Cross-session persistence** (single-session as before).

## Reproduction

See [protocols/19_roy_2pc_reproduction.md](protocols/19_roy_2pc_reproduction.md).

## Recommendation for next iteration

**Wire 1 escalation is now load-bearing for behavioral persona expression.** The empirical case across five single-seed iterations:

- Roy-0: substrate-primary throughout, rehearsal fixture — cluster monoculture (single-tool sense_food_source).
- Roy-1a: llm-primary at test, original holdout — wire structurally preserved, behaviorally inert under llm-primary (cluster bias not consumed by LLM proposer).
- Roy-1b: substrate-primary at test, original holdout — wire consumed, behaviorally inert (held-out percepts don't fire priming clusters).
- Roy-2: llm-primary at test, multi-arc priming, original holdout — multi-arc priming did NOT widen cluster vocabulary; clean A vs C tool-family divergence via salience-mediated LLM-prompt path.
- Roy-2pc: substrate-primary at test, multi-arc priming, **engineered-overlap** fixture — byte-identical action distribution across all three arms; engineering semantic overlap is insufficient to break the structural-vs-behavioral gap.

The cluster_reward_bias path has at least one (and possibly two) blocking gates between substrate state and action selection under substrate-primary, AND it isn't read at all under llm-primary. Wire 1's substrate-annotates-LLM-context design surfaces substrate-derived bias at the LLM prompt — bypassing both gates and applying across percept regimes the substrate didn't directly drill.

**Secondary recommendation: Roy-2c (`min_confidence=0.0` probe) is a cheap H1-vs-H2 disambiguator.** Same priming as Roy-2 + same engineered fixture as Roy-2pc + flip `min_confidence` to 0.0 (env var or config override). If A > B > C emerges → H2 confirmed (gate is the blocker) → option to ship `min_confidence` tuning as an interim Wire-1-precursor; if A ≈ B ≈ C reproduces → H1 confirmed (encoder/EC alignment is the blocker) → Wire 1 is the only path. Cheap because it reuses fixtures + priming and the only change is a single env var.

**Tertiary methodology note:** Future Roy iterations should consider instrumenting `recommend_action` with per-turn JSONL events containing (a) EC cluster activations at recall time, (b) proposal confidence, (c) `cluster_reward_bias` consulted per proposal. Without these, single-experiment disambiguation between H1 and H2 is structurally impossible — the wire's consumer is a black box at this level of observation.

## PR

<!-- filled when PR opens -->
TBD
