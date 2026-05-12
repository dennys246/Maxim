# Roy-1b — substrate-primary on held-out fixture (second methodology iteration)

**Date:** 2026-05-12 (run completed 2026-05-11 23:17 local)
**Plan:** [persona_convergence_crucible.md § "Iteration log"](../plans/persona_convergence_crucible.md)
**Companion:** [16_roy_1a.md](16_roy_1a.md) (A/B partner) · [15_g4_cluster_reward_wire.md](15_g4_cluster_reward_wire.md) (Roy-0 baseline)
**Spec:** [scenarios/roy/roy_1b_iteration.yaml](../../scenarios/roy/roy_1b_iteration.yaml)
**Held-out fixture:** [scenarios/roy/roy_1_holdout.yaml](../../scenarios/roy/roy_1_holdout.yaml)
**Reproduction:** [protocols/17_roy_1b_reproduction.md](protocols/17_roy_1b_reproduction.md)

## Status

Second "real" Roy iteration. Single-variable change vs Roy-1a: test-time AUT mode flips from `llm-primary` to `substrate-primary`. Priming, held-out fixture, and arms are byte-identical to Roy-1a.

Roy-1a's open question (left for Roy-1b): substrate priming wrote +1.0 cluster_reward_bias on six EC clusters for `tool:sense_food_source` — but arm A's llm-primary proposer never invoked the tool at test time, because the `cluster_reward_bias` path is consumer-coupled to substrate-primary's `recommend_action`. Roy-1b runs that consumer at test time so we can directly measure whether the priming bias is exploitable when its consumer fires.

## What shipped

- `scenarios/roy/roy_1b_iteration.yaml` — single-line change vs `roy_1a_iteration.yaml` at top-level `aut_mode` (`llm-primary` → `substrate-primary`).
- No new fixture (re-uses Roy-1a's `roy_1_holdout.yaml`).
- No new tests (the pre-Roy-1 scale stress test from `test_multi_agent_attribution_scale.py` covers Roy-1b's load too).

## Result

Wall: **1578.4s (~26.3 min)** — almost double Roy-1a's 830s. Substrate-primary at test time is *slower* despite no LLM in the AUT loop because every turn waits ~25s for the substrate-primary proposer to converge or fall through the `min_confidence=0.3` gate; under llm-primary the LLM call returns deterministically in ~6s. Pre-flight cleared via peer.yml (`outcome: ok`, `latency_ms: 300.2`).

### Per-arm

| Arm | Substrate | system_prompt | session_id | turns | duration_s | finish |
|---|---|---|---|---|---|---|
| a | from_priming | neutral | `20260511_230124` | 10 | 312.3 | cancel |
| b | blank | "You are a hungry infant" | `20260511_230636` | 10 | 335.6 | cancel |
| c | blank | neutral | `20260511_231212` | 10 | 301.2 | cancel |

25 `peer_backend_call` events (status 200), 10 narrator generations, 0 `dispatch_exhausted` warnings, 0 tracebacks.

### Headline pairwise diffs

| Pair | `reward_bias_l2` | **`cluster_reward_bias_l2`** | (keys differ) | `causal_link_Δ` | `episodes_Δ` | **`valence_KS`** (p) | **`salience_KS`** (p) |
|---|---|---|---|---|---|---|---|
| **a_vs_b** | 0.0 | **2.4678** | 10 | +157 | +654 | **0.998 (p=0.006)** | **0.997 (p=5.5e-5)** |
| **a_vs_c** | 0.0 | **2.4678** | 10 | +157 | +654 | **0.998 (p=0.006)** | (similar) |
| b_vs_c | 0.0 | 0.3000 | 4 | 0 | 0 | 0.000 (1.000) | 0.0 |

### Cluster-reward top deltas (a_vs_b)

```
6× tool:sense_food_source         delta=+1.0   (priming carryover — same six EC cluster ids as Roy-0/Roy-1a)
2× tool:infant_humanoid_pick_up   delta=+0.15  (test-time arm A pick_up succeeded on those EC clusters)
2× tool:infant_humanoid_pick_up   delta=-0.15  (test-time arm B pick_up succeeded on its own EC clusters)
```

The 4 new pick_up entries are mode-driven: substrate-primary's `recommend_action` fires at test time on every arm, producing cluster updates. Net L2 contribution from the 4 pick_up entries: sqrt(4 × 0.15²) = 0.30 — exactly the `b_vs_c.cluster_reward_bias_l2` (the noise floor under substrate-primary test arms).

### Test-phase tool distribution (the key Roy-1b signal)

```
Arm A (substrate-primed, neutral):       2× infant_humanoid_pick_up
Arm B (blank, "hungry infant"):          2× infant_humanoid_pick_up
Arm C (blank, neutral):                  2× infant_humanoid_pick_up
```

**All three arms produce the identical action distribution.** No `sense_food_source` calls in any arm despite arm A carrying +1.0 cluster_reward_bias for that tool. The priming substrate did NOT differentiate arm A's behavior from the blank arms.

8 of 10 turns per arm produced ZERO actions (sub-threshold proposals filtered by `min_confidence=0.3`); only 2 of 10 produced a single `pick_up`. The held-out fixture's percepts don't fire the EC clusters the priming bias is keyed on, so the bias is consulted but doesn't clear the gate.

### Direct A/B against Roy-1a

| Metric | Roy-1a (llm-primary @ test) | Roy-1b (substrate-primary @ test) | What the diff means |
|---|---|---|---|
| Wall time | 830s (13.8 min) | 1578s (26.3 min) | substrate-primary test arms are slower per turn (~25s waiting for proposer convergence/threshold filter vs ~6s for an LLM call) |
| `cluster_reward_bias_l2` (a_vs_b) | 2.4495 (6 keys) | **2.4678 (10 keys)** | Priming-carried 6 keys unchanged; substrate-primary at test added 4 new pick_up keys (cluster updates from test-phase activity). Cluster wire still works as designed. |
| `b_vs_c.cluster_reward_bias_l2` (noise) | 0.0 | **0.3000** | Substrate-primary test arms produce a 0.30 stochastic-cluster floor (same shape as Roy-0's 0.21). llm-primary has no floor because it doesn't update clusters. |
| A-vs-blank signal:noise (cluster) | ∞ | 8.2× | Roy-1a's "infinite" SNR was an artifact of llm-primary's zero floor; substrate-primary's 8.2× is the load-bearing-comparable SNR |
| Arm A `sense_food_source` count | 0 | **0** | **The headline result.** Substrate priming with +1.0 bias on sense_food_source did not produce a single test-time call under either AUT mode. The bias is structurally present and never crossed the `min_confidence` threshold on held-out percepts. |
| `salience_KS` (a_vs_b, p) | 0.879 (2.1e-9) | **0.997 (5.5e-5)** | Both highly significant; salience carryover slightly stronger under substrate-primary at test (more episodes captured during priming match the test-percept similarity space more cleanly). |
| `valence_KS` (a_vs_b, p) | 0.283 (0.402) | **0.998 (0.006)** | Roy-1a missed α=0.05; Roy-1b clears it strongly — BUT *sample-driven*: arm B captured only 1 episode under substrate-primary (mean -1.0 from a pick_up failure) vs arm A's 655 priming episodes (mean -0.09). The KS detects "these distributions differ" because B's distribution is a single point. Not a clean persona-convergence signal. |
| `goal_reward_bias_l2` (a_vs_b) | 0.2714 | 0.0011 | Roy-1a's goal bias came from llm-primary's tool-outcome credit; Roy-1b's substrate-primary AUT bypasses goal credit (cluster reward is the substrate-primary credit path). |

## What this proves

The headline finding of Roy-1b is **negative for the behavioral question** and **positive for the structural question:**

1. **The cluster wire works as designed.** Substrate-primary at test time DOES consume `_cluster_reward_bias` and DOES produce new cluster updates. The L2 grew from Roy-1a's 2.4495 (6 keys, all priming) to 2.4678 (10 keys, 6 priming + 4 test-time). The wire is structurally healthy across both AUT modes.

2. **The bias does NOT differentiate behavior on held-out percepts.** All three arms (primed-neutral, blank-prompt-injected, blank-neutral) produced the identical action distribution: 2 `pick_up` calls per arm, 8 sub-threshold turns. The +1.0 priming bias on six `sense_food_source` clusters was never read into a test-phase action proposal because the held-out fixture's percepts don't fire the priming-acquired EC clusters. The `min_confidence=0.3` gate filters proposals on the priming clusters out, and the fallback selection is the same across arms.

3. **Roy-1a + Roy-1b together show the structural-vs-behavioral gap is SYMMETRIC.** Neither llm-primary AUT (which doesn't consume the bias at all) nor substrate-primary AUT (which consumes it but on EC clusters that don't fire on held-out percepts) behaviorally expresses substrate priming under this fixture. The bias persists in NAc; it doesn't drive action selection.

4. **The Hippocampus salience signal Roy-1a discovered (KS=0.879) is reproduced and strengthened by Roy-1b (KS=0.997, p=5.5e-5).** Salience scoring is the one bio-pipeline metric where substrate carryover translates into a quantitative, statistically-significant test-time difference — across both AUT modes. **This is the load-bearing positive finding for the methodology.** Salience is consumed by ThoughtGate + WMS during deliberation; substrate priming modulates this signal regardless of which proposer fires.

5. **The valence_KS jump (0.283/p=0.402 → 0.998/p=0.006) is statistically real but methodologically misleading.** Arm B under substrate-primary captured only 1 episode total (the single pick_up failure produced a strongly-negative episode); arm A's distribution comes from 655 priming episodes. The KS test detects "distributions differ" but the sample asymmetry isn't a persona-convergence signal. Roy-2's larger priming + longer test will give this metric a clean read.

### Three concrete pointers for Roy-2 methodology refinement

This is the question Roy-1b *answers* and Roy-2 *must address before adversarial persona work*:

- **(a) Widen priming arc diversity.** Cradle_prelinguistic's 50 turns produce 6 distinct EC cluster ids all keyed to `sense_food_source` — a single-tool monoculture. The held-out percept regime ("heat blooms across your fingertips") doesn't pattern-match these clusters. Multi-arc priming (e.g., cradle_prelinguistic + cradle + a second cradle-flavor arc) would produce a richer EC cluster representation with more pattern-matching surface for held-out percepts.
- **(b) Tune `min_confidence` threshold.** Roy-0 flagged this as an open question (substrate-primary cold-start monoculture). Roy-1b adds: even with primed substrate, the threshold filter at test time produces the same default-behavior fallback as blank arms. Either drop the threshold to allow priming-acquired clusters to drive selection on weak matches, OR raise it and accept "no action" as informative.
- **(c) Ship `bio_emergent_persona_foundations.md` Wire 1 (substrate-annotates-LLM-context).** Neither current AUT mode reads `cluster_reward_bias` on percepts that don't match priming EC clusters. The Wire 1 design surfaces substrate-derived bias at the LLM-prompt level, where the LLM can apply it across percept regimes the substrate didn't directly drill. Roy-1b's negative behavioral result is the cleanest empirical argument for prioritising Wire 1 in 1.0.

## What this proves regardless of headline values

- **End-to-end Roy harness reproducibility across iterations.** Roy-0 → Roy-1a → Roy-1b: three single-seed iterations on the same priming substrate produced cluster_reward_bias_l2 within 1% (2.4587 → 2.4495 → 2.4678). The G4 wire is rock-solid; the priming side is mode-deterministic.
- **Substrate-primary at test time DOES exploit the cluster wire's consumer path.** 4 new cluster keys appeared in Roy-1b vs Roy-1a — substrate-primary's `recommend_action` is firing and writing. The wire's behavior is empirically validated under both AUT modes.
- **The held-out fixture is mode-agnostic.** Both Roy-1a and Roy-1b ran 30 of 30 test turns to completion on the same fixture; no fixture-shape regressions.

## What this still does NOT prove

- **Cross-session persistence** (still single-session).
- **Persona convergence on a real persona** (Roy-1a/1b are methodology validation).
- **Hybrid Wire 1 sufficiency** — neither tested.
- **That substrate priming would behaviorally express with priming-percept overlap.** Untested. Roy-2 should run a held-out fixture whose percepts *do* fire priming-acquired EC clusters as a positive control.

## Reproduction

See [protocols/17_roy_1b_reproduction.md](protocols/17_roy_1b_reproduction.md).

## Recommendation for next iteration

Roy-1a's recommendation ("Roy-1b should run next") is now executed and consumed. Roy-1b's finding shifts the next-iteration recommendation:

**Roy-2 is unblocked but needs a methodology refinement decision first.** Three options, prioritised by information-per-cost:

1. **Roy-2 with multi-arc priming + the existing held-out fixture.** Cheapest. Tests whether widening priming alone fixes the percept-overlap problem. ~30 min wall. Recommend running this BEFORE shipping Wire 1 — if multi-arc priming fixes behavioral expression on its own, Wire 1's complexity may not be needed yet.

2. **Roy-2 with a held-out fixture that explicitly overlaps with priming-acquired EC clusters.** Positive-control measurement. Tells us "the wire IS exploitable when percepts fire priming clusters" — useful for ruling out wire bugs, but doesn't move methodology forward toward real persona.

3. **Ship `bio_emergent_persona_foundations.md` Wire 1 first, then Roy-2.** Highest-confidence path to behavioral expression. Costs Wire 1's implementation budget; not a one-session task.

**My recommendation:** (1) next session. If multi-arc priming + the existing held-out fixture produces non-trivial behavioral divergence between primed and blank arms, the methodology is unblocked without needing Wire 1 yet. If it doesn't, (3) becomes the load-bearing prerequisite.

## PR

<!-- filled when PR opens -->
TBD
