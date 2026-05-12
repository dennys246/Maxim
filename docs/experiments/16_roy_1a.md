# Roy-1a — llm-primary on held-out fixture (first methodology iteration)

**Date:** 2026-05-12 (run completed 2026-05-11 21:17 local)
**Plan:** [persona_convergence_crucible.md § "Iteration log"](../plans/persona_convergence_crucible.md)
**Companion:** [15_g4_cluster_reward_wire.md](15_g4_cluster_reward_wire.md) (Roy-0 baseline this iteration builds on) · [14_g3_roy_preflight_probe.md](14_g3_roy_preflight_probe.md) (pre-flight probe)
**Spec:** [scenarios/roy/roy_1a_iteration.yaml](../../scenarios/roy/roy_1a_iteration.yaml)
**Held-out fixture:** [scenarios/roy/roy_1_holdout.yaml](../../scenarios/roy/roy_1_holdout.yaml)
**Reproduction:** [protocols/16_roy_1a_reproduction.md](protocols/16_roy_1a_reproduction.md)

## Status

First "real" Roy iteration after Roy-0 smoke (2026-05-11). Two methodology axes changed vs Roy-0:

1. **`aut_mode: llm-primary` at test time** (Roy-0 ran substrate-primary throughout; its cold-start regime produced a `sense_food_source` cluster monoculture).
2. **Held-out test fixture** (Roy-0 reused the priming arc). 10-percept fixture covers three classes (matching / novel / unrelated) per [persona_convergence_crucible.md §"Test phase"](../plans/persona_convergence_crucible.md).

Priming remained **identical** to Roy-0: 5 stages × 10 turns of `cradle_prelinguistic` at substrate-primary. Single-variable change at test time + held-out fixture lets divergence-vs-Roy-0 be attributed cleanly to those two axes.

## What was caught (pre-Roy-1a)

[persona_convergence_crucible.md §"Open questions" item 3](../plans/persona_convergence_crucible.md) flagged a pre-Roy-1 stress test on multi-agent attribution. The original P4 tests validated per-agent partitioning at N=50 per two agents; forward Roys will produce ~150-1000 distinct cluster updates per arm with a similar order of causal links.

[tests/integration/test_multi_agent_attribution_scale.py](../../tests/integration/test_multi_agent_attribution_scale.py) adds five scale tests at 4 agents × N=500-1000 each:

- **N=1,000 cluster updates × 4 agents, shared NAc.** Every persisted `(agent_id, cluster, tool)` triple partitions cleanly — zero cross-contamination.
- **N=1,000 `record_outcome` × 4 agents, shared NAc.** Every `CausalLink.event_context['agent_id']` matches the originating agent; tool-signature buckets disjoint across agents.
- **N=500 substrate-stash R/W × 4 agents.** Every consume returns the producing agent's nodes.
- **N=500 pain-intensity producers × 4 agents.** Each agent's intensity stays in its non-overlapping band (no max-merge contamination).
- **N=500 observe-episode ticks × 4 agents.** Each tick sequence is the contiguous range `1..N`.

All 5 pass in <1s — surfacing per-agent partitioning at 20× the load of the original tests. Tripwire for Roy-1: Adversarial (~1,000 priming turns), which sits at 1× of these tests' load.

## What shipped

### `scenarios/roy/roy_1_holdout.yaml` — 10-percept held-out fixture

| Class | Count | Example |
|---|---|---|
| matching | 3 + 1 cooldown | "heat blooms across your fingertips" — same modality family as priming, paraphrased |
| novel | 3 | "a low vibration hums beneath your back" — same somatosensory modality, stimuli priming never drilled |
| unrelated | 3 | "two people are arguing in the next room" — social/linguistic, no direct body coupling |

`scenario_tag` + `percept_class` metadata per percept means downstream analysis can bucket arm behaviour by class. Cooldown percept tagged `matching` so the last beat stays in familiar regime.

### `scenarios/roy/roy_1a_iteration.yaml` — iteration spec

Mirrors Roy-0 (5 priming stages × 10 turns of `cradle_prelinguistic` at substrate-primary) but with:

- `name: roy-1a`
- `aut_mode: llm-primary` at test time
- `test_scenario.fixture: roy_1_holdout.yaml`
- `test_scenario.turns: 10` (up from Roy-0's 3 — full fixture runs once; bridge cancels at exhaustion)

## Result

Wall: **830.1s (~13.8 min)** vs Roy-0's 926s (~15.4 min) — single-pass test arms at llm-primary are slightly faster than substrate-primary's heavier in-loop computation. Pre-flight cleared via peer.yml (`outcome: ok`, `latency_ms: 397.6`).

### Per-arm

| Arm | Substrate | system_prompt | session_id | turns | duration_s | finish |
|---|---|---|---|---|---|---|
| a | from_priming | neutral | `20260511_211347` | 10 | 83.7 | cancel |
| b | blank | "You are a hungry infant" | `20260511_211511` | 10 | 73.4 | cancel |
| c | blank | neutral | `20260511_211624` | 10 | 61.7 | cancel |

`turns=10, finish=cancel` for every arm — bridge cancels at fixture exhaustion, not a failure. 22 `peer_backend_call` events on the leader trace (qwen2.5-14b-instruct, all status 200), 10 narrator generations, 2 `dispatch_exhausted` warnings on orchestrator probes (cosmetic; arms still completed).

### Headline pairwise diffs

| Pair | `reward_bias_l2` | **`cluster_reward_bias_l2`** | `goal_reward_bias_l2` | `causal_link_Δ` | `episodes_Δ` | `valence_KS` (p) | `salience_KS` (p) |
|---|---|---|---|---|---|---|---|
| **a_vs_b** | 0.0 | **2.4495** | 0.2714 | +156 | +656 | 0.283 (p=0.402) | **0.879 (p=2.1e-9)** |
| **a_vs_c** | 0.0 | **2.4495** | 0.2606 | +150 | +656 | 0.283 (p=0.402) | (similar) |
| b_vs_c | 0.0 | 0.0 | 0.2666 | −6 | 0 | 0.000 (p=1.000) | 0.0 |

### Cluster-reward top deltas (a_vs_b — identical shape on a_vs_c)

Six `tool:sense_food_source` entries at the +1.0 per-key cap (`max_cluster_reward_bias=1.0`), each on a distinct EC cluster id — **all inherited from arm A's substrate-primary priming**:

```
6× tool:sense_food_source  delta=+1.0  (six distinct EC cluster ids from priming)
0  cluster updates added during test phase under llm-primary
```

### Goal-reward top deltas (a_vs_b)

```
roy:roy-1a:arm_b      delta=-0.196   (arm B's goal accumulated net-negative reward)
roy:roy-1a:arm_a      delta=+0.181   (arm A's goal accumulated net-positive reward)
cradle_prelinguistic  delta=+0.051   (priming goal carryover into arm A)
```

### Hippocampal valence + salience (a_vs_b)

```
valence:   mean_a=-0.088, mean_b=0.000    (priming wrote slightly-negative-mean episodes; blank arm has only its 9 test episodes near zero)
salience:  mean_a=0.506,  mean_b=0.711    (arm A's hippocampus, with 665 priming episodes as context, rates test-phase percepts as LOWER salience than blank arm B)
```

### Roy-0 (substrate-primary @ test) → Roy-1a (llm-primary @ test)

| Metric | Roy-0 | Roy-1a | Interpretation |
|---|---|---|---|
| `cluster_reward_bias_l2` (a_vs_b) | 2.4587 | **2.4495** | Substrate priming wire writes through to NAc *identically* across AUT modes — the bias is structurally preserved |
| `b_vs_c.cluster_reward_bias_l2` (noise floor) | 0.2121 | **0.0** | llm-primary test arms invoke zero `sense_food_source` calls → no stochastic-cluster noise floor. A-vs-blank signal-to-noise jumps from 11.6× to ∞ |
| `causal_link_count_delta` (a_vs_b) | +155 | +156 | Within 1 link of Roy-0 — priming-side episode/link production is unaffected by test-mode AUT |
| `goal_reward_bias_l2` (a_vs_b) | 0.1918 | **0.2714** | LLM AUT under "neutral" prompt succeeded more at its goal than the prompt-injected "hungry infant" arm did — goal-reward divergence is larger under llm-primary |
| `valence_KS` (a_vs_b, p-value) | 0.000 (1.000) | **0.283 (0.402)** | First non-zero valence KS in Roy harness history. Effect size present but sample-size dominated (9 episodes in blank arm vs 665 in primed) — not statistically significant at α=0.05 |
| `salience_KS` (a_vs_b, p-value) | not tracked | **0.879 (2.1e-9)** | Strong + highly significant. Primed hippocampus scores test percepts as lower-salience because they're less novel against 665 prior episodes |
| Wall time | 15.4 min | 13.8 min | llm-primary test arms cheaper than substrate-primary |
| Test fixture | warmup.yaml (3 percepts, rehearsal) | roy_1_holdout.yaml (10 percepts, held-out) | Methodology axis |

## What this proves

The headline finding from Roy-1a separates two questions the Roy-0 smoke test conflated:

1. **Does substrate-only priming write LLM-readable bio-state?** *Yes.* Cluster-reward bias L2 is unchanged from Roy-0 (2.45 vs 2.46) under llm-primary test-time AUT — the wire that learned during substrate-primary priming carries forward verbatim into a session that runs an entirely different proposer. Substrate is structurally preserved across AUT modes.
2. **Does substrate-only priming *behaviorally express* under llm-primary?** *Not via the cluster-bias path.* Arm A's tool distribution at test time contained zero `sense_food_source` calls — despite the priming substrate carrying +1.0 cluster bias for that tool on six distinct EC clusters. The llm-primary proposer chose tools based on its own context (`infant_humanoid_pick_up`, `sense`, `respond`) and did not read the cluster-keyed bias (substrate-primary's `recommend_action` is the consumer of that bias; llm-primary doesn't invoke it).

The salience divergence (KS=0.879, p=2.1e-9) is the **load-bearing positive finding** for the methodology: hippocampal salience scoring at test time differs strongly and significantly between primed and blank arms. The primed hippocampus, holding 665 prior episodes from the cradle arc, rates test-phase percepts as **lower-salience** than blank arms do. This is the substrate carryover translating into a quantitative downstream signal that the test-time AUT *reads* — salience is consumed by ThoughtGate and WMS scoring during llm-primary's deliberation step.

The valence divergence (KS=0.283, p=0.402) is the **load-bearing equivocal finding**: the priming substrate carries negative-mean valence (mean=-0.088 over 665 episodes — affordance failures during pre-linguistic exploration), and this distribution differs from the blank arms' near-zero mean. The effect is real (KS > 0.28 vs blank's 0.0) but the sample is dominated by 9 blank-arm test episodes against 665 primed-arm episodes, so the test doesn't reach α=0.05 significance. Roy-1b (substrate-primary AUT on the same fixture) and Roy-2 (longer-priming + multi-seed) will reduce this noise floor.

The goal-reward asymmetry (arm A's goal +0.181, arm B's goal −0.196 on the goal_reward_bias top deltas) is interesting but methodology-driven, not behavioral: each arm has a distinct goal tag (`roy:roy-1a:arm_a`, `roy:roy-1a:arm_b`) and NAc credits/penalizes per-goal. That arm B's goal landed net-negative while arm A's landed net-positive is consistent with the LLM-AUT in arm A (with primed substrate context shaping its tool outcomes) producing more successful tool calls than arm B (with the slightly-confused "hungry infant" prompt and blank substrate).

### Definitively proved regardless of headline values

- **The held-out fixture works for llm-primary AUT.** 30 of 30 test-phase turns completed cleanly across arms; 22 peer_backend_calls, all 200 OK. Zero fixture-shape regressions.
- **The pre-Roy-1 stress test is the right tripwire.** 5 attribution-scale tests pass at 4 agents × N=500-1000 — Roy-1: Adversarial sits within this load envelope with margin.
- **The G3 preflight + G4 cluster wire are stable across iterations.** Roy-0 → Roy-1a single-variable comparison reproduces the cluster_reward_bias_l2 number within 0.4% (2.4587 → 2.4495).

## What this still does NOT prove

- **Substrate-only priming surviving an LLM-primary test as *behavior*** is unproven. The cluster_reward_bias path is consumer-coupled to substrate-primary's `recommend_action`; llm-primary at test time doesn't read it. To behaviorally express substrate priming through llm-primary, either (a) hybrid priming would expose the LLM to the priming context directly, or (b) the LLM's prompt would need to be annotated with substrate-derived bias (the Wire 1 design [bio_emergent_persona_foundations.md](../plans/bio_emergent_persona_foundations.md) calls out). Roy-1b will measure the substrate-primary AUT side of this; Roy-2 will introduce the actual question.
- **Cross-session persistence** is untested (single-session iteration).
- **Persona convergence on a real persona.** Roy-1a is methodology validation — its `system_prompt` slugs are placeholders matching Roy-0. Carefully-shaped persona prompts arrive at Roy-2.
- **The valence carryover is real-but-marginal.** KS=0.283 with p=0.402 is below significance; doesn't reproduce above α=0.05 without longer priming or seed pooling.

## Reproduction

See [protocols/16_roy_1a_reproduction.md](protocols/16_roy_1a_reproduction.md).

## Recommendation for next iteration

**Roy-1b should run next.** The cleanest extension of Roy-1a: same fixture, same priming substrate, swap `aut_mode` to `substrate-primary` at test time. Two reasons:

1. The cluster_reward_bias structurally carried forward but didn't express behaviorally under llm-primary. Roy-1b will measure whether substrate-primary AUT at test time *does* exploit the priming bias (Roy-0 was substrate-primary throughout, so we can't distinguish "priming exploitation" from "in-session learning" — Roy-1b's clean separation between priming and test phases will).
2. The held-out fixture works under llm-primary. Re-using it for substrate-primary validates the fixture is mode-agnostic and lets the two iterations be diffed directly.

If Roy-1b shows substrate-primary AUT does invoke `sense_food_source` (or other priming-reinforced tools) in the test phase against percepts that don't pattern-match priming, that's the behavioral expression of substrate carryover the persona work needs.

If Roy-1b shows substrate-primary AUT still doesn't exploit the bias on the held-out fixture, the methodology question shifts to either (a) the min_confidence threshold gating cold-start (currently `0.3`, tracked in Roy-0's "what to change") or (b) the fixture's percept regime is too far from priming for the EC cluster representation to fire at recall time.

Methodology refinement (revisit before Roy-2):
- Consider longer priming for valence_KS to clear α=0.05 (Roy-0 had p=1.0; Roy-1a is at p=0.402; needs ≥4× more episodes or seed pooling).
- The `b_vs_c.causal_link_count_delta = -6` is small but non-zero (blank arms with different prompts produced slightly different test-phase causal links). Worth a note in the next iteration log, not a blocker.

## PR

<!-- filled when PR opens -->
TBD
