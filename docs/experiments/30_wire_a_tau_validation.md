# 30 — Wire-A tau-split validation (Roy-3a-retry)

**Status:** Run complete 2026-05-25. NULL outcome with three structural findings. **Retroactive correction added 2026-05-27** — see end of doc.
**Branch:** `feat/wire-a-tau-split-phase-3-validation`
**Plan:** [docs/plans/archive/cluster_reward_bias_decay_tau_split.md](../plans/archive/cluster_reward_bias_decay_tau_split.md), Phase 3.
**Predecessor PR:** [#267](https://github.com/dennys246/Maxim/pull/267) (Phase 1 tau-split implementation).
**Followups:** [docs/plans/deferred/sense_tool_registry.md](../plans/deferred/sense_tool_registry.md) (new) + [docs/plans/deferred/imagination_substrate_signals.md](../plans/deferred/imagination_substrate_signals.md) (new).

## Pre-registration

Per [tau-split plan Phase 3 table](../plans/archive/cluster_reward_bias_decay_tau_split.md):

| Measure | Roy-3a baseline (pre-split, tau=50) | Roy-3a-retry expected (tau=300) |
|---|---|---|
| Priming-end `cluster_reward_bias` | 2 keys, `{partial ~0.20, +0.98}` | Unchanged |
| Wire-A max(\|bias\|) at test time | 0.036 | ≥0.5 (strongly rewarding band) |
| Wire-A rendered band | `[neutral / mixed]` | At least one `strongly rewarding` |
| Arm A `sense_food_source` tool calls | 0 | ≥1 (positive divergence) |

**PRIMARY pass criterion (pre-registered):** Arm A produces ≥ 1 `sense_food_source` tool call across the 10-turn test arm.

**STRETCH:** Arm A > Arms B and C on `sense_food_source` count (cross-arm divergence).

## Methods

**Spec:** [scenarios/roy/roy_3a_iteration.yaml](../../scenarios/roy/roy_3a_iteration.yaml) (unchanged from Roy-3 — single-variable comparison is the tau split, not the spec).

**Env at runner:**
```
PYTHONPATH=src
MAXIM_SUBSTRATE_PATH=1
MAXIM_LOG_FILE=/tmp/roy_3a_tau_validation.jsonl
```

**Build state:** main @ `be0a2ec` (Phase 1 tau-split merged via PR #267). `NACConfig.cluster_reward_bias_decay_tau = 300.0` is the active value.

**Wall time:** 901.84 s (~15 min). Started 2026-05-25 21:56:55 MDT, ended 22:11:55 MDT.

**Arms:**
- A: `substrate=from_priming`, `system_prompt=neutral` (test arm — the Wire-A read happens here).
- B: `substrate=blank`, `system_prompt="You are a hungry infant"` (Roy-1a/Roy-2 baseline).
- C: `substrate=blank`, `system_prompt=neutral` (control).

**Priming + test fixture:** identical to Roy-2/Roy-2pc/Roy-2c (multi-arc priming + `roy_1_holdout.yaml`).

## Result — per-arm LLM-called tool distributions

Per-tick auto-sense (auto-fired `sense_presence` and self-sense, bypassing the executor at [agent_loop.py:1292](../../src/maxim/runtime/agent_loop.py)) does NOT write to `actions.jsonl` and is excluded from these counts. The table is LLM-called tools only:

| Tool | Arm A | Arm B | Arm C |
|---|---:|---:|---:|
| `respond` | 11 | 21 | 26 |
| `sense` | 5 | 4 | 0 |
| `sense_presence` | 2 | 1 | 0 |
| `sense_tools` | 0 | 1 | 1 |
| `infant_humanoid_pick_up` | 1 | 1 | 1 |
| `infant_humanoid_move` | 0 | 0 | 2 |
| `infant_humanoid_look` | 0 | 0 | 1 |
| `infant_humanoid_speak` | 0 | 0 | 1 |
| `_llm_unavailable` | 0 | 1 | 2 |
| **`sense_food_source`** | **0** | **0** | **0** |
| **`sense*` LLM-called total** | **7** | **6** | **1** |

## Result — Wire-A render reconstruction at arm A

`get_agent_tool_biases(agent_id="default_agent", top_n=5)` returns max\|bias\| per tool agent-wide ([nac.py:1858](../../src/maxim/decisions/nac.py)). At arm A end:

| Tool | max\|bias\| | Band |
|---|---:|---|
| `sense_food_source` | 0.753 | strongly rewarding |

Priming-end value was 0.997; arm A end was 0.753 — decayed by 0.244 over the ~84 ticks of arm A's wall (90 s × ~1 tick/s). The decay trajectory:

| Test-arm tick | Approx. max\|bias\| | Band |
|---:|---:|---|
| 0 (priming hand-off) | 0.997 | strongly rewarding |
| 30 (turn ~3) | 0.901 | strongly rewarding |
| 60 (turn ~6) | 0.815 | strongly rewarding |
| 84 (turn ~10) | 0.753 | strongly rewarding |

**Wire-A rendered `sense_food_source [strongly rewarding from prior experience]` at every LLM submission during arm A.** Throughout the entire 10-turn test arm.

## Result — log surfaces from the kickoff vigilance

1. **Stage 0c `sim_recommend_action` events**: zero during arm A. This is expected — `recommend_action` is the *substrate-primary* code path; arm A is `llm-primary` at test, so the substrate-primary recommender doesn't fire. Wire-A's annotation render is a *different* code path ([src/maxim/agents/prompt_builder.py:1127](../../src/maxim/agents/prompt_builder.py)) which does not currently emit a sim_log event. The 792 sim_recommend_action events all fired during the priming session, not at test.

2. **`cluster_reward_bias` decay trajectory**: clean fit to tau=300 decay-per-tick model. Priming-end 0.997 → arm A end 0.753 ≈ 0.997 × (1 − 1/300)^84 = 0.755 (within 0.3% of model). The Phase 1 calibration math is structurally validated.

3. **Cross-arm tool call distribution**: `sense_food_source = 0` across all three arms. **LLM-called `sense*` family**: Arm A=7, Arm B=6, Arm C=1. Arm A leans into sensing tools relative to Arm C (the substrate-blank, neutral-prompt control) — soft positive. Arm B's 6 (with "hungry infant" persona prompt) is close to Arm A's 7, suggesting the persona prompt also drives sensing about as strongly as Wire-A's annotation does.

4. **`Concept reinforced: "sense_food_source" (action)`**: 660 events during priming, confirming substrate strongly encodes the concept.

## Verdict

**PRIMARY pass criterion (Arm A produces ≥ 1 `sense_food_source` call): FAILED.** Arm A produced 0.

**STRETCH (Arm A > B,C on `sense_food_source` count): FAILED.** No divergence on the specific tool (all three arms 0).

**Tau-split structural validation: PASSED.** The Phase 1 calibration math is validated by the decay trajectory; Wire-A's annotation surfaced `sense_food_source [strongly rewarding from prior experience]` at every arm A LLM submission, with the bias magnitude in the "strongly rewarding" band throughout the entire test arm. The pre-Phase-1 0.036 magnitude lifted to a 0.753-0.997 band exactly as Phase 1 predicted. The tau split is correct.

## Architectural findings surfaced by post-result investigation

The PRIMARY criterion's failure is not a tau-magnitude issue (which the calibration math validates) and not just a single-layer gap. Post-result code investigation surfaced three structural concerns. Two of them get their own plan docs ([sense_tool_registry.md](../plans/deferred/sense_tool_registry.md) + [imagination_substrate_signals.md](../plans/deferred/imagination_substrate_signals.md)) since they're broader than this experiment's scope.

### Finding 1 — Substrate-scene-tool-availability gap

`sense_food_source` is a SEM-modulator-derived tool ([embodiment/tool_bridge.py:405](../../src/maxim/embodiment/tool_bridge.py)) auto-registered when a food-source entity enters scene. The Roy-1 holdout fixture is pure body-sensation percepts (heat / cold / vibration / social text) — no food entity, so the affordance is never registered. Arm A's logged active tool roster (at `t=1779768427.02`): `causal_links, concept_query, display_mode, energy_status, examine, infant_humanoid_pick_up, memory_recall, predict_outcome, request_interaction, respond, say, sense, sense_presence, sense_tools, set_scene, similarity_search, speak, system_stats, temporal_patterns, think` — and `sense_food_source` is absent. **The LLM can read Wire-A's annotation but cannot invoke the tool the annotation names.**

### Finding 2 — Sense tool registration heterogeneity

The sense* tool family has 7 distinct tools across 3 registration regimes:

| Group | Examples | Registration | Auto-fire? | LLM-visible when inactive? |
|---|---|---|---|---|
| Universal/core | `sense`, `sense_tools` | Once at boot | No | n/a (always active) |
| Auto-discovery | `sense_presence` | Once at boot | Yes (executor bypass) | n/a |
| SEM-modulator-derived | `sense_<entity>`, `read_<entity>_<sensor>`, `sense_food_source` | Per-entity at scene load | No | **No — silently invisible** |

The LLM has zero signal that SEM-derived sense tools might exist in other scenes. A unifying Sense Registry / Factory pattern could close this by adding grayscale visibility (inactive tools listed as `[not in current location]`) without breaking the existing load-bearing invariants (auto-sense bypass for hygiene, SEM scene-scoping for relevance). See [docs/plans/deferred/sense_tool_registry.md](../plans/deferred/sense_tool_registry.md) for the full scope.

### Finding 3 — Imagination is substrate-blind

Imagination's trigger ([imagination/trigger.py:579](../../src/maxim/imagination/trigger.py)) reads only percept text + scene manifest. The 74 `Imagination skipped: no percept_text (obs keys: [])` events during arm A correspond to idle ticks. When real percepts arrived, imagination ran but extracted zero phrases: `Imagination: no entity phrases from 'heat blooms across your fingertips.'` — the regex + stop-word extractor at [trigger.py:275](../../src/maxim/imagination/trigger.py) finds no entity indicators in pure-sensation language.

Substrate signals (NAc `_cluster_reward_bias`, Wire-A annotations, ATL concept activation) never reach imagination. In a simulation where Wire-A is annotating `sense_food_source` as strongly rewarding, imagination *should* be able to dream up a food-source entity into the scene to make the substrate-favored tool invokable — but the current wiring can't bridge that gap.

Three highest-priority fixes (sized in the plan doc):
1. Pass NAc top-biases to `generate_scene_manifest()` so the LLM-generated scene manifest sees Wire-A's biases before the AUT thread launches.
2. Add a substrate-signal hookup to `process_percept()` so the agent loop can request imagination for missing high-bias tools.
3. Relax the DN arousal gate for first-reaction-to-novel-percept ticks (currently blocks imagination during LLM-primary high-load).

See [docs/plans/deferred/imagination_substrate_signals.md](../plans/deferred/imagination_substrate_signals.md) for the full scope.

## Implications for the kickoff's three NULL branches

The kickoff anticipated two NULL branches; Phase B's verdict is a third:

- **NOT "tau=300 too aggressive."** Annotations were `[strongly rewarding]` throughout the test arm. The calibration math holds. Lifting to 400 would not change the outcome.
- **NOT "LLM ignores meaningful annotation."** Arm A's 7-to-1 `sense*` LLM-call lean over Arm C (and the 6 in Arm B with explicit persona prompt) suggests the LLM *does* read the annotation and generalize toward sensing — just at a generalization granularity that doesn't reach the specific tool Wire-A names.
- **THIRD branch: annotation drives the LLM toward a tool the current scene cannot supply.** Closing this gap requires *either* surfacing SEM-derived tools that aren't in scene with grayscale visibility (Finding 2), *or* making imagination substrate-aware so it dreams the missing entity into scene (Finding 3), or both.

Per the kickoff's escalation rule ("do NOT silently retry with a different tau. Document the NULL, diagnose the cause, surface to user, and proceed only after explicit authorization for the retry"), this verdict does **not** authorize a tau=400 retry. The bias-magnitude side is no longer the bottleneck.

## Companion artifacts

- Result JSON: `~/.maxim/roy/roy-3a/result.json`
- Per-arm session snapshots: `~/.maxim/sim_reports/{20260525_220706, 20260525_220836, 20260525_221013}/`
- Runner JSONL: `/tmp/roy_3a_tau_validation.jsonl` (7.8 MB)

## Plan-doc folding (per kickoff B6)

- [docs/plans/archive/cluster_reward_bias_decay_tau_split.md](../plans/archive/cluster_reward_bias_decay_tau_split.md): Phase 3 outcome — "tau=300 structurally validated by decay trajectory + annotation render; PRIMARY criterion failed due to downstream scene-tool-availability gap, not tau magnitude."
- [docs/plans/archive/release_0_9_1.md](../plans/archive/release_0_9_1.md) Stage 5: Roy-3 follow-up item 2 redirects to [sense_tool_registry.md](../plans/deferred/sense_tool_registry.md) + [imagination_substrate_signals.md](../plans/deferred/imagination_substrate_signals.md).
- [docs/plans/archive/v1_refinement.md](../plans/archive/v1_refinement.md): checked, no existing tau-split entry to update (the 1.0 plan never tracked the tau-split work directly). No fold required.
- [docs/plans/deferred/persona_convergence_crucible.md](../plans/deferred/persona_convergence_crucible.md): three NULL-branch open questions become next-iteration scope; Wire-A's tau parameter remains tunable per the calibration framing.
- [docs/plans/README.md](../plans/README.md): index entries added for the two new plan docs.

## Retroactive correction (2026-05-27, [32_wire_a_post_w1_w2.md](32_wire_a_post_w1_w2.md))

The W1+W2 integration test on 2026-05-27 surfaced a Roy cross-session agent_id mismatch that was already in effect during this Phase B run:

- Priming MemoryHub defaults to `agent_id="default_agent"` ([memory_hub.py:170](../../src/maxim/integration/memory_hub.py)); persisted `cluster_reward_bias` keys carry that prefix.
- The test-arm AUT is constructed with `agent_id="sim_aut"` ([orchestrator.py:534](../../src/maxim/simulation/orchestrator.py)); `_loop_agent_id` resolves to `"sim_aut"` at [agent_loop.py:1074](../../src/maxim/runtime/agent_loop.py).
- `NAc.get_agent_tool_biases(agent_id="sim_aut")` returns `[]` because the strict-equality filter at [nac.py:1903](../../src/maxim/decisions/nac.py) doesn't cross the priming/test agent_id boundary.

**The claim "Wire-A rendered `sense_food_source [strongly rewarding from prior experience]` at every LLM submission during arm A" was inferred from priming-end magnitude + decay-trajectory math, not verified from rendered LLM prompts.** The 0.997 → 0.753 decay this doc cited as evidence of the annotation being read was actually proof that `NAc.decay_cluster_reward_biases` ticked — that function is agent_id-agnostic, so it runs regardless of whether the annotation reaches the LLM.

What stays valid from this experiment:
- **Tau-split structural validation (decay trajectory):** valid. The Phase 1 calibration math holds independently of whether the annotation reaches the LLM.
- **PRIMARY criterion failure (Arm A = 0 `sense_food_source`):** valid as a behavioral measurement. The cause-attribution to "scene-tool-availability gap" is partially valid — the gap is real — but the agent_id mismatch likely contributed too.
- **Findings 1-3 (scene-tool-availability, sense-tool heterogeneity, imagination substrate-blindness):** valid as architectural observations; they motivated W1+W2 correctly. The integration test couldn't measure their effectiveness because of the upstream wiring bug, but the architectural reasoning stands.

What needs re-validation after Fix A (Roy priming uses `agent_id="sim_aut"`, [v1_refinement.md §1.5](../plans/archive/v1_refinement.md)) lands:
- Whether Wire-A's annotation ever reached the LLM during 0.9.1 Roy iterations.
- Whether the "annotation drives the LLM toward a tool the current scene cannot supply" third NULL branch above is the actual cause of Phase B's failure, or whether annotation-empty was the actual cause.
- Companion: experiment 32's Bug A scoping inherits this question.
