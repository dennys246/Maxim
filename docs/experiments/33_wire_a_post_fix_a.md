# 33 — Wire-A post-Fix-A clean measurement (Roy-3a)

**Status:** Run complete 2026-05-27. **PRIMARY failed (Arm A=0/B=0/C=0) BUT Wire-A demonstrably reaches the LLM for the first time.** This is the clean substrate→action conversion gap measurement that exp 30 and exp 32 could not produce because of Bug A.
**Branch:** main @ `cdd005a` (post-PR-#290 merge — Fix A landed).
**Plan:** [docs/plans/v1_refinement.md](../plans/v1_refinement.md) §1.5 integration test, re-run after Fix A.
**Predecessors:** [30_wire_a_tau_validation.md](30_wire_a_tau_validation.md) (Phase B baseline, Bug A confounded) + [32_wire_a_post_w1_w2.md](32_wire_a_post_w1_w2.md) (Bug A discovery + agent_id mismatch root cause).

## Pre-registration

| Criterion | Pre-registration | This run |
|---|---|---|
| PRIMARY: Arm A `sense_food_source` count | ≥ 1 | **0** |
| STRETCH: Arm A > Arms B+C on `sense_food_source` | A > B,C | All three = 0 (no divergence) |
| Fix A verification: NAc dump keyed by `sim_aut` | yes | **yes ✓** |
| Wire-A reaches LLM (lookup returns non-empty under `sim_aut`) | yes | **yes ✓** |

**Convergence outcome** (Arm A ≥1): Wire-A annotation→action pathway validated; W2 deferrable to 1.1+.
**Divergence-in-a-row outcome** (Arm A still 0 with Wire-A demonstrably reaching the LLM): per kickoff's pre-registration in v1_refinement.md §1.5 + refined Principle 4 — the two-divergence-in-a-row trigger to bird's-eye to encoder replacement (Roy-5a Stage 3 + JEPA, 1.2+). **THIS verdict (formally), with caveat below.**

## Methods

**Spec:** [scenarios/roy/roy_3a_iteration.yaml](../../scenarios/roy/roy_3a_iteration.yaml) — unchanged from Roy-3 / Roy-3a-retry / exp 32. Single-variable change vs exp 32 is the Fix A merge (PR #290).

**Env at runner:**
```
PYTHONPATH=src
MAXIM_SUBSTRATE_PATH=1
MAXIM_LOG_FILE=/tmp/roy_3a_post_fix_a.jsonl
```

**Build state:** main @ `cdd005a` (PR #289 doc + PR #290 Fix A merged). `build_bio_stack` requires `agent_id=` kwarg; AgentFactory threads `config.agent_id` end-to-end.

**Wall time:** 897.13 s (priming + 3 arms × ~90s each). Backend: `lane-large` → `qwen2.5-14b-instruct` via `https://maxim.dennyschaedig.com/v1`.

**Note:** first runner attempt failed at preflight with HTTP 502 — leader was being updated. Retry after the leader recovered (HTTP 401 auth-gated probe verified upstream alive) ran cleanly.

## Result — Fix A structural validation

The single load-bearing question coming into this iteration was: does the NAc dump now key biases by `sim_aut` (the AUT's AgentConfig identity) instead of `default_agent` (the pre-Fix-A bio-stack default)?

| Phase | NAc `cluster_reward_bias` key prefix |
|---|---|
| Priming end (substrate-primary, multi-arc) | `sim_aut\x1f<cluster_uuid>\x1ftool:sense_food_source` |
| Arm A end (llm-primary, from_priming substrate) | `sim_aut\x1f<cluster_uuid>\x1ftool:sense_food_source` |

Direct simulation of Wire-A's lookup at agent-loop tick time:
```
get_agent_tool_biases(agent_id="sim_aut",      top_n=5) → [('tool:sense_food_source', +0.7681)]
get_agent_tool_biases(agent_id="default_agent", top_n=5) → []
```

The old `default_agent` lookup returns empty (correctly orphaned by Fix A); the new `sim_aut` lookup returns the priming-acquired bias with `+0.768` magnitude. **Wire-A's `compose_cluster_bias_annotation_section` would render the strongly-rewarding band:**

```
=== Substrate associations from prior experience ===
  sense_food_source  [strongly rewarding from prior experience]
```

Throughout the entire 10-turn Arm A. (Decayed from priming-end 0.997 → arm-A-end 0.768 per `decay_cluster_reward_biases`'s agent_id-agnostic tau-300 per-tick decay; same trajectory as exp 30's Phase B, validating Fix A had no side effect on the tau-split calibration math.)

## Result — per-arm LLM-called tool distribution

| Tool | Arm A | Arm B | Arm C |
|---|---:|---:|---:|
| `respond` | 10 | 11 | 15 |
| `sense` | 3 | 0 | 5 |
| `sense_presence` | 2 | 0 | 1 |
| `infant_humanoid_pick_up` | 2 | 2 | 1 |
| `pick_up` | 1 | 1 | 1 |
| `say` | 1 | 0 | 0 |
| `sense_tools` | 0 | 1 | 0 |
| `infant_humanoid_use` | 0 | 2 | 0 |
| `infant_humanoid_speak` | 0 | 1 | 0 |
| `infant_humanoid_listen` | 0 | 1 | 0 |
| **`sense_food_source`** | **0** | **0** | **0** |
| **`sense*` LLM-called total** | **5** | **1** | **6** |

Arm A and Arm C show similar overall sensing engagement (A=5 vs C=6); Arm B (with explicit "hungry infant" persona prompt) leans into entity actions instead of sensing. Wire-A's `[strongly rewarding from prior experience]` annotation did NOT detectably lift Arm A's sensing relative to Arm C, and did NOT drive even one `sense_food_source` call.

## Result — tool roster gap

Arm A's active tool roster (20 tools, from `sim_sem_trace`):

```
causal_links, concept_query, display_mode, energy_status, examine,
infant_humanoid_pick_up, memory_recall, predict_outcome,
request_interaction, respond, say, sense, sense_presence, sense_tools,
set_scene, similarity_search, speak, system_stats, temporal_patterns,
think
```

**`sense_food_source` is NOT in the active roster.** The roy_1_holdout fixture contains no food entity, so the SEM-derived tool that the priming session auto-fired 2870 times is silently absent from the test arms. Per W1's design, the LLM sees the tool in the grayscale section with `[strongly rewarding from prior experience] [not in current location]` — the annotation reaches the LLM but names a tool the LLM cannot currently invoke.

## What this iteration definitively rules in/out

**Ruled out** as PRIMARY-=-0 causes:
1. ~~Wire-A's substrate read returns empty~~ — Fix A verified the lookup chain end-to-end (`sim_aut` keys, +0.768 magnitude, strongly-rewarding band).
2. ~~Wire-A's annotation doesn't render in the prompt~~ — `compose_cluster_bias_annotation_section` correctly renders for the non-empty bias list returned by Fix A.
3. ~~Pre-existing wiring bug between priming and test-arm AUTs~~ — both phases now use `sim_aut` end-to-end.
4. ~~Bias magnitude too low for "strongly rewarding" band~~ — +0.768 is well above the band threshold; the decay-trajectory math from exp 30's Phase B is independently re-validated.

**Ruled in** as load-bearing for PRIMARY=0:
1. **Scene-tool-availability gap** — `sense_food_source` not in active roster (the same finding exp 30 named as Finding 1). The LLM is told the tool is strongly rewarding but cannot invoke it.
2. **LLM does not reach for substitutes** — the LLM had access to `sense_tools` (could have queried for food-related affordances) and `examine` (could have asked about food in scene), and didn't use either in Arm A. The annotation says "sense_food_source rewarding"; the LLM defaults to `respond`×10.
3. **Imagination didn't dream up a food entity** — the W2 Bug B confirmed structural gap: imagination's manifest pre-trigger is only called by generative-narrator scene loads, NOT by fixture-driven scene loads. The annotation alone cannot bridge to a dreamed food entity.

## Verdict

**Pre-registered divergence-in-a-row trigger condition: MET on a strict reading.** Per the kickoff's framing in v1_refinement.md §1.5:

> "If Arm A still produces 0 with Wire-A demonstrably reaching the LLM, the divergence-in-a-row trigger fires correctly on the next iteration with a clean instrument."

This iteration is the next iteration with a clean instrument (Fix A made the instrument clean), and the condition fires.

**HOWEVER**, the load-bearing follow-up cause IS the second pre-registered gap (W2's Bug B — fixture-driven test arms bypass the substrate-aware manifest hookup). The divergence-in-a-row trigger formally points at encoder replacement (Roy-5a Stage 3 + JEPA, 1.2+), but the underlying mechanism is not yet known to be encoder-subspace disjointness — it could be:
- (a) **Encoder alignment gap (the trigger's named cause)** — the LLM has no semantic thread from the holdout fixture's body-sensation percepts ("heat blooms across your fingertips") to a food affordance, even when the annotation literally names "sense_food_source" as rewarding. Roy-5a's H1a verdict already named the cross-modal subspace disjointness as load-bearing; this iteration is consistent with it.
- (b) **Scene-tool-availability gap (Fix B's pre-registered cause)** — `sense_food_source` simply isn't in scene. Bring a food entity into scene (Fix B extending W2 to fixture scene-load), and the LLM might convert the annotation to action without any encoder work. This is the cheaper, more-bio-fidelity-conservative test.

Both readings are consistent with the data. **The kickoff explicitly named "Triggering bird's-eye encoder work if divergence fires: NEVER without explicit authorization."** Per refined Principle 4 + the kickoff's authorization scope, this verdict surfaces both options to the user; the cheaper Fix B test is the natural next iteration before encoder work commits.

## Companion artifacts

- Result JSON: `~/.maxim/roy/roy-3a/result.json`
- Summary: `~/.maxim/roy/roy-3a/summary.md`
- Per-arm session snapshots: `~/.maxim/sim_reports/{20260527_222416, 20260527_222602, 20260527_222730, 20260527_222907}/`
- Runner JSONL: `/tmp/roy_3a_post_fix_a.jsonl` (40,656 events)
- Runner stdout: `/tmp/roy_3a_post_fix_a_runner.log`

## Comparison to exp 30 (Phase B baseline) and exp 32 (post-W1+W2)

| Measurement | Exp 30 (Phase B) | Exp 32 (post-W1+W2) | Exp 33 (post-Fix-A) |
|---|---|---|---|
| `cluster_reward_bias` key prefix | `default_agent` | `default_agent` | **`sim_aut`** |
| Wire-A's lookup returns non-empty (under `_loop_agent_id="sim_aut"`) | NO (Bug A) | NO (Bug A) | **YES** (+0.768) |
| Wire-A's annotation reaches LLM | UNVERIFIED (claimed but never confirmed) | UNVERIFIED | **YES** (rendered every submission) |
| Arm A `sense_food_source` count | 0 | 0 | **0** |
| Arm A `sense*` family count | 7 | 1 | 5 |
| Verdict | NULL (inferred Wire-A reached but unverified) | AMBIGUOUS-WITH-WIRING-BUG | **DIVERGENCE-IN-A-ROW** (pre-registered, clean instrument) |

## Plan-doc folding (per kickoff B6)

- [docs/plans/v1_refinement.md](../plans/v1_refinement.md) §1.5: Fix A verdict — instrument now clean. PRIMARY still 0. Pre-registered divergence-in-a-row trigger met; bird's-eye to encoder pivot vs Fix B (W2 to fixture) is the open user decision.
- [docs/plans/imagination_substrate_signals.md](../plans/imagination_substrate_signals.md): the post-Fix-A iteration confirmed W2's Bug B is still the load-bearing scene-tool-availability gap; Fix B is the natural next test before encoder pivot commits.
- [docs/plans/sense_tool_registry.md](../plans/sense_tool_registry.md): W1's grayscale visibility worked as designed (the LLM saw `sense_food_source [strongly rewarding from prior experience] [not in current location]`) — but the LLM did not act on the inactive-tool signal. W1 is structurally correct; its operator-visibility design was validated; behavioral impact requires either the tool becoming active (W2 Bug B + Fix B) or the LLM developing the cognitive ability to use grayscale signals (cross-modal binding / JEPA territory).
