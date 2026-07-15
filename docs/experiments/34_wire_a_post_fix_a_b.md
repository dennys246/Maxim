# 34 — Wire-A post-Fix-A+B clean measurement (Roy-3a)

**Status:** Run complete 2026-05-28. **PRIMARY failed (Arm A=0/B=0/C=0) BUT the failure mode narrowed sharply** — Fix B's pipeline works end-to-end; the residual gap is a substrate→scene-entity semantic mismatch at the **manifest LLM**, not at the runtime annotation pipeline.
**Branch:** main @ `b1e20ee` (post-PR-#292 merge — Fix B landed).
**Plan:** [docs/plans/archive/v1_refinement.md](../plans/archive/v1_refinement.md) §1.5 integration test, re-run after Fix B.
**Predecessors:** [33_wire_a_post_fix_a.md](33_wire_a_post_fix_a.md) (post-Fix-A clean substrate→LLM measurement).

## Pre-registration

| Criterion | Pre-registration | This run |
|---|---|---|
| PRIMARY: Arm A `sense_food_source` count | ≥ 1 | **0** |
| STRETCH: Arm A > Arms B+C on `sense_food_source` | A > B,C | All three = 0 (no divergence) |
| Fix A still working (NAc keys `sim_aut`) | yes | **yes ✓** |
| Fix B fires in test arms | yes | **yes ✓** (1 entity resolved in Arm A) |
| Double-fire fixed (W2 doesn't fire in fixture path) | yes | **yes ✓** (W2 only in priming) |
| `sense_food_source` enters Arm A's active roster | yes | **NO — manifest LLM materialized "set of keys" instead** |

**Convergence outcome** (Arm A ≥1): substrate→action gap closes via Fix A + Fix B; W1+Wire-A+W2 validated end-to-end.
**Divergence-in-a-row outcome** (Arm A still 0): per v1_refinement.md §1.5 + refined Principle 4, two-divergence-in-a-row trigger formally fires toward encoder replacement (Roy-5a Stage 3 + JEPA, 1.2+).

**THIS verdict (formally) — with a sharply narrowed failure mode that opens a cheaper-than-encoder test.**

## Methods

**Spec:** [scenarios/roy/roy_3a_iteration.yaml](../../scenarios/roy/roy_3a_iteration.yaml) — unchanged from exp 33. Single-variable change vs exp 33 is the Fix B merge (PR #292).

**Env at runner:** `PYTHONPATH=src MAXIM_SUBSTRATE_PATH=1 MAXIM_LOG_FILE=/tmp/roy_3a_post_fix_b.jsonl`.

**Wall time:** 902.38 s (priming + 3 arms). Backend: qwen2.5-14b-instruct via `https://maxim.dennyschaedig.com/v1`. First runner attempt aborted at preflight (HTTP 502 — leader being updated); retry after leader recovered ran cleanly.

## Result — Fix B structural validation

**The pipeline works end-to-end.** Fix B's `_substrate_pretrigger` fired exactly once per test arm:

| Arm | Substrate | Fix B SEM_TRACE event(s) | Entities resolved |
|---|---|---|---|
| **a** | from_priming | "generating manifest (agent_id=sim_aut, biases=1)" → "1 entities resolved" | **`['set of keys']`** |
| **b** | blank | "skipped (NAc has no biases for agent_id=sim_aut)" | — |
| **c** | blank | "skipped (NAc has no biases for agent_id=sim_aut)" | — |

**Fix B's gates all worked correctly.** The substrate-aware kill-switch behavior is observable from JSONL (BLOCK 2 fold validated). Arms B+C correctly skip the manifest LLM call when the substrate has no biases.

**Double-fire fix validated.** Five `Scene manifest pre-trigger: N entities resolved` events fired during this run — but all five timestamps land in the **priming session** (11:31-11:39, before Arm A starts at 11:41). Zero W2-style events fired during the test arms; only Fix B's events did. The `if fixture_path is None` gate at orchestrator.py:1432 correctly disjoints the two paths.

## Result — the narrowed failure mode

Substrate signal at Arm A start: `sense_food_source` with bias `+0.997` (priming end) → `+0.768` (arm A end), in the `strongly rewarding` band throughout. Fix B passes this bias to the manifest LLM.

**The manifest LLM extracted two entity phrases from its output**: `['blue toy car', 'set of keys']`. Neither relates to `sense_food_source`. Only `set of keys` resolved into the scene (the blue toy car phrase didn't pass downstream design validation). The food entity that would have made `sense_food_source` invokable was never materialized.

The AUT LLM during Arm A's percept loop DID engage with the materialized entity. At t=1779990126.56:
```
[sim_nac] 🧠 new_observed: tool:infant_humanoid_pick_up → plan:I wonder what the soft thing
against my cheek is. It's probably the set of keys nearby.:success (RPE=+0.50, conf=0.50) 📈
```

So **the runtime annotation pipeline works**: the LLM saw the materialized entity, interpreted an incoming percept ("something soft drapes against your cheek") as that entity, picked it up, and recorded a positive RPE. The framework's substrate-aware-scene-augmentation is structurally working — the AUT LLM responds to materialized entities.

**The bottleneck is at the manifest LLM's substrate-bias → scene-entity translation.** It received `sense_food_source [strongly rewarding from prior experience]` and selected `set of keys` as the scene addition. The semantic bridge from "this tool is rewarding" to "the entity this tool senses" is weak in the current prompt.

## What this iteration definitively rules in/out

**Ruled OUT** as PRIMARY=0 causes:
1. ~~Fix B doesn't fire in the fixture path~~ — fired with `biases=1` and produced a SEM_TRACE event chain.
2. ~~Wire-A doesn't reach the LLM~~ — exp 33 ruled this out; Fix A's structural correctness re-validated here.
3. ~~W2 still fires double~~ — `Scene manifest pre-trigger` events are all from priming (cradle generative path); zero test-arm hits.
4. ~~The runtime annotation pipeline is broken~~ — the AUT picked up the materialized entity; sense_tools / examine paths are reachable.

**Ruled IN** as the load-bearing cause:
1. **Manifest LLM substrate-bias → scene-entity semantic gap.** Given `sense_food_source` as the strongly-rewarding tool, the LLM generated `['blue toy car', 'set of keys']` — entities unrelated to food. This is the encoder-alignment gap Roy-5a's H1a verdict named, manifested at a specific narrow layer (the manifest LLM's prompt interpretation, not the LLM's general semantic capability).

## Verdict

**Pre-registered divergence-in-a-row trigger condition: MET on a strict reading.** Per v1_refinement.md §1.5:

> "If Arm A still produces 0 with Wire-A demonstrably reaching the LLM, the divergence-in-a-row trigger fires correctly on the next iteration with a clean instrument."

This iteration has the cleanest instrument yet (Fix A + Fix B both fire + observably correct), and Arm A still produces 0. Per refined Principle 4 + the kickoff's framing, encoder replacement (Roy-5a Stage 3 + JEPA, 1.2+) is the formally-triggered next step.

**HOWEVER**, this iteration also narrowed the failure mode to a single layer. The substrate signal reaches the manifest LLM. The runtime annotation pipeline works. The AUT LLM engages with materialized entities. The only thing that doesn't work is the manifest LLM's interpretation of substrate biases as scene-entity preferences. **This is testable with cheaper-than-encoder experiments:**

- **Option X1 (prompt engineering retry)**: improve `_compose_substrate_context` in `narrator.py` to explicitly instruct the manifest LLM to bridge tool-name biases to scene-entity selections. E.g., add a line: "For each rewarding tool, include in this scene an entity that activates it (e.g., for `sense_food_source` include a food source)." Single-line prompt addition. Re-run Roy-3a. If Arm A=≥1, the gap is prompt-engineering and encoder pivot is deferrable to 1.1+. If Arm A=0 again with the LLM seeing the explicit bridge instruction and STILL producing wrong entities, encoder-alignment is fundamentally needed (the LLM doesn't have the cross-modal mapping even when told).

- **Option X2 (encoder pivot)**: commit to Roy-5a Stage 3 (cradle redesign to produce paired sensor+text training data) + JEPA cross-modal binding. 1.2+ research direction. Heaviest scope.

Per the kickoff's "Triggering bird's-eye encoder work if divergence fires: NEVER without explicit authorization", the encoder pivot stays unauthorized. The X1 prompt-engineering retry is the cheaper one-step diagnostic that disambiguates "weak prompt" from "fundamentally missing encoder mapping." User decision required.

## Companion artifacts

- Result JSON: `~/.maxim/roy/roy-3a/result.json`
- Summary: `~/.maxim/roy/roy-3a/summary.md`
- Per-arm session snapshots: `~/.maxim/sim_reports/{20260528_113941, 20260528_114137, 20260528_114308, 20260528_114443}/`
- Runner JSONL: `/tmp/roy_3a_post_fix_b.jsonl` (40,535 events)

## Comparison to exp 30 → 32 → 33 → 34

| Measurement | Exp 30 (Phase B) | Exp 32 (W1+W2) | Exp 33 (post-Fix-A) | **Exp 34 (post-Fix-A+B)** |
|---|---|---|---|---|
| NAc key prefix | `default_agent` | `default_agent` | `sim_aut` | **`sim_aut`** ✓ |
| Wire-A reaches LLM | UNVERIFIED | UNVERIFIED | YES (+0.768) | **YES (+0.768)** ✓ |
| Fix B fires in test arm | n/a | n/a (Bug B) | n/a (Bug B) | **YES (1 entity)** ✓ |
| Materialized entity = food? | n/a | n/a | n/a | **NO (`set of keys`)** ✗ |
| AUT engages materialized entity | n/a | n/a | n/a | **YES (`infant_humanoid_pick_up`)** ✓ |
| Arm A `sense_food_source` | 0 | 0 | 0 | **0** |
| Verdict | NULL (inferred) | AMBIGUOUS-WIRING | DIVERGENCE-IN-A-ROW (pre-reg) | **DIVERGENCE + narrowed failure mode at manifest LLM** |

## Plan-doc folding

- [docs/plans/archive/v1_refinement.md](../plans/archive/v1_refinement.md) §1.5: post-Fix-B verdict — pre-registered divergence-in-a-row trigger met; cheaper prompt-engineering retry (Option X1) is the natural next test before encoder pivot commits.
- [docs/plans/deferred/imagination_substrate_signals.md](../plans/deferred/imagination_substrate_signals.md): W2 Bug B fix validated end-to-end (Fix B fires structurally correctly); the surfacing layer (`_compose_substrate_context`) needs prompt-engineering refinement to bridge tool-bias → scene-entity selection. Hookup 2 (per-tick subscriber) and Hookup 3 (arousal-gate) remain 1.1+; Hookup 1's prompt refinement is the immediate cheaper move.
- [docs/plans/deferred/sense_tool_registry.md](../plans/deferred/sense_tool_registry.md): W1's grayscale visibility validated structurally; Arm A's tool roster still excludes `sense_food_source` (no food entity materialized because the manifest LLM didn't bridge from substrate bias to food).
- [docs/plans/archive/cross_modal_substrate_binding.md / jepa_plan_drafted.md] (if drafted): the post-Fix-B narrow gap (manifest LLM substrate-bias → scene-entity selection) is the strongest signal yet for the encoder-alignment thesis. JEPA plan should reference this iteration as direct motivation if Option X1 fails.
