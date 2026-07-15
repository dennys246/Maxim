# 32 — Wire-A post-W1+W2 integration test (Roy-3a)

**Status:** Run complete 2026-05-27. **AMBIGUOUS-WITH-WIRING-BUG verdict.** PRIMARY criterion failed but two upstream wiring gaps identified — neither sits inside W1 or W2 themselves.
**Branch:** main (post-PR-#288 merge at `5bf450b`).
**Plan:** [docs/plans/archive/v1_refinement.md](../plans/archive/v1_refinement.md) §1.5 integration test.
**Predecessor:** [30_wire_a_tau_validation.md](30_wire_a_tau_validation.md) (Phase B baseline, pre-W1+W2).
**Companion plans:** [sense_tool_registry.md](../plans/deferred/sense_tool_registry.md) (W1 MVP) + [imagination_substrate_signals.md](../plans/deferred/imagination_substrate_signals.md) (W2 MVP).

## Pre-registration

Per the kickoff prompt and v1_refinement.md §1.5:

| Criterion | Pre-registration | This run |
|---|---|---|
| PRIMARY: Arm A `sense_food_source` count | ≥ 1 | **0** |
| STRETCH: Arm A > Arms B+C on `sense_food_source` | A > B,C | All three = 0 (no divergence) |

**Convergence verdict** (PRIMARY pass): Wire-A annotation→action pathway validated; W1+W2 close the gap.
**Divergence-in-a-row verdict** (PRIMARY fail + new failure mode named): per refined Principle 4, two-divergence-in-a-row trigger; bird's-eye to encoder replacement.
**Ambiguous-with-wiring-bug** (PRIMARY fails AND Surfaces 3/4 show W1 or W2 didn't fire as designed): fix wiring before re-running.

## Methods

**Spec:** [scenarios/roy/roy_3a_iteration.yaml](../../scenarios/roy/roy_3a_iteration.yaml), unchanged from Roy-3 / Roy-3a-retry. Single-variable change is the W1+W2 MVP shipment (main commits `248992f` W1 + `5bf450b` W2).

**Env at runner:**
```
PYTHONPATH=src
MAXIM_SUBSTRATE_PATH=1
MAXIM_LOG_FILE=/tmp/roy_3a_post_w1_w2.jsonl
```

**Build state:** main @ `5bf450b` (W1 PR #287 + W2 PR #288 merged).

**Wall time:** 909.95 s (priming + 3 arms). Arm A duration 115.6s, B 98.5s, C 89.8s. Faster than the kickoff's 25-35 min estimate because the leader (qwen2.5-14b on RTX 5080 box) was warm.

**Backend:** `lane-large` → `qwen2.5-14b-instruct` via `https://maxim.dennyschaedig.com/v1` (peer mode, stage2 HTTP 200 at pre-flight).

**Arms:** identical to Roy-3a-retry / experiment 30.
- A: `substrate=from_priming`, `system_prompt=neutral` (the Wire-A test arm).
- B: `substrate=blank`, `system_prompt="You are a hungry infant"`.
- C: `substrate=blank`, `system_prompt=neutral`.

## Result — per-arm LLM-called tool distributions

| Tool | Arm A | Arm B | Arm C |
|---|---:|---:|---:|
| `respond` | 47 | 16 | 11 |
| `_llm_unavailable` | 4 | 1 | 0 |
| `infant_humanoid_move` | 2 | 0 | 0 |
| `sense_tools` | 1 | 1 | 1 |
| `infant_humanoid_pick_up` | 1 | 2 | 2 |
| `sense` | 0 | 3 | 2 |
| `sense_presence` | 0 | 1 | 1 |
| `say` | 0 | 1 | 1 |
| `pick_up` | 0 | 0 | 1 |
| **`sense_food_source`** | **0** | **0** | **0** |

**Comparison to Phase B (pre-W1+W2):** identical `sense_food_source = 0/0/0`. Arm A's `respond` count tripled (47 vs 11 in Phase B) — the LLM leaned even harder into verbalizing rather than acting.

## Result — substrate signal magnitude (Surface 2)

`cluster_reward_bias` from `aut_nac.json`:

| Phase | max\|bias\| | Band |
|---|---:|---|
| Priming-end | 0.9967 | strongly rewarding |
| Arm A end | 0.5822 | strongly rewarding |

The substrate signal is alive and inside the band Wire-A would render as `[strongly rewarding from prior experience]` if it reached the LLM prompt. Consistent with Phase B's reading.

## The wiring bugs surfaced

Direct JSONL search for W1's grayscale marker `not in current location` and W2's substrate-context marker `Substrate-acquired tool preferences` both return **0** occurrences across 42,003 events. This number was initially read as "W1+W2 didn't fire," but the broader inspection shows the diagnostic is itself blind: `compose_cluster_bias_annotation_section` and `compose_grayscale_tools_section` emit no `sim_log` event — they only mutate the LLM prompt text, which is NOT captured in the JSONL. The pattern-search returning zero is INCONCLUSIVE about whether the renderers ran.

Structural inspection then surfaced two distinct upstream wiring gaps, neither of which lives inside the W1 or W2 MVPs themselves:

### Bug A — Roy cross-session agent_id mismatch breaks Wire-A's substrate read

- The AUT MemoryHub in the orchestrator is constructed with `agent_id="sim_aut"` ([orchestrator.py:534](../../src/maxim/simulation/orchestrator.py)). `_loop_agent_id` resolves to `"sim_aut"` at [agent_loop.py:1074](../../src/maxim/runtime/agent_loop.py) (`memory_hub.agent_id` wins over `agent_name` fallback).
- The priming session for Roy uses a separate code path whose MemoryHub defaults to `agent_id="default_agent"` ([memory_hub.py:170](../../src/maxim/integration/memory_hub.py)). All 989 `sim_nac_recommend` events in this run carry `agent_id="default_agent"`; the 2 entries in priming-end `cluster_reward_bias` are keyed `default_agent\x1f<cluster_uuid>\x1ftool:sense_food_source`.
- The test arms inherit the priming NAc state via Roy's `substrate=from_priming` carryover. The dump still keys biases under `default_agent`.
- Wire-A at [agent_loop.py:2910](../../src/maxim/runtime/agent_loop.py) reads via `_loop_nac.get_agent_tool_biases(agent_id=_loop_agent_id, top_n=5)` — `_loop_agent_id` is `"sim_aut"` for the test arm AUT, so the strict-equality filter at [nac.py:1903](../../src/maxim/decisions/nac.py) returns `[]`.
- `context.cluster_bias_annotations` gets `[]`, the Wire-A section composer returns `""`, and the LLM never sees the annotation.

Direct simulation confirms:
```
get_agent_tool_biases(agent_id="default_agent") → [('tool:sense_food_source', 0.5822)]
get_agent_tool_biases(agent_id="sim_aut")        → []
```

The agent_id assigned to the AUT MemoryHub at orchestrator-construction time and the agent_id under which the priming-saved biases were written never aligned. Wire-A has been silently rendering empty in the Roy fixture path the entire 0.9.1 release window. The post-result reconstruction in [30_wire_a_tau_validation.md](30_wire_a_tau_validation.md) ("Wire-A rendered `sense_food_source [strongly rewarding from prior experience]` at every LLM submission") was inferred from priming-end magnitude + decay-trajectory math, not verified from actual rendered prompts. The 0.997 → 0.753 decay in Phase B that the doc cited as evidence of the annotation being read actually only proves that `NAc.decay_cluster_reward_biases` (agent_id-agnostic) ticked during the arm — it does not prove Wire-A's reader ran with a valid agent_id.

**W1's grayscale producer inherits the same blindness** because it explicitly reuses `context.cluster_bias_annotations` ([agent_loop.py:2959](../../src/maxim/runtime/agent_loop.py)) to avoid a second NAc call — the comment block says that's deliberate. When the reused value is `[]`, the grayscale section composer also returns `""`. W1's renderer is correct; its input is empty.

### Bug B — W2's hookup site is structurally bypassed by fixture-driven test arms

- W2's substrate-aware manifest fires at [orchestrator.py:1468](../../src/maxim/simulation/orchestrator.py) via `generate_scene_manifest(llm_router, goal, nac_top_biases=_aut_nac_biases)`.
- This code path runs only at scene-load time for AUT sessions that drive scenes via the generative narrator (cradle, free-form `--sim`). The 10 `narrator_generation` events in this run are all from the priming session (timestamps 1779932412.28 — 1779932555.05, ~19:40:12 — 19:42:35 MDT). The test arms (`20260527_194951` onward) emit zero `narrator_generation` events.
- Roy test arms use the `roy_1_holdout.yaml` fixture, which bypasses `generate_scene_manifest` entirely. W2's hookup point is never reached during the arm where the read-back would matter.
- Even if Bug A were fixed and `_aut_nac_biases` arrived populated, W2 still wouldn't influence the test arm percepts under the current Roy fixture-driven design.

W2's MVP plan explicitly cites cradle as the precedent for "the manifest pre-trigger runs once at scene load (already in production for cradle)." That precedent is correct — it just doesn't extend to Roy's holdout-fixture path. The W2 MVP itself is correctly wired for its named precedent. The integration test's pre-registration assumed fixture-driven test arms would benefit; that assumption is the gap.

## Verdict

**AMBIGUOUS-WITH-WIRING-BUG** per the kickoff's classification rule. PRIMARY fails AND structural analysis shows W1+W2 did not fire as designed in this fixture-driven test path (one renderer's input was silently empty; the other's call site was unreachable). The kickoff's escalation rule:

> "Wiring bug (Surfaces 3/4 show no firing): classify as 'ambiguous-with-wiring-bug'; fix in separate PR before re-running and classifying."

This is NOT a divergence-in-a-row trigger. Refined Principle 4 fires when "PRIMARY fails AND post-hoc findings each spawn new follow-up plans" — but the two findings here are **wiring bugs**, not new failure modes of the substrate→action conversion thesis. The thesis itself has not been measured by this iteration; the measurement instrument was broken.

The bird's-eye to encoder replacement (Roy-5a Stage 3 + JEPA) is therefore **not authorized** by this iteration. Fix the wiring, re-run, then classify.

## Recommended fixes (separate PRs)

Neither fix should ship without explicit two-lens pre-merge review — both involve cross-session contract decisions that affect Roy measurement integrity and the broader bio-system stash invariant. **Directions chosen 2026-05-27** post-verdict surfacing; scoping pending.

### Fix A — Align Roy priming/test agent_id

**Direction chosen: (1) Roy priming explicitly constructs its MemoryHub with `agent_id="sim_aut"`** so the cross-session NAc dump's keys match what the test arm AUT reads. Smallest delta; preserves the existing `default_agent` MemoryHub default for non-Roy callers; symmetry to the AUT config at [orchestrator.py:534](../../src/maxim/simulation/orchestrator.py). Bio-fidelity-clean per the per-agent stash invariant.

Rejected alternative (2): Roy test arm AUT inherits `agent_id="default_agent"`. Larger blast radius (the AUT's `sim_aut` identifier appears in nicknames, request contexts, log routing); risks coupling Roy specifically to the MemoryHub default rather than its own identity.

Scoping question that should land in the Fix A plan doc: how many OTHER 0.9.1 Roy iterations have been measuring "Wire-A rendered" when the actual rendering was silently empty? Experiment 30's [retroactive correction](30_wire_a_tau_validation.md) tracks this for the Phase B iteration; earlier Roy iterations (Roy-1a / Roy-2 / Roy-2pc / Roy-2c) used pre-annotation 0.9.0 codebases so Wire-A wasn't yet in play, but their Wire-A-claim citations in retrospective summaries should be checked.

### Fix B — Extend W2 hookup to fire on fixture scene-load

**Direction chosen: (b1) extend W2 to fixture scene-load.** Add a substrate-aware step to the fixture-loading code path parallel to `generate_scene_manifest`. The contract surface is larger (manifests have to mean something for a pre-set fixture) and bumps directly into W2's [Open Question 5](../plans/deferred/imagination_substrate_signals.md) (self-reinforcing preference loops — bio-fidelity reviewer's flag). Empirical-grounding constraint ("biased entities appear only if present in ≥N% of past sessions") should be re-read during scoping; if the constraint applies cleanly to fixture-driven scenes, it answers the self-reinforcement worry that originally drove the deferral.

Rejected alternative (b2): Roy switches test arms to a generative-narrator path. Sidesteps the mechanism question but abandons fixture-driven reproducibility — not the right move without an explicit Roy-redesign authorization.

### Re-run after Fix A only?

Fix A alone is necessary but not obviously sufficient. After Fix A:
- Wire-A's annotation should reach the LLM in the test arm (the priming biases are now keyed to match the test arm's `_loop_agent_id`).
- W1's grayscale rendering should fire (it consumes the Wire-A biases that Fix A unblocks).
- W2 remains structurally unreachable in the fixture-driven path.

This is enough to test the **Wire-A + W1 contribution to the substrate→action gap** independently of W2. If Arm A produces ≥1 `sense_food_source` call after Fix A alone, W1+Wire-A close the gap; W2 becomes a 1.1+ enhancement rather than a 1.0 critical path. If Arm A still produces 0 with Wire-A demonstrably reaching the LLM, the divergence-in-a-row trigger fires correctly on the next iteration with a clean instrument.

## Pre-existing observations

- Arm A's `respond=47` vs Phase B's `respond=11` quadrupled. With or without the wiring bugs above, the agent is talking more and acting less in this run than in Phase B — possibly an LLM-side variance (different leader warm state, different model version) but worth noting before the next iteration. The `_llm_unavailable=4` count is also non-trivial for a 10-turn arm; a handful of LLM calls failed mid-arm.
- Total LLM-called `sense*` family: Arm A=1, B=5, C=4. Inverted vs Phase B (A=7, B=6, C=1). Wire-A's "annotation lifts sensing tools in Arm A" hypothesis from Phase B is also weakened by this run if both readings hold.
- Substrate-primary priming was productive: 2 `cluster_reward_bias` entries for `sense_food_source` at +0.997/+0.786 max magnitude; per-cluster reward variance accumulated; ATL concepts +192 vs B.

## Companion artifacts

- Result JSON: `~/.maxim/roy/roy-3a/result.json`
- Summary: `~/.maxim/roy/roy-3a/summary.md`
- Per-arm session snapshots: `~/.maxim/sim_reports/{20260527_194804, 20260527_194951, 20260527_195146, 20260527_195325}/`
- Runner JSONL: `/tmp/roy_3a_post_w1_w2.jsonl` (42,003 events)
- Runner stdout: `/tmp/roy_3a_post_w1_w2_runner.log`

## Plan-doc folding (per kickoff B6)

- [docs/plans/archive/v1_refinement.md](../plans/archive/v1_refinement.md) §1.5: add ambiguous-with-wiring-bug outcome line; Fix A + Fix B become 1.0 critical path before the next Wire-A Roy iteration can run.
- [docs/plans/deferred/sense_tool_registry.md](../plans/deferred/sense_tool_registry.md) "MVP shipment" section: integration test ran but cannot validate W1's behavior until Bug A is fixed and a clean re-run lands. The MVP code itself is structurally correct; the upstream input is empty.
- [docs/plans/deferred/imagination_substrate_signals.md](../plans/deferred/imagination_substrate_signals.md) "W2 MVP shipment recap": integration test reveals W2's hookup site is structurally bypassed by Roy's fixture-driven test arms. Either move the hookup or change the Roy spec — surface decision to user.
- [docs/plans/deferred/persona_convergence_crucible.md](../plans/deferred/persona_convergence_crucible.md): Roy iteration log entry. NULL outcome on PRIMARY; two distinct upstream wiring bugs; not a divergence signal.
- [docs/plans/behavioral_graduation_candidates.md](../plans/behavioral_graduation_candidates.md): no Earned-tier movement (Wire-A still pre-Earned; the planned validation iteration is blocked on Fix A).
