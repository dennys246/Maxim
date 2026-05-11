# Grounded Language Acquisition + Substrate-Primary AUT (parallel-mode architecture)

**Status:** ACTIVE — promoted from deferred 2026-05-09 after E4 validation surfaced the LLM-band-aid drift (60-70% of recent engineering effort spent on LLM-mitigation scaffolding, ~845 LOC of band-aid code, growing). The MVP for negative-instruction tool-failure hints failed validation (n=6 per arm, no benefit observed; default flipped OFF), making the architectural pivot urgent.
**Begins:** Phase -1 + Phase 0 harness in 1.0 (parallel to docs work). Full Phase 0 validation, Phase 1, and Phase 2 in 1.1+. Substrate-primary AUT mode (the parallel-architecture work) interleaves with the language phases.
**Companion plans:** [persona_convergence_crucible.md](persona_convergence_crucible.md) (Roy methodology — same long-horizon shape), [behavioral_convergence_practice.md](behavioral_convergence_practice.md), [memory_consolidation_practice.md](memory_consolidation_practice.md), [v1_refinement.md](v1_refinement.md) (Phase 0 harness scope add)
**Operating context:** Roy long-horizon simulations (sim-years of subjective experience) with persistent substrate across sessions, plus deliberately text-heavy curricular sims (mom-reading, teacher-student, dialogue). The substrate-primary AUT runs in **parallel mode** — the existing LLM-AUT path remains available so users can continue running D&D campaigns and other long-horizon LLM-driven sims while this work matures.

## Architectural framing — parallel modes, not replacement

The work in this plan ships as a **parallel AUT mode** alongside the existing LLM-driven AUT. Users continue to get the LLM-AUT path by default; the substrate-primary mode is opt-in (e.g., `--aut-mode substrate-primary`) until convergence proves out.

**Where LLMs continue to live (and where prompt engineering continues):**
- **Orchestrator** (sim driver / DM) — LLM-driven; prompt-shaped scenario generation
- **Environment NPCs** — LLM-driven characters in scenes
- **LLM-AUT mode** (default) — the existing path, kept fully maintained
- **Imagination designer** — LLM-driven entity design at scene boundaries

**What substrate-primary mode replaces:**
- The AUT's action-selection LLM call only. NAc/reflexes/DN/imagination compose the action choice; no instruction-following layer between percept and motor.

**Why parallel and not replacement:** the existing LLM-AUT path delivers value today (D&D, Reachy demos, the published 1.0 thesis). Tearing it out would be reckless; building substrate-primary alongside lets us A/B at every stage and makes the kill criterion in the next section enforceable.

## D&D as the bidirectional kill criterion

A substrate-primary AUT that cannot survive a D&D-style campaign orchestrated by an LLM-DM is a failed bio-substrate. **AND** a simulation environment that cannot be navigated by a learning substrate is a failed simulation environment. The convergence test is **mutually load-bearing**:

| Outcome | Diagnosis |
|---|---|
| Substrate AUT runs the campaign cleanly | Substrate is real. Project thesis validated. |
| Substrate AUT fails; LLM-AUT succeeds in same scenario | **Substrate insufficient** for non-trivial cognition. Reframe required. |
| LLM-AUT also fails the same scenario | **Simulation environment is the failure** — the test isn't measuring what we think it is. |
| Both succeed but substrate is much weaker | Acceptable interim. Substrate scope clear; LLM remains in user-facing path. |

D&D is the right test because it has: long-horizon temporal structure, novel entities every session, decision-making with delayed reward, role-coherence demands, multi-agent dynamics. If the substrate works only in the cradle and breaks in D&D, we shipped a sensorimotor toy. The cradle is necessary; D&D is sufficient.

## What this is

A staged research program to remove the "pretrained LLM cheats" objection from Maxim's core thesis ("bio-systems carry the cognition; language is I/O"). The endpoint of the program — *if it gets there* — is a Maxim agent whose linguistic competence is acquired entirely through embodied simulation rather than imported from a pretrained model. The earlier stages are cheaper experiments that test the thesis at lower cost and gate the expensive ones.

This is **not** "build microGPT and wire it to NAc." That framing skips two cheaper experiments that would tell us whether the headline experiment is even worth running. It also misreads the role of the language model in current Maxim — a from-scratch sequence model trained on sim-scale reward signals will fail not because of data scale (Roy-scale curriculum addresses that) but because the *objective* is wrong: you can't bootstrap language from reward alone, and you can't bootstrap grounding from autoregressive prediction alone.

## Framing rule: each phase justifies itself

Each phase ships a finding about the substrate, not a deliverable that depends on later phases working. If Phase 0 reveals that bio-systems can't form proto-concepts without linguistic supervision, the whole program stops and we've still learned something load-bearing about the substrate. If Phase 2 ships a working symbol-binding layer but Phase 3's from-scratch sequence model never converges, Phase 2 is still a real result.

This framing is mandatory because the program spans years and we cannot tie its value to the most ambitious phase clearing.

## What's wrong with the naive framing

The user's first instinct — "wire microGPT from scratch and let NAc supervise it" — has three problems that need to be named so the plan can avoid them:

1. **Reward-only training of a randomly-initialized transformer doesn't work.** This is RLHF without the SFT base. Known to collapse. The pretrained LLM in current Maxim isn't optional decoration; it carries the *prior* over plausible language that lets reward-shaping tune behavior rather than discover language.
2. **Autoregressive prediction ≠ grounding.** A from-scratch GPT trained on sim transcripts will memorize sequences. It won't bind tokens to bio-substrate concepts unless you add a grounding objective alongside the prediction loss. That objective is the actually-novel piece, not the transformer.
3. **The thesis ("substrate carries cognition") doesn't require removing the LLM.** It requires showing that the substrate does the work. You can show that with the LLM constrained or absent in specific channels — Phase 0 and Phase 1 below — without rebuilding the language model.

So: do the cheap thesis-tests first, then earn the right to the expensive build.

## Phase -1 — Substrate action-generation prototype (~1 week, gate before all other phases) — **GATE CLEARED** (2026-05-09)

**Question (the most important one in this plan):** Can the substrate generate even a single non-reflex action without LLM proposal?

**Result:** **YES.** Shipped via PR #228 (commits `75d1112` + `b02a070`). `NAc.recommend_action()` produces an action proposal from causal-link confidence + reward bias + drive-relevance heuristic; `propose_via_substrate()` in [agent_loop.py](../../src/maxim/runtime/agent_loop.py) wraps that into an `LLMProposal` for the executor. 11 unit tests in [tests/unit/test_nac_recommend_action.py](../../tests/unit/test_nac_recommend_action.py) + 11 integration tests in [tests/integration/test_substrate_primary_aut.py](../../tests/integration/test_substrate_primary_aut.py). Phase 0 unblocked.

**Why this comes before Phase 0:** the bio-substrate readiness audit (2026-05-09) scored **2/10 for action selection**. NAc predicts and biases but does not propose. DN does autonomous gaze but no tool invocation. Reflexes cover only 2 thermal cases. **There is no `decision_engine.decide()` that does not go through an LLM proposal today.** Phase 0 strips language from the AUT prompt but leaves the LLM as motor-primitive selector — it does NOT prove the substrate can act on its own. This phase does.

**Setup:**
- Single-tick prototype, not a full sim. AUT given one entity (food) and one goal-relevant drive (hunger > 0.5).
- LLM call PATH disabled entirely on the AUT side (orchestrator can stay LLM).
- Substrate-only action generation: `NAc.recommend_action(state) → (tool_name, params, confidence)` — new method, prototype implementation. May initially be "the most-positive-bias tool from active EC nodes," `argmax` over reward_bias.
- Action dispatched if confidence > threshold; otherwise no-op.

**Success criterion:** the AUT calls `pick_up_food` (or any goal-relevant tool) on at least one tick across 20 trials without an LLM proposal. Just one. Proves the substrate can produce an action.

**Failure criterion:** zero non-reflex actions across 20 trials. Means NAc cannot serve as action-proposal source even in the simplest case. The substrate-primary architecture is not feasible as currently structured; the program redirects to building a minimum action-generation layer (new bio-system or significant NAc extension) **before** Phase 0 is worth running.

**Implementation surface:**
- New method `NAc.recommend_action(state, available_tools) → ActionProposal | None` (~50 LOC, prototype quality)
- New `--aut-mode substrate-primary` CLI flag (~20 LOC plumbing)
- Single-tick test harness in `tests/integration/test_substrate_action_generation.py` (~80 LOC)
- No new bio-systems; just expose existing NAc data in the right shape.

**Calibration:** this phase is a Boolean. It either produces an action or doesn't. No nuance, no curves, no instrumentation overhead. If it works, Phase 0 proceeds. If it doesn't, the plan stops and refactors.

## Phase 0 — Pre-linguistic cradle baseline (~2 weeks, ships finding regardless) — **HARNESS SHIPPED** (2026-05-09); validation pending

**Status:** harness shipped via PR #228 (commits `b02a070` + `78d9683`); validation blocked on Roy harness + EC sensor-encoding gap.

**Harness deliverables (shipped):**
- `cradle_prelinguistic` arc in [simulation/arcs.py](../../src/maxim/simulation/arcs.py) — same developmental scaffolding as `cradle`, all English instructions stripped.
- Motor-only AUT prompt renderer in [prompts/motor_only_aut.py](../../src/maxim/prompts/motor_only_aut.py) — numeric drives + sensors + bare tool names; English-leak sentinels in tests.
- Per-tick `SubstrateTelemetry` JSONL writer in [simulation/substrate_telemetry.py](../../src/maxim/simulation/substrate_telemetry.py) — captures EC node count, NAc reward bias, drive states, active proposal. Fail-soft (never crashes the loop).
- `--research` with `--aut-mode substrate-primary` enables telemetry; routes through `start_simulation_mode` (not the multi-agent paper harness).
- 13 harness tests in [tests/integration/test_phase0_harness.py](../../tests/integration/test_phase0_harness.py).
- Smoke run cleared: 38 actions, 61 causal links, hunger drift 0.0 → 0.65 over 5 turns. See [docs/experiments/13_phase0_harness_smoke.md](../experiments/13_phase0_harness_smoke.md).

**Validation gaps (pending):**
- ~~**EC sensor-encoding entry point**~~ — **CLOSED 2026-05-09 (second commit, branch `feat/phase0-sensor-encoding`):** `SensorEncoder` in [similarity/encoder.py](../../src/maxim/similarity/encoder.py) hashes drive snapshots into a 384-dim embedding via low/high basis interpolation and routes through EC `pattern_complete_or_separate` with modality `"interoception"`. Wired into [agent_loop.py::propose_via_substrate](../../src/maxim/runtime/agent_loop.py) — fires once per substrate-primary tick before reading drives. Smoke re-run produced EC `node_count` 0 → 1 with modality `"interoception": 1`. Surprising finding (smooth drift collapses to one cluster — EC centroid tracking + shared bases for unchanged sensors) documented in [13_phase0_harness_smoke.md](../experiments/13_phase0_harness_smoke.md); Phase 0+ refinement targets enumerated there. The bare measurement gap (anything > 0) is closed; cluster-purity tuning waits on Roy.
- **G4: Substrate-primary reward-feedback wire — Track 2 deferred, Roy-0 caught** (2026-05-11). Roy-0 ran 15 min end-to-end against a healthy leader with `aut_mode=substrate-primary` and produced **zero action proposals** (`proposal=none` × hundreds of loop ticks). Three parallel investigations (static gate trace + commit forensics + persisted-state inspection) converge on the same root cause: **the cluster-keyed reward update wire was explicitly deferred when cluster-keyed action *selection* shipped, so NAc has nothing learned to recommend from.**

  **What's wired (substrate-primary commit series):**
  - `78d9683` — Phase 0 harness: `propose_via_substrate()` called from [agent_loop.py:2654-2670](../../src/maxim/runtime/agent_loop.py) when `aut_mode == "substrate-primary"`.
  - `e293eb7` — `SensorEncoder.encode_sensors()` fires once per tick before `recommend_action`; drives hashed into 384-dim embedding, routed through `EC.pattern_complete_or_separate` with modality `"interoception"`.
  - `fad326f` — `frozen_centroid_modalities` config in [similarity/ec.py](../../src/maxim/similarity/ec.py) prevents interoception centroid from drifting.
  - `6d0e4a7` — `NAc._cluster_reward_bias` dict + `update_cluster_reward()` API + `recommend_action(current_cluster_id=...)` parameter; [agent_loop.py::propose_via_substrate](../../src/maxim/runtime/agent_loop.py) captures `cluster_id` from `encode_sensors()` and threads it into `recommend_action`.
  - `6643755` — orchestrator narration silenced when `aut_mode == "substrate-primary"` ([simulation/bridge.py](../../src/maxim/simulation/bridge.py) skips `inject_cli` and `percept_anxiety_hook`).

  **What's deferred (the G4 wire):** the `6d0e4a7` commit message states explicitly: *"Reward update wiring (cluster_id-aware `record_outcome`) is deliberately out of Track 2's scope — the API exists and is unit-covered, but the agent_loop's `_record_outcome` chain doesn't yet plumb cluster_id."* `cluster_id` is captured at proposal time but never travels to the post-execute outcome record, so `NAc._cluster_reward_bias` stays empty AND the per-tool `reward_bias` stays empty across every session.

  **Empirical confirmation from Roy-0 (`~/.maxim/sim_reports/20260510_213337/` priming final + `20260510_213527/` arm A test):**
  - `aut_nac.json::reward_bias` — **0 keys** in both priming-final and arm A. `goal_reward_bias` carries only the two test probes (`cradle_prelinguistic: 0.196` + an arm-specific key).
  - `aut_nac.json::links` — 133 causal-link targets accumulated, but the `outcome_signature` fields encode outcome *descriptions* (`"tool:sense_food_source:positive"`, `"plan:drive:hunger(0.50) → food:success"`) rather than affordance-name → reward valuations the proposer could query.
  - `aut_hippocampus.json::episodes` — 662 episodes captured but mostly `valence = -0.25` (passive observation), no `text`/`percept_refs`, no `concept_index` entries.
  - `ec.json` — not present in any session dir (EC writes go through the live `EC` instance during the run but aren't persisted under the `aut_*` prefix the way NAc/Hippocampus are; orthogonal to G4 but worth tracking — visible to the proposer in-process, invisible to substrate_diff across sessions).
  - `~/.maxim/roy/roy-0-smoke/result.json::pairwise_diffs` — `reward_bias_l2: 0.0` and `reward_bias_top_deltas: []` on every pair, confirming reward_bias never populated in any of the three arms.

  **Where the silent gate fires:** [decisions/nac.py::recommend_action](../../src/maxim/decisions/nac.py) at lines 1295 (`if not scores: return None`) and 1300 (`if best_score < min_confidence: return None`, default `min_confidence=0.3`). With empty `reward_bias` + empty `_cluster_reward_bias` + no causal-link confidence to score against + cold-start drives, every available tool scores 0.0. `recommend_action` returns None. [agent_loop.py:742](../../src/maxim/runtime/agent_loop.py) sees None and returns None. The substrate-primary branch at [agent_loop.py:2664](../../src/maxim/runtime/agent_loop.py) leaves `ctrl.pending_proposal` unset. Next tick: same gate fires the same way. No log line surfaces — the only positive log (`"substrate-primary proposal: tool=..."`) fires inside the `if substrate_proposal is not None:` branch.

  **What closing it costs (concrete change surface):** plumb `cluster_id` from `propose_via_substrate`'s capture site through to every `_record_outcome` call in [agent_loop.py](../../src/maxim/runtime/agent_loop.py), then have those sites invoke `NAc.update_cluster_reward(agent_id, cluster_id, tool_signature, reward)` alongside the existing reward-update calls. Per the `record_outcome_call_sites` feedback memo, there are **~7 `_record_outcome` sites in agent_loop.py** (grep `nac=_loop_nac`). The cluster_id needs to ride on the pending-proposal envelope (or be re-derived from sensor state at outcome time — re-derivation is the cheaper option since sensors are read every tick anyway, but it requires the outcome callback to know which tick's cluster matters and that may not be stable across tool-execution latency). Estimate: 1-2 days for the threading work + invariant tests, then a fresh Roy-0 run to confirm `reward_bias_l2 > 0` on the A-vs-blank pairs.

  **What does NOT close G4 alone:** populating `reward_bias` is necessary but not sufficient for the proposer to fire. The `min_confidence=0.3` gate means a few small updates won't cross threshold; the cold-start regime needs either (a) lowered threshold for the substrate-primary path specifically, (b) drive-affinity scoring that's non-zero from the first tick when a relevant drive crosses the 0.5 activation cutoff, or (c) cluster-bias accumulation tuned to dominate at small N. Track 2's design contemplates (c) — `update_cluster_reward` is meant to give cluster bias the loudest voice in selection — but only if there are *any* cluster reward observations in the bank, which loops back to the same threading gap. **Roy-1 should not run until a sub-Roy-0 (10-turn synthetic) shows `reward_bias_l2 > 0` after wire close.**

  **Why this belongs in Phase 0 not Roy:** the wire and its closure are properties of the substrate-primary architecture, not the persona-convergence harness. Roy-0 only surfaced the gap because it was the first long-horizon run with substrate-primary that captured persisted diffs across arms. Until G4 closes, every Roy iteration's "active substrate" measurements (reward_bias L2, valence KS, episode valence distribution) read 0 regardless of priming — the harness works, but the substrate has nothing active to compare.

  **Cross-references:** Roy-0 iteration log entry in [persona_convergence_crucible.md § Iteration log](persona_convergence_crucible.md), commit `6d0e4a7` message, [tests/integration/test_substrate_primary_aut.py](../../tests/integration/test_substrate_primary_aut.py) (5 unit-coverage tests for `update_cluster_reward` exist already; the wire is what's missing).

- **Roy long-horizon harness** — Phase 0 validation needs sim-years of subjective experience across persistent substrate sessions. Designed but unbuilt; tracked as the gating dependency for Phase 0/1 validation, persona convergence, and D&D survival testing.
- **Pre-linguistic orchestrator narration** — the LLM-DM still emits English narration percepts the substrate-primary AUT ignores. Phase 0's "no English" goal is met on the AUT side but not the orchestrator side; eventually the orchestrator should produce sensor writes only (or be silenced entirely).

**Question:** Can EC/ATL/NAc form coherent, persistent concepts with zero linguistic supervision?

**Setup:** Strip English narration entirely from cradle acts 1-2 in [_data/components/items/cradle_*.yaml](../../src/maxim/_data/components/items/) and the matching narrative phases in [simulation/arcs.py::BUILTIN_ARCS["cradle"]](../../src/maxim/simulation/arcs.py). The infant body still has drives (hunger, thermal, contact) and proprioceptive sensors via [_data/components/bodies/infant_humanoid.yaml](../../src/maxim/_data/components/bodies/infant_humanoid.yaml), and reflexes still fire from [_data/reflexes/infant.yaml](../../src/maxim/_data/reflexes/infant.yaml). What's removed: every scripted line of mother-narration, every English-named affordance description in the prompt, every word in `phase.summary`. The LLM is still present (the agent has to act), but it receives only sensorimotor channel data and produces only motor primitives — no English in either direction.

**Measurement:**
- EC node count over time: do clusters form?
- Cluster purity against held-out test stimuli (does "warm contact with caregiver" reliably activate the same EC region across sessions?)
- NAc reward_bias distribution: are biases forming around drive-resolution events?
- Cross-session persistence: load the substrate into a fresh sim — do the proto-concepts re-activate on matching sensory patterns?

**Implementation surface:**
- New cradle arc variant: `cradle_prelinguistic` in [simulation/arcs.py](../../src/maxim/simulation/arcs.py).
- Motor-only AUT prompt template — strips the Acting Coach's linguistic framing, presents drive/sensor state and available motor primitives only.
- Telemetry: per-tick EC activation snapshot, NAc reward_bias snapshot, sensor reading snapshot. Existing JSONL writers extend.
- No new bio-systems. No new buses. This is a sim-config and prompt-shaping experiment.

**Gate / kill criterion:**
- **Pass:** measurable EC cluster formation tied to repeating sensorimotor patterns; NAc reward_bias differentiates drive-resolution events from null events; cross-session re-activation > chance.
- **Fail:** no detectable substrate structure after 50+ sessions of Roy-scale exposure. **Stops the program.** If bio-systems can't form concepts without linguistic supervision, the thesis is wrong and Phases 1-4 have nothing to ground in.
- **Mixed:** clusters form but don't persist, or persist but don't generalize. Diagnostic, not fatal — feeds [memory_consolidation_practice.md](memory_consolidation_practice.md).

**What this DOESN'T test:** language. Phase 0 is intentionally pre-linguistic. Whether words can later bind to these clusters is Phase 2's question.

## Phase 1 — Vocabulary-constrained output (~2 weeks)

**Question:** When the LLM can only produce words that bind to substrate concepts, does behavior stay coherent or collapse?

This is the cheapest possible test of "the substrate carries the cognition" without rebuilding anything. Keep the pretrained LLM as input parser. At output time, mask logits so the model can only produce tokens that have an active binding to an EC node (Phase 0's clusters, plus any substrate concepts formed during the agent's prior life). Words for things the substrate has never encoded — including most of the LLM's pretrained world knowledge — are unavailable.

**Setup:**
- Symbol-binding registry: a persisted map `token_id → ec_node_id` populated from co-occurrence during normal language-on simulations. Until the registry is populated, the agent operates mute (motor-only).
- Output-time logit mask in the LLM call path. Gates words by binding strength.
- Test scenarios: a held-out task suite where success depends on the agent describing things it has and hasn't experienced. The gap between "describe an object you've handled" (substrate-bound, vocabulary available) and "describe an object from training data only" (substrate-unbound, vocabulary masked) is the load-bearing measurement.

**Measurement:**
- Behavioral degradation curve: how does task success scale with binding-registry size?
- "Knowledge leakage" detection: does the agent attempt to use pretrained-world-knowledge words at frequencies above its substrate exposure? Constraint failures are findings.
- Comparison against unconstrained baseline at matched substrate maturity.

**Implementation surface:**
- [models/language/maxim_peer_backend.py](../../src/maxim/models/language/maxim_peer_backend.py) gains an optional `logit_mask_provider` parameter. The peer-backend invariants (one HTTP call, typed failure mapping) stay intact — masking is local pre/post-processing, not an extra request.
- [decisions/nac.py](../../src/maxim/decisions/nac.py) or [similarity/ec.py](../../src/maxim/similarity/ec.py) exposes a `bound_tokens()` accessor.
- Co-occurrence learner runs as a passive observer subscribed to the existing percept/action buses; no new orchestrator.

**Gate / kill criterion:**
- **Pass:** behavior degrades gracefully with vocabulary size; substrate-bound vocabulary is sufficient for in-distribution task success; out-of-distribution tasks fail in the predicted way.
- **Fail:** behavior collapses even when the substrate has rich bindings, OR behavior survives masking trivially because the LLM routes around it. Both findings adjust Phase 2's design but don't kill the program — they tell us where the LLM is doing more than I/O.

**What this DOESN'T test:** producing language from scratch. The LLM is still doing all the sequence generation; we're just gating its vocabulary. Phase 2 builds the actual binding mechanism Phase 1 here borrows.

## Phase 2 — Symbol-binding layer (~2 months)

**Question:** Can a small, online-trained associative model bind words to bio-substrate concepts well enough to substitute for the LLM's vocabulary on bound concepts?

**Architecture (deliberately not a transformer):**
- Inputs: word-token sequences (subword level, but with a *new* tokenizer trained from the simulation corpus — not the pretrained LLM's tokenizer, which leaks subword priors).
- Outputs: distributions over EC node IDs.
- Training signal: co-occurrence in simulations where percept ground truth is known (mom-reading sims emit `(token_sequence, scene_entity_id, ec_node_id)` tuples; teacher-student sims emit explicit naming events).
- Model: simple — embedding lookup + small MLP, or a tiny RNN. Not a transformer. The point is to see what the smallest model that can bind looks like, then scale up only if it fails.

**Why not a transformer here:** because the question isn't "can a sequence model learn language" — it's "can the substrate provide enough grounding signal that any reasonable binding architecture works." If the simplest model works, the result is stronger. If it fails and a transformer succeeds, that's also informative — tells us the binding requires within-sequence context, not just word-to-concept mapping.

**Persistence:** weights live in `~/.maxim/util/grounded_language/` per-agent, persist across sessions, version-tagged via the standard `_format_version` contract per [CLAUDE.md](../../CLAUDE.md) v1.0 freeze. Online updates batched per-session-end alongside the Hippocampus consolidation pass. Catastrophic forgetting is the obvious risk — mitigation is replay-based: hippocampal episodes carrying linguistic content get sampled into the binding-layer's training batch during sleep replay, mirroring biological consolidation.

**Curriculum integration:**
- Mom-reading sims become the primary high-density linguistic input channel. New sim type: scripted narrator-driven sims where the LLM-mom emits text deterministically tied to known scene state. Token-to-EC ground truth is automatic.
- Teacher-student sims provide explicit naming events: "this is a sword. say 'sword.'" Used for direct supervision in early phases, weaning to indirect co-occurrence as the binding registry matures.
- Standard generative sims feed the binding layer passively — no curriculum metadata, just whatever co-occurrence the AUT produces.

**Measurement:**
- Binding accuracy on held-out token-to-concept pairs.
- Generalization: does "flame" map to the EC cluster for "fire" (re-using the affordance concept transfer machinery from [affordance_concept_transfer.md](archive/affordance_concept_transfer.md))?
- Phase 1 re-run with binding registry populated by Phase 2 (instead of co-occurrence): does Phase 1 behavior improve?

**Gate / kill criterion:**
- **Pass:** binding accuracy on held-out pairs > 70% after Roy-scale curriculum; generalization to substrate-equivalent concepts works; replay-based consolidation prevents catastrophic forgetting across sessions.
- **Fail:** binding doesn't stabilize, OR stabilizes but doesn't generalize, OR catastrophic forgetting overwhelms replay. Each failure mode informs whether Phase 3 is worth attempting and which of its objectives need to do more work. The program continues to Phase 3 only if Phase 2 ships a usable binding registry.

## Phase 3 — From-scratch sequence model (~6-12 months, the headline)

**Question:** Can a sequence model trained from scratch on the Roy curriculum, with substrate-grounding as a co-equal training objective alongside next-token prediction, replace the pretrained LLM?

**Why this is now plausible (vs my initial pushback in conversation):** Roy long-horizon sims plus text-heavy curriculum (mom-reading, teacher-student) plus persistent substrate accumulation across sessions can plausibly approach infant-scale linguistic exposure. ~5K words/turn × 100 turns/sim × 100 sims = 50M words in the same order as a 4-year-old's exposure. The data scale objection from the conversation thread was wrong on that axis. The objective and grounding objections still stand and are addressed below.

**Architecture choices to settle in this phase:**
- Transformer vs RNN vs state-space. Default to small transformer (~10-100M params) because tooling is best, but the choice is not load-bearing for the thesis — it's load-bearing for whether anything trains at all in our constraints. Honest pick: smallest model that converges on the curriculum.
- Tokenizer trained from the simulation corpus (continuing Phase 2's choice). No pretrained tokenizers — they leak.
- Two-objective loss: standard autoregressive next-token loss + a substrate-grounding loss (predict the active EC node ID given the word context, or given the produced token, predict which EC concept it references). The grounding loss is the load-bearing novel piece. Phase 2's binding registry seeds the grounding labels.

**Reward shaping (NAc):**
- NAc reward signals enter as a *third* objective with low weight, not as the primary loss. This is the actually-correct framing of "wire NAc to the language model": it tunes a model that already has language structure from the next-token objective, rather than trying to discover language from reward alone.
- Reward enters at the utterance level (was this utterance followed by drive-resolution? did it lead to a goal-related outcome?), not the token level. Token-level reward shaping is known to produce reward-hacking in language models.

**Curriculum scheduling:**
- Phase 3a: from-scratch model trained on the Roy curriculum corpus *passively*, no NAc, no grounding loss. Pure next-token. This is a sanity baseline — does a small transformer trained on 50M words of Maxim curriculum produce coherent text at all?
- Phase 3b: add grounding loss. Measure whether tokens now bind to EC concepts beyond co-occurrence statistics.
- Phase 3c: add NAc reward signal. Measure whether utterance-level behavior shifts toward NAc-rewarded outcomes.

**Persistence:** weights persist across sessions per-agent (same path convention as Phase 2). Per-session checkpoints to allow rollback if a curriculum batch destabilizes the model. Replay-based consolidation continues to be load-bearing.

**Gate / kill criterion:**
- **Pass-3a:** model produces coherent in-distribution text after Roy-scale curriculum. Coherence is a low bar here — "passes a basic syntactic-fluency check" is enough.
- **Pass-3b:** grounding loss measurably aligns model token outputs with substrate concepts; held-out grounding accuracy > Phase 2's binding-registry baseline.
- **Pass-3c:** NAc reward shaping changes utterance distribution in the predicted direction without collapsing fluency.
- **Fail at any sub-gate:** ships a finding about which objective is doing what. Stops Phase 4 but doesn't unshipphases 0-2.

## Phase 4 — Pretrained-vs-grounded A/B (~3 months, only if Phase 3 clears)

**Question:** Does the Maxim agent retain its substrate behaviors when the pretrained LLM is fully replaced by the Phase 3 model? Are claims about the substrate's role corroborated or refuted?

**Methodology:** mirror the persona-convergence three-arm comparison ([persona_convergence_crucible.md](persona_convergence_crucible.md) Methodology section).

| Arm | Language model | Substrate |
|---|---|---|
| **A** | Pretrained 14B (current) | Roy-trained |
| **B** | Phase 3 from-scratch | Roy-trained |
| **C** | Pretrained 14B | Blank (control) |

The interesting comparisons:
- A vs B: same substrate, different language model. Does behavior persist? If yes, strong evidence the substrate carries the cognition. If no, the LLM was doing more than I/O all along — load-bearing finding either way.
- B vs C: from-scratch language + Roy substrate, vs pretrained language + blank substrate. If B beats C on substrate-dependent tasks, the program's thesis is corroborated.

**Publishable result:** B vs C with explicit task-success deltas. If A ≈ B on substrate tasks, that's the cleanest possible demonstration that current Maxim's claimed substrate-driven behavior survives the LLM swap.

## Cross-cutting concerns

### Tokenization

Pretrained tokenizers (BPE from GPT-2/3/4, SentencePiece, etc.) carry priors. They've already decided that "anti" and "disestablishment" are subword units because of training-corpus statistics. Reusing them in Phase 2 or 3 leaks knowledge sideways and undermines the from-scratch claim. Train tokenizers from the Maxim curriculum corpus only. Start character-level in early Phase 2 prototypes if subword behavior is unclear; move to BPE only once curriculum scale justifies it.

### Catastrophic forgetting across sessions

Online learning + persistent weights = a known disaster mode. Mitigation across all phases:
- **Replay-based consolidation:** hippocampal episodes are already replayed during sleep ([memory_consolidation_practice.md](memory_consolidation_practice.md)). Extend the replay path to feed the binding-layer (Phase 2) and language model (Phase 3) training batches. This is biologically motivated and operationally necessary.
- **Per-session checkpoints with rollback:** if a session destabilizes the model (loss spike, eval collapse), revert. The cost is one sim's worth of learning; the benefit is bounded blast radius.
- **Importance-weighted updates:** linguistic events tied to high-valence outcomes weight more in the consolidation batch. This piggybacks on existing valence machinery.

### Curriculum scheduling

The Roy operator picks the curriculum. Start text-density-low (mom-reading bedtime stories, simple naming) and ramp toward text-density-high (teacher dialogues, peer conversation). The scheduler is operator-facing config in the Roy harness, not a new bio-system. Findings about which curricula are load-bearing accrue to this plan's iteration log over time.

### Compute

Phase 2's small associative model: trains on a CPU during sleep replay. No new infrastructure.
Phase 3's from-scratch transformer: trains on the user's RTX 5080 leader during off-sim periods or as a dedicated training session between Roy iterations. Not real-time. Roy's pacing supports this — sim-time and wall-clock-time are decoupled.

### Pretrained-LLM crutches to disable for Phase 0/1

The runtime ships several mitigations for pretrained-LLM behaviors that don't
exist in a from-scratch substrate. These are appropriate for current Maxim use
but **must be disabled** for Phase 0 (pre-linguistic) and Phase 1 (vocabulary-
constrained), or the experiments measure the crutch instead of the substrate.

| Crutch | What it does | Disable via |
|---|---|---|
| Tool-failure hint section | Adds a `=== Tools You've Hallucinated ===` block to the prompt listing names the agent previously called that don't exist (mitigates qwen-class training-prior fallback to `respond` etc.). Discovered in cradle E1 experiment 2026-05-09 — qwen2.5-14B hallucinated `respond` in 67% of runs even when it wasn't in the prompt. | `MAXIM_TOOL_FAILURE_HINTS=0`. Field on `LLMRequest.failed_tools`; population gated in [agent_loop.py](../../src/maxim/runtime/agent_loop.py) at the `submit_context` call site; section emitter is `build_failed_tools_section` in [prompt_builder.py](../../src/maxim/agents/prompt_builder.py). |

**Add new crutches to this table** as you discover them. The principle is the
same in every case: if a mitigation exists because the pretrained LLM does
something the substrate hasn't earned, the substrate-only experiments must
turn it off so we measure substrate competence and not the mitigation.

The substrate's bio-natural analog of the tool-failure hint exists already —
NAc records `tool:X → failure:not_registered` with negative valence and
biases against re-attempting that tool. With the hint disabled, that NAc
learning becomes the only signal, which is what we want to measure in
Phase 0/1: does the substrate's negative-valence avoidance suppress
hallucinated-tool calls without an explicit prompt warning? If yes, the
crutch was unnecessary all along; if no, we've found a real substrate gap
to address (perhaps via stronger NAc weight on tool-failure outcomes).

### What's deliberately NOT in this plan

- **No new bio-systems.** Every wire-up uses existing bio-systems (EC, ATL, NAc, Hippocampus) and existing buses (PainBus, ReactionBus, percept stream).
- **No new buses or coordinators.** The binding registry, the from-scratch model — both are passive observers / online consumers of existing event streams. Adding a new bus to coordinate them would violate the build_bio_stack invariants from [bio_stack_unification.md](archive/bio_stack_unification.md).
- **Maxim Oasis + Hivemind coupling is intentional but additive.** [maxim_hivemind.md](maxim_hivemind.md) (which superseded the old Mother Maxim plan on 2026-05-09) builds on this plan's substrate-primary work. The Hivemind is the peer-to-peer substrate-sharing layer; the Oasis is the persistent substrate-primary instance that distills LLM-AUT contributions and broadcasts patterns. **Phase -1 and Phase 0 must run with Hivemind disabled** (raw substrate, no bootstrap) so the headline experiment stays clean — the Hivemind is the end-user-convenience path, not the research path.

## File touch summary (estimated)

Phase 0: [simulation/arcs.py](../../src/maxim/simulation/arcs.py), [_data/components/bodies/infant_humanoid.yaml](../../src/maxim/_data/components/bodies/infant_humanoid.yaml), new prompt template under [src/maxim/prompts/](../../src/maxim/prompts/), telemetry extensions to [simulation/orchestrator.py](../../src/maxim/simulation/orchestrator.py).

Phase 1: [models/language/maxim_peer_backend.py](../../src/maxim/models/language/maxim_peer_backend.py) (logit mask hook), [decisions/nac.py](../../src/maxim/decisions/nac.py) or [similarity/ec.py](../../src/maxim/similarity/ec.py) (bound-tokens accessor), new module `src/maxim/language/binding_registry.py`.

Phase 2: new package `src/maxim/language/` — `binding_layer.py`, `tokenizer.py`, `consolidation.py`. New persistence path `~/.maxim/util/grounded_language/`. Hooks into [memory/hippocampus.py](../../src/maxim/memory/hippocampus.py) replay path.

Phase 3: extends `src/maxim/language/` — `from_scratch_lm.py`, training loop, curriculum runner. Likely a new optional extra `pymaxim[grounded-language]` to gate the torch dependency.

Phase 4: comparison harness alongside existing Roy machinery; no new core code.

## Iteration log

(Append findings here as phases run. Living doc convention per [persona_convergence_crucible.md](persona_convergence_crucible.md) and [behavioral_convergence_practice.md](behavioral_convergence_practice.md).)

### 2026-05-09 — Plan promoted from deferred; parallel-mode architecture decided

**Trigger:** E4 controlled validation of `MAXIM_TOOL_FAILURE_HINTS` (negative-instruction tool-failure hint section) — n=6 per arm — showed no benefit and possible backfire on qwen2.5-14B (treatment mean 4.67 respond-hallucinations vs control 3.33; total 37 vs 27; 2× `_llm_unavailable` cascades). Default flipped OFF. The MVP failure was the proximate cause; the deeper cause was a systemic audit:

**Audit findings (three parallel reviews, cross-confirmed):**
1. **LLM band-aid surface ~845 LOC** across the codebase. `json_parser.py` is 32.9% repair logic for malformed LLM output; `prompt_builder.py` is 22.5% mitigation scaffolding; the largest single band-aid is `orchestrator.py::_stall_detector` at **219 LOC** existing solely because the LLM gets stuck in observation loops. 7 distinct LLM failure-mode mitigations have dedicated code.
2. **Recent engineering split: 60-70% LLM mitigation, 30-40% bio-substrate.** The drift is real and accelerating.
3. **Bio-substrate readiness for action selection: 2/10.** NAc predicts and biases but does not propose actions. DN does autonomous gaze but no tool invocation. Reflexes cover 2 thermal cases. There is no `decision_engine.decide()` that does not go through an LLM proposal today. This is a much larger gap than "strip the LLM" implies — hence Phase -1's gating role.

**Strategic decisions made:**
- **Plan moved out of `deferred/`** to active. Status changed from "post-1.0" to "Phase -1 + Phase 0 harness in 1.0; full Phase 0 + 1 + 2 in 1.1+".
- **Parallel-mode architecture:** existing LLM-AUT path stays as the user-facing default. Substrate-primary AUT mode runs in parallel under `--aut-mode substrate-primary` (or similar). LLMs continue to drive orchestrator + NPCs + imagination + LLM-AUT mode; prompt engineering work continues for those layers. Substrate-primary mode is the only path with no LLM in the AUT itself.
- **D&D as the bidirectional kill criterion** (see top of plan). A substrate that can't survive a D&D campaign orchestrated by an LLM-DM is a failed bio-substrate. AND a simulation environment that no learning substrate can navigate is a failed simulation environment. Mutually load-bearing.
- **Phase -1 added** as the most important single experiment in the program — proves (or disproves) that the substrate can generate even ONE non-reflex action without LLM proposal. ~1 week of work; Boolean outcome; gates everything else.
- **Roy harness precondition acknowledged unmet** — designed but unbuilt. Phase 0 validation depends on Roy; Phase 0 HARNESS does not. Harness ships in 1.0 as experimental (`--research` flag); validation waits for Roy in 1.1+.

**1.0 scope add:** Phase -1 + Phase 0 harness (~600-700 LOC across NAc.recommend_action, --aut-mode flag, motor-only AUT prompt template, telemetry snapshots, single-tick test harness). Behind experimental flag; doesn't touch user-facing 1.0 surface; doesn't gate 1.0 docs work.

**Companion change:** `MAXIM_TOOL_FAILURE_HINTS` default flipped to OFF (was ON post-MVP). Validation showed no benefit; opt-in only for further experimentation. Documented in the "Pretrained-LLM crutches" table above.

### 2026-05-09 — Phase -1 GATE CLEARED + Phase 0 harness SHIPPED (PR #228)

Two-session sprint after the 2026-05-09 strategic pivot.

**Phase -1 — substrate action-generation Boolean: PASS.**
- `NAc.recommend_action(agent_id, available_tools, current_drives) -> dict | None` ([decisions/nac.py](../../src/maxim/decisions/nac.py)) scores each tool by causal-link confidence + reward bias + drive-relevance heuristic (substring + affinity table for cold-start). Returns None below threshold — silent IDLE, never random.
- `propose_via_substrate()` ([runtime/agent_loop.py](../../src/maxim/runtime/agent_loop.py)) wraps the recommendation as an `LLMProposal` with `strategy_used="substrate-primary"` and dispatches via the standard executor path.
- `--aut-mode {llm-primary,substrate-primary}` plumbed through `cli` → `start_simulation_mode` → `run_agentic_loop`. In substrate-primary mode the LLM submit branch is gated off; no inference call is ever issued.
- 22 tests across unit + integration. LLMRouter.dispatch tripwire confirms substrate path never touches the LLM.

**Phase 0 — harness shipped:**
- `cradle_prelinguistic` arc + `select_arc_for_goal` exact-name resolution.
- Motor-only AUT prompt renderer.
- `SubstrateTelemetry` JSONL writer wired into the substrate-primary tick.
- `--research` semantic split: with `--aut-mode substrate-primary` it means "telemetry on" (not the multi-agent paper harness).

**Smoke run cleared the success criterion:**
```
maxim --sim cradle_prelinguistic --embodiment bodies/infant_humanoid \
      --aut-mode substrate-primary --research --interactive false \
      --sim-max-turns 5
```
38 actions, 61 causal links, hunger drift 0.0 → 0.65, 195 telemetry rows. Substrate proposed `sense_food_source` on the cold-start drive-affinity heuristic (food substring matched hunger affinity).

**Surprising findings (full write-up: [13_phase0_harness_smoke.md](../experiments/13_phase0_harness_smoke.md)):**
1. **Substrate-primary mode owns its own clock.** First smoke run produced 0 actions because nothing was polling the embodiment — the LLM-primary path drives the embodiment via `EmbodimentPerceptSource.next_percept()` → `evaluate_failures()`, but substrate-primary skips that. Fixed by calling `evaluate_failures()` inside `propose_via_substrate`. Documented as a structural property: anything the LLM-primary path implicitly relied on (drive drift, percept polling) needs to be re-wired for substrate-primary.
2. **Substrate-primary bypasses `LinguisticEncoder`.** EC `node_count` stayed at 0 throughout the smoke run because substrate-primary doesn't feed text percepts through the encoding path. Phase 0's cluster-formation measurement is blocked on a sensor-percept encoding entry point — next work item.
3. **Drive-affinity heuristic + scene tools.** The infant body's `pick_up` lost to the cradle scene's `sense_food_source` because the affinity table matches "food" substrings. Same mechanism as the test suite caught with `pick_up_food`, just a different tool name. The heuristic is a Phase -1 placeholder; the full plan calls for replacement with EC embedding similarity once Phase 0 sensor-encoding lands.
4. **Pre-existing AttributeError.** End-of-run warning `'GenerativeCampaignResult' object has no attribute 'turns_completed'` surfaces on every termination path. Pre-existing bug in the generative campaign runner; flagged for future cleanup.

**1.0 implication:** B5's first ~700 LOC ships as planned. Hivemind shareability (~660 LOC) remains pending. The next concrete substrate-primary work item is the EC sensor-encoding entry point (small) — that's the cheapest experiment that converts the harness into an actual Phase 0 measurement.

---

## Honest recap

The user's instinct (replace pretrained LLM with from-scratch microGPT trained via NAc) was right in motivation, wrong in shape. The motivation is real: current Maxim's "substrate carries cognition" thesis has an asterisk because the LLM smuggles in conceptual priors. The shape is wrong because reward-only training of randomly-initialized transformers fails, autoregressive prediction doesn't ground, and the cheapest tests of the thesis don't require rebuilding the language model at all.

This plan addresses the motivation by sequencing four experiments of increasing cost. Phase 0 (pre-linguistic cradle) and Phase 1 (vocabulary constraint) test the thesis cheaply and ship findings regardless of later phases. Phase 2 (symbol binding) and Phase 3 (from-scratch sequence model) build the headline experiment, but only after gates clear. The Roy long-horizon context plus persistent substrate plus text-heavy curriculum makes Phase 3's data scale plausible — that part of my conversational pushback was wrong.

The hardest unsolved problem in the plan is **catastrophic forgetting across sessions**. Replay-based consolidation is the bet; it might not be enough. If Phase 2 fails primarily on forgetting rather than binding accuracy, the program fork is "do we invest in better consolidation or accept that this model needs offline batch training between Roys." Either fork is a real research finding worth publishing.

The program might not finish. That's fine — the framing rule says each phase justifies itself.
