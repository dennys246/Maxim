# Passive sense-tool discovery (ambient discovery channel)

> **⏸ DEFERRED 2026-08-27 — the only artifact is this doc (#406); Phase 0 never authorized; zero code; no roadmap item. **Revive when** an experiment is blocked by the cost of spending an action on `sense_tools`, or `sense_tool_registry.md`'s 1.1+ half is re-prioritized.**

**Status:** DRAFT — design pass pending. Written up from the 2026-07-19 deep-dive session.
**Trigger:** operator proposal — make tool discovery asynchronous with the agentic loop: a passive producer discovers/activates tools continuously and the loop grabs "what's new since my last turn" instead of the agent spending an action on `sense_tools`.
**Target:** 1.1 candidate. Not a 1.0 gate.
**Relation to prior plans:** this REVIVES the deferred 1.1+ half of [sense_tool_registry.md](sense_tool_registry.md) (`sensory_events.jsonl` split, LRU wiring/tuning, NAc predicate-outcome typing) — its revive trigger ("a substrate→action conversion experiment re-prioritizes tool visibility") is satisfied by operator authorization of this draft. Complementary to [imagination_substrate_signals.md](imagination_substrate_signals.md) (substrate dreams missing entities INTO scene; this plan surfaces what's already in scene without an action).

## One-paragraph proposal

Split sensing into two biologically distinct channels. An **ambient channel** (thalamic relay): a gated background discovery pass runs alongside the loop, activates salient scene tools through the existing registry, and surfaces a *delta* — "newly sensed since your last turn" — into the prompt's dynamic segment, the way §1.15 auto-sense and §1.16 audio orientation already surface passive perception. A **directed channel** (attention): `sense_tools(query)` survives unchanged as an LLM-chosen action, so the substrate keeps an operant-sensing surface it can learn over (the Exp 45 orienting line depends on sensing-as-action remaining learnable). Passive does not replace active; it removes the obligation to spend an action just to see.

## Current behavior (deep-dive findings, 2026-07-19)

How discovery works today, plus four pre-existing defects this plan either fixes or must not worsen:

1. **`sense_tools` is a synchronous chosen action** ([discovery.py:278-550](../../../src/maxim/tools/discovery.py)): keyword overlap over self-entity affordances, ComponentIndex semantic fallback when keyword score < 0.3, NAc-ranked top-8, then `registry.activate_scene(scene_id)` mid-turn ([discovery.py:406](../../../src/maxim/tools/discovery.py)). The enlarged roster reaches the next prompt via `mode_info.get_available_tools` ([agent_loop.py:3465](../../../src/maxim/runtime/agent_loop.py)).
2. **A passive channel already exists for presence, not discovery.** §1.15 auto-sense iterates `registry.get_auto_fire_tools()` per tick on new percepts and folds results into `_auto_sense_text` → `context.auto_sense_context`, bypassing `actions.jsonl` by design ([agent_loop.py:1370-1492](../../../src/maxim/runtime/agent_loop.py)). W1 grayscale visibility annotates inactive-but-substrate-known tools ([agent_loop.py:3178-3239](../../../src/maxim/runtime/agent_loop.py)).
3. **PRE-EXISTING HOLE — roster churn silently breaks the prompt cache.** The tools section is `cacheable=True` in autonomous modes ([prompt_builder.py:1065](../../../src/maxim/agents/prompt_builder.py)) on the *assumption* that the roster only changes at narrative-phase boundaries; the comment at [prompt_builder.py:1056-1064](../../../src/maxim/agents/prompt_builder.py) explicitly flags sense_tools/imagination churn as an unmitigated cache-miss risk. There is no deferral mechanism: a mid-phase `sense_tools` activation changes the stable prefix bytes on the next turn and re-spends the full cached prefix. Neither caching test covers this — [test_prompt_caching.py](../../../tests/integration/test_prompt_caching.py) and [test_prompt_builder_audit.py](../../../tests/integration/test_prompt_builder_audit.py) both hold `available_tools` constant across turns.
4. **PRE-EXISTING DEAD CODE — the discovery LRU never runs.** `evict_stale_discoveries` and `mark_tool_used` ([discovery.py:43-79](../../../src/maxim/tools/discovery.py)) have zero callers in `src/`; the documented "deactivate after 5 unused turns" behavior is inert. Only `mark_goal_selected` (orchestrator) and `reset_discovery_state` (cli) are live. The module-global LRU state (`_tool_last_used`, `_goal_selected`) is also unsynchronized — fine single-threaded, not with a background writer.
5. **PRE-EXISTING CREDIT NOISE — `sense_tools` self-reinforces in NAc.** Every successful call books a POSITIVE `tool:sense_tools` causal link via [tool_dispatch.py:155](../../../src/maxim/runtime/tool_dispatch.py) (plus the executor's `tool_pain_bridge.record_tool_complete`), with no discovery-tool exclusion. A positive-gated `recommend_action` can latch onto this.
6. **Cap eviction is eager and unaware of in-flight turns.** `activate_scene` under `DEFAULT_ACTIVE_TOOL_CAP = 20` evicts the oldest active scene ([registry.py:176-215](../../../src/maxim/tools/registry.py)). The executor's active-tool gate ([executor.py:206-215](../../../src/maxim/runtime/executor.py)) rejects calls to deactivated tools, increments `_consecutive_failures`, and **reports a failure to the pain detector**.

## Front-gate scope pressure (CLAUDE.md Principle 3)

*Does this need to be its own mechanism, or can it ride on existing infrastructure?*

| Candidate infrastructure | Verdict |
|---|---|
| `Tool.auto_fire` + §1.15 auto-sense dispatch ([sense_tool_registry.md](sense_tool_registry.md) MVP, PR #287) | **Ride.** The passive discovery pass is dispatched as (or exactly like) an auto-fire tool: per-tick, actions.jsonl-bypassing, output folded into the passive-perception prompt path. |
| PR #402 thalamic side-channel + §1.16 pattern (`sim.current_percept`, read-latest, change-gate, `_escalate_this_tick`) | **Ride.** The loop-side consumer copies §1.16: read-latest, `should_emit_orientation`-style change gate so unchanged rosters aren't re-announced, escalation flag for high-salience discoveries. |
| Hippocampus capture worker ([hippocampus.py:836](../../../src/maxim/memory/hippocampus.py)) — daemon thread + bounded `queue.Queue`, drop-oldest, `ThreadRegistry` | **Ride** (Phase 2 only). The template for a background thread if/when embedding-based discovery needs to leave the tick path. |
| `ImaginationTrigger` gate stack (cheap-first ordering; DN `imagination_allowed()` arousal gate; energy budget; per-key concurrency guard) ([trigger.py:706-748](../../../src/maxim/imagination/trigger.py)) | **Ride.** Same gates, same ordering, for "should I do discovery work this tick?" |
| `AdaptiveThresholdController` ([gating.py:156](../../../src/maxim/runtime/gating.py)) | **Ride.** Already tunes escalate-vs-fold thresholds against processing load. |
| `ComponentIndex.find_alias_only` ([component_index.py:229](../../../src/maxim/embodiment/component_index.py)) | **Ride.** The per-tick path uses alias + keyword only; `_embed()` (~5 ms, shared sentence-transformers singleton, contends with the GPU lane) is background-thread-only. |
| `ToolRegistry` RLock + activation machinery ([registry.py:70](../../../src/maxim/tools/registry.py)) | **Ride.** Cross-thread activation is already safe; what's missing is turn-coherence (see contract piece 3). |
| `PerceptTraceBuffer` watermark (`TraceEntry.tick` + `current_tick`) ([percept_trace_buffer.py](../../../src/maxim/memory/percept_trace_buffer.py)) | **Pattern to copy** for "what's new since my last read" — a watermark, not a per-consumer drain cursor. |

**Verdict: ride existing infrastructure for the entire producer/consumer machinery. Three genuinely new contract pieces are required** (below) — none is derivable from existing surfaces, and each closes a defect that already exists today:

1. a prompt-side **delta/manifest split** (no current mechanism defers roster changes to phase boundaries);
2. a **sensory-event vs causal-action typing** in NAc attribution (no current discriminator — this is precisely the deferred `sense_tool_registry.md` 1.1 scope);
3. a **turn-scoped roster snapshot** (no current mechanism makes prompt-build and executor-gate agree on a roster).

## The three new contract pieces

### 1. Prompt: delta in the dynamic segment; manifest folds at phase boundaries

The byte-stability invariant (CLAUDE.md, prompt-caching entry) stays intact by splitting the tools surface:

- **Stable manifest** (`cacheable=True`, unchanged section): re-rendered only at narrative-phase/act entry — the model the cache tag already assumes ([generative_runner.py:425-444](../../../src/maxim/simulation/generative_runner.py) is the sanctioned mutation point).
- **Discovery delta** (new dynamic section, rendered like `auto_sense_context`): "Newly available since your last turn: `<tool> — <description> [NAc annotation]`". Tools listed here are immediately callable — the executor gate reads the live registry, not the prompt — they just don't perturb the cached prefix until the next phase boundary folds them into the manifest.

This converts the async change from a cache-killer into a cache-**fix**: it also closes pre-existing hole #3 for the existing synchronous `sense_tools` path (its activations ride the same delta section instead of mutating the manifest mid-phase).

**Consequence for tests:** both prompt-caching byte-stability tests gain a roster-churn arm — activate tools mid-run, assert the stable prefix hash is *still* constant and the delta appears in the dynamic segment. Today they would pass while the cache silently breaks.

### 2. NAc: sensory events are predictive, never causal

Load-bearing invariant inherited from [sense_tool_registry.md](sense_tool_registry.md) invariant #1: **auto-fired sensing must never write to `actions.jsonl` or create causal action links** — phantom links corrupt NAc's causal model. Passive discovery output is perception, so:

- Discovery events route to `sensory_events.jsonl` (new event-type contract), not `actions.jsonl`.
- NAc attribution for passive discoveries is typed `predicate_outcome` ("in context X, tool Y tends to exist" — predictive), distinct from `tool_outcome` (causal, "I chose Z and Y happened"). This is the deferred plan's Phase 3, promoted from optional to correctness-backbone.
- The chosen-action `sense_tools` path keeps causal credit — but the pre-existing +1 self-credit (finding #5) should be re-examined in the same design pass: a discovery meta-tool booking the same `tool_outcome` credit as a physical affordance is at minimum a known confound for positive-gated `recommend_action`, and this plan's delta section reduces how often the LLM needs to call it at all.

### 3. Executor: turn-scoped roster coherence

Without it, the async producer introduces a TOCTOU race: prompt lists tool X → background activation under the 20-cap evicts X's scene → LLM calls X → executor gate rejects, increments `_consecutive_failures`, **feeds a phantom failure to the pain detector** — a negative outcome the agent did not cause, exactly the mis-attribution class B8 delta-attribution exists to kill.

Design options for the pass (pick one in Phase 0):
- **(a) Turn-scoped snapshot:** prompt build takes a roster snapshot; the executor gate honors `snapshot ∪ live-active` for that turn.
- **(b) Eviction grace:** `deactivate` from the async producer marks tools *evicting*; the gate accepts calls to evicting tools for one turn, then hard-deactivates.

Either way, the invariant is: **a tool the current prompt presented as callable must not fail the active-gate within that turn because of async churn.** (Agent-caused deactivation — e.g. `drop` releasing an entity — is exempt; that failure is genuinely informative.)

Supporting fixes in the same layer: wire the dead LRU eviction (finding #4) *inside* the producer so activation and eviction are one coherent policy rather than two racing ones, and put a lock around the module-global LRU state in `discovery.py` (or move it onto the producer object, which is cleaner than blessing module globals).

## Producer design (ride-along, no new mechanism class)

- **Phase 1 (cheap, in-loop):** the discovery pass runs in the §1.15 auto-fire slot — per tick, gated on (new percept) ∧ (change gate) ∧ (arousal) ∧ (energy) ∧ (adaptive threshold). Matching is keyword + `find_alias_only` ONLY. Output: registry activations (batched through contract piece 3) + the delta section + a `sensory_events.jsonl` record.
- **Phase 2 (background thread, only if Phase 1 telemetry shows alias-misses matter):** hippocampus-capture-worker pattern — daemon thread, bounded queue, drop-oldest, `ThreadRegistry`-registered, shut down at session end. Embedding-based matching (`find` / `find_similar`) lives here and nowhere else; results flow back through the same delta/watermark seam. Per-key concurrency guard copied from `ImaginationTrigger._designing`.
- **Escalation:** a high-salience discovery (goal-relevant per `select_goal_relevant_tools` scoring, or drive-relevant per `GatingContext.drive_states`) sets an `_escalate_this_tick`-style flag (§1.16's B1-fix pattern) so a discovery-only tick isn't dropped by the `has_meaningful_input` gate. Everything else folds silently.
- **Anti-bloat:** the cap, goal-top-k, and the revived LRU are the pressure valves. The delta section is additionally capped (top-N by salience, N≈5, matching W1 grayscale's `top_n`); overflow is logged, not silently dropped (no-silent-caps principle).

## Division of labor across the three visibility surfaces

| Surface | Covers | Mechanism |
|---|---|---|
| Passive delta (this plan) | Present in scene, salient now, newly available | Activation + dynamic prompt section |
| Directed `sense_tools` | Agent-driven query, operant sensing | Unchanged chosen action |
| W1 grayscale | Absent but substrate-remembered | Annotation only, no activation |

The Phase 0 design pass must check the three don't render conflicting descriptions of the same tool in one prompt (W1's Bio-C2 lesson: shared band naming via `bias_to_band`).

## Load-bearing invariants (DO NOT BREAK)

1. **Auto-sense / passive discovery never writes `actions.jsonl`** ([sense_tool_registry.md](sense_tool_registry.md) invariant #1). The delta channel is perception.
2. **System-prompt byte-stability within a session** (CLAUDE.md prompt-caching invariant). The stable manifest changes only at phase boundaries; everything else rides the dynamic segment.
3. **Scene-scoping at invocation is preserved** — grayscale/delta visibility never makes an absent-entity tool invokable.
4. **LRU eviction applies to scene tools only, never core tools** ([discovery.py:67](../../../src/maxim/tools/discovery.py) exemption survives the rewiring).
5. **`ModulatorAffordanceTool` sensor-delta feedback survives untouched** — discovery changes visibility/activation, never the affordance execution path (cerebellum forward model + pain bridge + `drive_potential_diff` motor credit).
6. **Directed sensing stays learnable.** `sense_tools` remains a chosen action with NAc credit so the substrate keeps an operant-sensing policy surface (Exp 45 orienting lineage).
7. **Non-blocking loop.** Nothing on the tick path blocks on embedding or I/O; Phase 2 work is thread-isolated per the capture-worker template.

## Phasing

- **Phase 0 — design pass** (two-lens review before any code): pick snapshot-vs-grace for contract piece 3; specify the `sensory_events.jsonl` schema + `predicate_outcome` NAc typing against current `NAc.observe`; specify the delta section renderer + manifest fold point; decide the `tool:sense_tools` self-credit question; LOC estimate.
- **Phase 1 — prompt split + turn coherence** (no new threads): delta section, phase-boundary manifest fold, roster snapshot/grace, caching-test churn arms. This alone fixes pre-existing hole #3 and race #6 and is independently shippable.
- **Phase 2 — passive producer, in-loop**: gated auto-fire discovery pass, LRU rewire + lock, `sensory_events.jsonl`, telemetry (`sim_discovery` events for every activation/eviction/fold).
- **Phase 3 — NAc predicate-outcome typing**: predictive channel wired; regression: causal `tool_outcome` distribution unchanged on existing fixtures.
- **Phase 4 — background thread** (conditional on Phase 2 telemetry): embedding matching off-tick.
- **Phase 5 — behavioral validation**: re-run the Roy-3a-lineage visibility experiment (does the delta convert substrate annotation to action where grayscale alone did not — [33_wire_a_post_fix_a.md](../../experiments/33_wire_a_post_fix_a.md) Fix-B territory) + an Exp 44-style A/B on action-economy (turns previously spent on `sense_tools` reallocated to affordance exploration).

## What this does NOT solve

- Imagination substrate-blindness ([imagination_substrate_signals.md](imagination_substrate_signals.md)) — the delta surfaces what exists; it doesn't dream missing entities into scene.
- Substrate-primary cold-start: passively activated tools arrive with zero bias and the positive-gated `recommend_action` won't select unknowns — that's the orient/motor-credit line (`drive_potential_diff`), not this plan.
- The narrator/Layer-2 non-determinism in LLM-primary embodiment (Exp 44 G1 territory).

## Open questions (for Phase 0)

1. Snapshot vs grace-period for turn coherence — snapshot is cleaner but touches the prompt-build path; grace is smaller but adds registry state.
2. Should the delta section list *deactivations* too ("no longer available: …"), or does silence + the executor's existing "not active" error suffice? (Bio prior: sensory fade is usually unannounced.)
3. Does `predicate_outcome` need a schema-version bump on persisted NAc snapshots, or is a new outcome type additive? (Same question deferred plan asked; verify against `_format_version` contract.)
4. Keep or gate the `tool:sense_tools` causal self-credit once the passive channel exists?
5. Substrate-primary: should `propose_via_substrate` consume the same delta watermark (e.g., as candidate-set expansion events) or continue reading `registry.list()` live? Live-read has no cache constraint, so the default is "leave it alone."

## Authorization gate

Drafted 2026-07-19 from the deep-dive session; operator requested the write-up. Phase 0 design pass starts on explicit authorization. Reviving the deferred [sense_tool_registry.md](sense_tool_registry.md) 1.1 scope is part of that authorization (its revive trigger is satisfied by this plan's Phase 3 dependency on predicate-outcome typing).
