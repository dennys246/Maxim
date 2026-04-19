# Maxim Plans

Current version: **0.3.2** (published on PyPI as `pymaxim`).
Target: **1.0** — cross-session learning demonstrated without LLM fine-tuning.

## Active (gating 1.0)

Foundations, reaction abstraction, simulator upgrades, P0 pilot, P1 recognition, P2 reward modulation, P3a episode binding, P3b channel integration, P3.5 persistence/snapshot, SEM execution hook, and **P4 cross-modal binding** are all complete. **All four 0.3-target substrate phases (P3a, P3b, P3.5, P4) are CLOSED. 0.3-target gate is CLOSED.**

- [substrate_binding_persistence.md](archive/substrate_binding_persistence.md) — **✅ SPLIT COMPLETE + ARCHIVED** (2026-04-17). Now a pure index. **All four 0.3-target phases CLOSED.** Per-phase plan files created for 0.5 track:
  - [substrate_p5_stress_persistence.md](substrate_p5_stress_persistence.md) — Draft. 10k+ node persistence stress. Depends on P3.5 + P4.
  - [substrate_p6_extinction.md](substrate_p6_extinction.md) — Draft. Decay without reinforcement vs LRU. Depends on P3a.
  - [substrate_p8_sleep_replay.md](substrate_p8_sleep_replay.md) — Draft. Minimum-viable sleep replay. Depends on P3a + P6.
  - [prompt_b3_b5_track.md](prompt_b3_b5_track.md) — Draft. Acting Coach + embodiment/narrative separation.
  - [prompt_b4_replanning.md](prompt_b4_replanning.md) — Draft. **1.0-GATING.** Replanning with failure diagnosis.
- [substrate_binding_split_proposal.md](archive/substrate_binding_split_proposal.md) — **✅ APPROVED + EXECUTED + ARCHIVED** (2026-04-17). The narrative that motivated the split.
- [archive/substrate_p0_pilot.md](archive/substrate_p0_pilot.md) — **✅ COMPLETE + ARCHIVED** (2026-04-12). Baseline pinned at 78.5%. Results: [experiments/p0_baseline_sweep.md](../experiments/p0_baseline_sweep.md).
- [archive/substrate_recognition.md](archive/substrate_recognition.md) — **✅ COMPLETE + ARCHIVED** (2026-04-14). P1+P2 all stages shipped. 0.3-minimum gate CLOSED.
- [archive/substrate_p3a_episode_binding.md](archive/substrate_p3a_episode_binding.md) — **✅ COMPLETE + ARCHIVED** (2026-04-14). Hebbian multi-hop F1=0.9955 vs TF-IDF 0.6600.
- [archive/substrate_p3b_channel_integration.md](archive/substrate_p3b_channel_integration.md) — **✅ Stage 1 SHIPPED + ARCHIVED** (2026-04-14). Stages 2+3 deferred — not version-gating.
- [archive/substrate_p3_5_persistence_snapshot.md](archive/substrate_p3_5_persistence_snapshot.md) — **✅ Stages 1+2 SHIPPED + ARCHIVED** (2026-04-14). Stage 3 deferred — not version-gating.
- [archive/substrate_p4_cross_modal_binding.md](archive/substrate_p4_cross_modal_binding.md) — **✅ COMPLETE + ARCHIVED** (2026-04-16). Stage 3 PASS: Arm B F1=1.000 vs Arm C F1=0.901, delta +0.099. Results: [experiments/p4_cross_modal_sweep.md](../experiments/p4_cross_modal_sweep.md).
- [archive/substrate_p4_option2_measurement.md](archive/substrate_p4_option2_measurement.md) — **✅ COMPLETE + ARCHIVED** (2026-04-16). Option 2 lift=0. Decision: defer.
- [archive/sem_execution_hook.md](archive/sem_execution_hook.md) — **✅ COMPLETE + ARCHIVED** (PRs #107, #110, #119, 2026-04-14). All four stages shipped. Stage 2c was structurally absorbed by [archive/executor_bootstrap_unification.md](archive/executor_bootstrap_unification.md). Stage 2b was deferred to [agent_factory_canonicalization.md](agent_factory_canonicalization.md) Stage F1+. The build_executor structural-enforcement pattern became the canonical example for [structural enforcement](../architecture/structural_enforcement.md).

The master reference for rationale, baselines, and statistical hygiene is archived at [archive/substrate_plan.md](archive/substrate_plan.md).

## Living practice docs (pair with substrate phases)

These accumulate evidence and refinement over time. They are not on the critical path to 1.0; they exist because the questions they address are scientific/ongoing, not engineering milestones.

- [behavioral_convergence_practice.md](behavioral_convergence_practice.md) — does the agent actually get better across sessions? **4 experiments logged, all 3 tiers PASS** (Exp 1: cross-session affective memory 11/11, Exp 2: energy consumable learning 13/13, Exp 3: LLM acts on bio-system learning 12/12, Exp 4: organic LLM learning 5/5 — teal rate 0% -> 25% -> 100%). 41/41 hypotheses confirmed. Pure living doc — no mechanism to ship, just hypotheses, scenarios, and results.
- [memory_consolidation_practice.md](memory_consolidation_practice.md) — refines the P8 sleep-replay mechanism with alternative strategies, promotion rules, interference analysis. Kicks in when P8 ships in 0.5 — needs the mechanism to exist before the practice has anything to refine.

## Parallel (ship anytime, not gating 1.0)

- [interactive_experience_031.md](interactive_experience_031.md) — **✅ SHIPPED** (2026-04-18, PR #156). Interactive UX fixes for 0.3.1: `RequestInteractionTool` honest reporting, narrator fallback immersion, handler logging, story context truncation, `MaximDisplay` → `sim_logger` wiring, prompt cleanup, 4 introspection tools. 8 stages, ~700 LOC. Known issue: display `print()` corruption → [docs/bugs/display_print_corruption.md](../bugs/display_print_corruption.md) for 0.3.2.
- [substrate_concept_decomposition.md](substrate_concept_decomposition.md) — **Stage 1 COMPLETE + VALIDATED** (2026-04-17). Protocol-based noun-phrase extraction. 100% concept-level recall vs 36.4% baseline. Stage 2 (role-tagged edges) pending.
- [substrate_episode_boundary_enrichment.md](substrate_episode_boundary_enrichment.md) — **PARTIAL** (2026-04-17). Stage 3 (pain/salience spike) SHIPPED via sem_learning_loop.md. `observe_episode_event` now wired into production agent loop via behavioral_convergence_wiring.md. Stages 1-2 (tool execution + semantic shift) remain — ship before P5.
- [biosystem_unification.md](biosystem_unification.md) — **central tracking doc** (2026-04-14, updated 2026-04-17). Waves 0-3 **ALL SHIPPED + ARCHIVED**. Wave 4 (agent_factory_canonicalization) not scheduled.
- [tool_refinement_plan.md](tool_refinement_plan.md) — living doc for agent tool surface curation
- [agent_factory_canonicalization.md](agent_factory_canonicalization.md) — **RUNNING DOC, trigger #4 activated** (2026-04-18). The Option D follow-up to `executor_bootstrap_unification.md` — make `AgentFactory.create_agent` the only door for constructing an agent in Maxim. **Wave G (Game/External Host) folded in** from game_npc_integration.md: wire Executor + bio-pipeline into `AgentPool.run_turn()`, HostContext protocol, async tool dispatch, emotional state readout, memory backend. F-wave ~1500-2500 LOC + G-wave ~940 LOC. Target: 0.5.
- [node_security_simplification.md](node_security_simplification.md) — Phase 1 ✅ SHIPPED. Phase 2 config-surface unification deferred.
- [reactive_peer_mesh_roadmap.md](reactive_peer_mesh_roadmap.md) — living roadmap for the full reactive peer mesh arc (C3→C9). C3-C4.6 COMPLETE. C5+ remain.
- [cross_platform_file_lock.md](cross_platform_file_lock.md) — shell plan to unify `utils/process_lock` and `filelock.FileLock`. Blocks nothing.
- [mesh_doc_transport.md](mesh_doc_transport.md) — shell plan for mesh-to-mesh structured doc exchange (C9). Not started.
- [pain_bus_bridge_subscriber_unification.md](pain_bus_bridge_subscriber_unification.md) — shell plan for bridge×subscriber attribution-asymmetry fix. Not started.
- [llm_path_refinement.md](llm_path_refinement.md) — meta-plan for the LLM routing path refactor. Plans 1-3.5 archived; Plan 3.6 R5 shipped; **Plan 4 Stages A+B + C1-C3.6 + C4+C4.5+C4.6 ALL SHIPPED.** Reactive mesh self-healing loop complete. Only stress phases B/C/E remain in scope. Architecture ref: [../architecture/llm_routing.md](../architecture/llm_routing.md).
  - [llm_path_peer_failover.md](llm_path_peer_failover.md) — Plan 3.6 R5 (VRAM spillover) ✅ SHIPPED. R1-R4 (multi-leader) remain draft, on hold until second GPU.
  - [llm_path_operator_visibility.md](llm_path_operator_visibility.md) — Plan 4. **Core stages ALL SHIPPED** (A, B, C1-C3.6, C4, C4.5, C4.6). Remaining deferred scope (admin API, rate limiting, key rotation) tracked in [reactive_peer_mesh_roadmap.md](reactive_peer_mesh_roadmap.md) as C6/C7.
  - Deferred: [deferred/llm_path_multi_peer_dispatch.md](deferred/llm_path_multi_peer_dispatch.md), [deferred/llm_mesh_capability_aware.md](deferred/llm_mesh_capability_aware.md), [deferred/llm_path_async_router.md](deferred/llm_path_async_router.md), [deferred/llm_path_fair_scheduling.md](deferred/llm_path_fair_scheduling.md)

  **Long-term mesh roadmap**: [reactive_peer_mesh_roadmap.md](reactive_peer_mesh_roadmap.md). Stages C3-C4.6 complete; C5 (capacity-aware routing), C6 (admin API), C7 (security hardening) remain.

## Deferred (post-1.0, revive on trigger)

Design work is preserved in [deferred/](deferred/). Each plan has an explicit "revive when" condition at the top.

- [deferred/bio_system_plugin_plan.md](deferred/bio_system_plugin_plan.md) — plugin discovery for bio-systems (extends the `maxim.robots` pattern). Depends on the `BioSystem` Protocol landing incrementally during substrate work. Revive when external contributors want to add bio-systems or a research collaborator needs substrate A/B testing.
- [deferred/mother_npc_stimulus_plan.md](deferred/mother_npc_stimulus_plan.md) — **needs heavy refinement.** Two-agent stimulus pattern: Baby Maxim is the AUT with frozen LLM and evolving substrate; Mother NPC is a separate agent with her own LLM that produces realistic, varied percepts Baby learns from. Interaction is percepts only, zero information leak beyond that surface. Gives behavioral convergence experiments scalable stimulus variety without breaking the "no fine-tuning" research claim. Revive when [behavioral_convergence_practice.md](behavioral_convergence_practice.md) has ≥2 successful experiments + 1 blocked-on-variety. Isolation leak vector list in the plan is a starting point, not a contract — heavy refinement needed at revive time.
- [deferred/pecking_order_graph_plan.md](deferred/pecking_order_graph_plan.md) — unified hierarchy DAG
- [deferred/mother_maxim_plan.md](deferred/mother_maxim_plan.md) — persistent collective memory
- [deferred/asset_foundry_plan.md](deferred/asset_foundry_plan.md) — automated SEM component generation
- [deferred/dungeon_master_extensions.md](deferred/dungeon_master_extensions.md) — DM post-MVP features

## Archive

Completed or superseded plans live in [archive/](archive/).

Recently archived (2026-04-17):
- [archive/sem_learning_loop.md](archive/sem_learning_loop.md) — **✅ COMPLETE + ARCHIVED** (2026-04-17). All 5 stages shipped. Complete SEM → bio-pipeline learning loop.
- [archive/behavioral_convergence_wiring.md](archive/behavioral_convergence_wiring.md) — **✅ COMPLETE + ARCHIVED** (2026-04-17). All 4 stages shipped. Valence in PromptAssembler, `observe_episode_event` in agent loop.
- [archive/cerebellum_activation.md](archive/cerebellum_activation.md) — **✅ COMPLETE + ARCHIVED** (2026-04-17). Absorbed into sem_learning_loop.md.
- [archive/substrate_valence_annotation.md](archive/substrate_valence_annotation.md) — **✅ COMPLETE + ARCHIVED** (2026-04-17). Stages 1-3 shipped. Stage 4 absorbed into sem_learning_loop.md.
- [archive/substrate_binding_persistence.md](archive/substrate_binding_persistence.md) — **✅ SPLIT COMPLETE + ARCHIVED** (2026-04-17). Pure index. All four 0.3-target phases CLOSED.
- [archive/substrate_binding_split_proposal.md](archive/substrate_binding_split_proposal.md) — **✅ APPROVED + EXECUTED + ARCHIVED** (2026-04-17). The narrative that motivated the split.
- [archive/bio_stack_unification.md](archive/bio_stack_unification.md) — **✅ Wave 3 SHIPPED + ARCHIVED** (PR #140). `build_bio_stack(*, persistence_dir)`. 4 sites migrated.
- [archive/router_drain_coupling.md](archive/router_drain_coupling.md) — **✅ COMPLETE + ARCHIVED** (2026-04-17). Router drain constraint wiring.
- [archive/auto_drain_persistent_failure.md](archive/auto_drain_persistent_failure.md) — **✅ COMPLETE + ARCHIVED** (2026-04-17). Type-aware auto-drain thresholds.

Previously archived (biosystem unification Waves 0-2 shipped 2026-04-14/16):
- [archive/executor_bootstrap_unification.md](archive/executor_bootstrap_unification.md) — **Wave 0** (PR #114, 2026-04-14). `build_executor(*, pain_bus)` required keyword arg. The canonical structural-enforcement example.
- [archive/pain_bus_unification.md](archive/pain_bus_unification.md) — **Wave 1** (PR #125, 2026-04-15). `build_pain_bus(*, hippocampus, nac)`. 3 CLI site migrations.
- [archive/reaction_bus_unification.md](archive/reaction_bus_unification.md) — **Wave 1** (PR #134, 2026-04-16). `build_reaction_bus(*)`. Cerebellum factory fix.
- [archive/memory_hub_unification.md](archive/memory_hub_unification.md) — **Wave 2** (PR #136, 2026-04-16). `build_memory_hub(*, hippocampus, scn, nac, ec)` always calls `.connect()`. 5 site migrations.
- [archive/default_network_unification.md](archive/default_network_unification.md) — **Wave 2** (PR #135, 2026-04-16). `build_default_network(*, nac)` + `pain_bus=` injection.

Previously archived (LLM path refinement Plans 1–3.5 shipped 2026-04-12/13):
- [archive/llm_path_foundation.md](archive/llm_path_foundation.md) — **Plan 1** (R0 + R1). R0 deleted ~1,250 LOC dead mesh scaffolding (commit `e811787`). R1 shipped `maxim/utils/http.py` with endpoint registry, typed `HTTPError` hierarchy, `RequestContext` + contextvars, `X-Maxim-*` header propagation. PRs #88, #90, #91.
- [archive/llm_path_typed_errors.md](archive/llm_path_typed_errors.md) — **Plan 2** (R2a–d). Role detection at CLI boot, typed `BackendError` hierarchy with `.fix_hint`, two-stage probe, SSRF moved to `utils/net.py`. PRs #92, #93.
- [archive/llm_path_fast_failover.md](archive/llm_path_fast_failover.md) — **Plan 3** (R2.5 + R2.6). `_MaximPeerBackend` purpose-built single-HTTP-call backend replaces `_OpenAIBackend` for self-hosted peers. Typed router dispatch with per-class backoff. Probe consolidation. **The 52s fail-slow is dead.** PR #94 (`ce5f034`).
- [archive/llm_path_cancellation_hygiene.md](archive/llm_path_cancellation_hygiene.md) — **Plan 3.5** (R1–R6). Cooperative cancellation primitives + "HTTP fires first" timeout contract (HTTP authoritative at 300s, agent layer strict safety net). PR #96 (`6a4f505`).

Earlier archives (2026-04-11/12, S1–S4 shipped 2026-04-12):
- [archive/foundations_plan.md](archive/foundations_plan.md) — F0.1–F0.8 all landed. NAc save/load, NarrativeModulator ghost removal, PerceptContext schema, agent_id threading + SCN race fix, PerceptTraceBuffer, tier assertions, SensoryTag population, Percept factory consolidation.
- [archive/cleanup_wave.md](archive/cleanup_wave.md) — C1–C4 shipped in 0.2.2. `--interactive` fix, dead CLI flags, display defaults, agent permissions.
- [archive/peer_leader_flexibility_plan.md](archive/peer_leader_flexibility_plan.md) — P1–P9 shipped. Dynamic n_ctx, KV quant, Apple Silicon tiers, auto-download, remote probes, lane decision log.
- [archive/unified_event_bus_plan.md](archive/unified_event_bus_plan.md) — Scope largely absorbed by reaction_abstraction (ReactionBus, typed protocols, sim_reaction hooks). Remaining AgentBus/LocalMessageBus merge is optional cleanliness.
- [archive/simulator_upgrades_plan.md](archive/simulator_upgrades_plan.md) — S1–S4 shipped (2026-04-12). FixtureDrivenOrchestrator, LLMBackend Protocol + MockLLMBackend, subprocess persistence harness, deterministic seeding. 72 tests, ~880 LOC.
- [archive/reaction_abstraction_plan.md](archive/reaction_abstraction_plan.md) — Phases 1–4 shipped (2026-04-11). Percept/Reaction dual-surface, ReactionBus, producer protocols, factories, runtime unification. Phase 5 folds into substrate P2.
- [archive/substrate_plan.md](archive/substrate_plan.md) — master substrate reference (superseded by the three focused plans above). Full rationale, baselines, statistical hygiene, and fallback strategies.

## Version path to 1.0

Five tracks run in parallel:
- **Track A — Substrate:** the bio-inspired research claim. ~~F0 → P0 → P1 → P2 → P3a → P3b → P3.5 → P4~~ ALL SHIPPED → P5 → P6 → P8.
- **Track B — Prompt layer:** ~~B1~~ SHIPPED → B3 → B4 → B5.
- **Track C — Infrastructure:** ~~LLM path Plans 1–3.5~~ SHIPPED → ~~Reactive peer mesh (C3.5/C3.6/C4.6)~~ ALL SHIPPED. Self-healing loop complete.
- **Track D — Behavioral convergence:** ~~Tier 1 + Tier 2 + Tier 3~~ ALL PASS (41/41 hypotheses) → Scale validation (20+ seeds).
- **Track E — Agent control surface (NEW):** Interactive UX fixes → AgentFactory canonicalization (F-wave) + Game/External Host (G-wave).

| Version | What ships | What it proves | Status |
|---|---|---|---|
| ~~**0.2.x**~~ | Foundations, cleanup, peer flexibility | Friction removed, infrastructure stable | ✅ SHIPPED |
| **0.3.0** | SEM learning loop, valence annotation, cerebellum activation, concept decomposition, behavioral convergence (Tier 1+2+3), reactive mesh (C4+C4.5) | **Cross-session learning without fine-tuning.** Agent learns from own actions, persists, behaves differently. 41/41 experiments. | ✅ SHIPPED |
| **0.3.1** | Interactive UX fixes (RequestInteractionTool honesty, narrator fallback, MaximDisplay wiring, prompt cleanup), agent introspection tools (nac_stats, memory_pressure, loop_stats, pain_triggers_active), bare `--interactive` flag | Agent can distinguish real user input from defaults. Rich panel UI for `--interactive`. Agent can introspect its own learning + pain state. | ✅ **SHIPPED** (PR #156) |
| **0.3.2** | Bidirectional interactive mode: raw terminal input (in-panel rendering), `request_interaction` agent→user prompting, `set_scene` dynamic scene header, `/pause` `/resume` `/display` commands, scrollable log with bio trace dimming, end-of-sim review prompt. Fixed display corruption, stdin contention, tool schema validation, LLM prompt context. | Full interactive simulation experience. User talks to agent, agent asks user questions, scene context updates dynamically. No display corruption, no double-Enter, no flickering. | ✅ **SHIPPED** |
| **0.4** | Tier 3 at scale (20+ seeds), episode boundary enrichment, P5 stress persistence, concept decomposition S2-3 | Learning is robust under variance + load. Substrate persists at 10k+ nodes. Not a fluke. | Planned |
| **0.5** | AgentFactory canonicalization (F+G waves), B3 (acting coach), B4 (replanning — **GATES 1.0**), P6 (extinction), P8 (sleep replay) | One door for every agent. NPCs learn from actions. Agent has coherent voice. Agent recovers from failures. Memory consolidates offline. | Planned |
| **1.0** | All exit criteria passing, behavioral convergence at scale with statistical rigor | Cross-session learning at realistic scale, coherent voice, failure recovery, one construction door | Target |

### 0.3.1 roadmap (detailed)

| Track | What | Plan | Scope | Why |
|---|---|---|---|---|
| **E — Interactive UX** | RequestInteractionTool honest reporting | [interactive_experience_031.md](interactive_experience_031.md) Stage 1 | ~60 LOC | Agent lies to itself about user input; critical fix |
| **E — Interactive UX** | Narrator fallback immersion | Stage 2 | ~70 LOC | Bracket tags break immersion |
| **E — Interactive UX** | Handler selection logging + unknown-mode error | Stage 3 | ~50 LOC | Silent misconfiguration |
| **E — Interactive UX** | Story context word-count truncation | Stage 4 | ~50 LOC | Char-count proxy breaks on unicode |
| **E — Interactive UX** | Wire MaximDisplay into sim_logger (thread-safe + atexit) | Stage 5 | ~180 LOC | Rich panel UI for `--interactive`, designed but never wired |
| **E — Interactive UX** | Light prompt cleanup (remove dead PromptTypes, freeze_context, poll_freeform) | Stage 6 | ~-60 LOC | Dead code with zero production callers |
| **E — Interactive UX** | Integration smoke test | Stage 7 | ~50 LOC | End-to-end validation |
| **Agent introspection** | `nac_stats` — total observations, top-rewarded tools, RPE | [tool_refinement_plan.md](tool_refinement_plan.md) | ~100 LOC | Agent can reason about what it's learned |
| **Agent introspection** | `memory_pressure` — per-tier counts, promotion rate | tool_refinement_plan.md | ~100 LOC | Agent can assess its own memory health |
| **Agent introspection** | `loop_stats` — Hz, cycle time, steps since boot | tool_refinement_plan.md | ~100 LOC | Agent can diagnose its own performance |
| **Agent introspection** | `pain_triggers_active` — current pain triggers + intensity | tool_refinement_plan.md | ~100 LOC | Agent can reason about its own discomfort |

**Why introspection tools in 0.3.1:** These are the cheapest high-value additions (~100 LOC each, no prerequisites). They give the agent self-awareness about its own learning state — which directly supports the 0.3 research claim ("the agent learns from its own actions"). The agent currently *has* causal learning, but can't *see* it. `nac_stats` and `memory_pressure` close that loop. They also provide immediate value for game NPC integration (Stage G3 emotional readout builds on the same data sources).

### 0.4 roadmap (detailed)

| Track | What | Scope | Why |
|---|---|---|---|
| **D — Tier 3 at scale** | Run organic learning experiment with 20+ seeds, report mean ± std | ~1 session | 0.3 proves the mechanism with 1 run; 0.4 proves it's not a fluke |
| **A — Episode boundaries** | Tool execution boundary + semantic shift detection (Rules 1-2) | ~200 LOC | Pre-P5 polish, observe_episode_event is now wired |
| **A — Concept decomposition** | Stages 2-3 (role-tagged edges + ConceptExtractor convergence) | ~250 LOC | Already validated at +63.6 pp, polish pass |
| **A — P5 stress persistence** | 10k+ node persistence stress test | ~500 LOC | Validates substrate robustness under realistic load |
| **C — Peer mesh completion** | ~~C3.5 (`--node update/restart`)~~ SHIPPED, ~~C3.6 (`--node llm`)~~ SHIPPED, ~~C4.6 (auto-undrain)~~ SHIPPED | ✅ COMPLETE | Self-healing reactive mesh |

### 0.5 roadmap (detailed)

Three parallel tracks. B4 replanning is the **1.0 gate** — everything else is supporting work.

| Track | What | Plan | Scope | Why |
|---|---|---|---|---|
| **B — Acting Coach** | B3 — personality scaffolds, speech register, DisplayExtension panels | [prompt_b3_b5_track.md](prompt_b3_b5_track.md) | ~450 LOC | NPCs get coherent, consistent voice |
| **B — Replanning** | B4 — failure diagnosis + prior attempt retrieval (**1.0 GATE**) | [prompt_b4_replanning.md](prompt_b4_replanning.md) | ~400 LOC | Agent recovers from failures instead of repeating them |
| **E — Factory F1** | Design pass: Z1/Z2/Z3 Executor lifetime decision | [agent_factory_canonicalization.md](agent_factory_canonicalization.md) | ~200 LOC | Central design question for all downstream factory work |
| **E — Factory F2-F5** | CLI/sim/Reachy/API migrations through factory | agent_factory_canonicalization.md | ~1500 LOC | 8 hand-rolled entry points → 1 factory door |
| **E — Factory F6** | Hard test enforcement (CI grep, cascade tests) | agent_factory_canonicalization.md | ~1000 LOC tests | Next bridge-wiring bug is a TypeError, not silent no-op |
| **E — Game NPC G1-G5** | Executor in run_turn, HostContext, emotional readout, async dispatch, memory backend | agent_factory_canonicalization.md Wave G | ~940 LOC | External hosts can use Maxim NPCs with full learning |
| **A — Extinction** | P6 — decay without reinforcement vs LRU | [substrate_p6_extinction.md](substrate_p6_extinction.md) | ~400 LOC | Agent forgets appropriately |
| **A — Sleep replay** | P8 — minimum-viable offline consolidation | [substrate_p8_sleep_replay.md](substrate_p8_sleep_replay.md) | ~500 LOC | Memory consolidates between sessions |

**0.5 sequencing:** B3 and factory F1 can start immediately (no dependencies). B4 depends only on B1+P3a (both shipped). Factory F2-F5 depends on F1. G-wave depends on F1. P6 and P8 are independent substrate work. **B4 is the critical path to 1.0** — if it slips, 1.0 slips. The factory refactor (F+G) is engineering hygiene that could defer to post-1.0 if needed without blocking the research claim.

### What 0.3 proved

The 0.3 release demonstrates the core 1.0 claim at prototype scale:

1. **Tier 1 (substrate):** Bio-systems learn affective associations and persist them across sessions (Exp 1: 11/11, Exp 2: 13/13)
2. **Tier 2 (LLM reads learning):** The LLM makes different decisions when it sees the agent's learned valence (Exp 3: 12/12, experienced 10/10 vs fresh 0/10)
3. **Tier 3 (organic learning):** The agent learns from its own actions without scripted training (Exp 4: 5/5, teal rate 0% → 25% → 100%, fresh control DIED)

All 41/41 hypotheses confirmed. No fine-tuning. No prompt engineering beyond surfacing the substrate's learned associations.

**P2 validation was originally scoped to run INSIDE Plan 3's stress test** (Phase A). In practice the P2 Stage 3 sweep is CPU-only and ~27s wall clock, so it shipped standalone on 2026-04-14 via `TestP2ValidationSweep::test_sweep_10_seeds` without waiting on the combined stress run. The reproduction runbook lives at [../experiments/protocols/p2_reward_modulation_reproduction.md](../experiments/protocols/p2_reward_modulation_reproduction.md). Stress phases B (multi-agent fan-out), C (`llama.cpp --parallel`), and E (fault injection) remain and will run under the combined [llm_path_stress_test.md](../experiments/protocols/llm_path_stress_test.md) protocol.

Channels (SMS, email, Slack, narrative speech) are **TEXT modality with context metadata**, not separate modalities. Channel rollout: SMS + narrative in 0.3, email + Slack in 0.5. See [substrate_plan.md](archive/substrate_plan.md) for phase definitions.

## How LLM path refinement interleaves with substrate P2

Timeline (rough, not calendar-committed). As of 2026-04-14, steps 1–3a, Plan 3.6 R5, Plan 4 Stage A+B, and substrate P2 Stage 3 are done. Only Plan 4 Stage C and stress phases B/C/E remain.

1. **✅ SHIPPED (2026-04-12):** Plan 1 R0 + R1 + R1 loose ends.
   - R0: dead mesh deleted (commit `e811787`)
   - R1 core: 9-step urllib migration + `maxim/utils/http.py` (PRs #88, #90)
   - R1 cleanup: dual-format logging + docs/memory/audit pass (commits `c8a07e9`, `845af61`)
   - R1 loose ends: CI grep wired, `internal=False` default, `make_http_response` helper (PR #91, commit `3a579de`)
   - Fast suite: 4004 passed on main. Leader updated + restarted cleanly. Pre-Plan-3 restart baseline: ~63s.

2. **✅ SHIPPED (2026-04-12):** Plan 2 (Typed Errors + Role Detection).
   - R2a: `runtime/role.py::detect_and_apply_role` called at the top of `cli.py::main()` BEFORE subcommand dispatch
   - R2b: `BackendError` hierarchy in `types.py` mirroring `HTTPError` shape + `_normalize_request_context` canonical shim in `agents/llm_worker.py` + `INFERENCE_BROKEN_BACKOFF_S = 15.0` single source of truth
   - R2c: two-stage probe with `enable_stage2=True` + per-outcome cache TTL table in `probe_cache.py` + corruption log promotion
   - R2d: SSRF check moved to `utils/net.py::validate_base_url` (shared helper)
   - PRs #92 + #93 (pre-merge review round caught 11 items, 2 real behavior bugs). 4073 tests passing post-R2.

3. **✅ SHIPPED (2026-04-12):** Plan 3 (Fast Failover) — `_MaximPeerBackend`, the 52s fail-slow is dead.
   - R2.5: `_MaximPeerBackend` purpose-built single-HTTP-call backend + typed router dispatch + `BACKEND_CLASSES` + `dispatch_exhausted` + safety-net counter + `stream_post` primitive (commit `824d737`, +2198 lines)
   - R2.6: probe consolidation through `health_check` + `for_url` factory + compat shims retained with CI allow-list (commit `d09b74d`, +313/-193)
   - Pre-merge review fix: 12 findings folded including 2 critical (`for_url` env-var race + `_emit_dispatch_exhausted` shim bypass) and a rewrite of `BACKEND_CLASSES` from dead identity map to real lazy-import dispatch (commit `b26ef4b`, +565/-120)
   - PR #94, merged as `ce5f034`. 4142 fast-suite tests passing, 3 CI grep invariants clean.
   - **Programmatic gate:** < 5s p99 against mocked-dead-peer. Real leader-restart re-measurement runs in stress protocol Phase D.

3a. **✅ POST-PLAN-3 HOTFIXES + IMPROVEMENTS (2026-04-13, main branch):**
   - **Stress test hotfixes (commit `6181329`):** (A) `_INFERENCE_PROXY_TIMEOUT_S` raised to 300s + `min(total,120)` cap removed — 60s was too short for 14B models. (B) `warmup()` calls `_ensure_endpoint_registered()` to pre-flight DNS at startup. (C) `reset_session_cost()` also resets `_provider_states` — stale backoffs no longer bleed across sim runs.
   - **Streaming proxy (Cloudflare 524 fix):** `leader_proxy._proxy_request()` now uses HTTP/1.1 chunked encoding via `raw_proxy_forward_streaming()` so the first byte reaches Cloudflare before the 125s edge timeout. `raw_proxy_forward_streaming()` added to `utils/http.py`.
   - **sim_roundup dispatch_exhausted fix:** `simulation/orchestrator.py` calls `reset_shutdown()` before `analyze_simulation()` — the LLM shutdown signal was causing every sim summary to hit `dispatch_exhausted` at 0.1ms.
   - **Empty-choices silent failure fixed:** `_MaximPeerBackend._parse_llm_response()` now raises `BackendInferenceBroken` (instead of returning empty content) when `choices` is empty. `complete_with_usage()` wraps the parse call to emit `peer_backend_failed` JSONL before re-raising.
   - **Structured `provider_silenced` event:** `LLMRouter._set_long_backoff()` / `_set_short_backoff()` now emit a WARNING-level `provider_silenced` structured log event with `provider`, `backoff_s`, `reason`, `consecutive_errors`.
   - **Doctor improvements:** `check_env_config()` (env var audit for MAXIM_ROLE, MAXIM_LLM_ENABLED, MAXIM_LLM_PROFILE, MAXIM_LLM_N_CTX, MAXIM_SKIP_REMOTE_PROBE, MAXIM_PEER_PROBE_KEY), `check_context_window()` (context window size detection), `check_role()` now surfaces divergence between `leader_mode.detect_role()` and `role.detect_role()` as a `role_divergence` warn. +11 new tests (4153 total).
   - **Security fixes (node_security_simplification.md Phase 1):** `secrets.compare_digest()` for auth comparison, rate-limiter buckets by source IP (not auth token), corrected peer help text in `tunnel/cli.py`.

4. **Stress test (one combined run):**
   - **Phase A: ✅ SHIPPED (2026-04-14)** as substrate P2 Stage 3. Ran standalone outside the combined stress test because the sweep is fast (~27s wall clock on CPU) and the substrate work didn't need to wait on multi-agent fan-out. Results: [experiments/p2_reward_modulation_sweep.md](../experiments/p2_reward_modulation_sweep.md). Mean target gain **+56.0 ± 29.0 pp**, distractor drift **0.0 ± 0.0 pp**, monotone **94%**, 9/10 seeds individually.
   - Phase B: multi-agent fan-out (exercises AgentPool under the new LLM path) — REMAINING
   - Phase C: `llama.cpp --parallel` batching PoC — REMAINING
   - **Phase D: ✅ SHIPPED (2026-04-13)** — [llm_path_stress_20260413.md](../experiments/results/llm_path_stress_20260413.md). All designed-for gates PASS: fast-fail, no stacked timeouts, no provider pollution, sim resumes. Recovery-time gate was inconclusive under sim workload (sim-cadence artifact), addressed in Phase D2 below.
   - **Phase D2: ✅ SHIPPED (2026-04-14)** — [llm_path_stress_plan4_20260414.md](../experiments/results/llm_path_stress_plan4_20260414.md). Uses the new `maxim bench recovery-time` tight-loop harness (Plan 4 Stage B) instead of sim workload. **58.68s recovery window** on real RTX 5080, matching leader self-reported 53s reload + ~5s proxy gap. Peer-side overhead ≈ 0s. 750/750 `agent_id` coverage (validates Plan 4 Stage A end-to-end). 199/199 failures typed as `BackendDown`.
   - Phase E: fault injection — REMAINING

5. **✅ SHIPPED (2026-04-14):** **Plan 3.6 R5 — VRAM spillover detection.** PR #99, commit `2884e58`. Doctor `check_vram_pressure` + spawn-time `_check_vram_spillover_risk` + shared `project_vram_usage` math in `lane_models.py`. Dynamic headroom `max(1.5, 0.55 × weights_gb)` calibrated to the 2026-04-13 incident. Plus fix for a pre-existing silent bug in `check_llm_model_active` (mutable-global import-by-name anti-pattern). 17 new tests. R1–R4 (multi-leader `peer.yml`) remain draft pending user's second-GPU bring-up.

6. **✅ SHIPPED (2026-04-14):** **Plan 4 Stage A + Stage B.** Commit `71f7c24` on `feat/llm-path-operator-visibility`, PR in pre-merge review.
   - **Stage A: agent_id observability fix.** Router capability-flag kwarg forwarding in `_invoke_backend` + `set_context` boundary binding in `LLMWorker._call_llm_with_timeout` (alongside existing `set_cancel_event`, before `copy_context()`) + contextvar fallback in `_normalize_request_context`. Closes the Phase D "agent_id=null in peer_backend_call" observability gap. 11 new regression tests.
   - **Stage B: recovery-time bench harness.** New `maxim bench recovery-time` CLI subcommand at `src/maxim/bench/` (package named `bench` not `benchmark` to avoid shadowing the existing `maxim.api.benchmark` public verb). Fires chat completions in a tight loop against a peer URL, extracts a rigorous recovery-time number from the first `success → failure → success` transition. 21 new tests. Real hardware validation is the Phase D2 run above.
   - **Stage C (mesh.yml + admin API + per-agent rate limiting)** — **C1+C2+C3.1 SHIPPED, C3.2 in PR, C3 remainder DEFERRED.** See the C1/C2/C3.1/C3.2 sub-bullets above for the per-stage status. Full scope preserved in [llm_path_operator_visibility.md](llm_path_operator_visibility.md) under "Phases".

7. **▶ NEXT (substrate track resumes):** Substrate P2 Stage 3 **COMPLETE (2026-04-14)** — `substrate_recognition.md` closed. Remaining stress-test phases B, C, E + Plan 4 Stage C can run in any order when their respective bottlenecks surface. Next substrate plan: [substrate_binding_persistence.md](archive/substrate_binding_persistence.md) (P3a onward).

8. **Release 0.4** with LLM path refinement complete + substrate P2 validated on real embeddings. Substrate P2 gate is CLOSED; 0.4 ships when the remaining stress phases (B, C, E) + Plan 4 Stage C are ready.

9. **Back to substrate work:** P3a + P3b + P4 in 0.5, built on the stabilized LLM path. Plan doc: [substrate_binding_persistence.md](archive/substrate_binding_persistence.md).

**Review pattern per plan (refined after R2):** each plan implements on a `feat/<plan>` branch. Before opening/merging the PR — **not after** — spawn two review Claudes (Executor lens + Architecture lens) in parallel read-only sessions against the branch tip. Findings get folded into the same branch via a follow-up commit, THEN the PR opens. One PR per plan, no `fix/<plan>-loose-ends` split.

R1 used the old "ship then review" timing → required PR #91 follow-up for CI grep gap + `HTTPEndpoint.internal` unsafe default + `make_http_response` helper + doc drift. R2 refined to "review before merge" → 11 findings (2 real behavior bugs: stage-2 probe mis-classification + argv scan subcommand gap) all folded into the same PR before it merged. Save the extra commit; catch the bugs earlier in the review loop.

**Every plan without a pre-merge review round is gambling.** Tests catch known failure modes; reviews catch unknown ones. Both R1 and R2 had bugs that passed 4000+ unit tests and would have shipped silently. See [feedback_review_before_ship.md](../../.claude/projects/-Users-dennyschaedig-Scripts-Maxim/memory/feedback_review_before_ship.md) for the full evidence trail and the review prompt templates.

**Why this order was load-bearing (decided 2026-04-12):**
- At 2026-04-12 the LLM infrastructure was **too broken to use for P2 testing at all.** The 52s retry loop + probe fragility + lack of per-agent observability would have made P2 validation data unreliable and un-attributable ("substrate bug vs. LLM flakiness?"). Running P2 first was considered and rejected.
- Substrate P2 needs a stable LLM path to validate correctly (for multi-agent stress runs — the 2026-04-14 Stage 3 sweep is single-process and CPU-only, so it didn't need the full LLM path; the multi-agent cross-session-learning 1.0 gate will).
- Multi-agent P2 runs need per-agent observability (which Plans 1-2 provide via `RequestContext`, and Plan 4 Stage A closed the last gap)
- Stress test needs typed exceptions to classify failures properly (Plan 2)
- Plan 4's admin API is where you'd inspect per-agent P2 reward modulation under load
- Shipping Plans 1-3 alone without Plan 4 means you can't debug concurrent-agent P2 validation effectively

**What actually happened (2026-04-14 retrospective):** the LLM path stabilization + substrate P2 ran in parallel after Plan 3.5 shipped. The single-process P2 sweep didn't need to wait on the combined stress test — it's fast and CPU-only. Multi-agent behavioral convergence (1.0 gate) still needs the full stable LLM path.

## 1.0 exit criteria

- **Prerequisite waves shipped:** Cleanup Wave (0.2.2), foundations_plan (F0.1–F0.8), simulator_upgrades_plan (S1–S4), P0 fixture-difficulty pilot (fixtures calibrated).
- **Substrate P1 through P8:** P1 and P2 pass mechanistic criteria. P3a and P3b pass (P3a head-to-head vs TF-IDF gate baseline, P3b with metadata-grep regression check). P3.5 certifies round-trip at scale. P4 passes in 0.3 with minimal real vision and in 0.4 with production vision, beating the OpenCLIP head-to-head gate baseline. P5 passes stress persistence. P6 passes extinction vs LRU head-to-head. **P8 passes minimum-viable sleep replay** — retrieval F1 improves on replayed probes without new input. Every phase passes both unit-sim and system-sim tiers and clears the persistence round-trip contract. Report mean + std across ≥10 seeds (≥20 for P4).
- **Track B B4 (gates 1.0):** Replanning recovers from induced failures instead of regenerating identical plans; NPCs exhibit distinct, consistent voices in blind A/B sim runs.
- **Living-doc discipline satisfied:** both [behavioral_convergence_practice.md](behavioral_convergence_practice.md) and [memory_consolidation_practice.md](memory_consolidation_practice.md) have logged at least one experiment entry each by the 1.0 tag. Soft discipline, not a hard gate — but the 1.0 release is the one version where you do enforce it on yourself.

## Rules for this directory

- **Active plans stay in the root.** Anything in the root is on the critical path.
- **Deferred plans must state a revive trigger.** If you can't state the trigger, it doesn't belong in deferred — it belongs in archive.
- **No ghost plans.** If a plan references a module that doesn't exist (e.g., the old `NarrativeModulator`), fix the plan or delete the reference.
- **Merge before multiplying.** If two plans overlap by more than a phase, merge them. Historical example: salience_abstraction was folded into substrate_plan because `WhereCoord` required embedding-space percepts anyway.
