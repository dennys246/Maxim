# Maxim Plans

Current version: **0.5.0** (published on PyPI as `pymaxim`). Next publish: **0.6.0**.
Target: **1.0** — cross-session learning demonstrated without LLM fine-tuning.

## Version Roadmap

| Version | Theme | Status |
|---------|-------|--------|
| **0.6** | **Generalizable embodiment** — E0 sim wiring + E1 Asset Foundry | Ready to publish |
| **0.7** | **Feature completion** — E2-E3 (foundry with real LLM + curation), B3 (Acting Coach), F3-F5 (Agent Factory canonicalization) | Next |
| **1.0** | **Validation + polish** — P5 stress persistence (the gate), API/CLI surface review for edge cases, final docs | Target |

**Discipline:** 0.7 is the last feature version. 1.0 is validation + polish only — no new features, just proving what's built holds up at scale and the API/CLI surfaces are complete for power users.

## Active (gating 1.0)

**All original 1.0 gates are CLOSED** (2026-04-19). The remaining 1.0 gate is **P5 stress persistence** (10k+ nodes).

Foundations, reaction abstraction, simulator upgrades, P0-P4, P6, P8, B4, SEM execution hook, E0 generalizable embodiment — all complete.

- [substrate_binding_persistence.md](archive/substrate_binding_persistence.md) — **✅ SPLIT COMPLETE + ARCHIVED** (2026-04-17). Now a pure index. **All four 0.3-target phases CLOSED.** Per-phase plan files created for 0.5 track:
  - [substrate_p5_stress_persistence.md](substrate_p5_stress_persistence.md) — Draft. 10k+ node persistence stress. Depends on P3.5 + P4.
  - [substrate_p6_extinction.md](archive/substrate_p6_extinction.md) — **✅ SHIPPED** (2026-04-19). Hebbian decay beats LRU across 10 seeds. Results: [experiments/p6_extinction_results.md](../experiments/p6_extinction_results.md).
  - [substrate_p8_sleep_replay.md](archive/substrate_p8_sleep_replay.md) — **✅ SHIPPED** (2026-04-19). Sleep replay F1 improves vs no-replay control, 10-seed sweep. Results: [experiments/p8_sleep_replay_results.md](../experiments/p8_sleep_replay_results.md). Activates [memory_consolidation_practice.md](memory_consolidation_practice.md).
  - [prompt_b3_b5_track.md](prompt_b3_b5_track.md) — Draft. Acting Coach + embodiment/narrative separation.
  - [prompt_b4_replanning.md](archive/prompt_b4_replanning.md) — **✅ COMPLETE** (2026-04-19). **1.0 GATE CLOSED.** All 3 stages shipped. Stage 3 blind A/B: treatment 100% vs control 0%, mean Jaccard 0.894. Results: [experiments/b4_replanning_results.md](../experiments/b4_replanning_results.md). 12 tests in `tests/substrate/test_b4_replanning_ab.py`.
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
- [memory_consolidation_practice.md](memory_consolidation_practice.md) — refines the P8 sleep-replay mechanism with alternative strategies, promotion rules, interference analysis. **ACTIVATED** (P8 shipped 2026-04-19). Next steps: alternative replay strategies, interference analysis.

## Parallel (ship anytime, not gating 1.0)

- [archive/interactive_experience_031.md](archive/interactive_experience_031.md) — **✅ SHIPPED** (2026-04-18, PR #156). 8 stages, ~700 LOC.
- [archive/substrate_concept_decomposition.md](archive/substrate_concept_decomposition.md) — **✅ SHIPPED** (2026-04-17). 100% concept-level recall vs 36.4% baseline.
- [substrate_episode_boundary_enrichment.md](substrate_episode_boundary_enrichment.md) — **PARTIAL** (2026-04-17). Stage 3 (pain/salience spike) SHIPPED via sem_learning_loop.md. `observe_episode_event` now wired into production agent loop via behavioral_convergence_wiring.md. Stages 1-2 (tool execution + semantic shift) remain — ship before P5.
- [biosystem_unification.md](biosystem_unification.md) — **central tracking doc** (2026-04-14, updated 2026-04-17). Waves 0-3 **ALL SHIPPED + ARCHIVED**. Wave 4 (agent_factory_canonicalization) not scheduled.
- [tool_refinement_plan.md](tool_refinement_plan.md) — living doc for agent tool surface curation
- [agent_factory_canonicalization.md](agent_factory_canonicalization.md) — **RUNNING DOC, trigger #4 activated** (2026-04-18). The Option D follow-up to `executor_bootstrap_unification.md` — make `AgentFactory.create_agent` the only door for constructing an agent in Maxim. **Wave G (Game/External Host) folded in** from game_npc_integration.md: wire Executor + bio-pipeline into `AgentPool.run_turn()`, HostContext protocol, async tool dispatch, emotional state readout, memory backend. F-wave ~1500-2500 LOC + G-wave ~940 LOC. Target: 0.5.
- [deferred/node_security_simplification.md](deferred/node_security_simplification.md) — Phase 1 ✅ SHIPPED. Phase 2 config-surface unification deferred.
- [reactive_peer_mesh_roadmap.md](reactive_peer_mesh_roadmap.md) — living roadmap for the full reactive peer mesh arc (C3→C9). C3-C4.6 COMPLETE. C5+ remain.
- [deferred/cross_platform_file_lock.md](deferred/cross_platform_file_lock.md) — shell plan to unify `utils/process_lock` and `filelock.FileLock`. Blocks nothing.
- [deferred/mesh_doc_transport.md](deferred/mesh_doc_transport.md) — shell plan for mesh-to-mesh structured doc exchange (C9). Not started.
- [deferred/pain_bus_bridge_subscriber_unification.md](deferred/pain_bus_bridge_subscriber_unification.md) — shell plan for bridge×subscriber attribution-asymmetry fix. Not started.
- **Interactive NAc attribution** (future concern, no plan file yet) — NAc tool-outcome learning is suppressed during interactive mode (0.4.0) because human-directed tool calls would corrupt the causal model with patterns that depend on human presence. The interim fix gates `record_tool_start`/`record_tool_complete` and the PainBus NAc subscriber on `get_interactive_mode() == ON`. Proper fix: add `human_influenced: bool` metadata to NAc links, or implement separate interactive vs autonomous learning modes so the agent can learn from interactive sessions without conflating human-directed causality with environmental causality. Revive when behavioral convergence experiments show drift between interactive-trained and autonomous-trained agents.
- [llm_path_refinement.md](llm_path_refinement.md) — meta-plan for the LLM routing path refactor. Plans 1-3.5 archived; Plan 3.6 R5 shipped; **Plan 4 Stages A+B + C1-C3.6 + C4+C4.5+C4.6 ALL SHIPPED.** Reactive mesh self-healing loop complete. Only stress phases B/C/E remain in scope. Architecture ref: [../architecture/llm_routing.md](../architecture/llm_routing.md).
  - [llm_path_peer_failover.md](archive/llm_path_peer_failover.md) — Plan 3.6 R5 (VRAM spillover) ✅ SHIPPED. R1-R4 (multi-leader) remain draft, on hold until second GPU.
  - [llm_path_operator_visibility.md](llm_path_operator_visibility.md) — Plan 4. **Core stages ALL SHIPPED** (A, B, C1-C3.6, C4, C4.5, C4.6). Remaining deferred scope (admin API, rate limiting, key rotation) tracked in [reactive_peer_mesh_roadmap.md](reactive_peer_mesh_roadmap.md) as C6/C7.
  - Deferred: [deferred/llm_path_multi_peer_dispatch.md](deferred/llm_path_multi_peer_dispatch.md), [deferred/llm_mesh_capability_aware.md](deferred/llm_mesh_capability_aware.md), [deferred/llm_path_async_router.md](deferred/llm_path_async_router.md), [deferred/llm_path_fair_scheduling.md](deferred/llm_path_fair_scheduling.md)

  **Long-term mesh roadmap**: [reactive_peer_mesh_roadmap.md](reactive_peer_mesh_roadmap.md). Stages C3-C4.6 complete; C5 (capacity-aware routing), C6 (admin API), C7 (security hardening) remain.

## Deferred (post-1.0, revive on trigger)

Design work is preserved in [deferred/](deferred/). Each plan has an explicit "revive when" condition at the top.

- [deferred/bio_system_plugin_plan.md](deferred/bio_system_plugin_plan.md) — plugin discovery for bio-systems (extends the `maxim.robots` pattern). Depends on the `BioSystem` Protocol landing incrementally during substrate work. Revive when external contributors want to add bio-systems or a research collaborator needs substrate A/B testing.
- [deferred/mother_npc_stimulus_plan.md](deferred/mother_npc_stimulus_plan.md) — **needs heavy refinement.** Two-agent stimulus pattern: Baby Maxim is the AUT with frozen LLM and evolving substrate; Mother NPC is a separate agent with her own LLM that produces realistic, varied percepts Baby learns from. Interaction is percepts only, zero information leak beyond that surface. Gives behavioral convergence experiments scalable stimulus variety without breaking the "no fine-tuning" research claim. Revive when [behavioral_convergence_practice.md](behavioral_convergence_practice.md) has ≥2 successful experiments + 1 blocked-on-variety. Isolation leak vector list in the plan is a starting point, not a contract — heavy refinement needed at revive time.
- [deferred/pecking_order_graph_plan.md](deferred/pecking_order_graph_plan.md) — unified hierarchy DAG
- [deferred/mother_maxim_plan.md](deferred/mother_maxim_plan.md) — persistent collective memory
- ~~[deferred/asset_foundry_plan.md](deferred/asset_foundry_plan.md)~~ — **PROMOTED to 0.6** (pre-1.0). **Stage 0 (sim affordance gap) ✅ SHIPPED** (2026-04-19). `--embodiment` now works with `--sim` across all modes. Stages E1-E3 (LLM-driven component generation + gauntlet + curation) remain.
- [deferred/dungeon_master_extensions.md](deferred/dungeon_master_extensions.md) — DM post-MVP features

## Archive

Completed or superseded plans live in [archive/](archive/).

Recently archived (2026-04-19):
- [archive/prompt_b4_replanning.md](archive/prompt_b4_replanning.md) — **✅ COMPLETE + ARCHIVED**. 1.0 gate closed. Blind A/B: treatment 100% vs control 0%.
- [archive/substrate_p6_extinction.md](archive/substrate_p6_extinction.md) — **✅ SHIPPED + ARCHIVED**. Hebbian decay beats LRU, 10 seeds.
- [archive/substrate_p8_sleep_replay.md](archive/substrate_p8_sleep_replay.md) — **✅ SHIPPED + ARCHIVED**. Sleep replay F1 improves, 10 seeds.
- [archive/agent_loop_state_repair.md](archive/agent_loop_state_repair.md) — **✅ COMPLETE + ARCHIVED**. State desync + observe-only + weak prompt fixed.
- [archive/interactive_experience_031.md](archive/interactive_experience_031.md) — **✅ SHIPPED + ARCHIVED** (PR #156). 8 stages.
- [archive/peer_update_pip_mode.md](archive/peer_update_pip_mode.md) — **✅ SHIPPED + ARCHIVED**. All 3 stages.
- [archive/substrate_concept_decomposition.md](archive/substrate_concept_decomposition.md) — **✅ SHIPPED + ARCHIVED**. 100% vs 36.4% baseline.
- [archive/game_npc_integration.md](archive/game_npc_integration.md) — **FOLDED** into agent_factory_canonicalization.md Wave G.
- [archive/llm_path_peer_failover.md](archive/llm_path_peer_failover.md) — **R5 SHIPPED + ARCHIVED**. R1-R4 on hold until second GPU.

Previously archived (2026-04-17):
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

| Version | What ships | What it proves | Status |
|---|---|---|---|
| ~~**0.2.x**~~ | Foundations, cleanup, peer flexibility | Friction removed, infrastructure stable | ✅ SHIPPED |
| **0.3.0** | SEM learning loop, valence annotation, cerebellum activation, concept decomposition, behavioral convergence (Tier 1+2+3), reactive mesh (C4+C4.5) | **Cross-session learning without fine-tuning.** Agent learns from own actions, persists, behaves differently. 41/41 experiments. | ✅ SHIPPED |
| **0.3.1** | Interactive UX fixes (RequestInteractionTool honesty, narrator fallback, MaximDisplay wiring, prompt cleanup), agent introspection tools (nac_stats, memory_pressure, loop_stats, pain_triggers_active), bare `--interactive` flag | Agent can distinguish real user input from defaults. Rich panel UI for `--interactive`. Agent can introspect its own learning + pain state. | ✅ **SHIPPED** (PR #156) |
| **0.3.2** | Bidirectional interactive mode: raw terminal input (in-panel rendering), `request_interaction` agent→user prompting, `set_scene` dynamic scene header, `/pause` `/resume` `/display` commands, scrollable log with bio trace dimming, end-of-sim review prompt. Fixed display corruption, stdin contention, tool schema validation, LLM prompt context. | Full interactive simulation experience. User talks to agent, agent asks user questions, scene context updates dynamically. No display corruption, no double-Enter, no flickering. | ✅ **SHIPPED** |
| **0.4** | Tier 3 at scale (20+ seeds), episode boundary enrichment, P5 stress persistence, concept decomposition S2-3 | Learning is robust under variance + load. Substrate persists at 10k+ nodes. Not a fluke. | Planned |
| **0.5** | AgentFactory canonicalization (F+G waves), B3 (acting coach), ~~B4 (replanning)~~ **✅ COMPLETE**, ~~P6 (extinction)~~ **✅ COMPLETE**, ~~P8 (sleep replay)~~ **✅ COMPLETE** | One door for every agent. NPCs learn from actions. Agent has coherent voice. **Agent recovers from failures (1.0 GATE CLOSED).** Memory consolidates offline. | In progress |
| **0.6** | **Generalizable embodiment** — Asset Foundry (promoted from deferred), SEM affordance tools in simulation, entity_ref wiring for sim AUT, component generation pipeline | Sim path and robot path use identical tool injection. Agent can interact with SEM entities in simulation with full pain-cascade learning. No code path divergence between sim and live. | Planned |
| **1.0** | All exit criteria passing, behavioral convergence at scale with statistical rigor, generalizable embodiment | Cross-session learning at realistic scale, coherent voice, failure recovery, one construction door, **unified sim/robot embodiment** | Target |

### 0.6 — Generalizable Embodiment (ready to publish)

| Stage | What | Status |
|---|---|---|
| **E0** | Wire `entity_ref` through sim orchestrator. `--embodiment` works with `--sim`. | ✅ SHIPPED |
| **E1** | Asset Foundry: generate → validate → SEM protocol tests → gauntlet → score → curate | ✅ SHIPPED |

Also includes: sim stall fix (dd6c29a), B4 replanning (1.0 gate closed), P6 extinction, P8 sleep replay, F2 factory migration.

### 0.7 — Feature Completion (next)

The last feature version before 1.0. Three parallel tracks:

| Track | What | Plan | Scope | Why |
|---|---|---|---|---|
| **E2-E3 — Foundry with real LLM** | Run foundry with actual LLM, curate promoted components into seed library | [asset_foundry_plan.md](deferred/asset_foundry_plan.md) | ~600 LOC + curation | Prove the foundry generates useful entities, not just valid YAML |
| **B3 — Acting Coach** | Personality scaffolds, speech register, embodiment tool guidance | [prompt_b3_b5_track.md](prompt_b3_b5_track.md) | ~450 LOC | Agent uses affordance tools effectively; NPCs have consistent voice |
| **F3-F5 — Agent Factory** | Make `AgentFactory.create_agent` the single construction door, sim/Reachy/API migration | [agent_factory_canonicalization.md](agent_factory_canonicalization.md) | ~1500 LOC | Eliminate dual-construction paths. One door for every agent. |

**0.7 sequencing:** B3 and E2-E3 are independent. F3-F5 depends on F2 (shipped in 0.5). All three tracks can run in parallel.

### 1.0 — Validation + Polish (target)

No new features. Proving what's built holds up at scale and the API/CLI surfaces are complete for power users.

| What | Plan | Why |
|---|---|---|
| **P5 stress persistence** (the gate) | [substrate_p5_stress_persistence.md](substrate_p5_stress_persistence.md) | 10k+ node persistence validates substrate at realistic scale |
| **API/CLI surface review** | Sweep for edge cases and missed opportunities in the public API (`api.py`, `__init__.py`) and CLI (`cli.py`, `cli_parser.py`). The API was designed before much of the current infrastructure — ensure all new capabilities (foundry, embodiment, bio-stack, factory) are properly accessible to power users. | Pre-1.0 accessibility |
| **Agent memory transfer docs** | Universal onboarding document set that allows any agent (Claude, human, or future AI) to quickly understand the repo's architecture, invariants, and conventions without reading every memory file. Refine `.claude/` memory, consolidate into a transferable format. | Knowledge continuity |
| **Final docs pass** | Publication guide, user docs, architecture docs | Ship-ready documentation |

**Discipline:** if it's not P5, a missing API/CLI surface, or the memory transfer docs, it doesn't go in 1.0.

### What 0.3 proved

The 0.3 release demonstrates the core 1.0 claim at prototype scale:

1. **Tier 1 (substrate):** Bio-systems learn affective associations and persist them across sessions (Exp 1: 11/11, Exp 2: 13/13)
2. **Tier 2 (LLM reads learning):** The LLM makes different decisions when it sees the agent's learned valence (Exp 3: 12/12, experienced 10/10 vs fresh 0/10)
3. **Tier 3 (organic learning):** The agent learns from its own actions without scripted training (Exp 4: 5/5, teal rate 0% → 25% → 100%, fresh control DIED)

All 41/41 hypotheses confirmed. No fine-tuning. No prompt engineering beyond surfacing the substrate's learned associations.

**P2 validation was originally scoped to run INSIDE Plan 3's stress test** (Phase A). In practice the P2 Stage 3 sweep is CPU-only and ~27s wall clock, so it shipped standalone on 2026-04-14 via `TestP2ValidationSweep::test_sweep_10_seeds` without waiting on the combined stress run. The reproduction runbook lives at [../experiments/protocols/p2_reward_modulation_reproduction.md](../experiments/protocols/p2_reward_modulation_reproduction.md). Stress phases B (multi-agent fan-out), C (`llama.cpp --parallel`), and E (fault injection) remain and will run under the combined [llm_path_stress_test.md](../experiments/protocols/llm_path_stress_test.md) protocol.

Channels (SMS, email, Slack, narrative speech) are **TEXT modality with context metadata**, not separate modalities. Channel rollout: SMS + narrative in 0.3, email + Slack in 0.5. See [substrate_plan.md](archive/substrate_plan.md) for phase definitions.

**Review pattern (refined after R2, validated through R3 + E0 + E1):** each plan implements on a `feat/<plan>` branch. Before opening/merging the PR — **not after** — spawn two review Claudes (Executor lens + Architecture lens). Findings get folded into the same branch, THEN the PR opens. See [feedback_review_before_ship.md](../../.claude/projects/-Users-dennyschaedig-Scripts-Maxim/memory/feedback_review_before_ship.md) for evidence + templates.

**Historical LLM path interleave timeline** (Plans 1-4, stress phases A-E, 2026-04-12 through 2026-04-14) is preserved in git history. All shipped. Only stress phases B (multi-agent fan-out), C (`llama.cpp --parallel`), and E (fault injection) remain — not version-gating.

## 1.0 exit criteria

All originally defined gates are **CLOSED** (2026-04-19). Remaining criteria for 1.0:

- ✅ **Substrate P1 through P4, P6, P8:** All pass. P5 (stress persistence) remains — the final substrate gate.
- ✅ **B4 replanning:** Treatment 100% vs control 0%, Jaccard 0.894.
- ✅ **Behavioral convergence:** 41/41 hypotheses, all 3 tiers.
- ✅ **Generalizable embodiment (E0):** `--embodiment` works with `--sim` across all modes.
- ✅ **Living-doc discipline:** Both practice docs have experiment entries.
- **P5 stress persistence:** 10k+ node persistence stress test. The final hard gate.
- **API/CLI surface review:** Sweep the public API and CLI for edge cases and missed opportunities. Not a hard gate — a quality bar.
- **Agent memory transfer docs:** Universal onboarding document set for knowledge continuity across agents.
- **0.7 feature completion:** E2-E3 (foundry with real LLM), B3 (Acting Coach), F3-F5 (Agent Factory).

## Rules for this directory

- **Active plans stay in the root.** Anything in the root is on the critical path.
- **Deferred plans must state a revive trigger.** If you can't state the trigger, it doesn't belong in deferred — it belongs in archive.
- **No ghost plans.** If a plan references a module that doesn't exist (e.g., the old `NarrativeModulator`), fix the plan or delete the reference.
- **Merge before multiplying.** If two plans overlap by more than a phase, merge them. Historical example: salience_abstraction was folded into substrate_plan because `WhereCoord` required embedding-space percepts anyway.
