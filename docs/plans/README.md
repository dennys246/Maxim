# Maxim Plans

Current version: **0.8.0** (published on PyPI as `pymaxim`).
Target: **1.0** — cross-session learning demonstrated without LLM fine-tuning.

## Version Roadmap

| Version | Theme | Status |
|---------|-------|--------|
| **0.6** | **Generalizable embodiment** — E0 sim wiring + E1 Asset Foundry | **Published** |
| **0.7** | **Self-generating simulations** — Imagination, B3 Acting Coach, E2-E3, F3-F5, SEM discovery, gating, deliberation | **SHIPPED** (2026-04-20) |
| **0.8** | **Cognitive maturity + embodiment** — WM+Exec, PFC cycle, temporal credit, display overhaul, reflex system, proprioceptive discovery, affordance concept transfer, entity ownership, component damage | **SHIPPED** (2026-04-25) |
| **1.0** | **Validation + stabilization + grounding** — cross-session proof, bio-system protocol freeze, sensorimotor cradle, SEM world enrichment, SCN feedback loop, cleanup, docs | Target |

**Discipline:** 0.7 was the last major feature version. 0.8 matured cognition and embodiment. 1.0 stabilizes, validates, and grounds — every bio-system fully operational with closed feedback loops, interfaces frozen with future-proof protocols, and the agent demonstrably learns from its own body.

## 1.0 — Validation + Stabilization + Grounding

The unified plan: [v1_refinement.md](v1_refinement.md)

| Section | Items | Status |
|---------|-------|--------|
| **Validation** | Cross-session sim experiment (prove the 1.0 claim) | **PARTIAL PASS** — 3 memories/turn on resume ([Exp 10](../experiments/10_cross_session_enrichment.md)) |
| **Bio-system stabilization** | Protocol enrichment (freeze-worthy interfaces), SCN oscillator feedback (close the loop), SEM world enrichment Phases 2-3 (rich environments) | Pending |
| **Sensorimotor grounding** | Cradle of Artificial Civilization (fire hurts, learned through sensors not language) | Pending |
| **Pipeline completion** | ToolPainBridge temporal migration (~50 LOC), episode semantic shift (Stage 2) | **P1 SHIPPED** (temporal events), P2 pending |
| **Cleanup** | Probe shim removal, dead code, DamageEntityTool shim, modulator sensors, health derived, raw constructor enforcement | **C1+C2+C3 SHIPPED** (PR #196, 2026-04-26); C4-C6 pending (0.9 deprecation cycle) |
| **Docs** | Agent memory transfer, API/CLI review, final docs pass | Pending |

**1.0 exit criteria:**
- ✅ Substrate P1-P8: all pass
- ✅ B4 replanning: treatment 100% vs control 0%, Jaccard 0.894
- ✅ Behavioral convergence: 41/41 hypotheses, all 3 tiers
- ✅ Generalizable embodiment (E0): `--embodiment` works with `--sim`
- ✅ 0.7+0.8 feature completion: all tracks shipped
- ��� P5 stress persistence: 1.0 gate CLOSED (2026-04-21)
- **PARTIAL PASS**: Cross-session validation (V1) — 3 memories/turn on resume, predictions/concepts pending
- Pending: Bio-system protocol enrichment (B1) — interfaces freeze at 1.0
- Pending: SCN oscillator feedback (B2) — close the last open feedback loop
- Pending: SEM world enrichment Phases 2-3 (B3) — rich learning environments
- Pending: Cradle experiment (B4) — sensorimotor learning without language
- Pending: API/CLI surface review + agent memory transfer docs

## Active (top-level)

Plans on the 1.0 critical path or actively shipping:

- [v1_refinement.md](v1_refinement.md) — **1.0 release plan.** Validation + bio-system stabilization + sensorimotor grounding + pipeline completion + cleanup + docs + contract clarification (Section 7). 1.1 track index is Section 8.
- [bio_system_protocol_enrichment.md](bio_system_protocol_enrichment.md) — Future-proof bio-system interfaces with `*Context` dataclass parameters. Cheap now, expensive post-1.0.
- [scn_oscillator_feedback.md](scn_oscillator_feedback.md) — Close the SCN→NAc feedback loop. Anticipatory temporal credit. ~100-150 LOC.
- [sem_world_enrichment.md](sem_world_enrichment.md) — **Phase 1 SHIPPED.** Phases 2-3 deferred to 1.1.
- [proprioceptive_discovery.md](proprioceptive_discovery.md) — **Mechanism A SHIPPED.** Mechanism B (entity acquisition) shipped with cradle.
- [reactive_peer_mesh_roadmap.md](reactive_peer_mesh_roadmap.md) — Living roadmap. C3-C4.6 complete. C5+ remain. Not gating 1.0.

### 1.1 track (concurrent development)

- [scene_actor_affordances.md](scene_actor_affordances.md) — `target_effect` + OrchestratorActorTool. ~110 LOC. Diagnostic for agent-backed entities. 1.1.
- [minecraft_benchmark.md](minecraft_benchmark.md) — Live demo + harness comparison. Stub. 1.1 splash launch.
- [mcp_compatibility.md](mcp_compatibility.md) — MCP server + client + schema interop. Stub. 1.0 ships CC9 (dual-format Tool schema) as the prerequisite.

## Living practice docs

Accumulate evidence over time. Not on the critical path — ongoing scientific/operational questions.

- [behavioral_convergence_practice.md](behavioral_convergence_practice.md) — Does the agent get better across sessions? 4 experiments, 41/41 hypotheses confirmed.
- [memory_consolidation_practice.md](memory_consolidation_practice.md) — Refines P8 sleep-replay. ACTIVATED (P8 shipped 2026-04-19).
- [tool_refinement_plan.md](tool_refinement_plan.md) — Ongoing tool surface curation.

## Deferred (post-1.0, revive on trigger)

Design work preserved in [deferred/](deferred/). Each has an explicit "revive when" condition.

- [deferred/b5_embodiment_narrative_separation.md](deferred/b5_embodiment_narrative_separation.md) — Formalize SEM/DM prompt boundary. Revive when prompt-bleed bug surfaces.
- [deferred/agent_backed_entities.md](deferred/agent_backed_entities.md) — 3-tier cognition + Cradle-trained cast + mesh-pressure budget. Revive if scene_actor_affordances diagnostic doesn't close the gap.
- [deferred/goal_depth_integration.md](deferred/goal_depth_integration.md) — GOAL WMS entry kind, goal-tagged episodes. Stage 3 absorbed by temporal_credit_integration. Remaining stages are enrichment, not gating.
- [deferred/bio_system_plugin_plan.md](deferred/bio_system_plugin_plan.md) — Plugin discovery for bio-systems. Revive when external contributors appear.
- [deferred/mother_npc_stimulus_plan.md](deferred/mother_npc_stimulus_plan.md) — Two-agent stimulus pattern. Revive when behavioral convergence hits variety ceiling.
- [deferred/pecking_order_graph_plan.md](deferred/pecking_order_graph_plan.md) — Unified hierarchy DAG. Revive when multi-node topology matters.
- [deferred/mother_maxim_plan.md](deferred/mother_maxim_plan.md) — Persistent collective memory. Revive when P8 convergence + external users.
- [deferred/dungeon_master_extensions.md](deferred/dungeon_master_extensions.md) — DM post-MVP features.
- [deferred/cross_platform_file_lock.md](deferred/cross_platform_file_lock.md) — Unify `process_lock` + `filelock`. Tech debt, blocks nothing.
- [deferred/mesh_doc_transport.md](deferred/mesh_doc_transport.md) — Mesh-to-mesh doc exchange (C9). Not started.
- [deferred/pain_bus_bridge_subscriber_unification.md](deferred/pain_bus_bridge_subscriber_unification.md) — Bridge/subscriber attribution-asymmetry fix. Monitor; open when pending-event context enriched.
- [deferred/node_security_simplification.md](deferred/node_security_simplification.md) — Phase 1 shipped. Phase 2 config unification deferred.
- [deferred/llm_path_multi_peer_dispatch.md](deferred/llm_path_multi_peer_dispatch.md) — Multi-peer load distribution. Revive when 2+ GPU nodes.
- [deferred/llm_mesh_capability_aware.md](deferred/llm_mesh_capability_aware.md) — Capability-aware routing. Revive when heterogeneous mesh.
- [deferred/llm_path_async_router.md](deferred/llm_path_async_router.md) — Async router. Revive if head-of-line blocking observed.
- [deferred/llm_path_fair_scheduling.md](deferred/llm_path_fair_scheduling.md) — Fair-share scheduling. Revive if rate limiting insufficient.

## Archive

Completed or superseded plans live in [archive/](archive/).

Recently archived (2026-04-25 plans audit + stall recovery verified shipped):
- [archive/affordance_concept_transfer.md](archive/affordance_concept_transfer.md) — **✅ Stages 0-4 SHIPPED + ARCHIVED.** Cross-entity learning, SCN temporal coupling.
- [archive/component_level_damage.md](archive/component_level_damage.md) — **✅ Stages 1-5 SHIPPED + ARCHIVED.** Per-modulator damage, `--deep-embodiment`. Stage 6 deferred.
- [archive/percept_reflex_system.md](archive/percept_reflex_system.md) — **✅ SHIPPED + ARCHIVED.** ReflexRegistry, wired into BioEnrichmentPipeline.
- [archive/sem_entity_ownership.md](archive/sem_entity_ownership.md) — **✅ SHIPPED + ARCHIVED.** Self/scene separation, discovery filter.
- [archive/orchestrator_sem_damage.md](archive/orchestrator_sem_damage.md) — **✅ SHIPPED + ARCHIVED.** Auto-damage + pain cascade. PR #190.
- [archive/deliberative_thought_stream.md](archive/deliberative_thought_stream.md) — **✅ Stages 1+2 SHIPPED + ARCHIVED.** Stages 3/3b/4 absorbed into temporal_credit_integration.
- [archive/temporal_credit_integration.md](archive/temporal_credit_integration.md) — **✅ Phases 1-7 SHIPPED + ARCHIVED.** TemporalEvent, Distributor, ValenceSignal, goal tags.
- [archive/gating_abstraction.md](archive/gating_abstraction.md) — **✅ G0+G1 SHIPPED + ARCHIVED.** SalienceScorer, AdaptiveThresholdController. G2/G3 deferred.
- [archive/agent_factory_canonicalization.md](archive/agent_factory_canonicalization.md) — **✅ F1-F5 SHIPPED + ARCHIVED.** Wave G deferred.
- [archive/llm_path_refinement.md](archive/llm_path_refinement.md) — **✅ Plans 1-4 ALL SHIPPED + ARCHIVED.** Meta-index. All children in archive.
- [archive/substrate_episode_boundary_enrichment.md](archive/substrate_episode_boundary_enrichment.md) — **✅ Stages 1+3 SHIPPED + ARCHIVED.** Stage 2 absorbed into v1_refinement.md.
- [archive/cross_session_sim_validation.md](archive/cross_session_sim_validation.md) — **ABSORBED into v1_refinement.md Section 1.**
- [archive/tool_pain_bridge_temporal_migration.md](archive/tool_pain_bridge_temporal_migration.md) — **ABSORBED into v1_refinement.md Section 2.**
- [archive/orchestrator_stall_recovery.md](archive/orchestrator_stall_recovery.md) — **✅ SHIPPED (PR #181, 2026-04-23).** Content-aware tool cap + feedback + diversity injection. Three-layer defense.

Previously archived (2026-04-23):
- [archive/pfc_deliberation_cycle.md](archive/pfc_deliberation_cycle.md), [archive/interactive_display_overhaul.md](archive/interactive_display_overhaul.md), [archive/working_memory_exec_loop.md](archive/working_memory_exec_loop.md), [archive/substrate_p5_stress_persistence.md](archive/substrate_p5_stress_persistence.md), [archive/concept_exploration.md](archive/concept_exploration.md), [archive/biosystem_unification.md](archive/biosystem_unification.md), [archive/llm_path_operator_visibility.md](archive/llm_path_operator_visibility.md), [archive/deliberation_observability.md](archive/deliberation_observability.md), [archive/circular_import_resolution.md](archive/circular_import_resolution.md), [archive/asset_foundry_plan.md](archive/asset_foundry_plan.md)

Previously archived (2026-04-19):
- [archive/prompt_b4_replanning.md](archive/prompt_b4_replanning.md), [archive/substrate_p6_extinction.md](archive/substrate_p6_extinction.md), [archive/substrate_p8_sleep_replay.md](archive/substrate_p8_sleep_replay.md), [archive/agent_loop_state_repair.md](archive/agent_loop_state_repair.md), [archive/interactive_experience_031.md](archive/interactive_experience_031.md), [archive/peer_update_pip_mode.md](archive/peer_update_pip_mode.md), [archive/substrate_concept_decomposition.md](archive/substrate_concept_decomposition.md), [archive/game_npc_integration.md](archive/game_npc_integration.md), [archive/llm_path_peer_failover.md](archive/llm_path_peer_failover.md)

Previously archived (2026-04-17):
- [archive/sem_learning_loop.md](archive/sem_learning_loop.md), [archive/behavioral_convergence_wiring.md](archive/behavioral_convergence_wiring.md), [archive/cerebellum_activation.md](archive/cerebellum_activation.md), [archive/substrate_valence_annotation.md](archive/substrate_valence_annotation.md), [archive/substrate_binding_persistence.md](archive/substrate_binding_persistence.md), [archive/substrate_binding_split_proposal.md](archive/substrate_binding_split_proposal.md), [archive/bio_stack_unification.md](archive/bio_stack_unification.md), [archive/router_drain_coupling.md](archive/router_drain_coupling.md), [archive/auto_drain_persistent_failure.md](archive/auto_drain_persistent_failure.md)

Previously archived (biosystem unification, 2026-04-14/16):
- [archive/executor_bootstrap_unification.md](archive/executor_bootstrap_unification.md), [archive/pain_bus_unification.md](archive/pain_bus_unification.md), [archive/reaction_bus_unification.md](archive/reaction_bus_unification.md), [archive/memory_hub_unification.md](archive/memory_hub_unification.md), [archive/default_network_unification.md](archive/default_network_unification.md)

Previously archived (LLM path, 2026-04-12/13):
- [archive/llm_path_foundation.md](archive/llm_path_foundation.md), [archive/llm_path_typed_errors.md](archive/llm_path_typed_errors.md), [archive/llm_path_fast_failover.md](archive/llm_path_fast_failover.md), [archive/llm_path_cancellation_hygiene.md](archive/llm_path_cancellation_hygiene.md)

Earlier archives (foundations, cleanup, peer/leader, simulator, substrate):
- [archive/foundations_plan.md](archive/foundations_plan.md), [archive/cleanup_wave.md](archive/cleanup_wave.md), [archive/peer_leader_flexibility_plan.md](archive/peer_leader_flexibility_plan.md), [archive/unified_event_bus_plan.md](archive/unified_event_bus_plan.md), [archive/simulator_upgrades_plan.md](archive/simulator_upgrades_plan.md), [archive/reaction_abstraction_plan.md](archive/reaction_abstraction_plan.md), [archive/substrate_plan.md](archive/substrate_plan.md), [archive/substrate_recognition.md](archive/substrate_recognition.md), [archive/substrate_p0_pilot.md](archive/substrate_p0_pilot.md), [archive/substrate_p3a_episode_binding.md](archive/substrate_p3a_episode_binding.md), [archive/substrate_p3b_channel_integration.md](archive/substrate_p3b_channel_integration.md), [archive/substrate_p3_5_persistence_snapshot.md](archive/substrate_p3_5_persistence_snapshot.md), [archive/substrate_p4_cross_modal_binding.md](archive/substrate_p4_cross_modal_binding.md), [archive/substrate_p4_option2_measurement.md](archive/substrate_p4_option2_measurement.md), [archive/sem_execution_hook.md](archive/sem_execution_hook.md), [archive/07_feature_completion.md](archive/07_feature_completion.md), [archive/sem_tool_discovery.md](archive/sem_tool_discovery.md), [archive/foundational_buildout_plan.md](archive/foundational_buildout_plan.md), [archive/api_surface_hardening_plan.md](archive/api_surface_hardening_plan.md), [archive/embodiment_core_plan.md](archive/embodiment_core_plan.md)

## What 0.3 proved

The 0.3 release demonstrates the core 1.0 claim at prototype scale:

1. **Tier 1 (substrate):** Bio-systems learn affective associations and persist them across sessions (Exp 1: 11/11, Exp 2: 13/13)
2. **Tier 2 (LLM reads learning):** The LLM makes different decisions when it sees the agent's learned valence (Exp 3: 12/12)
3. **Tier 3 (organic learning):** The agent learns from its own actions without scripted training (Exp 4: 5/5, teal rate 0% -> 25% -> 100%)

All 41/41 hypotheses confirmed. No fine-tuning. No prompt engineering beyond surfacing the substrate's learned associations.

## Review pattern

Each plan implements on a `feat/<plan>` branch. Before opening/merging the PR, spawn two review Claudes (Executor + Architecture lens). Findings fold into the same branch, THEN the PR opens. See [feedback_review_before_ship.md](../../.claude/projects/-Users-dennyschaedig-Scripts-Maxim/memory/feedback_review_before_ship.md).

## Rules for this directory

- **Active plans stay in the root.** Anything in the root is on the critical path or actively shipping.
- **Deferred plans must state a revive trigger.** No trigger = archive, not deferred.
- **No ghost plans.** If a plan references a module that doesn't exist, fix or delete the reference.
- **Merge before multiplying.** If two plans overlap by more than a phase, merge them.
