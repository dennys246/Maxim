# Maxim Plans

Current version: **0.2.1** (published on PyPI as `pymaxim`).
Target: **1.0** — cross-session learning demonstrated without LLM fine-tuning.

## Active (gating 1.0)

Foundations, reaction abstraction, simulator upgrades, P0 pilot, P1 recognition, P2 reward modulation, P3a episode binding, P3b channel integration, P3.5 persistence/snapshot, SEM execution hook, and **P4 cross-modal binding** are all complete. **All four 0.3-target substrate phases (P3a, P3b, P3.5, P4) are CLOSED. 0.3-target gate is CLOSED.**

- [substrate_binding_persistence.md](substrate_binding_persistence.md) — **SPLITTING IN PROGRESS** (2026-04-14, updated 2026-04-16). P3a → P8 + B3-B5 monolith. **All four 0.3-target phases CLOSED** (P3a, P3b Stage 1, P3.5 Stages 1+2, P4). Per-phase plan files for P5/P6/P8/B3-B5 can now open.
- [substrate_binding_split_proposal.md](substrate_binding_split_proposal.md) — **PROPOSAL** (2026-04-14). Narrative proposing the 8-file split of substrate_binding_persistence.md. Read before creating any of the remaining per-phase plan files (P5/P6/P8).
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

- [behavioral_convergence_practice.md](behavioral_convergence_practice.md) — does the agent actually get better across sessions? Pure living doc — no mechanism to ship, just hypotheses, scenarios, and results. Kicks in when P1 is green.
- [memory_consolidation_practice.md](memory_consolidation_practice.md) — refines the P8 sleep-replay mechanism with alternative strategies, promotion rules, interference analysis. Kicks in when P8 ships in 0.5 — needs the mechanism to exist before the practice has anything to refine.

## Parallel (ship anytime, not gating 1.0)

- [substrate_concept_decomposition.md](substrate_concept_decomposition.md) — **SHELL, TRIGGER FIRED** (2026-04-15, three-lens review folded 2026-04-16). Protocol-based noun-phrase extraction (`DecompositionStrategy` + `ConceptChunk`) at the `LinguisticEncoder` layer. spaCy default strategy, text-modality-only gate, config-driven enable. **P4 Stage 3 PASSED — unblocked.** Ideal moment: between P4 and P5 stress test. **Recommended sequence: concept decomposition → valence annotation → P5 stress test.**
- [substrate_valence_annotation.md](substrate_valence_annotation.md) — **SHELL** (2026-04-16). SEM reactions (pain/pleasure) annotate Hebbian edges with valence via `Edge.metadata["valence"]`. Fixes two broken links: (1) `CerebellumModulator._reaction_bus` never wired, (2) `PendingEpisodeState` has no reactions field. Spreading activation optionally propagates valence. Stage 3 is a simulation PoC (rusty sword pain scenario). **Recommended after concept decomposition Stage 1** — finer-grained nodes make valence targets more precise.
- [archive/substrate_p4_option2_measurement.md](archive/substrate_p4_option2_measurement.md) — **✅ COMPLETE + ARCHIVED** (2026-04-16). Option 2 lift = 0.0000 ± 0.0000. Decision: defer. Results: [experiments/p4_option2_measurement.md](../experiments/p4_option2_measurement.md).
- [substrate_episode_boundary_enrichment.md](substrate_episode_boundary_enrichment.md) — **SHELL, TRIGGER FIRED** (2026-04-15, trigger fired 2026-04-16). Three new episode boundary rules extending the existing pluggable detector: (1) tool execution boundary, (2) semantic shift detection (cosine divergence from running episode centroid), (3) pain/salience spike. Complementary to concept decomposition — finer-grained nodes need bounded episodes to keep Hebbian edge count manageable. **P4 Stage 3 PASSED — this plan is now unblocked.** Ideal moment: alongside concept decomposition, before P5 stress test.
- [biosystem_unification.md](biosystem_unification.md) — **central tracking doc** (2026-04-14, updated 2026-04-17) for the bio-system structural-enforcement work. Waves 0-2 **ALL SHIPPED + ARCHIVED**. Wave 3 (bio_stack umbrella) committed on `feat/bio-stack-unification`, PR pending. Wave 4 (agent_factory_canonicalization) not scheduled.
  - [archive/executor_bootstrap_unification.md](archive/executor_bootstrap_unification.md) — ✅ Wave 0 SHIPPED (PR #114). `build_executor(*, pain_bus)`.
  - [archive/pain_bus_unification.md](archive/pain_bus_unification.md) — ✅ Wave 1 SHIPPED (PR #125). `build_pain_bus(*, hippocampus, nac)`.
  - [archive/reaction_bus_unification.md](archive/reaction_bus_unification.md) — ✅ Wave 1 SHIPPED (PR #134). `build_reaction_bus(*)`.
  - [archive/memory_hub_unification.md](archive/memory_hub_unification.md) — ✅ Wave 2 SHIPPED (PR #136). `build_memory_hub(*, hippocampus, scn, nac, ec)`.
  - [archive/default_network_unification.md](archive/default_network_unification.md) — ✅ Wave 2 SHIPPED (PR #135). `build_default_network(*, nac)`.
  - [bio_stack_unification.md](bio_stack_unification.md) — Wave 3, committed on branch. `build_bio_stack(*, persistence_dir)`. 4 sites migrated.
- [tool_refinement_plan.md](tool_refinement_plan.md) — living doc for agent tool surface curation
- [agent_factory_canonicalization.md](agent_factory_canonicalization.md) — **RUNNING DOC, not scheduled** (2026-04-14). The Option D follow-up to `executor_bootstrap_unification.md` — make `AgentFactory.create_agent` the only door for constructing an agent in Maxim. Becomes a downhill rewrite once Wave 3 `build_bio_stack` merges. Subsumes `sem_execution_hook.md` Stage 2b. Trigger conditions documented inline.
- [node_security_simplification.md](node_security_simplification.md) — Phase 1 immediate security fixes (timing-safe auth comparison, rate-limiter bucket key, help-text corrections). Phase 2 config-surface unification deferred after Plan 4.
- [llm_path_refinement.md](llm_path_refinement.md) — meta-plan for the LLM routing path refactor. Motivated by two 2026-04-12 peer-leader incidents + an audit that revealed `_OpenAIBackend` has a hidden ~52s retry loop. **Ships as the 0.4 stability version.** Plans 1, 2, 3, 3.5 fully shipped and archived; Plan 3.6 R5 (VRAM spillover detection) shipped; Plan 4 Stages A+B (agent_id observability + recovery-time bench) shipped; **Plan 4 Stage C1+C2+C3.1 shipped** (`mesh.yml` schema + `list-nodes` + drain/resume + `init-mesh`); **C3.2 (`add-node`/`remove-node`) in PR**; only the rest of C3 (admin API, --node install, per-agent rate limiting, request-trace ring buffer, cluster key rotation) remains in scope. Authoritative architecture reference at [../architecture/llm_routing.md](../architecture/llm_routing.md); stress test protocol at [../experiments/protocols/llm_path_stress_test.md](../experiments/protocols/llm_path_stress_test.md).
  - **✅ Plan 1 (Foundation) — SHIPPED, ARCHIVED** → [archive/llm_path_foundation.md](archive/llm_path_foundation.md). R0 deleted ~1,250 LOC dead mesh (commit `e811787`); R1 shipped `maxim/utils/http.py` with endpoint registry + typed `HTTPError` + `RequestContext` contextvars + `X-Maxim-*` header propagation (PRs #88, #90, #91). See [project_llm_path_r1_shipped.md](../../.claude/projects/-Users-dennyschaedig-Scripts-Maxim/memory/project_llm_path_r1_shipped.md).
  - **✅ Plan 2 (Typed Errors + Role Detection) — SHIPPED, ARCHIVED** → [archive/llm_path_typed_errors.md](archive/llm_path_typed_errors.md). R2a-d: role detection at CLI boot, typed `BackendError` hierarchy with `.fix_hint`, two-stage probe, SSRF moved to `utils/net.py` (PRs #92, #93). See [project_llm_path_r2_shipped.md](../../.claude/projects/-Users-dennyschaedig-Scripts-Maxim/memory/project_llm_path_r2_shipped.md).
  - **✅ Plan 3 (Fast Failover) — SHIPPED, ARCHIVED** → [archive/llm_path_fast_failover.md](archive/llm_path_fast_failover.md). R2.5 `_MaximPeerBackend` purpose-built single-HTTP-call backend + router typed-exception dispatch + `BACKEND_CLASSES`; R2.6 probe consolidation. **The 52s fail-slow is dead.** PR #94, commit `ce5f034`. Programmatic gate: < 5s p99 against mocked-dead-peer fixture. See [project_llm_path_r3_shipped.md](../../.claude/projects/-Users-dennyschaedig-Scripts-Maxim/memory/project_llm_path_r3_shipped.md) for the 10 load-bearing invariants.
  - **✅ Plan 3.5 (Cancellation Hygiene) — SHIPPED, ARCHIVED** → [archive/llm_path_cancellation_hygiene.md](archive/llm_path_cancellation_hygiene.md). R1-R6: cooperative cancellation primitives in `maxim/utils/cancellation.py` + "HTTP fires first" timeout contract (HTTP authoritative at 300s, agent layer strict safety net above). PR #96, commit `6a4f505`. See [project_llm_path_cancellation_hygiene_shipped.md](../../.claude/projects/-Users-dennyschaedig-Scripts-Maxim/memory/project_llm_path_cancellation_hygiene_shipped.md).
  - [llm_path_peer_failover.md](llm_path_peer_failover.md) — **Plan 3.6: Peer Failover — PARTIAL SHIP (2026-04-14).** **R5 VRAM spillover detection ✅ SHIPPED** (PR #99, commit `2884e58`): doctor `check_vram_pressure` + spawn-time `_check_vram_spillover_risk` + shared `project_vram_usage` math + fix for pre-existing `check_llm_model_active` mutable-global bug. Dynamic headroom `max(1.5, 0.55 × weights_gb)` calibrated to the 2026-04-13 incident. R1–R4 (multi-leader `peer.yml`) **remain draft** — on hold until the user's second GPU comes online. See [project_vram_spillover_detection_shipped.md](../../.claude/projects/-Users-dennyschaedig-Scripts-Maxim/memory/project_vram_spillover_detection_shipped.md) for the 5 R5 load-bearing invariants.
  - [llm_path_operator_visibility.md](llm_path_operator_visibility.md) — **Plan 4: Operator Visibility — PARTIAL SHIP (2026-04-14).** Split into three sequential stages:
    - **✅ Stage A — agent_id observability fix** (PR in review on `feat/llm-path-operator-visibility`). Three complementary changes close the Phase D observability gap: router capability-flag kwarg forwarding, `set_context` boundary binding in `LLMWorker._call_llm_with_timeout`, and contextvar fallback in `_normalize_request_context`. 11 new regression tests.
    - **✅ Stage B — recovery-time bench harness** (same PR). New `maxim bench recovery-time` CLI subcommand at `src/maxim/bench/` (NOT `benchmark/` — name collision with `maxim.api.benchmark` public verb). Uses `_MaximPeerBackend` directly to measure peer recovery without sim-cadence workload artifacts. 21 new tests. **Phase D2 hardware validation:** 58.68s recovery window on real RTX 5080 (matches 53s leader self-report + ~5s proxy gap), 750/750 `agent_id` coverage, 199/199 typed `BackendDown` failures, fast-fail p99=614ms. See [llm_path_stress_plan4_20260414.md](../experiments/results/llm_path_stress_plan4_20260414.md) and [bench_recovery_time_rerun.md](../experiments/protocols/bench_recovery_time_rerun.md). See [project_llm_path_operator_visibility_ab_shipped.md](../../.claude/projects/-Users-dennyschaedig-Scripts-Maxim/memory/project_llm_path_operator_visibility_ab_shipped.md) for the 8 load-bearing invariants.
    - **✅ Stage C1 — `mesh.yml` schema + `list-nodes` + `--node status|health`** (PR #108, merged 2026-04-14). Hand-rolled `mesh.yml` parser (FROZEN dialect — no PyYAML), `peer/probe_classify.py` shared classifier (single source of truth for probe outcome → CheckResult mapping across mesh_cli + doctor), peer.yml→mesh fallback for zero-breaking-change. 2 review rounds caught 31 findings incl. silent `cluster_key` `#` truncation. See [project_plan4_c1_shipped.md](../../.claude/projects/-Users-dennyschaedig-Scripts-Maxim/memory/project_plan4_c1_shipped.md).
    - **✅ Stage C2 — drain/resume with runtime state layer** (PR #113, merged 2026-04-14). `~/.maxim/util/drained_nodes.{role}.txt` with `filelock.FileLock` cross-process serialization. Pivoted from Option A1 (config-only drain + TOML migration) to Option B (runtime state) after pre-design review caught 3 criticals. Pre-merge review caught 15 findings incl. 1 triple-confirmed orphan retry_id bug. Renamed in-house `maxim/utils/filelock.py` → `process_lock.py` to avoid name collision with 3rd-party. New `atomic_write_secret` wrapper. See [project_plan4_c2_shipped.md](../../.claude/projects/-Users-dennyschaedig-Scripts-Maxim/memory/project_plan4_c2_shipped.md).
    - **✅ Stage C3.1 — `init-mesh` verb + `mesh.yml` writer infrastructure** (PR #118, merged 2026-04-14). `MeshConfig.to_yaml()` round-trip serializer + `write_mesh_config()` disk-I/O wrapper using `atomic_write_secret`. Strict CI grep allow-list enforces `write_mesh_config` callers (only `mesh_setup.py` + tests). Pre-merge review caught 14 findings incl. 6 cross-confirmed. See `project_plan4_c3.1_shipped.md` (will be added to memory).
    - **🚧 Stage C3.2 — `add-node` + `remove-node` verbs** (PR pending, branch `feat/plan4-c3.2-add-remove-node`, fold complete). Closes the gap C3.1 left open: operators can now grow/shrink `mesh.yml::nodes` from the CLI without hand-editing. Renamed `init_mesh.py` → `mesh_setup.py` to group the 3 setup verbs. `MeshConfig.__post_init__` now validates `self_name in nodes` (hoisted from parser per A1 cross-confirmed fold). Pre-merge review caught 16 findings incl. 4 cross-confirmed. See `project_plan4_c3.2_shipped.md` (will be added to memory).
    - **Stage C3 remaining (DEFERRED):** `--node install` + VRAM precheck, `--node refresh`, `/v1/mesh/*` admin API, per-agent rate limiting, request-trace ring buffer, cluster key rotation. The full scope is still in [llm_path_operator_visibility.md](llm_path_operator_visibility.md) under "Phases".
  - Deferred shell plans (revive on stress-test-defined triggers):
    - [deferred/llm_path_multi_peer_dispatch.md](deferred/llm_path_multi_peer_dispatch.md) — multi-peer reactive overflow with rendezvous-hash distribution. **Partially triggered (2026-04-13)** by the user's RTX 3070 hardware; awaiting Plan 3.6 R1-R4 + Plan 4 Stage C ship.
    - [deferred/llm_mesh_capability_aware.md](deferred/llm_mesh_capability_aware.md) — capability advertisement + capability-aware router ranking. Revive when ≥2 nodes serve **different** loaded models.
    - [deferred/llm_path_async_router.md](deferred/llm_path_async_router.md) — async router if `_inference_lock` becomes the bottleneck
    - [deferred/llm_path_fair_scheduling.md](deferred/llm_path_fair_scheduling.md) — bio-inspired priority classes + fair-share

  **Long-term mesh roadmap** (current state → true reactive mesh): see the "Long-term roadmap" section in [llm_path_refinement.md](llm_path_refinement.md). Five concrete steps from leader/peer to peer-to-peer mesh with leader election.

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

Recently archived (biosystem unification Waves 0-2 shipped 2026-04-14/16):
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

Two tracks run in parallel:
- **Track A — Substrate:** the bio-inspired research claim. F0 → P0 → P1 → P2 → P3a → P3b → P3.5 → P4 → P5 → P6 → P8.
- **Track B — Prompt layer:** B1 → B3 → B4 → B5.
- **Track C — Infrastructure (2026-04-12, mostly shipped 2026-04-12/14):** LLM path refinement. Plans 1, 2, 3, 3.5 shipped and archived; Plan 3.6 R5 + Plan 4 Stage A+B shipped; substrate P2 Stage 3 shipped (stress phase A). Plan 4 Stage C + stress phases B/C/E remain. Ships as 0.4 stability version.

Track C is a pause-insertion between Track A's 0.3 and Track B's 0.4 because the 2026-04-12 peer-leader incidents + `_OpenAIBackend` retry-loop discovery made it clear the substrate work cannot be reliably stress-tested on the current LLM path.

Each substrate phase is a falsifiable claim validated with mechanistic criteria where the phase tests a mechanism, and head-to-head gate baselines where the baseline attacks the same claim (P3a TF-IDF, P4 OpenCLIP, P6 LRU). Pass criteria use effect sizes across ≥10 seeds (≥20 for P4); no p-values, no Bonferroni corrections. Persistence round-trip smoke tests fire at every phase.

| Version | What ships | What it proves |
|---|---|---|
| **0.2.2** | Cleanup Wave | Friction removed from the surface B1+P1 will rewrite |
| **0.3-pre** | foundations_plan, simulator_upgrades_plan, P0 pilot, B1+P1 combined migration | Foundations solid; substrate phases cheap to run; fixtures calibrated; text flows through percepts end-to-end |
| **0.3-minimum** | 0.3-pre plus P1, P2, P3.5 | Mechanism + reward modulation + persistence certification. Defensible version bump if P3a/b/P4 slip to 0.3.1. |
| **0.3-target** | 0.3-minimum plus P3a, P3b, P4 (OpenCLIP head-to-head) | Full substrate proven with cross-modal binding across real process boundary |
| **0.4 (Track C — stability)** | **LLM path refinement Plans 1–3.5 SHIPPED** (archived); Plan 3.6 R5 SHIPPED; Plan 4 Stage A+B SHIPPED; **substrate P2 Stage 3 SHIPPED** (real-embedding sweep PASS); Plan 4 Stage C + remaining stress phases (B/C/E) + `llama.cpp --parallel` batching PoC REMAINING | Infrastructure reliably supports multi-agent stress testing. `maxim peer restart` recovers in ~58s end-to-end on real hardware (peer-side overhead ≈ 0s, dominated by leader's 53s model reload). Per-agent observability via `agent_id` on every `peer_backend_call`/`peer_backend_failed` event. Rigorous recovery-time measurable via `maxim bench recovery-time`. Substrate P2 reward modulation validated on real embeddings at +56 pp target gain. See [llm_path_refinement.md](llm_path_refinement.md) + [substrate_recognition.md](substrate_recognition.md). |
| **0.5 (formerly 0.4)** | P4 re-pass (production vision + email/Slack), B3, B4 (gates 1.0), B5 | Architecture generalizes; NPCs coherent; replanning recovers from failure |
| **0.6 (formerly 0.5)** | P5 (stress persistence), P6 (extinction vs LRU), **P8 (minimum-viable sleep replay)** | Persists under load, forgets appropriately, actively strengthens rewarded associations offline |
| **1.0** | Stress-test sim combining all phases; B4 passing; practice docs with experiments logged | Cross-session learning without fine-tuning at realistic scale, with coherent voice, with ongoing research program |

**0.3-minimum vs 0.3-target:** a partial 0.3 can ship as a version bump if the ambitious target slips. Normal re-planning, not failure.

**0.4 is a pure infrastructure version bump.** No new substrate phases. No new prompt-layer features. It exists because the 2026-04-12 incidents + architecture audit made stability work non-optional. Substrate work continues in 0.5 on top of the stabilized LLM path.

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

7. **▶ NEXT (substrate track resumes):** Substrate P2 Stage 3 **COMPLETE (2026-04-14)** — `substrate_recognition.md` closed. Remaining stress-test phases B, C, E + Plan 4 Stage C can run in any order when their respective bottlenecks surface. Next substrate plan: [substrate_binding_persistence.md](substrate_binding_persistence.md) (P3a onward).

8. **Release 0.4** with LLM path refinement complete + substrate P2 validated on real embeddings. Substrate P2 gate is CLOSED; 0.4 ships when the remaining stress phases (B, C, E) + Plan 4 Stage C are ready.

9. **Back to substrate work:** P3a + P3b + P4 in 0.5, built on the stabilized LLM path. Plan doc: [substrate_binding_persistence.md](substrate_binding_persistence.md).

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
