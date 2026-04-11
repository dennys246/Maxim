# Maxim Plans

Current version: **0.2.1** (published on PyPI as `pymaxim`).
Target: **1.0** — cross-session learning demonstrated without LLM fine-tuning.

## Active (gating 1.0)

Three plans run in sequence before substrate phases start. Each is its own wave with its own gate. The sequence is Cleanup Wave → foundations_plan → simulator_upgrades_plan → P0 pilot → substrate phases.

- [foundations_plan.md](foundations_plan.md) — F0.1–F0.8 prerequisites (NAc wiring + save/load signature, PerceptTraceBuffer, NarrativeModulator ghost, Percept schema, agent_id threading + SCN race fix, factory consolidation, tier assertions, Sensor→Percept contract). Blocks simulator_upgrades_plan and substrate. ~1,130 LOC across eight PRs.
- [simulator_upgrades_plan.md](simulator_upgrades_plan.md) — S1–S4 test-harness infrastructure: fixture-driven orchestrator, **LLMBackend Protocol + MockLLMBackend** (Option B — S2 defines the Protocol itself as the first step because no formal protocol exists today), subprocess persistence harness, deterministic seeding CLI. Blocks substrate P0. ~850 LOC across four PRs. Drops substrate per-phase harness cost from ~200 LOC to ~100 LOC by leveraging existing `ConversationalSource`/`ScenarioSource`/`BenchmarkRunner` infrastructure instead of building bespoke harnesses.
- [substrate_plan.md](substrate_plan.md) — bio-stack convergence (Track A: P0, P1–P6, P8) + prompt layer (Track B: B1–B5, merged from `embodiment_voice_plan.md`). Includes P0 fixture-difficulty pilot, persistence as cross-phase contract, minimum-viable sleep replay (P8), sim-as-fixture-debugger workflow, 0.3-minimum vs 0.3-target fallback, incremental contracts layer, and living-doc discipline. Depends on foundations_plan + simulator_upgrades_plan landing first.

The prompt-layer plan was merged into substrate because B1's PromptAssembler and P1's text-to-prompt migration touch the same files. The foundations and simulator upgrades plans were split out because those items are prerequisite waves, not proof-obligation phases — keeping them together diluted all three documents.

## Living practice docs (pair with substrate_plan)

These accumulate evidence and refinement over time. They are not on the critical path to 1.0; they exist because the questions they address are scientific/ongoing, not engineering milestones.

- [behavioral_convergence_practice.md](behavioral_convergence_practice.md) — does the agent actually get better across sessions? Pure living doc — no mechanism to ship, just hypotheses, scenarios, and results. Kicks in when P1 is green.
- [memory_consolidation_practice.md](memory_consolidation_practice.md) — refines the P8 sleep-replay mechanism with alternative strategies, promotion rules, interference analysis. Kicks in when P8 ships in 0.5 — needs the mechanism to exist before the practice has anything to refine.

## Parallel (ship anytime, not gating 1.0)

- [cleanup_wave.md](cleanup_wave.md) — fix `--interactive`, delete dead flags, display defaults, agent permissions
- [peer_leader_flexibility_plan.md](peer_leader_flexibility_plan.md) — `--llm` precedence, Apple Silicon tier detection, graceful leader-down fallback, auto-download on first use
- [tool_refinement_plan.md](tool_refinement_plan.md) — living doc for agent tool surface curation

## Deferred (post-1.0, revive on trigger)

Design work is preserved in [deferred/](deferred/). Each plan has an explicit "revive when" condition at the top.

- [deferred/bio_system_plugin_plan.md](deferred/bio_system_plugin_plan.md) — plugin discovery for bio-systems (extends the `maxim.robots` pattern). Depends on the `BioSystem` Protocol landing incrementally during substrate work. Revive when external contributors want to add bio-systems or a research collaborator needs substrate A/B testing.
- [deferred/unified_event_bus_plan.md](deferred/unified_event_bus_plan.md) — consolidate the current five event transports (`LocalMessageBus`, `AgentBus`, `ConversationalSource`, direct cross-layer callbacks, `MemoryHub`) into one typed-topic bus. Protocol defined during substrate contracts-layer work; the 3–5 week refactor is deferred until a concrete trigger (cross-layer debugging pain, observability needs, external contributor friction).
- [deferred/mother_npc_stimulus_plan.md](deferred/mother_npc_stimulus_plan.md) — **needs heavy refinement.** Two-agent stimulus pattern: Baby Maxim is the AUT with frozen LLM and evolving substrate; Mother NPC is a separate agent with her own LLM that produces realistic, varied percepts Baby learns from. Interaction is percepts only, zero information leak beyond that surface. Gives behavioral convergence experiments scalable stimulus variety without breaking the "no fine-tuning" research claim. Revive when [behavioral_convergence_practice.md](behavioral_convergence_practice.md) has ≥2 successful experiments + 1 blocked-on-variety. Isolation leak vector list in the plan is a starting point, not a contract — heavy refinement needed at revive time.
- [deferred/pecking_order_graph_plan.md](deferred/pecking_order_graph_plan.md) — unified hierarchy DAG
- [deferred/mother_maxim_plan.md](deferred/mother_maxim_plan.md) — persistent collective memory
- [deferred/asset_foundry_plan.md](deferred/asset_foundry_plan.md) — automated SEM component generation
- [deferred/dungeon_master_extensions.md](deferred/dungeon_master_extensions.md) — DM post-MVP features

## Archive

Completed or superseded plans live in [archive/](archive/). See the archive for historical context on pre-publication refinement, repo management, and the v1 versions of plans folded into the active spine.

## Version path to 1.0

Track A runs substrate (F0 → P0 → P1 → P2 → P3a → P3b → P3.5 → P4 → P5 → P6 → P8). Track B runs the prompt layer (B1 → B3 → B4 → B5), interleaved with Track A. Each phase is a falsifiable claim validated with mechanistic criteria where the phase tests a mechanism, and head-to-head gate baselines where the baseline attacks the same claim (P3a TF-IDF, P4 OpenCLIP, P6 LRU). Pass criteria use effect sizes across ≥10 seeds (≥20 for P4); no p-values, no Bonferroni corrections. Persistence round-trip smoke tests fire at every phase.

| Version | What ships | What it proves |
|---|---|---|
| **0.2.2** | Cleanup Wave | Friction removed from the surface B1+P1 will rewrite |
| **0.3-pre** | foundations_plan, simulator_upgrades_plan, P0 pilot, B1+P1 combined migration | Foundations solid; substrate phases cheap to run; fixtures calibrated; text flows through percepts end-to-end |
| **0.3-minimum** | 0.3-pre plus P1, P2, P3.5 | Mechanism + reward modulation + persistence certification. Defensible version bump if P3a/b/P4 slip to 0.3.1. |
| **0.3-target** | 0.3-minimum plus P3a, P3b, P4 (OpenCLIP head-to-head) | Full substrate proven with cross-modal binding across real process boundary |
| **0.4** | P4 re-pass (production vision + email/Slack), B3, B4 (gates 1.0), B5 | Architecture generalizes; NPCs coherent; replanning recovers from failure. B4 depends on P3a — if 0.3 shipped as 0.3-minimum, B4 slips. |
| **0.5** | P5 (stress persistence), P6 (extinction vs LRU), **P8 (minimum-viable sleep replay)** | Persists under load, forgets appropriately, actively strengthens rewarded associations offline |
| **1.0** | Stress-test sim combining all phases; B4 passing; practice docs with experiments logged | Cross-session learning without fine-tuning at realistic scale, with coherent voice, with ongoing research program |

**0.3-minimum vs 0.3-target:** a partial 0.3 can ship as a version bump if the ambitious target slips. Normal re-planning, not failure.

Channels (SMS, email, Slack, narrative speech) are **TEXT modality with context metadata**, not separate modalities. Channel rollout: SMS + narrative in 0.3, email + Slack in 0.4. See [substrate_plan.md](substrate_plan.md) for phase definitions, convergence sims, plausible baselines, negative controls, pass criteria, swap points, and fixture requirements.

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
