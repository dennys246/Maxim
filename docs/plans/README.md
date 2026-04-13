# Maxim Plans

Current version: **0.2.1** (published on PyPI as `pymaxim`).
Target: **1.0** — cross-session learning demonstrated without LLM fine-tuning.

## Active (gating 1.0)

Three focused plans remain, split from the master substrate plan. Foundations, reaction abstraction, and simulator upgrades are all complete. P0 pilot and P1 recognition are complete. P2 core is merged, P2 validation is next. The sequence is: recognition P2 validation → binding/persistence (P3a through P8 + B3-B5).

- [substrate_p0_pilot.md](substrate_p0_pilot.md) — **COMPLETE** (2026-04-12). Baseline pinned at 78.5% (mpnet@0.50). P1 sanity floor = 73.5%. Results: [experiments/p0_baseline_sweep.md](../experiments/p0_baseline_sweep.md).
- [substrate_recognition.md](substrate_recognition.md) — **in progress.** B1+P1 **SHIPPED** (2026-04-12): 91.7% ± 2.9% collapse with paraphrase-mpnet@0.40 + centroid update. P2 core merged, P2 validation remaining. Results: [experiments/p1_recognition_sweep.md](../experiments/p1_recognition_sweep.md). ~2,230 LOC. Targets 0.3-pre through 0.3-minimum.
- [substrate_binding_persistence.md](substrate_binding_persistence.md) — blocked on recognition P2 validation. P3a episode binding through P8 sleep replay + B3-B5 prompt layer. Includes the 1.0-gating P4 cross-modal head-to-head vs OpenCLIP. ~4,100 LOC. Targets 0.3-target through 0.5.

The master reference for rationale, baselines, and statistical hygiene is archived at [archive/substrate_plan.md](archive/substrate_plan.md).

## Living practice docs (pair with substrate phases)

These accumulate evidence and refinement over time. They are not on the critical path to 1.0; they exist because the questions they address are scientific/ongoing, not engineering milestones.

- [behavioral_convergence_practice.md](behavioral_convergence_practice.md) — does the agent actually get better across sessions? Pure living doc — no mechanism to ship, just hypotheses, scenarios, and results. Kicks in when P1 is green.
- [memory_consolidation_practice.md](memory_consolidation_practice.md) — refines the P8 sleep-replay mechanism with alternative strategies, promotion rules, interference analysis. Kicks in when P8 ships in 0.5 — needs the mechanism to exist before the practice has anything to refine.

## Parallel (ship anytime, not gating 1.0)

- [tool_refinement_plan.md](tool_refinement_plan.md) — living doc for agent tool surface curation
- [llm_path_refinement.md](llm_path_refinement.md) — meta-plan for the LLM routing path refactor. Motivated by two 2026-04-12 peer-leader incidents + an audit that revealed `_OpenAIBackend` has a hidden ~52s retry loop. **Ships as 0.4 stability version**, contains four focused sub-plans + three deferred shell plans. Authoritative architecture reference at [../architecture/llm_routing.md](../architecture/llm_routing.md); stress test protocol at [../experiments/protocols/llm_path_stress_test.md](../experiments/protocols/llm_path_stress_test.md).
  - [llm_path_foundation.md](llm_path_foundation.md) — **Plan 1: Foundation — R0 + R1 FULLY SHIPPED (2026-04-12).** Deleted ~1,250 LOC dead mesh scaffolding (R0, commit `e811787`). Shipped `maxim/utils/http.py` with endpoint registry, typed `HTTPError` hierarchy, `RequestContext` dataclass + contextvars, automatic `X-Maxim-*` header propagation (R1, PRs #88 + #90). Post-ship cleanup shipped via PR #91 (`fix/r1-loose-ends`): CI grep invariant now actually wired in `.github/workflows/test.yml`, `HTTPEndpoint.internal` default flipped to `False` (data-sovereignty safe-by-default), `make_http_response` test helper in `tests/conftest.py`, architecture doc drift. All 11 urllib call sites migrated. Leader updated + restarted successfully. Fast suite baseline: **4004 passed, 1 skipped** (was 4003 pre-R1). See [project_llm_path_r1_shipped.md](../../.claude/projects/-Users-dennyschaedig-Scripts-Maxim/memory/project_llm_path_r1_shipped.md) for the 5 design divergences + 10 gotchas + load-bearing invariants for Plan 2+.
  - [llm_path_typed_errors.md](llm_path_typed_errors.md) — **Plan 2: Typed Errors + Role Detection — READY TO START.** Split from Plan 1 per user decision. Role detection (`detect_role()` as first runtime action), typed `BackendError` taxonomy with `.fix_hint`, two-stage probe, SSRF check moved to `utils/net.py`. ~280 LOC new. **Note:** R2c's `_probe_stage2_readiness` + `inference_broken` outcome are already scaffolded in `runtime/llm_server.py`, gated behind `enable_stage2=False`. R2c's remaining work is the per-outcome TTL map in `probe_cache.py`, flipping `enable_stage2=True` at the `lane_backends._validate_remote_urls` call site, and the tests — not a full implementation from scratch.
  - [llm_path_fast_failover.md](llm_path_fast_failover.md) — **Plan 3: Fast Failover**. New `_MaximPeerBackend` replaces `_OpenAIBackend` for self-hosted peers (single HTTP call, no retry loop, typed exceptions, streaming). Router catches typed exceptions for per-class backoff. Probe consolidation. ~420 LOC new. **The 52-second retry loop dies here.** Pre-Plan-3 baseline: `maxim peer restart` recovery measured at **~63s on 2026-04-12** (RTX 5080 leader + Mac peer, clean probe cache). Plan 3's success criterion: drop this significantly. Stress test protocol runs substrate P2 validation + `llama.cpp --parallel` batching PoC + multi-agent fan-out as triple duty.
  - [llm_path_operator_visibility.md](llm_path_operator_visibility.md) — **Plan 4: Operator Visibility** (renamed from "Reactive Mesh"). `mesh.yml` + `maxim peer --node X` CLI + `install` command + admin API with per-agent request-trace filtering + per-agent rate limiting to prevent runaway agent starvation. ~650 LOC new. Ships unconditionally — multi-peer dispatch moved to deferred.
  - Deferred shell plans (revive on stress-test-defined triggers):
    - [deferred/llm_path_multi_peer_dispatch.md](deferred/llm_path_multi_peer_dispatch.md) — multi-peer reactive overflow if batching doesn't solve saturation
    - [deferred/llm_path_async_router.md](deferred/llm_path_async_router.md) — async router if `_inference_lock` becomes the bottleneck
    - [deferred/llm_path_fair_scheduling.md](deferred/llm_path_fair_scheduling.md) — bio-inspired priority classes + fair-share (aspiration: improve on Kubernetes quotas, not copy them)

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

Recently archived (2026-04-11/12, S1–S4 shipped 2026-04-12):
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
- **Track C — Infrastructure (new, 2026-04-12):** LLM path refinement. Four sub-plans + stress test protocol. Ships as 0.4 stability version.

Track C is a pause-insertion between Track A's 0.3 and Track B's 0.4 because the 2026-04-12 peer-leader incidents + `_OpenAIBackend` retry-loop discovery made it clear the substrate work cannot be reliably stress-tested on the current LLM path.

Each substrate phase is a falsifiable claim validated with mechanistic criteria where the phase tests a mechanism, and head-to-head gate baselines where the baseline attacks the same claim (P3a TF-IDF, P4 OpenCLIP, P6 LRU). Pass criteria use effect sizes across ≥10 seeds (≥20 for P4); no p-values, no Bonferroni corrections. Persistence round-trip smoke tests fire at every phase.

| Version | What ships | What it proves |
|---|---|---|
| **0.2.2** | Cleanup Wave | Friction removed from the surface B1+P1 will rewrite |
| **0.3-pre** | foundations_plan, simulator_upgrades_plan, P0 pilot, B1+P1 combined migration | Foundations solid; substrate phases cheap to run; fixtures calibrated; text flows through percepts end-to-end |
| **0.3-minimum** | 0.3-pre plus P1, P2, P3.5 | Mechanism + reward modulation + persistence certification. Defensible version bump if P3a/b/P4 slip to 0.3.1. |
| **0.3-target** | 0.3-minimum plus P3a, P3b, P4 (OpenCLIP head-to-head) | Full substrate proven with cross-modal binding across real process boundary |
| **0.4 (Track C — stability)** | **LLM path refinement Plans 1-4** + substrate P2 validation + stress test + `llama.cpp --parallel` batching PoC | Infrastructure reliably supports multi-agent stress testing. `maxim peer restart` recovers in < 10s (vs ~63s measured 2026-04-12). Per-agent observability. Substrate P2 validated under multi-agent load. **Plan 1 R0+R1 shipped 2026-04-12** (see line above); Plans 2-4 + stress test remaining. See [llm_path_refinement.md](llm_path_refinement.md). |
| **0.5 (formerly 0.4)** | P4 re-pass (production vision + email/Slack), B3, B4 (gates 1.0), B5 | Architecture generalizes; NPCs coherent; replanning recovers from failure |
| **0.6 (formerly 0.5)** | P5 (stress persistence), P6 (extinction vs LRU), **P8 (minimum-viable sleep replay)** | Persists under load, forgets appropriately, actively strengthens rewarded associations offline |
| **1.0** | Stress-test sim combining all phases; B4 passing; practice docs with experiments logged | Cross-session learning without fine-tuning at realistic scale, with coherent voice, with ongoing research program |

**0.3-minimum vs 0.3-target:** a partial 0.3 can ship as a version bump if the ambitious target slips. Normal re-planning, not failure.

**0.4 is a pure infrastructure version bump.** No new substrate phases. No new prompt-layer features. It exists because the 2026-04-12 incidents + architecture audit made stability work non-optional. Substrate work continues in 0.5 on top of the stabilized LLM path.

**P2 validation runs INSIDE Plan 3's stress test** (Phase A). The substrate P2 reward modulation validation happens alongside Plan 3's fast-failover verification — one stress test serves both needs. See [../experiments/protocols/llm_path_stress_test.md](../experiments/protocols/llm_path_stress_test.md).

Channels (SMS, email, Slack, narrative speech) are **TEXT modality with context metadata**, not separate modalities. Channel rollout: SMS + narrative in 0.3, email + Slack in 0.5. See [substrate_plan.md](archive/substrate_plan.md) for phase definitions.

## How LLM path refinement interleaves with substrate P2

Timeline (rough, not calendar-committed). As of 2026-04-12, step 1 is done.

1. **✅ SHIPPED (2026-04-12):** Plan 1 R0 + R1 + R1 loose ends.
   - R0: dead mesh deleted (commit `e811787`)
   - R1 core: 9-step urllib migration + `maxim/utils/http.py` (PRs #88, #90)
   - R1 cleanup: dual-format logging + docs/memory/audit pass (commits `c8a07e9`, `845af61`)
   - R1 loose ends: CI grep wired, `internal=False` default, `make_http_response` helper (PR #91, commit `3a579de`)
   - Fast suite: 4004 passed on main. Leader updated + restarted cleanly. Pre-Plan-3 restart baseline: ~63s.

2. **▶ NOW:** Plan 2 (Typed Errors + Role Detection) — correctness primitives.
   - R2a: `detect_role()` as first runtime action (memorize the subcommand-dispatch gap from R1 — `cli.py::main` early-call pattern is load-bearing)
   - R2b: `BackendError` hierarchy mirroring `HTTPError` shape from R1's `utils/http.py`
   - R2c: two-stage probe — **R2c's `_probe_stage2_readiness` + `inference_broken` outcome already pre-landed** in `runtime/llm_server.py`, gated behind `enable_stage2=False`. R2c's remaining work: wire per-outcome TTL in `probe_cache.py`, flip the gate, tests. ~40 LOC, not ~100.
   - R2d: SSRF check moved from `openai_backend.py` → `maxim/utils/net.py`

3. **Then:** Plan 3 (Fast Failover) — `_MaximPeerBackend`, kills the ~50s retry loop. Measured leader restart recovery goes from ~63s → target < 10s.

4. **Stress test (one combined run):**
   - Phase A: substrate P2 validation (satisfies 0.3-minimum P2 requirement) + Plan 3 baseline
   - Phase B: multi-agent fan-out (exercises AgentPool under the new LLM path)
   - Phase C: `llama.cpp --parallel` batching PoC (decides whether Plan 4 needs multi-peer dispatch or just visibility)
   - Phase D: leader restart recovery test (the "~50s is dead" proof — compare against the 63s 2026-04-12 baseline)
   - Phase E: fault injection (verifies typed exception coverage)

5. **Then:** Plan 4 (Operator Visibility) — CLI + admin API + per-agent rate limiting. Scope depends on Phase C outcome (scoped-only vs. scoped + multi-peer revival).

6. **Release 0.4** with LLM path refinement complete + substrate P2 validated.

7. **Back to substrate work:** P3a + P3b + P4 in 0.5, built on the stabilized LLM path.

**Review pattern per plan:** each plan ships to a `feat/` branch, then spawns two review Claudes (Executor lens + Architecture lens) in parallel read-only sessions. Findings triage into a `fix/<plan>-loose-ends` PR, merge both, then start the next plan. R1 proved this pattern works — the executor review caught the CI grep gap and the `HTTPEndpoint.internal` unsafe default; the architecture review caught doc drift. Both would have rotted into Plan 2 if not flagged.

**Why this order is load-bearing (decided 2026-04-12):**
- Current LLM infrastructure is **too broken to use for P2 testing at all.** The 52s retry loop + probe fragility + lack of per-agent observability would make P2 validation data unreliable and un-attributable ("substrate bug vs. LLM flakiness?"). Running P2 first was considered and rejected.
- Substrate P2 needs a stable LLM path to validate correctly
- Multi-agent P2 runs need per-agent observability (which Plans 1-2 provide via `RequestContext`)
- Stress test needs typed exceptions to classify failures properly (Plan 2)
- Plan 4's admin API is where you'd inspect per-agent P2 reward modulation under load
- Shipping Plans 1-3 alone without Plan 4 means you can't debug concurrent-agent P2 validation effectively

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
