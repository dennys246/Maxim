# Maxim Plans

Current version: **0.9.3** (published on PyPI as `pymaxim`).
Target: **1.0** — cross-session learning demonstrated without LLM fine-tuning.

## Version Roadmap

| Version | Theme | Status |
|---------|-------|--------|
| **0.6** | **Generalizable embodiment** — E0 sim wiring + E1 Asset Foundry | **Published** |
| **0.7** | **Self-generating simulations** — Imagination, B3 Acting Coach, E2-E3, F3-F5, SEM discovery, gating, deliberation | **SHIPPED** (2026-04-20) |
| **0.8** | **Cognitive maturity + embodiment** — WM+Exec, PFC cycle, temporal credit, display overhaul, reflex system, proprioceptive discovery, affordance concept transfer, entity ownership, component damage | **SHIPPED** (2026-04-25) |
| **0.9.1** | **Substrate-annotates-LLM-context** — Roy-2c probe, Stage 0 telemetry, Wires A+1+2+3, Roy-3 validation, EC centroid-drift fix | **SHIPPED** (2026-05-25, plan: [release_0_9_1.md](release_0_9_1.md)) |
| **0.9.2** | **Config unification + Hivemind shareability + LLM timeout scalability** — `~/.config/maxim/config.json`, `maxim config/model/substrate` CLIs, `hivemind/` substrate bundle, TTFT keepalive, per-tier timeout, context-overflow admission, stall detector, leader-local harness singleton guard | **SHIPPED** (2026-06-05) |
| **0.9.3** | **Loud optional-dependency failures** — `utils/optional_deps.py` centralises 45+ import sites; missing requested backend raises `OptionalDependencyError` instead of silently returning empty responses | **SHIPPED** (2026-06-06) |
| **1.0** | **Validation + stabilization + grounding** — cross-session proof, bio-system protocol freeze, sensorimotor cradle, SEM world enrichment, SCN feedback loop, cleanup, docs | Target |

**Discipline:** 0.7 was the last major feature version. 0.8 matured cognition and embodiment. 1.0 stabilizes, validates, and grounds — every bio-system fully operational with closed feedback loops, interfaces frozen with future-proof protocols, and the agent demonstrably learns from its own body.

## 1.0 — Validation + Stabilization + Grounding

The unified plan: [v1_refinement.md](v1_refinement.md)
Release writeup: [The Honest Benchmark](../../html-guides/maxim-1-0-release.html) ([announcement copy](../announcements/maxim_1_0_release.md)) — what shipped + the pre-registered cross-session experiments ([Exp 37](../experiments/37_cross_session_graduation.md) Goldilocks zone · [Exp 38](../experiments/38_counter_prior_substrate.md) counter-prior dominance · [Exp 40](../experiments/40_counter_prior_goldilocks.md) Goldilocks counter-prior) that mapped where the substrate helps vs where the LLM prior dominates.

| Section | Items | Status |
|---------|-------|--------|
| **Validation** | Cross-session sim experiment (prove the 1.0 claim) | **PARTIAL PASS** — 3 memories/turn on resume ([Exp 10](../experiments/10_cross_session_enrichment.md)) |
| **Bio-system stabilization** | Protocol enrichment (freeze-worthy interfaces), SCN oscillator feedback (close the loop), SEM world enrichment Phases 2-3 (rich environments) | Pending |
| **Sensorimotor grounding** | Cradle of Artificial Civilization (fire hurts, learned through sensors not language) | Pending |
| **Pipeline completion** | ToolPainBridge temporal migration (~50 LOC), episode semantic shift (Stage 2) | **P1 SHIPPED** (temporal events), P2 pending |
| **B5 substrate-primary** | Phase -1 + Phase 0 harness + Hivemind shareability | **ALL SHIPPED** — Phase -1 + Phase 0 harness (PR #228, 2026-05-09); EC sensor-encoding (branch `feat/phase0-sensor-encoding`); Hivemind shareability (PRs #305–#311, 2026-05-31) |
| **Cleanup** | Probe shim removal, dead code, DamageEntityTool shim, modulator sensors, health derived, raw constructor enforcement | **C1–C6 ALL SHIPPED** (C1+C2+C3 PR #196, 2026-04-26; C4-C5-C6 hard-error flips PRs #299–#303, 2026-05-29) |
| **Docs** | Agent memory transfer, API/CLI review, final docs pass | Pending |

**1.0 exit criteria:**
- ✅ Substrate P1-P8: all pass
- ✅ B4 replanning: treatment 100% vs control 0%, Jaccard 0.894
- ✅ Behavioral convergence: 41/41 hypotheses, all 3 tiers
- ✅ Generalizable embodiment (E0): `--embodiment` works with `--sim`
- ✅ 0.7+0.8 feature completion: all tracks shipped
- ✅ P5 stress persistence: 1.0 gate CLOSED (2026-04-21)
- ✅ B5 substrate-primary Phase -1 + Phase 0 harness SHIPPED (PR #228, 2026-05-09) — substrate produces non-LLM actions; cradle-prelinguistic harness writes per-tick telemetry
- ✅ B5 Phase 0 sensor-encoding entry point SHIPPED — `SensorEncoder` in `similarity/encoder.py` hashes drive snapshots into EC with modality `"interoception"` (branch `feat/phase0-sensor-encoding`, 2026-05-09)
- ✅ B5 Hivemind shareability infrastructure SHIPPED (PRs #305–#311, 2026-05-31) — provenance + substrate-domain tags, `nac_merge`/`ec_merge`, identity detection, substrate bundle + `maxim substrate` CLI
- **PARTIAL PASS**: Cross-session validation (V1) — 3 memories/turn on resume, predictions/concepts pending (Exp 37 in flight)
- Pending: Bio-system protocol enrichment (B1) — interfaces freeze at 1.0
- Pending: SCN oscillator feedback (B2) — close the last open feedback loop
- Pending: SEM world enrichment Phases 2-3 (B3) — rich learning environments
- Pending: Cradle experiment (B4) — sensorimotor learning without language (Exp 37)
- Pending: API/CLI surface review + agent memory transfer docs

## Active (top-level)

Plans on the 1.0 critical path or actively shipping:

- [release_0_9_1.md](release_0_9_1.md) — **0.9.1 release plan.** Substrate-annotates-LLM-context pattern. Roy-2c probe + Stage 0 telemetry + Wire-A (cluster-bias annotation, NEW) + Wires 1+2+3 from bio_emergent_persona_foundations.md + Roy-3 validation. Supersedes the foundations doc's "deferred to 1.1+" disposition (Roy-2pc shifted the empirical floor).
- [v1_refinement.md](v1_refinement.md) — **1.0 release plan.** Validation + bio-system stabilization + sensorimotor grounding + pipeline completion + cleanup + docs + contract clarification (Section 7). 1.1 track index is Section 8.
- [structural_invariant_tests.md](structural_invariant_tests.md) — **2026-05-27 DRAFT.** Three-stage test discipline companion to the regression-guard convention (PRs #274-#277). Stage 1 statistic-shape tests for accumulators (~100 LOC), Stage 2 scripted-action trajectory tests (~250 LOC), Stage 3 multi-agent marker + fixture + CI lint (~150 LOC). Catches Wire 1 / sequential-drift / P4 silent-merge classes structurally. Kickoff prompt at [memory/kickoff_structural_invariant_tests.md](../../.claude/projects/-Users-dennyschaedig-Scripts-Maxim/memory/kickoff_structural_invariant_tests.md).
- [behavioral_graduation_candidates.md](behavioral_graduation_candidates.md) — **Living doc + 1.0 gate, paired alongside benchmarking** (2026-05-27 v0.1 seed). Two phases of the same discipline: (1) pre-1.0 graduation push — which `[engineering]` invariants from CLAUDE.md must graduate to `[behavioral]` via cited Roy/equivalent experiments before 1.0; (2) post-1.0 behavioral-regression discipline — Earned invariants get re-run on triggering events (encoder swap, bio-system refactor, substrate-pipeline change, minor-version heartbeat). Status states span the lifecycle: `Pending` / `Earned` / `Maintained` / `Stale` / `Broken` / `Retired` / `Dropped`. `Stale` and `Broken` entries block the next release. Three-tier scope; v0.1 seeds 20 candidates; honest accounting flags 4 as "can't predicate yet — claim fuzzy."
- [grounded_language_acquisition.md](grounded_language_acquisition.md) — **Phase -1 + Phase 0 harness + EC sensor-encoding ALL SHIPPED** (PR #228 + branch `feat/phase0-sensor-encoding`, 2026-05-09). Substrate-primary AUT mode is real; `SensorEncoder` hashes drive snapshots through EC with modality `"interoception"`. Phase 0 *validation* + Phase 1 + Phase 2 are 1.1+ (1.1-T7).
- [maxim_hivemind.md](maxim_hivemind.md) — Companion to grounded_language_acquisition. Three-layer architecture (LLM-AUT default + Maxim Oasis substrate-primary + Maxim Hivemind P2P). **B5 Hivemind shareability SHIPPED** (PRs #305–#311, 2026-05-31); Oasis software in 1.1; full Hivemind P2P in 1.2.
- [sem_world_enrichment.md](sem_world_enrichment.md) — **Phases 1+2 SHIPPED.** Phase 3 (composable body archetypes) partial — archetype YAMLs in `_data/components/archetypes/`, no avatar migration yet. 1.0 vs 1.1 scope decision pending.
- [persona_cleanup_and_mode_transition.md](persona_cleanup_and_mode_transition.md) — **Stage 1 SHIPPED** (PR #217, 2026-04-30): additive `--mode` flag + deprecation warnings on `--persona` and `register_persona`. Stages 2-6 (resolve testing strategy, dispatch hook migration, public API migration, hard-delete, docs+memory) are 1.1 deprecation cleanup work.
- [bio_emergent_persona_foundations.md](bio_emergent_persona_foundations.md) — **Field reservations SHIPPED** (PR #216, 2026-04-30) under V1 Phase A clean-pass branch. Full Stages 0-3 implementation + Stages 4-5 deferred to 1.1+ since substrate alone reproduced V1 cross-session recall.
- [scene_actor_affordances.md](scene_actor_affordances.md) — **Stages 1+2 SHIPPED** (PR #213, 2026-04-30): `target_effect` field + `OrchestratorActorTool`. Stages 3-5 (orchestrator prompt update, designer template hint, validation experiment) are 1.1 work.
- [reactive_peer_mesh_roadmap.md](reactive_peer_mesh_roadmap.md) — Living roadmap. C3-C4.6 complete. C5+ remain. Not gating 1.0.
- [cluster_reward_bias_decay_tau_split.md](cluster_reward_bias_decay_tau_split.md) — **Phase 1 SHIPPED ([PR #267](https://github.com/dennys246/Maxim/pull/267), 2026-05-26); Phase 3 validation complete ([30_wire_a_tau_validation.md](../experiments/30_wire_a_tau_validation.md), 2026-05-26).** Tau split structurally validated — Wire-A's annotation rendered `[strongly rewarding]` (max\|bias\| 0.753-0.997) throughout the Roy-3a-retry test arm; decay trajectory fit the model within 0.3%. PRIMARY criterion (Arm A ≥1 `sense_food_source`) still failed due to **two downstream gaps** the kickoff didn't anticipate: substrate-scene-tool-availability + imagination substrate-blindness. The bias-magnitude side of Wire-A is no longer the bottleneck. Two new 1.1+ plan docs surfaced.
- [sense_tool_registry.md](sense_tool_registry.md) — **DRAFT (2026-05-26, 1.0 MVP + 1.1+ full).** Surfaced by Roy-3a-retry verdict; reframed 2026-05-27 from 1.1+ → 1.0 MVP critical path. **1.0 MVP (~150-200 LOC):** grayscale visibility for SEM-derived inactive tools (`[not in current location]` tag), `auto_fire` tool metadata, registration-time `kind=` classifier. Closes the substrate-favored-tool-not-in-scene gap that blocked Wire-A annotation→action conversion. Deferred to 1.1+: `sensory_events.jsonl` separation, LRU tuning, NAc predicate-outcome typing.
- [imagination_substrate_signals.md](imagination_substrate_signals.md) — **DRAFT (2026-05-26, 1.0 MVP + 1.1+ full).** Surfaced by Roy-3a-retry verdict; reframed 2026-05-27 from 1.1+ → 1.0 MVP critical path. **1.0 MVP (~20-30 LOC):** Hookup 1 only — substrate-aware manifest generation that passes `NAc.get_agent_tool_biases()` to `Narrator.generate_scene_manifest()`. Complement to [sense_tool_registry.md](sense_tool_registry.md). Deferred to 1.1+: Hookup 2 (per-tick subscriber), Hookup 3 (arousal-gate relaxation). Gating on outcome of next Wire-A Roy iteration with both MVPs landed.
- [scn_decay_anchoring.md](scn_decay_anchoring.md) — **DRAFT (2026-05-26, 1.0 nice-to-have / 1.1 acceptable).** Phase C of the tau-split kickoff sequence; status resolved 2026-05-27 from "1.0 or 1.1 open question" to "1.0 nice-to-have / 1.1 acceptable." Does NOT touch the substrate→action conversion pathway (which the cycle-C MVPs handle); addresses hardware portability of decay timescales. Ships if Roy outcome creates a multi-machine benchmarking need; otherwise stays for 1.1. Prerequisite for [decay_consolidation_calibration_plan.md](decay_consolidation_calibration_plan.md) at whatever release it lands.
- [auth_format_freeze_audit.md](auth_format_freeze_audit.md) — **DRAFT (2026-06-04, CC13).** Narrow format-freeze pass on four security-shaped surfaces shipping in 1.0 (api_key_ref URI namespace, Hivemind bundle `signature_algorithm` registry, `mesh.yml::cluster_key` shape, leader proxy `Authorization:` scheme dispatch). Does **not** implement authentication — full pluggable auth provider abstraction stays a 1.1+ track alongside Hivemind P2P. ~50 LOC + ~16 tests + 1 doc page, ~0.5-1 day wall. Parallel to benchmarking — touches doc + freeze-shape additions only, no behavior under measurement changes. v1_refinement.md Section 7 / CC13 entry.
- [ec_centroid_drift_fix.md](ec_centroid_drift_fix.md) — **SHIPPED (2026-05-24).** Five-phase fix for text-modality centroid drift surfaced by [24_roy_paraphrase_diagnostic.md](../experiments/24_roy_paraphrase_diagnostic.md). Phase 1 (PR #259, matrix sweep) + Phase 2 (PR #260, 0.01 fine-sweep refinement) named threshold **0.44** as the strictly-dominant cell on both P1 paraphrase and Roy paraphrase fixtures. Phase 3 (PR #261 + #262) shipped `ECConfig.pattern_complete_threshold = 0.44` with structural pinning of the coupled NAc constant + Roy-5 H1C boundary. Phase 3.5 (PR #263) parameterized `NAc.get_threshold_overrides` to thread the live EC threshold from `LinguisticEncoder`; P2 validation sweep improved from +56.0pp ± 29.0pp to +58.4pp ± 9.1pp, 10/10 seeds. Phase 4 (PR #264) ran Roy-2c on the post-fix substrate: **behavioral signal unchanged** (A ≈ B ≈ C, 0× sense_food_source × 3 arms — H1 cross-source alignment gap is structurally upstream of drift) but **structural fragmentation reduced 79%** (a_vs_b cluster_reward_bias_l2 2.566 → 0.535, 10 keys → 4). Drift was real; drift is not the dominant Roy mechanism. Fix is V1 cross-session prerequisite + substrate hygiene. JEPA / cross-modal binding priority unchanged — stays a 1.0-or-1.1 gate.

- [config_unification.md](config_unification.md) — **SHIPPED (PR #318, 2026-06-03).** `~/.config/maxim/config.json` single-source operator config; `resolve_setting` precedence chain (CLI > env > config.json > default); `maxim config` CLI verbs; role-detector unification (seven-rank single source of truth in `runtime/role.py`); per-tier remote routing migration; doctor "Resolved Config" section; deprecation warnings for absorbed env vars.
- [leader_ux_profile_management.md](leader_ux_profile_management.md) — **SHIPPED (PR #314, 2026-05-31).** L1: bundled profiles expanded (qwen2.5-32b, llama-3.1-70b, mixtral-8x7b). L2: `~/.config/maxim/profiles.yml` user-profile loader. L3: `maxim model add|remove|list` CLI verbs.
- [llm_timeout_scalability.md](llm_timeout_scalability.md) — **Stages 1–3 SHIPPED (PRs #320–#323, 2026-06-03/04).** Stage 1: per-tier `MAXIM_LANE_<TIER>_TIMEOUT_S` + `lanes.<tier>.timeout_s` config field. Stage 2: context-overflow admission gate (HTTP 413 + `MAXIM_PROXY_CONTEXT_ADMISSION`). Stage 3: TTFT keepalive emitter (`MAXIM_PROXY_KEEPALIVE_INTERVAL_S`). Stage 4 (adaptive throughput model) is 1.1+.
- [stall_detector_timeout_awareness.md](stall_detector_timeout_awareness.md) — **Stage 1 SHIPPED (PR #324, 2026-06-04).** `runtime/llm_call_registry.py` in-flight call registry + `runtime/stall_threshold.py::compute_stall_threshold`. Orchestrator stall detector now suppresses nudges during legitimate inference.
- [mesh_perception_transport.md](mesh_perception_transport.md) — **1.0 prep SHIPPED (PR #329, 2026-06-04).** `Percept.to_wire_dict`/`from_wire_dict` wire format; substrate fields excluded from wire; non-blocking `PerceptSource` protocol contract reserved. Full transport ships in 1.1.
- [benchmarking_1_0.md](benchmarking_1_0.md) — **Scope doc (2026-05-29).** Defines the 1.0 benchmark gate — what "passes" means, what is out of scope. Sibling to [behavioral_graduation_candidates.md](behavioral_graduation_candidates.md).
- [cradle_activation_fixes.md](cradle_activation_fixes.md) — **ACTIVE (revised 2026-06-04).** Empirical reframing of the three compounding bugs that blocked Exp 37 cradle substrate-transfer measurement. Blocks [37_cross_session_graduation.md](../experiments/37_cross_session_graduation.md).
- [exp37_metric_pivot.md](exp37_metric_pivot.md) — **ACTIVE (2026-06-04).** Primary metric pivot to `positive_approach_engagement_fraction` (Path 2) after `per_action_failure_rate` proved structurally 0 on Arm A. Blocks Exp 37 re-fire.
- [exp37_sd_shift.md](exp37_sd_shift.md) — **AMENDMENT IN FLIGHT (2026-06-05).** Statistical test swap to mean-shift-in-SD-units. Blocks Exp 37 re-fire.
- [cloud_dispatch_debug.md](cloud_dispatch_debug.md) — **DRAFT (2026-06-05).** Cloud-dispatch path debug + cleanup during Exp 37's 33-hour background window.
- [key_drift_detection.md](key_drift_detection.md) — **DRAFT (2026-06-03).** Proactive API key mismatch surface for peer↔leader. Small follow-on to config_unification (~150 LOC src + ~40 tests). Independent of other 1.0 work.

### 1.1 track (concurrent development)

- [substrate_exploration_policy.md](substrate_exploration_policy.md) — **DRAFT (2026-06-17, 1.1 target).** Adds an exploration policy to substrate-primary action selection (`NAc.recommend_action`) to break the deterministic-argmax fixation that stalled Exp 39 and left graduation row **#6** reframed-settled at 1.0. Rides on the existing selection chokepoint — no new bus/bio-system, per the front-gate — at ~145 LOC across selection + a session-scoped visit-count + per-tick decay + config; default-off ≡ legacy argmax (the regression anchor). Validated by [Exp 41](../experiments/41_substrate_primary_exploration.md). Surfaced by the 2026-06-17 post-1.0 roadmap review as the **highest-leverage post-1.0 mechanism**: it is the gate the substrate-primary → Oasis → Hivemind value chain implicitly depends on (sharing a substrate is only worth the engineering once the substrate is shown behaviorally load-bearing).
- [substrate_primary_cradle_readiness.md](substrate_primary_cradle_readiness.md) — **FINDINGS + PLAN (2026-06-18).** Consolidated bird's-eye of the substrate-primary cradle blocker chain found while building toward Exp 41. Unifying root cause: substrate-primary removes the LLM but the cradle's scene-entity harm loop was only wired through the LLM/narrator (Layer-2 proximity) path, so scene-affordance `self_effect` is inert (`embodiment=None` on phase-activated tools) and a scene action has no bodily consequence. B2 (exploration) + B3 (drive-need derivation) shipped; B4 (thread embodiment into substrate-primary phase activation, gated so Exp 37/38 stay byte-identical) + B5 (embodiment-aware `record_outcome`) are scoped + awaiting authorization. The keystone for a *valid* Exp 41 run.
- [../experiments/41_substrate_primary_exploration.md](../experiments/41_substrate_primary_exploration.md) — **PRE-REGISTERED (2026-06-17), metrics FROZEN.** The test that closes the loop Exp 38/40 (override fails *under the LLM*) and Exp 39 (substrate-primary fixates) only pointed at: substrate-primary + counter-prior + exploration. Asks whether the unmasked substrate learns from embodied pain alone to override its **own built-in drive-affinity prior**. 2×2 ($0, local, 40 runs). PASS/PASS = GRADUATE row #6 (thesis earned in the only regime that can test it); FAIL/FAIL = honest "fixation is deeper than selection." Depends on the exploration policy + a `cradle_prelinguistic_deceptive` arc + a new analyzer landing first.
- [decay_consolidation_calibration_plan.md](decay_consolidation_calibration_plan.md) — **DRAFT (2026-05-26, 1.1+ target).** Calibration-by-simulation framework replaces hand-picked NAc decay-tau defaults and consolidation parameters with tier-transition-driven calibrated values. Depends on `cluster_reward_bias_decay_tau_split.md` Phase 1 (shipped) and `scn_decay_anchoring.md` as prerequisites. 6 phases, ~1,400 LOC, 7–10 weeks for Phases 0–5. Revive trigger: multi-machine benchmarking confirms decay timescales are the remaining bottleneck, or any new NAc tau consumer inherits the wrong default again.
- [roy_5_encoder_alignment_disambiguator.md](roy_5_encoder_alignment_disambiguator.md) — **Stage 1 SHIPPED (2026-05-14, PRs #249 + #251).** Cosine-localization analyzer disambiguates Roy-2c's H1 into H1c (threshold tuning) / H1b (encoder A/B) / H1a (subspace incompatibility). Roy-5a verdict: **H1a confirmed across three runs**, with the stronger-than-modeled finding that `SensorEncoder` (384-dim) and `LinguisticEncoder` (768-dim) live in **different-dimensional** embedding spaces — cross-modality cosine is mathematically undefined without a learned projection. Stage 3 (cradle-arc redesign) is the next implementation track and is the data-production prereq for [jepa_cross_modal_alignment.md](jepa_cross_modal_alignment.md).
- [cross_modal_substrate_binding.md](cross_modal_substrate_binding.md) — **CANCELLED by Roy-4 (2026-05-13).** Hebbian binding edges between EC nodes via temporal co-activation was the original design; Roy-4 confirmed priming↔test EC node IDs never co-fire (zero would-have-bound edges across the full parameter sweep). Stage 4a's resurrection conditions need what [jepa_cross_modal_alignment.md](jepa_cross_modal_alignment.md) ships (a learned projection into a shared latent — Hebbian binding can then operate in the dim-consistent shared space). Stays cancelled until Roy-5 Stage 3 + JEPA Stage 5 both PASS.
- [jepa_cross_modal_alignment.md](jepa_cross_modal_alignment.md) — **DRAFT (2026-05-14).** 1.2+ research direction (NOT 1.0 / 1.1 critical path). Two-headed JEPA projection learns a shared latent across `SensorEncoder` + `LinguisticEncoder` from substrate-emergent paired data the redesigned cradle arc produces. Bio-defensible answer to roy_5's Stage 4b "encoder replacement to 1.2+"; provides the architecture [grounded_language_acquisition.md](grounded_language_acquisition.md)'s Phase 2 symbol-binding layer currently sketches as "small MLP, or a tiny RNN." Stays DRAFT until Stage 3-of-roy-5 ships and Stage 0 audit (~50 LOC, days) confirms paired-data sufficiency.
- [minecraft_benchmark.md](minecraft_benchmark.md) — Live demo + harness comparison. Stub. 1.1 splash launch.
- [mcp_compatibility.md](mcp_compatibility.md) — MCP server + client + schema interop. Stub. 1.0 ships CC9 (dual-format Tool schema) as the prerequisite.

## Living practice docs

Accumulate evidence over time. Not on the critical path — ongoing scientific/operational questions.

- [behavioral_convergence_practice.md](behavioral_convergence_practice.md) — Does the agent get better across sessions? 4 experiments, 41/41 hypotheses confirmed.
- [memory_consolidation_practice.md](memory_consolidation_practice.md) — Refines P8 sleep-replay. ACTIVATED (P8 shipped 2026-04-19).
- [tool_refinement_plan.md](tool_refinement_plan.md) — Ongoing tool surface curation.
- [persona_convergence_crucible.md](persona_convergence_crucible.md) — Long-horizon persona emergence ("Roy" iterations). Three-arm comparison methodology, substrate-only priming. Begins post-1.0; depends on bio_emergent_persona_foundations Stages 0-3.

## Deferred (post-1.0, revive on trigger)

Design work preserved in [deferred/](deferred/). Each has an explicit "revive when" condition.

- [deferred/b5_embodiment_narrative_separation.md](deferred/b5_embodiment_narrative_separation.md) — Formalize SEM/DM prompt boundary. Revive when prompt-bleed bug surfaces.
- [deferred/agent_backed_entities.md](deferred/agent_backed_entities.md) — 3-tier cognition + Cradle-trained cast + mesh-pressure budget. Revive if scene_actor_affordances diagnostic doesn't close the gap.
- [deferred/goal_depth_integration.md](deferred/goal_depth_integration.md) — GOAL WMS entry kind, goal-tagged episodes. Stage 3 absorbed by temporal_credit_integration. Remaining stages are enrichment, not gating.
- [deferred/bio_system_plugin_plan.md](deferred/bio_system_plugin_plan.md) — Plugin discovery for bio-systems. Revive when external contributors appear.
- [deferred/mother_npc_stimulus_plan.md](deferred/mother_npc_stimulus_plan.md) — Two-agent stimulus pattern. Revive when behavioral convergence hits variety ceiling.
- [deferred/pecking_order_graph_plan.md](deferred/pecking_order_graph_plan.md) — Unified hierarchy DAG. Revive when multi-node topology matters.
- ~~deferred/mother_maxim_plan.md~~ → **SUPERSEDED** (2026-05-09) by [maxim_hivemind.md](maxim_hivemind.md). Old plan moved to [archive/mother_maxim_plan.md](archive/mother_maxim_plan.md). Reframed for the substrate-primary world: peer-to-peer Maxim Oases (sustaining gathering places) instead of central Mother server; Maxim Hivemind (collective cognition layer) instead of Pecking Order Graph; distilled bio-substrate snapshots instead of raw episodes. ~2,100 LOC instead of 3,800. Phasing: shareability infrastructure SHIPPED in 1.0 (B5, PRs #305–#311); Oasis software in 1.1; full Hivemind P2P protocol in 1.2.
- [deferred/dungeon_master_extensions.md](deferred/dungeon_master_extensions.md) — DM post-MVP features.
- [deferred/cross_platform_file_lock.md](deferred/cross_platform_file_lock.md) — Unify `process_lock` + `filelock`. Tech debt, blocks nothing.
- [deferred/mesh_doc_transport.md](deferred/mesh_doc_transport.md) — Mesh-to-mesh doc exchange (C9). Not started.
- [deferred/pain_bus_bridge_subscriber_unification.md](deferred/pain_bus_bridge_subscriber_unification.md) — Bridge/subscriber attribution-asymmetry fix. Monitor; open when pending-event context enriched.
- [deferred/node_security_simplification.md](deferred/node_security_simplification.md) — Phase 1 shipped. Phase 2 config unification deferred.
- [deferred/llm_path_multi_peer_dispatch.md](deferred/llm_path_multi_peer_dispatch.md) — Multi-peer load distribution. Revive when 2+ GPU nodes.
- [deferred/llm_mesh_capability_aware.md](deferred/llm_mesh_capability_aware.md) — Capability-aware routing. Revive when heterogeneous mesh.
- [deferred/llm_path_async_router.md](deferred/llm_path_async_router.md) — Async router. Revive if head-of-line blocking observed.
- [deferred/llm_path_fair_scheduling.md](deferred/llm_path_fair_scheduling.md) — Fair-share scheduling. Revive if rate limiting insufficient.
- ~~deferred/grounded_language_acquisition.md~~ → **PROMOTED** to [grounded_language_acquisition.md](grounded_language_acquisition.md) (2026-05-09). **Phase -1 GATE CLEARED + Phase 0 harness + EC sensor-encoding ALL SHIPPED** (PR #228 + branch `feat/phase0-sensor-encoding`, 2026-05-09): substrate produces non-LLM actions; `cradle_prelinguistic` arc + motor-only AUT prompt + per-tick telemetry; `SensorEncoder` hashes drive snapshots through EC with modality `"interoception"`. Hivemind shareability SHIPPED (PRs #305–#311, 2026-05-31). Full Phase 0 validation + Phase 1 + Phase 2 in 1.1+ (1.1-T7). Trigger: 2026-05-09 audit found 60-70% of recent engineering effort going to LLM-mitigation scaffolding (~845 LOC of band-aids) rather than substrate work; the parallel-mode pivot is the structural fix. D&D campaign survival is the bidirectional kill criterion (substrate failure OR simulation-environment failure). The existing LLM-AUT path stays as the user-facing default.

## Archive

Completed or superseded plans live in [archive/](archive/).

Recently archived (2026-05-02 — V1 phased re-run wave complete):
- [archive/confound_quarantine.md](archive/confound_quarantine.md) — **✅ PURPOSE SERVED + ARCHIVED.** Flags shipped (PR #214), V1 phased re-run executed against commit `f742527` (2026-04-30, [Experiment 12](../experiments/12_v1_phased_attribution.md)) → all 7 phases recalled `BLUE-7-DAWN`, Phase A clean pass forced flag removal in 1.0 (PR #215). Plan stays in archive as the artifact explaining the disposition; harness `scripts/run_v1_phases.sh` and Experiment 12 stay in-tree for academic-ML reproducibility.

Previously archived (2026-04-29 1.0 audit — verified shipped against code):
- [archive/bio_system_protocol_enrichment.md](archive/bio_system_protocol_enrichment.md) — **✅ COMPLETE + ARCHIVED.** All 5 `*Context` dataclasses live at `src/maxim/models/bio_context.py`.
- [archive/scn_oscillator_feedback.md](archive/scn_oscillator_feedback.md) — **✅ SHIPPED + ARCHIVED.** B2 anticipatory temporal credit. `OscillatorNetwork.predict_event_imminence`, `TemporalCreditDistributor.anticipatory_pre_activate`, `scn.enable_oscillator()` all wired.
- [archive/proprioceptive_discovery.md](archive/proprioceptive_discovery.md) — **✅ Mechanisms A+B SHIPPED + ARCHIVED.** Latent-affordance discovery + entity acquisition.

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
