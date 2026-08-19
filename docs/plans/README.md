# Maxim Plans

Current version: **1.0.6** (git tag; the latest release published on PyPI as `pymaxim` is still 1.0.0, 2026-06-17 — closing that gap is roadmap item 16).
Now: **1.1 "Sensorimotor" release closure** — ship the merged embodiment work plus
the correctness, contract, truth, and verification debt it incurred. **Zero new
mechanisms.** Reconciled 2026-08-19 after the D13/D14 investigation and repository
review; see
[roadmap_1_1_to_1_3.md](roadmap_1_1_to_1_3.md) for the cut line, the blockers, and
why the reflex tier / loudness / Oasis moved.

**Directory last deeply audited 2026-07-15; release authority reconciled
2026-08-19.** Root = active work only. Superseded release checklists belong in
`archive/`; the 1.1 cut exists only in the roadmap above.

**New here?** [glossary.md](glossary.md) decodes the coded IDs (`CC3`, `Wire-A`, `Roy-4`, `P2`, `NAc`, GRADUATE, …) — including the prefixes that mean different things in different plans.

## Version Roadmap

| Version | Theme | Status |
|---------|-------|--------|
| **0.6** | **Generalizable embodiment** — E0 sim wiring + E1 Asset Foundry | **Published** |
| **0.7** | **Self-generating simulations** — Imagination, B3 Acting Coach, E2-E3, F3-F5, SEM discovery, gating, deliberation | **SHIPPED** (2026-04-20) |
| **0.8** | **Cognitive maturity + embodiment** — WM+Exec, PFC cycle, temporal credit, display overhaul, reflex system, proprioceptive discovery, affordance concept transfer, entity ownership, component damage | **SHIPPED** (2026-04-25) |
| **0.9.1** | **Substrate-annotates-LLM-context** — Roy-2c probe, Stage 0 telemetry, Wires A+1+2+3, Roy-3 validation, EC centroid-drift fix | **SHIPPED** (2026-05-25, plan: [release_0_9_1.md](archive/release_0_9_1.md)) |
| **0.9.2** | **Config unification + Hivemind shareability + LLM timeout scalability** — `~/.config/maxim/config.json`, `maxim config/model/substrate` CLIs, `hivemind/` substrate bundle, TTFT keepalive, per-tier timeout, context-overflow admission, stall detector, leader-local harness singleton guard | **SHIPPED** (2026-06-05) |
| **0.9.3** | **Loud optional-dependency failures** — `utils/optional_deps.py` centralises 45+ import sites; missing requested backend raises `OptionalDependencyError` instead of silently returning empty responses | **SHIPPED** (2026-06-06) |
| **1.0** | **Validation + stabilization + grounding** — cross-session proof, bio-system protocol freeze, sensorimotor cradle, SEM world enrichment, SCN feedback loop, cleanup, docs | **SHIPPED** (2026-06-17) |
| **1.1** | **"Sensorimotor" release closure** — merged embodiment work plus D13/D14 liveness, stable Python API truth, hermetic fast tests, atomic NAc+EC invalidation, architecture-audit regression enforcement, S4, remaining heartbeat chapters, release mechanics, and agent-guidance convergence. **Zero new mechanisms.** | In progress — authoritative gates in [roadmap](roadmap_1_1_to_1_3.md#11-cut-line--reconciled-2026-08-19) |
| **1.2** | **Oasis + Hivemind** — peer substrate sharing (~800 LOC) + P2P protocol (~600 LOC). Entry is gated on encoder provenance/compatibility, read-side EC safety, bundle-version threat modeling, and the 1.1 verification gates staying green. | Planned, gated |
| **1.3** | **Perception fabric + reflex tier** — cochlear front-end, population coding, vision encoder, orient-windowed binding, three-factor calibration ([cross_modal_perception_fabric.md](cross_modal_perception_fabric.md), [three_factor_credit_assignment.md](three_factor_credit_assignment.md) — both self-targeted 1.3), and the DN-canonical orienting reflex. Contains the pivotal may-fail experiment (Stage 0c). | Planned |

**Discipline:** 1.1 is a mechanism freeze. Existing behavior is not assumed live
because a type or bridge exists: release claims require an executed contract test or
graduated behavioral row. New mechanism work resumes only after the 1.1 cut is
releasable and the 1.2 entry gates are satisfied.

## 1.0 — SHIPPED (2026-06-17)

The unified plan lives in the archive: [archive/v1_refinement.md](archive/v1_refinement.md) — all hard requirements closed 2026-06-15 (C1–C6 cleanup, CC1–CC13 contract clarifications, B1/B2/B4 bio-system stabilization, P1–P4 pipeline completion, D1–D3 docs).
Release writeup: [The Honest Benchmark](../../html-guides/maxim-1-0-release.html) ([announcement copy](../announcements/maxim_1_0_release.md)) — what shipped + the pre-registered cross-session experiments ([Exp 37](../experiments/37_cross_session_graduation.md) Goldilocks zone · [Exp 38](../experiments/38_counter_prior_substrate.md) counter-prior dominance · [Exp 40](../experiments/40_counter_prior_goldilocks.md) Goldilocks counter-prior) that mapped where the substrate helps vs where the LLM prior dominates. The 1.0 benchmark gate scope + disposition: [archive/benchmarking_1_0.md](archive/benchmarking_1_0.md). Post-1.0, the substrate-primary safe-vs-harm claim was EARNED mechanism-level by [Exp 42](../experiments/42_substrate_primary_preference.md) (GRADUATE, PR #380).

## Active

Everything in the root is in-flight or a living doc. Deep-audited 2026-07-15;
release authority reconciled 2026-08-19.

### 1.1 release closure

- [roadmap_1_1_to_1_3.md](roadmap_1_1_to_1_3.md) — **SOLE 1.1 SCOPE
  AUTHORITY.** Gate order: planning liveness → stable API/hermetic tests →
  persistence/architecture enforcement → evidence closure → release transaction.
- [../bugs/repository_review_2026_08_19.md](../bugs/repository_review_2026_08_19.md)
  — D15–D20 evidence and required contracts.
- [../limits/score_cards/](../limits/score_cards/) — repo-grade baselines and
  grade-improvement criteria (2026-08-19: Codex + Claude cards).

### Exp 44 — LLM-primary embodied choice (1.1.x experiment line; not a 1.1 gate)

- [controlled_llm_primary_embodied_harness.md](controlled_llm_primary_embodied_harness.md) — **ACTIVE Exp 44 vehicle.** G1 deterministic scene embodiment SHIPPED (`MAXIM_DETERMINISTIC_SCENE_EMBODIMENT`, `7c052a3d`); `MAXIM_DISABLE_IMAGINATION` universal gate shipped (`d024ba63`). G2 (drive-gating) rerouted to [grounded_language_acquisition.md](grounded_language_acquisition.md); G3 (terse narrator) HELD as hygiene. Next: one validation seed confirming safe-vs-harm learning appears now that harm is deterministic, then the ablation arms below.
- [acting_coach_body_state_ablation.md](acting_coach_body_state_ablation.md) — **Pre-registered Exp 44 ablation; prerequisites SHIPPED (PR #391)** — `MAXIM_ENABLE_BODY_STATE_PROMPT` opt-in wiring, `MAXIM_DISABLE_COACH_BODY_LAYERS` arm-B toggle, harness fork, observability. First launch (2026-07-15) blocked on the ops stack (wrong model served + n_ctx mismatch); arms not yet validly run. Gated behind G1 validation.

### Substrate-native orienting (Reachy Mini hardware line)

- [substrate_native_orienting.md](substrate_native_orienting.md) — **ACTIVE umbrella (consolidated 2026-07-15).** The authoritative plan for the orient-to-center line: the substrate learning its first **real-hardware sensorimotor policy** (turn to drive bearing-error → 0, credited by drive-pain *reduction* via NAc). Absorbs the former `audiovisual_orienting.md`. Two-layer framing: Layer 1 = the orient policy (hardware-validated); Layer 2 = spatial co-location as the grounding substrate for language + vision (gated north star). Carries the learned-vs-servo rigor bar.
- [reachy_orient_live.md](reachy_orient_live.md) — **ACTIVE hardware runbook — Step 1 PASSED 2026-07-15 (PR #392).** WS-era SDK ≥1.5 bring-up on the physical Reachy Mini. Steps 2 (`live_2_reactive.py`, reactive orient + sign calibration) and 3 (`live_3_learn.py`, the learning loop with `potential_diff` credit + NAc persistence) are the immediate next work.
- [perception_pipeline_placement.md](perception_pipeline_placement.md) — **PARTIALLY landed (PRs #382–#385 merged 2026-06-27/28); doc-truth corrected 2026-08-07.** The exteroceptive `"audio"` EC modality, DoA audio front-end, and orient affordances on `reachy_mini.yaml` are real and production-wired. `runtime/perception_placement.py` (stage DAG, pinned/placeable resolver, coherence validator) merged as a TYPE LAYER but has **zero production callers** — marked Dormant per Principle 2 (module docstring has the marker + resurrection trigger: the 1.3 perception fabric actually placing stages). The earlier "1.1 core LANDED" wording overclaimed. Remaining: the on-demand 1.2+ tail (leader-side segmentation / frame-transport cut point).

### Substrate-primary spine (1.1 → 1.2)

- [grounded_language_acquisition.md](grounded_language_acquisition.md) — **ACTIVE umbrella for the substrate-primary line.** Phase -1 + Phase 0 harness + EC sensor-encoding shipped (PR #228, 2026-05-09); substrate-primary mode GRADUATED via Exp 42 (PR #380). Actively absorbing intake: G2 substrate→LLM action-salience primitive rerouted here 2026-07-15 (`eec7369e`). Phase 0 validation + Phases 1–3 (vocabulary-constrained → symbol-binding → from-scratch sequence model) are the 1.1+/1.2 arc.
- [maxim_hivemind.md](maxim_hivemind.md) — **Roadmap/architecture companion** to grounded_language_acquisition (three layers: LLM-AUT default + Oasis substrate-primary + Hivemind P2P). B5 shareability SHIPPED (PRs #305–#311; `src/maxim/hivemind/` verified). Oasis + the P2P protocol are now 1.2, gated by D1/D3/D4/D8 and the sharing threat model; they are not part of 1.1 release closure.
- [sem_environmental_proximity_sensing.md](sem_environmental_proximity_sensing.md) — **NEW (2026-07-15) design doc, P1-ready.** Grounds interoception in sensed exteroception: field-query reader for `heat_output`/ambient entity fields (currently declared-but-inert), per-turn environmental-sense pass, dual-path wiring (LLM-primary §1.15 + substrate-primary tick). Explicitly NOT an Exp 44 blocker (that was G1); the medium-term "do it right" for embodied sensing.

### Substrate representation & provenance (1.1 → 1.2) — NEW 2026-08-11

Three plans opened by the Exp 44b pilot + the Exp 48 heartbeat re-run, which hit the
same wall from opposite directions: **the substrate knows more than its artifacts
record.**

- [modality_resolution_and_alignment.md](modality_resolution_and_alignment.md) — **DRAFT, heavily corrected by a four-lens review.** How much each modality can DISCRIMINATE, and whether one threshold can serve incompatible geometries. Key measured facts: `_sensor_embed` is a two-basis interpolation so a scalar resolves ~3 EC nodes at the production sensor threshold (**0.85**, `SensorEncoderConfig` — NOT `ECConfig`'s 0.44, which is the text path); static co-sensors make per-dimension resolution *worse*, not better; the partition for azimuth is left/centre/right, i.e. coarse but well-placed. **Read §0 first** — the first draft's central thesis was false and the recommendation flipped to "bound the claims" (F-C). The one open item shipped as PR #499.
- [deferred/retrosplenial_spatial_frames.md](deferred/retrosplenial_spatial_frames.md) — **DEFERRED design + trigger registration.** Egocentric↔allocentric frame translation. Its falsifiable pre-check RAN (`scripts/rsc_precheck.py`) and **refuted the plan's own predicted failure mode**; §2 records the corrections, incl. the struck circular/ring-code recommendation (azimuth is a folded semicircle: −1 = hard left, +1 = hard right, so a ring code would destroy the discrimination Exp 45/48 rest on). Carries the re-validation registry for any EC-clustering change.
- [annotation_context_and_provenance.md](annotation_context_and_provenance.md) — **DRAFT.** The prompt channel is a lossy encoder of what NAc learned: it drops the cluster id (context-dependent value is inexpressible), keys on exact tool signatures (transfer is name-brittle), and renders a band with no WHY. S1 write-side shipped as PR #501; S2 (context-aware view, ADDITIVE to the agent-wide one — switching outright reproduces the Roy-2c bug Wire-A exists to fix) and S3 (concept-keyed transfer) remain.
- [decision_provenance.md](decision_provenance.md) — **PROPOSED (concurrent session).** The READ-side twin of S1: why an action was *chosen* (score decomposition + `explore_decisive`) at `recommend_action`. Motivated by Exp 48's re-run, where exploration outscored the learned signal ~20:1 and a learned bias below the ~0.11 novelty gap is **invisible**. S1 = write side (why a value exists), this = read side (did it drive the choice); neither substitutes for the other.

### Tool surface / perception hygiene

- [passive_sense_discovery.md](passive_sense_discovery.md) — **DRAFT (2026-07-19), Phase 0 design pass pending authorization.** Ambient (passive/async) tool-discovery channel: gated background discovery + "newly sensed since last turn" prompt delta, keeping `sense_tools` as directed attention. Rides auto-fire + the PR #402 side-channel pattern; three new contract pieces (delta/manifest prompt split, sensory-event vs causal NAc typing, turn-scoped roster coherence). Fixes two pre-existing holes found in the 2026-07-19 deep dive (mid-phase roster churn silently breaking the prompt cache; unwired discovery LRU). Phase 3 revives the deferred [sense_tool_registry.md](deferred/sense_tool_registry.md) 1.1 scope.

### Living discipline docs

- [../bugs/README.md](../bugs/README.md) — **KNOWN-DEFECTS LEDGER.** What is verifiably wrong or bounded now, including the D13/D14 liveness pair and the D15–D20 API/architecture/test cluster from the 2026-08-19 review. Every row carries evidence and a disposition.
- [../limits/score_cards/](../limits/score_cards/) — **REPOSITORY SCORECARDS** (dual-assessor cadence from 2026-08-19: Codex + Claude, same eight axes, independent evidence). Evidence-backed grades for research integrity, runtime correctness, maintainability, test/CI truthfulness, documentation, and release governance, with explicit conditions for improving each grade. Re-scored at each release cut; divergence between the two cards is itself signal.
- [external_critique_response.md](external_critique_response.md) — **LIVING EXTERNAL-CRITIQUE RESPONSE.** The bio-docstring truth pass ([bio_docstring_truth_pass.md](bio_docstring_truth_pass.md)) and CLAUDE.md diet ([claude_md_diet.md](claude_md_diet.md)) shipped; fail-loud Stages 2–3 ([measurement_path_fail_loud.md](measurement_path_fail_loud.md)), god-function decomposition ([god_function_decomposition.md](god_function_decomposition.md)), and CI gate scope remain. The 2026-08-19 scorecards are the current repo-wide assessment rather than a replacement for this critique-specific history.

- [behavioral_graduation_candidates.md](behavioral_graduation_candidates.md) — **Living doc + post-1.0 behavioral-regression discipline.** Actively maintained (Exp 42 GRADUATE recorded 2026-06-23; row #11 feeds Exp 44). Earned invariants re-run on triggers (encoder swap, bio-system refactor, minor-version heartbeat); `Stale`/`Broken` entries block the next release.
- [tool_refinement_plan.md](tool_refinement_plan.md) — **Living tool-curation doc**, updated 2026-07-14 (records the introspection-tools hard-delete from the embodiment truth-restoration pass, PR #390, + rebuild guidance).
- [reactive_peer_mesh_roadmap.md](reactive_peer_mesh_roadmap.md) — **Living mesh roadmap.** Reactivity core complete (C3–C4.6 + C10 1.0-prep); C5 (capacity-aware routing), C6 (admin API), C7/C8 (security/cross-version) cold until multi-tenant need; C10 1.1-ship slice is the near-term item, driven by the Reachy line. Note: §5 version-mapping table predates 1.0 and needs a refresh on next touch.

## Deferred (revive on trigger)

Design work preserved in [deferred/](deferred/). Each has an explicit "revive when" condition (full rationale in the banner at the top of each doc).

Deferred in the 2026-07-15 audit — **partially shipped, remainder paused:**
- [deferred/llm_timeout_scalability.md](deferred/llm_timeout_scalability.md) — Stages 1–3.5 SHIPPED (per-tier timeout, admission gate, TTFT keepalive; PRs #320/#321). Stage 4 adaptive throughput model unbuilt. Revive when a big-model soak yields the `(TTFT, tok/sec)` data it needs.
- [deferred/stall_detector_timeout_awareness.md](deferred/stall_detector_timeout_awareness.md) — Stage 1 SHIPPED (PR #324). Stage 2 heartbeat migration pending (CI grep still allowlists `heartbeat.py` for it). Revive on the next 1.0.x maintenance pass or first heartbeat false-positive.
- [deferred/mesh_perception_transport.md](deferred/mesh_perception_transport.md) — 1.0 prep SHIPPED (PR #329); single-cut-point framing superseded by [perception_pipeline_placement.md](perception_pipeline_placement.md). Revive when a perception stage is placed across the wire (1.2+ frame transport).
- [deferred/bio_emergent_persona_foundations.md](deferred/bio_emergent_persona_foundations.md) — Stages 0–3 + Wire-A SHIPPED via 0.9.1. Only Wires 4/5 remain; evidence points at encoder alignment, not these wires, as the persona blocker. Revive if a Roy iteration shows Wire 4 or 5 is load-bearing.
- [deferred/persona_cleanup_and_mode_transition.md](deferred/persona_cleanup_and_mode_transition.md) — **COMPLETE (2026-08-09).** Stage 1 shipped in PR #217; Option A decided (owner, 2026-08-07); Stages 3–6 shipped in the 1.1 persona hard-remove PR (`personas.py` deleted, `--persona` flags + `/persona` removed, `register_persona()` raises, persona→mode rename with legacy readers). Kept in deferred/ because shipped error messages reference the path.
- [deferred/scene_actor_affordances.md](deferred/scene_actor_affordances.md) — Stages 1+2 SHIPPED (PR #213). Stages 3–5 (prompt rule, designer hint, validation experiment) revive when an adversarial sim needs narrated entity actions to hit real AUT body mechanics, or when re-evaluating agent_backed_entities.
- [deferred/sense_tool_registry.md](deferred/sense_tool_registry.md) — 1.0 MVP SHIPPED (PR #287: `auto_fire`, `kind`, grayscale visibility). 1.1+ full plan revives if substrate→action tool visibility is re-prioritized or predictive learning needs `sensory_events.jsonl` / predicate-outcome typing.
- [deferred/imagination_substrate_signals.md](deferred/imagination_substrate_signals.md) — Hookup 1 MVP SHIPPED (PRs #288/#292). Hookups 2+3 revive after encoder replacement, if per-tick substrate-driven imagination is the next lever.

Deferred in the 2026-07-15 audit — **never started / blocked:**
- [deferred/mcp_compatibility.md](deferred/mcp_compatibility.md) — Zero MCP code; CC9 prerequisite shipped. Revive when the MCP spec stabilizes AND a concrete user need appears.
- [deferred/key_drift_detection.md](deferred/key_drift_detection.md) — Never implemented; the reactive 401-hint fold reduced urgency. Revive if stale-key confusion recurs outside the 401 path.
- [deferred/transition_based_drive_pain.md](deferred/transition_based_drive_pain.md) — Root-cause fix for per-tick drive-pain re-firing (B8 covers channel 1 today). Revive on a second attribution consumer, a channel-2 mis-attribution incident, or any `evaluate_failures` cadence change.
- [deferred/minecraft_benchmark.md](deferred/minecraft_benchmark.md) — Stub; prerequisites (CC1/CC8/CC9) all shipped so nothing decays. Revive when the 1.1 splash launch is greenlit + someone commits to the M0 protocol research.
- [deferred/colibri_worldgen_smoke.md](deferred/colibri_worldgen_smoke.md) — Hard-blocked on an upstream colibrì server bug (degenerate output; engine itself works). Revive when upstream fixes `openai_server.py`; escalate to archive if dead for a quarter.
- [deferred/jepa_cross_modal_alignment.md](deferred/jepa_cross_modal_alignment.md) — 1.2+ research direction; motivating 384-vs-768-dim split still true, but Exp 35/36 resolved via threshold tuning. Revive on a structurally cross-modal problem unsolvable by threshold tuning + a passing Stage 0 paired-data audit.
- [deferred/scn_decay_anchoring.md](deferred/scn_decay_anchoring.md) — Hardware portability of decay timescales; unstarted. Revive when a second hardware baseline joins benchmarking or decay calibration is greenlit (hard prereq of the next entry).
- [deferred/decay_consolidation_calibration_plan.md](deferred/decay_consolidation_calibration_plan.md) — Calibration-by-simulation framework; blocked on scn_decay_anchoring. Revive when decay timescales are confirmed as the bottleneck or a new tau consumer mis-inherits a default.

Deferred in the 2026-07-15 audit — **stalled practice docs** (were "living," stopped accumulating):
- [deferred/persona_convergence_crucible.md](deferred/persona_convergence_crucible.md) — 8 Roy iterations of real data (May 2026); halted on the encoder-alignment gap its own runs root-caused (768-vs-384-dim). Load-bearing negative evidence — do NOT archive. Revive when encoder replacement / cross-modal alignment ships; Roy-1 Adversarial then becomes runnable.
- [deferred/behavioral_convergence_practice.md](deferred/behavioral_convergence_practice.md) — Exp 1–5 (41/41) logged in April; the validation energy migrated to [behavioral_graduation_candidates.md](behavioral_graduation_candidates.md). Revive on a new system-level "does the agent get better" experiment.
- [deferred/memory_consolidation_practice.md](deferred/memory_consolidation_practice.md) — Zero entries ever, despite P8 shipping. Revive when a consolidation-tuning question becomes load-bearing; downgrade to archive if 1.1 ships without touching consolidation.

Earlier deferrals:
- [deferred/b5_embodiment_narrative_separation.md](deferred/b5_embodiment_narrative_separation.md) — Formalize SEM/DM prompt boundary. Revive when prompt-bleed bug surfaces.
- [deferred/agent_backed_entities.md](deferred/agent_backed_entities.md) — 3-tier cognition + Cradle-trained cast + mesh-pressure budget. Revive if scene_actor_affordances diagnostic doesn't close the gap.
- [deferred/goal_depth_integration.md](deferred/goal_depth_integration.md) — GOAL WMS entry kind, goal-tagged episodes. Stage 3 absorbed by temporal_credit_integration. Remaining stages are enrichment, not gating.
- [deferred/bio_system_plugin_plan.md](deferred/bio_system_plugin_plan.md) — Plugin discovery for bio-systems. Revive when external contributors appear.
- [deferred/mother_npc_stimulus_plan.md](deferred/mother_npc_stimulus_plan.md) — Two-agent stimulus pattern. Revive when behavioral convergence hits variety ceiling.
- [deferred/pecking_order_graph_plan.md](deferred/pecking_order_graph_plan.md) — Unified hierarchy DAG. Revive when multi-node topology matters.
- ~~deferred/mother_maxim_plan.md~~ → **SUPERSEDED** (2026-05-09) by [maxim_hivemind.md](maxim_hivemind.md). Old plan moved to [archive/mother_maxim_plan.md](archive/mother_maxim_plan.md). Reframed for the substrate-primary world: peer-to-peer Maxim Oases (sustaining gathering places) instead of central Mother server; Maxim Hivemind (collective cognition layer) instead of Pecking Order Graph; distilled bio-substrate snapshots instead of raw episodes. ~2,100 LOC instead of 3,800. Phasing: shareability infrastructure SHIPPED in 1.0 (B5, PRs #305–#311); Oasis + full Hivemind protocol gated for 1.2.
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

- [archive/release_1_1_checklist.md](archive/release_1_1_checklist.md) —
  **SUPERSEDED 2026-08-19.** Preserved July snapshot; its Oasis/Exp 44 critical path
  was replaced by the current release-closure roadmap.

Archived in the 2026-07-15 deep audit (each verified against code + git history; disposition banner at the top of each doc):
- [archive/v1_refinement.md](archive/v1_refinement.md) — **✅ 1.0 SHIPPED (2026-06-17).** The release master plan; all hard requirements closed 2026-06-15. Residual 1.1 pointers live in sibling plans.
- [archive/benchmarking_1_0.md](archive/benchmarking_1_0.md) — **✅ Gate fired + dispositioned (2026-06-13).** Exp 37/38 executed; performance claim honestly pulled, mechanism/persistence claims stand. Continuation owned by behavioral_graduation_candidates.
- [archive/structural_invariant_tests.md](archive/structural_invariant_tests.md) — **✅ All 3 stages SHIPPED (PRs #279/#280/#281)** — statistic-shape tests, trajectory invariants, multi-agent marker + CI lint.
- [archive/lane_capability_placement_split.md](archive/lane_capability_placement_split.md) — **✅ All phases SHIPPED (PRs #357–#362)** + CLAUDE.md invariant. Hardware tuning per-placement is a documented 1.1+ extension.
- [archive/prompt_caching_for_cloud_backends.md](archive/prompt_caching_for_cloud_backends.md) — **✅ SHIPPED (PR #350).** Byte-stable prefix split; measured ~38% Anthropic ITPM reduction (6× premise honestly falsified in Phase 0).
- [archive/cradle_activation_fixes.md](archive/cradle_activation_fixes.md) — **✅ CONCLUDED.** All P0/P1/P2 fixes shipped (PRs #330/#331); Exp 37 concluded. P3 superseded by `MAXIM_DISABLE_IMAGINATION`.
- [archive/substrate_exploration_policy.md](archive/substrate_exploration_policy.md) — **✅ SHIPPED + VALIDATED.** Exploration policy landed with Exp 41 plumbing (PR #379); Exp 42 GRADUATED using it (PR #380).
- [archive/substrate_primary_cradle_readiness.md](archive/substrate_primary_cradle_readiness.md) — **✅ CONCLUDED.** B1–B5 all resolved; Exp 41 VOID → Exp 42 GRADUATE. Open question A1 handed to the Exp 44 G1 work.
- [archive/sem_world_enrichment.md](archive/sem_world_enrichment.md) — **✅ Phases 1–3 substantially SHIPPED** (`archetype.py`, 7 archetypes, 75 tagged seeds). Only the `maxim_sim_avatar` default-embodiment item was descoped.
- [archive/roy_5_encoder_alignment_disambiguator.md](archive/roy_5_encoder_alignment_disambiguator.md) — **✅ Stages 1–4 ALL RESOLVED.** H1a confirmed (384-dim vs 768-dim); Stage 3 scaffold shipped (PR #295) and adjudicated by Exp 35/36. The open "Stage 5" question is new-plan-shaped.
- [archive/cross_modal_substrate_binding.md](archive/cross_modal_substrate_binding.md) — **❌ CANCELLED, do-not-resurrect.** Roy-4 found zero would-have-bound edges; the resurrection condition was then falsified by Exp 36 (gap closure = EC drift fix, not binding). Cross-modal work continues via jepa_cross_modal_alignment (deferred).
- [archive/cluster_reward_bias_decay_tau_split.md](archive/cluster_reward_bias_decay_tau_split.md) — **✅ All 5 phases COMPLETE** (PR #267 + Exp 30 validation). Downstream gaps handed to sense_tool_registry + imagination_substrate_signals (both since MVP-shipped).

Previously archived (2026-07-15 — plans consolidation + shipped/concluded sweep):
- [archive/audiovisual_orienting.md](archive/audiovisual_orienting.md) — **MERGED UP into [substrate_native_orienting.md](substrate_native_orienting.md).** Its coordination content (shared orient backbone, two-signal split, Phase 0b credit resolution, phased sequence, fusion endgame) is preserved in the new umbrella plan.
- [archive/release_0_9_1.md](archive/release_0_9_1.md) — **✅ 0.9.1 SHIPPED (2026-05-25).** All stages shipped; Roy-3 outcome recorded; follow-ups redirected to `sense_tool_registry.md` / `imagination_substrate_signals.md` / `cluster_reward_bias_decay_tau_split.md`.
- [archive/config_unification.md](archive/config_unification.md) — **✅ SHIPPED (PR #318, 2026-06-03).** `config.json` operator layer + `resolve_setting` + `maxim config` + role-detector unification.
- [archive/leader_ux_profile_management.md](archive/leader_ux_profile_management.md) — **✅ SHIPPED (PR #314, 2026-05-31).** Bundled + user profiles + `maxim model` verbs.
- [archive/ec_centroid_drift_fix.md](archive/ec_centroid_drift_fix.md) — **✅ SHIPPED (2026-05-24).** Text-modality centroid drift fix (threshold 0.44), PRs #259–#264.
- [archive/auth_format_freeze_audit.md](archive/auth_format_freeze_audit.md) — **✅ SHIPPED (2026-06-15, CC13).** Four security-shaped format-freeze surfaces.
- [archive/doctor_robot_reachable.md](archive/doctor_robot_reachable.md) — **✅ IMPLEMENTED (2026-07-15, PR #392).** `check_robot_reachable` doctor check (resolve → :8000 → daemon status → era-coherence).
- [archive/exp37_metric_pivot.md](archive/exp37_metric_pivot.md) · [archive/exp37_sd_shift.md](archive/exp37_sd_shift.md) · [archive/exp37_cross_model_characterization.md](archive/exp37_cross_model_characterization.md) · [archive/counter_prior_substrate_experiment.md](archive/counter_prior_substrate_experiment.md) — **✅ Exp 37/38/40 CONCLUDED** (Goldilocks zone + counter-prior dominance; 1.0 gates settled PR #371, 2026-06-15). These metric/statistical/pre-reg amendments served their purpose.
- [archive/cloud_dispatch_debug.md](archive/cloud_dispatch_debug.md) — **✅ SERVED PURPOSE.** Cloud-dispatch path validated during the Exp 37 cross-model window.


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

Each implementation plan receives two independent pre-merge agent reviews: one
execution/correctness lens and one architecture/maintenance lens. Findings fold into
the implementation branch before the PR opens. The review protocol must be tracked in
the repository; private `.claude/projects/.../memory` files are historical evidence,
not load-bearing instructions.

## Rules for this directory

- **Active plans stay in the root.** Anything in the root is on the critical path or actively shipping.
- **`roy/` is not a plan.** It holds `roy_0_smoke.yaml`, a scenario fixture consumed by the test suite (`test_roy_log.py`, `test_roy_runner.py`, `test_curriculum_runner.py`) — it stays put.
- **Deferred plans must state a revive trigger.** No trigger = archive, not deferred.
- **No ghost plans.** If a plan references a module that doesn't exist, fix or delete the reference.
- **Merge before multiplying.** If two plans overlap by more than a phase, merge them.
