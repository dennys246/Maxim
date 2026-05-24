# v1 Refinement — Validation + Stabilization + Cleanup for 1.0 Release

**Status:** PLANNING (pre-v1.0 release)
**Target version:** 1.0
**Branch:** TBD

---

## Motivation

1.0 claims "cross-session learning without fine-tuning." All substrate gates are closed (P1-P8, B4, behavioral convergence 41/41). What remains is:

1. **Proving the claim end-to-end** in a user-facing scenario
2. **Stabilizing the bio-systems** — every core bio-system (SCN, SEM, PainBus, NAc) must have standardized protocols and fully operational feedback loops before the interfaces freeze at 1.0
3. **Grounding basic knowledge** — the agent should understand fundamental sensorimotor truths (fire hurts, falling damages) through its own bio-pipeline, not LLM world knowledge
4. **Enriching the world layer** — SEM entities need rich environments to learn from, not bare-bones single-entity scenes
5. **Closing pipeline gaps and removing silent backward compat**

---

## Outstanding for 1.0 (as of 2026-05-03)

Substrate work is fully shipped (V1+V2 + B1+B2+B4 + P1-P4 + CC1-CC12 + C1-C3). What remains:

**Hard requirements:**
- **C4-C6** (Section 5) — Cleanup deprecation cycle. Needs a 0.9 release shipping parse-time / construction-time warnings before 1.0 flips them to hard errors. C4 SHIPPED (PR #219). C6 SHIPPED (2026-05-03). C5 design intact in §C5. C4/C6 hard-error flip prerequisites tracked in §1.1-T4.
- **D1-D3** (Section 6) — Docs passes (agent memory transfer, API/CLI surface review, final docs pass). No code, just writing.

**Optional polish:**
- **B3 Phase 3** (Section 2) — Composable body archetypes. YAMLs partially landed; `maxim_sim_avatar` migration pending. 1.0 vs 1.1 scope decision still open. Doesn't gate the substrate-attribution claim.

**Continuing past 1.0** (not 1.0 gates — listed in Section 8 "1.1 track" + per-plan partial-status headers):
- bio_emergent_persona_foundations Stages 0-3 full implementation (1.0 shipped reservations only per V1 Phase A clean-pass branch)
- persona_cleanup_and_mode_transition Stages 2-6 (1.0 shipped Stage 1 deprecation only)
- scene_actor_affordances Stages 3-5 (1.0 shipped Stages 1+2 only)
- 1.1-T1 Minecraft live demo + harness benchmark
- 1.1-T5 Agent-backed entities revival
- 1.1-T6 B5 embodiment/narrative separation
- MCP compatibility (CC9 prereq shipped)
- **1.1-T7 Substrate-primary AUT mode** — full Phase 0 validation, Phase 1 (vocabulary-constrained), Phase 2 (symbol binding) per [grounded_language_acquisition.md](grounded_language_acquisition.md). Phase -1 + Phase 0 harness ship in 1.0 (see new B5 below); the substrate-primary AUT mode itself ships in 1.1.
- **1.1-T8 Maxim Oasis** — first hostable Oasis instance per [maxim_hivemind.md](maxim_hivemind.md). Single-Oasis software (~800 LOC); LLM-AUT users opt in to contribute substrate via `maxim contribute --to oasis://...`; direct Oasis-to-Oasis sync supported (no mesh discovery yet). Builds on B5's shareability infrastructure.
- **1.2 Maxim Hivemind protocol** — peer-to-peer substrate exchange + conflict resolution + poison resistance (~600 LOC). Substrate-primary Maxims pull bootstrap from Hivemind, contribute back as they learn. Multi-Oasis federation goes live.

**New 1.0 add (parallel to docs work):**
- **B5. Substrate-primary AUT mode — Phase -1 + Phase 0 harness + Hivemind shareability infrastructure** (see [grounded_language_acquisition.md](grounded_language_acquisition.md) for substrate-primary phases; [maxim_hivemind.md](maxim_hivemind.md) for the Hivemind/Oasis layer that B5's shareability infrastructure enables). Ships under experimental flag (`--aut-mode substrate-primary` or similar). **Three components:**
  - **Phase -1** (~150 LOC) — **GATE CLEARED + SHIPPED** (PR #228, 2026-05-09). `NAc.recommend_action()` + `propose_via_substrate()` + `--aut-mode` CLI plumbing. Boolean YES on substrate-driven action generation. 22 tests across unit + integration.
  - **Phase 0 harness** (~550 LOC) — **SHIPPED** (PR #228, 2026-05-09). `cradle_prelinguistic` arc + motor-only AUT prompt renderer + `SubstrateTelemetry` JSONL writer + `--research` routing fix. 13 harness tests. Smoke run cleared: 38 actions, 61 causal links, hunger drift 0.0 → 0.65 over 5 turns. Validation gap: substrate-primary bypasses `LinguisticEncoder` so EC `node_count` stayed at 0 — the cluster-formation measurement Phase 0 actually wants is blocked on a sensor-percept encoding entry point. Next concrete work item; small (~1-2 sessions).
  - **Hivemind shareability infrastructure** (~660 LOC) — **PENDING.** Substrate snapshot bundle format (zip + manifest + signature, ~150) + `nac.merge()` / `ec.merge()` Bayesian-aggregation library functions (~200) + provenance tags on NAc links + EC nodes (~100) + identity-bearing concept detection ported from old Mother plan (~80) + substrate domain tagging (~50) + `maxim substrate export` / `maxim substrate import` CLI verbs (~80). Enables the 1.1 Oasis software and the 1.2 Hivemind P2P protocol without retrofitting.
  - **Scope shipped to date:** ~700 LOC of ~1,360. Hivemind shareability (~660 LOC) remains.
  - **Doesn't touch user-facing 1.0 surface; doesn't gate D1-D3 docs.** Motivated by the 2026-05-09 audit that found 60-70% of recent engineering effort going to LLM-mitigation scaffolding (~845 LOC of band-aids) rather than substrate work — the parallel-mode architecture is the structural fix to the drift, and baking shareability in from day one means the Maxim Hivemind ships as a 1.2 turn-on rather than a 1.3+ retrofit.

**Net assessment:** 1.0 substantively done. Path to ship is one 0.9 release carrying C4/C5/C6 deprecation warnings → deprecation window of operator's choosing → 1.0 release flipping warnings to hard errors. D1-D3 docs work proceeds in parallel with B5 (substrate-primary harness); B3 Phase 3 is optional polish. No more substrate uncertainty, no more confound risk, no more unshipped invariants. The B5 add accepts a small scope expansion to begin the parallel-architecture pivot; 1.0 still ships when D1-D3 + C4/C5/C6 deprecation cycle complete, regardless of where B5 lands (B5 finishing in 1.0 is preferred but not gating).

---

## Section 1: Validation (prove the claim)

### V1. Cross-session sim validation — VALIDATED (2026-04-26)

**Absorbed from:** `cross_session_sim_validation.md`
**Results:** [docs/experiments/10_cross_session_enrichment.md](../experiments/10_cross_session_enrichment.md)
**Status:** PARTIAL PASS — memories surface (3/turn on resume), predictions/concepts/affordances need more session history.

Prove that Layer 1 pre-deliberation enrichment produces measurably different behavior when the agent has prior session history vs. a fresh start.

**Experiment design:**

| Phase | Command | Purpose |
|---|---|---|
| 1. Baseline (fresh) | `maxim --sim "escape a dungeon with a sleeping guard" --interactive false --sim-max-turns 8` | Record session_id, action sequence, hippocampus captures, NAc links |
| 2. Resume (cross-session) | Same goal + `--resume-sim <session_id>` | Compare: does "WHAT YOUR EXPERIENCE TELLS YOU" populate? Do actions differ? |
| 3. Negative transfer | Different scenario + `--resume-sim <session_id>` | Verify dungeon memories don't dominate unrelated garden scenario |

**What to measure:**

| Metric | Fresh | Resume | Expected |
|--------|-------|--------|----------|
| Enrichment sections populated | 0-1 (WMS only) | 3-4 (memories + predictions + concepts + WMS) | Resume >> Fresh |
| NAc predictions in prompt | 0 | 2-5 | Resume has learned causal links |
| Hippocampal recalls in prompt | 0 | 1-3 | Resume has episodes to retrieve |
| Action diversity | ~0.3 | ~0.5+ | Prior experience prevents repetition |

**Success criteria:**
1. Resume session shows 2+ enrichment sections populated that were empty in fresh session
2. Action sequence in resume differs in a way traceable to enrichment content
3. Negative transfer test shows enrichment doesn't dominate unrelated scenarios

**Results doc:** `docs/experiments/10_cross_session_enrichment.md`

**Substrate prerequisite — EC centroid drift fix (2026-05-24, PRs #259–#264).** Pre-fix V1 runs silently degraded as more text accumulated on a shared EC: the text-modality `pattern_complete_threshold` at 0.40 admitted marginal paraphrases (cosine 0.42-0.48) whose centroids drifted toward a generic "second-person body sensation" prototype that then pattern-completed everything subsequent. Cross-session recall would therefore drift toward "anything second-person-sensory" the more text the substrate had seen — the V1 result "3 memories/turn on resume" was measured under this regime. The Phase 1-5 fix in [ec_centroid_drift_fix.md](ec_centroid_drift_fix.md) bumped the default to 0.44, structurally pinned the coupling, parameterized the NAc override base, and behaviorally validated on Roy-2c. The Roy-2c behavioral signal didn't move (the H1 cross-source alignment gap that Roy-2c measures is structurally upstream of drift — see [27_ec_drift_phase_4_behavioral.md](../experiments/27_ec_drift_phase_4_behavioral.md)) but the structural improvement is real (a_vs_b cluster_reward_bias_l2 2.566 → 0.535, six +1.0 priming UUIDs collapsed to one near-zero cluster). **V1 re-runs after this fix lands should show steadier cross-session recall — fewer spurious sibling clusters per concept means fewer near-miss recall failures driven by drifted-centroid pattern-completion onto the wrong prototype.** Re-running V1 on post-fix substrate is not strictly required for the 1.0 claim (V2 Phase A already passed under V1's contaminated regime, which is a stronger result than passing on the cleaner substrate), but a sanity-check re-run is on the table for any future V1 audit.

### V2. Confound quarantine for V1 re-run — gates substrate attribution claim — **CLEAN PASS** (2026-04-30, PR #214 + Experiment 12)

**Companion plan:** [confound_quarantine.md](confound_quarantine.md)
**Companion experiment:** [docs/experiments/12_v1_phased_attribution.md](../experiments/12_v1_phased_attribution.md)
**Status:** SHIPPED + VALIDATED. Implementation: PR #214 (confound flags + report block + tests). Re-run: 7 phases run 2026-04-30 against commit `f742527` over peer-routed qwen2.5-14b. **All 7 phases (7/7) successfully recalled `BLUE-7-DAWN` across sessions, including Phase A (substrate-only baseline)**, see Experiment 12 for the full table + recall evidence excerpts.

**Disposition (per §R1 clean-pass branch):** flags removed in 1.0. `MAXIM_DISABLE_PFC_PREAMBLE`, `MAXIM_DISABLE_ACTING_COACH`, `MAXIM_DISABLE_SIM_SANDBOX_TEXT`, `MAXIM_NO_DEFAULT_PERSONA`, `--no-acting-coach`, `--no-persona`, `runtime/confound_flags.py`, the `report.json::confound_quarantine` block, and the test surface (autouse scrub + `test_confound_flags.py` + `test_v1_phased_metrics.py`) are scheduled for removal in a follow-up `chore(v1): remove confound flags after Phase A clean pass` PR. `--no-embodiment` and `MAXIM_DATA_HOME` are unrelated escape hatches and stay. The harness `scripts/run_v1_phases.sh` and the experiment doc remain in-tree for academic-ML reproducibility — re-running against post-removal main shows the env vars no-op; checking out `f742527` reproduces the full protocol.

A multi-lens audit (default-on prompt injectors / goal-keyword path divergence / hidden state leakage) found **five default-on confound systems** that fire silently on every `maxim --sim` invocation, including V1. The original V1 PARTIAL PASS result is contaminated: it cannot be attributed to the substrate alone while four other systems are simultaneously shaping AUT behavior.

**The five confounds:**

| # | Confound | Site | Impact |
|---|---|---|---|
| 1 | PFC deliberation preamble (~1k token reasoning scaffold w/ hardcoded "fight dragon → slash" example) | [exec_prompts.py:13](../../src/maxim/agents/exec_prompts.py#L13), gate at [prompt_builder.py:979](../../src/maxim/agents/prompt_builder.py#L979) | Primes combat behavior on every embodied sim. *Widened* on 2026-04-24 to fire on cold-start. |
| 2 | Acting Coach + embodied identity rewrite (`role_values=("curiosity","survival")` baked in) | [prompt_builder.py:117](../../src/maxim/agents/prompt_builder.py#L117), [acting_coach.py:73](../../src/maxim/prompts/acting_coach.py#L73) | Pre-decides what counts as a good outcome — the very thing affordance-learning experiments measure. |
| 3 | "SIMULATION ENVIRONMENT — all tool actions are sandboxed and safe" prompt block | [prompt_builder.py:148](../../src/maxim/agents/prompt_builder.py#L148) | Plausibly deflates risk-aversion behavior. |
| 4 | Orchestrator NPC global state at `~/.maxim/orchestrator/` | [orchestrator.py:1181](../../src/maxim/simulation/orchestrator.py#L1181) | Persists across every sim run. Run N+1's narrator carries learned biases from Run N. |
| 5 | Persona default `adversarial` | [personas.py:352](../../src/maxim/simulation/personas.py#L352) | Already known. The original 9-experiment confound. |

Plus second-tier: arc keyword routing (goals containing "learn"/"memory"/"cradle" silently route through generative narrator path); default `bodies/base_humanoid` injection; global `~/.maxim/util/{semantic_embeddings.npz, escalation_learning.json, fear_learning.json, ...}` bridge state.

**Phased re-run protocol** (full design in [confound_quarantine.md](confound_quarantine.md) §"Phased re-run protocol"):

| Phase | Description |
|-------|-------------|
| **A** (substrate-only baseline) | All confound flags ON (disable everything), isolated `MAXIM_DATA_HOME` |
| B–F | Re-enable each confound one-at-a-time; measure delta vs Phase A |
| **G** (control) | All defaults ON, isolated `MAXIM_DATA_HOME` only — confirms isolation alone doesn't move metrics |

The phase deltas attribute the V1 result to specific contributors. Phase A is the actual substrate-attribution number for the 1.0 claim.

**Existential risk flagged:** if Phase A shows the cross-session recall signal disappears once the scaffold is removed, the 1.0 substrate-attribution claim must be re-scoped. That's the point of the experiment — discover this before 1.0, not after.

**Flag lifecycle — decided in 1.0, not deferred.** Flags ship in 0.9.x as experimental. Their disposition is forced at 1.0, conditional on Phase A results: (a) clean pass → flags removed in 1.0, V1 reproducibility via 0.9.x commit-hash pinning; (b) conditional pass → flags graduate from experimental to public-stable, claim documents which scaffolds it depends on; (c) fail → re-scope the 1.0 claim, keep flags as evidence. No experimental limbo through 1.1+.

**Why before 1.0:** the substrate-attribution claim is the central 1.0 marketing claim. Shipping it on contaminated data is a credibility risk. The flag surface is ~40 production LOC; the test surface and harness are larger but cheap.

---

## Section 2: Bio-System Stabilization (freeze-worthy interfaces)

These are the systems that must be fully operational and standardized before 1.0 freezes the interfaces. Post-1.0, interface changes are expensive.

### B1. Bio-system protocol enrichment — SHIPPED (2026-04-26)

**Companion plan (archived):** [archive/bio_system_protocol_enrichment.md](archive/bio_system_protocol_enrichment.md)
**Shipped surface:** all 5 `*Context` frozen dataclasses live at [src/maxim/models/bio_context.py](../../src/maxim/models/bio_context.py).

Add standardized `*Context` dataclass parameters to each bio-system's primary methods NOW (optional, defaults to None) so future backends can consume richer input without Protocol breaks. Current implementations ignore the context; future ones use it.

**Systems enriched:**
- Hippocampus: `RetrievalContext` (temporal, emotional, goal state, consolidation level)
- NAc: `PredictionContext` (arousal, temporal discount, context richness)
- ATL: `SemanticContext` (abstraction level, domain hints)
- EC: `EncodingContext` (decomposition strategy hints, resolution)
- SCN: `TemporalContext` (oscillator phase, prediction confidence)

### B2. SCN oscillator feedback — anticipatory temporal credit — SHIPPED (2026-04-26)

**Companion plan:** [scn_oscillator_feedback.md](scn_oscillator_feedback.md) (~120 LOC)
**Branch:** `feat/v1-scn-oscillator`

Closed the SCN→NAc feedback loop. Three-path credit in TemporalCreditDistributor: fast-decay → phase-similarity → anticipatory (oscillator-predicted imminent events). 21 new tests.

**Changes shipped:**
1. `OscillatorNetwork.observe_event()` — per-event-type circadian phase tracking
2. `OscillatorNetwork.predict_event_imminence()` — circular mean + concentration scoring
3. `TemporalCreditDistributor.distribute()` — third anticipatory credit path
4. `build_bio_stack` — `scn.enable_oscillator()` by default

### B3. SEM world enrichment (Phase 3 only — Phases 1+2 shipped)

**Companion plan:** [sem_world_enrichment.md](sem_world_enrichment.md)

Phase 1 shipped (scene manifest pre-trigger via `ImaginationTrigger.process_manifest`, head-noun alias fallback). Phase 2 shipped (`BioEnrichmentPipeline` consumes `resolved_entities` and queries `ComponentIndex.find_alias_only` to enrich percept context with SEM-aware affordances; `agent_loop.py` forwards the imagination resolution into the enrichment context).

**Remaining (1.0 vs 1.1 scope decision pending):**

- **Phase 3: Composable body archetypes** — Mix-and-match body templates (biped + wings, quadruped + tail weapon) so imagination can compose novel entities from known parts rather than designing from scratch. Archetype YAMLs partially landed in `_data/components/archetypes/`; the `maxim_sim_avatar` body migration and the `archetype: <name>` field plumbing on existing components are not yet done.

---

## Section 3: Sensorimotor Grounding (the agent understands its world)

### B4. Cradle of Artificial Civilization

**Companion plan (archived):** [archive/cradle_sensorimotor_development.md](archive/cradle_sensorimotor_development.md). **SHIPPED 2026-04-25** (PR #200), validated by [exp 11](../experiments/11_cradle_sensorimotor_poc.md).

A newborn agent that learns from sensation, not language. Three-layer sensation model (contact via entity acquisition, proximity via orchestrator sensor writes, narrative fallback via keyword reflexes) all converging on the same downstream pipeline. Multi-act developmental scenario mimicking Piaget's sensorimotor stages.

**Key deliverables:**
1. **Drive protocol** — `HomeostaticDriveSpec` (temperature, pressure, stamina — body self-regulates toward set point) + `EntropicDriveSpec` (hunger, thirst, fatigue — drift away from equilibrium). Interface-level `CouplingSpec` and `ModulationSpec` ship for 1.0 freeze, implementation deferred.
2. **Three-layer sensation model** — standardized external→internal signal translation. Contact (entity acquisition / Mechanism B from proprioceptive_discovery.md), proximity (orchestrator sensor writes), narrative fallback (keyword reflexes). All produce sensor changes → `evaluate_failures()` → PainBus → NAc.
3. **Energy system unification** — `EnergyReactionBridge` replaced by generic drive protocol. `MovementEnergyTracker` → writes stamina sensor (homeostatic drive). Prevents two parallel threshold systems from freezing in 1.0 API.
4. **Narrative acts** — `act` + `world_entities` fields on `NarrativePhase`. Cradle uses 4 developmental acts. General sim framework for long-horizon narrative structure.
5. **Validation experiment** — 4-act developmental scenario with 8 measurements, 8 pass/fail criteria. Cross-session transfer via `--resume-sim`.

---

## Section 4: Pipeline Completion (small gaps)

### P1. ToolPainBridge temporal event migration (~50 LOC) — SHIPPED (2026-04-25)

**Absorbed from:** `tool_pain_bridge_temporal_migration.md`
**Status:** COMPLETE. 7 new tests, 5925 tests passing. Deep 2-lens review folded.

ToolPainBridge has 4 `scn.register()` call sites that write temporal signatures to SCN bins but never emit `TemporalEvent`s for credit attribution via the `TemporalCreditDistributor`.

**Changes:**
1. Add `distributor: TemporalCreditDistributor | None = None` to bridge constructor
2. At each SCN registration site, also emit a `TemporalEvent`
3. Thread `Reaction.scn_tag` (CircadianContext) through the reaction subscriber, converting to `TemporalSignature` at the boundary

**Constraint:** Existing NAc calls stay unchanged. Temporal event emission is additive.

### P2. Episode boundary — semantic shift detection (Stage 2) — SHIPPED (PR #208, 2026-04-28)

**Shipped via:** PR #208 (initial) + PR #208 follow-up (review fold)

**Absorbed from:** `substrate_episode_boundary_enrichment.md` Stage 2

Add semantic-shift episode boundary detection. When incoming text embedding diverges from the episode centroid, close the episode.

**Changes:**
1. Add `embedding: ndarray | None = None` to `CaptureEvent`
2. Add `centroid_embedding: ndarray | None = None` to `PendingEpisodeState` with incremental centroid update
3. Implement `semantic_shift_rule(threshold=0.40)` in `episode.py`
4. Calibration sweep against real conversational data

### P3. Dead energy code removal + drive protocol migration — SHIPPED (PR #200, 2026-04-26)

**Shipped via:** PR #200 (cradle Stage 1c, per the original plan)

**Audit finding:** `EnergyReactionBridge` (125 LOC) and `MovementEnergyTracker` (362 LOC) are **completely dead code** — zero callers anywhere in the codebase. Never instantiated, never wired. Additionally, 6 of 10 `EnergyType` enum values are unused (COMPUTE_TIME, MOTOR_CURRENT, VISION_INFERENCE, AUDIO_PROCESSING, ATTENTION, MEMORY_ACCESS).

**Changes:**
1. Hard-remove `energy/reactions.py` (entire file — EnergyReactionBridge, create_energy_reaction_bridge)
2. Hard-remove `energy/movement_tracker.py` (entire file — MovementEnergyTracker, MovementEnergyConfig)
3. Remove 6 unused `EnergyType` enum values from `energy/signal.py`
4. Fix `simulation/introspection.py:149` — `get_stats()` → `get_summary()` (AttributeError bug)
5. The generic drive protocol (B4 Stage 1) replaces what the bridge was SUPPOSED to do

**What stays (live code):** `LLMEnergyTracker` (wired in llm_worker.py), `EnergyRegistry` (budget gating + imagination energy gate), `EnergyBudget`, `EnergySignal`.

### P4. Multi-agent learning attribution gaps (~50 LOC) — SHIPPED (PR #202, 2026-04-28)

**Shipped via:** PR #202 (`feat(v1-p4): per-agent stash + agent_id attribution`)

**Surfaced by:** [deferred/agent_backed_entities.md](deferred/agent_backed_entities.md) audit. Independently necessary regardless of cast direction.

Two correctness gaps in multi-agent paths surfaced while investigating SEM-backed Maxim entities. Both gaps would manifest the moment any path runs more than one cognitive agent in parallel — including the AUT + orchestrator pair shipped today, though the orchestrator's hippocampus is currently disabled so the gap is latent.

**Changes:**

1. **`bio_integration.py` global stash → per-agent dict.** Replace module-level `_latest_substrate_nodes` and `_latest_pain_intensity` ([bio_integration.py:175-205, 247-259](../../src/maxim/memory/bio_integration.py#L175-L259)) with per-`agent_id` dicts at the `Hippocampus.observe_episode_event()` callsite. Code comment currently admits "acceptable because multi-agent substrate encoding is not yet a production path" — this lifts the bound.

2. **`agent_id` on `tool_dispatch.record_outcome()`.** Add `agent_id: str` parameter ([tool_dispatch.py::record_outcome](../../src/maxim/runtime/tool_dispatch.py)) and update all 7 callers in `agent_loop.py` (per `feedback_record_outcome_call_sites.md`). Today `ToolPainBridge` handles attribution via direct lookup; this closes the loop-level gap.

**Test:** two parallel `AgentFactory` agents running `run_agentic_loop` simultaneously must produce isolated learning, verified by inspecting their separate `nac.json` files for non-overlapping causal links.

---

## Section 5: Cleanup (breaking changes)

Several backward-compatibility shims silently accept under-specified inputs. Per CLAUDE.md "push silent-no-op invariants into types, not helpers," these are silent-failure risks.

**Timeline contract (refined 2026-04-28):**
- **C1-C3 (internal hard-removes):** ship in **0.9 or 1.0**. Zero user impact, no deprecation cycle needed — these only remove internal shims.
- **C4-C6 (user-facing breaking changes):** ship **deprecation warnings in 0.9, hard errors in 1.1**. NOT in 1.0. Per semver discipline, breaking-change-on-X.0 is fine when it's the first major release, but 1.0 should be the stable contract — users adopting 1.0 should not be surprised by 1.0.1 or 1.1 hard errors. The 0.9 deprecation cycle gives users a release to react.

### C1. Probe compat shim removal (internal) — SHIPPED (2026-04-26)

**Shipped via:** PR #196 (commit `1f20df7`)

Removed `probe_llm_server`, `llm_server_responding_at`, `_probe_once` from `runtime/llm_server.py` (-271 LOC). Probe primitives (`_probe_once`, `_probe_stage2_readiness`, `_build_probe_url`, `_classify_probe_cause`) moved into `models/language/maxim_peer_backend.py` where `health_check()` lives. 4 production callers migrated to `_MaximPeerBackend.for_url(url, api_key=k, model=m).health_check()` (lane_backends.py ×2, doctor/checks.py ×2). Eliminated the bidirectional lazy-import cycle between `llm_server.py` and `maxim_peer_backend.py`. CI grep allow-list at `.github/workflows/test.yml` updated to zero-tolerance — any re-introduction now fails CI.

### C2. `SendMessageTool._detect_attack` dead code removal (internal) — SHIPPED (2026-04-26)

**Shipped via:** PR #196 (commit `1f20df7`)

No-op at ship time — the reflex system landing in 0.8 had already removed `_detect_attack()` and `_ATTACK_KEYWORDS`. Verified zero hits across `src/maxim/` and `tests/`.

### C3. `DamageEntityTool` shim removal (internal) — SHIPPED (2026-04-26)

**Shipped via:** PR #196 (commit `1f20df7`)

Removed `DamageEntityTool` class, import, registration, and `TOOL_DESCRIPTIONS` entry from `simulation/tools.py`. Removed from `docs/user/tools.md`. Orchestrator prompts in `simulation/orchestrator.py` updated to use `damage_component`. `DamageComponentTool` is the sole damage tool going forward.

### C4. Modulators without sensors (deprecation phase) — SHIPPED (PR #219, 2026-05-03)

Require every modulator to declare at least one sensor. Capability-only modulators declare `abstract: true`. 0.9 deprecation warning, 1.x hard error.

**Shipped:** warning lives in `SpecModulator.__init__` (not `_parse_entity`) per CLAUDE.md "push silent-no-op invariants into types, not helpers" — covers parser, `Entity.from_dict`, foundry-generated specs, future programmatic builders. The 1.x flip is one-line. Symmetric `to_dict` emission of the `abstract` boolean. `Entity.from_dict` reconstructs per-modulator sensors (pre-existing roundtrip gap that the new constructor warning would have made spuriously fire) + legacy-pre-C4 dict-shape compat (no `sensors` and no `abstract` → load as `abstract=True`). Bundled audit: 115 modulators in `_data/components/` + 11 in `scenarios/embodiment/` declare `abstract: true`. Regression-guarded by `test_bundled_components_emit_no_deprecation_warning`. Pre-merge two-lens review folded.

**Known follow-ups (track for 1.x track, before the hard-error flip in §1.1-T4):**

- **C4-followup-1: Imagination + foundry pipelines emit bare LLM specs.** `imagination/trigger.py` (3 call sites) and `simulation/foundry.py` (2 call sites) call `_parse_entity` on LLM-generated specs that don't declare `abstract: true` on capability-only modulators. v0.9 surfaces deprecation warnings on every imagined entity — the intended deprecation signal. v1.x flip will hard-fail those entry points until fixed. **Fix shape:** small post-process normalizer in `spec.py` (`if not mod.get("sensors") and "abstract" not in mod: mod["abstract"] = True`) called at the 5 LLM-spec entry points + update `simulation/entity_designer.py` JSON template to ask the LLM to emit `abstract: true` for capability-only modulators. The post-process is a safety net for LLM forgetfulness; the prompt update is the right-shaped ask. Hard 1.x deadline (warning → error). ~50-100 LOC.

- **C4-followup-2: Sensor-promotion audit.** The 115 bundled modulators were marked `abstract: true` uniformly by an audit script. A small subset (~5-15) arguably should grow real sensors instead so `compute_integrity()` reflects "this capability is degraded" — `cradle_lever_door.mechanism` should own `lever_position`, `wizard.magic` should own `mana`, etc. **Approach:** group the 115 by modulator-name category (combat/social/maintenance/usage are clearly verbs and stay abstract — ~95+ of them); the real audit shrinks to the ~15 ambiguous ones (`magic`, `mechanism`, `lifecycle`, `physical`, `expression`). For each: does the entity carry sensors that conceptually belong to this modulator's working order? Net diff is small (5-10 promotions); the architectural signal it sends ("we know what state belongs where") is large. Polish pass — no hard deadline, can land any time before 1.x.

### C5. Entity health as direct sensor (deprecation phase)

If entity has modulators with sensors AND a direct `health` sensor (not `derived`), parse-time warning in 0.9, hard error in 1.0.

### C6. Raw constructor enforcement (deprecation phase) — SHIPPED (2026-05-03)

**Shipped:** Added keyword-only `_allow_raw: bool = False` to `PainBus.__init__`, `ReactionBus.__init__`, and `MemoryHub` (as a `kw_only=True` dataclass field, kept out of `repr` / `compare`). When `_allow_raw` is False, each constructor emits a `DeprecationWarning` naming the canonical builder (`build_pain_bus` / `build_reaction_bus` / `build_memory_hub`) plus the load-bearing rationale (Wave 1+2 silent-no-op bug class). Each warning also prints to stderr (`print(f"DeprecationWarning: {msg}", file=sys.stderr)`) for human visibility — `DeprecationWarning` is silenced by Python's default warning filter outside `__main__` and by pytest's global `ignore::DeprecationWarning`, so the stderr line is the load-bearing signal that callers actually see. Mirrors the C4/C5 + `cli_utils._resolve_persona_mode` pattern.

`build_pain_bus`, `build_reaction_bus`, and `build_memory_hub` now pass `_allow_raw=True` internally so production paths are silent. `PainBus.__init__`'s internal `ReactionBus(...)` call also passes `_allow_raw=True` — only the *outer* type's warning fires (one warning per raw construction, not two).

`simulation/orchestrator.py::_setup_sim_sandbox` migrated from raw `PainBus()` to `build_pain_bus(hippocampus=None, nac=None)` — the early sandbox bus pattern routes through the canonical door. `default_network/network.py::_init_pain_circuit` keeps its raw `PainBus(_allow_raw=True)` opt-out with an explicit `TODO(wave-2)` comment naming the deferred plan (`pain_bus_unification.md` "Latent bridge × subscriber attribution-asymmetry trap" + this doc's §1.1-T4 C6 prerequisite).

The `ReactionBus` warning text deliberately distinguishes itself from `PainBus`/`MemoryHub`: it has no current production silent-no-op bug class (PainBus constructs ReactionBus internally), but the door is enforced now to be forward-protective for the Wave-3 ordering where `build_bio_stack` will construct a standalone ReactionBus and hand it to `build_pain_bus(reaction_bus=...)`. The text says so explicitly so a user hitting the warning doesn't hunt for a non-existent retroactive bug.

CI grep guard at `.github/workflows/test.yml` blocks new `_allow_raw=True` opt-outs in `src/maxim/`. The allow-list covers exactly the 5 legitimate sites (4 internal builder calls + DefaultNetwork's deferred opt-out). New production opt-outs require updating both the allow-list and CLAUDE.md in the same commit, with a `TODO(wave-N)` comment at the call site. Same shape as the existing `write_mesh_config` / install-core / admin-update / admin-llm allow-lists.

Test surface: ~60 raw-construct sites updated to `_allow_raw=True` across 15 test files + 5 scripts. Regression-guarded by `tests/unit/test_c6_raw_construction_warnings.py` (17 tests covering: raw-warns, `_allow_raw=True`-silent, builder-silent, kw-only-field-shape, builders-propagate-allow-raw-internally).

Pre-merge two-lens review: 5 findings folded (ReactionBus message clarification, stderr-print visibility, CI grep allow-list, DN TODO marker, plan §1.1-T4 prereq mirror). 1 NIT deferred (future tag-bypass-with-reason-string enhancement).

1.0 flip is tracked in §1.1-T4: change the default warning to a `TypeError` (or remove the parameter and require the builder).

---

## Section 6: Docs

### D1. Agent memory transfer docs

Universal onboarding document set for knowledge continuity across agents.

### D2. API/CLI surface review

Sweep `api.py`, `__init__.py`, `cli.py`, `cli_parser.py` for edge cases. Ensure all capabilities are accessible.

### D3. Final docs pass

Publication guide, user docs, architecture docs — ship-ready state.

---

## Execution order

The persona-cleanup track (1.1-T-persona-triad: scene_actor_affordances → bio_emergent → persona_cleanup) interleaves with the V1 re-run because Phase E of the dial-down protocol exercises persona-disabled runs and Phase F exercises embodiment-disabled runs. Both need infrastructure landed before the re-run is meaningful. Final ordering:

1. **B1, P1, P2** — already shipped; baseline.
2. **scene_actor_affordances Stages 1-2** (target_effect field on `AffordanceSchema` + `OrchestratorActorTool`) — absorbs the world-physics-engine job from the adversarial persona prompt, so killing persona doesn't break narrative→SEM coupling.
3. **confound_quarantine flags** — opt-in disable env vars + `--no-acting-coach` / `--no-persona` CLI flags + autouse scrub fixture + per-flag pin tests. ~40 production LOC. See [confound_quarantine.md](confound_quarantine.md).
4. **V1 phased re-run** (Phases A–G) — the dial-down experiment. Phase A produces the substrate-only baseline number for the 1.0 claim. Phase A's outcome forces the flag-lifecycle decision in 1.0 (clean pass / conditional pass / fail).
5. **bio_emergent_persona_foundations** — scope decided after Phase A. If Phase A reveals the substrate needs richer disposition mechanics (learned aversions, risk sensitivity), implement Stages 0-3 before 1.0. Otherwise reserve fields on `GatingContext` and `OutcomePrediction` and ship implementation in 1.1.
6. **persona_cleanup_and_mode_transition Stage 1** — additive `--mode` flag + deprecation warning on `--persona` and `register_persona`. Hard-remove in 1.1.
7. **B2** (SCN oscillator) — already shipped (PR #198).
8. **B3** (SEM world enrichment Phase 3) — Phases 1+2 shipped; Phase 3 (composable body archetypes) optional for 1.0.
9. **B4** (cradle) — already shipped (PR #200).
10. **C1-C3** — already shipped (PR #196, 2026-04-26).
11. **C4-C6** (deprecation phase) — 0.9 warnings, 1.0 hard errors.
12. **D1-D3** (docs) — last, after content stabilizes.

## Timing

- Items 2-6 are the new triad-plus-validation chain. Ordering is load-bearing: target_effect must exist before persona is disabled in Phase E; confound_quarantine flags must exist before V1 re-run; V1 results must land before bio_emergent and persona-cleanup scopes are decided.
- Items 7-11 are largely shipped or low-LOC; the critical-path constraint is the 2-6 chain.
- Date pressure is light per user direction (no external dependents on a specific 1.0 ship date) — foundation correctness matters more than calendar.

## 1.0 interface freeze checklist

These items must ship before 1.0 because they define frozen interfaces. Post-1.0, adding or changing fields is a breaking change:

- [ ] B1: `*Context` parameters on bio-system methods (shipped)
- [ ] B4 Stage 1a: `HomeostaticDriveSpec`, `EntropicDriveSpec`, `CouplingSpec`, `ModulationSpec` dataclasses
- [ ] B4 Stage 1a: `pain_model: str` field on `HomeostaticDriveSpec`
- [ ] B4 Stage 1c: Dead energy code hard-removed (prevents dead code from becoming frozen API surface)
- [ ] B4 Stage 3: `self_effect` YAML key on `AffordanceSchema`
- [ ] B4 Stage 4: `GatingContext.drive_states` field in `runtime/gating.py`
- [ ] B4 Stage 4: DRIVE annotations in `body_state_summary()` output format
- [ ] B4 Stage 5: `act` and `world_entities` fields on `NarrativePhase`

## Pre-1.0 but post-cradle (interface reservations that need implementation before freeze)

- [ ] `GatingContext.drive_states` scoring modulation in `TextSalienceScorer`
- [ ] Novelty tracker formalized as `HomeostaticDriveSpec` on a `novelty_drive` sensor

---

## Section 7: Contract clarification — freeze hardening (1.0 work, not features)

The architectural work for 1.0 is done. This section is about *clarifying which parts of that work are public contract vs internal implementation*, so post-1.0 changes don't accidentally break users or force a 2.0. None of these items change how Maxim behaves; all of them change whether we owe a 2.0 to fix mistakes.

**Why this section exists separately from Cleanup:** Section 5 removes things. This section adds stability markers, version fields, classification docs, and forward-compat hooks. Different work, different review lens.

### CC1. Persistence format versioning audit (~50 LOC + 1 test) — SHIPPED (PR #203, 2026-04-28)

**Shipped via:** PR #203 (+ `src/maxim/utils/format_version.py` + `tests/integration/test_persistence_compat.py`)

Some persisted JSON files reference `format_version` / `schema_version` (`nac.py`, `atl.py`, `scn.py`, `cross_layer.py`, `percept_trace_buffer.py`, `snapshot.py`, `hippocampus_persistence.py`); others may not. Standardize: every persisted JSON file carries a `_format_version: "1.0"` field at the root. Loaders default to `"0.x"` when absent and emit a one-time warning. Adds a single integration test that loads a pre-1.0 fixture and asserts no errors.

### CC2. Public API stability classification (mostly doc) — SHIPPED (PR #212, 2026-04-29)

**Shipped via:** PR #212 (bundled with CC4 + CC12)

[`api.py`](../../src/maxim/api.py) exports 17 verbs + ~10 dataclasses. Today there's no marker for "stable in 1.0" vs "experimental, may change." Add a `docs/user/stable_api.md` page listing what's frozen at 1.0. Anything not on that page is fair game for post-1.0 changes. Mark experimental functions in their docstring header with a clear note.

**Candidates for "experimental" tag:** `research()` (research orchestrator surface still evolving), the event subscription API (`on()`, `EventHandle`) if it's still maturing, anything else flagged during D2 review.

### CC3. Frozen dataclass forward-compat audit (~2 hours) — SHIPPED (PR #207, 2026-04-28)

**Shipped via:** PR #207 (initial) + PR #209 (review fold)

Every frozen dataclass shipping in 1.0 must have either (a) all fields with defaults, or (b) an `extra: dict[str, Any] = field(default_factory=dict)` escape hatch. Auditees: `PainSignal`, `Reaction`, `ReactionContext`, `PerceptContext`, `TraceSnapshot`, `AffordanceSchema`, `HomeostaticDriveSpec`, `EntropicDriveSpec`, `CouplingSpec`, `ModulationSpec`, `BackendError` subclasses, `HTTPError` subclasses, persisted dataclasses (Episode, MemoryRecord, etc.).

### CC4. Environment variable + CLI flag classification (~doc work) — SHIPPED (PR #212, 2026-04-29)

**Shipped via:** PR #212 (bundled with CC2 + CC12)

CLAUDE.md documents ~30 `MAXIM_*` env vars mixing public-contract (`MAXIM_LLM_PROFILE`, `MAXIM_DATA_BUDGET_GB`) with debug/internal (`MAXIM_BACKEND_TRACE`, `MAXIM_HEARTBEAT`). Classify in user docs:
- **Public** — stable in 1.0, removal is a breaking change.
- **Debug / experimental** — explicitly may change without notice; documented with that disclaimer.

Same treatment for CLI flags. Mark experimental flags in `--help` output (e.g., `[experimental]` suffix). Candidates: `--auto-curate`, `--research`, possibly cradle role flags.

### CC5. State migration policy (~50 LOC + doc) — SHIPPED (PR #211, 2026-04-29)

**Shipped via:** PR #211 (bundled with CC6 + CC7)

When users pip-upgrade 0.8 → 1.0 with existing `~/.maxim/agents/` state, what happens? Two minimum bars:
- `tests/integration/test_persistence_compat.py` loads a pre-1.0 fixture state and asserts no errors. Becomes the regression guard for future format changes.
- `docs/user/upgrading.md` documents the contract — what survives upgrades, what doesn't, what `maxim` should do if it can't load old state.

Future work (post-1.0): a `maxim migrate` verb that detects and warns. Not required for 1.0 if the test + doc exist.

### CC6. Plugin / extension API documentation (~1 page) — SHIPPED (PR #211, 2026-04-29)

**Shipped via:** PR #211 (bundled with CC5 + CC7)

`maxim.robots` entry point group is documented. Other extension surfaces are not: custom tools (Tool ABC), custom backends (Plan 3 dispatch table), custom bio-system bridges, custom percept sources. Without a "Maxim Extension API" doc, third parties write against internal modules and we can't refactor.

**Deliverable:** `docs/user/extension_api.md` listing every supported extension point with stable vs experimental tags. ~1 page.

### CC7. Tool side_effects registry centralization (~doc) — SHIPPED (PR #211, 2026-04-29)

**Shipped via:** PR #211 (bundled with CC5 + CC6)

CLAUDE.md says side_effects keys are "well-known, append-only registry, documented in class docstring." Move the registry to `docs/user/tool_side_effects.md` so third-party tool authors know what keys exist. Append-only is the right invariant; visibility makes it enforceable.

### CC8. Sim → general adapter contract audit (~1 hour audit + maybe 50 LOC) — SHIPPED (PR #210, 2026-04-29)

**Shipped via:** PR #210 (bundled with CC10 + CC11)

For Minecraft (1.1) and game-NPC integration to slot in cleanly, `PerceptSource`, `ActionSink`, and the bridge protocols in [`simulation/bridge.py`](../../src/maxim/simulation/bridge.py) need to NOT assume sim orchestrator. Run a focused audit: can a non-sim adapter (e.g., a Mineflayer adapter) implement these cleanly? If the protocols leak sim-specific assumptions (orchestrator dependency, narrative phase coupling, etc.), clarify or generalize before freeze.

### CC9. Tool schema dual-format support — MCP-compat preparation (~100 LOC) — SHIPPED (PR #204, 2026-04-28)

**Shipped via:** PR #204 (+ `tests/unit/test_tool_dual_schema.py`)

Today `Tool.input_schema` uses a custom format `{"name": type}` or `{"name": (type, default)}`. The wider ecosystem (MCP, OpenAI, Anthropic, OpenAPI) uses JSONSchema. To keep the door open for MCP server/client modes in 1.1+ without breaking user-written tools:

1. Add `Tool.to_json_schema()` method that converts the custom format to JSONSchema.
2. Allow `Tool.input_schema` to accept *either* the custom format OR a JSONSchema dict (auto-detect at construction).
3. Internally, normalize to the existing custom format (no behavior change in 1.0).
4. Document that JSONSchema is the canonical format going forward; the custom format is supported indefinitely as a convenience.

### CC10. Async wrappability check (~1 hour audit) — SHIPPED (PR #210, 2026-04-29)

**Shipped via:** PR #210 (bundled with CC8 + CC11)

Modern Python frameworks (FastAPI, Pydantic AI, LangGraph) are async-native. Maxim's `api.py` verbs are sync. As long as users can wrap them in `asyncio.to_thread()` cleanly, no work is needed. But if any verb does something that breaks under threadpool wrapping (e.g., uses asyncio internally without an event loop, depends on stdin/stdout, depends on the current working directory), retrofitting async support post-1.0 is invasive.

**Deliverable:** focused 1-hour audit of every verb in `api.py`. For each, confirm:
- No internal asyncio usage that requires a running event loop
- No stdin/stdout dependency (terminal output is fine; required input is not)
- No CWD assumption

If anything fails the check, fix it (small surface adjustments, likely <30 LOC each). Document the threadpool wrap pattern in the stability page from CC2.

### CC11. Tool cancellation contract (~30 LOC + doc) — SHIPPED (PR #210, 2026-04-29)

**Shipped via:** PR #210 (bundled with CC8 + CC10)

Tools have a `timeout` field but no cancellation hook. A user pressing Ctrl-C, an exceeded budget, or an upstream cancellation has no graceful path — long-running tools (web scraping, large file ops, MCP subprocess calls in 1.1+) become stuck threads.

**Deliverable:** add `Tool.cancel()` hook to the ABC with a default no-op implementation. Document semantics: when called, the tool should attempt to abort its current execution and release resources. Backwards-compatible for existing tools (default no-op = today's behavior); explicit for new ones. Wire one or two heavy built-in tools (web fetch, MCP subprocess) to actually implement cancel.

### CC12. Token telemetry standardization (~doc + small surface adjustment) — SHIPPED (PR #212, 2026-04-29)

**Shipped via:** PR #212 (bundled with CC2 + CC4)

Modern frameworks expose `input_tokens`, `output_tokens`, `cached_tokens` in standardized dicts. Maxim has cost tracking per-session via `LLMEnergyTracker`. Verify the per-call telemetry exposes the standard fields in `ToolOutput`, `BackendError`, or wherever — so users building dashboards / fine-tuning datasets / cost analyses don't have to reverse-engineer the format.

**Deliverable:** audit the call-site telemetry in `models/language/router.py`, `_MaximPeerBackend`, `_OpenAIBackend`, `LLMEnergyTracker`. Confirm `input_tokens` / `output_tokens` / `cached_tokens` are exposed under those exact field names in any user-visible structure. Where missing, add them. Document the contract in CC4's classification doc.

---

## Section 8: 1.1 track (concurrent development, not 1.0 gating)

Items deliberately scoped OUT of 1.0 to keep the stabilization release tight. Shipped concurrently or in 1.1 splash.

### 1.1-T1. Minecraft live demo + harness benchmark

**Plan stub:** [minecraft_benchmark.md](minecraft_benchmark.md) (design exploration, post-1.0)

Compare Maxim against Voyager / GITM / SPRING on a Minecraft live demo. Builds on Cradle's embodied learning foundation. Why 1.1 not 1.0:

1. Comparison protocol design (same seeds, same turn budgets, same embodiment) is research-grade work that benefits from time, not haste.
2. 1.0 is the interface freeze; 1.1 is the showpiece. Two news cycles, less risk per release.
3. Cradle (B4) already provides the cross-session learning evidence 1.0 needs. Minecraft strengthens the story without gating it.

### 1.1-T2. Scene actor affordances

**Plan:** [scene_actor_affordances.md](scene_actor_affordances.md)

`target_effect` field + `OrchestratorActorTool` so scene entities produce real SEM mechanics on the AUT body. ~110 LOC. Diagnostic for whether agent-backed entities are needed. Slips from 1.0 because it's a quality-of-life refinement, not a freeze gate.

### 1.1-T3. SEM world enrichment Phases 2-3

**Plan:** [sem_world_enrichment.md](sem_world_enrichment.md)

Bio-enrichment routing through ComponentIndex, composable body archetypes. Phase 1 shipped. Phases 2-3 enrich the learning environment for the Minecraft demo but don't gate the cross-session claim — the Cradle and Experiment 10 already validate the claim with simpler worlds.

### 1.1-T4. C4-C6 hard errors

After 0.9 deprecation cycle. Flip `warnings.warn(DeprecationWarning, ...)` → `raise ConfigurationError(...)` (or `TypeError` for the C6 raw-construction path) at each shipped warning site.

**C4 prerequisites (must clear before flipping the C4 warning):**
- C4-followup-1 (imagination + foundry pipeline normalizer + prompt update) — see §C4. Hard prerequisite: without it, the flip hard-fails every imagined entity.
- C4-followup-2 (sensor-promotion audit on ambiguous bundled modulators) — see §C4. Soft prerequisite: the bundled YAMLs already pass the C4 invariant, but the audit is the right time to promote ~5-15 from `abstract: true` to "owns these sensors."

**C6 prerequisite (must clear before flipping the C6 warning):**
- `default_network/network.py::_init_pain_circuit` still constructs `PainBus(_allow_raw=True)` when no bus is injected. The Wave-2 follow-on (split-subscriber-ownership fix in `pain_bus_unification.md`) couples DefaultNetwork to MemoryHub so the bus comes from `build_pain_bus`. Until that lands, C6's hard-error flip would break DN's standalone path. Either (a) ship the Wave-2 follow-on first, or (b) leave the explicit `_allow_raw=True` opt-out in DN and flip C6 elsewhere — if you take (b), keep the comment at the call site current.

### 1.1-T5. Agent-backed entities (revival path)

**Plan (deferred):** [deferred/agent_backed_entities.md](deferred/agent_backed_entities.md)

Revives if scene_actor_affordances doesn't close the dragon-narration symptom OR if Minecraft demo exposes a cognition gap (zombies need pathfinding, villagers need trade memory).

### 1.1-T6. B5 embodiment/narrative separation

**Plan (deferred):** [deferred/b5_embodiment_narrative_separation.md](deferred/b5_embodiment_narrative_separation.md)

B3 Acting Coach shipped. B5 is a shell.

---
