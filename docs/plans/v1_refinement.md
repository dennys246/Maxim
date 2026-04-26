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

---

## Section 2: Bio-System Stabilization (freeze-worthy interfaces)

These are the systems that must be fully operational and standardized before 1.0 freezes the interfaces. Post-1.0, interface changes are expensive.

### B1. Bio-system protocol enrichment

**Companion plan:** [bio_system_protocol_enrichment.md](bio_system_protocol_enrichment.md)

Add standardized `*Context` dataclass parameters to each bio-system's primary methods NOW (optional, defaults to None) so future backends can consume richer input without Protocol breaks. Current implementations ignore the context; future ones use it.

**Systems to enrich:**
- Hippocampus: `RetrievalContext` (temporal, emotional, goal state, consolidation level)
- NAc: `PredictionContext` (arousal, temporal discount, context richness)
- ATL: `SemanticContext` (abstraction level, domain hints)
- EC: `EncodingContext` (decomposition strategy hints, resolution)
- SCN: `TemporalContext` (oscillator phase, prediction confidence)

**Why before 1.0:** Adding an optional `context=None` parameter is non-breaking. Removing one or changing its shape post-1.0 IS breaking. The cost of adding these now is near-zero; the cost of not having them later is a 2.0.

### B2. SCN oscillator feedback — anticipatory temporal credit — SHIPPED (2026-04-26)

**Companion plan:** [scn_oscillator_feedback.md](scn_oscillator_feedback.md) (~120 LOC)
**Branch:** `feat/v1-scn-oscillator`

Closed the SCN→NAc feedback loop. Three-path credit in TemporalCreditDistributor: fast-decay → phase-similarity → anticipatory (oscillator-predicted imminent events). 21 new tests.

**Changes shipped:**
1. `OscillatorNetwork.observe_event()` — per-event-type circadian phase tracking
2. `OscillatorNetwork.predict_event_imminence()` — circular mean + concentration scoring
3. `TemporalCreditDistributor.distribute()` — third anticipatory credit path
4. `build_bio_stack` — `scn.enable_oscillator()` by default

**Why before 1.0:** SCN was the only bio-system with a one-way data flow. Now every bio-system has a closed feedback loop.

### B3. SEM world enrichment (Phases 2-3)

**Companion plan:** [sem_world_enrichment.md](sem_world_enrichment.md)

Phase 1 shipped (scene manifest pre-trigger, head-noun alias fallback). Remaining:

- **Phase 2: Bio-enrichment routing** — BioEnrichmentPipeline queries ComponentIndex for SEM-aware context. Entity affordances appear in enrichment sections ("your experience with fire_breath suggests...").
- **Phase 3: Composable body archetypes** — Mix-and-match body templates (biped + wings, quadruped + tail weapon) so imagination can compose novel entities from known parts rather than designing from scratch.

**Why before 1.0:** Sim agents currently start in sparse worlds (1-2 entities). Real learning requires a rich environment — doors, torches, traps, multiple creatures. Without Phases 2-3, the cradle (B4) has nothing to learn from.

---

## Section 3: Sensorimotor Grounding (the agent understands its world)

### B4. Cradle of Artificial Civilization

**Companion plan:** [cradle_sensorimotor_development.md](cradle_sensorimotor_development.md) (~500-800 LOC)

A newborn agent that learns from sensation, not language. Strip away LLM world knowledge and force learning through direct sensorimotor experience. The agent doesn't "know" fire is dangerous because GPT said so — it knows because touching fire triggered pain in its thermal sensors, and NAc formed a causal link between the thermal spike and negative valence.

**Key deliverables:**
1. **Somatosensory registry** — distributed sensor patches (thermal, pressure, sharp, texture) across body regions
2. **Interoceptive drives** — hunger, fatigue, curiosity as internal signals that modulate behavior
3. **Cradle environment** — controlled scenario where sensorimotor contingencies are discoverable (hot stove, sharp objects, soft bedding)
4. **Validation experiment** — fresh agent + 3-5 sessions → demonstrate learned avoidance of harmful stimuli AND approach toward beneficial ones, purely through bio-pipeline learning

**Why before 1.0:** This is the strongest possible demonstration of the 1.0 claim. If the agent can learn "fire hurts" through its own sensors without any linguistic scaffolding, the cross-session learning claim is proven at the deepest level. It's also a compelling demo for the bio-inspired framing.

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

### P2. Episode boundary — semantic shift detection (Stage 2)

**Absorbed from:** `substrate_episode_boundary_enrichment.md` Stage 2

Add semantic-shift episode boundary detection. When incoming text embedding diverges from the episode centroid, close the episode.

**Changes:**
1. Add `embedding: ndarray | None = None` to `CaptureEvent`
2. Add `centroid_embedding: ndarray | None = None` to `PendingEpisodeState` with incremental centroid update
3. Implement `semantic_shift_rule(threshold=0.40)` in `episode.py`
4. Calibration sweep against real conversational data

---

## Section 5: Cleanup (breaking changes)

Several backward-compatibility shims silently accept under-specified inputs. Per CLAUDE.md "push silent-no-op invariants into types, not helpers," these are silent-failure risks.

### C1. Probe compat shim removal (internal)

Remove `probe_llm_server`, `llm_server_responding_at`, `_probe_once`. Migrate remaining 4 callers to `_MaximPeerBackend.for_url(...).health_check()`. Hard-remove.

### C2. `SendMessageTool._detect_attack` dead code removal (internal)

Delete `_detect_attack()`, `_ATTACK_KEYWORDS`, and related comments. Dead code after reflex system shipped. Hard-remove.

### C3. `DamageEntityTool` shim removal (internal)

Remove `DamageEntityTool` entirely. Update orchestrator prompts to use `damage_component`. Hard-remove.

### C4. Modulators without sensors (deprecation phase)

Require every modulator to declare at least one sensor. Capability-only modulators declare `abstract: true`. 0.9 deprecation warning, 1.0 hard error.

### C5. Entity health as direct sensor (deprecation phase)

If entity has modulators with sensors AND a direct `health` sensor (not `derived`), parse-time warning in 0.9, hard error in 1.0.

### C6. Raw constructor enforcement (deprecation phase)

Add `_allow_raw=False` to `PainBus()` / `ReactionBus()` / `MemoryHub()`. Production uses builders, tests pass `_allow_raw=True`. 0.9 warning, 1.0 hard error.

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

1. **V1** (validation) — proves the claim. If it fails, everything else is moot.
2. **B1** (protocol enrichment) — cheap now, expensive later. Do early while interfaces are still warm.
3. **P1 + P2** (pipeline gaps) — small, can run in parallel with B1.
4. **B2** (SCN oscillator) — depends on P1 (ToolPainBridge temporal migration provides diverse TemporalEvents for the oscillator to learn from).
5. **B3** (SEM world enrichment Phases 2-3) — enriches the learning environment.
6. **B4** (cradle) — depends on B2 (SCN feedback) and B3 (rich world). The capstone demo.
7. **C1-C3** (internal cleanup) — ship anytime.
8. **C4-C6** (deprecation phase) — 0.9 warnings, 1.0 hard errors.
9. **D1-D3** (docs) — last, after content stabilizes.

## Timing

- B1 and P1+P2 are the quickest wins — ship first.
- B2→B3→B4 is the critical chain for the sensorimotor grounding story.
- C1-C3 are internal hard-removes, zero user impact, ship anytime.
- C4-C6 need a 0.9 deprecation release before 1.0 hard errors.
