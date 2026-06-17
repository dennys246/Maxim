# v1 Refinement — Validation + Stabilization + Cleanup for 1.0 Release

**Status:** PLANNING (pre-v1.0 release)
**Target version:** 1.0
**Branch:** TBD

---

## Front-gate scope pressure (retroactive)

Added 2026-05-27 per CLAUDE.md Principle 3.

**Note:** this is a **1.0 release coordination doc** spanning multiple sub-plans, not a single-mechanism plan. Per the kickoff rule "do the front-gate analysis per-mechanism, not per-plan" for multi-mechanism plans, the analyses for individual mechanisms referenced here live in their respective sub-plans:

- B5 substrate-primary AUT + Hivemind shareability → [grounded_language_acquisition.md](grounded_language_acquisition.md) + [maxim_hivemind.md](maxim_hivemind.md)
- B3 SEM world enrichment → [sem_world_enrichment.md](sem_world_enrichment.md)
- bio_emergent_persona_foundations wires → [bio_emergent_persona_foundations.md](bio_emergent_persona_foundations.md)
- scene_actor_affordances → [scene_actor_affordances.md](scene_actor_affordances.md)
- persona_cleanup → [persona_cleanup_and_mode_transition.md](persona_cleanup_and_mode_transition.md)
- ec_centroid_drift_fix (V1 substrate prerequisite) → [ec_centroid_drift_fix.md](ec_centroid_drift_fix.md)

**For the 1.0 work owned directly by this plan (Section 3 cleanup C4-C6, Section 6 docs D1-D3, Section 7 contract clarification):**

| Item | Front-gate verdict |
|---|---|
| C4-C6 deprecation cycle (raw constructor enforcement, hard-error flip) | Pure removal / hardening of existing surface. No new mechanism — flips already-shipped deprecation warnings to hard errors |
| D1-D3 docs passes | Documentation only. Not a mechanism |
| Section 7 contract clarification | Documentation + clarification of existing contracts. Not a mechanism |
| V1 cross-session validation | Experiment, not a mechanism |

**Verdict aggregate:** v1_refinement.md as a release plan doesn't introduce new mechanisms of its own. All mechanism-bearing sub-plans now carry their own front-gate analyses per Principle 3.

**Specific reason this plan exists without new mechanism:** the 1.0 release is **stabilization + validation + cleanup**, not a feature release. The substrate work is shipped; what remains is hardening (C4-C6), documenting (D1-D3), proving the claim (V1), and shipping the experimental substrate-primary harness (B5, scoped in sub-plans). New mechanisms enter through sub-plans with their own scope-pressure discipline.

---

## Motivation

1.0 claims "cross-session learning without fine-tuning." All substrate gates are closed (P1-P8, B4, behavioral convergence 41/41). What remains is:

1. **Proving the claim end-to-end** in a user-facing scenario
2. **Stabilizing the bio-systems** — every core bio-system (SCN, SEM, PainBus, NAc) must have standardized protocols and fully operational feedback loops before the interfaces freeze at 1.0
3. **Grounding basic knowledge** — the agent should understand fundamental sensorimotor truths (fire hurts, falling damages) through its own bio-pipeline, not LLM world knowledge
4. **Enriching the world layer** — SEM entities need rich environments to learn from, not bare-bones single-entity scenes
5. **Closing pipeline gaps and removing silent backward compat**

---

## Outstanding for 1.0 — ALL HARD REQUIREMENTS CLOSED (2026-06-15)

Substrate work is fully shipped (V1+V2 + B1+B2+B4 + P1-P4 + CC1-CC13 + C1-C6). Both 1.0 behavioral gates are SETTLED (Tier 1 behavioral-graduation closed; benchmarking executed + dispositioned via the Exp 37/38/39 line). The 1.0 release announcement landed on main (#373). **The version is bumped to 1.0.0.** Nothing remains gating the 1.0 ship.

**Hard requirements — all DONE:**
- **C4-C6** (Section 5) — Cleanup deprecation cycle COMPLETE. All three hard-error flips landed in Phase 1 of the sequenced 1.0 plan (2026-05-29): C4 (PR-A.3) + C5 (PR #299) + C6 (PR #301, path (b) with DefaultNetwork opt-out retained). C6 Wave-2 split-subscriber-ownership fix in `pain_bus_unification.md` is post-1.0 polish; C4-followup-2 sensor-promotion audit is post-1.0 polish.
- **CC13** (Section 7) — Auth format-freeze SHIPPED (2026-06-15, branch `feat/1-0-cc13-auth-format-freeze`). Four security-shaped surfaces frozen for 1.1+ hardware-token / signed-bundle / mTLS work.
- **D1-D3** (Section 6) — Docs passes SHIPPED (2026-06-15). Honesty reconciliation against the locked release thesis + version bump. See Section 6.
- **W1-W2 (Wire-A substrate→action conversion)** — see Section 1.5 below. Added 2026-05-27 after Phase B Roy-3a-retry's verdict named the substrate-scene-tool-availability + imagination-substrate-blindness gaps as the 1.0 thesis-demonstration bottleneck.

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
  - **Hivemind shareability infrastructure** (~660 LOC) — **SHIPPED** (2026-05-30 / 2026-05-31, four reviewed PRs):
    - PR A (#305, MERGED) — provenance + substrate-domain + fan-in-contributors fields on CausalLink + EC nodes; new `EntorhinalCortex.substrate_node_metadata` accessor.
    - PR B (#308, MERGED) — `nac_merge` / `ec_merge` Bayesian-aggregation pure-function utilities at `src/maxim/hivemind/merge.py`. Zero-prior for unobserved entries; valence-distinct CausalLinks stay separate; count-weighted EC centroid mean (respecting `frozen_centroid_modalities` per the bio-fidelity-lens fold); Chan parallel-Welford for variance state.
    - PR C (#309, MERGED) — identity-bearing concept detection at `src/maxim/hivemind/identity.py` (proper-noun + identity-keyword heuristic with conservative-tilt threshold).
    - PR D (#310, OPEN) — substrate snapshot bundle (ZIP + manifest + reserved signature slot) at `src/maxim/hivemind/bundle.py` + `maxim substrate export | import | inspect` CLI verbs at `src/maxim/hivemind/cli.py` + bundle migration-registry seam for 1.1+. Three-lens review (Executor + Architecture + Bio-fidelity) closed 1 CRITICAL ZIP-slip + 5 IMPORTANT findings before push.
    - Reserved hooks for 1.2 P2P: `trusted_sources` / `validate_link` / `validate_node` callback slots on merge functions; manifest `signature` + `signature_algorithm` slots. Reserved namespaces: `_consensus` (`CONSENSUS_SOURCE`) + `_identity` (`IDENTITY_DOMAIN_MARKER`); the `_*` prefix is rejected at every public entry point via shared `_validate_source`.
    - Real-session smoke test passed end-to-end (`maxim substrate export ... && maxim substrate import ...` on a captured sim_report). 1.1 Oasis ingestion + 1.2 P2P protocol build on this surface without retrofitting.
  - **Scope shipped to date:** ~1,360 LOC of ~1,360. Hivemind shareability complete.
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

## Section 1.5: Wire-A substrate→action conversion (added 2026-05-27)

The Phase B Roy-3a-retry verdict ([30_wire_a_tau_validation.md](../experiments/30_wire_a_tau_validation.md)) confirmed Wire-A's annotation reaches the LLM at strong magnitude (`[strongly rewarding]` throughout the test arm) but **cannot convert to action** because (a) the substrate-favored tool wasn't in the scene's active roster and (b) imagination didn't dream the missing entity into existence. Until both gaps close, Roy iterations stay structurally capped at "annotation is present" findings — the 1.0 thesis-demonstration is blocked.

Both gaps get MVP-scoped 1.0 plans. Full versions of each plan are deferred to 1.1+; only the MVP scope is 1.0 critical path.

### W1. Sense tool registry — grayscale visibility MVP (~150-200 LOC)

**Plan:** [sense_tool_registry.md](sense_tool_registry.md) (1.0 MVP scope section).

**MVP scope:**
- `tools_block` rendering distinguishes always-active core tools from SEM-derived inactive tools, with `[not in current location]` tag for the latter. Inactive SEM tools the substrate has accumulated bias for (per `NAc.get_agent_tool_biases`) appear in the LLM-facing list.
- `auto_fire: bool` metadata on Tool dataclass (declarative replacement for the implicit executor bypass).
- Registration-time classifier with `kind=` discriminator.

**Deferred to 1.1+:** `sensory_events.jsonl` separation, LRU eviction tuning, predicate-outcome NAc typing, description unifier.

### W2. Imagination substrate-signal — Hookup 1 MVP (~20-30 LOC)

**Plan:** [imagination_substrate_signals.md](imagination_substrate_signals.md) (1.0 MVP scope section).

**MVP scope:**
- Substrate-aware manifest — pass `NAc.get_agent_tool_biases()` results to `Narrator.generate_scene_manifest()` so the LLM-generated manifest sees Wire-A's biases at scene-load time. The manifest can then include entities that activate substrate-favored tools.

**Deferred to 1.1+:** Hookup 2 (per-tick subscriber), Hookup 3 (arousal-gate relaxation).

### Integration test: next Wire-A Roy iteration

After W1 + W2 MVPs ship, run Roy-3a-retry's spec unchanged with both gaps closed.

**Convergence outcome** (Arm A ≥1 `sense_food_source` call): Wire-A annotation→action pathway validated; 1.0 has a positive thesis signal to point at. Other 1.0 gates (D1-D3, etc.) proceed; W1+W2 close.

**Divergence-in-a-row outcome** (Roy finds a *new* failure mode beyond the two now-closed gaps): per refined Principle 4, this is the two-divergence-in-a-row trigger to bird's-eye to encoder replacement (Roy-5a Stage 3 + JEPA, 1.2+). W1+W2 stay shipped but their behavioral weight is gated on encoder work; the 1.0 thesis-demonstration claim re-scopes.

**Outcome (2026-05-27, [32_wire_a_post_w1_w2.md](../experiments/32_wire_a_post_w1_w2.md)):** AMBIGUOUS-WITH-WIRING-BUG. PRIMARY failed (0/0/0), but structural analysis identified two upstream wiring gaps neither of which lives inside the W1 or W2 MVPs:

- **Bug A** — Roy cross-session agent_id mismatch. Priming MemoryHub defaults to `agent_id="default_agent"`; the AUT orchestrator constructs the test-arm MemoryHub with `agent_id="sim_aut"` ([orchestrator.py:534](../../src/maxim/simulation/orchestrator.py)). `cluster_reward_bias` persists with the priming-time key; the test arm reads via `_loop_agent_id="sim_aut"` and `get_agent_tool_biases` strict-filters to empty. Wire-A and W1's grayscale (which reuses Wire-A's biases) both silently render empty. Affects every 0.9.1 Roy iteration that has claimed Wire-A reached the LLM — Experiment 30's reconstruction needs re-validation.
- **Bug B** — W2's hookup site (`generate_scene_manifest` at [orchestrator.py:1468](../../src/maxim/simulation/orchestrator.py)) is structurally bypassed by Roy's fixture-driven test arms (`roy_1_holdout.yaml`). The W2 MVP plan correctly cites cradle as its precedent; that precedent doesn't extend to the Roy fixture path.

Both fixes are 1.0 critical path before the next Wire-A Roy iteration can run with a working measurement instrument. The bird's-eye to encoder replacement is **not authorized** by this iteration — the thesis was not measured (the instrument was broken). Refined Principle 4 fires on new failure modes of the thesis, not on instrument bugs. Fix A's two candidate shapes and Fix B's two options are scoped in the experiment doc.

**Bug A fix scope (2026-05-27, post-audit reframe):** the audit during Fix A scoping surfaced that Bug A is NOT Roy-specific — `AgentFactory.create_full_agent` calls `build_bio_stack` without threading `config.agent_id`, so EVERY production agent (CLI non-sim, sim AUT, sim NPC, Reachy, headless `pymaxim`) silently gets `memory_hub.agent_id="default_agent"` regardless of its AgentConfig. The root-cause fix lives in `AgentFactory.create_full_agent` + `agentic_runtime.py`'s Reachy path, NOT in Roy priming. Per architecture-lens BLOCK-1 from the pre-merge review, the fix also pushes the invariant into the type: `build_bio_stack`, `build_memory_hub`, and `_create_memory_hub` make `agent_id=` REQUIRED keyword-only (no default), matching the canonical `build_executor(pain_bus=...)` and `build_pain_bus(hippocampus=..., nac=...)` patterns. The next miss is a `TypeError` at the door, not a silent agent_id divergence.

**Backward-compat decision for Bug A fix (architecture-lens BLOCK-3, accepted 2026-05-27):** existing persisted NAc snapshots under `~/.maxim/sim_reports/` and `~/.maxim/agents/` are keyed under `agent_id="default_agent"`. Post-fix, any agent loading those snapshots reads with its real agent_id and gets empty results — the old `default_agent`-keyed entries become orphan data. **The choice is "ship clean, no migration."** Rationale: sim sessions write to per-session tmpdirs that don't persist across runs; the V1 cross-session-learning thesis is structurally about the *invariant* being correct, not about preserving prior-session-data through the fix that makes the invariant correct. The migration-shim alternative (rewrite `default_agent` keys on load) was considered and rejected — it adds load-path complexity to preserve data the V1 thesis can regenerate cleanly. Pre-1.0 users with valuable persisted state can manually rename their NAc dump's `default_agent` keys to their canonical agent_id via `jq`.

**Post-Fix-A integration test verdict (2026-05-27, [33_wire_a_post_fix_a.md](../experiments/33_wire_a_post_fix_a.md)):** Fix A shipped (PR #290 merged at `cdd005a`). Re-ran Roy-3a with the same spec. **Fix A structurally validated** — NAc `cluster_reward_bias` keys now use `sim_aut`, Wire-A's lookup returns `[('tool:sense_food_source', +0.768)]` at Arm A end (strongly-rewarding band), the annotation `sense_food_source [strongly rewarding from prior experience]` is rendered to the LLM every submission. The first clean substrate→LLM measurement in the 0.9.1 release window. **PRIMARY criterion still failed** (Arm A=0/B=0/C=0). The load-bearing follow-up cause is the scene-tool-availability gap: `sense_food_source` is NOT in Arm A's active roster (the roy_1_holdout fixture has no food entity), so W1 surfaces the tool in grayscale (`[not in current location]`) but the LLM cannot invoke it and does not reach for substitutes. **Per the pre-registered framing in this section, the divergence-in-a-row trigger condition is met on a strict reading**, formally pointing at encoder replacement (Roy-5a Stage 3 + JEPA, 1.2+). But the cheaper, more-bio-fidelity-conservative next test is **Fix B (extend W2 to fixture scene-load)** — bring a food entity into Arm A's scene and re-measure. Per the kickoff's "Triggering bird's-eye encoder work if divergence fires: NEVER without explicit authorization" rule, the encoder pivot stays unauthorized; the Fix B vs encoder-pivot decision is the next user-facing scoping question.

**Post-Fix-A+B integration test verdict (2026-05-28, [34_wire_a_post_fix_a_b.md](../experiments/34_wire_a_post_fix_a_b.md)):** Fix B shipped (PR #292 merged at `b1e20ee`). Re-ran Roy-3a. **Fix B structurally validated end-to-end** — fires once per test arm, emits SEM_TRACE events at every branch, Arm A's pre-trigger called the manifest LLM with `biases=1` (sense_food_source +0.997) and materialized 1 entity into scene; Arms B+C correctly skipped on empty biases. Double-fire fix also validated (W2's hookup fired only during the priming session, not during test arms). **PRIMARY criterion still failed** (Arm A=0/B=0/C=0). **Narrowed failure mode:** the manifest LLM generated `['blue toy car', 'set of keys']` instead of a food entity when given `sense_food_source [strongly rewarding from prior experience]`. The substrate→manifest-LLM bridge has a **semantic gap at the manifest LLM's interpretation of tool-name biases as scene-entity preferences**. The AUT LLM DID engage with the materialized entity (picked up the set of keys on a relevant percept) — so the runtime pipeline works; the bottleneck is at the manifest LLM's prompt-side substrate-bias-to-entity translation. **Per refined Principle 4 + pre-registered framing, the divergence-in-a-row trigger fires** toward encoder replacement (Roy-5a Stage 3 + JEPA, 1.2+). But this iteration's narrow-failure-mode finding opens a cheaper diagnostic before encoder commit: **Option X1 — improve `_compose_substrate_context` in narrator.py to explicitly instruct the manifest LLM to bridge tool biases to scene entities** (one-line prompt addition, single Roy iteration cost). If Option X1 fails with the explicit bridging instruction, encoder-alignment work is fundamentally needed and JEPA pivot is authorized cleanly. If Option X1 succeeds, encoder pivot defers to 1.1+. Per the kickoff's authorization rule, encoder pivot stays unauthorized; Option X1 vs encoder-pivot decision is the next user scoping question.

### What this section does NOT include

- **scn_decay_anchoring is NOT a 1.0 item** despite originating from the same tau-split kickoff sequence. It's 1.0 nice-to-have / 1.1 acceptable per its own status header — addresses hardware portability, not substrate→action conversion. See [scn_decay_anchoring.md](scn_decay_anchoring.md) for the resolution.
- Full versions of W1 + W2 (deferred 1.1+ items in each plan's MVP scope section).

---

## Section 1.6: Post-Roy-5b roadmap (RESOLVED 2026-05-29) + sequenced 1.0 plan

Added 2026-05-28; updated 2026-05-29 after Roy-5b verdict + Roy-5b-confound-isolation Branch A resolution. **Both Branch A and Branch B are now closed** — the encoder-pivot question is resolved (not via either branch, but via a third path the disambiguator plan didn't pre-register). See [docs/experiments/36_roy_5b_confound_isolation.md](../experiments/36_roy_5b_confound_isolation.md) for the verdict + [project_roy_5b_confound_isolation.md](../../../.claude/projects/-Users-dennyschaedig-Scripts-Maxim/memory/project_roy_5b_confound_isolation.md) memory.

### What happened (the resolved encoder-pivot question)

The Roy-2c recognition gap closes from the **EC drift fix alone** (PR #264, `pattern_complete_threshold` 0.40 → 0.44, 2026-05-24). No naming-event scaffold needed; no Hebbian binding mechanism needed; no JEPA projection needed. Four-experiment matrix:

| Experiment | Body | Codebase | Threshold | Arm A overlap with priming |
|---|---|---|---|---|
| Roy-4 (2026-05-13) | infant_humanoid | pre-fix | 0.40 | 0/10 |
| Roy-4-replica (2026-05-29) | infant_humanoid | pre-fix | 0.40 | 0/10 |
| Roy-5b-confound-isolation (2026-05-29) | infant_humanoid | HEAD | 0.44 | 10/10 |
| Roy-5b (2026-05-28) | naming_v1 (scaffold) | HEAD | 0.44 | 10/10 |

The threshold tweak is solely responsible. **Outcomes:**
- [cross_modal_substrate_binding.md](cross_modal_substrate_binding.md) ARCHIVED. Do not resurrect.
- [jepa_cross_modal_alignment.md](jepa_cross_modal_alignment.md) stays at its pre-Roy-5b "Stage 4b candidate" status — 1.1+/1.2 research direction motivated by the dimensional fact (384 vs 768-dim cross-modal cosine undefined). Roy-5b's specific Branch B promotion trigger (clean FAIL across the sweep) did not fire, so JEPA does NOT promote to "1.2 in flight" today. But the plan's underlying motivation is independent of any specific Roy iteration verdict — it stands on the structural fact about different-dimensional encoders. Neither Roy-5b nor Roy-5b-confound-isolation cancel it; neither promotes it.
- [embodiment/naming_events.py](../../src/maxim/embodiment/naming_events.py) marked Dormant per CLAUDE.md Principle 2.
- [roy_5_encoder_alignment_disambiguator.md](roy_5_encoder_alignment_disambiguator.md) Stage 4 disposition resolved; new Stage 5 question: what does the threshold-driven gap closure buy behaviorally?

### Sequenced 1.0 plan (authorized 2026-05-29)

User-authorized order. Two tracks run in parallel through the middle; benchmarking scope gets defined alongside the warm-up so the graduation criteria can incorporate it.

**Phase 1 — Warm-up (1-2 small PRs + 1 planning doc):**
1. **C5 + C4/C6 hard-error flip — SHIPPED** (PRs #299 + PR-A.3 + #301, 2026-05-29). Small mechanical PRs; first momentum win of the sequenced 1.0 plan.
2. **Benchmarking scope-definition planning doc.** Per [behavioral_graduation_candidates.md](behavioral_graduation_candidates.md) "Benchmarking is the **sibling** 1.0 gate to behavioral graduation." Currently undefined in plan docs. Scoping during warm-up means the criteria can inform how Tier 1 graduations get measured. Don't tail-load this — risk of late surprises that re-open graduation work.

**Phase 2 — Parallel tracks (the calendar middle):**

*Foreground track — Tier 1 behavioral graduations, in confidence order:*
1. **Cross-session learning** (Tier 1, PARTIAL → EARNED). Closest to EARNED — 3 memories/turn already measured per [Exp 10](../experiments/10_cross_session_enrichment.md); needs predictions + concepts pending. **Tackle first** because lowest risk of taking calendar.
2. **Affordance concept transfer** (Tier 1, PARTIAL → EARNED). Has 0.785 cosine measurement; needs broader Roy-5+ fixture validation. **Tackle second.**
3. **Substrate-primary action selection** (Tier 1, PARTIAL → EARNED). Hardest — Roy-5b's findings inform but don't graduate this. The disambiguator's new Stage 5 question is the path forward but currently undefined. **Tackle last** because earlier graduations may inform what "threshold-driven closure buys behaviorally" looks like in practice.

"Hammer each one until we **graduate or reframe**." Each Tier 1 item ends in `EARNED` status OR explicit retraction in 1.0 release notes (no silent omission). Per [behavioral_graduation_candidates.md](behavioral_graduation_candidates.md) "1.0 commitment" line.

*Background track — Hivemind shareability infrastructure (~660 LOC):*
- Substrate snapshot bundle format (zip + manifest + signature, ~150 LOC)
- `nac.merge()` / `ec.merge()` Bayesian-aggregation library functions (~200 LOC)
- Provenance tags on NAc links + EC nodes (~100 LOC)
- Identity-bearing concept detection ported from old Mother plan (~80 LOC)
- Substrate domain tagging (~50 LOC)
- `maxim substrate export` / `maxim substrate import` CLI verbs (~80 LOC)

Runs PARALLEL to graduations because each Roy iteration is ~25 min wall + analysis — leader would sit idle during graduation analysis windows otherwise. Two-track approach saves ~2-3 weeks. Doesn't touch user-facing 1.0 surface; doesn't gate D1-D3.

**Phase 3 — Benchmarking execution:**

Once graduation criteria are firm + Tier 1 graduations are in flight or done, execute against the benchmarking scope from Phase 1's planning doc. Sibling 1.0 gate per [behavioral_graduation_candidates.md](behavioral_graduation_candidates.md).

**Phase 4 — D1-D3 docs:**

Last, so docs reflect the settled state. No code; just writing. Agent memory transfer + API/CLI surface review + final docs pass per §6.

### Why this sequence

- **C5/C4/C6 first:** small mechanical wins that build momentum, no calendar risk, surface a "done" early.
- **Benchmarking scope EARLY (parallel to C5/C4/C6):** scoping is a planning artifact at warm-up stage; defining it informs how graduations get measured. Tail-loading is risky.
- **Tier 1 graduations as the long-pole middle:** the binary "graduate or reframe" framing is the honest 1.0 discipline. 3 of 5 Tier 1 items are PARTIAL today. The Roy harness is mature enough that each graduation is ~1-3 weeks; cross-session-first ordering puts the highest-confidence graduations first.
- **Hivemind PARALLEL to graduations:** independent code work that fills the "wait for Roy iteration" calendar windows. Strict serial would leave the leader idle.
- **Benchmarking execution after scope + graduations:** scope is firm by then; can be executed against settled graduation criteria.
- **Docs LAST:** D1-D3 is "no code, just writing"; writing once everything is settled prevents rework.

### Walls (rough)

| Phase | Wall | Notes |
|---|---|---|
| Phase 1 (warm-up) | ~1 week | C5/C4/C6 PRs + benchmarking scope doc |
| Phase 2 (parallel tracks) | ~4-6 weeks | graduations: cross-session ~1-2wk + affordance ~1-2wk + substrate-primary ~2-3wk. Hivemind in background. |
| Phase 3 (benchmarking exec) | ~1-2 weeks | depends on scope from Phase 1 |
| Phase 4 (docs) | ~1-2 weeks | D1-D3 |
| **Total 1.0 wall** | **~7-11 weeks** | with realistic risk buffer |

### Cadence prediction

~8-12 PRs total across 1.0. Each PR matches the recent pattern (~3-5 hour sessions, two-lens review where applicable, single durable artifact). Hivemind track may stack multiple smaller LOC PRs.

### Open questions worth pre-flagging now

- **What does "graduated" actually look like per Tier 1 item?** The graduation candidates doc lists the bio-mechanism + status, but each PARTIAL item needs a pre-registered experiment + diagnostic that produces a clean PASS/FAIL (Roy-style discipline). Each Tier 1 kickoff should pre-register its graduation criterion before the experiment runs.
- **Benchmarking scope definition.** What tasks, what metrics, what acceptance criteria? Phase 1's scoping doc owns this. Inputs: 1.0 thesis ("cross-session learning without fine-tuning"), Tier 1 graduation criteria, comparable benchmark precedents in the bio-inspired-LLM-harness space.
- **The "reframe" option per Tier 1 item.** If a PARTIAL item can't graduate, which framings get retracted vs which force a 1.0 delay? Cross-session learning is load-bearing for the entire thesis; substrate-primary action selection is currently experimental flag and could ship as such. Worth thinking through per-item before "hammering."
- **0.9.x parallel cadence.** Wire-A, Fix A, Fix B all live and useful. A 0.9.2 / 0.9.3 cadence with operator-visible surface improvements is independent of the 1.0 sequence. Cheap; worth doing if user value warrants.
- **Disambiguator Stage 5 question.** "What does the threshold-driven gap closure buy behaviorally?" is the disambiguator plan's new next-question. May surface a problem that informs (or blocks) substrate-primary graduation. Light scoping pass before substrate-primary graduation kicks off.

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

### C4. Modulators without sensors — SHIPPED

Require every modulator to declare at least one sensor. Capability-only modulators declare `abstract: true`.

Deprecation phase shipped via PR #219 (2026-05-03): parse-time `DeprecationWarning` at `SpecModulator.__init__` for any bare modulator without `abstract: true`. Lives at the constructor (not `_parse_entity`) per CLAUDE.md "push silent-no-op invariants into types, not helpers" — covers parser, `Entity.from_dict`, foundry-generated specs, future programmatic builders. Symmetric `to_dict` emission of the `abstract` boolean. `Entity.from_dict` reconstructs per-modulator sensors (pre-existing roundtrip gap that the new constructor warning would have made spuriously fire) + legacy-pre-C4 dict-shape compat (no `sensors` and no `abstract` → load as `abstract=True`). Bundled audit: 115 modulators in `_data/components/` + 11 in `scenarios/embodiment/` declare `abstract: true`.

Hard-error flip shipped via PR-A.3 (2026-05-29): warning replaced with `ConfigurationError` raised from `SpecModulator.__init__`. Marker shortened from `(C4 v0.9 deprecation)` → `(C4)` matching the C5+C6 post-flip convention. `import warnings` dropped (unused after flip). Regression guard: [tests/unit/test_modulator_abstract.py](../../tests/unit/test_modulator_abstract.py) (15 tests — bundled audit re-anchored on "no ConfigurationError on `_parse_entity`").

**Follow-up status:**

- **C4-followup-1: Imagination + foundry pipelines emit bare LLM specs — SHIPPED** (PR #300, 2026-05-29). Per-source contract enforced: LLM-derived `_parse_entity` callers (3 sites in `imagination/trigger.py`, 2 in `simulation/foundry.py`) route through `normalize_llm_entity_spec` which fills `abstract: True` on capability-only modulators. Bundled YAMLs (`component_registry`), user-authored YAMLs (`campaign_runner` DM specs), and curated arc data (`generative_runner` world entities) deliberately DO NOT normalize — the 1.0 `ConfigurationError` is the user-facing migration signal asking the author to declare `abstract: true` explicitly. The LLM prompt in `simulation/entity_designer.py::_SEM_SCHEMA_PROMPT` is updated with the right-shaped ask (split example modulators — verb-group with `abstract: true` vs component-part with sensors — prevents cargo-cult of both fields on the same modulator). Audited scenarios/ YAMLs: zero bare modulators across 51 files. **Post-flip the normalizer is LOAD-BEARING**: without it, every LLM-imagined entity with a bare modulator would crash `_parse_entity` on the runtime path. Regression-guarded by: (a) CI grep in `.github/workflows/test.yml` allow-listing the four non-LLM input source files + requiring `normalize_llm_entity_spec` on every other `_parse_entity` caller, (b) [tests/unit/test_normalize_llm_entity_spec.py](../../tests/unit/test_normalize_llm_entity_spec.py) (15 tests), (c) CLAUDE.md "Architectural invariants" entry naming the contract.

- **C4-followup-2: Sensor-promotion audit.** The 115 bundled modulators were marked `abstract: true` uniformly by an audit script. A small subset (~5-15) arguably should grow real sensors instead so `compute_integrity()` reflects "this capability is degraded" — `cradle_lever_door.mechanism` should own `lever_position`, `wizard.magic` should own `mana`, etc. **Approach:** group the 115 by modulator-name category (combat/social/maintenance/usage are clearly verbs and stay abstract — ~95+ of them); the real audit shrinks to the ~15 ambiguous ones (`magic`, `mechanism`, `lifecycle`, `physical`, `expression`). For each: does the entity carry sensors that conceptually belong to this modulator's working order? Net diff is small (5-10 promotions); the architectural signal it sends ("we know what state belongs where") is large. Polish pass — no hard deadline, can land any time post-1.0.

### C5. Entity health as direct sensor — SHIPPED

Deprecation phase shipped via PR #220 (2026-04-30): parse-time `DeprecationWarning` when an entity has modulators with sensors AND a direct `health` sensor without `health: derived` opt-in.

Hard-error flip shipped via PR #299 (2026-05-29): warning replaced with `ConfigurationError` raised from `_check_c5_direct_health` in [`src/maxim/embodiment/spec.py`](../../src/maxim/embodiment/spec.py). Single canonical opt-in spelling (string `"derived"` only — boolean `True` rejected) preserved from the deprecation cycle. Bundled audit (80 components + scenarios) confirms zero offenders. Regression guard: [tests/unit/test_c5_direct_health.py](../../tests/unit/test_c5_direct_health.py) (8 tests).

**Promoted from §1.1-T4 to 1.0** because the bundled audit cleared all offenders and C5 has no follow-up prerequisites (unlike C4's imagination/foundry normalizer or C6's DefaultNetwork standalone-path coupling).

**Marker convention post-flip:** bare `(C5)` in the exception message — the `"deprecation"` suffix used by C4's pre-flip warning (`"(C4 v0.9 deprecation)"`) and C6's pre-flip warning (`"(C6 deprecation)"`) becomes anachronistic once the warning is an error. The natural endpoint for all three is `(C{N})` once C4 and C6 also flip.

### C6. Raw constructor enforcement — SHIPPED

Deprecation phase SHIPPED (PR #221, 2026-05-03); hard-error flip SHIPPED (PR #301, 2026-05-29) via path (b) — `DefaultNetwork._init_pain_circuit` keeps its explicit `PainBus(_allow_raw=True)` opt-out (Wave-2 split-subscriber-ownership fix in `pain_bus_unification.md` is the proper resolution but ships separately).

**Technical debt acknowledgment:** path (b) is an interim 1.0 contract, not a permanent endpoint. The Wave-2 fix in `pain_bus_unification.md` is the final architecture; until it lands, the DN opt-out site is the **single allow-listed production caller** of `PainBus(_allow_raw=True)`. The CI grep allow-list ([.github/workflows/test.yml](../../.github/workflows/test.yml) "C6 hard-error allow list") covers it explicitly; new production opt-outs require updating both the allow-list and CLAUDE.md in the same commit, with a `TODO(wave-N)` comment at the call site. The `TestDefaultNetworkOptOutStillWorks` tripwire in [tests/unit/test_c6_raw_construction.py](../../tests/unit/test_c6_raw_construction.py) pins that the DN-shape construction stays valid.

**Shipped:** Added keyword-only `_allow_raw: bool = False` to `PainBus.__init__`, `ReactionBus.__init__`, and `MemoryHub` (as a `kw_only=True` dataclass field, kept out of `repr` / `compare`). When `_allow_raw` is False, each constructor emits a `DeprecationWarning` naming the canonical builder (`build_pain_bus` / `build_reaction_bus` / `build_memory_hub`) plus the load-bearing rationale (Wave 1+2 silent-no-op bug class). Each warning also prints to stderr (`print(f"DeprecationWarning: {msg}", file=sys.stderr)`) for human visibility — `DeprecationWarning` is silenced by Python's default warning filter outside `__main__` and by pytest's global `ignore::DeprecationWarning`, so the stderr line is the load-bearing signal that callers actually see. Mirrors the C4/C5 + `cli_utils._resolve_persona_mode` pattern.

`build_pain_bus`, `build_reaction_bus`, and `build_memory_hub` now pass `_allow_raw=True` internally so production paths are silent. `PainBus.__init__`'s internal `ReactionBus(...)` call also passes `_allow_raw=True` — only the *outer* type's warning fires (one warning per raw construction, not two).

`simulation/orchestrator.py::_setup_sim_sandbox` migrated from raw `PainBus()` to `build_pain_bus(hippocampus=None, nac=None)` — the early sandbox bus pattern routes through the canonical door. `default_network/network.py::_init_pain_circuit` keeps its raw `PainBus(_allow_raw=True)` opt-out with an explicit `TODO(wave-2)` comment naming the deferred plan (`pain_bus_unification.md` "Latent bridge × subscriber attribution-asymmetry trap" + this doc's §1.1-T4 C6 prerequisite).

The `ReactionBus` warning text deliberately distinguishes itself from `PainBus`/`MemoryHub`: it has no current production silent-no-op bug class (PainBus constructs ReactionBus internally), but the door is enforced now to be forward-protective for the Wave-3 ordering where `build_bio_stack` will construct a standalone ReactionBus and hand it to `build_pain_bus(reaction_bus=...)`. The text says so explicitly so a user hitting the warning doesn't hunt for a non-existent retroactive bug.

CI grep guard at `.github/workflows/test.yml` blocks new `_allow_raw=True` opt-outs in `src/maxim/`. The allow-list covers exactly the 5 legitimate sites (4 internal builder calls + DefaultNetwork's deferred opt-out). New production opt-outs require updating both the allow-list and CLAUDE.md in the same commit, with a `TODO(wave-N)` comment at the call site. Same shape as the existing `write_mesh_config` / install-core / admin-update / admin-llm allow-lists.

Test surface: ~60 raw-construct sites updated to `_allow_raw=True` across 15 test files + 5 scripts. Regression-guarded by `tests/unit/test_c6_raw_construction_warnings.py` (17 tests covering: raw-warns, `_allow_raw=True`-silent, builder-silent, kw-only-field-shape, builders-propagate-allow-raw-internally).

Pre-merge two-lens review: 5 findings folded (ReactionBus message clarification, stderr-print visibility, CI grep allow-list, DN TODO marker, plan §1.1-T4 prereq mirror). 1 NIT deferred (future tag-bypass-with-reason-string enhancement).

**1.0 flip SHIPPED via PR #301 (2026-05-29):** `PainBus.__init__`, `ReactionBus.__init__`, and `MemoryHub.__post_init__` now `raise TypeError(...)` instead of emitting `DeprecationWarning + stderr` when `_allow_raw=False`. The `_allow_raw=True` opt-out keyword is preserved (used by builders + the DefaultNetwork deferred site + ~60 test sites). Message marker shortened from `"(C6 deprecation)"` to `"(C6)"` matching the C5 convention. Regression guard: [tests/unit/test_c6_raw_construction.py](../../tests/unit/test_c6_raw_construction.py) (18 tests covering raw-raises, opt-out-silent, builder-silent, DefaultNetwork opt-out tripwire).

### C7. Dormancy audit follow-up — broader cleanup candidate list (1.0 cleanup track)

**Surfaced by:** PR #282 (2026-05-26) — CLAUDE.md Principle 2 dormancy markers for four canonical mechanisms. PR #282 also ran a parallel Explore-subagent sweep across `imagination/`, `memory/`, `simulation/`, `tools/`, `models/language/`, `reactions/`, `proprioception/`, `embodiment/`, `runtime/` that surfaced **27+ additional candidates** — over the kickoff's 20-threshold which says "something larger is going on (dead-code accumulation, refactor leftovers)."

**False-positive lesson from PR #282 spot-check:** an early subagent flagged all 6 `*_from_snapshot` helpers in [memory/snapshot.py:406-446](../../src/maxim/memory/snapshot.py#L406-L446) as dormant, but those are production-used via an internal dispatch table at lines 608-613. Grep-only sweeps produce noise; **every candidate needs hand-validation against indirect-dispatch patterns** (dispatch tables, plugin discovery, reflection, duck typing) before marking.

**Candidates to triage before 1.0** (sorted by suspicion strength from PR #282's spot-check; verify each independently):

- `memory/atl.py:576` — `get_by_modality` — retrieval by sensory modality; only definition site
- `memory/atl.py:621` — `propose_relationship_type` — semantic relationship proposal; only definition site
- `memory/semantics.py:269,278` — `save_registry` / `load_registry` — registry persistence helpers; only definition sites
- `memory/hippocampus.py:1089` — `get_memories_by_index` — index-key lookup; only definition site
- `memory/hippocampus.py:1179` — `repair_consistency` — internal consistency maintenance; only definition site
- `memory/hippocampus_retrieval.py:367` — `get_associated_ids` — associated-memory id lookup
- `memory/spatial.py:19` — `SpatialContext` dataclass — forward-compat type per module docstring
- `memory/hippo_tracer.py:145,177` — `traced_recall` / `traced_recall_associated` nested functions in tracer
- `imagination/cache.py:75` — `mention_count` — phrase mention-count accessor
- `imagination/trigger.py:1052` — `clear_session` — session-cache clear (verify session-end indirection)
- `tools/learned_index.py:165` — `register_manual_keywords`
- `tools/discovery.py:53` — `evict_stale_discoveries`
- `tools/mode_switch.py:123` — `get_switch_history`
- `proprioception/pain_bus.py:284` — `recent_by_type` — bus history accessor
- `embodiment/cerebellum.py:539` — `observe_action_sequence` — motor program crystallization
- `embodiment/cerebellum.py:553` — `find_programs_for_goal` — goal-keyed program query
- `embodiment/cerebellum.py:661` — `query_engrams` — engram retrieval
- `embodiment/reflex.py:150` — `reset_state` — reflex-registry state clear
- `simulation/arcs.py:51,74` — `turn_range` / `phase_names` — narrative-arc accessors
- `simulation/campaign_runner.py:63` — `run_dm_campaign`
- `simulation/conversational_source.py:130` — `inject_sensor`
- `simulation/dm_runtime.py:882` — `get_active_entity`
- `simulation/entity_designer.py:166,202` — `design_npc` / `design_item`

**Separately flagged for deeper investigation:** the `simulation/` subsystem subagent reported "30+ additional dormant methods in logging, validation, foundry, experiment introspection, and tool modules" beyond the 7 surfaced above. That magnitude is a **dead-code-accumulation signal**, not individual-mechanism dormancy — treat as a distinct audit pass (closer to P3's energy-code hard-remove shape than C7's per-method marker shape).

**Approach for the 1.0 cleanup track:**
1. **Per-candidate hand-validation** against indirect-dispatch patterns (~2-3 minutes per candidate; cap at the ~23 atomic candidates above).
2. **For true-positive dormant mechanisms** (no production caller, no bio-thesis claim attached): apply the `Dormant since <date>: <reason>` marker per CLAUDE.md Principle 2.
3. **For true-dead-code** (no production caller AND no scoped revival path AND no future-proofing rationale): hard-remove following P3's pattern. Bias toward dormancy when ambiguous — cascade-deletion risk is documented in CLAUDE.md.
4. **Simulation-subsystem mass cleanup** is its own audit; not bundled into this candidate list.

**Cost ceiling:** ~3-4 hours wall to triage the 23 atomic candidates above; the simulation-subsystem sweep is open-ended and scoped separately.

**Why a follow-up rather than blocking 1.0:** PR #282's canonical 4 markers carry the principle's hard-shipped form. The broader candidates need per-mechanism validation that doesn't fit a single PR's review surface. Cross-confirmed false-positive rate (1-of-7 sample) means the discipline must be hand-validation, not subagent-trust.

---

## Section 6: Docs

### D1. Agent memory transfer docs — SHIPPED (2026-06-15)

Covered by the existing `docs/user/` set — `cross-session-learning.md` (the end-to-end recall→inspect→resume loop) + `memory-user-guide.md` (tier lifecycle + what gets remembered) + `concept-decomposition.md`. The 1.0 honesty pass (2026-06-15) reconciled `cross-session-learning.md` with the locked release thesis: the substrate *persistence/recall* is the earned claim; behavioral change is the LLM choosing to act on recalled context, not a guarantee (the prior often dominates). No remaining "agents learn and improve" framing in user docs.

### D2. API/CLI surface review — SHIPPED (covered by CC2 + CC4)

The verb/dataclass surface is classified in `docs/user/stable_api.md` (CC2 — 17 stable verbs + experimental tags); env-var + CLI-flag classification is in `docs/user/configuration.md` (CC4); the full CLI is in `docs/user/cli-reference.md`. No undocumented-but-public capability gaps surfaced.

### D3. Final docs pass — SHIPPED (2026-06-15)

1.0 release announcement landed on main (#373 — `docs/announcements/maxim_1_0_release.md` + HTML guide, the honest-benchmark framing). The 1.0 honesty pass aligned `README.md` (already framed as a bio-inspired LLM harness that *augments* LLM context), `cross-session-learning.md`, `docs/index.md`, and `docs/publication_guide.md` with that thesis. Version bumped 0.9.3 → 1.0.0 (`pyproject.toml` + `src/maxim/__init__.py`, kept in sync) + version refs in `docs/index.md` / `docs/publication_guide.md`.

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
11. **C4-C6** — all three hard-error flips SHIPPED (PRs #299 + PR-A.3 + #301, 2026-05-29).
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

### CC13. Auth format-freeze audit (~50 LOC + ~16 tests + 1 doc page) — SHIPPED (2026-06-15)

**Plan:** [auth_format_freeze_audit.md](auth_format_freeze_audit.md) (drafted 2026-06-04, SHIPPED 2026-06-15)

**Shipped via:** branch `feat/1-0-cc13-auth-format-freeze` — all four surfaces (A1 `api_key_ref` scheme reservation doc; A2 bundle `signer_identity` field + `signature_algorithm` registry doc; A3 `MeshConfig` reserved-null `cluster_keys`/`cluster_trust_anchors`/`cluster_auth_mode`; A4 leader-proxy auth scheme dispatch) in one PR. Two-lens pre-merge review folded (no CRITICAL; hardened non-ASCII Bearer → clean 401, CC3 docstring cross-reference, `op://` form, A1 closed-enum note, `signer_identity` namespace note, `mesh_setup.py` TODO markers). Full fast suite green (7997 passed). See the plan doc for the per-surface detail + regression-guard test paths.

Narrow format-freeze pass on the four security-shaped surfaces shipping in 1.0, so the future hardware-token / signed-bundle / mTLS / WebAuthn work planned for 1.1+ alongside Hivemind P2P is not boxed out by 1.0 schema choices. **Does not implement authentication** — full pluggable auth provider abstraction stays a 1.1+ track. This is purely about whether the slots already shipping in 1.0 admit the future without a breaking change.

**Surfaces audited:**

1. **`lanes.<tier>.remote_api_key_ref` URI scheme** — currently accepts file paths + `keyring:` URIs. Reserve `pkcs11:` / `fido2:` / `tpm:` / `vault:` / `op:` / `env:` schemes in the doc; validator stays deny-by-default.
2. **Hivemind bundle `signature_algorithm`** — publish the recognized-values registry (`ed25519`, `ed25519-pgp`, `webauthn`, `pkcs7`, etc.) so 1.2 P2P verifiers share a vocabulary. Reserve `signer_identity: str | None` field on the manifest (10 LOC) so 1.1+ can bind verified identity to `contributor_id` without retrofitting.
3. **`mesh.yml::cluster_key` shape** — add three reserved-null sibling fields (`cluster_keys: list[str] | None` for rotation, `cluster_trust_anchors: list[str] | None` for asymmetric mesh auth, `cluster_auth_mode: str | None`). Parser already tolerates absent fields per the frozen dialect.
4. **Leader proxy `Authorization:` scheme dispatch** — ~20 LOC refactor to parse the scheme before the credential, return 401 on unknown schemes rather than treating as malformed Bearer. Reserves the dispatch table for future `Signature` / `HSM-Sig` / mTLS variants.

**Why parallel to benchmarking is the right timing:** all four are doc + freeze-shape additions with `None` defaults, no behavior change at 1.0. The benchmarking track measures behavior under load — this track touches none of that surface. Genuinely independent work.

**Pre-merge review:** two-lens (Executor + Architecture) — freeze-decisions are exactly the case where pre-merge review pays off. Specifically watch for boxed-out future schemes, name conflicts with existing Hivemind identifiers, and reserved namespace scope.

**Wall:** 0.5–1 day implementation + 0.5 day review. Owner unassigned; can fold into any natural slot during the Phase 3 benchmarking window.

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

### 1.1-T4. C-series hard-error flips — SHIPPED

All three C-series hard-error flips landed in Phase 1 of the sequenced 1.0 plan (2026-05-29). The original §1.1-T4 marker is retained for traceability — content moved into the per-flip sections.

**Exception type per C-series rule:** C4 and C5 use `ConfigurationError` (parse-time YAML schema violation — config invariant); C6 uses `TypeError` (constructor-signature violation — the wrong constructor was called). The split is intentional and follows Python convention.

- **C5** hard-error flip SHIPPED (PR #299, 2026-05-29) — see §C5.
- **C6** hard-error flip SHIPPED (PR #301, 2026-05-29) via path (b) with `DefaultNetwork._init_pain_circuit` retaining its `PainBus(_allow_raw=True)` opt-out — see §C6. Wave-2 split-subscriber-ownership fix in `pain_bus_unification.md` is the proper resolution but ships separately.
- **C4** hard-error flip SHIPPED (PR-A.3, 2026-05-29) — see §C4. Hard prerequisite C4-followup-1 (LLM-spec normalizer, PR #300) cleared first; soft prerequisite C4-followup-2 (sensor-promotion audit) is a polish pass that can land any time post-1.0.

### 1.1-T5. Agent-backed entities (revival path)

**Plan (deferred):** [deferred/agent_backed_entities.md](deferred/agent_backed_entities.md)

Revives if scene_actor_affordances doesn't close the dragon-narration symptom OR if Minecraft demo exposes a cognition gap (zombies need pathfinding, villagers need trade memory).

### 1.1-T6. B5 embodiment/narrative separation

**Plan (deferred):** [deferred/b5_embodiment_narrative_separation.md](deferred/b5_embodiment_narrative_separation.md)

B3 Acting Coach shipped. B5 is a shell.

---
