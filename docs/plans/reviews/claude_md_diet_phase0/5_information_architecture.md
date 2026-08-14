# Phase 0 — Lens 5: Information Architecture

**Input:** CLAUDE.md @ chore/claude-md-diet worktree (255,759 chars, ~62.8K tokens, byte-identical to main).
**Inventory:** 36 Lessons-learned entries (L1–L36 below), 64 Architectural-invariant bullets (A1–A64), 5 Working principles, 31 quick-reference table rows, 83 env vars, plus 12 other sections.
**Job:** validate the Appendix's six-brief cut against the actual clusters; per-brief ToC + exact assignment; straddler homes; routing table; orphans.

---

## 1. Entry inventory (stable IDs used throughout)

### Lessons learned (document order)

| ID | Slug (short title) | Tag |
|---|---|---|
| L1 | harness-own-repo-provenance (Exp 42b) | eng |
| L2 | harness-not-on-leader (Exp 37 cascade) | eng |
| L3 | ec-centroid-drift | beh |
| L4 | key-embedded-degenerate-stats (Welford placement) | eng |
| L5 | silent-noop-invariants-into-types | eng |
| L6 | autosave-under-rwlock-deadlock | eng |
| L7 | mutable-globals-module-extraction | eng |
| L8 | per-agent-stash-dicts | eng |
| L9 | auth-in-health-probes | eng |
| L10 | nac-class-name | eng |
| L11 | lane-tier-names | eng |
| L12 | lane-capability-vs-placement | eng |
| L13 | leaderproxy-before-normalize-args | eng |
| L14 | dead-code-accumulates | eng |
| L15 | optin-env-vars-autouse-scrubs | eng |
| L16 | http-via-utils-http | eng |
| L17 | role-detection-first-runtime-action | eng |
| L18 | llm-profile-vs-active-model-drift | eng |
| L19 | backenderror-fix-hint-static | eng |
| L20 | subcommand-dispatch-bypasses-logging | eng |
| L21 | review-round-before-merge (process) | eng |
| L22 | fold-commits-on-merge-target (process; untagged) | — |
| L23 | peer-backend-one-http-call | eng |
| L24 | streaming-contract-difference | eng |
| L25 | probe-entry-point-health-check | eng |
| L26 | per-tier-timeout-flow | eng |
| L27 | proxy-context-admission-gate | eng |
| L28 | ttft-keepalive-write-lock | eng |
| L29 | httpx-stream-ctx-lifetime | eng |
| L30 | context-similarity-denominator | beh |
| L31 | probe-classify-single-place | eng |
| L32 | mesh-yml-parser-frozen | eng |
| L33 | mesh-yml-declarative-vs-util-state | eng |
| L34 | direct-key-over-context-similarity | beh |
| L35 | painbus-reactionbus-coexistence | eng |
| L36 | optional-deps-canonical-surface | eng |

### Architectural invariants (document order)

| ID | Slug | | ID | Slug |
|---|---|---|---|---|
| A1 | memory-tier-one-way | | A33 | tool-input-schema-dual (CC9) |
| A2 | separate-episodic-instances | | A34 | record-event-canonical-no-scn-bus |
| A3 | tool-results-via-bus | | A35 | build-executor-canonical |
| A4 | atomic-write-json | | A36 | build-pain-bus-canonical |
| A5 | frozen-dataclass-audit (CC3) | | A37 | build-default-network-canonical |
| A6 | format-version-contract (CC1) | | A38 | build-reaction-bus-canonical |
| A7 | llm-via-router | | A39 | build-memory-hub-canonical |
| A8 | prompt-byte-stable (caching) | | A40 | build-bio-stack-canonical |
| A9 | one-http-call (dup of L23) | | A41 | recall-touch |
| A10 | probe-entry-point (dup of L25) | | A42 | promotion-pressure |
| A11 | for-url-concurrency-safe | | A43 | memoryrecord-fields-compat |
| A12 | transports-typed-per-purpose | | A44 | episode-valence-default |
| A13 | percept-wire-format | | A45 | spreading-activation-shapes |
| A14 | workerpool-owned-by-llmworker | | A46 | reward-bias-clamp |
| A15 | no-new-silent-swallows | | A47 | save-cerebellum-session-end |
| A16 | requestcontext-contextvar | | A48 | nac-decay-wired-loop-8.5 |
| A17 | http-errors-typed | | A49 | scn-temporal-coupling [beh] |
| A18 | raw-proxy-forward-reserved | | A50 | scn-oscillator-default-on (B2) |
| A19 | tool-pain-direct-attribution | | A51 | affordance-encoder-separate |
| A20 | b8-delta-attribution | | A52 | signed-sensor-range-encoding |
| A21 | drive-pain-channel-split (+SCN gap, 42b) | | A53 | drive-spec-types |
| A22 | motor-credit-value-progress-sign | | A54 | entity-acquisition-side-effects |
| A23 | side-effects-typed-channel | | A55 | self-effect-writeback |
| A24 | tool-cancel-noop (CC11) | | A56 | three-interaction-levels |
| A25 | stable-hash-persistence | | A57 | narrative-phase-act |
| A26 | nac-ec-pair-persistence | | A58 | energy-bridges-deleted |
| A27 | reachy-ws-transport | | A59 | tick-in-evaluate-failures |
| A28 | reachy-head-world-frame | | A60 | normalize-llm-entity-spec |
| A29 | goto-target-clamped-locked + retained axes | | A61 | hivemind-layer |
| A30 | perceptsource-actionsink-minimal (CC8) | | A62 | config-writer-canonical |
| A31 | cwd-relative-api-verbs (CC10) | | A63 | detect-role-single-source |
| A32 | llm-call-registry + stall-threshold | | A64 | peer-yml-auto-migration |

Note for the condensation lens: A9≈L23 and A10≈L25 are near-duplicates (invariant bullet restates the lesson). Both land in the same brief either way, so the IA is unaffected; merge is safe.

---

## 2. Verdict on the Appendix's six-brief cut

**The cut HOLDS.** Every one of the 100 entries lands naturally in one of the six briefs or in core. Three adjustments against the actual clusters:

1. **llm-routing is by far the largest brief** (~30 entries + ~40 env vars + doctor + remote-update). Keep it ONE file but structure it in two hard-titled halves — *Router & backends* (router, typed errors, peer backend, prompt-cache mechanics x-ref) and *Topology: peer/mesh/proxy/tunnel/doctor* (mesh.yml, drain, proxy gates, role interplay, doctor maintenance). If a future split is ever needed, that seam is where it goes. Do not split now: the two halves share the failure vocabulary (typed BackendError, probe entry point, one-call rule).
2. **Reachy/hardware safety gets a named top-level section inside embodiment** ("Hardware safety — read before commanding motion": A27, A28, A29). Motors were physically destroyed twice; this cluster must be the first screenful of the embodiment brief, not interleaved with SEM semantics. A separate `robot-hardware` brief was considered and rejected — the motion-safety rules constantly reference SEM frames (body-relative yaw, drive sensors on the head) and a 7th file would break the "exactly one brief" promise for anyone touching `embodied_runtime/`.
3. **hivemind (A61) has no obvious home in the six.** Homed in **bio-memory** (it is substrate sharing: nac_merge/ec_merge semantics, frozen-modality preservation) with a cross-ref from persistence-config (bundle/ZIP/scrub format). Flag: when 1.2 P2P work starts, promote to a 7th brief `hivemind.md`; until then it is one section.

Also validated: **imagination + foundry + cradle cluster with simulation-experiments**, not bio-memory — they are scene/content generation machinery (the Appendix's cut didn't name them; they'd otherwise be orphans).

---

## 3. Per-brief design

Common skeleton for every brief (synthesis template — the connective prose is NEW writing):

```
# <subsystem> — working brief
1. Mental model (5–15 lines, synthesized)
2. Key files (relocated quick-ref rows)
3. Invariants (one-liners + Regression guard + docs/lessons/ link)
4. Live gotchas / known gaps
5. Env vars owned
6. Lesson archive links (docs/lessons/<slug>.md)
```

### 3.1 `docs/agents/bio-memory.md`

**Mental model (SYNTHESIS — the highest-value new prose in the whole exercise):** one data-flow paragraph + diagram that currently exists nowhere: percept → LinguisticEncoder (decomposition strategies) → EC `pattern_complete_or_separate` (thresholds, frozen modalities) → ATL / Hippocampus tiers → NAc (reward_bias / cluster_reward_bias / causal links / Welford) → SCN temporal credit → back into prompt annotations (Wire-A, Wire-1) and substrate-primary action selection. Today this chain must be reassembled from ~15 scattered entries.

- **Lessons assigned:** L3 (ec-centroid-drift), L4 (key-embedded-stats), L10 (nac-class-name), L30 (context-similarity-denominator — home here: it's a `decisions/nac.py` function contract; x-ref from embodiment).
- **Invariants assigned:** A1, A2, A26, A34, A41, A42, A43, A44, A45, A46, A47, A48, A49, A50, A51, A52, A61.
- **Quick-ref rows relocated:** Memory, Causal learning, Temporal credit, Substrate encoding, Valence, Cross-layer wiring (memory_hub), Percept schema? — no, Percept schema → runtime-tools; Valence → here.
- **Env vars owned:** MAXIM_SUBSTRATE_PATH, MAXIM_CONCEPT_DECOMPOSITION, MAXIM_NAC_MIN_CONFIDENCE, MAXIM_NAC_REWARD_BIAS_DISABLED (x-ref sim: Exp 37 ablation arm), MAXIM_EC_TRACE_ACTIVATIONS, MAXIM_DISABLE_CLUSTER_BIAS_ANNOTATION, MAXIM_DISABLE_VARIANCE_ANNOTATION, MAXIM_NAC_CLUSTER_REWARD_BIAS_DECAY_TAU, MAXIM_NAC_TEMPORAL_CREDIT_WEIGHT, MAXIM_PLACE_CODE_EXTEROCEPTION (x-ref embodiment: audio channel), MAXIM_HIPPO_TRACE, MAXIM_ATL_TRACE, MAXIM_NAC_TRACE.
- **Synthesis vs relocation:** mental-model paragraph + the EC-threshold table (text 0.44 / sensor 0.85 / frozen modalities {interoception, audio}) are SYNTHESIS — the sensor-vs-text threshold confusion already caused a real incident (memory: "Sensor threshold is 0.85 NOT 0.44"). Everything else is relocation.
- **Gotchas section:** isolated-vs-sequential drift detection rule (from L3), decay is tick-anchored not wall-clock except NAc.load(), graduation-candidates doc pointer.

### 3.2 `docs/agents/llm-routing.md`

**Mental model (SYNTHESIS):** the dispatch chain — FunctionRouter picks a capability tier → LaneConfig (+placement axis) compiles to provider_priority → LLMRouter._try_provider failover over typed BackendErrors → backend (_MaximPeerBackend one-call | _OpenAIBackend) → leader proxy (admission gate, keepalive) → llama-cpp / cloud. Plus the topology half: leader/peer/solo roles, mesh.yml (read-only) vs ~/.maxim/util (mutable), tunnel, doctor.

- **Lessons assigned:** L9, L11, L12, L16, L18 (x-ref persistence-config: declarative-vs-runtime state), L19, L23, L24, L25, L26, L27, L28, L29, L31, L32, L33 (x-ref persistence-config), and L2's *hardening half* (singleton spawn guard + preflight — the incident narrative itself homes in simulation-experiments; see straddlers).
- **Invariants assigned:** A7, A9 (merge w/ L23), A10 (merge w/ L25), A11, A12, A13 (x-ref runtime-tools: Percept type), A16, A17, A18, A32.
- **Quick-ref rows relocated:** LLM routing, Lane tiers, Mesh.
- **Other sections absorbed:** `maxim doctor` maintenance guide (the "Maintaining this over time" subsection — recommend HERE rather than docs/user/: its audience is an agent adding a check, which is exactly the brief's reader; core keeps the 5-line summary per the plan); Remote Update Workflow details (core keeps the 2 command lines).
- **Env vars owned (~40):** 8 provider API keys; MAXIM_ROLE, MAXIM_LLM_ENABLED, MAXIM_LLM_PROFILE, MAXIM_LLM_N_CTX, MAXIM_AUTO_DOWNLOAD_MODELS, MAXIM_SKIP_REMOTE_PROBE, MAXIM_REMOTE_PROBE_{FIRST,RETRY}_TIMEOUT_S, MAXIM_REMOTE_PROBE_CACHE_TTL_S, MAXIM_LLM_CALL_TIMEOUT_S, MAXIM_HTTP_TRACE, MAXIM_BACKEND_TRACE, MAXIM_HEARTBEAT{,_INTERVAL_S,_STALL_S}, MAXIM_LANE_TRACE, MAXIM_PEER_LOG_REQUESTS, MAXIM_DRAIN_CACHE_TTL_S, MAXIM_AUTO_DRAIN_THRESHOLD, MAXIM_AUTO_UNDRAIN_PROBE_INTERVAL_S, MAXIM_PROXY_MAX_CONCURRENT, MAXIM_PROXY_RATE_LIMIT_RPM, MAXIM_PROXY_KEEPALIVE_INTERVAL_S, MAXIM_PROXY_CONTEXT_ADMISSION, MAXIM_PROXY_CONTEXT_OVERHEAD_TOKENS, MAXIM_LLM_CLOUD_ENABLED, MAXIM_MAX_CLOUD_LANES, MAXIM_LLM_REDACTION_POLICY, MAXIM_CLOUD_SESSION_BUDGET, MAXIM_LANE_<TIER>_REMOTE_{URL,MODEL,API_KEY}, MAXIM_LANE_<TIER>_TIMEOUT_S, MAXIM_AUTO_SPAWN_* (5), MAXIM_LEADER_PROXY_PORT.
- **Synthesis:** dispatch-chain diagram; a 6-row "which timeout fires first" table (three-layer timeout anatomy exists as a memory file, not in repo docs — this brief is its natural repo home). Everything else relocation.

### 3.3 `docs/agents/embodiment.md`

**Mental model (SYNTHESIS — second-highest value):** the **pain/credit channel map**. Currently A19, A20, A21, A22, L34, L35 each describe one face of a three-channel system; no single place states it. The brief opens with a table:
| Channel | Path | Semantics | Filter |
|---|---|---|---|
| 1 direct | FailureEvent → side_effects → ToolPainBridge → NAc.record_outcome | state-based, per-call | B8 delta-attribution |
| 2 bus | _publish_drive_pain → PainBus → subscribers | severity-latched on Entity | hysteresis + deepen fraction |
| 3 motor credit | drive_potential_diff → record_outcome SIGN | value-progress, not pain | producer-owned collateral gate |

- **Hardware-safety section FIRST:** A27 (WS transport, version-match), A28 (head world-frame counter-rotation), A29 (clamped+locked dispatch, retained-axes stash). These are the "motors get destroyed" rules.
- **Lessons assigned:** L34 (direct-key-over-context-similarity — home: bridges/tool_pain_bridge.py; x-ref bio-memory), L35 (painbus-reactionbus — home: proprioception/; x-ref runtime-tools for build_pain_bus).
- **Invariants assigned:** A19, A20, A21 (x-ref bio-memory for the SCN-oscillator sub-bullet), A22 (x-ref bio-memory: NAc consumer), A27, A28, A29, A53, A54, A55, A56, A58, A59 (x-ref runtime-tools: agent_loop tick sites), A60.
- **Quick-ref rows relocated:** Embodiment, Drives, Reactions, DN behaviors, Seed data, Tools row's embodiment sub-entries (tool_bridge, entity_map — split the row; see runtime-tools).
- **Env vars owned:** MAXIM_DEEP_EMBODIMENT, MAXIM_PAIN_CHAIN_TRACE, MAXIM_MOTOR_CREDIT_TRACE, MAXIM_ENABLE_BODY_STATE_PROMPT (x-ref sim: Exp 44 ablation), MAXIM_DISABLE_COACH_BODY_LAYERS (x-ref sim).
- **Gotchas:** `head=None` re-solves IK against retained world pose; `enable_motors()` gate; SDK/daemon era-match first; the two latch polarities (FailureMode.persistent vs drive_breach_severity); EmbodimentPerceptSource is Dormant.
- **Synthesis:** channel map + hardware-safety checklist. Relocation: everything else.

### 3.4 `docs/agents/simulation-experiments.md`

**Mental model (SYNTHESIS):** what a valid experiment looks like here — provenance stamped, apparatus declared, pre-registration, graduation ladder ([engineering] → [behavioral] via Roy/Exp), and the sim-cost discipline.

- **Lessons assigned:** L1 (harness-own-repo-provenance), L2 (harness-not-on-leader — home HERE: it is harness-topology discipline; x-ref llm-routing where the spawn-guard/preflight code lives).
- **Invariants assigned:** A57 (narrative-phase-act). (Most sim knowledge in CLAUDE.md is sections, not invariant bullets.)
- **Sections absorbed:** "Running simulations — keep them small" (bulk of it; core RETAINS three always-on bullets — see §6 core skeleton: `--interactive false`, config via `maxim config` not env, never co-locate leader+harness), "Simulation Reports" paths, the graduation-candidates pointer, apparatus-standards pointer (simulation_apparatus_standards.md S1–S8).
- **Quick-ref rows relocated:** Simulation, Substrate test infra, Generative campaigns, DM campaigns, Asset Foundry, Benchmarks, Research, Cradle, Imagination (x-ref bio-memory: NAc.decay_imagined_links + substrate-signal), Interactive UI.
- **Env vars owned:** MAXIM_PROVENANCE_VERBOSITY, MAXIM_DETERMINISTIC_SCENE_EMBODIMENT, MAXIM_DISABLE_IMAGINATION, MAXIM_DISABLE_IMAGINATION_SUBSTRATE_SIGNAL, MAXIM_EXP44_CAPTURE_LOG, MAXIM_OPERANT_ONLY_CREDIT, MAXIM_SUBSTRATE_TOOL_WHITELIST, MAXIM_CRADLE_MOTHER_DISABLE_CARE, MAXIM_SUBSTRATE_ACTIONS_PER_TURN.
- **Synthesis:** the "valid experiment checklist" (provenance assert + executed_code_provenance + apparatus declaration + pre-registration + two-process guard where hashing is involved). Relocation: sim-discipline bullets, report paths, env vars.

### 3.5 `docs/agents/persistence-config.md`

**Mental model (SYNTHESIS):** the **three-layer state map** — (1) declarative operator intent (config.json, mesh.yml, peer.yml, profiles.yml — never runtime-written), (2) mutable runtime state (~/.maxim/util/*.{role}.txt, active_llm_model), (3) persisted bio/session state (_format_version + schema_version envelopes, session dirs). Today this model is smeared across L18, L33, A6, A62–A64.

- **Lessons assigned:** L6 (autosave-rwlock-deadlock — home: it is a persistence-under-lock rule; x-ref bio-memory since the subject is Hippocampus), L17 (role-detection-first — x-ref runtime-tools cli ordering cluster).
- **Invariants assigned:** A4, A5 (frozen-dataclass audit — the registry of path-(a)/(b) dataclasses lives here; x-ref from every brief that owns one), A6, A25 (stable-hash — x-ref bio-memory where all five converted sites live), A62, A63, A64.
- **Quick-ref rows relocated:** Persistence.
- **Env vars owned:** MAXIM_DATA_BUDGET_GB.
- **Sections absorbed:** Versioning stays in core (short), but the two-places-in-sync rule gets a one-liner here too? — NO: keep single-home in core; this brief doesn't duplicate it.
- **Synthesis:** three-layer state map + the "which file may I write from code" decision table (config_writer allow-list, mesh_setup allow-list, atomic_write_secret vs atomic_write_text). Relocation: the rest.

### 3.6 `docs/agents/runtime-tools.md`

**Mental model (SYNTHESIS):** boot + loop anatomy — cli.py::main ordering (configure_logging → detect_and_apply_role → LeaderProxy → _normalize_args → dispatch), the canonical-builder family and WHY it exists (L5's push-into-types principle instantiated six times), the agent-loop tick sections (8.5 decay, drift tick, imagination hook), Tool ABC contracts.

- **Lessons assigned:** L5 (silent-noop-into-types — home here as the builder family's rationale; ALSO gets a one-line echo in core's Working-principles vicinity, see §5), L7 (mutable-globals-module-extraction), L8 (per-agent-stash-dicts — x-ref bio-memory), L13 (leaderproxy-ordering — x-ref llm-routing), L15 (optin-env-autouse-scrubs), L20 (subcommand-dispatch-logging), L36 (optional-deps).
- **Invariants assigned:** A3, A8 (prompt-byte-stable — home here: the actionable trigger is "adding a prompt section"; x-ref llm-routing for cache_control mechanics), A14, A23 (side-effects registry — x-ref embodiment: key producers/consumers), A24, A30 (x-ref llm-routing A12/A13), A31, A33, A35, A36, A37, A38, A39, A40 (all six builders; each x-refs the brief owning the wired subsystem).
- **Quick-ref rows relocated:** Agent loop, Tools (runtime half: registry, base, discovery, executor; embodiment sub-entries cross-listed in embodiment), Prompt composition, Percept schema, Multi-agent, Adding-env-vars row becomes part of core's routing table footnote.
- **Sections absorbed:** Python API (pymaxim) maintenance rules + package management; Architecture Essentials' thread-model bullet.
- **Env vars owned:** MAXIM_CONTEXT_POOL_* (8).
- **Synthesis:** boot-ordering diagram + builder-family table (builder → required kwargs → what forgetting used to silently break). Relocation: the rest.

---

## 4. Straddlers — home + cross-ref (explicit, per the brief's requirement)

| Entry | Home | Cross-ref from | Reason for home |
|---|---|---|---|
| L2 harness-not-on-leader | simulation-experiments | llm-routing | It's harness-run discipline; the code hardenings (spawn guard, preflight) are routing-side |
| L5 push-into-types | runtime-tools | core one-liner | Canonical example + all six instantiations live in bootstrap |
| L6 autosave-rwlock | persistence-config | bio-memory | Rule is "no lock-taking persistence inside write block"; subject happens to be Hippocampus |
| L8 per-agent-stash | runtime-tools | bio-memory | runtime/bio_integration.py + tool_dispatch.py |
| L13/L20 cli startup ordering | runtime-tools | llm-routing | Both are cli.py::main structure rules; proxy is the payload |
| L17 role-first | persistence-config | runtime-tools | Appendix assigns role detection to persistence-config; keep it with A63/A64 |
| L18 profile-vs-active drift | llm-routing | persistence-config | Symptom + resolution are LLM-server-side |
| L30 context-similarity denominator | bio-memory | embodiment | nac.py function contract |
| L32/L33 mesh.yml | llm-routing | persistence-config | Appendix puts mesh/peer in llm-routing; the declarative-vs-mutable PRINCIPLE is restated in persistence-config's state map (one-line, pointing home) |
| A5 frozen-dataclass audit | persistence-config | all briefs | Audit surface = persisted/wire-crossing types |
| A8 prompt-byte-stable | runtime-tools | llm-routing | Trigger is prompt-section authorship |
| A13 percept-wire-format | llm-routing | runtime-tools | It's a transport contract (pairs with A12) |
| A21 SCN sub-bullet (oscillator gap) | bio-memory (with A34) | embodiment | The gap is an SCN-intake fact; the dead emitter is embodiment-side |
| A22 motor-credit sign | embodiment | bio-memory | Producer (tool_bridge/sem.py) owns the semantics; NAc is consumer |
| A23 side_effects channel | runtime-tools | embodiment | Tool ABC + registry doc own the contract |
| A25 stable-hash | persistence-config | bio-memory | General persistence-boundary rule |
| A26 NAc+EC pair | bio-memory | persistence-config | Pairing rationale is bio (dangling node ids) |
| A48 NAc decay in loop 8.5 | bio-memory | runtime-tools | NAc semantics; loop is just the call site |
| A52 signed-sensor range | bio-memory | embodiment | Encoder (similarity/encoder.py) owns the fold |
| A59 tick cycle | embodiment | runtime-tools | Body.evaluate_failures owns it; agent_loop hosts two call sites |
| A61 hivemind | bio-memory | persistence-config | Substrate-merge semantics dominate; bundle format is the minor half |
| Builders A35–A40 | runtime-tools | bio-memory, embodiment | Single family, single pattern, single file (bootstrap/bio_stack) |
| MAXIM_NAC_REWARD_BIAS_DISABLED, MAXIM_ENABLE_BODY_STATE_PROMPT, MAXIM_DISABLE_COACH_BODY_LAYERS | owning subsystem (bio-memory / embodiment) | simulation-experiments | Var lives on the mechanism; the sim brief lists them in an "ablation-arm index" table pointing home |

Rule adopted for all straddlers: **home = the brief whose trigger paths an editor is most likely inside when the rule must fire**; the cross-ref is one line ("also see …"), never a restatement.

---

## 5. Orphans — stay in core CLAUDE.md (explicit list)

Per the plan's Exception clause (prose IS the enforcement) plus cross-cutting rules with no subsystem trigger path:

1. **All 5 Working principles** (two-tier tracking, dormancy-over-deletion, front-gate scope pressure, convergence-vs-divergence + actuation checklist, regression-guard citation). Keep full text (light trim per plan).
2. **L21 review-round-before-merge** — process, no mechanical guard. Full text (it is long; the condensation lens may trim the PR-number archaeology into a lesson file while keeping the rule + SCOPE TRIGGER + "different reader" rationale — that trim is compatible with the Exception clause since the guard line says "process invariant").
3. **L22 fold-commits-on-merge-target** — process, no mechanical guard. Same treatment. (Note: this entry is UNTAGGED — the split should add `[engineering]` or a `[process]` tag; flag for the truth lens.)
4. **L14 dead-code-accumulates** — process invariant ("periodic grep before publish"), 2 lines already.
5. **A15 no-new-silent-swallows** — cross-cutting code rule, already short; its guard is an ad-hoc grep. Core.
6. **"When making changes — required checks"** section incl. no-band-aid rule, worktree rule — core (already tight).
7. **Key Commands, Remote Update (2-line form), Versioning, Testing/Testing-efficiently, Simulation Reports pointer, Python API 2-line pointer, Active initiatives (trimmed to links per plan)** — core.
8. **Cross-cutting one-liner echoes** in core (full text in briefs/lessons): L5 (push-into-types), L7 (module extraction), L15 (env-var scrubs), L36 (optional-deps). These four are general engineering reflexes an agent needs *anywhere*; one line each in a "Cross-cutting code rules" list under the required-checks section, pointing at their home brief. This is the only deliberate double-listing; everything else is single-home + cross-ref.
9. **Three sim-safety bullets** retained in core from the Running-sims section: `--interactive false` from scripts; configure via `maxim config` (n_ctx drift → silent 0-action runs); never co-locate leader + harness/sim. These are session-killing hazards that fire before anyone would think to open a brief.

Nothing else fits no brief — audit complete: 36 lessons + 64 invariants each have exactly one assignment above (home) with cross-refs enumerated.

---

## 6. CLAUDE.md core routing table (proposed)

Place directly after "When making changes". Globs are repo-relative; a touch of ANY listed path ⇒ read that brief before editing.

| Touching | Read first |
|---|---|
| `src/maxim/memory/`, `decisions/`, `similarity/`, `integration/memory_hub.py`, `hivemind/`, `time/`, `agents/bus.py` (tiers/valence), `imagination/` (substrate side) | [docs/agents/bio-memory.md] |
| `src/maxim/models/language/`, `runtime/lane_*.py`, `runtime/function_router.py`, `runtime/leader_proxy.py`, `runtime/llm_server.py`, `runtime/llm_call_registry.py`, `runtime/stall_threshold.py`, `peer/`, `mesh/`, `tunnel/`, `doctor/`, `utils/http.py` | [docs/agents/llm-routing.md] |
| `src/maxim/embodiment/`, `proprioception/`, `bridges/`, `reactions/`, `default_network/`, `embodied_runtime/`, `motion/`, robot YAMLs, anything commanding Reachy motion | [docs/agents/embodiment.md] — **hardware-safety section is mandatory reading before motion code** |
| `scripts/benchmark_*`, `scripts/exp*`, `scripts/orient_*`, `simulation/`, `interactive/`, `tests/behavioral/`, `docs/experiments/`, running any sim | [docs/agents/simulation-experiments.md] |
| `utils/atomic_io.py`, `utils/format_version.py`, `utils/seeding.py`, `utils/paths.py`, `runtime/config_loader.py`, `runtime/config_writer.py`, `runtime/role.py`, `runtime/leader_mode.py`, any persisted-JSON shape, any frozen dataclass | [docs/agents/persistence-config.md] |
| `runtime/agent_loop.py`, `runtime/executor.py`, `runtime/bootstrap.py`, `runtime/bio_stack.py`, `runtime/agent_factory.py`, `runtime/agent_pool.py`, `runtime/tool_dispatch.py`, `tools/`, `agents/` (prompt_builder, percept, working memory), `cli.py`, `api.py` | [docs/agents/runtime-tools.md] |

Footnotes to the table: (1) multiple rows may match — read all matched briefs (rare; the straddler homes minimize it); (2) adding an env var → add to the owning brief's env table AND the core env index (see below); (3) `docs/lessons/` is archive — follow links only when the stub's trigger fires.

**Env vars in core:** replace the 146-line table with a one-line-per-var INDEX grouped by owning brief (name + 5-word purpose + brief link), or — cheaper — drop the index entirely and keep only MAXIM_LOG_FILE + the "adding env vars" rule, since every var is discoverable via its brief. Recommend the second; the retrieval lens should confirm.

---

## 7. Estimated post-split sizes

| File | Est. tokens |
|---|---|
| CLAUDE.md core (skeleton in §5/§6) | 6–8K (within the ≤10K target with headroom for compressed invariant stubs if the operator wants any duplicated in core — recommend NOT duplicating; stubs live in briefs) |
| bio-memory.md | ~5K |
| llm-routing.md | ~7K (largest; two-halves structure) |
| embodiment.md | ~6K |
| simulation-experiments.md | ~4K |
| persistence-config.md | ~3.5K |
| runtime-tools.md | ~5K |
| docs/lessons/*.md (~40 files) | full prose, unbounded |

Interaction with the plan's Stage 1: the compression contract says stubs stay in CLAUDE.md. **Amend (as the Appendix anticipates): compressed stubs live in the OWNING BRIEF, not in core** — core keeps only the routing table + orphans. Otherwise core cannot reach 10K (64 invariants × 4 lines ≈ 10K tokens by themselves). The Principle-5 lint must then run over core + the six briefs (extend `scripts/lint_claude_md_invariants.py`'s file list) so `Regression guard:` lines remain grep-able repo-wide — this satisfies the hard constraint "every Regression guard line survives" in spirit; if the operator insists on the letter (survives *in CLAUDE.md*), the fallback is a core appendix listing `slug → guard` one-liners (~2K tokens). Decide at synthesis.

---

## 8. Notes for the synthesis fold

- **Two entries duplicated verbatim** (A9/L23, A10/L25) — merge during split (condensation lens confirms).
- **L22 is untagged** — add a tag during the split (truth lens).
- **A21 is the largest single invariant (~7K chars, 6 sub-bullets)** — it decomposes cleanly: channel-split rule (embodiment stub) + SCN-gap note (bio-memory gotcha) + 42b validation narrative (lesson file) + regression guards (stay on the stub).
- **The five synthesis artifacts that justify the briefs' existence** (vs pure relocation): bio-memory data-flow chain; embodiment three-channel pain/credit map; llm-routing dispatch chain + timeout table; persistence three-layer state map; runtime-tools boot-ordering + builder-family table. Budget real writing time for these five; the rest is mechanical.
- **Memory-file dedup** (plan's third risk): several `~/.claude/.../memory/` reference files (three-layer-timeout-anatomy, scn-temporal-intake, recommend-action-reward-driven, prompt-construction) restate content that now gets a canonical repo home in a brief — after the split, those memory files should be re-pointed at the briefs.
