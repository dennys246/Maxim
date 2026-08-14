# Phase 0 — Lens 3: RETRIEVAL

**Input:** CLAUDE.md @ 255,759 chars / 669 lines (verified byte-identical to main checkout).
**Question:** what does a fresh session consult before its first edit (ALWAYS), what does it need only when touching a subsystem (ON-DEMAND → `docs/agents/<brief>.md`), and what is forensics-only (ARCHIVE → `docs/lessons/<slug>.md`)?

**Layer definitions used below**
- **CORE** = stays in slim CLAUDE.md, loaded every session.
- **BRIEF(x)** = moves to `docs/agents/x.md` (x ∈ bio-memory, llm-routing, embodiment, simulation-experiments, persistence-config, runtime-tools, per the plan appendix).
- **ARCHIVE** = narrative moves to `docs/lessons/<slug>.md`.

For every Lessons/Invariants entry the split is three-way: the *rule stub* (imperative + guard line) goes CORE or BRIEF; the *narrative* (dates, PRs, cascades, dead hypotheses) goes ARCHIVE. Column "Stub" says where the stub belongs from a retrieval standpoint.

---

## 1. Top-level section map (measured)

| Section | Lines | Chars | Layer verdict |
|---|---|---:|---|
| Header + Project Overview | 1–6 | 0.4K | CORE verbatim |
| Required checks + guardrails | 7–32 | 2.6K | CORE verbatim (this is the highest-retrieval block in the file) |
| Lessons learned (37 entries) | 33–106 | 71.0K | Split per-entry (§2) |
| Working principles | 107–124 | 10.3K | CORE per plan Exception, but ~2–3K of it is retold incident narrative (2026-07-16 sensor story, Roy-3c bisect detail) → ARCHIVE; keep triggers/checklists. Target ~6.5K |
| Running simulations | 125–139 | 5.6K | CORE — cost/safety critical. But line 129 (n_ctx drift, 4.1K) is 80% incident history → rule + `maxim config` procedure stays (~1K), three-leg bug archaeology → ARCHIVE. Target ~2.7K |
| Architectural invariants (~57 bullets) | 140–242 | 112.4K | Split per-entry (§3) |
| `maxim doctor` | 243–283 | 3.5K | CORE 5-line summary (~0.5K); "Maintaining this over time" → doctor module docstring / docs/user (plan already says so) |
| Key Commands | 284–346 | 3.5K | CORE (trim foundry/auto-curate examples slightly → ~3K) |
| Remote Update Workflow | 347–360 | 1.0K | CORE — contains the safe/unsafe peer-command split (state-mutation safety) |
| Versioning | 361–368 | 0.5K | CORE |
| Architecture Essentials | 369–392 | 5.9K | Mostly BRIEF material (per-subsystem mental models). Keep a ~1.2K compressed map in CORE; Percept/Reaction detail, imagination pipeline, mode system → briefs |
| Quick reference table | 393–428 | 6.5K | CORE but **merge with the routing table**: "Area → key files → `docs/agents/<brief>`" one row per area. Target ~3.5K. This is the natural seed of the plan's routing table |
| Environment Variables | 429–574 | 23.2K | Split (§4): ~2K CORE, rest BRIEF; paragraph rationales → module docstrings/lessons |
| Testing | 575–605 | 1.6K | CORE (threading pitfalls could go to runtime-tools brief, but at 3 lines they're cheap — keep) |
| Simulation Reports | 606–609 | 0.3K | CORE |
| Python API | 610–628 | 1.4K | 2-line CORE pointer; maintenance rules → BRIEF(runtime-tools) or api.py docstring. Only consulted when touching api.py |
| Active initiatives | 629–669 | 4.9K | **~85% ARCHIVE.** "Recently shipped 2026-04-24 / 04-20 / 04-17 / 04-11" blocks + "Version 0.7.0" are stale history (repo is at 1.0.x→1.1 — flag to Truth lens). CORE keeps: link to docs/plans/README.md, current version, one line per LIVE gate. Target ~0.5K |

---

## 2. Lessons learned — per-entry layer assignment

37 entries, 71.0K chars. "Stub" = where the compressed ≤4-line rule lives; narrative always → ARCHIVE.

| L# | Entry (short) | Chars | Stub | Rationale |
|---|---|---:|---|---|
| 35 | Harness must assert own-repo import (provenance) | 3,130 | **CORE** (safety) | Prevents retracted experiments — a session running ANY harness needs this before first sub-sim. 2-line stub + `assert_repo_interpreter` pointer |
| 37 | Don't run harness on leader machine | 4,007 | BRIEF(sim-experiments) | Only fires when running benchmark harnesses; the one-line "never co-locate" already lives in Running-sims (keep that line CORE). Cascade narrative = classic ARCHIVE |
| 39 | EC centroid drift (isolated-vs-sequential) | 4,105 | BRIEF(bio-memory) | Detection + fix rules are subsystem method; nothing a non-EC session needs |
| 41 | Key-embedded values → degenerate statistics | 1,805 | BRIEF(bio-memory) | Design principle, but fires only when adding statistics to keyed entities |
| 43 | Push silent-no-op invariants into types | 1,951 | **CORE** | Cross-cutting design principle cited by ~6 other entries; 3-line compression |
| 45 | Auto-save under RWLock write self-deadlocks | 1,573 | BRIEF(bio-memory) | Hippocampus-local; guard test exists |
| 47 | Mutable globals + module extraction | 593 | BRIEF(runtime-tools) | Fires only during module-extraction refactors |
| 49 | Per-agent stash dicts for multi-agent state | 3,504 | BRIEF(runtime-tools) | Multi-agent wiring detail; guard tests exist |
| 51 | Auth in health probes (401 = alive) | 676 | BRIEF(llm-routing) | |
| 53 | NAc class name | 477 | BRIEF(bio-memory) | CI-grep-enforced; core presence adds nothing the grep doesn't |
| 55 | Lane tier names | 442 | BRIEF(llm-routing) | CI-enforced |
| 57 | Lane capability vs placement axes | 3,517 | BRIEF(llm-routing) | Pure subsystem mental model — prime brief content |
| 59 | LeaderProxy before `_normalize_args()` | 460 | BRIEF(llm-routing) | |
| 61 | Dead code accumulates (orphan grep) | 344 | BRIEF(runtime-tools) | Pre-publish checklist item |
| 63 | Opt-in env vars need autouse scrubs | 985 | **CORE** (1-liner) | Cross-cutting: ANY session adding an env var hits it; belongs next to Testing |
| 65 | HTTP through utils/http.py | 1,027 | BRIEF(llm-routing) | CI-enforced |
| 67 | Role detection is first runtime action | 1,139 | BRIEF(persistence-config) | |
| 69 | llm.profile vs active_llm_model drift | 1,987 | BRIEF(persistence-config) | Error message now self-documents the fix |
| 71 | BackendError.fix_hint never user-controllable | 995 | BRIEF(llm-routing) | |
| 73 | Subcommand dispatch bypasses logging | 1,444 | BRIEF(runtime-tools) | |
| 75 | Review round BEFORE merge + SCOPE TRIGGER | 3,012 | **CORE full text** | Plan Exception: prose IS the enforcement (process invariant). Light trim of R1/R2/R3 attribution → ARCHIVE possible (~2.2K kept) |
| 77 | Fold commits must be ON THE MERGE TARGET | 1,832 | **CORE full text** | Same Exception; PR #435 narrative can shrink (~1.3K kept) |
| 79 | `complete_with_usage` exactly one HTTP call | 2,045 | BRIEF(llm-routing) | CI-grep-enforced; duplicated at invariant L155 — keep ONE one-liner |
| 81 | Streaming contract peer vs cloud intentional | 949 | BRIEF(llm-routing) | |
| 83 | Probe entry point health_check | 1,500 | BRIEF(llm-routing) | CI-enforced; duplicated at L156 |
| 85 | Per-tier timeout_s flow | 1,862 | BRIEF(llm-routing) | |
| 87 | Proxy context-overflow admission gate | 4,256 | BRIEF(llm-routing) | 90% mechanism description = brief body, not lesson |
| 89 | TTFT keepalive write lock | 2,273 | BRIEF(llm-routing) | |
| 91 | httpx stream ctx must outlive consumer | 1,360 | BRIEF(llm-routing) | |
| 93 | `_context_similarity` denominator directional | 1,814 | BRIEF(bio-memory) | |
| 95 | Probe classification single source | 2,215 | BRIEF(llm-routing) | |
| 97 | mesh.yml parser dialect FROZEN | 1,336 | BRIEF(llm-routing) | |
| 99 | mesh.yml declarative vs ~/.maxim/util mutable | 5,272 | BRIEF(persistence-config) | Largest lesson; the spec-vs-status rule is brief content, the five review-round citations are ARCHIVE |
| 101 | Direct lookup key beats context similarity | 1,996 | BRIEF(embodiment) | |
| 103 | PainBus rich carrier / ReactionBus typed | 2,303 | BRIEF(embodiment) | |
| 105 | optional_deps canonical surface | 2,756 | **CORE** (2-liner) | Cross-cutting: any new import of an optional package; the three entry points fit in 2 lines, table detail → brief |

**Lessons totals:** CORE-stub entries: 6 (≈5.5K after compression incl. the two full-text process rules). BRIEF stubs: 31 (≈6–7K as one-liners in briefs). Narrative to ARCHIVE: ≈58K of the current 71K.

---

## 3. Architectural invariants — per-entry layer assignment

~57 bullets, 112.4K chars. Multi-line entries measured as blocks.

### 3a. CORE candidates (cross-cutting or safety — a session in ANY area can hit these)

| L# | Entry | Chars | Compressed CORE form |
|---|---|---:|---|
| 145 | atomic_write_json for persistence | 342 | 1 line, near-verbatim |
| 152 | `_format_version` on every persisted JSON | 1,789 | 2 lines; envelope/schema_version detail → BRIEF(persistence-config) |
| 153 | LLM access through router only | 1,512 | 1–2 lines; allow-list detail → BRIEF(llm-routing) |
| 161 | No NEW silent exception swallows | 622 | 2 lines, near-verbatim |
| 180 | stable_hash for persistence-crossing values | 1,668 | 2 lines (silent cross-process breakage; hits anyone persisting) |
| 186 | **Reachy head world-frame / head=None counter-rotates** | 3,657 | 3 lines — HARDWARE-behavior safety + the generalized "verify actuation" rule; story → ARCHIVE |
| 187 | **`goto_target` single clamped dispatch point** | 4,170 | 3 lines — motors 2+3 were DESTROYED; the don't-hand-roll-SDK-calls rule must be un-missable |
| 146–151 | Frozen dataclass path (a)/(b) audit | 3,455 | 1-line trigger ("new frozen dataclass → pick path (a)/(b), see brief"); lists → BRIEF(persistence-config) |
| 197..204 | Canonical builders (build_executor / pain_bus / reaction_bus / memory_hub / default_network / bio_stack) | 13,932 combined | ONE collective 3-line CORE entry ("all bio wiring goes through canonical builders; raw construction raises TypeError; pain_bus/agent_id are required kw-only") — full per-builder text → BRIEF(runtime-tools). These six entries are the biggest single compression win in the file |

CORE subtotal after compression: ≈ **4.5–5K chars**.

### 3b. BRIEF assignments (stub = 1-liner w/ guard, in the brief; narrative → ARCHIVE)

| Brief | Entries (L#) | Current chars |
|---|---|---:|
| bio-memory | 142 (tiers), 143 (separate EpisodicMemory), 182 (NAc+EC pair persist), 206–213 (recall touch, pressure promotion, MemoryRecord fields, valence default, spreading_activation, reward-bias clamp, save_cerebellum, per-tick decay), 214 (SCN temporal coupling), 215 (oscillator default-on), 216 (affordance encoder), 196 (record_event canonical intake) | ≈13.5K |
| llm-routing | 154 (byte-stable prompt), 155–157 (peer backend trio — dedupe w/ lessons 79/83), 158 (typed transports), 163 (typed HTTP errors), 164 (raw_proxy_forward reserved), 193 (llm_call_registry + stall threshold) | ≈9.5K |
| embodiment | 165 (direct pain attribution), 166 (B8 delta-attribution), 167–175 (channel-split drive pain, 7.4K), 176 (value-progress motor credit, 3.3K), 177 (side_effects registry), 184 (Reachy WS transport), 217 (signed sensor ranges), 218–223 (drive protocol, acquisition, self_effect, interaction levels, NarrativePhase, deleted trackers), 224 (tick cycle, 5.6K), 225 (normalize_llm_entity_spec) | ≈28K |
| runtime-tools | 144 (bus not direct calls), 160 (WorkerPool owner), 162 (RequestContext), 178 (Tool.cancel), 189 (PerceptSource/ActionSink protocols), 191 (CWD-relative verbs), 195 (dual-format input_schema), 197–204 (builders full text) | ≈12K + builders 14K |
| persistence-config | 159 (Percept wire vs session format), 227–236 (hivemind layer, 8.3K), 237 (config.json writer), 239 (role detection SSoT), 241 (peer.yml migration) | ≈17K |

Narrative → ARCHIVE from this section: ≈ **75–80K** of 112.4K. Note heavy intra-file duplication (peer-backend rules appear as lessons 79/83 AND invariants 155/156/157; drive-pain appears at 166, 167–175, 176, and 224) — condensation lens should merge before splitting.

---

## 4. Environment variable table (23.2K chars; 75 var lines + 30 comment lines)

Session-critical (**CORE**, one line each, ≈1.8K total): provider API keys (collapse 8 lines → 1), `MAXIM_ROLE`, `MAXIM_LLM_ENABLED`, `MAXIM_LLM_PROFILE`, `MAXIM_LLM_N_CTX`, `MAXIM_LOG_FILE`, `MAXIM_BACKEND_TRACE`, `MAXIM_HTTP_TRACE`, `MAXIM_PROVENANCE_VERBOSITY`, `MAXIM_SUBSTRATE_PATH`, `MAXIM_SKIP_REMOTE_PROBE`, `MAXIM_HEARTBEAT`, plus a pointer line: "full table per subsystem → docs/agents/*".

**BRIEF assignments** (one line per var; paragraph rationales → owning module docstring or lesson file, per plan):
- **llm-routing** (~30 vars): probe timeouts/cache, `MAXIM_LLM_CALL_TIMEOUT_S`, proxy family (admission, overhead, keepalive, concurrency, rate, port), cloud family, lane remote config + timeout_s family, drain routing, auto-spawn family, auto-download/data-budget.
- **bio-memory** (~10): `MAXIM_NAC_MIN_CONFIDENCE`, `MAXIM_NAC_REWARD_BIAS_DISABLED`, `MAXIM_EC_TRACE_ACTIVATIONS`, Wire-A/W1 disables, NAc tau, bio trace vars (HIPPO/ATL/NAC/PAIN_CHAIN).
- **embodiment** (~3): `MAXIM_DEEP_EMBODIMENT`, `MAXIM_PLACE_CODE_EXTEROCEPTION` (its 2.5K rationale → lesson/module docstring), `MAXIM_MOTOR_CREDIT_TRACE`.
- **simulation-experiments** (~12): `MAXIM_DETERMINISTIC_SCENE_EMBODIMENT`, `MAXIM_DISABLE_IMAGINATION(_SUBSTRATE_SIGNAL)`, body-state ablation pair, `MAXIM_EXP44_CAPTURE_LOG`, cradle_mother trio, `MAXIM_SUBSTRATE_ACTIONS_PER_TURN`, `MAXIM_CONCEPT_DECOMPOSITION`.
- **runtime-tools** (~9): context-pool family, heartbeat tuning, `MAXIM_LANE_TRACE`, `MAXIM_PEER_LOG_REQUESTS`.

The biggest single-line rationales (L513 cradle_mother 1.5K header, L539 place-code 2.5K, L535 timeout_s 1.0K, L505-509 Exp44 block) are ARCHIVE/docstring material — the var name + one clause suffices in the brief.

---

## 5. Predicted core composition vs budget

| Core block | Est. chars |
|---|---:|
| Header + overview | 0.4K |
| Required checks + guardrails | 2.6K |
| Working principles (narratives trimmed) | 6.5K |
| Running simulations (n_ctx history → archive) | 2.7K |
| Process invariants kept full (review-round, merge-target) | 3.5K |
| Safety rules block (provenance, Reachy ×2, co-locate) | 2.0K |
| Cross-cutting compressed invariants (§2 CORE + §3a) | 5.5K |
| Doctor 5-line summary | 0.5K |
| Key Commands | 3.0K |
| Remote update + Versioning | 1.4K |
| Architecture map (compressed) | 1.2K |
| Routing table (merged Quick-ref: area → files → brief) | 3.5K |
| Env vars (session-critical subset) | 1.8K |
| Testing + Sim reports + API pointer | 2.2K |
| Active initiatives (live-only) | 0.5K |
| **Total (subsystem-local stubs live in briefs)** | **≈37K chars ≈ 9.3K tokens — FITS** |

**Constraint conflict to resolve in the design fold:** the appendix's hard constraint "every `Regression guard:` line survives in CLAUDE.md" is incompatible with the ≤10K-token target unless the ~85 subsystem-local entries are kept as ultra-compact one-liners (~180 chars each ≈ +15K chars → ≈52K total ≈ 13K tokens, OVER budget). Recommendation from the retrieval standpoint: amend the constraint to "every guard line survives in the repo — in the owning `docs/agents/` brief (still lint-checkable) — and CLAUDE.md keeps the routing table + the CORE subset". If the operator insists on all guards in CLAUDE.md, the format must be a grouped table (Rule | Guard) at ~140 chars/row (≈12K) and Working principles must be cut to ~4K to squeak under 40K chars.

**Layer totals over the current 254.5K:** ALWAYS-worthy (post-compression) ≈37K; ON-DEMAND source material ≈120K (synthesizes into ~6 briefs of 8–15K each); ARCHIVE ≈95–100K narrative.

**Retrieval-specific observations for other lenses:**
1. Active initiatives is the stalest block (claims v0.7.0; repo is 1.0.x→1.1) — Truth lens.
2. Intra-file duplication (peer backend ×2, drive-pain ×4 entries, role detection lesson+invariant, mesh ×2) — Condensation lens should merge before extracting.
3. The Quick-reference table is the routing table — don't build a second one; add a "brief" column.
4. The six canonical-builder invariants are the single largest compress-to-one-entry win (~14K → ~0.4K core + brief).
5. The env table's real payload is 30 experiment toggles a normal session never touches — the CORE subset is genuinely ~14 vars.