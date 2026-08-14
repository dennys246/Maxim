# Phase 0 / Lens 1 — Enforcement audit of CLAUDE.md invariants

**Date:** 2026-08-13. **Worktree:** `/Users/dennyschaedig/Scripts/Maxim-wt-claudemd` (branch `chore/claude-md-diet`, HEAD `888b07f8`; CLAUDE.md byte-identical to main checkout, 255,759 chars).

**Method:** every entry tagged `[engineering]` / `[behavioral]` in the "Lessons learned" and "Architectural invariants" sections was enumerated (36 lessons-side entries incl. the untagged review-round corollary, 45 architectural bullets = **81 entries**, with 2 deliberate duplicates). Every cited `Regression guard:` / `Roy experiment:` reference was checked against the worktree with `grep`/`test -f` (batch script: scratchpad `verify_guards.sh`, 150 checks). Verification column shows the decisive check; all commands ran against the worktree root `$WT`, CI = `$WT/.github/workflows/test.yml`.

**Classification key:** MECHANICAL = real test/CI-grep/CI-lint enforces → compress hard. STRUCTURAL = co-located source shape (required kw-only param, frozen dataclass, typed exception, canonical symbol) verified → compress hard. PROSE = guard is convention/process/absent, or cited guard does not actually enforce the stated rule → keep (near-)full text per the plan's Exception. BROKEN = citation does not resolve.

---

## Part 1 — Lessons learned

| # | Entry | Tag | Guard citation | Verified? | Class | Compress hard? |
|---|---|---|---|---|---|---|
| L01 | Experiment harness must assert repo interpreter (Exp 42b) | eng | `scripts/_provenance.py` (+3 harness callers) + `docs/experiments/42b_drive_pain_fold_revalidation.md` | YES — `assert_repo_interpreter` + `executed_code_provenance` defined; all 3 harnesses (`benchmark_exp42_preference.py`, `benchmark_exp41_exploration.py`, `benchmark_cross_session.py`) call it; 42b doc exists. BONUS: `scripts/lint_harness_provenance.py` now runs in CI (test.yml:695) — stronger than the text claims | MECHANICAL | yes |
| L02 | Don't run benchmark harness on leader machine (Exp 37) | eng | `tests/unit/test_llm_server.py::TestCheckExistingLlmServer` + `tests/behavioral/test_exp37_harness_smoke.py::TestHarnessPreflight` | YES — both classes grep-confirmed | MECHANICAL | yes |
| L03 | Running-mean centroid drift in EC pattern completion | behav | `tests/unit/test_ec_centroid_drift_fix.py` + `test_roy_5_cosine_localization.py::test_h1c_lower_bound_tracks_ec_default`; Roy: exp 27 + 22_roy_5a | YES — both tests + both docs exist; `test_nac_threshold_override_base_tracks_ec_default` also found | MECHANICAL | yes |
| L04 | Key-embedded values → degenerate statistics (Wire 1) | eng | `src/maxim/decisions/nac.py` `_event_outcome_welford` on parent aggregation | YES — symbol present in nac.py | STRUCTURAL | yes |
| L05 | Push silent-no-op invariants into types | eng | `bootstrap.py::build_executor` required kw-only `pain_bus=` | YES — signature parsed: `pain_bus` after `*`, no default (TypeError on omission) | STRUCTURAL | yes |
| L06 | Auto-save must not run under hippocampus RWLock write block | eng | `tests/integration/test_persistent_agent_campaign.py::test_sleep_with_autosave_does_not_deadlock` | YES — function grep-confirmed | MECHANICAL | yes |
| L07 | Mutable globals + module extraction | eng | "pattern-lesson, no test enforces; rely on reviewer attention" | N/A — guard admits absence | **PROSE** | **no — keep** |
| L08 | Per-agent stash dicts for multi-agent state | eng | `test_multi_agent_attribution.py` (TestSharedNacIsolation) + `bio_integration.py::_check_agent_id` | YES — both grep-confirmed | MECHANICAL | yes |
| L09 | Auth in health probes | eng | `maxim_peer_backend.py::health_check` sends Authorization | YES — `def health_check` + `Authorization` present | STRUCTURAL | yes |
| L10 | NAc class name (not NucleusAccumbens) | eng | CI grep in test.yml | YES — pattern present in "1.0 guard promotion" step | MECHANICAL | yes |
| L11 | Lane tier names (no infer/review/record) | eng | CI grep on lane_models.py | YES — test.yml:589 `grep -nE "\"(infer\|review\|record)\"..." lane_models.py` | MECHANICAL | yes |
| L12 | Lane = capability tier; placement is separate axis | eng | `test_lane_placement.py` (`_legacy_classify_oracle`) + `test_lane_placement_config.py` + `test_lane_placement_runtime.py` | YES — all three files + oracle symbol exist | MECHANICAL | yes |
| L13 | LeaderProxy starts BEFORE `_normalize_args` | eng | code structure at top of `cli.py::main` | YES — ordering verified in `_main_impl` (LeaderProxy ~L771-805 precedes `_normalize_args` L835). ⚠ minor drift: logic lives in `_main_impl` (L519), `main` (L2258) is a thin error-surfacing wrapper — citation says "main()" | STRUCTURAL | yes |
| L14 | Dead code accumulates silently | eng | "process invariant — periodic grep before publish; no automated test" | N/A — guard admits absence | **PROSE** | **no — keep** |
| L15 | Opt-in env vars in hot startup paths need autouse scrubs | eng | `tests/conftest.py` autouse fixtures pattern | PARTIAL — cited exemplars `_isolate_maxim_llm_profile_env` + `_isolate_maxim_auto_download_env` exist, but "new env-var branches must add a scrub in the same commit" is unenforced convention | **PROSE** (exemplar-backed) | **no — keep rule + trigger; narrative can move** |
| L16 | HTTP call sites must use `utils/http.py` | eng | CI grep `urllib.request.urlopen` | YES — pattern in test.yml | MECHANICAL | yes |
| L17 | Role detection is the first runtime action | eng | `role.py::detect_and_apply_role` + cli.py call site precedes dispatch | YES — symbol exists; call at cli.py:556, after `configure_logging` (546), before subcommand dispatch. Same `_main_impl` naming drift as L13 | STRUCTURAL | yes |
| L18 | `config.json::llm.profile` vs `active_llm_model.{role}.txt` separate | eng | `llm_server.py::check_existing_llm_server` error message includes `maxim config set llm.profile` | YES — literal string present | STRUCTURAL | yes |
| L19 | `BackendError.fix_hint` never user-controllable | eng | `types.py` — class-level static fix_hint | YES — `class BackendError` + `fix_hint` present | STRUCTURAL | yes |
| L20 | Subcommand dispatch bypasses logging setup by default | eng | `configure_logging` at top of `main()` before dispatch | YES — cli.py:546 (in `_main_impl`), before all subcommand short-circuits | STRUCTURAL | yes |
| L21 | Plan review round runs BEFORE PR merge | eng | "process invariant — no automated test enforces" | N/A — guard admits absence | **PROSE** | **no — keep (named in plan's Exception)** |
| L22 | Review round not complete until folds are ON THE MERGE TARGET | (untagged) | "process invariant — no automated test" | N/A — guard admits absence. Note: entry carries no `[engineering]` tag (lint conformance worth checking during split) | **PROSE** | **no — keep (named in plan's Exception)** |
| L23 | `_MaximPeerBackend.complete_with_usage()` = exactly one HTTP call | eng | CI grep `retry\|backoff\|gateway` on maxim_peer_backend.py | YES — test.yml:150 exact pattern with the two allowed exclusions | MECHANICAL | yes |
| L24 | Streaming contract difference peer vs cloud is intentional | eng | 3 named `test_streaming_*_raises_backend_down` tests | YES — all three grep-confirmed under tests/ | MECHANICAL | yes |
| L25 | Probe entry point is `health_check` (shims removed) | eng | zero-match CI grep on `probe_llm_server` / `llm_server_responding_at` | YES — both names in test.yml grep block | MECHANICAL | yes |
| L26 | Per-tier `lanes.<tier>.timeout_s` flow | eng | `test_config_loader.py::TestLaneTierTimeoutField` + `test_leader_proxy.py::TestLaneTimeoutFieldFlow` | YES — both classes exist | MECHANICAL | yes |
| L27 | Proxy context-overflow admission gate | eng | 6 named test classes in test_leader_proxy.py | YES — all 6 (TestContextOverheadResolver, TestAdmissionEnableGate, TestInputTokenEstimator, TestAdmissionCheck, TestContextWindowResolver, TestBackendsAlwaysSendMaxTokens) exist | MECHANICAL | yes |
| L28 | TTFT keepalive emitter writes under shared lock | eng | `TestKeepaliveChunkFrameFormat` + `TestKeepaliveEmitter` | YES — both classes exist | MECHANICAL | yes |
| L29 | httpx stream contexts must outlive consumers | eng | `utils/http.py` `StreamingResponse._stream_ctx` field | YES — `_stream_ctx` present | STRUCTURAL | yes |
| L30 | `NAc._context_similarity` denominator is `len(ctx1)` | behav | `test_nac.py::TestContextSimilarity` + `test_pain_bus.py::test_pain_attributes...` + `tests/substrate/test_sem_pain_cascade.py`; Roy: p2_sem_pain_cascade.md | YES — all four resolve | MECHANICAL | yes |
| L31 | Probe outcome → classification lives in ONE place | eng | `peer/probe_classify.py::classify_probe_outcome` ("verified by code search") | YES symbol exists; ⚠ nothing blocks a future caller from bypassing — enforcement is code-search convention on top of the structural single module | STRUCTURAL (weak) | yes, keep the "don't mutate returned fields" imperative |
| L32 | `mesh.yml` parser dialect is FROZEN | eng | `mesh_config.py::parse_mesh_config` + `tests/unit/test_mesh_config.py` | YES — both exist | MECHANICAL | yes |
| L33 | `mesh.yml` declarative; `~/.maxim/util/` mutable state | eng | CI grep allow-list on `write_mesh_config` callers | YES — pattern in test.yml (line ~253 area) | MECHANICAL | yes (long entry, big win) |
| L34 | Direct lookup key beats context-similarity attribution | behav | Roy: docs/experiments/p2_sem_pain_cascade.md | YES — doc exists; end-to-end also pinned by tests/substrate/test_sem_pain_cascade.py + `record_tool_embodiment_failure` in tool_pain_bridge.py | MECHANICAL | yes |
| L35 | PainBus rich-context carrier; ReactionBus typed isolation | eng | pain_bus.py module docstring + tests/unit/test_pain_bus.py | YES — both exist | STRUCTURAL | yes |
| L36 | `utils/optional_deps.py` canonical optional-dep surface | eng | tests/unit/test_optional_deps.py | YES — file exists. BONUS: CI now also blocks `except ImportError: X = None` fallbacks (test.yml:507-512) — stronger than cited | MECHANICAL | yes |

## Part 2 — Architectural invariants

| # | Entry | Tag | Guard citation | Verified? | Class | Compress hard? |
|---|---|---|---|---|---|---|
| A01 | Memory tier progression one-way | eng | `agents/bus.py` TierTransitionError | YES — symbol present | STRUCTURAL | yes |
| A02 | Hippocampus/NAc/ATL separate EpisodicMemory instances | eng | `memory_hub.py::build_memory_hub` constructor params | YES — builder exists; per-system instance claim is co-located shape (weak but real) | STRUCTURAL | yes |
| A03 | Tool results flow through agent bus | eng | "convention — enforced by reviewer attention" | N/A — guard admits absence | **PROSE** | **no — keep (it's 1 line already)** |
| A04 | Persistence uses `atomic_write_json` | eng | atomic_io.py canonical + ad-hoc `grep os.replace` | ⚠ MIXED — `atomic_write_json` exists, but the cited ad-hoc grep **currently surfaces ~7 hand-rolled `os.replace` sites** in src/maxim (report.py:151, llm_server.py:133, plan_logger.py:171, plan_dashboard.py:173, plotting.py:154, simulation/report.py, plan_document.py) and nothing fails on them. The guard describes a detection command, not enforcement | **PROSE** (guard does not enforce) | **no — keep rule; flag the 7 sites to the truth lens** |
| A05 | Frozen dataclasses forward-compat-audited (CC3) | eng | CC3 audit list + `SHAPE-FROZEN at 1.0 (CC3)` docstring markers; "new frozen dataclasses must pick (a)/(b) before merge" | PARTIAL — markers present in 15 src files; the pick-before-merge gate is review convention | STRUCTURAL (markers) + PROSE (merge gate) | mostly — keep the (a)/(b) decision rule, move class lists |
| A06 | `_format_version` persistence contract | eng | tests/integration/test_persistence_compat.py | YES — file exists | MECHANICAL | yes |
| A07 | LLM access via router; backends not imported outside | eng | `lane_backends.py::BACKEND_CLASSES` + CI grep blocking backend imports | YES — dispatch table exists; test.yml:76 `BACKEND_IMPORT_ALL` grep with documented allow-list (llm_agent.py, exec_agent.py) | MECHANICAL | yes |
| A08 | System prompt byte-stable across turns | eng | `test_prompt_caching.py::test_system_prompt_byte_stable_across_turns` + test_prompt_builder_audit.py + `TestBuildSegmented` | YES — all three resolve | MECHANICAL | yes |
| A09 | One-HTTP-call (duplicate of L23) | eng | same CI grep | YES (dup) | MECHANICAL | yes — merge with L23 |
| A10 | Canonical probe entry point (duplicate of L25) | eng | same CI grep | YES (dup) | MECHANICAL | yes — merge with L25 |
| A11 | `for_url` concurrency-safe via `_api_key_override` | eng | instance attribute in maxim_peer_backend.py | YES — symbol present | STRUCTURAL | yes |
| A12 | Peer transports typed per purpose (1.1 prep) | eng | "the existing CI grep on maxim_peer_backend.py is the template the next transport copies" | PARTIAL — the grep exists but only guards the EXISTING transport; the per-purpose design rule for FUTURE transports has no guard (nothing to guard yet) | **PROSE** (forward-looking design rule) | **no — keep rule + playbook list; narrative can move** |
| A13 | `Percept` wire format distinct from session format | eng | test_percept_wire_format.py + test_percept_source_protocol.py | YES — both exist | MECHANICAL | yes |
| A14 | WorkerPool owned by LLMWorker | eng | llm_worker.py + worker_pool.py; "review attention catches parallel-pool constructions" | PARTIAL — classes exist; ownership enforced only by review | **PROSE** (thin) | borderline — 2 lines already; keep as-is |
| A15 | No NEW silent exception swallows | eng | "ad-hoc grep ...; a diff-scoped CI lint ... is the tracked follow-up" | ⚠ **STALE — GUARD IS NOW STRONGER THAN CITED**: `scripts/lint_no_silent_swallows.py` runs in CI (test.yml:675-686, commit 30b31e2f, zero-total + per-file grandfathered ratchet). Entry text still calls it a follow-up | MECHANICAL (update citation!) | yes — after citation fix |
| A16 | RequestContext + ContextVar multi-agent contract | eng | `http.py::_build_headers` gated by `HTTPEndpoint.internal` | YES — both symbols present | STRUCTURAL | yes |
| A17 | HTTP errors typed, not string-matched | eng | http.py + types.py hierarchies + `_try_provider` order | YES — `class HTTPError`, BackendError, `_try_provider` all present (catch ORDER itself is not test-pinned — convention within a structural frame) | STRUCTURAL | yes |
| A18 | `raw_proxy_forward*` reserved for leader_proxy | eng | CI grep | YES — pattern in test.yml | MECHANICAL | yes |
| A19 | Tool-invoked embodiment pain attributes directly | eng | tests/substrate/test_sem_pain_cascade.py | YES — file exists; `record_tool_embodiment_failure` in tool_pain_bridge.py | MECHANICAL | yes |
| A20 | B8 delta-attribution (causer vs bystander) | eng | test_substrate_primary_scene_harm.py incl. `test_execute_delta_attribution_causing_vs_bystander_on_chilled_body` + test_sem_pain_cascade.py + test_self_effect.py | YES — all resolve | MECHANICAL | yes (5.6K chars → big win) |
| A21 | Drive-pain channel-split (state-based vs severity-latched) | eng | test_transition_drive_pain.py (19 tests — count verified: exactly 19) + the A20 test | YES | MECHANICAL | yes (longest entry in file) |
| A22 | Motor credit = SIGN of `drive_potential_diff` | eng | test_drive_pain_helper.py + test_motor_credit_emission.py + `TestClusterRewardMotorCredit` + scripts/orient_substrate/2_full_path_probe.py | YES — all four resolve | MECHANICAL | yes |
| A23 | `ToolOutput.side_effects` typed channel + registry | eng | tools/base.py + docs/user/tool_side_effects.md | YES — both exist | STRUCTURAL | yes |
| A24 | `Tool.cancel()` non-abstract no-op | eng | `test_tool_cancel.py::test_cancel_has_no_caller_in_executor_dispatch` | YES — function exists | MECHANICAL | yes |
| A25 | Persistence-crossing values use stable_hash | eng | test_stable_hash_two_process.py | YES — file exists (5 tests) | MECHANICAL | yes |
| A26 | NAc+EC persist as a PAIR; decay-on-load in `NAc.load()` | eng | test_nac_persistence_decay.py + test_cross_session_persistence.py | YES — both exist | MECHANICAL | yes |
| A27 | Reachy transport is WS-era | eng | test_reachy_connection_options.py + ad-hoc `grep 7447` | YES — test file exists; 7447 appears only in 2 historical comments (grep verified) | MECHANICAL | yes |
| A28 | Reachy head pose is WORLD-frame | eng | test_reachy_head_frame.py ("5 offline tests") | YES — file exists; ⚠ now contains **7** tests, text says 5 (harmless drift) | MECHANICAL | yes (huge entry; keep the one-line actuation-checklist rule per plan's risk note) |
| A29 | `goto_target` single clamped dispatch + retained-axes stash | eng | test_reachy_workspace_safety.py + test_reachy_retained_axes.py + CI grep `mini\.(goto_target\|set_target\|look_at_image)` | YES — both test files exist; CI grep at test.yml:627 with the two sanctioned-file exclusions | MECHANICAL | yes |
| A30 | PerceptSource/ActionSink minimal protocols (CC8) | eng | sources.py Protocol + sim_adapter.py flag site | YES — `class PerceptSource` + `is_sim_mode` present; also pinned by test_percept_source_protocol.py | STRUCTURAL | yes |
| A31 | CWD-relative path resolution documented per-verb (CC10) | eng | api.py docstrings; benchmark error surfaces CWD | YES — api.py:1329-1330 ConfigurationError includes `Path.cwd()`; docstrings reference CC10 | STRUCTURAL | yes |
| A32 | llm_call_registry + stall_threshold canonical | eng | 3 test files + CI grep `STALL_S` | YES — all three files + CI pattern exist | MECHANICAL | yes |
| A33 | `Tool.input_schema` dual-format; `to_json_schema()` canonical | eng | `test_tool_dual_schema.py::TestAgentLoopParamRendering` | YES — class exists | MECHANICAL | yes |
| A34 | `record_event` is canonical SCN intake — no SCN bus | eng | "**none yet** — documents an ABSENCE by design"; scripts/check_oscillator_coldstart.py pointer | Script exists and reports drive-gap (grep confirmed); guard deliberately absent | **PROSE** (missing-is-the-signal) | **no — keep rule + producer checklist; incident inventory can move** |
| A35 | `build_executor` canonical bridge wiring (dup of L05) | eng | required kw-only pain_bus | YES (dup) | STRUCTURAL | yes — merge with L05 |
| A36 | `build_pain_bus` canonical PainBus construction | eng | required kw-only hippocampus/nac + tripwire `test_subscriber_does_not_link_pending_tool_event` | YES — builder + tripwire test exist. NOTE: the tripwire's "if this fails DO NOT relax — open deferred plan" instruction is prose that must survive compression (deferred plan file exists) | STRUCTURAL + MECHANICAL | yes, keep the tripwire instruction line |
| A37 | `build_default_network` canonical (nac required kw-only) | eng | bootstrap.py | YES — `def build_default_network` present | STRUCTURAL | yes |
| A38 | `build_reaction_bus` canonical | eng | reactions/bus.py | YES — symbol present | STRUCTURAL | yes |
| A39 | `build_memory_hub` canonical + auto-connect | eng | memory_hub.py + tests/integration/test_memory_hub.py | YES — both exist | STRUCTURAL + MECHANICAL | yes |
| A40 | `build_bio_stack` canonical (agent_id required) | eng | bio_stack.py + `TestCreateFullAgentBioStackAgentIdPropagation` | YES — both exist | STRUCTURAL + MECHANICAL | yes |
| A41 | `Hippocampus.recall()` always touches | eng | hippocampus_retrieval.py RetrievalMixin + test_memory_hub.py | YES — both exist | STRUCTURAL | yes |
| A42 | Promotion is pressure-based | eng | hippocampus_consolidation.py symbols + test_memory_hub.py | YES — `_compute_access_score` + `_PROMOTION_PRESSURE_THRESHOLD` present | STRUCTURAL | yes |
| A43 | MemoryRecord new fields back-compat | eng | memory/types.py + test_persistence_compat.py | YES — both | MECHANICAL | yes |
| A44 | `Episode.valence` defaults 0.0 | eng | episode.py + test_persistence_compat.py | YES — both | MECHANICAL | yes |
| A45 | `spreading_activation` return shapes | eng | @overload signatures in agents/bus.py | YES — overload adjacent to def | STRUCTURAL | yes |
| A46 | NAc `_reward_bias` clamps to [0, max] | eng | nac.py clamp | YES — `max_reward_bias` present (clamp itself unverified line-level; acceptable) | STRUCTURAL | yes |
| A47 | `BioStack.save_cerebellum()` at session end | eng | bio_stack.py + "session-end handlers in integration/memory_hub.py" | ⚠ PARTIAL — method exists and IS called (bio_stack.py:97, agent_factory.py:220, api.py) but **memory_hub.py contains no save_cerebellum call** — the second guard citation is wrong-file | STRUCTURAL (fix citation) | yes after citation fix |
| A48 | NAc per-tick decay wired in agent_loop 8.5 | eng | co-located decay calls | YES — `decay_eligibility` + `decay_reward_biases` in agent_loop.py | STRUCTURAL | yes |
| A49 | SCN temporal coupling for eligibility traces | behav | Roy: docs/experiments/temporal_credit_validation.md ("citation pending stricter Roy validation") | YES — doc exists; entry itself flags the validation as pending | **PROSE-leaning behavioral** | light compress only — the pending-validation caveat must survive |
| A50 | SCN oscillator enabled by default (B2) | eng | bio_stack.py enable_oscillator + temporal_credit.py | YES — both symbols present | STRUCTURAL | yes |
| A51 | Affordance names use separate LinguisticEncoder | eng | decomposer.py AFFORDANCE_STRATEGY + trigger.py `_make_aff_encoder` | YES — both present | STRUCTURAL | yes |
| A52 | Signed sensors MUST encode with range (P1 fold) | eng | test_normalize_value_range_aware.py + orient probe script | YES — both exist | MECHANICAL | yes |
| A53 | Drive protocol compartmentalized frozen types | eng | sem.py SHAPE-FROZEN dataclasses | YES — HomeostaticDriveSpec + SHAPE-FROZEN marker present | STRUCTURAL | yes |
| A54 | Entity acquisition via side_effects | eng | entity_map.py transfer_to_self + executor.py `_handle_entity_acquisition` | YES — both present | STRUCTURAL | yes |
| A55 | `self_effect` writes to body sensors | eng | tool_bridge.py write-back | YES — present | STRUCTURAL | yes |
| A56 | Three interaction levels converge on evaluate_failures | eng | body.py::evaluate_failures single convergence point | YES — present | STRUCTURAL | yes |
| A57 | `NarrativePhase.act` + `world_entities` | eng | arcs.py + generative_runner.py `_activate_phase_entities` | YES — both present | STRUCTURAL | yes |
| A58 | EnergyReactionBridge/MovementEnergyTracker DELETED | eng | CI grep | YES — `EnergyReactionBridge` in test.yml guard-promotion step | MECHANICAL | yes |
| A59 | Embodiment tick cycle in `evaluate_failures()` | eng | TestEvaluateFailuresAutoDrift + TestProposeViaSubstrateTick + TestTickEmbodimentDriftLLMPrimary + CI `tick_vital_drift(` grep + protocol test | YES — all four test classes/fns + CI grep (test.yml:597, restricted to body.py) verified | MECHANICAL | yes (very long entry) |
| A60 | LLM entity specs route through `normalize_llm_entity_spec` | eng | CI grep "v1 C4-followup-1" allow-list | YES — `C4-followup-1` present in test.yml | MECHANICAL | yes |
| A61 | `hivemind/` substrate-sharing layer (4-PR contract) | eng | test_hivemind_merge.py + test_hivemind_identity.py + test_hivemind_bundle.py + test_artifact_stamping.py | YES — all four files + `test_hivemind_frozen_modalities_match_ec_default` + `scrub_nac_state_for_bundle` verified | MECHANICAL | yes (8K+ chars → big win) |
| A62 | config.json operator layer; config_writer canonical | eng | CI grep allow-list ("config_unification.md C2 + C6") + 3 unit-test files | YES — CI pattern + test_config_writer.py + test_config_cli.py + test_config_loader.py all exist | MECHANICAL | yes |
| A63 | `detect_role` single source of truth (C3) | eng | test_role_unification.py + test_role_detection.py + test_leader_mode.py | YES — all three exist | MECHANICAL | yes |
| A64 | `_maybe_migrate_from_peer_yml` auto-migration | eng | test_lane_routing_via_config.py | YES — file exists | MECHANICAL | yes |

**Also verified (untagged but guard-carrying, in "Running simulations" section):** `tests/unit/test_llm_worker_n_ctx_clamp.py` and `tests/unit/test_lane_served_nctx.py` both exist (n_ctx three-leg bug entry). `scripts/lint_claude_md_invariants.py` exists and runs in CI (test.yml:654) — the diet's own lint constraint is live.

---

## Summary counts

| Classification | Count | Notes |
|---|---|---|
| MECHANICAL | 45 | incl. 2 duplicates (A09/A10 of L23/L25); safe to compress hard |
| STRUCTURAL | 26 | incl. 1 duplicate (A35 of L05); safe to compress hard |
| PROSE (keep full / near-full) | 10 | L07, L14, L15, L21, L22, A03, A04, A12, A14, A34 (+ A49 light-compress-only behavioral) |
| BROKEN | 0 | no cited guard failed to resolve |

**≈88% of entries are mechanically or structurally guarded → hard compression is safe for the overwhelming bulk of the document**, exactly as the plan's key insight predicted.

## PROSE list (keep per the Exception)
1. **L07** Mutable globals + module extraction — no guard by admission.
2. **L14** Dead code accumulates — process invariant.
3. **L15** Autouse env scrubs — exemplars exist; "same commit" rule is convention.
4. **L21** Review round before merge — process invariant (explicitly named in the plan's Exception).
5. **L22** Fold commits on merge target — process invariant; **also missing its `[engineering]` tag** (check lint behavior during split).
6. **A03** Tool results via bus — reviewer-attention convention (already 1 line).
7. **A04** atomic_write_json — cited ad-hoc grep is detection-only and **currently matches ~7 hand-rolled `os.replace` sites** (report.py, llm_server.py, plan_logger.py, plan_dashboard.py, plotting.py, simulation/report.py); guard does not enforce the rule.
8. **A12** Typed transports per purpose — forward-looking design rule; grep guards only the existing transport.
9. **A14** WorkerPool ownership — review attention.
10. **A34** record_event canonical SCN intake — "none yet" by design (missing-is-the-signal).

## Drift findings (hand to the Truth lens)
1. **A15 is stale in the favorable direction:** `scripts/lint_no_silent_swallows.py` now runs in CI (test.yml:675, commit 30b31e2f) — the entry still says the CI lint "is the tracked follow-up". Update citation; entry then becomes MECHANICAL and compressible.
2. **L01 is also stronger than cited:** `scripts/lint_harness_provenance.py` runs in CI (test.yml:695) — not mentioned in the entry.
3. **L13/L17/L20 cite `cli.py::main()`** — the ordering logic verified but lives in `_main_impl` (cli.py:519); `main()` (cli.py:2258) is now a thin BackendError-surfacing wrapper.
4. **A28 says test_reachy_head_frame.py has 5 tests; it has 7.**
5. **A47's second citation is wrong-file:** session-end `save_cerebellum` callers are `bio_stack.py:97` (`on_session_end` path) and `agent_factory.py:220`, not `integration/memory_hub.py` (zero matches there).
6. **A21 count claim verified exact** (19 tests in test_transition_drive_pain.py) — no drift.
7. **L22 lacks an `[engineering]` tag** — verify `lint_claude_md_invariants.py` treats it as an invariant before restructuring.

## Compression-safety note for Stage 1
Where an entry embeds a **conditional instruction to future readers** inside otherwise-mechanical prose, the compressed stub must retain that one line even under hard compression: A36's tripwire instruction ("if `test_subscriber_does_not_link_pending_tool_event` fails, do NOT relax — open `docs/plans/deferred/pain_bus_bridge_subscriber_unification.md`"), A28's actuation-verification checklist line (named in the plan's own risk section), A34's producer construction checklist (`temporal_sig`/`context` field names, no bare except), and A05's pick-(a)-or-(b)-before-merge rule.
