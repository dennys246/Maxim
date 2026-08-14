# Phase 0 — Lens 2: CONDENSATION report

**Input:** `CLAUDE.md` @ worktree `Maxim-wt-claudemd` (255,759 chars, 669 lines). Full coverage: 36 Lessons-learned entries (L01–L36, lines 35–105) + 64 Architectural-invariants bullets (A01–A64, lines 142–241) = 100 entries, not the ~55-60 the plan estimated.

**Conventions used below:**
- Each entry: original title (abbreviated), source line, slug, ready-to-paste compressed form in a fenced block, notes.
- `KEEP AS-IS` = original already satisfies the ≤4-line contract; no lesson file needed (creating one would only duplicate). Slug still reserved in case the operator wants uniform archiving.
- `KEEP FULL (Exception)` = process invariant with no mechanical guard — the prose IS the enforcement, per the plan's Exception clause.
- Every `Regression guard:` / `Roy experiment:` line is copied byte-exact from the original, including markdown links. Multi-guard entries keep all references.
- A few entries justifiably exceed 4 lines (marked `>4L justified`): the plan's Risks section explicitly requires keeping load-bearing checklist lines (head-frame actuation checklist), and entries with 2–3 guard lines can't fit 4 total lines without truncating guards, which is forbidden.

---

## Part A — Lessons learned (36 entries)

### L01 — Experiment harness MUST assert its sub-sims import its OWN repo (line 35)
Slug: `harness-provenance-assert-repo-interpreter`
```
**[engineering] An experiment harness MUST assert that the `maxim` its sub-sims import is its OWN repo — `git_hash` answers the wrong question.** Any harness that spawns `maxim` calls `scripts/_provenance.py::assert_repo_interpreter(repo_root, binary, exempt=<mock>)` before its first sub-sim (exit 3 on mismatch), and SHOULD stamp `executed_code_provenance(...)` into every run record. Trigger: stale editable `.pth` files / relative `PYTHONPATH` can silently run another checkout while every sub-sim "succeeds" (Exp 42b retraction — a result whose code-under-test cannot be established is not a validation). Operator hygiene: `export PYTHONPATH="$PWD/src"` (absolute) on its OWN line, never chained after a `source` with `&&`. Full history: [docs/lessons/harness-provenance-assert-repo-interpreter.md](docs/lessons/harness-provenance-assert-repo-interpreter.md). Regression guard: [scripts/_provenance.py](scripts/_provenance.py) (shared guard; the three spawning harnesses — `benchmark_exp42_preference.py`, `benchmark_exp41_exploration.py`, `benchmark_cross_session.py` — all call it and exit 3, verified by running each with `PYTHONPATH` unset) + the post-mortem in [docs/experiments/42b_drive_pain_fold_revalidation.md](docs/experiments/42b_drive_pain_fold_revalidation.md).
```
Notes: moves out — the PEP-660 `.pth`-vs-MetaPathFinder analysis, the three failure-mode narrative, the "probably fine" corollary prose (rule kept as one clause). The operator-hygiene line is load-bearing (recurs in memory files) and stays.

### L02 — Don't run the benchmark harness on the leader machine (line 37)
Slug: `no-harness-on-leader-machine`
```
**[engineering] Don't run the benchmark harness on the same machine as the leader** — co-locating requires the harness's children to bypass role-detection, which has no clean entry point; run it from a peer machine. Post-2026-06-05 hardenings (singleton spawn guard in `runtime/llm_server.py::check_existing_llm_server` + harness preflight `assert_subsim_routed_not_local`) make leader-local firing safe. Diagnostic signature: `lane_decisions.jsonl` shows `tier_decisions.large.source: "tier_table"` AND >1 llama-cpp process on the leader. Full history: [docs/lessons/no-harness-on-leader-machine.md](docs/lessons/no-harness-on-leader-machine.md). Regression guard: [tests/unit/test_llm_server.py::TestCheckExistingLlmServer](tests/unit/test_llm_server.py) (spawn/reuse-200/reuse-401/fail-loud-wrong-model) + [tests/behavioral/test_exp37_harness_smoke.py::TestHarnessPreflight](tests/behavioral/test_exp37_harness_smoke.py) (rejects tier_table, accepts env/reused_server).
```
Notes: moves out — the six-step cascade narrative, the "what does NOT work" env-var digression. DUPLICATE: the "never co-locate leader + experiment" rule is restated in the "Running simulations" section (n_ctx bullet, last line) — keep the sims-section one-liner as a cross-ref to this entry.

### L03 — Running-mean centroid drift in cosine-similarity pattern completion (line 39)
Slug: `ec-centroid-drift`
```
**[behavioral] Running-mean centroid drift in cosine-similarity pattern completion.** Detection rule: when validating a new EC modality or substrate-encoding path, ALWAYS measure both isolated (fresh EC per item) and sequential (one EC, all items) — sharp disagreement = drift. Fix rule: frozen-prototype semantics (`ECConfig.frozen_centroid_modalities`) or raise `pattern_complete_threshold` (text: 0.40 → 0.44). NAc coupling: production callers MUST pass `self.ec.config.pattern_complete_threshold` to `NAc.get_threshold_overrides(base_threshold=)`. Sweeps default to 0.05 granularity; 0.01 only at a regression boundary. Full history: [docs/lessons/ec-centroid-drift.md](docs/lessons/ec-centroid-drift.md). Regression guards: [tests/unit/test_ec_centroid_drift_fix.py](tests/unit/test_ec_centroid_drift_fix.py) (4 tests pinning default + parameterization + fallback + clamp floor), [tests/unit/test_roy_5_cosine_localization.py::test_h1c_lower_bound_tracks_ec_default](tests/unit/test_roy_5_cosine_localization.py) (Roy-5 H1C boundary tracks EC default). Roy experiment: [docs/experiments/27_ec_drift_phase_4_behavioral.md](docs/experiments/27_ec_drift_phase_4_behavioral.md) + [docs/experiments/22_roy_5a.md](docs/experiments/22_roy_5a.md) (H1C boundary).
```
Notes: >4L justified (three distinct rules + dual guard/Roy lines, all mandatory). Moves out — the 19-of-20 diagnostic narrative, Phase-3.5 widening math, Roy-2c behavioral verdict prose.

### L04 — Key-embedded values produce structurally-degenerate statistics (line 41)
Slug: `key-embedded-degenerate-statistics`
```
**[engineering] Key-embedded values produce structurally-degenerate statistics.** Before adding a statistic field to an entity, list the entity's key fields and confirm the statistic varies over them; if the measured dimension is part of the key, the accumulator belongs on the parent aggregation, not the keyed entity (canonical: reward variance moved from `CausalLink` — keyed on valence-embedding `outcome_signature` — up to `NAc._event_outcome_welford`). Full history: [docs/lessons/key-embedded-degenerate-statistics.md](docs/lessons/key-embedded-degenerate-statistics.md). Regression guard: [src/maxim/decisions/nac.py](src/maxim/decisions/nac.py) — variance accumulator lives on `NAc._event_outcome_welford` (parent aggregation), not on keyed `CausalLink`; class docstring documents the move.
```
Notes: moves out — generalisation examples (bandit arms etc.), cross-link to `_context_similarity` lesson (goes in the file body).

### L05 — Push silent-no-op invariants into types, not helpers (line 43)
Slug: `silent-noop-invariants-into-types`
```
**[engineering] Push silent-no-op invariants into types, not helpers.** Count silent failures, not loud ones: one silent-failure miss in a critical path → consider structural enforcement; three silent-failure misses in any path → no longer a question, push the invariant DOWN into the type/constructor signature so forgetting becomes a `TypeError`, not a silent no-op. Canonical example: `build_executor(pain_bus=...)` required keyword-only. Full history: [docs/lessons/silent-noop-invariants-into-types.md](docs/lessons/silent-noop-invariants-into-types.md). Regression guard: [src/maxim/runtime/bootstrap.py::build_executor](src/maxim/runtime/bootstrap.py) — required keyword-only `pain_bus=` parameter is the canonical example; signature enforces the rule structurally so forgetting becomes a `TypeError`, not a silent no-op.
```
Notes: this is the parent principle of the six canonical-builder invariants (A37/A38/A40–A43) — the lesson file should cross-link them.

### L06 — Auto-save must not run under the hippocampus RWLock write block (line 45)
Slug: `autosave-outside-rwlock-write-block`
```
**[engineering] Auto-save must not run under the hippocampus RWLock write block — read-lock-under-write self-deadlocks.** Before calling anything inside a held write block on a non-reentrant RWLock, audit the callee chain for ANY lock acquisition (save → dump → `read()`); a lock-taking persistence call belongs in the public wrapper after release (NOTE tombstones forbid re-adding it at the old sites). Corollary: a conftest fixture that globally disables a default-on config flag is a CI blind-spot signal. Full history: [docs/lessons/autosave-outside-rwlock-write-block.md](docs/lessons/autosave-outside-rwlock-write-block.md). Regression guard: [tests/integration/test_persistent_agent_campaign.py](tests/integration/test_persistent_agent_campaign.py) (`test_sleep_with_autosave_does_not_deadlock` — sleep runs in a thread with a bounded join, so a regression fails fast instead of hanging the suite).
```
Notes: moves out — the PR #428 incident narrative and reachability analysis.

### L07 — Mutable globals + module extraction (line 47)
Slug: `mutable-globals-module-extraction`
KEEP AS-IS — already 3 lines, guard is "pattern-lesson, no test enforces" (no mechanical guard → borderline Exception, but nothing to compress).

### L08 — Per-agent stash dicts for multi-agent state (line 49)
Slug: `per-agent-stash-dicts`
```
**[engineering] Per-agent stash dicts (not module-level globals) for multi-agent state.** Any per-agent runtime stash MUST be a `dict[agent_id, value]` from day one, with `agent_id: str` required keyword-only + rejected-if-empty (`_check_agent_id`) at every entry point; producer and consumer MUST agree on one canonical key (`memory_hub.agent_id`). Pain-intensity read-modify-write needs an explicit `threading.Lock`, not the GIL. Test discipline: per-agent isolation must be provable via `event_context["agent_id"]` even when two agents share ONE NAc. Full history: [docs/lessons/per-agent-stash-dicts.md](docs/lessons/per-agent-stash-dicts.md). Regression guard: [tests/integration/test_multi_agent_attribution.py](tests/integration/test_multi_agent_attribution.py) (TestSharedNacIsolation) + [src/maxim/runtime/bio_integration.py](src/maxim/runtime/bio_integration.py) (`_check_agent_id` validator rejects empty `agent_id` at every entry point).
```
Notes: moves out — the three failure-mode narrative, CPython 3.11 bytecode-specialisation detail, `record_outcome(agent_id=...)` mirror description.

### L09 — Auth in health probes (line 51)
Slug: `auth-in-health-probes`
KEEP AS-IS — already 3 lines. Rule verbatim ("MUST include the auth header … treat 401 as 'server is up'"). Cross-ref candidate: probe canon entry (L25/A15).

### L10 — NAc class name (line 53)
Slug: `nac-class-name`
KEEP AS-IS — 2 lines, CI-grep guarded. See MERGE table (M7: removed-identifiers roll-up).

### L11 — Lane tier names (line 55)
Slug: `lane-tier-names`
KEEP AS-IS — 2 lines, CI-grep guarded. See MERGE table (M7).

### L12 — Lane = capability tier; placement is a SEPARATE ordered axis (line 57)
Slug: `lane-capability-placement-split`
```
**[engineering] Lane = capability tier (`large`/`medium`/`small`); placement/origin (`LOCAL`/`CLOUD`/`PEER`) is a SEPARATE ordered axis.** Placement rides on `LLMRouter`'s existing `provider_priority`/`_try_provider` failover — NOT a new resolver (a second resolver re-introduces the multi-call hazard the one-HTTP-call invariant kills). Empty `placement == ()` derives from legacy fields byte-identically; non-empty placement is authoritative (CLI edits on top); multi-element placements compile via tail-injection; `MAXIM_MAX_CLOUD_LANES` keys off `placement[0].origin` only; coherence is enforced at the producer boundary, not in `ProviderPlacement.__post_init__`. Full history: [docs/lessons/lane-capability-placement-split.md](docs/lessons/lane-capability-placement-split.md). Regression guard: [tests/unit/test_lane_placement.py](tests/unit/test_lane_placement.py) (`_legacy_classify_oracle` equivalence matrix + explicit-placement dispatch) + [tests/unit/test_lane_placement_config.py](tests/unit/test_lane_placement_config.py) (schema + coherence-at-load) + [tests/unit/test_lane_placement_runtime.py](tests/unit/test_lane_placement_runtime.py) (producer + tail-injection + CLI re-expression).
```
Notes: >4L justified (five interlocking sub-rules, three guards). Moves out — dispatch-table walkthrough, deprecated-alias schedule, hardware-tuning placement note.

### L13 — Startup ordering in cli.py (line 59)
Slug: `leaderproxy-before-normalize-args`
KEEP AS-IS — 2 lines.

### L14 — Dead code accumulates silently (line 61)
Slug: `dead-code-orphan-grep`
KEEP AS-IS — 2 lines; guard is a process invariant (no mechanical guard).

### L15 — Opt-in env vars in hot startup paths need autouse scrubs (line 63)
Slug: `env-var-autouse-scrubs`
```
**[engineering] Opt-in env vars in hot startup paths need autouse scrubs.** Any new `if os.environ.get("MAXIM_FOO"): do_side_effect()` branch reachable from `build_primary_router` MUST be paired in the same commit with an `@pytest.fixture(autouse=True)` env-scrub in tests/conftest.py — a leaked var makes the side effect run for real in every later test (P5: 9-minute pytest hang on a real 1 GB GGUF download). Full history: [docs/lessons/env-var-autouse-scrubs.md](docs/lessons/env-var-autouse-scrubs.md). Regression guard: [tests/conftest.py](tests/conftest.py) — autouse env-scrub fixtures pattern; new env-var branches must add a matching scrub in the same commit.
```
Notes: near-verbatim; only template-fixture naming moves to the file.

### L16 — HTTP call sites must use maxim/utils/http.py (line 65)
Slug: `http-via-utils-http`
```
**[engineering] HTTP call sites must use `maxim/utils/http.py`.** New outbound HTTP calls pick `http.get`/`http.post` (registered endpoint), `http.fetch_url` (arbitrary URL), or `http.download_to_file` (streaming); the `raw_proxy_forward` escape hatch is reserved for `leader_proxy._proxy_request` ONLY — do not use it elsewhere. (Origin: the 2026-04-12 Cloudflare Bot Fight Mode missing-User-Agent incident.) Full history: [docs/lessons/http-via-utils-http.md](docs/lessons/http-via-utils-http.md). Regression guard: CI grep `grep -r "urllib.request.urlopen" src/maxim/ | grep -v "utils/http.py"` must return zero matches; enforced in [.github/workflows/test.yml](.github/workflows/test.yml).
```
Notes: overlaps A23 (`raw_proxy_forward` reserved) — keep both; A23 is the invariant, this is the how-to.

### L17 — Role detection is the first runtime action (line 67)
Slug: `role-detection-first`
```
**[engineering] Role detection is the first runtime action.** `cli.py::main()` calls `runtime/role.py::detect_and_apply_role(raw_argv)` immediately after `configure_logging`, BEFORE subcommand dispatch; downstream code reads `os.environ["MAXIM_ROLE"]` — never re-detects, never calls `detect_role()` a second time, never infers from `peer.yml` existence. Persisted state is split per role (`active_llm_model.{role}.txt`). Full history: [docs/lessons/role-detection-first.md](docs/lessons/role-detection-first.md). Regression guard: [src/maxim/runtime/role.py::detect_and_apply_role](src/maxim/runtime/role.py) + [src/maxim/cli.py::main](src/maxim/cli.py) — call site structurally precedes subcommand dispatch.
```
Notes: MERGE candidate M3 with A62 (`detect_role` single source of truth) — the stale 5-rank decision order listed here is superseded by A62's 7-rank order; the lesson file should carry only A62's ordering (truth-lens overlap).

### L18 — config.json::llm.profile vs active_llm_model.{role}.txt are separate state files (line 69)
Slug: `declarative-vs-runtime-model-state`
```
**[engineering] `config.json::llm.profile` (declarative operator intent) and `active_llm_model.{role}.txt` (runtime what's-actually-loaded) are intentionally separate state files.** `maxim --llm <model>` is a one-shot runtime swap and does NOT update `config.json`; automatic runtime writes to declarative config files are FORBIDDEN (C2 invariant). Symptom of the by-design drift: sub-sim startup RuntimeError "configured X but server is serving Y" — resolve via `maxim config set llm.profile`, restart, or kill the server. Full history: [docs/lessons/declarative-vs-runtime-model-state.md](docs/lessons/declarative-vs-runtime-model-state.md). Regression guard: [src/maxim/runtime/llm_server.py::check_existing_llm_server](src/maxim/runtime/llm_server.py) — the error message includes the concrete `maxim config set llm.profile` command + the resolution branches.
```
Notes: shares the declarative-never-runtime-mutated principle with L33 and A61 — state the principle once in the lesson file, cross-link.

### L19 — BackendError.fix_hint is never user-controllable (line 71)
Slug: `backend-error-fix-hint-static`
```
**[engineering] `BackendError.fix_hint` is never user-controllable.** Format strings are always static (validated identifiers may be interpolated); access patterns are exactly three: `.status`, `.response`, `.fix_hint` — do NOT add `raw_body` or any parallel attribute (the router bridge counts on the shape matching `HTTPError`). Import `INFERENCE_BROKEN_BACKOFF_S`, don't duplicate it. Full history: [docs/lessons/backend-error-fix-hint-static.md](docs/lessons/backend-error-fix-hint-static.md). Regression guard: [src/maxim/models/language/types.py](src/maxim/models/language/types.py) — `BackendError` hierarchy with class-level `fix_hint` strings (static format strings, not user-controllable).
```
Notes: near-verbatim already.

### L20 — Subcommand dispatch in cli.py::main bypasses logging setup by default (line 73)
Slug: `configure-logging-before-subcommand-dispatch`
```
**[engineering] Subcommand dispatch in `cli.py::main` bypasses logging setup by default.** Anything depending on early logging setup needs `configure_logging` called at the TOP of `main()` before subcommand dispatch (the later `force=True` call dedupes handlers, so early+late is safe). Corollary: any code that runs early in `main()` and consumes `argv` must explicitly handle subcommand entry paths (`maxim tunnel --llm X` once mis-detected role as `solo`). Full history: [docs/lessons/configure-logging-before-subcommand-dispatch.md](docs/lessons/configure-logging-before-subcommand-dispatch.md). Regression guard: [src/maxim/cli.py::main](src/maxim/cli.py) — `configure_logging` is called at the top of `main()` before subcommand dispatch; co-located code structure enforces ordering.
```

### L21 — Plan review round runs BEFORE PR merge, not after (line 75)
Slug: `review-round-before-merge`
**KEEP FULL (Exception clause).** Process invariant; guard line reads "process invariant — review-round discipline enforced by author + reviewer attention; no automated test enforces". The prose (SCOPE TRIGGER, "the value of the round is a DIFFERENT reader") is the enforcement. Optional light trim: the R1/R2/R3 comparison history and PR #395 narrative could move to `docs/lessons/review-round-before-merge.md` with the rule + SCOPE TRIGGER + practical-form lines kept — but per the Exception this is operator's call, not default.

### L22 — A review round is not complete until fold commits are ON THE MERGE TARGET (line 77)
Slug: `review-fold-on-merge-target`
**KEEP FULL (Exception clause).** Process invariant, no mechanical guard. The three numbered rules + the `gh pr list` corollary are the enforcement. Same optional-trim caveat as L21 (PR #435/#443 narrative → lesson file).

### L23 — _MaximPeerBackend.complete_with_usage() makes EXACTLY one HTTP call (line 79)
Slug: `peer-backend-one-http-call`
```
**[engineering] `_MaximPeerBackend.complete_with_usage()` makes EXACTLY one HTTP call.** Adding a `try: ... except: <call again>` block anywhere in `maxim_peer_backend.py` re-introduces the ~52s fail-slow incident — failover is the router's job (`_try_provider` catches typed `BackendError`s specific-before-general). Per-provider cooldown goes through `LLMRouter._note_provider_overload`/`_set_long_backoff`/`_set_short_backoff`, never inside the backend; long-cooldown branches deliberately skip `_note_provider_failure` — do NOT add it for symmetry. Full history: [docs/lessons/peer-backend-one-http-call.md](docs/lessons/peer-backend-one-http-call.md). Regression guard: CI grep `grep -nE "retry|backoff|gateway" src/maxim/models/language/maxim_peer_backend.py | grep -vE "retry_after_s|retry_timeout_s"` must return zero matches; enforced in [.github/workflows/test.yml](.github/workflows/test.yml).
```
Notes: **DUPLICATE of A14** (invariants bullet, line 155) which already says "See the Plan 3 lesson above". Proposal: keep ONE compressed stub — in Architectural invariants (A14's slot) since it's the contract statement — and delete this Lessons entry; its narrative becomes the shared lesson file. See DUP table D1.

### L24 — Streaming contract difference between peer and cloud backends is intentional (line 81)
Slug: `peer-vs-cloud-streaming-contract`
```
**[engineering] Streaming contract difference between `_MaximPeerBackend` and `_OpenAIBackend` is intentional.** Cloud silently collects partial output mid-stream; peer raises `BackendDown` on ANY mid-stream failure so the router can fail over. Do NOT "fix" the peer backend to match the cloud one — that re-introduces the silent-partial-output bug class Plan 3 eliminated. Full history: [docs/lessons/peer-vs-cloud-streaming-contract.md](docs/lessons/peer-vs-cloud-streaming-contract.md). Regression guards: `test_streaming_mid_stream_malformed_chunk_raises_backend_down` + `test_streaming_connection_error_mid_stream_raises_backend_down` + `test_streaming_empty_content_raises_backend_down`.
```
Notes: near-verbatim; original is already close to contract length.

### L25 — Probe entry point is _MaximPeerBackend.health_check (line 83)
Slug: `probe-entry-point-health-check`
```
**[engineering] Any liveness/readiness probe against a peer URL MUST use `_MaximPeerBackend.for_url(url, api_key=k, model=m).health_check(enable_stage2=...)`** — never the removed shims `probe_llm_server` / `llm_server_responding_at` / `_probe_once`. `for_url` is concurrency-safe via instance-level `_api_key_override` and does NOT mutate `os.environ`; new backend factories use the same instance-attribute pattern — do NOT mutate process-global state from a factory call. Full history: [docs/lessons/probe-entry-point-health-check.md](docs/lessons/probe-entry-point-health-check.md). Regression guard: zero-match CI grep in [.github/workflows/test.yml](.github/workflows/test.yml) ("deprecated probe shims" block) — any `probe_llm_server` / `llm_server_responding_at` reference in `src/maxim/` fails CI.
```
Notes: **DUPLICATE of A15 + A16** (invariant bullets 156, 157 restate both halves verbatim). Proposal: merge A15+A16 into one invariant bullet carrying this compressed form; delete this Lessons entry; narrative (R2.5 env-var race story) → shared lesson file. See DUP table D2.

### L26 — Per-tier lanes.<tier>.timeout_s flows through LaneConfig.remote_timeout_s (line 85)
Slug: `lane-timeout-flow`
```
**[engineering] Per-tier `lanes.<tier>.timeout_s` flows through `LaneConfig.remote_timeout_s` → `providers[key]["timeout_s"]` with no backend wiring changes.** Do NOT change either backend's default timeout to honor a missing config: 300s self-hosted / 60s cloud is the right pair, and config absence means "use the backend default" — not "use 0". Loader raises on invalid values; `apply_lane_env_overrides` silent-ignores as belt-and-suspenders. Full history: [docs/lessons/lane-timeout-flow.md](docs/lessons/lane-timeout-flow.md). Regression guard: [tests/unit/test_config_loader.py::TestLaneTierTimeoutField](tests/unit/test_config_loader.py) (13 tests pinning field validation, JSON parser, env coercion, resolve_setting precedence) + [tests/unit/test_leader_proxy.py::TestLaneTimeoutFieldFlow](tests/unit/test_leader_proxy.py) (4 tests pinning env passthrough into LaneConfig.remote_timeout_s).
```
Notes: MERGE candidate M8 — L26/L27/L28 all stem from llm_timeout_scalability.md; three stubs, ONE shared lesson file with sections.

### L27 — Proxy context-overflow admission gate (line 87)
Slug: `proxy-context-admission-gate`
```
**[engineering] The leader proxy rejects oversize inference requests BEFORE forwarding** (HTTP 413, OpenAI-shaped error envelope) when estimated prompt + max_tokens + overhead exceed the context window. The agent-side `PromptBudgeter` is the PRIMARY mechanism; the gate is the SECONDARY safety net — both read `n_ctx` from the same source. Malformed bodies bypass the gate (upstream returns a cleaner 400). Both backends always populate `max_tokens` (required keyword) — keep it that way. Full history: [docs/lessons/proxy-context-admission-gate.md](docs/lessons/proxy-context-admission-gate.md). Regression guard: [tests/unit/test_leader_proxy.py::TestContextOverheadResolver / TestAdmissionEnableGate / TestInputTokenEstimator / TestAdmissionCheck / TestContextWindowResolver / TestBackendsAlwaysSendMaxTokens](tests/unit/test_leader_proxy.py) — 30+ tests pinning env parsing, gate logic, estimator across body shapes, OpenAI-compatible error envelope, cache thread-safety, and the always-send-max_tokens backend invariant.
```
Notes: moves out — char-ratio estimator math, cache tuple shape, gate-fired triage list (a/b/c). Part of M8.

### L28 — TTFT keepalive emitter writes to self.wfile under a shared lock (line 89)
Slug: `keepalive-wfile-write-lock`
```
**[engineering] The TTFT keepalive emitter and the main chunk writer share `self.wfile`; the emitter MUST acquire `write_lock` around each write** — unsynchronised concurrent writes interleave bytes mid-frame and corrupt the chunked encoding. `stop_event.set()` happens BEFORE acquiring `write_lock` in the chunk loop (reversing the order emits one extra keepalive after the first real chunk). Only fires for `text/event-stream` responses — injecting into buffered JSON would corrupt it. Full history: [docs/lessons/keepalive-wfile-write-lock.md](docs/lessons/keepalive-wfile-write-lock.md). Regression guard: [tests/unit/test_leader_proxy.py](tests/unit/test_leader_proxy.py) — `TestKeepaliveChunkFrameFormat` pins the wire format, `TestKeepaliveEmitter` pins emit cadence + stop semantics + write-error handling + lock release.
```
Notes: moves out — cloudflared idle-timeout motivation, frame format detail, exit-condition enumeration. Part of M8.

### L29 — httpx stream contexts must outlive their consumers (line 91)
Slug: `httpx-stream-ctx-lifetime`
```
**[engineering] Any code that enters an httpx stream context manager manually via `.__enter__()` MUST store a reference to the context manager that lives at least as long as the consumer reading the stream** — GC calls `__exit__()` (closing the stream) the moment the ctx goes out of scope. The `_stream_ctx` field in `StreamingResponse` is load-bearing — do not set it to `None` or remove it. Full history: [docs/lessons/httpx-stream-ctx-lifetime.md](docs/lessons/httpx-stream-ctx-lifetime.md). Regression guard: [src/maxim/utils/http.py](src/maxim/utils/http.py) — `StreamingResponse._stream_ctx` field declaration + `close()` enters the context manager's `__exit__`; structural enforcement via the dataclass shape.
```
Notes: moves out — the 0-chunks-through-Cloudflare incident narrative.

### L30 — NAc._context_similarity denominator is len(ctx1), not the key union (line 93)
Slug: `context-similarity-directional-denominator`
```
**[behavioral] `NAc._context_similarity` denominator is `len(ctx1)`, not the key union — the function is directional** (ctx1 = pending-event/stored-link side; ctx2 = outcome/query side; extra outcome-side keys do NOT dilute the score). If you add a new caller that needs symmetric similarity, build a separate function — do NOT touch `_context_similarity`'s denominator. Full history: [docs/lessons/context-similarity-directional-denominator.md](docs/lessons/context-similarity-directional-denominator.md). Regression guards: `tests/unit/test_nac.py::TestContextSimilarity` (7 tests) + `tests/unit/test_pain_bus.py::TestCreatePainNacSubscriber::test_pain_attributes_to_pending_action_via_context_similarity` + `tests/substrate/test_sem_pain_cascade.py` end-to-end. Roy experiment: [docs/experiments/p2_sem_pain_cascade.md](docs/experiments/p2_sem_pain_cascade.md) (end-to-end cascade validation on rusty_sword fixture).
```
Notes: the "(borderline: …)" tag-hedge parenthetical moves to the lesson file. MERGE candidate M6 with L31 (same incident family, same Roy experiment file).

### L31 — Context-similarity attribution is wrong when a direct lookup key exists (line 101)
Slug: `direct-key-over-context-similarity`
```
**[behavioral] Context similarity is the *fallback* for out-of-band attribution; it is NEVER the right mechanism when you have a direct lookup key.** If a new code path wants `record_outcome_full` + context similarity, ask first whether a direct key (e.g. `(tool_name, invocation_id)`) is available and prefer `record_outcome`. `ToolPainBridge._on_embodiment_pain` guards on `bool(self._pending_tools)` to prevent double-recording while a tool is in flight. Full history: [docs/lessons/direct-key-over-context-similarity.md](docs/lessons/direct-key-over-context-similarity.md). Roy experiment: [docs/experiments/p2_sem_pain_cascade.md](docs/experiments/p2_sem_pain_cascade.md) (validates tool-invoked SEM affordance learning end-to-end via direct-attribution path).
```
Notes: DUPLICATE cluster with A24 (invariant bullet 165 restates the whole mechanism). Proposal: A24 keeps the wiring invariant; this keeps the principle; ONE shared lesson file (`direct-key-over-context-similarity.md`) serves both + L30. See DUP table D3 / MERGE M6.

### L32 — Probe outcome → classification lives in ONE place (line 95)
Slug: `probe-classification-single-source`
```
**[engineering] Probe outcome → classification lives in ONE place: `peer/probe_classify.py::classify_probe_outcome` — and callers do NOT override the returned message.** Want a richer message? Pass a richer `detail` parameter and let the classifier compose it; DO NOT mutate the returned `ProbeClassification` fields. Specific-before-general ordering is load-bearing (`auth_rejected`/`inference_broken` before the generic network-down bucket). `retry_id` and status are orthogonal — producers set a stable `retry_id` regardless of status. Full history: [docs/lessons/probe-classification-single-source.md](docs/lessons/probe-classification-single-source.md). Regression guard: [src/maxim/peer/probe_classify.py::classify_probe_outcome](src/maxim/peer/probe_classify.py) — single source of truth; callers route through this function (verified by code search of `ProbeClassification` usage sites).
```
Notes: moves out — round-1/round-2 review attribution, caller enumeration.

### L33 — mesh.yml parser dialect is FROZEN (line 97)
Slug: `mesh-yml-parser-frozen`
```
**[engineering] `mesh.yml` parser dialect is FROZEN — DO NOT bolt features onto `peer/mesh_config.py::parse_mesh_config`.** If you need quoted strings, YAML anchors, multi-line values, or tab indentation: do not extend this parser — the two escape hatches are (a) switch to TOML + stdlib `tomllib`, or (b) promote PyYAML to core dep; either is an architectural decision, not a drive-by patch. Full history: [docs/lessons/mesh-yml-parser-frozen.md](docs/lessons/mesh-yml-parser-frozen.md). Regression guard: [src/maxim/peer/mesh_config.py::parse_mesh_config](src/maxim/peer/mesh_config.py) + corresponding unit tests in [tests/unit/test_mesh_config.py](tests/unit/test_mesh_config.py).
```
Notes: moves out — dialect feature list, the `#`-comment edge case (goes in file; the parser's own tests pin it).

### L34 — mesh.yml is declarative; ~/.maxim/util/ is mutable state (line 99)
Slug: `mesh-declarative-vs-util-state`
```
**[engineering] `mesh.yml` is declarative (operator-edited, runtime-read-only); every mutable peer/mesh state lives in `~/.maxim/util/{name}.{role}.txt|json`.** Only `peer/mesh_setup.py` may mutate `mesh.yml`, routing through `write_mesh_config` (CI allow-list); the default answer for automatic/runtime/admin-API mutations is ALWAYS `~/.maxim/util/`. New mutable mesh surface checklist: (1) `~/.maxim/util/`, (2) role-scoped filename via `MAXIM_ROLE`, (3) `filelock.FileLock` around the full RMW, (4) `atomic_write_secret` for credential-bearing files / `atomic_write_text` otherwise — never pass `preserve_mode=True` to `atomic_write_text` directly, (5) validate against `mesh.yml`'s node set at write, surface orphans as warnings at read. Wanting a `mesh.yml::<mutable-field>` = a two-source-of-truth reconciliation problem — stop. Full history: [docs/lessons/mesh-declarative-vs-util-state.md](docs/lessons/mesh-declarative-vs-util-state.md). Regression guard: CI grep allow-list in [.github/workflows/test.yml](.github/workflows/test.yml) restricts callers of `write_mesh_config` to `mesh_setup.py` + its test file; new callers fail CI.
```
Notes: >4L justified — the 5-step checklist is the load-bearing part (plan Risks: keep checklists). Moves out — verb enumeration history, review-round cross-confirmation credits, lock-unification aside, `_role()` DrainError detail.

### L35 — PainBus is the rich-context carrier; ReactionBus is the typed isolation surface (line 103)
Slug: `painbus-rich-reactionbus-typed`
```
**[engineering] PainBus (rich free-form `PainSignal.context`) and ReactionBus (typed, isolation-ruled) coexist by design.** Do NOT route rich cause-description through `ReactionContext.bindings` (violates the isolation docstring) and do NOT re-add a `ContextVar`-based signal stash as a back channel (re-entrancy hazard). PainBus's finer `(entity, failure_mode)` refractory gate is intentional — without it two entities firing embodiment pain in one tick collapse into one dispatch. Full history: [docs/lessons/painbus-rich-reactionbus-typed.md](docs/lessons/painbus-rich-reactionbus-typed.md). Regression guard: [src/maxim/proprioception/pain_bus.py](src/maxim/proprioception/pain_bus.py) (module docstring + dual-dispatch implementation) + [tests/unit/test_pain_bus.py](tests/unit/test_pain_bus.py).
```

### L36 — utils/optional_deps.py is the canonical optional-dependency surface (line 105)
Slug: `optional-deps-canonical-surface`
```
**[engineering] `utils/optional_deps.py` is the canonical optional-dependency surface — do NOT add any new `try: import X except ImportError:` variant (silent pass / non-deduped warning / swallowed return-None) anywhere in `src/maxim/`.** Pick by intent: `require_optional_dependency` (explicitly-requested feature → raises typed `OptionalDependencyError`), `optional_dependency_available` (capability probe → bool, never logs), `warn_optional_fallback` (real fallback exists → ONE deduped WARNING). Add new extras in `EXTRA_FOR_IMPORT`, not at call sites; `OptionalDependencyError` access patterns are exactly `.import_name`/`.extra`/`.fix_hint` — no parallel attributes. Full history: [docs/lessons/optional-deps-canonical-surface.md](docs/lessons/optional-deps-canonical-surface.md). Regression guard: [tests/unit/test_optional_deps.py](tests/unit/test_optional_deps.py) — covers `require_optional_dependency` raise/return, `optional_dependency_available` bool, `warn_optional_fallback` dedup, `OptionalDependencyError` subclass shape, and LLM-router reraise behaviour.
```
Notes: moves out — the $0-sim incident narrative.

---

## Part B — Architectural invariants (64 entries)

Short bullets that already satisfy the contract are listed as KEEP AS-IS with slug only. Compressed forms are given for every entry that needs one.

### A01 — Memory tier progression is one-way (line 142) — slug `memory-tier-progression` — KEEP AS-IS (2 lines).
### A02 — Separate EpisodicMemory instances (line 143) — slug `separate-episodic-instances` — KEEP AS-IS (2 lines).
### A03 — Tool results flow through the agent bus (line 144) — slug `tool-results-via-agent-bus` — KEEP AS-IS (2 lines; guard is convention/reviewer attention — no mechanical guard, borderline Exception).
### A04 — Persistence uses atomic_write_json (line 145) — slug `atomic-write-json` — KEEP AS-IS (2 lines).

### A05 — Frozen dataclasses are forward-compat-audited (CC3) (line 146)
Slug: `frozen-dataclass-forward-compat`
```
**[engineering] Every `@dataclass(frozen=True)` that persists or crosses a wire MUST declare its path in the class docstring before merge:** (a) escape-hatch — defaults on all fields + `extra: dict = field(default_factory=dict, hash=False, compare=False)` (JSON-serializable values only; `__post_init__` rejects extra keys colliding with declared fields), or (b) `SHAPE-FROZEN at 1.0 (CC3)` marker with the rejection rationale. Typed exception hierarchies follow the same spirit via explicit keyword-only `__init__`s — no `**kwargs`/`extra`. Runtime-ephemeral config dataclasses are out of scope. Class rosters: see lesson file. Full history: [docs/lessons/frozen-dataclass-forward-compat.md](docs/lessons/frozen-dataclass-forward-compat.md). Regression guard: CC3 audit list + the `SHAPE-FROZEN at 1.0 (CC3)` docstring marker on each frozen-without-extra dataclass; new frozen dataclasses must pick path (a) or (b) before merge.
```
Notes: the two long class rosters (5 escape-hatch + 22 shape-frozen names) move to the lesson file — they are lookup data, not rule text.

### A06 — _format_version is the persistence-format contract (line 152)
Slug: `format-version-contract`
```
**[engineering] Every persisted JSON file carries `"_format_version": "1.0"` at root.** Writers wrap via `with_format_version(payload)` + `atomic_write_json`; loaders call `check_format_version(data, "<file_type>", log=logger)` (missing → `"0.x"` sentinel + one warning per file_type; old files still load). Envelope `schema_version` (int) and `_format_version` (string) coexist by design; do NOT bump the tombstoned legacy payload-layer `"version": "1.0"` strings. List-rooted files are wrapped in a thin dict so the field has a slot. Full history: [docs/lessons/format-version-contract.md](docs/lessons/format-version-contract.md). Regression guard: [tests/integration/test_persistence_compat.py](tests/integration/test_persistence_compat.py).
```

### A07 — LLM access goes through models/language/router.py (line 153)
Slug: `llm-router-only-access`
```
**[engineering] LLM access goes through `models/language/router.py`; concrete backends are not imported outside `models/language/`.** Sanctioned exceptions: `_MaximPeerBackend.for_url(...).health_check()` as the cross-module PROBE surface (inference DISPATCH stays router-only) and `bench/recovery_time.py` (deliberate benchmark bypass). Adding a backend type = one line in `runtime/lane_backends.BACKEND_CLASSES` + one `_classify_backend` branch — no router edit. Full history: [docs/lessons/llm-router-only-access.md](docs/lessons/llm-router-only-access.md). Regression guard: [src/maxim/runtime/lane_backends.py::BACKEND_CLASSES](src/maxim/runtime/lane_backends.py) (single dispatch table) + CI grep in [.github/workflows/test.yml](.github/workflows/test.yml) ("1.0 guard promotion" step) blocking backend imports outside `models/language/` (allow-listed: `agents/llm_agent.py` — grandfathered pre-router standalone agent, migration to the router is tracked follow-up work; `agents/exec_agent.py` — imports the `PROPOSED_GOAL_TOOL` constant, not a backend class; `_MaximPeerBackend` imports are sanctioned via the probe-entry-point invariant).
```

### A08 — System prompt MUST be byte-stable across turns (line 154)
Slug: `system-prompt-byte-stable`
```
**[engineering] The system prompt MUST be byte-stable across turns within a session** (prompt-caching prerequisite). A section is in the stable prefix iff it calls `budgeter.add(..., cacheable=True)` — reserved for genuinely session/phase-stable content; per-turn-dynamic content (drive states, datetime, memories, bio annotations, observations, conversation) MUST NOT be tagged cacheable. When adding a new prompt section, default to dynamic; tag `cacheable=True` only after confirming it cannot vary within a session. Full history: [docs/lessons/system-prompt-byte-stable.md](docs/lessons/system-prompt-byte-stable.md). Regression guard: [tests/integration/test_prompt_caching.py::test_system_prompt_byte_stable_across_turns](tests/integration/test_prompt_caching.py) + [tests/integration/test_prompt_builder_audit.py](tests/integration/test_prompt_builder_audit.py) (segment-level byte-stability) + [tests/unit/test_prompt_budgeter.py::TestBuildSegmented](tests/unit/test_prompt_budgeter.py).
```
Notes: moves out — segment-delimiter wire format, `_STABLE_BUDGET_FRACTION`, measured-win figure.

### A09 — One HTTP call (line 155) — slug `peer-backend-one-http-call` — **DUPLICATE D1.** Keeper slot. Compressed form = L23's block (identical rule). Guard line here ("CI grep above, enforced in [.github/workflows/test.yml](.github/workflows/test.yml)") is subsumed by L23's fuller guard line.
### A10 — health_check canonical probe entry point (line 156) — slug `probe-entry-point-health-check` — **DUPLICATE D2.** Merge with A11 into one bullet = L25's block.
### A11 — for_url concurrency-safe via _api_key_override (line 157) — slug `probe-entry-point-health-check` — **DUPLICATE D2** (same fact as L25 sentence 2). Its guard line ("[src/maxim/models/language/maxim_peer_backend.py::_MaximPeerBackend.for_url](src/maxim/models/language/maxim_peer_backend.py) — `_api_key_override` is an instance attribute, not a module-level variable; pre-merge review caught the env-var race and pinned the fix") must be APPENDED to the merged bullet's guards.

### A12 — Peer transports are typed per purpose, not generic (line 158)
Slug: `typed-peer-transports`
```
**[engineering] Peer transports are typed per purpose, not generic — do NOT extend `_MaximPeerBackend` to carry non-LLM payloads.** Each future transport (perception, substrate bundle) lands in its own file next to its domain and copies the playbook: single-purpose backend, typed exception hierarchy with `.fix_hint`, no internal retry, `for_url(api_key=k)` factory with instance-level override, `health_check()` probe entry point — plus its own CI grep on the one-call pattern. Full history: [docs/lessons/typed-peer-transports.md](docs/lessons/typed-peer-transports.md). Regression guard: the existing CI grep on `maxim_peer_backend.py` (enforcing the one-HTTP-call rule via `grep -nE "retry|backoff|gateway"`) is the template the next transport copies; any new transport class adds its own CI grep allow-list entry on the same pattern so cross-transport leakage stays out.
```
Notes: moves out — per-transport landing-path enumeration, failover-semantics rationale.

### A13 — Percept wire format is distinct from session-persistence format (line 159)
Slug: `percept-wire-format`
```
**[engineering] `Percept.to_wire_dict/from_wire_dict` (cross-process transport) and `to_dict/from_dict` (leader session persistence) are separate contracts, versioned independently under `_format_version`.** `embedding` and `substrate_node_id` are NEVER on the wire — the leader owns the substrate; the peer ships raw observations (leader-derived `salience`/`novelty` + `maxim_runtime` also excluded). Network-backed `PerceptSource`s are non-blocking-by-design: `next_percept()` MUST NOT make a synchronous network call; a blocking variant requires a parallel Protocol. Full history: [docs/lessons/percept-wire-format.md](docs/lessons/percept-wire-format.md). Regression guard: [tests/unit/test_percept_wire_format.py](tests/unit/test_percept_wire_format.py) pins the wire-dict whitelist + the substrate-fields-never-on-wire invariant + the explicit wire/session divergence + [tests/unit/test_percept_source_protocol.py](tests/unit/test_percept_source_protocol.py) pins `isinstance(stub_RemotePerceptSource, PerceptSource)` so a Protocol-shape change that breaks the 1.1 adapter fails here.
```

### A14 — WorkerPool owned by LLMWorker (line 160) — slug `workerpool-owned-by-llmworker` — KEEP AS-IS (2 lines).
### A15 — No NEW silent exception swallows (line 161) — slug `no-new-silent-swallows` — KEEP AS-IS (3 lines; note: guard text "a diff-scoped CI lint … is the tracked follow-up" may be stale — commit 30b31e2f "no-silent-swallows lock" suggests it shipped; truth-lens item).
### A16 — RequestContext + ContextVar multi-agent contract (line 162) — slug `requestcontext-contract` — KEEP AS-IS (4 lines).

### A17 — HTTP errors are typed, not string-matched (line 163)
Slug: `typed-http-errors`
```
**[engineering] HTTP errors are typed, not string-matched.** Callers branch on `HTTPError`/`BackendError` subclasses (`.status`/`.fix_hint`), never parse messages; do NOT introduce a parallel exception type — extend the hierarchy in `types.py` + add the matching `_try_provider` router branch in specific-before-general order (order violation = the R2c auth-misclassified-as-inference_broken bug class). `INFERENCE_BROKEN_BACKOFF_S = 15.0` is the single source of truth linking router backoff to probe cache TTL — import, don't duplicate. Full history: [docs/lessons/typed-http-errors.md](docs/lessons/typed-http-errors.md). Regression guard: [src/maxim/utils/http.py](src/maxim/utils/http.py) + [src/maxim/models/language/types.py](src/maxim/models/language/types.py) (typed hierarchies) + `LLMRouter._try_provider` specific-before-general catch order in [src/maxim/models/language/router.py](src/maxim/models/language/router.py).
```
Notes: DUP D6 — the specific-before-general + INFERENCE_BROKEN_BACKOFF_S facts also appear in L19/L23/L32; canonical home is here, others cross-ref.

### A18 — raw_proxy_forward reserved for leader_proxy (line 164) — slug `raw-proxy-forward-reserved` — KEEP AS-IS (3 lines; cross-refs L29's `_stream_ctx` lesson).

### A19 — Tool-invoked embodiment pain attributes directly (line 165)
Slug: `direct-key-over-context-similarity`
```
**[engineering] Tool-invoked embodiment pain attributes directly, not via context similarity.** `runtime/executor.py` routes non-empty `side_effects["embodiment_failures"]` to `ToolPainBridge.record_tool_embodiment_failure(tool_name, invocation_id, failures)` → `nac.record_outcome` with a direct event_id — NO context similarity. `_on_embodiment_pain` early-returns while any tool is in flight (guard assumes one Executor + one bridge per agent; if a future refactor shares a bridge across concurrent tools, narrow the guard by matching `signal.context["entity"]`). Full history: [docs/lessons/direct-key-over-context-similarity.md](docs/lessons/direct-key-over-context-similarity.md). Regression guard: [tests/substrate/test_sem_pain_cascade.py](tests/substrate/test_sem_pain_cascade.py) (end-to-end direct-attribution path validated).
```
Notes: DUPLICATE D3 — same mechanism as L31; shares one lesson file.

### A20 — Drive-spec embodiment failures are delta-attributed (B8) (line 166)
Slug: `b8-delta-attribution`
```
**[engineering] A tool is blamed only for drive-spec breaches its OWN effect caused (B8 delta-attribution).** `ModulatorAffordanceTool.execute` includes a drive-spec failure in `side_effects["embodiment_failures"]` only when the affordance's own delta is intrinsically harmful to that sensor; standard (non-drive) `failure_mode`s pass through unchanged; on any filter error fall back to the UNFILTERED events, never to empty. B8 is NOT redundant and must NOT be removed — being state-independent, it is the only mechanism separating causer from bystander at sensor saturation (latching this channel inverts a repeat harmful affordance to POSITIVE credit). Full history: [docs/lessons/b8-delta-attribution.md](docs/lessons/b8-delta-attribution.md). Regression guard: [tests/unit/test_substrate_primary_scene_harm.py](tests/unit/test_substrate_primary_scene_harm.py) (B8 helper unit tests + execute-level causing-vs-bystander on the chilled body) + [tests/substrate/test_sem_pain_cascade.py](tests/substrate/test_sem_pain_cascade.py) + [tests/unit/test_self_effect.py](tests/unit/test_self_effect.py) (causing-tool path preserved).
```
Notes: the mid-body inversion-guard citation (`test_execute_delta_attribution_causing_vs_bystander_on_chilled_body`) is preserved inside the trailing guard set via the lesson file; MERGE M5 shares a lesson file with A21.

### A21 — Drive-pain emission is CHANNEL-SPLIT (line 167)
Slug: `drive-pain-channel-split`
```
**[engineering] Drive-pain emission is CHANNEL-SPLIT: the direct `FailureEvent` channel stays state-based (one event per call while breached — do NOT latch it; latching starves B8); the PainBus channel is severity-latched** (fires on band entry + material re-injury past `_BREACH_DEEPEN_FRACTION`, clears with hysteresis). The latch lives on `Entity.drive_breach_severity` (`__slots__`, never serialized), NOT the `Embodiment` wrapper; an unreadable sensor clears the latch; motivation rides the drive VALUE and is untouched by both channels. Note opposite polarities: `FailureMode.persistent` = keep firing; `drive_breach_severity` = stay quiet unless deepening — declaring `persistent: true` on a drive does nothing. Full history: [docs/lessons/drive-pain-channel-split.md](docs/lessons/drive-pain-channel-split.md). Regression guard: [tests/unit/test_transition_drive_pain.py](tests/unit/test_transition_drive_pain.py) (19 tests — channel-1 stays state-based incl. at saturation, channel-2 entry/deepen/marginal-creep/recovery/band-edge-jitter/entropic-satisfaction-threshold, entity-owned latch across ephemeral wrappers + same-name siblings + reparent, unreadable-sensor clear, non-serialization, motivation preserved) + [tests/unit/test_substrate_primary_scene_harm.py::test_execute_delta_attribution_causing_vs_bystander_on_chilled_body](tests/unit/test_substrate_primary_scene_harm.py) (end-to-end repeat-causer arm on the real Exp 42 fixtures).
```
Notes: >4L justified (two-channel contract + latch-location + polarity trap are each load-bearing). Moves out — the SCN-oscillator "CONCERN VOID" sub-bullet (belongs with A36's SCN-gap lesson file), the 42b revalidation results sub-bullet (belongs in the experiment doc, already linked). MERGE M5.

### A22 — Substrate-primary action credit = SIGN of drive value-progress (line 176)
Slug: `motor-credit-value-progress`
```
**[engineering] Substrate-primary action selection is credited by the SIGN of drive VALUE-PROGRESS toward comfort (`drive_potential_diff`), not tool-execution success.** `drive_comfort_progress` is value-based, NOT pain-based (a pain-based reward books zero for sub-threshold entropic relief and starves warmth/feeding of credit); the consumer takes the SIGN (±1), never the magnitude (a graded magnitude loses the argmax to a flat +1); `target_effect` is NOT scored; exactly-0 net progress → tool-success fallback. The collateral-harm gate lives in the PRODUCER (suppresses to `None` on unaccounted-sensor failures); a `not embodiment_failed` gate at the consumer would break orient. Full history: [docs/lessons/motor-credit-value-progress.md](docs/lessons/motor-credit-value-progress.md). Regression guard: [tests/unit/test_drive_pain_helper.py](tests/unit/test_drive_pain_helper.py) (`drive_comfort_progress` value-based + entropic-subthreshold-positive) + [tests/unit/test_motor_credit_emission.py](tests/unit/test_motor_credit_emission.py) (emission incl. sub-threshold entropic, collateral gate on real `warmth_alpha_harm`/`_safe`, orient-preserved) + [tests/unit/test_tool_dispatch.py::TestClusterRewardMotorCredit](tests/unit/test_tool_dispatch.py) (signed consumer + producer-owns-harm-gate) + [scripts/orient_substrate/2_full_path_probe.py](scripts/orient_substrate/2_full_path_probe.py) (full-path policy) + Exp 42 behavioral re-validation.
```
Notes: >4L justified (4 guard refs + the sign-not-magnitude and producer-owns-gate rules are all failure-mode-preventing). Moves out — the #405 regression narrative and step-function math.

### A23 — ToolOutput.side_effects is the typed bio-signal channel (line 177)
Slug: `tool-side-effects-registry`
```
**[engineering] `ToolOutput.side_effects` is the typed channel for bio-pipeline signals; the append-only key registry lives at [docs/user/tool_side_effects.md](docs/user/tool_side_effects.md).** Add new keys via PR that updates the registry table AND wires the consumer (or marks the key `informational`). Do NOT hijack `metadata` (caller-facing extras) or `output` (main result) — collapsing them silently couples the tools layer to bio concepts. Full history: [docs/lessons/tool-side-effects-registry.md](docs/lessons/tool-side-effects-registry.md). Regression guard: [src/maxim/tools/base.py::ToolOutput](src/maxim/tools/base.py) (dataclass + docstring) + [docs/user/tool_side_effects.md](docs/user/tool_side_effects.md) (key registry).
```
Notes: current-key enumeration moves out (registry doc is the authority anyway).

### A24 — Tool.cancel() is a non-abstract no-op on the Tool ABC (line 178)
Slug: `tool-cancel-noop`
```
**[engineering] `Tool.cancel()` stays a non-abstract no-op on the Tool ABC — adding `@abstractmethod` post-1.0 breaks every third-party subclass; do not do it.** Overrides are called from a different thread than `execute()`: they must be thread-safe (`threading.Event` is the canonical shape), never raise, and `execute()` clears the event at entry. No 1.0 dispatch path calls `cancel()` (forward-compat for 1.1+ MCP/async-cancel; `Tool.timeout` likewise unenforced). Full history: [docs/lessons/tool-cancel-noop.md](docs/lessons/tool-cancel-noop.md). Regression guard: [tests/unit/test_tool_cancel.py::test_cancel_has_no_caller_in_executor_dispatch](tests/unit/test_tool_cancel.py) — if a future refactor wires `cancel()` into the executor, update that test and document the new caller here.
```

### A25 — Persistence-boundary values use stable_hash, never builtin hash() (line 180)
Slug: `stable-hash-persistence`
```
**[engineering] Values that cross a persistence boundary MUST be hashed with `utils/seeding.py::stable_hash_32` / `stable_hash_64_signed`, never builtin `hash()`** (PYTHONHASHSEED randomization makes persisted hashes permanently unmatchable across processes; a seed PARAMETER routed through `hash()` only looks deterministic). Sum-then-branch-on-sign sites use the SIGNED 64-bit variant. Persisted files carry `hash_scheme: "stable-sha256-v1"`; loaders WARN when absent. A same-process test passes over this entire bug class — the guard MUST be two-process with differing PYTHONHASHSEED. Full history: [docs/lessons/stable-hash-persistence.md](docs/lessons/stable-hash-persistence.md). Regression guard: [tests/unit/test_stable_hash_two_process.py](tests/unit/test_stable_hash_two_process.py) (verified to fail 5/5 against the pre-fix code).
```
Notes: moves out — the five converted-site list, measured similarity numbers, CI-flake attribution.

### A26 — NAc and EC persist as a PAIR; decay-on-load in NAc.load() only (line 182)
Slug: `nac-ec-persist-pair`
```
**[engineering] NAc and EC persist as a PAIR in `build_bio_stack` (`nac.json` + `ec.json`); decay-on-load lives in `NAc.load()` and NEVER in `load_state()`** (biases key on EC node ids — restoring either alone leaves them silently dangling; `load_state` stays byte-faithful for hivemind `nac_merge`). `apply_decay=False` is REQUIRED where wall-clock gaps are not agent-experienced time: the `--resume-sim` restore and read-only observers (`maxim.load.nac`, `maxim.observe`). `NAc.save()` — not `dump()` — stamps `saved_at`. The orchestrator NPC passes `AgentConfig(load_persisted=False)` (write-but-don't-read). Full history: [docs/lessons/nac-ec-persist-pair.md](docs/lessons/nac-ec-persist-pair.md). Regression guard: [tests/unit/test_nac_persistence_decay.py](tests/unit/test_nac_persistence_decay.py) (decay schedules, opt-out, bool-`saved_at`, corrupt-value coercion, recovery reset) + [tests/integration/test_cross_session_persistence.py](tests/integration/test_cross_session_persistence.py) (two-session two-process content round-trip; verified to fail on both no-persistence and a simulated save-only truncating implementation).
```
Notes: moves out — half-life table, bio-claim disclaimer.

### A27 — Reachy Mini transport is WS-era (line 184)
Slug: `reachy-ws-transport`
```
**[engineering] Reachy Mini transport is WS-era (SDK >= 1.5): control `ws://<host>:8000/ws/sdk`, liveness `GET /api/daemon/status`, network DoA `GET /api/state/doa`.** Never re-introduce zenoh `:7447` probes/tunnels, multicast-discovery debugging, or client-side `mini.media.get_DoA()` for off-robot consumers. After any robot reflash, version-match client vs daemon FIRST (`curl .../api/daemon/status` vs `importlib.metadata.version("reachy-mini")`) — skew fails silently; torque is a separate explicit gate (`enable_motors()`); `goto_target(head=...)` takes a 4x4 pose matrix. Full history: [docs/lessons/reachy-ws-transport.md](docs/lessons/reachy-ws-transport.md). Regression guard: [tests/unit/test_reachy_connection_options.py](tests/unit/test_reachy_connection_options.py) (WS-era pins: probe :8000 not :7447, host/port threading, tunnel :8000 retarget, era gate fails loud) + ad-hoc `grep -rn "7447" src/maxim/` must match only historical comments.
```

### A28 — Reachy head pose is WORLD-frame; head=None counter-rotates (line 186)
Slug: `reachy-head-world-frame`
```
**[engineering] Reachy head pose is WORLD-frame and sits ABOVE `body_yaw` — `goto_target(body_yaw=X)` with `head=None` COUNTER-ROTATES the head, so head-mounted sensors (mics, camera) DO NOT turn with the body.** Rule: any code that turns the body and then reads a head-mounted sensor MUST ship an explicit `head=` matrix with the body delta added to head yaw; call `set_automatic_body_yaw(False)` when your loop owns the yaw axis. `head=None` means "re-solve IK against the RETAINED world head target", NOT "leave the head alone". Generalized lesson: when a measurement disagrees with the model, verify the ACTUATION assumption — did the thing you are sensing with actually move? — BEFORE theorizing about the sensor; and read the vendor's docs before reverse-engineering their kinematics. Full history: [docs/lessons/reachy-head-world-frame.md](docs/lessons/reachy-head-world-frame.md). Regression guard: [tests/unit/test_reachy_head_frame.py](tests/unit/test_reachy_head_frame.py) — 5 offline tests against a fake SDK pinning the production `ReachyMiniController` path (body-only ships a head matrix; head world-yaw tracks body; `head_yaw` is body-relative and composed onto the body; a head-only command composes against the body's angle — since the 2026-08-09 F1 fold that is the last COMMANDED body once one exists, with the readback as the one-shot seed (see the retained-axes invariant below); `get_current_pose()` exposes `body_yaw`). **Verified to fail 5/5 on the pre-fix controller.**
```
Notes: >4L justified — the plan's Risks section names this entry's actuation-checklist + vendor-docs lines as must-keep. Moves out — the six-hypothesis debugging saga, measured gains, the guard-citation correction history (the parenthetical about the earlier invalid guards moves to the lesson file).

### A29 — goto_target is the single clamped+locked motion dispatch point (line 187)
Slug: `reachy-motion-dispatch-safety`
```
**[engineering] `ReachyMiniController.goto_target` is the single clamped+locked motion dispatch point; `motion/movement.py::move_head` is the only other sanctioned SDK motion primitive. Do NOT hand-roll `mini.goto_target(...)` / `mini.set_target(...)` / `mini.look_at_image(...)` anywhere else** (motors 2+3 were destroyed by an unclamped pose). Head-yaw clamps apply in the BODY-RELATIVE frame under `_motion_lock`; callers reporting pose outcomes MUST read `last_clamped_axes` or the frame readback, never echo the commanded value. Retained axes fill from the per-axis last-COMMANDED stash (post-clamp), never live readback (positive feedback ratchet); readback seeds an axis exactly once; any raw head mover MUST wire `controller.note_external_head_motion()` or the next command snaps the head to a stale pose. Full history: [docs/lessons/reachy-motion-dispatch-safety.md](docs/lessons/reachy-motion-dispatch-safety.md). Regression guard: [tests/unit/test_reachy_workspace_safety.py](tests/unit/test_reachy_workspace_safety.py) (verified to fail 10/14 on the pre-fold code) + [tests/unit/test_reachy_retained_axes.py](tests/unit/test_reachy_retained_axes.py) (biased-plant fake SDK; core ratchet tests verified to fail on the pre-fix controller) + CI grep in [.github/workflows/test.yml](.github/workflows/test.yml) ("1.0 guard promotion" step) restricting raw `mini.(goto_target|set_target|look_at_image)` to the sanctioned primitives.
```
Notes: >4L justified (hardware-destruction safety rule + F1 ratchet rule). Moves out — numeric limits table, `move_head` frame-composition mechanism, H1/Exp 49 measurement provenance, divergence-WARNING channel detail.

### A30 — PerceptSource and ActionSink are minimal protocols (line 189)
Slug: `percept-source-action-sink-protocols`
```
**[engineering] `PerceptSource` and `ActionSink` are minimal protocols with no orchestrator, narrative-phase, conversational-turn, or tool-registry coupling.** Optional extensions are duck-typed via `hasattr`/`getattr` with sensible defaults — new extensions follow the same pattern. `is_sim_mode` really means "an external adapter is driving percepts"; it gates DN-startup skip, sim logging, LLM-fallback skip, AND lightweight session-end (no full consolidation — revisit when the first non-sim adapter ships). The `isinstance(..., SimulationBridge)` confirmation checks are intentional escape hatches. Full history: [docs/lessons/percept-source-action-sink-protocols.md](docs/lessons/percept-source-action-sink-protocols.md). Regression guard: [src/maxim/simulation/sources.py](src/maxim/simulation/sources.py) (`PerceptSource` Protocol shape) + [src/maxim/runtime/sim_adapter.py](src/maxim/runtime/sim_adapter.py) (`is_sim_mode` flag site).
```

### A31 — CWD-relative path resolution in public API verbs (line 191)
Slug: `cwd-relative-api-verbs`
```
**[engineering] Three public verbs resolve bare/relative paths against CWD: `maxim.benchmark(suite=)`, `maxim.imagine(scenario=)`, `maxim.campaign(path=)` — pass absolute paths from async / pip-install / arbitrary-CWD callers.** Preserved as a developer-checkout convenience; fixing post-1.0 is a non-breaking add. Full history: [docs/lessons/cwd-relative-api-verbs.md](docs/lessons/cwd-relative-api-verbs.md). Regression guard: [src/maxim/api.py](src/maxim/api.py) — three verbs documented per-verb in module docstrings; `benchmark` ConfigurationError surfaces CWD context.
```

### A32 — llm_call_registry + compute_stall_threshold are canonical (line 193)
Slug: `llm-call-registry-stall`
```
**[engineering] `runtime/llm_call_registry.py` is the canonical in-flight LLM call surface; new stall detectors MUST consult `runtime/stall_threshold.py::compute_stall_threshold` rather than defining their own hardcoded thresholds.** The instrumentation site is `LLMRouter._complete_text_locked` (register at dispatch entry, end in try/finally spanning the provider-fallback loop) — wrapping per-backend is INCORRECT (spurious nudges mid-failover). `register_call_start`/`register_call_end` use stack-token LIFO semantics. Known limitation: the orchestrator's `_stall_detector` hardcodes `lane_tier="large"` + reads the env var directly (tracked follow-up). Full history: [docs/lessons/llm-call-registry-stall.md](docs/lessons/llm-call-registry-stall.md). Regression guard: [tests/unit/test_llm_call_registry.py](tests/unit/test_llm_call_registry.py) (15 tests incl. LIFO + nested-dispatch) + [tests/unit/test_stall_threshold.py](tests/unit/test_stall_threshold.py) (20 tests) + [tests/unit/test_stall_detector_with_registry.py](tests/unit/test_stall_detector_with_registry.py) (6 integration tests) + CI grep in [.github/workflows/test.yml](.github/workflows/test.yml) blocking hardcoded `STALL_S=30` assignment literals outside the canonical module (heartbeat.py allowlisted pending Stage 2; pattern is assignment-only — arithmetic expressions and non-`STALL`-prefixed identifiers are intentionally not enforced).
```

### A33 — Tool.input_schema is dual-format; to_json_schema() canonical (line 195)
Slug: `tool-dual-schema`
```
**[engineering] `Tool.input_schema` is dual-format (legacy custom OR JSONSchema 2020-12), never normalized at `__init__`; format-sensitive consumers MUST route through `tool.to_json_schema()` — never iterate `input_schema.items()` directly** (pre-CC9, that pattern silently rendered every `@maxim.tool` tool with empty params behind a swallowed exception). The description-as-value pattern dispatches as required — new tool code migrates to `{NAME: (str, None)}` + JSONSchema descriptions. Contract frozen at 1.0: adding a third format or collapsing one is breaking. Full history: [docs/lessons/tool-dual-schema.md](docs/lessons/tool-dual-schema.md). Regression guard: [tests/unit/test_tool_dual_schema.py::TestAgentLoopParamRendering](tests/unit/test_tool_dual_schema.py) — pins both formats render correctly via `Tool.to_json_schema()`.
```

### A34 — record_event is the canonical temporal intake; do NOT build an SCN bus (line 196)
Slug: `scn-record-event-intake`
```
**[engineering] `TemporalCreditDistributor.record_event` is the canonical intake for EVERY temporally-anchored event — do NOT build an "SCN bus"; a new consumer belongs INSIDE `record_event`.** When adding a producer: use the required fields (`event_id`, `event_type`, `event_signature`, `agent_id`, `temporal_sig` — note `temporal_sig`/`context`, NOT `temporal_signature`/`metadata`), pass the distributor explicitly (required keyword-only, `None` = explicit opt-out), and do NOT wrap the emit in a bare `except Exception` — a swallowed `TypeError` is exactly how the drive path stayed dead unnoticed (only 1 of 6 declared event_type categories has a producer; see [docs/plans/deferred/scn_event_producer_gap.md](docs/plans/deferred/scn_event_producer_gap.md)). Full history: [docs/lessons/scn-record-event-intake.md](docs/lessons/scn-record-event-intake.md). Regression guard: none yet — this documents an ABSENCE by design (missing-is-the-signal); `scripts/check_oscillator_coldstart.py` reports `drive=0` with a pointer to the plan, and the "every declared category has a producer or is marked reserved" test is the tracked follow-up that would make it enforceable.
```
Notes: no-mechanical-guard entry, but the rule compresses cleanly; front-gate rationale prose moves out.

### A35 — build_executor canonical bridge wiring site (line 197)
Slug: `build-executor-canonical`
```
**[engineering] `runtime/bootstrap.py::build_executor` is the canonical bridge wiring site; `pain_bus` is REQUIRED keyword-only (explicit `None` = opt-out).** Bridges cannot be retrofitted — wrapping executors (`FearGatedExecutor` etc.) happens AFTER `build_executor` returns; the deleted `bootstrap_embodiment_and_pain_bridge` helper must not be re-introduced. Fail-fast invariants check at the top BEFORE construction: `pain_bus` XOR `pain_detector`; `entity_ref` requires `pain_bus` + `component_registry`; any subscription path requires `nac`. Bridge construction gates on `nac is not None`, NOT on `pain_bus`. Full history: [docs/lessons/build-executor-canonical.md](docs/lessons/build-executor-canonical.md). Regression guard: [src/maxim/runtime/bootstrap.py::build_executor](src/maxim/runtime/bootstrap.py) — required keyword-only `pain_bus` parameter; signature enforces explicit decision at every call site.
```
Notes: MERGE M4 (canonical-builders roll-up) — see merge table.

### A36 — build_pain_bus canonical PainBus construction site (line 198)
Slug: `build-pain-bus-canonical`
```
**[engineering] `proprioception/pain_bus.py::build_pain_bus` is the canonical PainBus construction site: `hippocampus` and `nac` are REQUIRED keyword-only (`None` = explicit opt-out); raw `PainBus()` raises `TypeError` (tests pass `_allow_raw=True`).** Auto-subscribes the standard memory + NAc pain learners; takes only learning subjects (no ReactionBus/flags/filters per L4). Tripwire: if `test_subscriber_does_not_link_pending_tool_event` fails, DO NOT relax the assertion — the bridge/subscriber no-double-count property is load-bearing on the context-similarity mismatch, not any guard; open [docs/plans/deferred/pain_bus_bridge_subscriber_unification.md](docs/plans/deferred/pain_bus_bridge_subscriber_unification.md) and ship the deeper fix. Full history: [docs/lessons/build-pain-bus-canonical.md](docs/lessons/build-pain-bus-canonical.md). Regression guard: [src/maxim/proprioception/pain_bus.py::build_pain_bus](src/maxim/proprioception/pain_bus.py) — required keyword-only `hippocampus`/`nac` parameters + [tests/unit/test_pain_bus.py::TestBuildPainBus::test_subscriber_does_not_link_pending_tool_event](tests/unit/test_pain_bus.py) tripwire.
```
Notes: >4L justified — the tripwire instruction is the highest-value line (tells a future session exactly what NOT to do on red). Moves out — DefaultNetwork Wave-2 deferral, api.py F5 note, the full latent-trap analysis.

### A37 — build_default_network canonical (line 200)
Slug: `build-default-network-canonical`
```
**[engineering] `runtime/bootstrap.py::build_default_network` is the canonical DefaultNetwork construction site; `nac` is REQUIRED keyword-only (`None` = explicit opt-out).** `maxim=None` is the headless/sim opt-out (DN still provides pain detection + novelty without motor control); `pain_bus=` injects the already-subscribed bus so DN is a bus CONSUMER, not constructor; `ImportError` → `None` + warning, config/type errors propagate. Full history: [docs/lessons/build-default-network-canonical.md](docs/lessons/build-default-network-canonical.md). Regression guard: [src/maxim/runtime/bootstrap.py::build_default_network](src/maxim/runtime/bootstrap.py) — required keyword-only `nac=` parameter.
```

### A38 — build_reaction_bus canonical (line 201)
Slug: `build-reaction-bus-canonical`
```
**[engineering] `reactions/bus.py::build_reaction_bus` is the canonical ReactionBus construction site; raw `ReactionBus()` raises `TypeError` (tests pass `_allow_raw=True`).** No required learning-subject params (generic typed pub/sub); exists for Wave-3 sequencing (construct before `build_pain_bus(..., reaction_bus=rb)`). `cerebellum_modulator_factory` accepts `reaction_bus=` — pre-audit it silently dropped the parameter, discarding every SEM modulator failure reaction. Full history: [docs/lessons/build-reaction-bus-canonical.md](docs/lessons/build-reaction-bus-canonical.md). Regression guard: [src/maxim/reactions/bus.py::build_reaction_bus](src/maxim/reactions/bus.py) (canonical builder; first production caller is `build_bio_stack`).
```

### A39 — build_memory_hub canonical (line 202)
Slug: `build-memory-hub-canonical`
```
**[engineering] `integration/memory_hub.py::build_memory_hub` is the canonical MemoryHub construction site: four core bio-systems required keyword-only, always calls `.connect()` internally** (raw `MemoryHub()` raises `TypeError`; tests pass `_allow_raw=True`). Motivating bug: two production sites constructed `MemoryHub()` and never called `.connect()` — all three bridges silently `None`. A second `.connect()` later is safe (bridges are stateless at construction; overwrite). Full history: [docs/lessons/build-memory-hub-canonical.md](docs/lessons/build-memory-hub-canonical.md). Regression guard: [src/maxim/integration/memory_hub.py::build_memory_hub](src/maxim/integration/memory_hub.py) (required keyword-only core bio-systems + auto `.connect()`) + [tests/integration/test_memory_hub.py](tests/integration/test_memory_hub.py).
```

### A40 — build_bio_stack canonical (line 204)
Slug: `build-bio-stack-canonical`
```
**[engineering] `runtime/bio_stack.py::build_bio_stack` is the canonical bio-pipeline construction site, composing the four Wave 1+2 builders in dependency order; `agent_id=` is REQUIRED keyword-only** (a defaulted agent_id silently diverged `MemoryHub.agent_id` from every other surface, breaking Wire-A's cross-session read — same silent-no-op pattern, pushed into the type). `persistence_dir` derives sub-paths internally; `pain_bus=` accepts a pre-built bus; `build_memory_hub` carries the same required-`agent_id=` contract. Full history: [docs/lessons/build-bio-stack-canonical.md](docs/lessons/build-bio-stack-canonical.md). Regression guard: [src/maxim/runtime/bio_stack.py::build_bio_stack](src/maxim/runtime/bio_stack.py) (composes Wave 1+2 builders in correct dep order; required `agent_id=` parameter) + [tests/integration/test_multi_agent_attribution.py::TestCreateFullAgentBioStackAgentIdPropagation](tests/integration/test_multi_agent_attribution.py) (pins `config.agent_id` propagation through AgentFactory into the bio-stack's MemoryHub).
```

### A41 — Hippocampus.recall() touch (line 206) — slug `recall-touch` — KEEP AS-IS (2 lines).
### A42 — Pressure-based promotion (line 207) — slug `pressure-based-promotion` — KEEP AS-IS (3 lines).
### A43 — MemoryRecord new fields backward-compatible (line 208) — slug `memory-record-compat-fields` — KEEP AS-IS (2 lines).
### A44 — Episode.valence default 0.0 (line 209) — slug `episode-valence-default` — KEEP AS-IS (2 lines).
### A45 — spreading_activation overloads (line 210) — slug `spreading-activation-overloads` — KEEP AS-IS (2 lines).
### A46 — NAc _reward_bias clamps to [0, max] (line 211) — slug `reward-bias-clamp` — KEEP AS-IS (2 lines).
### A47 — BioStack.save_cerebellum() at session end (line 212) — slug `save-cerebellum-session-end` — KEEP AS-IS (2 lines).
### A48 — NAc per-tick decay wired into agent_loop 8.5 (line 213) — slug `nac-per-tick-decay` — KEEP AS-IS (3 lines).

### A49 — SCN temporal coupling for eligibility traces (line 214)
Slug: `scn-temporal-coupling-eligibility`
```
**[behavioral] SCN temporal coupling for eligibility traces (first SCN-substrate PoC).** When fast-decay eligibility traces expire, `distribute_reward` falls back to temporal-phase similarity via `NAc._temporal_anchors` at `NACConfig.temporal_credit_weight` (default 0.3x). Session-scoped — NOT persisted; cross-session transfer uses persisted `reward_bias`; anchors prune when both trace expired AND older than `temporal_window_seconds`. Full history: [docs/lessons/scn-temporal-coupling-eligibility.md](docs/lessons/scn-temporal-coupling-eligibility.md). Roy experiment: [docs/experiments/temporal_credit_validation.md](docs/experiments/temporal_credit_validation.md) (named-experiment citation pending stricter Roy validation per the borderline note).
```

### A50 — SCN oscillator enabled by default in build_bio_stack (line 215)
Slug: `scn-oscillator-default-on`
```
**[engineering] The SCN oscillator is enabled by default in `build_bio_stack`** (`scn.enable_oscillator()` after construction); anticipatory pre-activation runs via `TemporalCreditDistributor.anticipatory_pre_activate(agent_id)` once per tick before `distribute()`; cold-start guard: <3 observations per event type → 0.0 imminence; `_event_phases` is written only under the distributor's RLock and persists via `scn.dump()`. Full history: [docs/lessons/scn-oscillator-default-on.md](docs/lessons/scn-oscillator-default-on.md). Regression guard: [src/maxim/runtime/bio_stack.py::build_bio_stack](src/maxim/runtime/bio_stack.py) (oscillator enable at construction) + [src/maxim/decisions/temporal_credit.py](src/maxim/decisions/temporal_credit.py) (TemporalCreditDistributor composition).
```

### A51 — Affordance names use a SEPARATE LinguisticEncoder (line 216)
Slug: `affordance-encoder-separate`
```
**[engineering] Affordance names are encoded through a SEPARATE `LinguisticEncoder` from the percept encoder** (shared EC/ATL/NAc backing; affordance side uses `AffordanceDecompositionStrategy`, percept side `SpaCyNounChunkStrategy`). Use the `AFFORDANCE_STRATEGY` singleton for annotation lookups and the shared `_make_aff_encoder()` factory for new affordance-encoder constructions. Full history: [docs/lessons/affordance-encoder-separate.md](docs/lessons/affordance-encoder-separate.md). Regression guard: [src/maxim/similarity/decomposer.py](src/maxim/similarity/decomposer.py) (`AFFORDANCE_STRATEGY` singleton) + [src/maxim/imagination/trigger.py](src/maxim/imagination/trigger.py) (`_make_aff_encoder` factory).
```

### A52 — Signed sensors MUST be encoded WITH their range or they FOLD (line 217)
Slug: `sensor-encoding-range-aware`
```
**[engineering] Signed (`[-1,1]`) sensors MUST be encoded WITH their range through `SensorEncoder`, or they FOLD** (the range-blind map aliases center 0.0 with hard-left −1.0 — left/right azimuth collapse into one EC cluster). The no-range path stays byte-identical to pre-P1; callers needing sign preserved thread `ranges={name:(lo,hi)}`. `_read_drive_ranges` and `_read_drive_states` MUST emit the same drive set (a value with no range silently re-folds), and a malformed YAML range is skipped per-sensor, never raised (a raise silently disables ALL substrate encoding). Ranges must be in the same UNITS as the values. Full history: [docs/lessons/sensor-encoding-range-aware.md](docs/lessons/sensor-encoding-range-aware.md). Regression guard: [tests/unit/test_normalize_value_range_aware.py](tests/unit/test_normalize_value_range_aware.py) (byte-identical legacy incl. the fold, monotonic range-aware, `[0,1]` identity, left/right separation, the two-walk drift guard on real reachy+infant bodies, malformed-range skip) + [scripts/orient_substrate/2_full_path_probe.py](scripts/orient_substrate/2_full_path_probe.py).
```
Notes: moves out — fuzz-verification detail, centroid-drift interaction analysis, behavioral numbers.

### A53 — HomeostaticDriveSpec / EntropicDriveSpec are compartmentalized types (line 218) — slug `drive-spec-types` — KEEP AS-IS (4 lines).
### A54 — Entity acquisition via side_effects (line 219) — slug `entity-acquisition-side-effects` — KEEP AS-IS (4 lines).
### A55 — self_effect writes to agent body sensors (line 220) — slug `self-effect-write-back` — KEEP AS-IS (3 lines).
### A56 — Three interaction levels for entities (line 221) — slug `entity-interaction-levels` — KEEP AS-IS (3 lines).
### A57 — NarrativePhase.act + world_entities (line 222) — slug `narrative-phase-acts` — KEEP AS-IS (3 lines).
### A58 — EnergyReactionBridge/MovementEnergyTracker DELETED (line 223) — slug `energy-bridge-deleted` — KEEP AS-IS (2 lines). See MERGE M7.

### A59 — Embodiment tick cycle lives in Body.evaluate_failures() (line 224)
Slug: `embodiment-tick-cycle`
```
**[engineering] The embodiment tick cycle lives in `Body.evaluate_failures()`** — it applies `tick_vital_drift` automatically from elapsed wall-clock, so drives advance on every path that touches the body. **Do NOT add a second raw `tick_vital_drift(` call site** (double-drift); adding another `evaluate_failures()` CALLER is safe (idempotent w.r.t. elapsed time). LLM-primary ticks per live loop iteration via `tick_embodiment_drift` — a no-op on substrate-primary (which ticks itself in `propose_via_substrate`) and when unembodied. `EmbodimentPerceptSource` is Dormant (CC8 protocol-shape template only). Body-state prompt wiring stays behind `MAXIM_ENABLE_BODY_STATE_PROMPT` (default OFF) pending the pre-registered ablation. Full history: [docs/lessons/embodiment-tick-cycle.md](docs/lessons/embodiment-tick-cycle.md). Regression guard: [tests/unit/test_drive_specs.py::TestEvaluateFailuresAutoDrift](tests/unit/test_drive_specs.py) (wall-clock auto-drift + no-double-drift + first-call baseline) + [tests/unit/test_substrate_primary_scene_harm.py::TestProposeViaSubstrateTick](tests/unit/test_substrate_primary_scene_harm.py) (substrate-primary tick fires before drive read) + [tests/unit/test_substrate_primary_scene_harm.py::TestTickEmbodimentDriftLLMPrimary](tests/unit/test_substrate_primary_scene_harm.py) (llm-primary per-iteration tick applies real drift; no-op on substrate-primary + unembodied; swallows exceptions) + CI grep in [.github/workflows/test.yml](.github/workflows/test.yml) restricting `tick_vital_drift(` call sites to `embodiment/body.py` + [tests/unit/test_percept_source_protocol.py::test_embodiment_percept_source_satisfies_protocol](tests/unit/test_percept_source_protocol.py) (dormant module stays protocol-conformant).
```
Notes: >4L justified by 5 guard refs. Moves out — the 2026-07-14 correction history, cadence-caveat resolution narrative, Exp 44 re-validation note (truth-lens: still-open item, keep in lesson file prominently).

### A60 — LLM-derived entity specs route through normalize_llm_entity_spec (line 225)
Slug: `llm-entity-spec-normalizer`
```
**[engineering] LLM-derived entity specs route through `normalize_llm_entity_spec` before `_parse_entity`** — it backstops LLM forgetfulness by filling `abstract: True` on capability-only modulators (post-C4 hard-error flip, unnormalized specs crash with `ConfigurationError`). Do NOT auto-normalize bundled or user-authored YAMLs — the `ConfigurationError` there is the deliberate user-facing migration signal. New LLM-derived entry points must update the CI allow-list and this invariant in the same commit. Full history: [docs/lessons/llm-entity-spec-normalizer.md](docs/lessons/llm-entity-spec-normalizer.md). Regression guard: CI grep in [.github/workflows/test.yml](.github/workflows/test.yml) ("v1 C4-followup-1 LLM-spec normalizer rule") allow-lists the four non-LLM input source files (`embodiment/spec.py` internal, `embodiment/component_registry.py` bundled, `simulation/campaign_runner.py` user YAML, `simulation/generative_runner.py` curated arc) and rejects any new `_parse_entity(` caller outside them that omits the normalizer — new LLM-derived entry points must update both the allow-list and this invariant in the same commit; new non-LLM bundled/user-YAML sources also update the allow-list to extend the migration-signal coverage.
```

### A61 — src/maxim/hivemind/ is the substrate-sharing layer (line 227)
Slug: `hivemind-substrate-sharing`
```
**[engineering] `src/maxim/hivemind/` is the substrate-sharing layer.** `nac_merge`/`ec_merge` are pure functions (inputs never mutated) requiring keyword-only `left_source=`/`right_source=` routed through `_validate_source` (rejects non-strings, empties, and the reserved `_*` namespace — any new source-taking entry point must route through it). `ec_merge` respects `frozen_centroid_modalities` — do NOT bypass or mutate the default (it must equal `ECConfig.frozen_centroid_modalities`; the equality is test-pinned after a silent divergence shipped drift on `"audio"`). Bundles NEVER include hippocampus episodes, and the `nac.json` slice is content-scrubbed at composition UNCONDITIONALLY (never at capture); every ZIP entry routes through `_safe_join` (ZIP-slip). New merge entry points MUST reserve the `trusted_sources`/`validate_link`/`validate_node` parameter shape for 1.2 poison resistance; new bundle SLICES require the schema bump + migration (additive manifest keys follow the `signer_identity` `.get` precedent). Full history: [docs/lessons/hivemind-substrate-sharing.md](docs/lessons/hivemind-substrate-sharing.md). Regression guard: [tests/unit/test_hivemind_merge.py](tests/unit/test_hivemind_merge.py) (Welford parallel-merge correctness, valence-distinct-stays-separate, sorted-key determinism, frozen-modality preservation, reserved-namespace rejection) + [tests/unit/test_hivemind_identity.py](tests/unit/test_hivemind_identity.py) (short-proper-noun coverage, threshold semantics) + [tests/unit/test_hivemind_bundle.py](tests/unit/test_hivemind_bundle.py) (ZIP-slip rejection, migration-seam, float-precision survival, identity filter, end-to-end round-trip through real NAc + EC instances) + [tests/unit/test_artifact_stamping.py](tests/unit/test_artifact_stamping.py) (16 tests incl. the CLI read-from-payload seam + old-file compat both directions).
```
Notes: >4L justified — this entry currently spans 7 sub-bullets (~7,600 chars); the compressed form keeps every DO/DO-NOT while the scrub field-by-field inventory, encoder-provenance mechanics, identity-heuristic limitations, and CLI-verb notes move to the lesson file. The two per-sub-bullet guard lines (bundle scrub AST guard; artifact stamping) are folded into the single trailing guard set — nothing dropped.

### A62 — ~/.config/maxim/config.json operator layer; config_writer.py canonical writer (line 237)
Slug: `config-json-writer-canonical`
```
**[engineering] `~/.config/maxim/config.json` is the operator-config layer and `runtime/config_writer.py` is its ONLY sanctioned writer** (`write_config`/`mutate_config`/`set_field`: atomic_write_json + with_format_version under a FileLock acquired BEFORE the read). Precedence: CLI > env > config.json > builtin defaults via `resolve_setting`; empty-string env vars are UNSET; shadow logs WARNING, convergence INFO. Declarative config files take operator-explicit one-shot writes only — no automatic runtime / admin-API writes (mesh.yml rule). API key refs accept ONLY file paths or `keyring:` URIs — inline plaintext keys are rejected at load. Full history: [docs/lessons/config-json-writer-canonical.md](docs/lessons/config-json-writer-canonical.md). Regression guard: CI grep in [.github/workflows/test.yml](.github/workflows/test.yml) ("config_unification.md C2 + C6 invariants (IM2 fold)") allow-lists callers of `write_config` / `mutate_config` / `set_field` — new callers fail CI; tests in [tests/unit/test_config_writer.py](tests/unit/test_config_writer.py) + [tests/unit/test_config_cli.py](tests/unit/test_config_cli.py) + [tests/unit/test_config_loader.py](tests/unit/test_config_loader.py) pin every fold from the pre-implementation review.
```

### A63 — runtime/role.py::detect_role is the single source of truth (line 239)
Slug: `detect-role-single-source`
```
**[engineering] `runtime/role.py::detect_role` is the single source of truth for role detection** — never add a second detector (the pre-C3 pair silently disagreed and regressed the Mac Mini on day one). Seven-rank order: `MAXIM_ROLE` env → `config.json::role` → `mesh.yml` (peer) → cloudflared config (leader) → `peer.yml` (peer) → `--llm` local + no peer config (solo) → default leader. `leader_mode.py::detect_role` is a thin back-compat wrapper (legacy `client` term drops in 1.2); `ConfigurationError` from config.json surfaces as WARNING, never silently corrupts rank order. Full history: [docs/lessons/detect-role-single-source.md](docs/lessons/detect-role-single-source.md). Regression guard: [tests/unit/test_role_unification.py](tests/unit/test_role_unification.py) (the eight-cell matrix pinning every Mac Mini failure mode + the ConfigurationError surface) + [tests/unit/test_role_detection.py](tests/unit/test_role_detection.py) (the legacy seven-rank tests updated for C3 ordering) + [tests/unit/test_leader_mode.py](tests/unit/test_leader_mode.py) (wrapper translation contract).
```
Notes: MERGE M3 — absorbs L17's stale 5-rank ordering (this entry's 7-rank order is the current truth).

### A64 — _maybe_migrate_from_peer_yml auto-migration (line 241)
Slug: `peer-yml-auto-migration`
```
**[engineering] `_maybe_migrate_from_peer_yml` writes peer.yml → config.json on first startup iff config.json absent + peer.yml present + cloudflared config ABSENT** (the absent-cloudflared condition protects the leader-with-stale-peer.yml case — never auto-flip a leader to peer). peer.yml is NEVER deleted; the API key goes to `~/.config/maxim/api_key` via `atomic_write_secret` under `umask(0o077)`; `_migration_attempted` is set only AFTER a successful write (transient OSError retries next load). `_apply_lane_config_to_env` is idempotent via `_lane_env_applied` — without the guard a second call re-attributes source config→env and breaks doctor attribution. Full history: [docs/lessons/peer-yml-auto-migration.md](docs/lessons/peer-yml-auto-migration.md). Regression guard: [tests/unit/test_lane_routing_via_config.py](tests/unit/test_lane_routing_via_config.py) (migration shim + idempotency + retry-on-transient-failure + self-hosted classification + peer.yml fallback).
```

---

## DUPLICATES — content retold across entries

| # | Content | Where it appears | Proposal (keeper / cross-ref) |
|---|---|---|---|
| D1 | `_MaximPeerBackend` one-HTTP-call rule | L23 (lessons, line 79, full story) + A09 (invariant, line 155, short restate "See the Plan 3 lesson above") | **Keep ONE compressed stub in Architectural invariants** (the contract home); delete the Lessons entry; both feed `docs/lessons/peer-backend-one-http-call.md`. Net saving: one full entry. |
| D2 | Probe entry point `for_url(...).health_check()` + `_api_key_override` concurrency safety | L25 (line 83, full story incl. env-var race) + A10 (line 156) + A11 (line 157) | **Merge A10+A11 into one invariant bullet** carrying L25's compressed form (append A11's guard line to the merged guards); delete the Lessons entry. Lesson file: `probe-entry-point-health-check.md`. Net saving: two entries. |
| D3 | Direct-lookup-key-over-context-similarity mechanism (`record_tool_embodiment_failure`, `_pending_tools` guard) | L31 (line 101, principle + incident) + A19 (line 165, wiring restated nearly verbatim incl. the same guard-narrowing note) | Keep **A19** as the wiring invariant and **L31** as the one-line principle stub; ONE shared lesson file `direct-key-over-context-similarity.md`. Alternatively fold L31 entirely into A19 (safe — A19 contains every imperative). |
| D4 | Context-similarity denominator directionality | L30 (line 93, canonical) + retold inside L31/A19 (the 0/1=0.0 arithmetic) + inside A36 (line 198, the "latent trap" paragraph re-derives it) | L30 is the canonical home. L31/A19/A36 stubs reference it implicitly via shared lesson files; the arithmetic retellings move out. |
| D5 | "Declarative config files are NEVER mutated by runtime operations" | L18 (line 69), L34 (line 99), A62 (line 237) — three restatements of the C2 principle | Keep the principle sentence in **A62** (config layer home) and in L34's mesh context; L18's stub cites it as "(C2 invariant)" without re-deriving. Lesson files cross-link. |
| D6 | Specific-before-general typed-exception catch order | L23 (line 79), L32 (line 95), A17 (line 163) | Canonical home **A17**; L23/L32 keep only their local application clause (already reflected in the stubs above). |
| D7 | `INFERENCE_BROKEN_BACKOFF_S` single-source | L19 (line 71) + A17 (line 163) | Keep in **A17**; drop from L19's stub if further trimming is needed (currently kept in both at one clause each). |
| D8 | "Never co-locate leader + harness/experiment on one box" | L02 (line 37) + Running-simulations section (n_ctx bullet, final sentence) + MEMORY.md | Keep L02 as the entry; the sims-section sentence becomes "(see the harness-co-location lesson)". |
| D9 | `_stream_ctx` keeps the httpx stream alive | L29 (line 91) + A18 (line 164, restates the field's purpose) | Keep L29; A18's restatement compresses to a cross-ref clause (already ≤3 lines, low priority). |
| D10 | Singleton spawn guard / `check_existing_llm_server` behavior | L02 (line 37, Update paragraph) + L18 (line 69, symptom/resolution) | Keep the mechanism in L02's lesson file; L18 keeps only the symptom+resolution (as compressed above). |
| D11 | `_allow_raw=True` raw-construction TypeError pattern | A36, A38, A39 (each restates the C6 hard-error flip + test-site counts) | State once in the M4 merged builders entry (or in each stub as one clause, as drafted); the C6 history lives once in a shared lesson file. |
| D12 | Push-into-types principle retold as motivation | L05 (principle) + A35/A36/A40 each re-explain "same silent-no-op pattern" | Keep the principle only in L05; builder stubs drop the re-explanations (done in drafts above). |
| D13 | Exp 42 safe-vs-harm inversion consequence | A20 (B8) + A21 (channel-split) both narrate the latched-direct-channel inversion | Shared lesson file (M5); each stub keeps one clause. |
| D14 | Wire-A / annotation env-var parser lineage ("mirrors MAXIM_DISABLE_CLUSTER_BIAS_ANNOTATION's parser") | Env-var table, ~8 entries | Out of Part A/B scope but flagged for Stage 2: state the canonical parser once above the table; per-var comments shrink to one line. |

## MERGE candidates — one compressed stub + one lesson file

| # | Entries | Proposal |
|---|---|---|
| M1 | L23 + A09 | Single invariant stub (D1). Lesson file `peer-backend-one-http-call.md`. |
| M2 | L25 + A10 + A11 | Single invariant stub (D2). Lesson file `probe-entry-point-health-check.md`. |
| M3 | L17 + A63 | Single role-detection stub under Architectural invariants: A63's form + L17's "first runtime action / read env, never re-detect" clause (one added line). Lesson file `detect-role-single-source.md` (absorbs L17's narrative; L17's outdated 5-rank order is dropped as superseded — truth-lens confirm). |
| M4 | A35, A36, A37, A38, A39, A40 (six canonical builders) | Optionally one **"Canonical builders"** invariant with a 6-row table (builder → required kwargs → guard) + per-builder special clauses (A36's tripwire, A39's auto-connect, A40's agent_id). Individual guard lines ALL retained in the table rows. If the operator prefers 1:1 stubs (drafted above), still share ONE lesson file `biosystem-unification-builders.md` with six sections — the six current entries repeat the same C6/_allow_raw/silent-no-op boilerplate (~9,900 chars total → ~2,500 as a table). L05 stays separate as the governing principle. |
| M5 | A20 + A21 | Two stubs (different imperatives: don't-remove-B8 vs don't-latch-channel-1), ONE lesson file `drive-pain-attribution.md` holding the shared Exp 42 inversion narrative, the SCOPE history, and the SCN-concern-void note (cross-linked to `scn-record-event-intake.md`). |
| M6 | L30 + L31 (+ A19 wiring) | Both [behavioral], same Roy experiment doc (`p2_sem_pain_cascade.md`), same incident family. Merge L30+L31 into one "context-similarity attribution" entry with the two rules (directional denominator; direct-key-first); A19 remains the executor-wiring invariant. One lesson file serves all three. |
| M7 | L10 + L11 + A58 (+ the probe-shim grep already merged via M2) | Optional roll-up: one **"Removed/renamed identifiers"** bullet — `NucleusAccumbens`→`NAc`, `infer/review/record`→`large/medium/small`, `EnergyReactionBridge`/`MovementEnergyTracker` deleted — each with its existing CI-grep guard reference kept. Saves 3 slots; zero information loss (all three are pure grep-guarded name bans). |
| M8 | L26 + L27 + L28 | Three stubs kept (distinct rules), ONE shared lesson file `llm-timeout-scalability.md` with three sections (they share the same plan lineage and the narrative overlaps: n_ctx overflow → gate → keepalive). |
| M9 | L21 + L22 | Both process invariants kept FULL per the Exception clause; if the operator opts to trim, they should share one lesson file `review-round-discipline.md` (L22 is explicitly "the same principle … applied to the merge"). |

## Tallies and expected savings

- 36 lessons + 64 invariants = 100 entries. 27 KEEP AS-IS (already ≤4 lines), 2 KEEP FULL (Exception: L21, L22), 71 compressed.
- 9 entries marked `>4L justified` (L01, L03, L12, L34, A21, A22, A28, A29, A36, A59, A61 — mostly due to multi-guard lines that must be copied unchanged; the guard lines alone often exceed 2 lines).
- Merges (M1–M3, M6, optionally M4/M7) remove 6–13 entries outright with zero rule loss.
- The Lessons+Invariants sections are ~200K of the file's 256K chars; the compressed forms above total ~60K chars — on their own they bring the document to roughly the 10K-token target before the env-table/doctor-section Stage-2 trims.
- Lesson files needed: 62 unique slugs (shared files per M2/M3/M4/M5/M6/M8 reduce the count below one-per-entry).

## Cautions for Stage 1 (mechanical split)

1. **`Regression guards:` (plural) and `Roy experiment:` variants exist** (L03, L24, L30) — the extraction script and the Principle 5 lint must match both spellings; copy exactly, do not normalize.
2. Two entries carry BOTH a guard line and a Roy line (L03, L30) — keep both.
3. L22 (line 77) has **no `[engineering]` tag** in the original — the lint's opener pattern may not match it today; don't "fix" the tag during the split without operator sign-off (it's a process invariant kept full anyway).
4. A11's guard line must not be dropped when merging A10+A11 (D2).
5. Several stubs embed in-body guard citations (A20's inversion test, A36's tripwire test) — these are load-bearing DO-NOT-RELAX instructions, not decoration; they stay in the stub, not just the lesson file.
6. The head-frame entry (A28) and motion-safety entry (A29) are hardware-damage / retraction-class lessons — over-compression risk is highest here; the drafted forms keep every imperative and the plan-mandated checklist lines.

