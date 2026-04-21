# CLAUDE.md

## Project Overview

Maxim is a bio-inspired cognitive architecture for AI agents. It combines a 5-agent pipeline (Perception, Memory, Exec, Goal, Statistician) with biological memory systems (Hippocampus, ATL, Angular Gyrus, SCN, NAc) and a reactive Default Network. Works headless, in simulation, or connected to a robot.

## When making changes — required checks

Run these before considering any non-trivial task done:

```bash
# Lint + format
ruff check src/ tests/
ruff format src/ tests/

# Tests (fast suite)
python -m pytest tests/ -x -q -m "not slow" --ignore=tests/integration/test_memory_hub.py

# If touching memory/, decisions/, integration/memory_hub.py:
python -m pytest tests/integration/test_memory_hub.py -q
```

Additional guardrails:
- **Test interactive changes with logging.** When touching interactive mode (display, prompts, stdin reader, orchestrator sim loop), capture a session with `MAXIM_LOG_FILE=/tmp/maxim.jsonl maxim --sim "test basic recall" --interactive --sim-max-turns 3` and read the JSONL to verify percepts, tool calls, and followups flow correctly. Check for `ACTION_FOLLOWUP` entries to confirm user responses reach the LLM. Use `MAXIM_BACKEND_TRACE=1` for per-call token/latency data.
- **No band-aid fixes.** If you spot a bug while working on a task, determine whether the fix addresses the root cause or merely hides the symptom. If it's the latter — a special case, a swallowed exception, a flag that toggles around broken behavior, a fix that would need to be repeated elsewhere — stop, describe the root cause and the scope of the proper fix, and ask the user how to proceed. Never silently choose the smaller fix because it's easier.
- Prefer editing existing modules over creating new ones — this codebase favors many small files already
- Don't rename bio-system classes (Hippocampus, ATL, NAc, SCN, EC, AngularGyrus) — names are load-bearing for the mental model
- If you touch provenance, run a sim with `MAXIM_PROVENANCE_VERBOSITY=2` and eyeball the trace
- **Run `mypy` on public API files** after changing api.py, session.py, create.py, load.py, or __init__.py: `mypy src/maxim/__init__.py src/maxim/api.py src/maxim/session.py src/maxim/create.py src/maxim/load.py --ignore-missing-imports`
- **Run `ruff format`** after any changes: `ruff format src/ tests/`

## Lessons learned (bugs that bit us)

**Push silent-no-op invariants into types, not helpers:** The PRIMARY trigger is the *failure mode*, not the count. When a bug class has a SILENT failure mode (no exception, no log, just wrong behavior or missing learning) AND has more than one possible call site, push the invariant DOWN into the type/constructor signature so forgetting becomes a `TypeError`, not a silent no-op. Three repeated instances of the same shape is *strong corroborating evidence* that helper-discipline is too weak — but it's the silent failure mode that makes the rule load-bearing, not the headcount. Loud-failure recurring bugs (raise → log → user sees it) can stay on helper-discipline without harm; silent-failure bugs that are forgotten even ONCE in a security-critical or correctness-critical path should jump straight to structural enforcement. The `build_executor(pain_bus=...)` keyword-only requirement is the canonical example: three identical "forgot to wire `ToolPainBridge`" bugs in `sem_execution_hook.md` Stages 1+2+2c silently skipped NAc tool-outcome learning on three different agent entry points. The bug never raised; the next entry point would have reproduced it. `executor_bootstrap_unification.md` made forgetting impossible at the constructor level — the fourth, fifth, and sixth instances surfaced during the migration audit (api.py headless mode, simulation/tools.py sub-AUT, etc.) became explicit `pain_bus=None` decisions instead of unnoticed silent gaps. **Rule:** count silent failures, not loud ones. One silent-failure miss in a critical path → consider structural enforcement. Three silent-failure misses in any path → no longer a question, push it down.

**Mutable globals + module extraction:** When extracting module-level mutable globals (like `_active_spawner`) into a new file, do NOT re-import them by name (`from new_module import _active_spawner`). Python binds by value at import time — assignments in the importing module diverge from the source. Use module reference instead: `import new_module as _mod; _mod._active_spawner = value`. Functions are safe to re-import (they close over their own module's namespace).

**Auth in health probes:** Any HTTP health check that probes an endpoint behind API key auth MUST include the auth header. The leader's `_probe_upstream_ready()` was silently getting 401s from an auth-gated llama-cpp-server, causing `llm_ready` to be permanently false. Always send auth in probes, and treat 401 as "server is up" (auth-gated but alive).

**NAc class name:** The class is `NAc` (in `decisions/nac.py`), NOT `NucleusAccumbens`. Old code may reference the wrong name — always grep for `NucleusAccumbens` after touching NAc-related code.

**Lane tier names:** The canonical tier names are `"large"`, `"medium"`, `"small"`. The old names `"infer"`, `"review"`, `"record"` have been fully removed. Do not re-introduce them.

**Startup ordering in cli.py:** The LeaderProxy MUST start BEFORE `_normalize_args()` because arg normalization can trigger heavy CUDA imports (5-15s on GPU systems). Peers polling for the proxy during restart will time out if the proxy starts after these imports.

**Dead code accumulates silently:** Before publishing or after major refactors, grep for orphan modules: `.py` files whose basename doesn't appear in any `import` statement. We found 15 dead modules (~8,500 LOC) shipping in the wheel.

**Opt-in env vars in hot startup paths need autouse scrubs:** When you wire a new `if os.environ.get("MAXIM_FOO"): do_side_effect()` branch into anything reachable from `build_primary_router` (auto-spawn, tier detection, ensure_available, ...), pair it in the same commit with a `@pytest.fixture(autouse=True)` env-scrub fixture in [tests/conftest.py](tests/conftest.py). Without it, ANY test that sets the env var (e.g., a `normalize_args` unit test asserting `--auto-download` populates the var) leaks into every later test that constructs the runtime — and the leaked side effect runs for real. P5 cost a 9-minute pytest hang on a real 1 GB GGUF download to `~/.maxim/` before this was caught. The two existing scrubs (`_isolate_maxim_llm_profile_env`, `_isolate_maxim_auto_download_env`) are the template.

**HTTP call sites must use `maxim/utils/http.py`:** Plan 1 R1 consolidated ~11 scattered `urllib.request` call sites into one registry-backed module. The invariant is CI-enforced: `grep -r "urllib.request.urlopen" src/maxim/ | grep -v "utils/http.py"` must return zero matches. The 2026-04-12 Cloudflare Bot Fight Mode incident (commit `8b52cbd`) was a missing `User-Agent` header in one of those call sites — the `_external` endpoint in `utils/http.py` sets it once at registration, so that class of bug is structurally impossible now. When adding a new outbound HTTP call: pick `http.get`/`http.post` (registered endpoint), `http.fetch_url` (arbitrary URL), or `http.download_to_file` (streaming file). The `raw_proxy_forward` escape hatch is reserved for `leader_proxy._proxy_request` ONLY — do not use it elsewhere.

**Role detection is the first runtime action:** Plan 2 R2a made role explicit. `cli.py::main()` calls `runtime/role.py::detect_and_apply_role(raw_argv)` immediately after `configure_logging`, BEFORE subcommand dispatch. It exports `MAXIM_ROLE` + emits `role_detected` as the first structured log event. Downstream code (`runtime/llm_server.py::_model_state_file`) reads `MAXIM_ROLE` from env — never re-detects, never infers from `peer.yml` existence. If you're adding a new feature whose behavior depends on role, read `os.environ["MAXIM_ROLE"]`; never call `detect_role()` a second time. Decision order: env var → mesh.yml → peer.yml → `--llm` flag + no peer config → default leader. Persisted state is split per role (`active_llm_model.{role}.txt`). The call site is co-located with `configure_logging` at the top of `main()` — if you move it downstream you'll re-encounter the subcommand-dispatch logging gap described below.

**`BackendError.fix_hint` is never user-controllable:** Plan 2 R2b added the typed `BackendError` hierarchy in `models/language/types.py` mirroring `utils/http.py::HTTPError`. Every subclass has a class-level `fix_hint`. Subclasses may interpolate validated identifiers (model names, URLs) into hint strings, but the format strings themselves are always static. Prevents log injection via user-controlled exception content. Access patterns are exactly three: `.status`, `.response`, `.fix_hint`. Do NOT add `raw_body` or any parallel attribute — Plan 3's router bridge counts on the shape matching `HTTPError`. The `INFERENCE_BROKEN_BACKOFF_S = 15.0` constant in the same module is the single source of truth linking router backoff to probe cache TTL; import it, don't duplicate.

**Subcommand dispatch in `cli.py::main` bypasses logging setup by default:** `cli.py::main` short-circuits to `run_doctor_subcommand` / `run_peer_connect_subcommand` / `run_tunnel_subcommand` before reaching the sim loop that previously was the only caller of `configure_logging`. Any feature that depends on early logging setup (MAXIM_LOG_FILE JSONL handler, future structured event emission, Plan 2 R2a's `detect_role` log event) needs `configure_logging` called at the TOP of `main()` before subcommand dispatch, not at the sim-loop entry. This was a real bug during Plan 1 R1 — MAXIM_LOG_FILE silently did nothing for `maxim doctor` until commit `c8a07e9` added the early call. The sim loop's later `configure_logging(force=True)` call dedupes JSONL handlers by absolute path, so the early call + late call is safe. **Plan 2 R2 re-encountered this class of bug in a different form:** `_has_local_llm_flag` scanned raw `argv` including subcommand names, so `maxim tunnel --llm X` mis-detected role as `solo`. Any code that runs early in `main()` and consumes `argv` must explicitly handle subcommand entry paths — either skip the scan when `argv[0]` is in `{doctor, peer, tunnel, ...}` or only apply the logic to the sim/agent entry path.

**Plan review round runs BEFORE PR merge, not after** (refined 2026-04-12 after R1 vs R2 comparison, validated again in R3): every completed sub-plan on a `feat/<plan>` branch triggers a pre-merge review round. Spawn two parallel review Claudes (Executor lens + Architecture lens — templates in [docs/plans/llm_path_refinement.md](docs/plans/llm_path_refinement.md) context or the R1/R2/R3 session histories). Fold findings into the same branch via a follow-up commit BEFORE opening/merging the PR. Do NOT merge first and ship a `fix/<plan>-loose-ends` PR after — that pattern works but splits the bisect surface and leaves known-buggy code on main. R1 used the old timing (required PR #91 follow-up); R2 and R3 used the refined pre-merge timing. R3 found 18 issues including 2 critical behavior bugs (`_MaximPeerBackend.for_url` env-var race + `_emit_dispatch_exhausted` bypassing the canonical `_normalize_request_context` shim). Both reviewers caught the env-var race INDEPENDENTLY — the cross-confirmation is a strong signal to trust the finding even when it feels minor. All 12 blocking findings folded into one fix commit before PR #94 opened. Tests caught zero of those bugs — they're correctness issues in input spaces tests don't cover. Review rounds are non-optional, not ceremony. See [feedback_review_before_ship.md](../../.claude/projects/-Users-dennyschaedig-Scripts-Maxim/memory/feedback_review_before_ship.md) for the full evidence + prompt templates.

**`_MaximPeerBackend.complete_with_usage()` makes EXACTLY one HTTP call — the load-bearing invariant for Plan 3:** the whole point of Plan 3 was killing the ~52s fail-slow caused by `_OpenAIBackend`'s internal gateway-retry loop amplified by the per-lane `_inference_lock`. `_MaximPeerBackend` in `models/language/maxim_peer_backend.py` replaces that path for self-hosted peer traffic. It raises typed `BackendError` subclasses on failure and lets the router's provider-fallback loop handle failover. **Adding a `try: ... except: <call again>` block anywhere in this file re-introduces the incident.** CI grep enforces the rule: `grep -nE "retry|backoff|gateway" src/maxim/models/language/maxim_peer_backend.py | grep -vE "retry_after_s|retry_timeout_s"` must return zero matches. The two allowed parameter-name matches are `BackendOverloaded.retry_after_s` (Plan 2 R2b contract) and `health_check.retry_timeout_s` (inherited from the pre-R2.6 probe signature, used for the liveness two-attempt budget — that's a retry *budget*, not a retry loop). If you need per-provider cooldown, use `LLMRouter._note_provider_overload` / `_set_long_backoff` / `_set_short_backoff` — they apply at the router layer, not inside the backend. The router's `_try_provider` catches the typed exceptions in specific-before-general order (`BackendOverloaded` → `BackendAuthFailed` → `BackendModelMissing` → `BackendInferenceBroken` → `BackendTimeout` → `BackendDown` → `BackendError` → `Exception` safety net). Long-cooldown branches (auth 300s, model_missing 60s, inference_broken 15s) do NOT call `_note_provider_failure` — that would overwrite the hard value with the exponential ramp. Do NOT "helpfully" add a `_note_provider_failure` call for symmetry; it's load-bearing that those branches skip it.

**Streaming contract difference between `_MaximPeerBackend` and `_OpenAIBackend` is intentional:** `_OpenAIBackend._stream_response` silently collects partial output when a chunk iteration raises mid-stream (cloud providers' first-token-latency UX expects "got some tokens" > "nothing"). `_MaximPeerBackend._stream_response` raises `BackendDown` on any mid-stream failure (malformed JSON chunk, `HTTPConnectionError` during `iter_lines`, or empty content) so the router can fail over to a different provider. These are different contracts for different backends, not a bug in either. Do NOT "fix" the peer backend to match the cloud one — that re-introduces the class of silent-partial-output bugs Plan 3 was designed to eliminate. Regression guards: `test_streaming_mid_stream_malformed_chunk_raises_backend_down` + `test_streaming_connection_error_mid_stream_raises_backend_down` + `test_streaming_empty_content_raises_backend_down`.

**Probe entry point is `_MaximPeerBackend.health_check` — Plan 3 R2.6:** any liveness or readiness probe against a peer URL MUST use `_MaximPeerBackend.for_url(url, api_key=k, model=m).health_check(enable_stage2=...)`, not the deprecated `runtime.llm_server.probe_llm_server` / `llm_server_responding_at` / `_probe_once`. The three historical functions still exist as DEPRECATED thin compat shims kept to avoid mass-migrating ~35 test mock sites, but CI grep in `.github/workflows/test.yml` allow-lists the 4 existing call sites (in `llm_server.py`, `lane_backends.py`, `doctor/checks.py`, `maxim_peer_backend.py`) — any new match is a CI failure with a migration hint. If you're writing new code that probes a peer URL, go through the backend's method directly. **`_MaximPeerBackend.for_url` is concurrency-safe via instance-level `_api_key_override` — it does NOT mutate `os.environ`.** The R2.5 original shipment wrote the probe key to `os.environ["MAXIM_PEER_PROBE_KEY"]` which races under concurrent probes; the pre-merge review round caught this as a critical finding and the fix stores the key on the returned instance. If you add a new backend factory, use the same instance-attribute pattern — do NOT mutate process-global state from a factory call.

**httpx stream contexts must outlive their consumers:** Calling `ctx = client.stream(...)` then `raw = ctx.__enter__()` opens a live HTTP stream, but Python GC will call `ctx.__exit__()` (which closes the stream) as soon as `ctx` goes out of scope. `raw_proxy_forward_streaming()` in `utils/http.py` originally returned `StreamingResponse(_raw=raw)` without storing `ctx`. The function returning caused `ctx` to fall out of scope; GC closed the stream before `_proxy_request` could call `iter_bytes()` — resulting in `httpx.StreamClosed`, 0 chunks forwarded through Cloudflare, and `JSONDecodeError` on the Mac peer. Every inference call silently returned an empty body. Fix: `StreamingResponse._stream_ctx: Any | None = None` holds the context alive; `close()` calls `_stream_ctx.__exit__(None, None, None)` in cleanup. **Rule: any code that enters an httpx stream context manager manually via `.__enter__()` MUST store a reference to the context manager that lives at least as long as the consumer reading the stream.** The `_stream_ctx` field in `StreamingResponse` is load-bearing — do not set it to `None` or remove it.

**`NAc._context_similarity` denominator is `len(ctx1)`, not the key union** (Substrate P2 Stage 2, 2026-04-13): The function is directional. Every in-file caller passes the pending-event / stored-link side as `ctx1` and the outcome / current-query side as `ctx2`. Semantics: "how much of the pending event's context is matched by the outcome context?" Extra keys on the outcome side do NOT dilute the score — they represent cause-description the pending event was not conditioned on. The pre-Stage-2 denominator was `len(ctx1 | ctx2)` (union), which silently broke every `record_outcome_full` caller that passed a rich outcome context (notably `ToolPainBridge._on_embodiment_pain` with 7 keys). Pending events would match at ~0.29 instead of 1.0, fall below the 0.5 `context_similarity_threshold`, and never link. A Stage 2 draft tried to work around this by passing a slim 2-key context from `create_pain_nac_subscriber` only — the architecture review caught it as a band-aid (per the no-band-aid rule above) because the same bug still existed in the bridge. Fix is in `decisions/nac.py::_context_similarity`. Regression guards: `tests/unit/test_nac.py::TestContextSimilarity` (7 tests) + `tests/unit/test_pain_bus.py::TestCreatePainNacSubscriber::test_pain_attributes_to_pending_action_via_context_similarity` + `tests/substrate/test_sem_pain_cascade.py` end-to-end. If you add a new caller that needs symmetric similarity (treats ctx1 and ctx2 as interchangeable), build a separate function — do NOT touch `_context_similarity`'s denominator.

**Probe outcome → classification lives in ONE place and callers do NOT override the returned message** (Plan 4 C1, 2026-04-14, refined by round 2 review): `maxim.peer.probe_classify.classify_probe_outcome(outcome, detail, latency_ms, url)` is the single source of truth for mapping a `ProbeResult.outcome` literal to a `ProbeClassification(status, message, fix)` frozen dataclass. Every caller that probes a peer URL via `_MaximPeerBackend.for_url(...).health_check()` MUST route its classification through this helper: `peer/mesh_cli.py::_probe_node`, `doctor/checks.py::_probe_mesh_node_to_check`, and any future C2 `--node install` / C3 admin-API dashboard code. Round 1 review caught the same logic already drifting between the two initial call sites (fix-hint wording diverged on the first ship). Round 2 review caught the follow-up footgun: the doctor site was *overriding* the classifier's returned message post-hoc to inject role+url, which silently re-creates the drift the shared helper was supposed to kill. **The rule:** if your caller wants a richer message, pass a richer `detail` parameter through the classifier and let it compose the output — DO NOT mutate the returned `ProbeClassification` fields. `doctor/checks.py::_probe_mesh_node_to_check` passes `detail=f"{node.role}, {node.url}"` for `ok` outcomes so the "reachable" string naturally includes that context. **Specific-before-general ordering is load-bearing (Plan 2 R2c):** `auth_rejected` and `inference_broken` must be classified BEFORE the generic network-down bucket. **`retry_id` and status are orthogonal:** producers set `retry_id` to a stable identity regardless of status; the retry loop in `doctor/cli.py` filters on status to decide whether to re-run. Coupling status and retry_id at the producer was a round 1 fold bug that round 2 review caught. The shared helper lives in its own module (`peer/probe_classify.py`) so callers can use it without pulling the mesh parser layer.

**`mesh.yml` parser dialect is FROZEN** (Plan 4 C1, 2026-04-14): `peer/mesh_config.py::parse_mesh_config` is a deliberately trivial hand-rolled YAML-ish parser — flat top-level `key: value` scalars plus a single nested `nodes:` list of `- name: foo` blocks with indented continuation lines. **DO NOT bolt features onto it.** It rejects tabs, bare `- ` entries, duplicate node names, and strips `#` inline comments ONLY when preceded by whitespace (so `cluster_key: sk-abc#literal` is preserved, not silently truncated — round 2 review E1). If you need quoted strings (URL fragments beyond what the whitespace-# rule handles), YAML anchors, multi-line values, or tab indentation: **do not extend this parser.** The two escape hatches are (a) switch `mesh.yml` to TOML and use stdlib `tomllib`, or (b) promote PyYAML from optional extra to core dep. Either change is a C2/C3 architectural decision, not a drive-by patch. Round 1 review flagged five silent-mis-parse classes the original implementation tolerated; round 2 review flagged a sixth (E1 silent `#` truncation). A seventh finding is highly likely if the dialect grows.

**`mesh.yml` is declarative; `~/.maxim/util/` is mutable state** (Plan 4 C2, 2026-04-14, refined by the C2 pre-merge fold round): the two-layer split is load-bearing across the entire Plan 4 Stage C surface. **Every mutable peer/mesh state — drained nodes, rate limit overrides, per-agent quotas, request traces — lives in `~/.maxim/util/{name}.{role}.txt` (or `{name}.{role}.json`)** with (a) `filelock.FileLock` serialization around the full read-modify-write cycle for cross-platform concurrency safety, and (b) `maxim.utils.atomic_io.atomic_write_secret(path, content)` for any file that contains secrets (API keys, cluster keys, bearer tokens). Non-secret operator-visible state — like drain state — uses plain `atomic_write_text`; the function name is the "this contains secrets" signal. **Never pass `preserve_mode=True` to `atomic_write_text` directly** — use `atomic_write_secret` so the intent is visible at the call site (C2 pre-merge review A3 fold: makes the safe path verbose, not the unsafe path). `mesh.yml` stays strictly read-only from every runtime Maxim code path — operators edit it by hand and restart the daemon to reload topology. **Any verb that mutates `mesh.yml` lives in `src/maxim/peer/mesh_setup.py` and routes through `peer/mesh_config.py::write_mesh_config`** — that single file is the sanctioned writer surface, enforced via the CI grep allow-list in `.github/workflows/test.yml` (only `mesh_setup.py` + its test file may call `write_mesh_config`). Adding a new verb requires updating the allow-list AND **first asking whether the state belongs in `~/.maxim/util/` instead**. The default answer for "automatic" or "runtime" or "admin-API-driven" mutations is always `~/.maxim/util/`; the only writes that earn a place in `mesh_setup.py` are operator-explicit one-shot setup verbs the operator invokes consciously from the CLI. As of Plan 4 C3.2 this is `init-mesh` (synthesize from `peer.yml`), `add-node` (append/replace), and `remove-node` (drop + auto-clear drain). The file rename (`init_mesh.py` → `mesh_setup.py`) happened in C3.2 when the file grew from one verb to three. **Do NOT enumerate verbs in this lesson** — the question to ask is "does this belong in `mesh_setup.py`?", not "is this in the list of three?" The friction modes for the strict allow-list are tracked in the `feedback_strict_grep_caller_allowlist.md` memory file; if you see repeated additions in successive PRs, inline reimplementations of `write_mesh_config`, or this lesson drifting from the allow-list, the rule needs revisiting. **Do NOT add automatic runtime / admin-API write paths to `mesh.yml`** — C3's admin API, cluster key rotation, per-agent rate limiting, and request-trace ring buffer all write to `~/.maxim/util/` (the mutable state layer), never to `mesh.yml`. **C3's admin API writes to the mutable state layer, never to `mesh.yml`.** This Kubernetes-style spec-vs-status split means the two layers serve strictly disjoint purposes and need NO reconciliation contract; `mesh.yml::drain` is NOT a field, drain lives exclusively in `~/.maxim/util/drained_nodes.{role}.txt`. The C2 design pass that landed this rule folded four cross-confirmed findings (role detection timing, read/write race, orphan validation, permission preservation) into a single architectural invariant. When you add a new mutable mesh surface (C3 cluster key rotation, per-agent rate limits, request trace ring buffer): (1) put it in `~/.maxim/util/`, (2) role-scope the filename via `MAXIM_ROLE`, (3) wrap the RMW in a `filelock.FileLock` from the third-party `filelock` package, (4) use `atomic_write_secret` for credential-bearing files and plain `atomic_write_text` for everything else, (5) validate against `mesh.yml`'s node set at write time and surface orphans as warnings (not fails) at read time. **Invariants enforced at the state layer, not the CLI layer** — the self-drain guard and orphan validation in `drain_state.drain_node` are mandatory so future C3 admin-API writers can't bypass them by calling `drain_node` directly (C2 pre-merge review A2 fold). **`_role()` raises `DrainError` on unexpected `MAXIM_ROLE` values** — silent fallback to `"leader"` was a band-aid per the no-band-aid rule; genuinely-absent env var still defaults to `"leader"` for test isolation (C2 pre-merge review A1 fold). **Note on locking:** Maxim currently has two advisory-file-lock abstractions (`maxim.utils.process_lock` for the model-download path, `filelock.FileLock` for drain state). Unification is tracked in `docs/plans/cross_platform_file_lock.md` as a shell plan — revisit after C3 lands to consolidate. **If you find yourself wanting a `mesh.yml::<mutable-field>` for the convenience of "operator committed their intent," stop — you're about to introduce a two-source-of-truth reconciliation problem. Put it in `~/.maxim/util/` and call `maxim peer list-<thing>` to surface it.** CC2 in the C1 pre-merge review + A1R2 + A3R2 in the C2 pre-design review + A1/A2/A3/CCR1/J1 in the C2 pre-merge fold round all independently cross-confirmed this rule.

**Context-similarity attribution is the wrong mechanism when a direct lookup key exists** (SEM execution hook Stage 1, 2026-04-14): the pre-fix `ToolPainBridge._on_embodiment_pain` tried to attribute tool-invoked embodiment pain via `nac.record_outcome_full`'s context-similarity path. The pending tool event's context at `record_tool_start` was `{"params": {...}}` (1 key). The outcome context from `body.py::_publish_pain` was `{source, entity, entity_type, failure_mode, composes, intensity, sensor_readings}` (7 keys, zero overlap). Per the P2 Stage 2 directional-denominator rule (`len(ctx1)` on the pending side), similarity = `0 / 1` = 0.0, below the 0.5 threshold. Every tool-invoked SEM affordance silently failed to produce NAc learning. The fix is NOT a context-similarity tweak — it's architectural: when a direct lookup key exists (`(tool_name, invocation_id)` from the executor's pending map), use it. `ToolPainBridge.record_tool_embodiment_failure` pops the pending entry by `(tool_name, invocation_id)` and calls `nac.record_outcome` (not `record_outcome_full`) with NEGATIVE valence. The executor routes to this method when `result.side_effects["embodiment_failures"]` is non-empty. `_on_embodiment_pain` guards on `bool(self._pending_tools)` and short-circuits while any tool is in flight to prevent double-recording. **Rule:** context similarity is the *fallback* for out-of-band attribution (autonomous SEM ticks with no pending tool); it is NEVER the right mechanism when you have a direct lookup key. If a new code path wants to use `record_outcome_full` with context similarity, ask first whether a direct key is available and prefer `record_outcome`.

**PainBus is the permanent rich-context carrier; ReactionBus is the typed isolation surface** (Substrate P2 Stage 2, 2026-04-13): The two buses coexist by design and serve different audiences. `Reaction` / `ReactionContext` enforces the typed isolation rules in `reactions/types.py` docstring (no cross-agent intent, no private state, no scenario oracles, no learned-policy hints) — its `bindings: dict[str, TraceSnapshot]` is deliberately strict. `PainSignal.context: dict[str, Any]` is the rich free-form carrier for bio-pipeline-internal **cause-description** metadata (`source`, `entity`, `entity_type`, `failure_mode`, `composes`, `sensor_readings`) that feeds NAc causal learning but is NOT a learning hint in the isolation-rule sense. Bio-internal publishers (`body.py`, `sandbox.py`, tool-failure bridges) call `PainBus.publish(PainSignal(...))` — this dispatches to direct PainSignal subscribers with full context AND forwards the converted Reaction to `reaction_bus` for typed subscribers (lossy, by design). Pre-Stage-2 this module was labeled "backward-compatible, Phase 2b deprecation planned"; the rewrite makes PainBus load-bearing. Do NOT route rich cause-description through `ReactionContext.bindings` — that violates the isolation docstring. Do NOT re-add a `ContextVar`-based signal stash as a back channel (the original Stage 2 draft used one; pre-merge review flagged it as a re-entrancy hazard under `@resilient` retry and async contexts). PainBus has its own `(entity, failure_mode)` refractory gate distinct from reaction_bus's coarser `(kind, source)` gate — this is intentional: `pain_signal_to_reaction` synthesizes `source` from `pain_type` alone, so without the finer PainBus gate, two distinct entities firing embodiment pain in the same tick would silently collapse into one dispatch. Any new call site that subscribes via `PainBus.subscribe` receives the full `signal.context`; code that wants the strict typed view subscribes to `self.reaction_bus.subscribe("pain", ...)` directly. Reference: `proprioception/pain_bus.py` rewritten module docstring.

## Running simulations — keep them small

Simulations call a live LLM for every turn and can burn cost + time quickly. When running sims from this CLI (for diagnostics, verification, or debugging):

- **IMPORTANT: Use `--interactive false` when running sims from Claude Code or scripts.** Interactive mode is ON by default in CLI with a TTY (0.3.2). The raw terminal reader conflicts with non-human stdin. Always pass `--interactive false` for automated/scripted sim runs.
- **Set a narrow goal.** `--goal "test X specifically"` beats `--goal "test safety"` — specific goals converge faster.
- **Cap duration.** Hit Ctrl+C after 30–90 seconds when you've seen what you need. Sims report partial results on cancel.
- **Prefer --sandbox tmpdir for debugging** unless you're specifically testing Docker — tmpdir has no pull/startup cost.
- **Use --debug sparingly.** The verbose-trace output is great for diagnosing stalls but floods the terminal for routine runs.
- **Don't invoke sims from test suites** unless the test is specifically for sim machinery. The sim runner spins up real LLM calls and can 2-3x test-suite runtime.
- **Re-use sessions with --resume-sim SESSION_ID** to avoid re-running setup + warm-up costs when iterating on a specific run.
- **Local models > Claude for loop-testing.** Use `--language-model mistral-7b` for sanity checks; save Claude for verifying final behavior.
- **Watch for Cost:** in the final report. $0.05–$0.15 per short run is normal; $0.50+ for a single debug session suggests the sim is too broad or too long.

## Architectural invariants (do not break without discussion)

- **Memory tier progression is one-way**: FORMING → WORKING → SHORT_TERM → LONG_TERM. Don't skip or reverse.
- **Hippocampus, NAc, and ATL maintain SEPARATE EpisodicMemory instances** — this is intentional coexistence, not tech debt. Don't merge.
- **Tool results flow through the agent bus**; don't call agents directly from tools.
- **Persistence uses `maxim.utils.atomic_io.atomic_write_json`** (fsync + tmp cleanup). Don't hand-roll `open().write()` + `os.replace()`.
- **LLM access goes through `models/language/router.py`**; backends (anthropic/llama/openai/transformers/**maxim_peer**) should not be imported directly from outside `models/language/`. Self-hosted peer routes go through `_MaximPeerBackend`; cloud routes stay on `_OpenAIBackend`. Selection is driven by `runtime/lane_backends.BACKEND_CLASSES` + `resolve_backend_class` — adding a new backend type is exactly one line in the dispatch table + one branch in `_classify_backend`, no router edit. The `"maxim_peer"` / `"maxim-peer"` spelling is normalised by `resolve_backend_class`.
- **`_MaximPeerBackend.complete_with_usage()` makes EXACTLY one HTTP call.** See the Plan 3 lesson above. The CI grep `grep -nE "retry|backoff|gateway" src/maxim/models/language/maxim_peer_backend.py | grep -vE "retry_after_s|retry_timeout_s"` enforces this. Failover is the router's job, not the backend's.
- **`_MaximPeerBackend.for_url(url, api_key=k, model=m).health_check()` is the canonical probe entry point** (Plan 3 R2.6). `probe_llm_server` / `llm_server_responding_at` / `_probe_once` still exist as DEPRECATED thin compat shims; the CI grep allow-list in `test.yml` blocks new callers. Do NOT add new usages of the shims.
- **`_MaximPeerBackend.for_url` is concurrency-safe via instance-level `_api_key_override`** — it does NOT mutate `os.environ`. If you add a new backend factory that needs to accept an override, store it on the instance, never on process-global state.
- **The WorkerPool is owned by LLMWorker**, which shuts it down on `stop()`. Don't create a parallel pool.
- **`@resilient` decorator (runtime/resilient.py) wraps any callback that can fail** — use it instead of bare `except Exception: pass`.
- **`RequestContext` + `contextvars.ContextVar` is the multi-agent contract.** Set at the request boundary (agent loop, sim orchestrator), read automatically by `maxim/utils/http.py::_build_headers` to populate `X-Maxim-*` outbound headers on internal endpoints. Callers don't thread the context through function signatures — set the contextvar, read at the leaves. The `HTTPEndpoint.internal: bool` flag gates which endpoints propagate X-Maxim-* — third-party URLs (HuggingFace, DuckDuckGo, tool fetches) use `internal=False` so request IDs don't leak. When adding a new cluster-internal endpoint, set `internal=True`; when adding an external one, set `internal=False`.
- **HTTP errors are typed, not string-matched.** `maxim/utils/http.py` defines `HTTPError` + subclasses (`HTTPTimeout`, `HTTPConnectionError`, `HTTPAuthError`, `HTTPServerError`, `HTTPClientError`, `HTTPRateLimited`) with `.status` + `.fix_hint`. Callers branch on these instead of parsing exception messages. **Plan 2 R2b SHIPPED** the parallel `BackendError` hierarchy in `models/language/types.py` (`.status`, `.response`, `.fix_hint` — same three access patterns, no `raw_body` or parallel attributes). **Plan 3 R2.5 SHIPPED** the router bridge in `LLMRouter._try_provider` that catches each subclass in specific-before-general order. Backends convert HTTP errors to Backend errors via one-line `except HTTPRateLimited as e: raise BackendOverloaded(...) from e` pairs; the router branches on the typed Backend exceptions. Do NOT introduce a parallel exception type — extend the existing hierarchy in `types.py` + add the corresponding router branch in specific-before-general order. Order violation is the same class of bug the R2c stage-2 probe review round caught (auth mis-classified as inference_broken). `INFERENCE_BROKEN_BACKOFF_S = 15.0` is the single source of truth linking router backoff to probe cache TTL — import, don't duplicate.
- **`raw_proxy_forward` and `raw_proxy_forward_streaming` are both reserved for `leader_proxy._proxy_request` ONLY.** Do not call either from other modules. `raw_proxy_forward_streaming` returns a `StreamingResponse` whose `_stream_ctx` field keeps the underlying httpx stream context alive until `close()` is called — see the "httpx stream contexts must outlive their consumers" lesson above.
- **Tool-invoked embodiment pain attributes directly, not via context similarity.** When a tool in flight produces SEM failures (`ModulatorAffordanceTool.execute` populates `ToolOutput.side_effects["embodiment_failures"]`), `runtime/executor.py` routes the post-execute call to `ToolPainBridge.record_tool_embodiment_failure(tool_name, invocation_id, failures)` which calls `nac.record_outcome` with a direct event_id — NO context similarity. `ToolPainBridge._on_embodiment_pain` guards on `bool(self._pending_tools)` and early-returns while any tool is in flight to avoid double-recording and to avoid the broken context-similarity path that the pre-fix code silently failed through (pending event `{"params":...}` vs rich outcome `{source, entity, failure_mode, ...}` had zero key overlap → similarity 0.0 → no link). The guard assumes serialized single-executor semantics (one `Executor` + one `ToolPainBridge` per agent instance); if a future refactor shares a bridge across concurrent in-flight tools, narrow the guard by matching on `signal.context["entity"]` against `_pending_contexts`. See [src/maxim/bridges/tool_pain_bridge.py::record_tool_embodiment_failure](src/maxim/bridges/tool_pain_bridge.py) and the ToolOutput.side_effects docstring in [src/maxim/tools/base.py](src/maxim/tools/base.py).
- **`ToolOutput.side_effects` is the typed channel for bio-pipeline signals.** A `dict[str, Any] | None` field on `ToolOutput`. Well-known keys are documented in the class docstring (append-only registry). Current keys: `"embodiment_failures"` (list of SEM failure event dicts). Add new keys in the docstring when adding a new bio signal. Do NOT hijack `metadata` (caller-facing extras) or `output` (main result) — those serve different audiences and collapsing them silently couples the tools layer to bio concepts.
- **`runtime/bootstrap.py::build_executor` is the canonical bridge wiring site.** The `pain_bus` parameter is REQUIRED (keyword-only, no default). Every caller makes an explicit `pain_bus=<bus>` (opt in to bio-learning) or `pain_bus=None` (explicit opt-out for sandbox executors, headless agents, tests) decision. Bridges cannot be retrofitted onto an Executor — wrapping (`FearGatedExecutor`, `PainInterceptorExecutor`, `AnticipatoryPainExecutor`) MUST happen AFTER `build_executor` returns. The previous helper `runtime/embodiment_bootstrap.py::bootstrap_embodiment_and_pain_bridge` is DELETED — do not re-introduce it. The fail-fast invariants are checked at the top of `build_executor` BEFORE any object construction: `pain_bus` XOR `pain_detector`, `entity_ref` requires `pain_bus` + `component_registry`, any subscription path requires `nac`. Six logical call sites in `src/maxim/` (cli.py, simulation/orchestrator.py × 2, embodied_runtime/agentic_runtime.py, api.py, simulation/tools.py) — seven physical calls because two sites have an `if/else` fallback. Each makes an explicit `pain_bus=` decision. The AUT call site in `simulation/orchestrator.py` conditionally passes `entity_ref` + `component_registry` + `pain_bus=aut_pain_bus` when `--embodiment` is used with `--sim` (E0, 0.6). Bridge construction is gated on `nac is not None`, NOT on `pain_bus`/`pain_detector` — the bridge's primary value is direct attribution via `record_tool_embodiment_failure`; subscription is the secondary out-of-band path. See `docs/plans/executor_bootstrap_unification.md`.
- **`proprioception/pain_bus.py::build_pain_bus` is the canonical PainBus construction site** (Wave 1 of biosystem_unification, 2026-04-14). Required keyword-only `hippocampus` and `nac`; forgetting either is a `TypeError`, not a silent no-op. `None` is the explicit opt-out for sandboxes / tests / api.py headless. Auto-subscribes `create_pain_memory_subscriber` and `create_pain_nac_subscriber` for the non-None subjects; optional `additional_subscribers` tuple appends after the standard learners. The motivating bug class: three CLI entry points (non-sim, `--sim agent`, `--sim interactive`) constructed `PainBus()` and subscribed only `create_pain_memory_subscriber`, silently skipping `create_pain_nac_subscriber` even though `_cli_nac` was in scope — out-of-band SEM pain reached hippocampus but never NAc on three of four agent paths. Tool-invoked pain still reached NAc via the direct-attribution path through `ToolPainBridge`, so the bug was invisible to the substrate P2 cascade test. The structural fix mirrors `build_executor` — push the wiring invariant DOWN into the constructor signature so the next sibling entry point cannot reproduce the bug. **Per L4** (gate on the learning subject, not the signal source), `build_pain_bus` takes only `hippocampus`/`nac`/extras — no ReactionBus, no enabled flag, no kind filter. **Per L5** (declared fields, not stashes), `PainBus.__init__` keeps its existing declared fields untouched. **Raw `PainBus()` construction is intentionally still allowed** — the ~30 test sites across 7 files don't migrate (same precedent as `Executor()` raw construction surviving the executor unification). The structural enforcement lives at the production door, not the type. **DefaultNetwork** ([default_network/network.py:360](src/maxim/default_network/network.py)) currently constructs `PainBus()` directly with split subscriber ownership (NAc via `PainCircuitBridge`, hippocampus via the external consumer in `embodied_runtime/agentic_runtime.py:719`); this is **deliberately deferred to Wave 2** (`memory_hub_unification.md`) because the proper fix couples DefaultNetwork to MemoryHub. **api.py headless** ([api.py:436](src/maxim/api.py)) keeps its explicit `pain_bus=None` opt-out — the structural side is resolved (the door exists) but the user-facing API decision (default-on vs default-off bio-learning for headless `pymaxim` agents) belongs to `agent_factory_canonicalization.md` Stage F5. **Latent bridge × subscriber attribution-asymmetry trap** (surfaced by the pre-merge architecture review): post-Wave-1, CLI paths now wire BOTH `create_pain_nac_subscriber` AND `ToolPainBridge` to the same bus. The bridge has a `_pending_tools` guard at [tool_pain_bridge.py::_on_embodiment_pain](src/maxim/bridges/tool_pain_bridge.py); the subscriber does NOT. Today, the subscriber's `record_outcome_full` silently fails to link pending tool events because `_context_similarity({"params":...}, {7-key-body-context}) = 0/1 = 0.0 < 0.5 threshold`. Net: no double-counting **today**. **The correctness is load-bearing on this similarity mismatch, not on any guard.** If anyone enriches `record_tool_start`'s pending-event context (e.g., adds `entity` for the broad-guard narrowing contemplated at `tool_pain_bridge.py:367-371`), double-counting starts silently. Tripwire: `tests/unit/test_pain_bus.py::TestBuildPainBus::test_subscriber_does_not_link_pending_tool_event`. If that test fails, **DO NOT relax the assertion** — open [docs/plans/pain_bus_bridge_subscriber_unification.md](docs/plans/pain_bus_bridge_subscriber_unification.md) and ship the deeper Option-B fix (`build_pain_bus(*, tool_pain_bridge=...)` parameter + bridge-aware subscriber, OR invert the wiring so the bridge mediates NAc attribution and the subscriber is the no-bridge fallback). Adding the `_pending_tools` guard to the subscriber directly is a coupling band-aid that violates layer boundaries (subscriber lives in `proprioception/pain_bus.py`, bridge state lives in `bridges/tool_pain_bridge.py`). See [docs/plans/pain_bus_unification.md](docs/plans/pain_bus_unification.md) "Latent risk surfaced during pre-merge review" section.

- **`runtime/bootstrap.py::build_default_network` is the canonical DefaultNetwork construction site** (Wave 2 of biosystem_unification, 2026-04-16). Layer 4 → Layer 5 upgrade: `nac` is now REQUIRED keyword-only; forgetting it is a `TypeError`. Pass `None` to explicitly opt out. `maxim=None` is the headless/sim opt-out — DN still provides pain detection + novelty tracking without motor control (previous version returned `None` early, which was wrong for sim). `pain_bus=` parameter closes Gap B from `pain_bus_unification.md`: when provided, DN uses the injected bus (which already has hippocampus + NAc subscribers from `build_pain_bus`) instead of constructing its own internally. This inverts DN from bus **constructor** to bus **consumer**, closing the split-subscriber-ownership problem where hippocampus was wired externally at `agentic_runtime.py:719`. Exception handling narrowed: `ImportError` → `None` + warning (optional dep); config/type errors propagate. `config=` parameter accepts pre-built `DefaultNetworkConfig` (sim mode) with precedence over `config_path=`. Two production callers: `agentic_runtime.py` (Reachy robot path — injects PainBus) and `simulation/orchestrator.py` (sim path — injects aut_pain_bus, `maxim=None`). CLI non-sim and api.py headless are explicit opt-outs (no DN — no robot → no vision → no reactive behaviors; documented per Gaps D+E). See [docs/plans/default_network_unification.md](docs/plans/default_network_unification.md).
- **`reactions/bus.py::build_reaction_bus` is the canonical ReactionBus construction site** (Wave 1 of biosystem_unification, 2026-04-16). Accepts optional `per_kind_subscribers`, `all_subscribers`, `history_size`, `refractory_overrides`. Unlike `build_pain_bus`, does NOT have required learning-subject parameters — ReactionBus is a generic typed pub/sub whose subscribers vary by caller. The builder exists for **downstream Wave 3 sequencing**: `bio_stack_unification.md` prescribes `reaction_bus = build_reaction_bus(...)` constructed BEFORE `pain_bus = build_pain_bus(..., reaction_bus=reaction_bus)` because PainBus depends on ReactionBus at construction time. Today the builder has ZERO production callers — `PainBus.__init__` constructs `ReactionBus()` directly. Wave 3's `build_bio_stack` will be the first production caller when it constructs a standalone ReactionBus and passes it to `build_pain_bus(..., reaction_bus=rb)`. The thin wrapper is intentional — the interface is the deliverable. Raw `ReactionBus()` construction is intentionally still allowed for tests (~14 sites across 3 files). **`cerebellum_modulator_factory` now accepts `reaction_bus=`** — pre-audit the factory silently dropped the parameter, so every SEM modulator failure reaction was silently discarded (`_emit_failure_reaction` returned at `if self._reaction_bus is None`). The factory itself has zero production callers today (the entire Cerebellum backend path is infrastructure ready for SEM wiring but not yet invoked from any runtime startup path); the fix is preemptive so the next caller wires it correctly. See [docs/plans/reaction_bus_unification.md](docs/plans/reaction_bus_unification.md).
- **`integration/memory_hub.py::build_memory_hub` is the canonical MemoryHub construction site** (Wave 2 of biosystem_unification, 2026-04-16). Takes the four core bio-systems (`hippocampus`, `scn`, `nac`, `ec`) as required keyword-only args plus optional bio-systems (`atl`, `angular_gyrus`, `worker_pool`, `cerebellum`, `embodiment`) and bridge deps (`spatial`, `attention`, `salience`, `fear_agent`, `novelty_tracker`). **Always calls `.connect()` internally**, so the three always-created bridges (PlanHistoryBridge, EscalationLearningBridge, FearCircuitBridge) are alive on every hub returned by the builder. The motivating bug class: two production sites (cli.py non-sim agent, AgentFactory NPC agents) constructed `MemoryHub()` and **never called `.connect()`** — all three bridges were permanently `None`, silently disabling plan-template learning, escalation-threshold learning, and risk-adjustment learning. A third site (orchestrator orch hub) was dead code — `MemoryHub(hippocampus=..., nac=...)` missing required `scn`/`ec` dataclass fields, TypeError swallowed by `except Exception`. Raw `MemoryHub()` construction is intentionally still allowed for tests (~16 sites across 8 files). Site #4 (Reachy embodied runtime) uses the builder at construction then calls `.connect()` again later to wire spatial/salience from DefaultNetwork — this is safe because bridges are stateless at construction (the second call overwrites). See [docs/plans/memory_hub_unification.md](docs/plans/memory_hub_unification.md).

- **`runtime/bio_stack.py::build_bio_stack` is the canonical bio-pipeline construction site** (Wave 3 of biosystem_unification, 2026-04-17). Composes the four individual Wave 1+2 builders (`build_reaction_bus`, `build_pain_bus`, `build_memory_hub`, `build_default_network`) in the correct dependency order. Returns a frozen `BioStack` dataclass containing all wired bio-systems. `persistence_dir: Path | str | None` is the primary configuration — sub-paths (`hippocampus.json`, `atl.json`, `angular_gyrus.json`) are derived internally. `pain_bus=` parameter accepts a pre-built PainBus (sim AUT pattern where the sandbox needs the bus before the rest of the stack); standard learners are subscribed to the pre-existing bus. `with_default_network=True` constructs a DefaultNetwork (Reachy + sim AUT only). Four production callers: cli.py non-sim, simulation/orchestrator.py AUT + orch NPC, embodied_runtime/agentic_runtime.py Reachy. AgentFactory (site #7) deferred to `agent_factory_canonicalization.md` Wave 4 — conditional `remembers`/`learns` + auto_load doesn't fit the umbrella. CLI sim modes stay as-is (just `build_pain_bus`). See [docs/plans/bio_stack_unification.md](docs/plans/bio_stack_unification.md).

- **`Episode.valence` defaults to 0.0 on old data.** Backward compatible. Old episode dicts without the valence field deserialize cleanly.
- **`spreading_activation(propagate_valence=False)` returns `dict[str, float]` unchanged.** The `propagate_valence=True` path returns `dict[str, tuple[float, float]]`. Existing callers are unaffected.
- **NAc `_reward_bias` clamps to [0, max_reward_bias].** Negative rewards (pain) produce 0.0 bias. Bias only widens EC recognition, never narrows. Pain avoidance is handled by valence annotation on edges, not by reward bias.
- **`BioStack.save_cerebellum()` must be called at session end.** Without it, learned forward models are lost.

## `maxim doctor` — environment diagnostics

Runs platform-aware checks + prints fix hints with the user's actual IPs filled in.
Lives in [src/maxim/doctor/](src/maxim/doctor/) — three modules:

- `platform_detect.py` — OS + runtime (native/WSL1/WSL2/docker) + Linux distro
- `checks.py` — individual check functions, each returns a `CheckResult` (status: ok/warn/fail/info)
- `cli.py` — `maxim doctor` and `maxim peer test` subcommands

**Check surface (v2):** GPU/CUDA, tier detection, llama-cpp-server, auto-spawn reachability, inference coherence, leader role, LAN access, cloudflared, tunnel config/sync, API key (presence + age + permissions + auth smoke), disk space, RAM headroom, lane metrics. Peer mode: URL reachability, key check, auth, model availability, latency.

**CLI flags:** `--retry` (interactive fix loop), `--json` (machine-readable output), `--as peer <url>` / `--as leader` / `--as solo` (role override).

Companion: `maxim tunnel` subcommand in [src/maxim/tunnel/](src/maxim/tunnel/) (cloudflared wrapping + API key management).

### Maintaining this over time

**Adding a new check:**
1. Write a pure function in `doctor/checks.py` that takes `PlatformInfo` (if platform-aware) and returns a `CheckResult`.
2. Add it to the correct section in `run_all_checks()`. For peer-only checks, add to the `detected_role == "peer"` branch; for leader/solo checks, add to the `else` branch.
3. If the fix differs per platform, branch on `info.runtime` / `info.os` / `info.distro` inside the check and produce platform-specific `fix` strings with user-visible commands (users copy-paste, so make them runnable as-is).
4. Use actual detected values (IPs, paths) in fix strings — call `detect_wsl_ip()` / `detect_lan_ip()` rather than `<your-ip>` placeholders when possible.
5. Add a unit test in `tests/unit/test_doctor.py`; mock out network/process calls so tests run offline.

**When a check references another module's function** (e.g., `find_cloudflared`, `_llm_server_responding_at`), import inside the function body (not module-level) to keep `maxim doctor` fast when unused features aren't installed. Tests must patch the **original** module path (`maxim.tunnel.cloudflared.find_cloudflared`), not `maxim.doctor.checks.find_cloudflared`.

**Retry loop** (`maxim doctor --retry`): the loop is data-driven — any `CheckResult` with a `retry_id` and non-ok status is automatically included. Add the retry_id to the `retryable_fns` dict in `cli._retry_loop` with a callable that re-runs the check.

**Role detection** (`_detect_doctor_role()`): auto-detects from `MAXIM_LANE_LARGE_REMOTE_URL`. A non-localhost URL triggers peer mode. Override with `--as peer/leader/solo`.

**Adding a new platform:** extend `PlatformInfo`'s `OSName` / `Runtime` / `Distro` Literal types + the detection branches in `platform_detect.py`, and add fix-hint branches in every platform-aware check.

**`maxim peer test`** should stay self-contained — no imports from the agent runtime. It's run from peer machines that may not have the full dependency set installed.

**Don't:**
- Don't auto-execute fixes without the user asking (`--fix` flag is explicit opt-in).
- Don't make checks slow (> 1s). Network probes use short timeouts (1.5–2s). Long-running benchmarks belong in a future `maxim benchmark` subcommand.
- Don't silently drop failures — any failing check needs a user-actionable `fix` string.

Remaining doctor enhancements are tracked inline as TODO comments in [src/maxim/doctor/](src/maxim/doctor/); no standalone plan doc.

## Key Commands

```bash
# Quick start — interactive menu (no args needed)
maxim                                        # Rich menu: campaigns, chat, doctor, help

# Agent runtime
maxim --llm mistral-7b                       # local LLM
maxim --llm claude-sonnet                    # Claude (needs ANTHROPIC_API_KEY)

# Model management
maxim --list-models                          # show models + download status
maxim --delete-model llama-2-13b-chat        # free disk space

# Simulation (interactive mode ON by default for CLI with TTY)
maxim --sim "test memory recall"             # generative campaign (interactive)
maxim --sim interactive                      # interactive chat (full generative sim stack)
maxim --sim scenarios/campaigns/heist_v1.yaml  # DM campaign (human picks choices + free-text roleplay)
maxim --sim "test safety" --persona adversarial --research  # with research report
maxim --sim benchmark --models mistral-7b,qwen2.5-14b      # benchmark
maxim --sim scenarios/substrate/P0_paraphrase_collapse.yaml --seed 42  # fixture-driven (S1+S4)
# In-sim commands: /cancel /pause /resume /status /report /display clean|bio|debug
# /new <goal> /persona <name> — arrow keys scroll the log
# DM campaigns: type choice number/name, or free-text to roleplay before choosing

# Embodiment in sim (0.6+) — AUT gets SEM affordance tools + pain cascade
maxim --sim "test sword combat" --embodiment weapons/rusty_sword  # generative + entity
maxim --sim interactive --embodiment bodies/reachy_mini            # interactive + entity

# Non-interactive (for Claude Code, CI, scripting, or debugging)
maxim --sim "test memory recall" --interactive false  # raw output, no Rich panel

# Asset Foundry (0.6+) — LLM-driven SEM component generation
maxim --foundry "cyberpunk weapons" --foundry-genre cyberpunk  # generate + test + score
maxim --foundry "fantasy creatures" --foundry-count 20         # larger batch
maxim --foundry "test" --foundry-dry-run                       # generate + validate only

# Auto-Curation (0.7+) — pre-sim coverage gap filling
maxim --sim "test combat" --embodiment weapons/rusty_sword --auto-curate  # fill gaps
maxim --sim "explore" --embodiment X --auto-curate --curate-threshold 8   # higher bar
maxim --sim "test" --embodiment X --no-curate                             # explicit opt-out

# Diagnostics + networking
maxim doctor                                 # environment check
maxim doctor --retry                         # interactive fix loop
maxim tunnel setup                           # Cloudflare tunnel
maxim peer update && maxim peer restart      # remote update (auto-detects pip/git mode)
maxim peer update --version 0.3.1            # pin specific PyPI version
maxim peer update --dev                      # force git mode (origin/main)
maxim peer update --dev feat/foo             # force git mode (specific branch)
maxim peer install semantic                  # install optional extra on leader
maxim peer deps                              # show leader's installed packages

# Tests
python -m pytest tests/ -x -q -m "not slow" --ignore=tests/integration/test_memory_hub.py
```

Full CLI reference: [docs/user/cli-reference.md](docs/user/cli-reference.md)

## Remote Update Workflow

```bash
# Pip-installed leaders (auto-detected):
maxim peer update && maxim peer restart

# Git-checkout leaders (dev workflow):
git push origin main && maxim peer update --dev && maxim peer restart
```

Use `--dry-run` first if unsure. The update command auto-detects whether the leader is pip-installed or a git checkout. Use `--dev` to force git mode, `--version X.Y.Z` to pin a specific PyPI version. Use `--force` (dev mode only) if the leader has untracked files blocking the pull. Troubleshooting: [docs/troubleshooting/remote_update.md](docs/troubleshooting/remote_update.md).

**Important for Claude agents:** `maxim peer update --dry-run`, `maxim peer version`, `maxim peer logs`, `maxim peer llm --status`, and `maxim peer deps` are safe and read-only. `maxim peer update`, `maxim peer restart`, `maxim peer llm <model>`, and `maxim peer install <extras>` modify leader state — only run when explicitly asked by the user.

## Versioning

Version is defined in two places that **must stay in sync**: `pyproject.toml` and `src/maxim/__init__.py`.

**When to bump:** Any change that affects runtime behavior, CLI interface, or peer/leader protocol. Docs-only or test-only changes do not require a bump.

**Check versions:** `python -c "from maxim import get_version_info; print(get_version_info())"` or `maxim peer version` to compare local vs leader. Version mismatch means the leader needs `maxim peer update && maxim peer restart`.

## Architecture Essentials

Project structure is documented in [docs/reference.md](docs/reference.md).

- **Agent loop** lives in `runtime/agent_loop.py` with `LoopController` in `runtime/loop_controller.py`
- **Multi-agent runtime**: `AgentFactory` in `runtime/agent_factory.py` creates independent agent instances (NPC agents with isolated Hippocampus, NAc, ATL). `AgentPool` in `runtime/agent_pool.py` orchestrates concurrent multi-agent execution with `LocalMessageBus`.
- **LLM routing** lives in `models/language/router.py` (config in `models/language/config.py`). 8 cloud providers (Anthropic, OpenAI, Google Gemini, Groq, Together, Fireworks, Mistral, DeepSeek) across 15 cloud profiles, plus 15 local profiles (llama-cpp and PyTorch/Transformers backends). Self-hosted peer tunnels go through `models/language/maxim_peer_backend.py::_MaximPeerBackend` (Plan 3 R2.5) — purpose-built single-HTTP-call backend with typed failure mapping; cloud providers stay on `_OpenAIBackend`. Backend class dispatch is driven by `runtime/lane_backends.BACKEND_CLASSES` + `resolve_backend_class`.
- **Simulation** orchestrator in `simulation/orchestrator.py`, bridge in `simulation/bridge.py`. Campaign runners in `simulation/campaign_runner.py` (generative + DM + fixture). Fixture-driven testing in `simulation/fixture_orchestrator.py` (S1). Types in `simulation/sim_types.py`.
- **Interactive runtime** in `interactive/` — universal prompt protocol (`PromptRequest`/`PromptHandler`), rich terminal display with split panels, DM display extensions.
- **Mode system**: ProcessingState (awake/sleep) x OperationalMode (planning/supervised/autonomous). Sleep is a tool the agent calls; it wakes automatically on user input.
- **Memory tiers**: FORMING -> WORKING -> SHORT_TERM -> LONG_TERM (enforced by `TierTransitionError` in `agents/bus.py` — see F0.7)
- **Memory store protocols**: `EpisodicStore`, `CausalStore`, `SemanticStore` in `memory/store.py` — split persistence protocols with `File*Store` defaults and database implementations for Mother Maxim.
- **Percept/Reaction dual surface** (reaction_abstraction_plan, Phases 1–4 shipped):
  - **Percept** = sensory/environmental input. Typed `PerceptContext` (channel, sender, agent_id, scn_tag) in `agents/percept_context.py`. Named factories in `agents/percept_factory.py`: `make_text_percept`, `make_scene_percept`, `make_intero_percept`. `SensoryTag` populated at all producers via `agents/modality.py`.
  - **Reaction** = evaluative signal driving learning. Types in `reactions/types.py` (`Reaction`, `ReactionContext`, `TraceSnapshot`, `ReactionKind`). `ReactionBus` in `reactions/bus.py` (generalized from PainBus, per-kind dispatch). `PerceptProducer`/`ReactionProducer` protocols in `reactions/protocols.py`.
  - **SEM integration**: sensors → PerceptProducer (via `EmbodimentPerceptSource`), modulators → ReactionProducer (via `CerebellumModulator` mediation). SEM specs don't import Reaction types.
  - **Runtime unification**: both MaximAgent and AgentPool produce typed Percepts. AgentPool.run_turn wraps string input via `make_text_percept`.
  - **Isolation hygiene**: PerceptContext and ReactionContext must NOT carry cross-agent intent, private state, scenario oracles, or learned-policy hints. Rules documented in module docstrings of `percept_context.py` and `reactions/types.py`.
- **Lane tier system**: Functions route to capability tiers (large/medium/small) via `FunctionRouter` in `runtime/function_router.py`. `detect_tiers()` in `lane_models.py` auto-detects from hardware.
- **Data paths**: Bundled seed data in `src/maxim/_data/` (components, encounters, prompts, templates). User data at `~/.maxim/` (memory, sessions, benchmarks, config). Resolution via `utils/paths.py`.
- **SEM Component Registry**: `embodiment/component_registry.py` discovers SEM entity templates from campaign-local, `~/.maxim/components/`, and `_data/components/`. 65 seed components across 7 categories (bodies, creatures, environments, items, npcs, vehicles, weapons). Genre-gated: fantasy, cyberpunk, scifi, horror, historical, modern, devops. **Asset Foundry** (`simulation/foundry.py`) generates new components via LLM + template fallback, validates, tests (8 SEM protocol tests + 3-encounter gauntlet), and scores on 4 bio-system engagement dimensions. **ComponentIndex** (`embodiment/component_index.py`) provides two-layer semantic discovery: alias hash table (from `component.synonyms` YAML field, O(1)) + embedding cosine similarity (sentence-transformers, threshold 0.65). Reuses `similarity.encoder._get_encoder` singleton — no duplicate model. Thread-safe via RLock. Persistence via `.npy` + `.json` sidecar (no pickle). Used by imagination trigger (I1) and auto-curation dedup (E3).
- **Imagination system** (`imagination/`): Real-time entity design from novel percept mentions. Pipeline: entity noun-phrase extraction → ImaginationCache check → ComponentIndex two-layer lookup → DN arousal gate → energy budget check → EntityDesigner LLM call → quick validation → `register_ephemeral()` + `ComponentIndex.add()` → scene-scoped tool registration. Session-scoped ephemeral overlay on ComponentRegistry (`_ephemeral_index`, separate from persistent `_index`). Episodes and CausalLinks from imagined entities carry `imagined=True` provenance; on session end, `NAc.decay_imagined_links(0.5)` reduces confidence by 50%. Wired into agent loop post-`state.update()` via `imagination_trigger` parameter on `run_agentic_loop`. Per-phrase design guard prevents concurrent LLM calls for the same phrase (AUT + orchestrator race). Thread-safe throughout via RLock.
- **Thread model**: Main loop at 2-30Hz + WorkerPool (tier-based lanes: large/medium/small, owned by LLMWorker) + Hippocampus capture thread (owned + shut down by MemoryHub.on_session_end)

## Quick reference — where to look

| Area | Key files |
|---|---|
| Agent loop | `runtime/agent_loop.py`, `runtime/loop_controller.py` |
| Tools | `tools/registry.py` (scene-scoped activation, active tool cap, per-tool `deactivate_tool`), `tools/base.py` (Tool ABC), `tools/discovery.py` (SEM discovery: `DiscoverToolsTool`, `UniversalSenseTool`, goal top-k, LRU eviction), `runtime/executor.py` (dispatch + active-tool gate, aliases), `embodiment/tool_bridge.py` (entity tool generation), `embodiment/entity_map.py` (name→Entity resolution) |
| LLM routing | `models/language/router.py` (provider fallback, typed exception branches, `dispatch_exhausted` aggregated WARN), `models/language/maxim_peer_backend.py` (self-hosted peer backend — one HTTP call, typed failure, streaming with strict mid-stream fail, `health_check` + `for_url` factory), `runtime/lane_backends.py::BACKEND_CLASSES` (dispatch table), `models/language/config.py` (profiles), `models/language/json_parser.py` (JSON repair) |
| Memory | `memory/hippocampus.py`, `memory/concept_extractor.py`, `memory/store.py` (protocols), `memory/percept_trace_buffer.py` (τ-decay ring buffer) |
| Causal learning | `decisions/nac.py` (reward bias, eligibility traces, distribute_reward), `decisions/causal_link.py` (CausalLink, percept_refs) |
| Substrate encoding | `similarity/encoder.py` (LinguisticEncoder), `similarity/ec.py` (pattern_complete_or_separate, centroid update) |
| Prompt composition | `prompts/assembler.py` (PromptAssembler, MemorySummary), `agents/prompt_builder.py` (legacy), `prompts/acting_coach.py` (B3: Acting Coach — bio-modulated affordance exploration meta-prompt) |
| Percept schema | `agents/percept_context.py` (PerceptContext), `agents/percept_factory.py` (factories), `agents/modality.py` (SensoryTag, SubstrateModality) |
| Reactions | `reactions/types.py` (Reaction, ReactionContext, TraceSnapshot), `reactions/bus.py` (ReactionBus), `reactions/protocols.py` (PerceptProducer, ReactionProducer) |
| Cross-layer wiring | `integration/memory_hub.py` (single coordinator) |
| Persistence | `utils/atomic_io.py`, `utils/paths.py` (data path resolution) |
| Simulation | `simulation/orchestrator.py`, `simulation/bridge.py`, `simulation/fixture_orchestrator.py`, `simulation/personas.py` |
| Substrate test infra | `models/language/backend_protocol.py` (S2), `utils/seeding.py` (S4), `tests/substrate/` (S2+S3+P1 metrics) |
| Generative campaigns | `simulation/arcs.py`, `simulation/narrator.py`, `simulation/generative_runner.py` |
| DM campaigns | `simulation/dm_schema.py`, `simulation/dm_runtime.py` |
| Asset Foundry | `simulation/foundry.py` (FoundryRunner, generate, validate, gauntlet, score) |
| Benchmarks | `simulation/benchmark.py`, `simulation/validation.py` |
| Research | `simulation/research_agents.py`, `simulation/research_orchestrator.py` |
| Valence | `memory/episode.py` (Episode.valence, apply_hebbian_on_close, salience_spike_rule), `agents/bus.py` (propagate_valence), `memory/hippocampus.py` (capture_reaction, include_valence) |
| Embodiment | `embodiment/sem.py`, `embodiment/body.py`, `embodiment/cerebellum.py` (forward models), `embodiment/backends/cerebellum_modulator.py` (predict/fallback/train + success reactions), `embodiment/motor.py` |
| Imagination | `imagination/trigger.py` (entity extraction + ComponentIndex lookup + design dispatch), `imagination/designer.py` (ImaginationDesigner wrapping EntityDesigner), `imagination/cache.py` (session-scoped ImaginationCache) |
| Mesh | `mesh/identity.py`, `mesh/knowledge.py`, `mesh/task_delegation.py`, `mesh/clock.py` |
| Lane tiers | `runtime/function_router.py`, `runtime/lane_models.py`, `runtime/lane_backends.py` |
| Multi-agent | `runtime/agent_factory.py`, `runtime/agent_pool.py` |
| Interactive UI | `interactive/prompts.py`, `interactive/display.py` |
| Seed data | `_data/components/`, `_data/encounters/` |
| Adding env vars | Add to the env table below + touch whatever reads it |

## Environment Variables

```bash
ANTHROPIC_API_KEY          # Required for Claude backend
OPENAI_API_KEY             # Required for OpenAI backend
GOOGLE_API_KEY             # Required for Gemini backend
GROQ_API_KEY               # Required for Groq backend
TOGETHER_API_KEY           # Required for Together backend
FIREWORKS_API_KEY          # Required for Fireworks backend
MISTRAL_API_KEY            # Required for Mistral API backend
DEEPSEEK_API_KEY           # Required for DeepSeek backend
MAXIM_ROLE=leader          # Plan 2 R2a. Explicit role: leader|peer|solo. Exported by cli.py::main at startup; downstream reads it from env (runtime/llm_server.py::_model_state_file picks active_llm_model.{role}.txt).
MAXIM_LLM_ENABLED=1        # Enable LLM inference
MAXIM_LLM_PROFILE=claude-sonnet  # Default model profile
MAXIM_LLM_N_CTX=4096       # Override auto-computed llama.cpp n_ctx (P4c). Same as --llm-n-ctx.
MAXIM_AUTO_DOWNLOAD_MODELS=1     # Skip the auto-download prompt (P5). Same as --auto-download.
MAXIM_DATA_BUDGET_GB=50          # Optional soft cap on ~/.maxim disk usage; refuses downloads over budget.
MAXIM_SKIP_REMOTE_PROBE=1        # Bypass the remote-URL probe (P6) — CI escape hatch.
MAXIM_REMOTE_PROBE_FIRST_TIMEOUT_S=1.5   # First probe attempt timeout (clamped 0.2-5.0; cold httpx≈710ms)
MAXIM_REMOTE_PROBE_RETRY_TIMEOUT_S=8.0   # Retry probe timeout (clamped 0.5-10.0; dormant Cloudflare tunnel re-establishment >2.5s)

# Plan 3.5 R2 — agent-level LLM call timeout (strict safety net above HTTP layer)
MAXIM_LLM_CALL_TIMEOUT_S=300              # LLMWorker agent-level timeout (clamped 10-1800; default 300s, was 60s pre-plan). Strictly larger than _INFERENCE_PROXY_TIMEOUT_S so HTTP layer fires first with typed BackendTimeout → clean lock release. If this ever fires, it's a LOUD bug signal (HTTP layer is wedged).
MAXIM_REMOTE_PROBE_CACHE_TTL_S=60        # Probe cache freshness window (clamped 0-600)

# Decision log (P9). Append-only JSONL at ~/.maxim/util/lane_decisions.jsonl.
# Inspect with: maxim doctor --last-decision   (or tail/jq the file directly)
MAXIM_PROVENANCE_VERBOSITY=1     # 0=off, 1=compact, 2=verbose

# Substrate path (P1). Enables LinguisticEncoder → EC → ATL dual-write.
MAXIM_SUBSTRATE_PATH=1           # Enable substrate encoding path (Phase 1 dual-write)
MAXIM_CONCEPT_DECOMPOSITION=1    # Enable concept decomposition (noun-phrase extraction before EC). Requires spaCy + en_core_web_sm. Opt-in.

# HTTP client trace (Plan 1 R1)
MAXIM_HTTP_TRACE=1               # Bumps http_request events from DEBUG to INFO (every outbound call logged)
MAXIM_LOG_FILE=/tmp/maxim.jsonl  # Attaches a JSONL file handler via StructuredFormatter. Dual-format: stdout stays human-readable, file is machine-parseable. Root logger runs at DEBUG when this is set; stdout still applies its own verbosity filter.

# Peer backend trace (Plan 3 R2.5)
MAXIM_BACKEND_TRACE=1            # Bumps _MaximPeerBackend peer_backend_call events from DEBUG to INFO. Pair with MAXIM_LOG_FILE for per-call JSONL with full multi-agent context (agent_id/session_id/request_id/lane/input_tokens/output_tokens). Off by default.

# Heartbeat + trace (debug/diagnostics)
MAXIM_HEARTBEAT=1                # System health heartbeat every 10s (GPU/CPU/RAM/disk/WiFi + stall detection)
MAXIM_HEARTBEAT_INTERVAL_S=10    # Heartbeat sample interval
MAXIM_HEARTBEAT_STALL_S=30       # Warn after this many seconds with no LLM calls
MAXIM_LANE_TRACE=1               # Per-request LLM trace logs (also enables heartbeat)
MAXIM_PEER_LOG_REQUESTS=1        # JSON log per outbound peer call

# Drain routing (Plan 4 C4+C4.5+C4.6)
MAXIM_DRAIN_CACHE_TTL_S=1.0              # DrainConstraint mtime cache freshness (clamped 0-60)
MAXIM_AUTO_DRAIN_THRESHOLD=5             # Transient failure count before auto-drain (clamped 2-20; permanent=1)
MAXIM_AUTO_UNDRAIN_PROBE_INTERVAL_S=90   # Auto-undrain probe cycle interval (clamped 30-600)

# Leader proxy admission control
MAXIM_PROXY_MAX_CONCURRENT=4     # Max in-flight requests to upstream (0=unlimited)
MAXIM_PROXY_RATE_LIMIT_RPM=0     # Per-peer requests/minute (0=unlimited)

# Cloud provider integration
MAXIM_LLM_CLOUD_ENABLED=1       # Enable cloud dispatch (required for --cloud-* flags)
MAXIM_MAX_CLOUD_LANES=1          # Max lanes using cloud providers (default: 0)
MAXIM_LLM_REDACTION_POLICY=standard  # Redaction policy for cloud dispatch (standard/relaxed/strict)
MAXIM_CLOUD_SESSION_BUDGET=5.00  # Hard ceiling on cloud spending per session

# Peer/tier remote configuration (tier names: large, medium, small)
MAXIM_LANE_LARGE_REMOTE_URL=     # Override large tier to use remote server
MAXIM_LANE_LARGE_REMOTE_MODEL=   # Model name to request from remote server
MAXIM_LANE_LARGE_REMOTE_API_KEY= # Auth token for remote server
# Tier names only: large, medium, small (legacy infer/review/record removed in v1.0)
```

## Testing

```bash
# Full suite
python -m pytest tests/ -x -q -m "not slow" --ignore=tests/integration/test_memory_hub.py

# Specific test file
python -m pytest tests/unit/test_simulation_agent.py -v

# Just the module you changed (fast feedback)
python -m pytest tests/unit/test_lane_metrics.py -v
```

### Testing efficiently

**Run narrow first, then wide.** Test the specific module you changed before running the full suite (~3 min). The full suite has ~4,000 tests; don't wait for all of them on every edit.

**Kill stale sims before running tests.** A running `maxim --sim agent` process holds GPU + port resources and can cause test hangs:
```bash
pkill -f "maxim.*sim" 2>/dev/null; sleep 2
python -m pytest tests/ -x -q -m "not slow" --ignore=tests/integration/test_memory_hub.py
```

**Threading pitfalls (learned the hard way):**
- Use `threading.RLock` (not `Lock`) if a method acquires the lock and then calls another method that also acquires it (e.g. `snapshot()` calling `self.failure_rate`). Regular `Lock` deadlocks on re-entry.
- Thread-safety tests with many workers (8+ threads × 100 calls) can appear to hang if a deadlock exists — they're not slow, they're stuck.

**Don't run sims from tests.** Sims call real LLMs and can 2-3x test-suite runtime. The sim runner is for manual/CLI testing only (`maxim --sim agent`). Tests should mock LLM calls.

**Peer/tunnel testing requires the leader.** Use `curl -si -H "Authorization: Bearer $KEY" https://maxim.yourdomain.com/v1/models` for quick checks. See [docs/troubleshooting/](docs/troubleshooting/) for in-depth peer connectivity guides.

## Simulation Reports

Sim runs save to `~/.maxim/sessions/{session_id}/` (report.json, actions.jsonl, aut_hippocampus.json, aut_nac.json). Research protocol details and campaign execution flow are documented in `docs/simulation.md` and `docs/experiments/`.

## Python API (pymaxim)

Published to PyPI as `pymaxim` (import name stays `maxim`). 17 verb-based functions, all lazy-loaded from `src/maxim/api.py`. Key files: `api.py` (facades), `__init__.py` (lazy wiring), `simulation/introspection.py` (Observer).

**Rules for maintaining the API:**
- **Verbs are facades, not logic.** Delegate to existing internals. Don't put business logic in `api.py`.
- **Lazy imports only.** `import maxim` must not trigger loading of optional dependencies.
- **Return structured data, not prints.** `diagnose()` returns `DiagnosticReport`, `imagine()` returns `SimulationResult`, etc.
- **`introspect` is an alias for `observe`.** Don't add behavior to one without the other.
- **`Observer`** is the canonical name (no aliases).

**Package management:**
- **Package name:** `pymaxim` on PyPI, `maxim` as import
- **Core deps:** `numpy`, `scipy`, `pyyaml`, `json-repair` only. Everything else is optional extras.
- **Optional extras:** `llm-llama`, `llm-server`, `llm-torch`, `llm-anthropic`, `llm-openai`, `vision`, `audio`, `reachy`, `comms`, `search`, `temporal`, `training`, `tts`, `yolo`, `semantic`, `database`
- **Robot plugins:** Auto-discovered via `maxim.robots` entry-point group.
- **Build validation:** `python -m build && twine check dist/*` before any publish
- Publication guide: [publication_guide.md](docs/publication_guide.md)

## Active initiatives

See [docs/plans/README.md](docs/plans/README.md) for the roadmap index. Current version: v0.7.0 on PyPI as `pymaxim` ([publication guide](docs/publication_guide.md)).

**Recently shipped (2026-04-20):**
- **0.7 Feature Completion** — Self-generating simulations. All tracks landed:
  - R0 Prerequisites: ComponentRegistry thread safety, sim-mode consolidation, TOOL_ALIASES lock
  - B3.1 Acting Coach: config + prompt section with bio-system modulation (NAc caution, pain anticipation, cerebellum predictions)
  - F3-F5 Agent Factory: sim orchestrator + Reachy + headless API migrated to `create_full_agent`
  - E2 Real LLM: foundry wired to real LLM with entity context injection + synonym generation
  - E2.5 ComponentIndex: two-layer semantic discovery (alias hash O(1) + embedding cosine similarity)
  - E3 Auto-Curation: `--auto-curate` CLI for pre-sim coverage gap filling via foundry
  - I1 Imagination Trigger: entity extraction → ComponentIndex lookup → DN arousal gate → design dispatch
  - I2 Real-time Design: ImaginationDesigner with quick validation + synonym generation
  - I3 Scene-scoped Tools: tool window with cap (20 scene tools), deactivation, executor gate
  - Integration wiring: ImaginationTrigger constructed in orchestrator AUT path, session-end cleanup (imagined link decay + ephemeral entity clearing)
- **Version bump to 0.7.0.** Experiment: [docs/experiments/07_imagination_wiring.md](docs/experiments/07_imagination_wiring.md).

**Previously shipped (2026-04-17):**
- Valence annotation, SEM Learning Loop, Behavioral convergence wiring, Experiments 1-4 (41/41 hypotheses confirmed). Version 0.3.0.

**Previously shipped (2026-04-11/12):**
- Foundations wave F0.1–F0.8, Reaction abstraction Phases 1–4, Cleanup wave C1–C4, Peer/leader flexibility P1–P9, Simulator upgrades S1–S4. All archived.

**Gating 1.0** (three focused substrate plans, split from the master plan):
- [substrate_p0_pilot.md](docs/plans/substrate_p0_pilot.md) — **COMPLETE** (2026-04-12). Baseline pinned at 78.5%. Results: [docs/experiments/p0_baseline_sweep.md](docs/experiments/p0_baseline_sweep.md).
- [substrate_recognition.md](docs/plans/substrate_recognition.md) — **COMPLETE** (2026-04-14). B1+P1 shipped 2026-04-12 at 91.7% collapse (`paraphrase-mpnet@0.40` + centroid update). P2 Stages 1+2 shipped via PR #100 (SEM pain cascade end-to-end on real `rusty_sword` + NAc `_context_similarity` directional fix + PainBus dual-layer rewrite). P2 Stage 3 shipped via PR #102 — real-embedding sweep at `paraphrase-mpnet@0.70, reward 2.0` cleared with **+56.0 ± 29.0 pp target gain / 0.0 ± 0.0 pp distractor drift / 94% monotone / 9-of-10 seeds**, after three metric pivots (node-count → raw pair-collapse → plurality-ownership self-collapse) + a fixture pivot. Results: [docs/experiments/p1_recognition_sweep.md](docs/experiments/p1_recognition_sweep.md) + [docs/experiments/p2_reward_modulation_sweep.md](docs/experiments/p2_reward_modulation_sweep.md) + [docs/experiments/p2_sem_pain_cascade.md](docs/experiments/p2_sem_pain_cascade.md). Reproduction runbook: [docs/experiments/protocols/p2_reward_modulation_reproduction.md](docs/experiments/protocols/p2_reward_modulation_reproduction.md). 0.3-minimum gate CLOSED.
- [substrate_binding_persistence.md](docs/plans/archive/substrate_binding_persistence.md) — **SPLIT COMPLETE + ARCHIVED.** Now a pure index. All four 0.3-target phases CLOSED. Per-phase plan files created for 0.5 track.

**Living practice docs (paired with substrate_plan):**
- [behavioral_convergence_practice.md](docs/plans/behavioral_convergence_practice.md) — does the agent actually get better across sessions? Living doc, not a gate.
- [memory_consolidation_practice.md](docs/plans/memory_consolidation_practice.md) — refines the P8 sleep-replay mechanism. Kicks in when P8 ships in 0.5.

**Parallel:**
- [tool_refinement_plan.md](docs/plans/tool_refinement_plan.md) — living doc for agent tool curation.

**Deferred (post-1.0, revive on trigger):** Bio-System Plugin Discovery, Unified Event Bus, Mother NPC Stimulus, Mother Maxim, Pecking Order Graph, Asset Foundry, DM Extensions. See [docs/plans/deferred/](docs/plans/deferred/).
