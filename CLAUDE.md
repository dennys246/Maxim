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
- **No band-aid fixes.** If you spot a bug while working on a task, determine whether the fix addresses the root cause or merely hides the symptom. If it's the latter — a special case, a swallowed exception, a flag that toggles around broken behavior, a fix that would need to be repeated elsewhere — stop, describe the root cause and the scope of the proper fix, and ask the user how to proceed. Never silently choose the smaller fix because it's easier.
- Prefer editing existing modules over creating new ones — this codebase favors many small files already
- Don't rename bio-system classes (Hippocampus, ATL, NAc, SCN, EC, AngularGyrus) — names are load-bearing for the mental model
- If you touch provenance, run a sim with `MAXIM_PROVENANCE_VERBOSITY=2` and eyeball the trace
- **Run `mypy` on public API files** after changing api.py, session.py, create.py, load.py, or __init__.py: `mypy src/maxim/__init__.py src/maxim/api.py src/maxim/session.py src/maxim/create.py src/maxim/load.py --ignore-missing-imports`
- **Run `ruff format`** after any changes: `ruff format src/ tests/`

## Lessons learned (bugs that bit us)

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

## Running simulations — keep them small

Simulations call a live LLM for every turn and can burn cost + time quickly. When running sims from this CLI (for diagnostics, verification, or debugging):

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
# Agent runtime
maxim --llm mistral-7b                       # local LLM
maxim --llm claude-sonnet                    # Claude (needs ANTHROPIC_API_KEY)

# Model management
maxim --list-models                          # show models + download status
maxim --delete-model llama-2-13b-chat        # free disk space

# Simulation
maxim --sim "test memory recall"             # generative campaign
maxim --sim scenarios/campaigns/heist_v1.yaml  # DM campaign
maxim --sim "test safety" --persona adversarial --research  # with research report
maxim --sim benchmark --models mistral-7b,qwen2.5-14b      # benchmark
maxim --sim scenarios/substrate/P0_paraphrase_collapse.yaml --seed 42  # fixture-driven (S1+S4)

# Diagnostics + networking
maxim doctor                                 # environment check
maxim doctor --retry                         # interactive fix loop
maxim tunnel setup                           # Cloudflare tunnel
maxim peer update && maxim peer restart      # remote update
maxim peer install semantic                  # install optional extra on leader
maxim peer deps                              # show leader's installed packages

# Tests
python -m pytest tests/ -x -q -m "not slow" --ignore=tests/integration/test_memory_hub.py
```

Full CLI reference: [docs/user/cli-reference.md](docs/user/cli-reference.md)

## Remote Update Workflow

```bash
git push origin main && maxim peer update && maxim peer restart
```

Use `--dry-run` first if unsure. Use `--force` if the leader has untracked runtime files blocking the pull. Troubleshooting: [docs/troubleshooting/remote_update.md](docs/troubleshooting/remote_update.md).

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
- **SEM Component Registry**: `embodiment/component_registry.py` discovers SEM entity templates from campaign-local, `~/.maxim/components/`, and `_data/components/`. 54 seed components across 7 categories (bodies, creatures, environments, items, npcs, vehicles, weapons). Genre-gated: fantasy, cyberpunk, scifi, horror, historical, modern, devops.
- **Thread model**: Main loop at 2-30Hz + WorkerPool (tier-based lanes: large/medium/small, owned by LLMWorker) + Hippocampus capture thread (owned + shut down by MemoryHub.on_session_end)

## Quick reference — where to look

| Area | Key files |
|---|---|
| Agent loop | `runtime/agent_loop.py`, `runtime/loop_controller.py` |
| Tools | `tools/` (register in registry), `runtime/executor.py` (aliases) |
| LLM routing | `models/language/router.py` (provider fallback, typed exception branches, `dispatch_exhausted` aggregated WARN), `models/language/maxim_peer_backend.py` (self-hosted peer backend — one HTTP call, typed failure, streaming with strict mid-stream fail, `health_check` + `for_url` factory), `runtime/lane_backends.py::BACKEND_CLASSES` (dispatch table), `models/language/config.py` (profiles), `models/language/json_parser.py` (JSON repair) |
| Memory | `memory/hippocampus.py`, `memory/concept_extractor.py`, `memory/store.py` (protocols), `memory/percept_trace_buffer.py` (τ-decay ring buffer) |
| Causal learning | `decisions/nac.py` (reward bias, eligibility traces), `decisions/causal_link.py` (CausalLink, percept_refs) |
| Substrate encoding | `similarity/encoder.py` (LinguisticEncoder), `similarity/ec.py` (pattern_complete_or_separate, centroid update) |
| Prompt composition | `prompts/assembler.py` (PromptAssembler, MemorySummary), `agents/prompt_builder.py` (legacy) |
| Percept schema | `agents/percept_context.py` (PerceptContext), `agents/percept_factory.py` (factories), `agents/modality.py` (SensoryTag, SubstrateModality) |
| Reactions | `reactions/types.py` (Reaction, ReactionContext, TraceSnapshot), `reactions/bus.py` (ReactionBus), `reactions/protocols.py` (PerceptProducer, ReactionProducer) |
| Cross-layer wiring | `integration/memory_hub.py` (single coordinator) |
| Persistence | `utils/atomic_io.py`, `utils/paths.py` (data path resolution) |
| Simulation | `simulation/orchestrator.py`, `simulation/bridge.py`, `simulation/fixture_orchestrator.py`, `simulation/personas.py` |
| Substrate test infra | `models/language/backend_protocol.py` (S2), `utils/seeding.py` (S4), `tests/substrate/` (S2+S3+P1 metrics) |
| Generative campaigns | `simulation/arcs.py`, `simulation/narrator.py`, `simulation/generative_runner.py` |
| DM campaigns | `simulation/dm_schema.py`, `simulation/dm_runtime.py` |
| Benchmarks | `simulation/benchmark.py`, `simulation/validation.py` |
| Research | `simulation/research_agents.py`, `simulation/research_orchestrator.py` |
| Embodiment | `embodiment/sem.py`, `embodiment/body.py`, `embodiment/cerebellum.py`, `embodiment/motor.py` |
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
MAXIM_REMOTE_PROBE_CACHE_TTL_S=60        # Probe cache freshness window (clamped 0-600)

# Decision log (P9). Append-only JSONL at ~/.maxim/util/lane_decisions.jsonl.
# Inspect with: maxim doctor --last-decision   (or tail/jq the file directly)
MAXIM_PROVENANCE_VERBOSITY=1     # 0=off, 1=compact, 2=verbose

# Substrate path (P1). Enables LinguisticEncoder → EC → ATL dual-write.
MAXIM_SUBSTRATE_PATH=1           # Enable substrate encoding path (Phase 1 dual-write)

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

See [docs/plans/README.md](docs/plans/README.md) for the roadmap index. Current version: v0.2.1 on PyPI as `pymaxim` ([publication guide](docs/publication_guide.md)).

**Recently shipped (2026-04-11/12):**
- Foundations wave F0.1–F0.8 — all landed. Archived.
- Reaction abstraction Phases 1–4 — Percept/Reaction dual-surface architecture. Archived. Phase 5 folds into substrate P2.
- Cleanup wave C1–C4, Peer/leader flexibility P1–P9 — all archived.
- Simulator upgrades S1–S4 **SHIPPED** (2026-04-12). Archived.

**Gating 1.0** (three focused substrate plans, split from the master plan):
- [substrate_p0_pilot.md](docs/plans/substrate_p0_pilot.md) — **COMPLETE** (2026-04-12). Baseline pinned at 78.5%. Results: [docs/experiments/p0_baseline_sweep.md](docs/experiments/p0_baseline_sweep.md).
- [substrate_recognition.md](docs/plans/substrate_recognition.md) — **in progress.** B1+P1 **SHIPPED** (2026-04-12): 91.7% collapse with paraphrase-mpnet@0.40 + centroid update. P2 core merged, P2 validation remaining. Results: [docs/experiments/p1_recognition_sweep.md](docs/experiments/p1_recognition_sweep.md). 0.3-pre → 0.3-minimum.
- [substrate_binding_persistence.md](docs/plans/substrate_binding_persistence.md) — blocked on recognition P2. P3a–P8 + B3-B5. Includes 1.0-gating P4 cross-modal head-to-head. ~4,100 LOC. 0.3-target → 0.5.

**Living practice docs (paired with substrate_plan):**
- [behavioral_convergence_practice.md](docs/plans/behavioral_convergence_practice.md) — does the agent actually get better across sessions? Living doc, not a gate.
- [memory_consolidation_practice.md](docs/plans/memory_consolidation_practice.md) — refines the P8 sleep-replay mechanism. Kicks in when P8 ships in 0.5.

**Parallel:**
- [tool_refinement_plan.md](docs/plans/tool_refinement_plan.md) — living doc for agent tool curation.

**Deferred (post-1.0, revive on trigger):** Bio-System Plugin Discovery, Unified Event Bus, Mother NPC Stimulus, Mother Maxim, Pecking Order Graph, Asset Foundry, DM Extensions. See [docs/plans/deferred/](docs/plans/deferred/).
