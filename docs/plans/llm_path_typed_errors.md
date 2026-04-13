# LLM Path Refinement — Plan 2: Role Detection + Typed Error Taxonomy

**Status:** Draft — proposed 2026-04-12, split from Plan 1 per user request. **Plan 1 R0+R1 SHIPPED 2026-04-12 (PRs #88, #90, pending cleanup PR for `c8a07e9`).** R2a-d ready to start.
**Scope:** ~280 LOC new
**Target version:** 0.4 (single stability version containing all LLM path sub-plans)
**Part of:** [llm_path_refinement.md](llm_path_refinement.md) — the LLM path refinement meta-plan
**Depends on:** [llm_path_foundation.md](llm_path_foundation.md) (Plan 1) — R0 + R1 SHIPPED, unblocks Plan 2
**Blocks:** [llm_path_fast_failover.md](llm_path_fast_failover.md) (Plan 3, formerly Plan 2)

## R1 learnings to apply before starting R2

**Read before writing any code:** [project_llm_path_r1_shipped.md](../../.claude/projects/-Users-dennyschaedig-Scripts-Maxim/memory/project_llm_path_r1_shipped.md) + [docs/architecture/llm_routing.md](../architecture/llm_routing.md) Layer 7. The R1 implementation diverged from its plan spec in five load-bearing ways; Plan 2's own implementation should mirror R1's patterns, not invent new ones.

**Specific R1 patterns Plan 2 must match:**

1. **`BackendError` class shape mirrors `HTTPError`** — same three access patterns (`.status`, `.response`, `.fix_hint`). See R2b below.
2. **`_normalize_request_context` is the single canonical shim** — R1 introduced `RequestContext` + `contextvars.ContextVar` in `utils/http.py`. R2b's `_normalize_request_context` in `agents/llm_worker.py` is the ONE place the legacy dict shape gets bridged to the typed dataclass. Plan 3's `_MaximPeerBackend` imports this function; do not define a parallel shim.
3. **`probe_llm_server` + `_classify_probe_cause` are already migrated** — R2c adds stage-2 readiness on top of the already-migrated stage-1 probe. Do NOT re-write `_classify_probe_cause`; it stays until Plan 3 R2.5 supersedes the whole function.
4. **`detect_role()` must be called from the EARLY `cli.py::main` block** — before subcommand dispatch, not at sim-loop entry. The subcommand-dispatch bug bit Plan 1 R1 (commit `c8a07e9` fixed it); Plan 2 R2a will re-encounter it if the `role_detected` log event doesn't fire for `maxim doctor` / `maxim peer X`. See [feedback_subcommand_logging_gap.md](../../.claude/projects/-Users-dennyschaedig-Scripts-Maxim/memory/feedback_subcommand_logging_gap.md).
5. **JSONL log format uses single-letter top-level keys** (`t`/`l`/`s`/`e`) — runbook examples for R2a's `event=role_detected` MUST use `jq 'select(.e=="role_detected")'`, not `.event`. See [feedback_structured_formatter_short_keys.md](../../.claude/projects/-Users-dennyschaedig-Scripts-Maxim/memory/feedback_structured_formatter_short_keys.md).

## Goal

Four small independent correctness primitives that Plan 3 (Fast Failover) depends on. Split out of Plan 1 into its own sub-plan so each can ship and test individually without waiting on the broader HTTP client migration.

Four concrete outcomes:

1. **Explicit role detection.** `detect_role()` called as first runtime action, exports `MAXIM_ROLE` to env, downstream reads from env only. Ends ambiguity about whether a process is leader/peer/solo.
2. **Typed error taxonomy.** `BackendError` hierarchy with `.fix_hint` on every class replaces string-matching. Plan 3's router integration catches these instead of wrestling `str(exc).lower()`.
3. **Two-stage probe.** Liveness + readiness, with `inference_broken` outcome and 15s cache TTL. Plan 3's `_MaximPeerBackend.health_check()` is built on this.
4. **SSRF check lives in utils.** Moved out of `openai_backend.py` to `maxim/utils/net.py`. Both backends import it.

## Non-goals

- **Not touching the HTTP call sites.** That's Plan 1 R1 (already shipped by the time Plan 2 starts).
- **Not building the custom backend.** That's Plan 3.
- **Not migrating existing log calls to JSONL.** See the [logging approach](#logging-approach-dual-format) — existing calls stay unchanged.
- **Not adding per-agent rate limiting.** Plan 4 (Operator Visibility).

## Context — why split from Plan 1

Plan 1 v3 bundled R0 (delete dead mesh), R1 (HTTP client), and R2 (this plan's content) into a single sub-plan ~720 LOC. Per user request, R2 is now its own sub-plan for three reasons:

1. **R2's pieces are independent.** Role detection, typed exceptions, two-stage probe, and SSRF move don't need to ship together. Each is ~40-100 LOC and individually testable.
2. **Tighter testing feedback loops.** Plan 1's hard testing checkpoint is ~4,100 tests. Splitting R2 out means R2 can reuse the existing probe cache + HTTP client test infrastructure without waiting for R1's CI grep enforcement to pass.
3. **Cleaner architecture doc mapping.** The new [architecture/llm_routing.md](../architecture/llm_routing.md) has distinct sections for "how HTTP gets built" (Plan 1) vs "how errors are classified" (Plan 2). Splitting the plans matches the doc structure.

## Phases

### R2a — Explicit role detection — ~80 LOC new

**Problem:** role is currently inferred implicitly in multiple places (`peer/config.py::read_peer_config`, `lane_backends._apply_local_llm_override`, `cli.py::main`). This is how the 2026-04-12 persisted-profile incident happened — the peer config was loaded but the role wasn't set explicitly, so downstream code made inconsistent assumptions.

**Fix:** create `runtime/role.py::detect_role() -> Literal["leader", "peer", "solo"]`. Single source of truth. Inference order:

1. Explicit `MAXIM_ROLE` env var wins
2. `mesh.yml` exists → `peer` (or `leader` if local node's entry has `role: leader`)
3. `peer.yml` exists → `peer` (legacy path)
4. `--llm <local>` CLI flag + no peer config → `solo`
5. Default → `leader`

**Call from `cli.py::main()` as the very first runtime action**, before anything else. Export: `os.environ["MAXIM_ROLE"] = detected_role`. Downstream reads from env only.

**First log line after role detection** logs the detected role + source at INFO:
```python
log_structured(logger, logging.INFO, event="role_detected", data={
    "role": detected_role,
    "source": source,  # "env_var" | "mesh_yml" | "peer_yml" | "cli_flag" | "default"
})
```

This is the first structured event every startup emits. Makes "what role did this process start as?" answerable from the first log line.

**Persisted state split:** rename `~/.maxim/util/active_llm_model.txt` → `active_llm_model.{role}.txt`. Only the matching role writes/reads it. Migration logic on first startup:

- `old_path` exists + peer role → delete (log WARN with old filename)
- `old_path` exists + solo role + CLI flag → rename to `.solo.txt`
- `old_path` exists + leader role → rename to `.leader.txt`
- `old_path` exists + unclear role → rename to `.leader.txt` (conservative default)

Migration test covers all four pre-existing user states.

**Concurrency:** `detect_role()` is called once at startup before any threads spin up. No locking needed.

### R2b — Typed exception taxonomy — ~60 LOC new

**MANDATORY reference shape — Plan 1 R1's HTTPError hierarchy.** R1 shipped the exact pattern this section will mirror, including the three load-bearing access patterns that Plan 3's router integration will consume. Do not invent a parallel pattern — read `src/maxim/utils/http.py` before writing a single line of `types.py`. Matching the HTTPError shape keeps the `except HTTPRateLimited as e: raise BackendOverloaded(e.retry_after_s, ...)` bridge in Plan 3 trivial.

**The three access patterns (match `HTTPError` exactly):**
1. `e.status` — the HTTP status code (int or None for transport failures)
2. `e.response.json()` — parsed error body, attached to the exception by `_classify_status` when the response had a body. Use `e.response` only — do NOT add a parallel `raw_body` attribute.
3. `e.fix_hint` — human-actionable repair string. Class attribute with optional per-instance override.

Extend `models/language/types.py`:

```python
class BackendError(Exception):
    """Base for all backend-raised errors. Mirrors the shape of
    maxim.utils.http.HTTPError — same three access patterns (status,
    response, fix_hint) so Plan 3's router can bridge HTTP → Backend
    exceptions with a simple except/raise pair.
    """
    provider_key: str
    fix_hint: str = ""

    def __init__(self, provider_key: str, *, status: int | None = None,
                 fix_hint: str = "", **kwargs):
        self.provider_key = provider_key
        self.status = status
        if fix_hint:
            self.fix_hint = fix_hint
        self.response: Any | None = None  # attached by router on body-bearing errors
        for k, v in kwargs.items():
            setattr(self, k, v)
        super().__init__(f"{type(self).__name__}[{provider_key}]: {self.fix_hint}")

class BackendOverloaded(BackendError):
    retry_after_s: float = 0.0
    suggested_peer: str | None = None
    queue_depth: int = 0
    fix_hint = "Peer is at capacity. Try a different peer or wait."

class BackendDown(BackendError):
    fix_hint = "Peer is not responding. Run `maxim peer --node <name> status`."

class BackendTimeout(BackendError):
    elapsed_s: float = 0.0
    fix_hint = "Peer exceeded timeout. Check network or MAXIM_LANE_*_REMOTE_TIMEOUT_S."

class BackendAuthFailed(BackendError):
    fix_hint = "Cluster key rejected. Verify mesh.yml::cluster_key matches peer config."

class BackendModelMissing(BackendError):
    requested_model: str = ""
    fix_hint = "Run `maxim peer --node <name> install <model>`."

class BackendInferenceBroken(BackendError):
    """Stage-2 probe failed: listener alive, chat endpoint broken."""
    fix_hint = "Model loading, llama-cpp crashed, or chat template broken. Check peer logs."
```

**Invariants:**
- Every subclass sets `fix_hint` as a class attribute (mutable via `__init__` kwargs if needed for interpolation)
- `fix_hint` content is **never user-controllable** — all strings are static or interpolated from validated identifiers only. Prevents log injection.
- Static test iterates all `BackendError` subclasses and asserts `fix_hint != ""`.
- `status` and `response` are set by the raiser (usually Plan 3's router `_try_provider` bridge), never by caller-supplied kwargs. This matches how `utils/http.py::_classify_status` attaches `response` to raised `HTTPError` subclasses.
- **R1-tested helper pattern**: subclass `__init__` should NOT override `BackendError.__init__` — set class-level defaults and let the base handle kwargs. See `HTTPRateLimited.__init__` in `utils/http.py` for the one acceptable override pattern (explicit named kwargs for documentation purposes).

**Backcompat shim — CANONICAL LOCATION:** `request_context["agent"]` → `request_context["agent_id"]` migration. Plan 1 R1 introduced `RequestContext` as a typed replacement. This plan's R2b adds the **single canonical** `_normalize_request_context` function in `agents/llm_worker.py`. **Plan 3's `_MaximPeerBackend` does not define a parallel shim** — it imports and calls this one. If you see `_build_request_context` in the peer backend, it is a thin wrapper that delegates to this function. One normalization path, one migration owner:

```python
def _normalize_request_context(ctx: dict[str, Any] | None) -> RequestContext:
    if ctx is None:
        return RequestContext(request_id=generate_request_id())
    # Read both legacy "agent" and new "agent_id"
    agent_id = ctx.get("agent_id") or ctx.get("agent")
    return RequestContext(
        request_id=ctx.get("request_id") or generate_request_id(),
        agent_id=agent_id,
        session_id=ctx.get("session_id"),
        lane=ctx.get("lane"),
    )
```

One-minor-version compatibility window. Then drop the `"agent"` key read in 0.5.

### R2c — Two-stage probe — ~100 LOC new

**R1 status:** `probe_llm_server` + `llm_server_responding_at` + `_probe_once` have already been migrated to use `maxim.utils.http.fetch_url` (step 2 of R1, commit `4cb748e`). The pre-R1 urllib implementation no longer exists. R2c adds the stage-2 readiness probe on top of the already-migrated stage-1 probe.

**R1 already built** `_classify_probe_cause(exc, fallback)` — walks the `__cause__` chain of a caught `HTTPConnectionError` looking for `socket.gaierror` / `ssl.SSLError` / `TimeoutError` / `ConnectionRefusedError`, returning the coarse `ProbeOutcome` literal that downstream `lane_backends._validate_remote_urls` expects. **R2c must NOT re-invent this** — it stays in place until Plan 3 R2.5's `_MaximPeerBackend.health_check()` supersedes the whole function.

**What R2c adds:** stage-2 readiness probe (micro-completion) + `BackendInferenceBroken` outcome + per-outcome cache TTL table. The plumbing from `http.fetch_url` to `ProbeResult` is already there.

Does **not** delete `probe_llm_server` yet — that happens in Plan 3 after `_MaximPeerBackend.health_check()` takes over.

**Stage 1 — liveness:** `GET /v1/models` via Plan 1's `http.get("leader", "/models", context=...)`, 1.5s timeout. Classifies into existing `ProbeOutcome` plus new `inference_broken`.

**Stage 2 — readiness:** runs only if stage 1 returned `ok`. Micro-completion via `http.post("leader", "/chat/completions", json={...}, context=...)`:
```python
{
    "model": model_name,
    "messages": [{"role": "user", "content": "."}],
    "max_tokens": 1,
    "temperature": 0.0,
}
```
3s timeout. Success = inference verified. Failure = new outcome `inference_broken`.

**Cache TTL by outcome** (extend `probe_cache.py`):
- `ok` → 60s (current default)
- `auth_rejected` → 60s (listener alive, user needs to fix key)
- `inference_broken` → **15s** (retry sooner — might be mid-load)
- `http_5xx`, `timeout`, `connection_refused`, `dns_fail` → 60s

**Load-bearing shared constant:** the 15s `inference_broken` TTL must match the `BackendInferenceBroken` backoff used in Plan 3's router integration. These two values must not drift. Plan 2 R2b exports a single constant in `models/language/types.py`:

```python
INFERENCE_BROKEN_BACKOFF_S: float = 15.0  # single source of truth
```

Both `probe_cache.py::CACHE_TTL_BY_OUTCOME` and `router.py::_set_short_backoff` (Plan 3 addition) import this constant. A Plan 2 test asserts the import + value; Plan 3's integration test asserts both sites use the same value. Changing it in one place without the other is impossible.

**Fallback safety:** if stage 2 throws a non-HTTP exception (JSON parse error, library crash), treat stage 1's `ok` as final + log warning. Probe fragility can't make the whole system look dead.

**Probe cache corruption handling:** `probe_cache.load_cache` already catches `JSONDecodeError` + `OSError` and returns empty dict (verified at [probe_cache.py:72](../../src/maxim/runtime/probe_cache.py#L72)). **Current log level is `debug`; Plan 2 promotes to `warning`** — a corrupted cache file is operationally significant enough that operators should see it without needing `-v`. ~5 LOC change.

**Two-stage probe emits structured events:**
```python
log_structured(logger, logging.INFO, event="probe_started", data={
    "endpoint": endpoint_name,
    "stage": "liveness",
    "cached": False,
})
log_structured(logger, logging.INFO, event="probe_completed", data={
    "endpoint": endpoint_name,
    "stage": "both",
    "outcome": "ok",
    "liveness_ms": 45,
    "readiness_ms": 320,
})
```

### R2d — SSRF check moves to utils — ~40 LOC new

Move `_validate_base_url` from `models/language/openai_backend.py` (lines 109-137) to `maxim/utils/net.py`. Import unchanged from `_OpenAIBackend`; Plan 3's `_MaximPeerBackend` also imports it.

The function validates:
- `https://` required for public endpoints
- `http://` acceptable for private-IP LAN servers (llama-cpp-server on 10.x.x.x, etc.)
- Host resolves to a private IP when `allow_local=True`, else public only
- Rejects redirects to different hosts (prevents SSRF via redirect)

**New in this plan:** add a test fixture `tests/unit/test_net_ssrf.py` with a full coverage sweep. Previously this logic was tested indirectly via `_OpenAIBackend` tests; now it's directly tested as a utility.

## Logging approach — dual format

**Critical finding from the grep audit:** `src/maxim/utils/structured_logging.py` already exists with a `StructuredFormatter` outputting compact JSON, plus a `log_structured(logger, level, event, data)` helper. The existing infrastructure is what Plans 1-4 should use.

**Dual-format strategy:**

- **stdout:** human-readable format (existing `DEFAULT_FORMAT` from `utils/logging.py`) — unchanged. Users running `maxim --sim` still see readable output.
- **file (optional, enabled via `MAXIM_LOG_FILE=/path/to/log.jsonl`):** JSONL via `StructuredFormatter`. Machine-parseable. Agents can grep/jq.
- **New log calls from Plans 1-4** use `log_structured(logger, level, event=..., data={...})`. The function already exists and routes to both the human console log AND the JSONL buffer.
- **Existing ~1,450 `logger.info/warning/error` calls stay unchanged.** They continue to emit human-readable lines. The JSONL file captures them as generic `event="log"` entries (via `StructuredFormatter.format`).

**No mass migration of existing log calls.** Plans 1-4 add ~50-80 new structured log events total. That's the migration scope. Existing calls are compatible because they flow through the same logger; the `StructuredFormatter` just wraps them with minimal metadata.

**Existing env var alignment:** `MAXIM_AGENTIC_VERBOSITY` already exists in `structured_logging.py` (0=quiet, 1=normal, 2=verbose, 3=debug). Plan 4's `MAXIM_MESH_VERBOSITY` follows the same pattern with the same numeric range. This matches the project's existing convention.

**What this means for the plan:**
- Plan 1 R1 still introduces `maxim/utils/http.py` — but log events use `log_structured()` from the existing module, not a new formatter
- Plan 2 R2 (this plan) emits ~10 new structured events (role_detected, probe_started/completed, exception paths)
- Plan 3 R2.5/R2.6 emits ~15 new structured events (backend_call, backend_failed, stream_start, etc.)
- Plan 4 R3.x emits ~20 new structured events (dispatch, admin, rate_limit, drain)
- **Total new structured events: ~45-50.** Very manageable migration.

**Grep check commitment:** Plan 1's hard testing checkpoint includes a regression check that existing log output is byte-compatible with pre-plan output (for the first 1000 lines of a reference sim run). This catches accidental format drift.

## Logging & verbosity requirements (this plan's contribution)

**New structured events introduced in R2a-R2d:**

```python
# R2a
"role_detected"        # role + source
"persisted_model_migrated"  # old_path + new_path + role
# R2b  
"backend_error_classified"  # exception_class + fix_hint + provider_key
# R2c
"probe_started"        # endpoint + stage + cached
"probe_completed"      # endpoint + stage + outcome + liveness_ms + readiness_ms
"probe_cache_corrupt"  # path + error (when cache file unreadable)
"probe_stage2_fallback" # endpoint + reason (when stage 2 crashes unexpectedly)
# R2d
"ssrf_rejected"        # url + reason (allow_local=False, private IP, etc.)
```

All use `log_structured()` — no new formatter, no new infrastructure.

## Multi-agent lens (applied to R2)

**Role detection is process-scoped, not agent-scoped.** One process = one role. Multiple agents under that process share the role. This is correct for single-tenant deployments.

**Typed exceptions carry `provider_key` but not `agent_id`.** The `agent_id` is added to log lines via the contextvar + `log_structured()` at the call site, not embedded in the exception itself. Keeps the exception lightweight and serializable.

**Two-stage probe is cluster-scoped, not agent-scoped.** A probe runs against an endpoint, not on behalf of an agent. Probe cache is shared across agents.

**SSRF check is stateless.** No multi-agent consideration.

## Success criteria — R2a/b/c/d

**R2a:**
- `detect_role()` unit-tested for all five decision paths
- First structured log event at startup is `role_detected`
- Persisted-model migration test covers all four pre-existing user states
- `grep -r "peer.yml" src/maxim/` shows no direct reads outside `runtime/role.py` + `peer/config.py`

**R2b:**
- All `BackendError` subclasses have non-empty `fix_hint` (auto-test)
- Backcompat shim reads both `"agent"` and `"agent_id"` keys
- Exception repr includes `fix_hint` + provider_key

**R2c:**
- Two-stage probe test: killed chat endpoint + live `/v1/models` → probe reports `inference_broken` with 15s TTL
- Non-HTTP probe exception test: graceful fallback to stage-1 `ok` + warning log
- Corrupted probe cache test: `probe_cache.json` with malformed JSON → empty cache + warning
- Structured events `probe_started` / `probe_completed` emitted

**R2d:**
- SSRF check works from `utils/net.py` location
- Test sweep: `http://public`, `http://private+allow_local`, `https://public`, redirect to different host, malformed URL
- `_OpenAIBackend` still uses the function via import (no regression)

## Hard testing requirement

**Non-negotiable before Plan 3 begins.**

**Automated:**
```bash
# Full fast suite
python -m pytest tests/ -x -q -m "not slow" --ignore=tests/integration/test_memory_hub.py

# Targeted R2 tests
python -m pytest tests/unit/test_role_detection.py -v
python -m pytest tests/unit/test_backend_error_taxonomy.py -v
python -m pytest tests/unit/test_two_stage_probe.py -v
python -m pytest tests/unit/test_net_ssrf.py -v
python -m pytest tests/unit/test_probe_cache_corruption.py -v
python -m pytest tests/unit/test_persisted_model_migration.py -v

# Mypy
mypy src/maxim/runtime/role.py src/maxim/models/language/types.py \
     src/maxim/utils/net.py --ignore-missing-imports

# Lint + format
ruff check src/maxim/runtime/role.py src/maxim/utils/net.py \
           src/maxim/models/language/types.py
ruff format --check src/maxim/runtime/role.py src/maxim/utils/net.py \
                    src/maxim/models/language/types.py

# Every BackendError subclass has a fix_hint
python -c "
from maxim.models.language.types import BackendError
import inspect
for cls in BackendError.__subclasses__():
    fix_hint = getattr(cls, 'fix_hint', None)
    assert fix_hint, f'{cls.__name__} missing fix_hint'
    print(f'{cls.__name__}: {fix_hint[:60]}...')
"
```

**Manual smoke:**
```bash
# Verify first log line is role_detected
rm -f ~/.maxim/util/active_llm_model.txt
maxim --sim "test" 2>&1 | head -1
# Expected: line contains event=role_detected (in whatever format your logging uses)

# Verify persisted model migration
touch ~/.maxim/util/active_llm_model.txt
maxim doctor  # triggers migration
ls ~/.maxim/util/active_llm_model*  # should show .{role}.txt, not .txt

# Two-stage probe manual check
# (requires a leader to be running — test on existing infrastructure)
maxim doctor 2>&1 | grep -E "probe|liveness|readiness"
```

**Rollback drill:** each R2 sub-phase is a separate commit. Revert individually if issues arise.

## Documentation & memory update

**1. Update [../reference.md](../reference.md):**
- **"Role detection"** section: `runtime/role.py::detect_role()` as single source of truth
- **"Error taxonomy"** section: `BackendError` hierarchy with `fix_hint` convention
- **"Two-stage probe"** section: liveness vs readiness, `inference_broken` outcome, cache TTL map
- **"Network utilities"** section: SSRF check location

**2. Update [../architecture/llm_routing.md](../architecture/llm_routing.md):**

Extends the architecture doc drafted in Plan 1 (R0 commit). Adds:
- "Error taxonomy" section with the full `BackendError` diagram
- "Probe lifecycle" section with two-stage flow + cache TTL decisions
- "Role detection flow" section with the 5-step decision tree

**3. Update [../../CLAUDE.md](../../CLAUDE.md):**

- **Lessons learned:**
  > **Role detection is the first runtime action.** `cli.py::main()` calls `detect_role()` before anything else, exports `MAXIM_ROLE`, and downstream code reads from env only. Never re-detect role; never infer role from `peer.yml` existence in downstream code.
  > 
  > **`BackendError.fix_hint` is never user-controllable.** Subclasses may interpolate validated identifiers (model names, URLs) into hint strings, but the format strings themselves are always static. Prevents log injection via user-controlled exception content.

**4. Update project memory:**

Add `project_llm_path_typed_errors_shipped.md` with:
- What shipped (role detection, typed errors, two-stage probe, SSRF move)
- Why it was split from Plan 1
- How it differs from the legacy string-matching error classification
- Load-bearing invariants for future refinement

## Migration notes

- `active_llm_model.txt` → `active_llm_model.{role}.txt` auto-migration. Operators don't need to do anything.
- `request_context["agent"]` → `request_context["agent_id"]` backcompat shim. Existing agent code continues to work; new code uses `agent_id`.
- Existing string-matching error paths still work (they hit the generic `except Exception` safety net in Plan 3's router integration). Typed classification is additive.

## Related docs

- **Previous plan:** [llm_path_foundation.md](llm_path_foundation.md) — Plan 1 (prerequisite)
- **Next plan:** [llm_path_fast_failover.md](llm_path_fast_failover.md) — Plan 3
- **Meta plan:** [llm_path_refinement.md](llm_path_refinement.md)
- **Architecture:** [../architecture/llm_routing.md](../architecture/llm_routing.md)
- **Project guide:** [../../CLAUDE.md](../../CLAUDE.md)
