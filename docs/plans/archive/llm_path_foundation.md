# LLM Path Refinement — Plan 1: Foundation Cleanup

**Status:** ✅ **SHIPPED 2026-04-12**. R0 as commit `e811787`. R1 across PRs #88 (step 1) + #90 (steps 2-9) + pending cleanup PR for commit `c8a07e9` (dual-format logging fix). All 4003 fast-suite tests passing, CI grep invariant clean. See [project_llm_path_r1_shipped.md](../../.claude/projects/-Users-dennyschaedig-Scripts-Maxim/memory/project_llm_path_r1_shipped.md) for the 5 design divergences, 10 gotchas, and load-bearing invariants for Plan 2+.
**Scope:** ~450 LOC new + ~1,330 LOC deleted (as shipped, close to spec)
**Target version:** 0.4 (single stability version containing all LLM path sub-plans)
**Part of:** [llm_path_refinement.md](llm_path_refinement.md) — the LLM path refinement meta-plan
**Unblocks:** [llm_path_typed_errors.md](llm_path_typed_errors.md) (Plan 2) — READY TO START

## Goal

Delete dead code and collapse eleven scattered HTTP call sites into one registry-backed module. No user-visible behavior changes — this plan is pure refactoring + structural observability.

**Two concrete outcomes:**

1. **One HTTP client, one place for bugs.** Eleven `urllib.request` call sites collapse into `maxim/utils/http.py` with an endpoint registry, connection pooling, typed errors, and automatic multi-agent header propagation. The 2026-04-12 Cloudflare User-Agent incident becomes structurally impossible.
2. **Dead mesh scaffolding deleted.** ~1,250 LOC of unused modules (`peer_registry`, `peer_info`, `peer_channel`, `task_delegation`, `knowledge`, `clock`, `agent_identity`, `admission`) removed. The remaining `bus.py`, `identity.py`, `message.py`, `naming.py` are simulation-only and stay.

**Not in this plan (split to other sub-plans):**
- Role detection + typed error taxonomy + two-stage probe + SSRF move → [llm_path_typed_errors.md](llm_path_typed_errors.md) (Plan 2)
- `_MaximPeerBackend` custom backend → [llm_path_fast_failover.md](llm_path_fast_failover.md) (Plan 3)
- `mesh.yml` + admin API + per-agent rate limiting → [llm_path_operator_visibility.md](llm_path_operator_visibility.md) (Plan 4)

## Non-goals

- **Not changing the router's provider fallback loop.** The audit confirmed it already does reactive fallback correctly.
- **Not touching `_OpenAIBackend`.** Cloud providers keep existing behavior.
- **Not migrating existing log calls to JSONL.** ~1,450 existing `logger.info/warning` calls stay unchanged. New events from Plans 1-4 use `log_structured()` from the **existing** `utils/structured_logging.py` module. See [logging approach](#logging-approach-dual-format).
- **Not introducing runtime behavior changes.** After this plan, `maxim --sim` runs behave identically.

## Context

Two incidents on 2026-04-12 exposed structural fragility:
1. Stale persisted profile clobbered peer config — commit `d875fb9`
2. Missing `User-Agent` header in `probe_llm_server` — commit `8b52cbd`

An architecture audit found ~1,250 LOC of dead mesh scaffolding — zero production imports, swept along during the task→size lane refactor and never re-wired.

**Multi-agent lens finding:** the existing `request_context` dict carries `agent` but doesn't propagate it to HTTP headers, log lines, or metrics. Under AgentPool workloads, "which agent's request went slow?" is nearly un-answerable. Plan 1 closes this gap as a structural requirement via `RequestContext` + `contextvars.ContextVar`.

**Logging finding (from grep audit):** `src/maxim/utils/structured_logging.py` already exists with a `StructuredFormatter` (JSON output) and `log_structured()` helper. **Plans 1-4 use this existing infrastructure, not a new formatter.** Existing ~1,450 log calls stay as-is.

## Phases

### R0 — Delete dead mesh scaffolding — ~-1,250 LOC

**Delete** (verified via zero production imports):
- `src/maxim/mesh/peer_registry.py`
- `src/maxim/mesh/peer_info.py`
- `src/maxim/mesh/peer_channel.py`
- `src/maxim/mesh/task_delegation.py`
- `src/maxim/mesh/knowledge.py`
- `src/maxim/mesh/clock.py`
- `src/maxim/mesh/agent_identity.py`
- Corresponding lines in `src/maxim/mesh/__init__.py`

**First cherry-pick `mesh/admission.py`** per-peer rate-limiting logic into `runtime/rate_limit.py` — ~150 LOC of solid code that will serve Plan 4's admin API + per-agent rate limiting. Then delete `admission.py`.

**Keep unchanged:** `bus.py`, `identity.py`, `message.py`, `naming.py` — simulation-only consumers in `simulation/` and `create.py`.

**Verification:**
```bash
# Zero production imports of deleted modules
! grep -rE "from maxim.mesh.(peer_registry|peer_info|peer_channel|task_delegation|knowledge|clock|agent_identity|admission)" src/maxim/
# Should return nothing
```

**Documentation audit:** `docs/troubleshooting/mesh.md` references `PeerRegistry.get_peer()` and `PeerChannel`. Add a "LEGACY" header to the file noting R0 deleted these classes. Plan 4 eventually rewrites this as `mesh_debug.md` with the new operator runbook.

**Forward reference in `tool_refinement_plan.md`** (line 98) points at deleted `PeerRegistry`. Update to "Phase 7c (requires Plan 4's mesh.yml — see llm_path_operator_visibility.md)."

### R1 — `maxim/utils/http.py` unified HTTP client — ~450 LOC new

The foundation for everything else.

**Module shape:**

```python
# maxim/utils/http.py

@dataclass(frozen=True)
class TimeoutPolicy:
    connect_s: float = 3.0
    read_s: float = 30.0
    total_s: float = 60.0

@dataclass(frozen=True)
class HTTPEndpoint:
    name: str
    base_url: str | None
    default_headers: Mapping[str, str]
    auth_provider: Callable[[], str | None] | None  # late-bound bearer token
    timeouts: TimeoutPolicy
    max_pool_connections: int = DEFAULT_POOL_PER_ENDPOINT

DEFAULT_POOL_PER_ENDPOINT = 10
# Concurrent-dispatch ceiling: N peers × DEFAULT_POOL_PER_ENDPOINT
# = max simultaneous outbound HTTP calls from one node. Metric
# http_pool_exhausted_total fires when a call waits for a pool slot.

@dataclass(frozen=True)
class RequestContext:
    """Multi-agent routing + observability context.
    
    Set via contextvars.ContextVar at request boundary. Read automatically
    by the logging formatter + HTTP client. Callers don't have to thread
    this through function signatures.
    """
    request_id: str              # auto-generated if absent
    agent_id: str | None = None  # canonical key (legacy "agent" is aliased)
    session_id: str | None = None
    lane: str | None = None
    parent_request_id: str | None = None  # for fan-out tracing

# Contextvar for request-scoped context
_current_context: ContextVar[RequestContext | None] = ContextVar("maxim_request_context", default=None)

def set_context(ctx: RequestContext) -> None: ...
def current_context() -> RequestContext | None: ...

class HTTPError(Exception):
    endpoint: str
    status: int | None
    fix_hint: str

class HTTPTimeout(HTTPError): ...
class HTTPConnectionError(HTTPError): ...
class HTTPAuthError(HTTPError): ...      # 401/403
class HTTPServerError(HTTPError): ...    # 5xx
class HTTPClientError(HTTPError): ...    # 4xx (other)
class HTTPRateLimited(HTTPError):
    retry_after_s: float
    suggested_peer: str | None

ENDPOINT_REGISTRY: dict[str, HTTPEndpoint] = {}

def register_endpoint(endpoint: HTTPEndpoint) -> None: ...
def get(name: str, path: str = "", *, context: RequestContext | None = None, **kwargs) -> Response: ...
def post(name: str, path: str = "", json: Any = None, *, context: RequestContext | None = None, **kwargs) -> Response: ...
def stream(name: str, path: str = "", *, context: RequestContext | None = None, **kwargs) -> Iterator[bytes]: ...
```

**Automatic `X-Maxim-*` header propagation** from `RequestContext`:
- `X-Maxim-Request-Id: <request_id>` (always)
- `X-Maxim-Agent-Id: <agent_id>` (when set)
- `X-Maxim-Session-Id: <session_id>` (when set)
- `X-Maxim-Lane: <lane>` (when set)
- `X-Maxim-Parent-Request-Id: <parent_request_id>` (when set)
- `X-Maxim-Protocol-Version: 1` (always)

These are the wire protocol between nodes. Documented in [../architecture/llm_routing.md](../architecture/llm_routing.md) as a versioned contract.

**Input sanitization at boundary:** every header value passes through a sanitizer that rejects control chars, CR/LF (log injection risk), non-ASCII bytes, and lengths > 256. Bad values raise `HTTPClientError` with a `fix_hint` pointing at the offending context field.

**Why a registry, not helper functions:**
- Headers (`User-Agent: maxim-peer/1.0`) set **once** at registration — `grep 'User-Agent' src/` returns one match after this plan, not twelve
- Auth is late-bound — cluster key rotation doesn't touch call sites
- Timeouts typed + per-endpoint
- Connection pools bounded per endpoint
- Tests mock one registry, not 11 urllib patches
- CI grep enforces: `grep -r "urllib.request.urlopen" src/maxim/ | grep -v utils/http.py` returns zero

**Backend:** `httpx`. Already a transitive dep via `openai`. Consistent with what inference uses. Better error taxonomy than urllib.

**Migration order — hot path LAST:**
1. Register the `leader` endpoint in `peer/config.py::apply_peer_config_to_env`
2. Migrate `llm_server.py::probe_llm_server` (will be replaced in Plan 3 but migrate first)
3. Migrate `doctor/checks.py` + `doctor/cli.py` (low blast radius)
4. Migrate `tools/http_fetch.py` + `tools/internet_search.py` (isolated)
5. Migrate `api.py` (auth token verify)
6. Migrate `mesh_trace.py`, `peer/cli.py`
7. Migrate `models/download.py` (uses `http.stream`)
8. Migrate `local_server_spawner.py` readiness check
9. Migrate `leader_proxy.py::_proxy_request` **last**

Each migration is a separate commit with its own tests. Old urllib imports stay until step 9 is fully tested. No feature flag for the HTTP client swap — rollback is `git revert` of the specific migration commit.

**Parallelizability:** step 1 (leader endpoint registration) MUST go first — subsequent migrations import from the registry it populates. Step 9 (leader_proxy) MUST go last — it's the critical inference path and carries the highest blast radius. **Steps 2–8 can run in any order or in parallel**: each touches a different file, none depend on each other, and they can be divided across parallel Claude sessions or batched by one agent. A practical split is "diagnostics + tools" (steps 3-5) in one session, "download + spawner" (steps 6-8) in another. Don't attempt steps 2-8 as a single atomic commit — one-file-per-commit keeps rollback clean.

## Logging approach — dual format

**Critical finding from the grep audit:** `src/maxim/utils/structured_logging.py` already exists with a `StructuredFormatter` (JSON output) + `log_structured()` helper. Plans 1-4 use this existing infrastructure, not a new formatter.

**Dual-format strategy:**

- **stdout:** human-readable format (existing `DEFAULT_FORMAT` from `utils/logging.py`) — **unchanged**. Users running `maxim --sim` still see readable output.
- **file (optional via `MAXIM_LOG_FILE=/path/to/log.jsonl`):** JSONL via `StructuredFormatter`. Machine-parseable. Agents can grep/jq.
- **New log events from Plans 1-4** use `log_structured(logger, level, event=..., data={...})`. Routes to both console (human) AND file (JSONL).
- **Existing ~1,450 `logger.info/warning/error` calls stay unchanged.** They continue to emit human-readable lines. JSONL file captures them with generic `event="log"` metadata.

**No mass migration.** Plans 1-4 add ~50 new structured log events total.

**Existing `MAXIM_AGENTIC_VERBOSITY` env var** (0-3) already exists. Plan 4's `MAXIM_MESH_VERBOSITY` follows the same pattern.

**Plan 1's contribution to logging:**

New structured events via `log_structured()`:
```python
"startup_phase"       # per-phase duration at startup
"http_request"        # every outbound call (DEBUG, INFO at MAXIM_HTTP_TRACE=1)
"http_request_failed" # WARN on HTTP errors with typed class + fix_hint
```

**Startup phase timer** — `cli.py::main` logs phase durations:
```json
{"ts":"...","level":"INFO","event":"startup_phase","phase":"http_registry","duration_ms":8}
{"ts":"...","level":"INFO","event":"startup_phase","phase":"peer_config","duration_ms":43}
{"ts":"...","level":"INFO","event":"startup_phase","phase":"ready","total_duration_ms":1680}
```

Makes "what's slow at startup" answerable via `jq` on the JSONL file.

**Metrics emitted** (added to existing `lane_metrics.metrics_snapshot()`):
- `http_requests_total{endpoint, status}` counter
- `http_latency_seconds{endpoint}` histogram
- `http_pool_in_use{endpoint}` gauge
- `http_pool_exhausted_total{endpoint}` counter
- `startup_phase_duration_seconds{phase}` histogram

No `agent_id` on metric labels — bounded cardinality.

## Env var inventory (from this plan)

Adds two env vars. Full six-var inventory is in [llm_path_refinement.md](llm_path_refinement.md).

| Var | Default | Purpose |
|---|---|---|
| `MAXIM_HTTP_TRACE` | 0 | Verbose HTTP logging (INFO level + full headers + payload sizes) |
| `MAXIM_LOG_FILE` | unset | Path for JSONL structured log file (optional) |

Both documented in CLAUDE.md env var table. `MAXIM_HTTP_TRACE` has a conftest autouse scrub.

## Success criteria — R0 + R1

**R0:**
- Zero production imports of deleted mesh modules (CI grep)
- `docs/troubleshooting/mesh.md` has LEGACY header
- Fast suite (~4,100 tests) stays green
- Fast suite runtime unchanged ±5%

**R1:**
- Zero `urllib.request.urlopen` outside `utils/http.py` (CI grep)
- `http_requests_total`, `http_latency_seconds`, `http_pool_in_use`, `http_pool_exhausted_total` all exposed
- Startup phase timer logs visible as `event=startup_phase` entries
- `RequestContext` contextvar plumbing works: logging formatter pulls agent_id automatically
- Every outbound HTTP call includes `X-Maxim-Request-Id` + `X-Maxim-Protocol-Version: 1` headers; `X-Maxim-Agent-Id` when context is set
- Header sanitizer test: control chars, CR/LF, non-ASCII all rejected
- No regression in existing probe / download / tool suites
- Manual smoke: `maxim doctor` runs clean

## Hard testing requirement (checkpoint before Plan 2)

**Non-negotiable.**

**Automated:**
```bash
# Full fast suite
python -m pytest tests/ -x -q -m "not slow" --ignore=tests/integration/test_memory_hub.py

# Targeted HTTP client tests
python -m pytest tests/unit/test_http_client.py -v
python -m pytest tests/unit/test_request_context.py -v
python -m pytest tests/unit/test_header_sanitization.py -v

# Integration
python -m pytest tests/integration/test_memory_hub.py -q

# Mypy
mypy src/maxim/utils/http.py --ignore-missing-imports

# Lint + format
ruff check src/ tests/
ruff format --check src/ tests/

# CI grep enforcement
! grep -r "urllib.request.urlopen" src/maxim/ | grep -v "utils/http.py"
! grep -rE "from maxim.mesh.(peer_registry|peer_info|peer_channel|task_delegation|knowledge|clock|agent_identity)" src/maxim/

# JSONL format sanity (if MAXIM_LOG_FILE is set)
MAXIM_LOG_FILE=/tmp/maxim-r1-test.jsonl maxim --sim "quick test" 2>&1
cat /tmp/maxim-r1-test.jsonl | python -c "
import sys, json
count = 0
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    obj = json.loads(line)  # will raise on malformed JSON
    assert 'event' in obj, f'Missing event field: {obj}'
    count += 1
print(f'Validated {count} JSONL log lines.')
"

# Request context propagation check — every http_request event has request_id
jq -c 'select(.event=="http_request") | .request_id' /tmp/maxim-r1-test.jsonl | \
  awk 'NF==0 || $0=="null" {print "MISSING request_id"; exit 1}'
```

**Manual smoke:**
```bash
rm -f ~/.maxim/util/probe_cache.json

# Leader smoke
MAXIM_HTTP_TRACE=1 MAXIM_LOG_FILE=/tmp/maxim-leader.jsonl maxim --sim "A cyberpunk heist" --seed 42

# Peer smoke
MAXIM_HTTP_TRACE=1 MAXIM_LOG_FILE=/tmp/maxim-peer.jsonl maxim --sim "A cyberpunk heist" --seed 42

# Verify log format looks right
head -20 /tmp/maxim-peer.jsonl | jq -c '{ts, event, request_id, agent_id}'
```

**Rollback drill:**
```bash
# Revert the most recent R1 migration commit, verify system still works
git revert HEAD
maxim --sim "rollback test"
git revert HEAD  # restore
```

**Only proceed to Plan 2 (Typed Errors) after all checkboxes green.**

## Documentation & memory update

**1. Update [../reference.md](../reference.md):**
- **"HTTP client"** section: `maxim/utils/http.py`, registry pattern, `RequestContext`, header contract, SSRF check reference
- **"Removed: dead mesh scaffolding"** paragraph explaining R0

**2. Architecture doc** ([../architecture/llm_routing.md](../architecture/llm_routing.md)) is **drafted in the R0 commit** as the first action of this plan. R1 extends it incrementally as each migration step lands. By Plan 1 completion, the HTTP client layer is fully documented.

**3. Update [../../CLAUDE.md](../../CLAUDE.md):**
- **Env var table:** add `MAXIM_HTTP_TRACE`, `MAXIM_LOG_FILE`
- **Lessons learned:**
  > **HTTP call sites must use `maxim/utils/http.py`:** raw urllib is banned in `src/maxim/` (CI grep enforced). The 2026-04-12 Cloudflare incident was a missing `User-Agent` header in one of eleven scattered call sites. The registry sets headers once at endpoint registration.
  > 
  > **`RequestContext` propagates via `contextvars.ContextVar`.** Set at the request boundary; read automatically by the logging formatter and HTTP client. Don't thread it through function signatures.

**4. Create [../troubleshooting/http_debugging.md](../troubleshooting/http_debugging.md):**

Short runbook (~100 lines):
- Enable `MAXIM_HTTP_TRACE=1` + `MAXIM_LOG_FILE=...`
- `jq` queries: startup phase durations, failing endpoints, per-agent latency (when Plan 2 ships agent_id)
- `http_pool_exhausted_total > 0` interpretation
- Cloudflare 403 diagnosis
- Rolling back via git revert

**5. Update existing troubleshooting docs** affected by R0 deletions:
- `docs/troubleshooting/mesh.md` → add LEGACY header
- `docs/troubleshooting/peer_diagnosis_runbook.md` → audit for `mesh.*` references
- `docs/troubleshooting/peer_leader_connectivity.md` → audit
- `docs/troubleshooting/leader_proxy_debug.md` → cross-link to new `http_debugging.md`

**6. Update project memory:**

Add `project_llm_path_foundation_shipped.md`:

```markdown
---
name: LLM path foundation shipped
description: R0 + R1 — dead mesh deleted, unified HTTP client, RequestContext contract
type: project
---

**Shipped:** <date> as part of [llm_path_foundation.md](docs/plans/llm_path_foundation.md).

**What changed:**
- `src/maxim/mesh/` lost ~1,250 LOC of dead scaffolding
- New `maxim/utils/http.py` is the ONLY place urllib/httpx calls live (CI enforced)
- New `RequestContext` dataclass + contextvars propagate agent_id/session_id/request_id automatically
- Structured logging via existing `utils/structured_logging.py` — new events from Plans 1-4 use `log_structured()`
- `X-Maxim-*` header contract documented as versioned wire protocol

**Why this matters:**
- Plan 2 (typed errors) builds on this foundation
- Plan 3 (fast failover) builds `_MaximPeerBackend` on this HTTP client
- Plan 4 (operator visibility) builds admin API on this HTTP client
- The 2026-04-12 Cloudflare incident is structurally impossible now

**Load-bearing invariants:**
- NEVER add urllib call sites. Use `maxim/utils/http.py`.
- NEVER log without `RequestContext` set at the request boundary — the formatter pulls from contextvar automatically
- NEVER rename `X-Maxim-*` headers without a protocol version bump
```

Update `MEMORY.md` with pointer.

**Documentation step ships when:**
- `docs/reference.md` has HTTP client + removed-mesh sections
- `docs/architecture/llm_routing.md` covers the HTTP client + RequestContext layers
- `CLAUDE.md` has new env vars + lessons
- `docs/troubleshooting/http_debugging.md` exists
- Legacy `mesh.md` has the LEGACY header
- Project memory file exists, linked from `MEMORY.md`

## Migration notes

- `MAXIM_HTTP_TRACE=1` is the new debug env var
- `MAXIM_LOG_FILE=path/to/log.jsonl` enables JSONL file output (optional)
- No config file schema changes
- No CLI flag changes
- Existing `logger.info/warning` calls unchanged — JSONL dual-format means human readers and machine parsers both work

## Related docs

- **Meta plan:** [llm_path_refinement.md](llm_path_refinement.md)
- **Next plan:** [llm_path_typed_errors.md](llm_path_typed_errors.md) — Plan 2
- **Architecture:** [../architecture/llm_routing.md](../architecture/llm_routing.md)
- **Project guide:** [../../CLAUDE.md](../../CLAUDE.md)
- **Related incident commits:** `d875fb9` + `8b52cbd` on main
