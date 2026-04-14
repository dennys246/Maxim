# LLM Path Refinement — Plan 4: Operator Visibility

**Status:** Draft v3 — renumbered 2026-04-12 after Plan 2 (Typed Errors) split out of former Plan 1
**Scope:** ~650 LOC new
**Target version:** 0.4 (single stability version)
**Part of:** [llm_path_refinement.md](llm_path_refinement.md)
**Depends on:**
- [llm_path_fast_failover.md](llm_path_fast_failover.md) (Plan 3) — typed-exception router loop ✅ shipped
- [llm_path_cancellation_hygiene.md](llm_path_cancellation_hygiene.md) (Plan 3.5) — "HTTP fires first" contract ✅ shipped
- [llm_path_peer_failover.md](llm_path_peer_failover.md) (Plan 3.6) — multi-leader `peer.yml` precursor (not strictly required but recommended; `mesh.yml` is the canonical successor)
**Note:** renamed from "Reactive Mesh" in v1. Multi-peer dispatch moved to [deferred/llm_path_multi_peer_dispatch.md](deferred/llm_path_multi_peer_dispatch.md). Capability-aware ranking is [deferred/llm_mesh_capability_aware.md](deferred/llm_mesh_capability_aware.md).
**Bake-in target (2026-04-13):** the user's RTX 5080 + RTX 3070 setup is the concrete two-node deployment for testing `mesh.yml`'s schema validation, drain/resume, per-node admin endpoints, and per-agent rate limiting. Plan 3.6 unblocks failover testing without waiting for the full Plan 4 admin API.

## Goal

Give operators the tools to inspect, manage, and debug the LLM routing path across one or more nodes. **Without** committing to multi-peer reactive overflow (which moved to deferred per user decision based on the stress test results).

Four concrete outcomes:

1. **User-facing mesh contract.** `maxim peer list-nodes`, `--node X status`, `install`, `drain/resume`. The cluster stops being a black box even when there's only one peer.
2. **Observable dispatch.** Every routing decision is recorded in a ring buffer and accessible via `GET /v1/mesh/request-trace/<req_id>`. Per-agent filtering via `?agent_id=X`. Debugging "why did this request go there and take that long?" takes minutes, not hours.
3. **Per-agent accounting.** The leader tracks per-agent in-flight counts, total counts, p50/p99 latency. Exposed via admin API for debugging + future fair-share scheduling.
4. **Runaway agent protection.** Simple per-agent token-bucket rate limiting prevents one misbehaving agent from starving others. Configurable via `mesh.yml`.

## Non-goals

- **Not multi-peer reactive overflow.** Moved to [deferred/llm_path_multi_peer_dispatch.md](deferred/llm_path_multi_peer_dispatch.md) per user decision + stress test results. R3.0 + R3.5-lite + R3.6-lite ship unconditionally; multi-peer dispatch extends later when triggered.
- **Not proactive load-aware routing.** Deferred to [deferred/llm_mesh_capability_aware.md](deferred/llm_mesh_capability_aware.md).
- **Not auto-download.** Explicit `install` only.
- **Not per-node JWT auth.** Shared cluster key.
- **Not fair-share scheduling across agents.** Rate limiting is a hard cap per agent; fair-share is [deferred/llm_path_fair_scheduling.md](deferred/llm_path_fair_scheduling.md).
- **Not async router.** Concurrent per-lane agent calls still serialize under `_inference_lock`. Plan 3 shortens the worst-case lock duration; full async is [deferred/llm_path_async_router.md](deferred/llm_path_async_router.md).

## Context

Stress test results (from Plan 3's Phase C batching sweep) determine what ships here:

**If `llama.cpp --parallel` solves leader saturation:** we don't need multi-peer dispatch. Plan 4 ships as scoped (R3.0 + R3.5-lite + R3.6-lite) to give operators visibility without the routing complexity.

**If batching doesn't solve saturation:** same Plan 4 scope still ships, AND the deferred `llm_path_multi_peer_dispatch.md` gets revived to add overflow routing on top.

Either way, Plan 4 as scoped is valuable: it converts the cluster from "opaque single-leader box" into "inspectable, manageable set of nodes." That's platform-grade regardless of distribution decisions.

## Phases

### R3.0 — `maxim peer --node` CLI + `mesh.yml` config — ~250 LOC new

User-facing contract.

**New CLI verbs:**
```bash
maxim peer list-nodes                         # show registered nodes + live status
maxim peer --node <name> status               # one node's capabilities + queue
maxim peer --node <name> install <profile>    # explicit model install
maxim peer --node <name> logs                 # tail a node's logs (ssh-based, best effort)
maxim peer --node <name> drain                # stop accepting new requests
maxim peer --node <name> resume               # re-enable after drain
maxim peer --node <name> health               # run _MaximPeerBackend.health_check()
maxim peer --node <name> refresh              # clear backoff state + force fresh probe
maxim peer add-node <name> <url> --key <k>    # register a new node
maxim peer remove-node <name>                 # unregister
maxim peer rotate-cluster-key                 # generate + distribute new key
maxim peer agent-stats <agent_id>             # per-agent stats (from leader's accounting)
```

**New config file: `~/.config/maxim/mesh.yml`**
```yaml
cluster_key: <shared-cluster-key>   # bearer token trusted by all nodes
self: rtx-leader                    # name of THIS node — load-bearing for self-dispatch protection
protocol_version: 1                  # header contract version (Plan 1 R2)
nodes:
  - name: rtx-leader
    url: https://maxim.dennyschaedig.com/v1
    role: leader
  - name: mac-peer
    url: https://mac.dennyschaedig.com/v1
    role: peer
drain:
  - spare-gpu                       # nodes to skip until resumed
agent_rate_limits:                  # per-agent rate limiting (new in v2)
  default_rpm: 0                    # 0 = unlimited (default)
  overrides:
    runaway-test-agent: 10          # cap specific agents
```

**Self-dispatch protection:** the `self:` field names THIS node's entry. Even without multi-peer dispatch, this is used by:
- `X-Maxim-Node-Id` response header (Plan 3 R2.5 — the leader identifies itself)
- The admin API's `/v1/mesh/state` endpoint (reports local node name)
- The doctor check ("am I reachable as a peer?")

**Startup-time fail-loud:** if `self:` doesn't match any entry in `nodes:`, startup fails with a line-numbered error. Prevents misconfigured deployments from shipping silently.

**Schema validation on read:** malformed URLs, missing fields, unknown role values fail at startup with actionable errors. Same validation pattern as `peer/config.py`.

**Fallback:** if `mesh.yml` doesn't exist, R3.0 loads `peer.yml` and synthesizes a one-node mesh (leader only). Zero behavior change for existing users.

**Doctor integration:** `maxim doctor` gets a new check that calls `_MaximPeerBackend.health_check()` against every `mesh.yml` node with the local cluster key. Reports per-node outcome via the typed exception classes (BackendAuthFailed → key sync issue, BackendInferenceBroken → chat endpoint broken, BackendDown → network).

### R3.5-lite — `install` command + VRAM precheck — ~100 LOC new

Explicit model management on individual nodes.

**`maxim peer --node <name> install <profile>`:**
- Looks up the node in `mesh.yml`
- **Pre-flight VRAM check:** client-side lookup against the node's last health-check response (which advertises `vram_total_gb` in the models list from stage-1 probe). If `profile.required_vram_gb * 1.2 > node.vram_total_gb`, **refuse the install** with a structured warning. Prevents the "Mac downloads 14B, can't load it" failure mode from 2026-04-12.
- POST to `<node_url>/v1/admin/install` with `{profile: "qwen2.5-14b"}`
- Node runs existing `models/download.py::ensure_available` locally
- Blocks with progress display; Ctrl+C cancels cleanly

**No request-triggered auto-download.** Explicit only.

**Stranded-download metric:** if a download somehow slips past the precheck and the subsequent model load fails, emit `mesh_stranded_downloads_total{profile, reason}` + structured WARN with remediation.

### R3.6-lite — Admin API + metrics + per-agent tracking — ~300 LOC new

The operator surface. Without this, Plan 4 is just config files.

**Admin endpoints on `/v1/mesh/`:**

**Cluster state:**
- `GET /v1/mesh/state` → full local view: known peers, drain state, last probe result per peer
- `GET /v1/mesh/health` → one-line summary: `ok | degraded | dead`
- `GET /v1/mesh/request-trace/<req_id>` → specific request's routing decisions from the ring buffer
- `GET /v1/mesh/request-trace/recent?limit=50&agent_id=X&session_id=Y` → filtered tail of recent decisions

**Cluster operations:**
- `POST /v1/mesh/drain` → mark self drained (triggers reactive router rebuild)
- `POST /v1/mesh/resume` → clear drain state
- `POST /v1/mesh/refresh/<peer>` → force health check + clear backoff state for that peer
- `POST /v1/admin/install` → R3.5-lite install endpoint

**Per-agent observability (new in v2):**
- `GET /v1/mesh/agents` → list of agent_ids seen in the last N requests with counts
- `GET /v1/mesh/agents/<agent_id>/stats` → {in_flight, total_requests, avg_latency_ms, p50_latency_ms, p99_latency_ms, last_request_ts, errors_by_class}
- `GET /v1/mesh/agents/<agent_id>/requests?limit=50` → recent request-trace entries filtered by agent

**Cluster key rotation:**
- `POST /v1/mesh/rotate-cluster-key` → receives new key, verifies caller has old key, atomically installs new key, returns success
- Called by `maxim peer rotate-cluster-key` verb on each peer in the cluster + rollback if any fail

All endpoints require `Authorization: Bearer <cluster_key>`.

**Admin API rate limiting (new in v2):** all admin endpoints use `runtime/rate_limit.py` (cherry-picked from mesh/admission.py in Plan 1 R0). Default: 60 requests/minute per source IP. Prevents `drain` abuse (rapid drain/resume cycles), prevents `refresh` storms. Returns HTTP 429 with `Retry-After` when limit exceeded.

**Per-agent rate limiting (new in v2 — prevents runaway agent starvation):**

**Critical ordering invariant:** the rate-limit check runs in `LLMRouter.complete_text()` (the **public, unlocked** entry point) BEFORE it calls the private `_complete_text_locked()` helper that acquires `_inference_lock`. Rate-limited agents are rejected without ever touching the lock. Putting this check inside `_complete_text_locked` would defeat the purpose — a rate-limited agent would still hold the lock while being rejected, starving other agents exactly as if there were no rate limit.

```python
# In LLMRouter — PUBLIC entry point, BEFORE any lock acquisition
def complete_text(self, system, user, *, request_context=None, ...):
    # ── Per-agent rate limit check (Plan 4 R3.6-lite) ──
    # MUST run here, not inside _complete_text_locked (which holds _inference_lock).
    # Checking inside the lock would let a rate-limited agent hold the lock while
    # being rejected, defeating the starvation-prevention purpose.
    agent_id = (request_context or {}).get("agent_id")
    if agent_id and self._agent_rate_limiter:
        admitted, reason = self._agent_rate_limiter.check(agent_id, key_class="agent")
        if not admitted:
            raise BackendOverloaded(
                provider_key="agent-rate-limit",
                retry_after_s=self._agent_rate_limiter.estimate_retry_after(agent_id),
                fix_hint=f"Agent '{agent_id}' rate limited: {reason}. Check mesh.yml::agent_rate_limits.",
            )
    
    # Only now acquire the lock and run the actual inference
    with self._inference_lock:
        return self._complete_text_locked(system, user, request_context=request_context, ...)
```

~40 LOC of integration. The token bucket itself is the `KeyedRateLimiter` from `runtime/rate_limit.py` (cherry-picked in Plan 1 R0 from the deleted `mesh/admission.py`; already shipped as dormant code).

**Why this matters for multi-agent:** without per-agent rate limiting, one runaway agent (bug, infinite loop, adversarial) can saturate `_inference_lock` and starve every other agent in the process. With the check at the public entry point (BEFORE the lock), the runaway agent gets 429'd cheaply and other agents proceed normally. **Cost: ~40 LOC + the already-cherry-picked rate limiter. Benefit: cluster-level liveness property.**

**Dispatch-trace ring buffer** — in-memory only (no persistence per platform standard), holds last 100 decisions by default. Each entry:

```python
@dataclass(frozen=True)
class RequestTraceEntry:
    timestamp: float
    request_id: str
    agent_id: str | None       # new in v2 — multi-agent context
    session_id: str | None     # new in v2
    lane: str                  # large/medium/small
    providers_tried: list[str] # provider_keys attempted
    selected: str | None       # final winner
    outcomes: dict[str, str]   # provider_key → outcome
    total_latency_ms: float
    reason: str                # "single_peer", "local_only", etc.
```

**Per-agent filtering:** `GET /v1/mesh/request-trace/recent?agent_id=X` returns only entries for that agent. This is the **single most useful debugging surface** for multi-agent deployments — "show me the last 20 calls from agent npc-mother" is one curl away.

**Ring buffer overflow logging (new in v2):** when the buffer wraps (oldest entry drops), increment `request_trace_dropped_total`. Log at DEBUG once per 100 drops. **Warn at WARN if drops exceed 1000/minute** — indicates the ring buffer size is too small for the request rate. Configurable via `MAXIM_REQUEST_TRACE_SIZE=N` env var or `mesh.yml::request_trace_size`.

**Feature flags:**
- `MAXIM_MESH=0|1` — enable Plan 4 mesh features (default 0 initially, 1 after bake-in)
- `MAXIM_MESH_VERBOSITY=0|1|2` — 0=metrics only, 1=one JSONL line per decision, 2=verbose decision tree

Both documented in CLAUDE.md env var table. Both have `conftest.py` autouse scrubs.

## Logging & verbosity requirements

Plan 1 established JSONL format + `agent_id` / `session_id` / `request_id` enrichment. Plan 3 added backend-call events. Plan 4 adds dispatcher + admin layer.

**Note on verbosity env vars:** `MAXIM_MESH_VERBOSITY` (0/1/2) is orthogonal to the existing `MAXIM_PROVENANCE_VERBOSITY` (0/1/2). Mesh verbosity controls dispatch-decision and admin-API log detail; provenance verbosity controls the lane-decision trace written to `~/.maxim/util/lane_decisions.jsonl`. Setting one does not affect the other. Both use the same 0/1/2 scale for consistency with `MAXIM_AGENTIC_VERBOSITY` (0-3) from `utils/structured_logging.py`.

**Request trace decisions** — format depends on `MAXIM_MESH_VERBOSITY`:

At `MAXIM_MESH_VERBOSITY=1` (recommended default):
```json
{"ts":"...","level":"INFO","event":"request_trace","request_id":"abc123","lane":"large","selected":"lane-large-rtx-leader","providers_tried":1,"total_latency_ms":340,"reason":"single_peer_success","agent_id":"npc-mother","session_id":"sim-42"}
```

At `MAXIM_MESH_VERBOSITY=2` (verbose):
```json
{"ts":"...","level":"DEBUG","event":"dispatch_start","request_id":"abc123","lane":"large","providers":["lane-large-rtx-leader"],"policy":"single_peer","agent_id":"npc-mother","session_id":"sim-42"}
{"ts":"...","level":"DEBUG","event":"dispatch_attempt","request_id":"abc123","provider":"lane-large-rtx-leader","attempt":1,"outcome":"ok","latency_ms":340,"agent_id":"npc-mother"}
{"ts":"...","level":"INFO","event":"request_trace","request_id":"abc123","lane":"large","selected":"lane-large-rtx-leader","providers_tried":1,"total_latency_ms":340,"agent_id":"npc-mother","session_id":"sim-42"}
```

At `MAXIM_MESH_VERBOSITY=0`: metrics only.

**Ring buffer overflow warnings:**
```json
{"ts":"...","level":"WARN","event":"request_trace_overflow","drops_per_minute":1240,"current_size":100,"fix_hint":"Increase MAXIM_REQUEST_TRACE_SIZE or reduce request rate"}
```

**Admin API access** at INFO:
```json
{"ts":"...","level":"INFO","event":"mesh_admin","endpoint":"/v1/mesh/drain","method":"POST","caller_ip":"10.0.0.50","outcome":"ok","affected_lanes":["large"]}
```

**Per-agent rate limit events** at WARN:
```json
{"ts":"...","level":"WARN","event":"agent_rate_limited","agent_id":"runaway-test-agent","rate_rpm":10,"retry_after_s":4.2,"lane":"large","session_id":"sim-42"}
```

**Cluster key rotation events** at INFO (each step):
```json
{"ts":"...","level":"INFO","event":"cluster_key_rotation_step","step":"verify_old_key","peer":"mac-peer","outcome":"ok"}
{"ts":"...","level":"INFO","event":"cluster_key_rotation_step","step":"install_new_key","peer":"mac-peer","outcome":"ok"}
{"ts":"...","level":"INFO","event":"cluster_key_rotation_complete","peers_updated":2,"duration_ms":840}
```

**New metrics:**
- `request_trace_entries_total` — counter (requests traced)
- `request_trace_dropped_total` — counter (buffer overflow events)
- `mesh_admin_requests_total{endpoint, status}` — counter
- `mesh_admin_rate_limited_total{endpoint}` — counter (admin API rate limit hits)
- `mesh_probe_outcome_total{peer, outcome}` — counter (from doctor integration)
- `agent_rate_limited_total{agent_id}` — counter (only populated if rate limits configured; cardinality bounded by config)
- `agent_in_flight_requests{agent_id}` — gauge (only populated when per-agent accounting enabled)

**Note on agent-scoped metric cardinality:** `agent_rate_limited_total` and `agent_in_flight_requests` are labeled by `agent_id`, which would normally explode cardinality. Mitigation: **only configured agents appear as labels**, not every observed agent. Agents without explicit rate limits don't contribute labels. Agents above a configurable threshold (`mesh.yml::agent_stats_top_n: 20`) get per-agent labels; the rest aggregate into `agent_id="__other__"`.

## Multi-agent / multi-user lens findings (applied to Plan 4)

**1. Per-agent rate limiting lands here.** Prevents runaway agent starving cluster. ~70 LOC. User decision: fold in, don't defer.

**2. Per-agent observability via admin API.** `GET /v1/mesh/agents/<agent_id>/stats` provides the debugging hook. ~50 LOC.

**3. Request-trace ring buffer is per-agent filterable.** The single most useful debug surface for multi-agent workloads.

**4. Drain state uses reader-writer lock from Plan 2 R2.** Plan 2 introduces the primitive; Plan 4 consumes it. No race conditions between drain and dispatch.

**5. Admin API DoS protection.** Cherry-picked `rate_limit.py` from R0 handles this. Drain/refresh can't be abused to DoS the cluster.

**6. Cluster key rotation is an atomic operation.** POST to every peer in sequence, rollback on any failure. ~50 LOC. User decision: POST-to-every-peer approach.

**7. Data sovereignty documented.** The architecture doc has the section; Plan 4's operator doc references it.

**8. Audit: what happens when `mesh.yml` changes mid-session?** Not supported — requires restart. Plan 4 validates schema at startup, doesn't hot-reload. Hot reload is a future enhancement noted in `deferred/mesh_hot_reload.md` (shell plan).

**9. Audit: mixed-version clusters.** Plan 1 R1 introduces `X-Maxim-Protocol-Version: 1` header. Plan 4 R3.0 exposes `protocol_version: 1` in `mesh.yml`. Receivers log a warning on unknown versions but process requests normally (forward compatibility). This is documented in the header contract; full version negotiation deferred.

**10. Audit: privilege escalation via admin API.** Every endpoint requires cluster key. Rate limiting prevents brute force. Cluster key in HTTPS only (check protocol). `rotate-cluster-key` verifies old key before accepting new one.

## Success criteria — R3.0 + R3.5-lite + R3.6-lite

**R3.0:**
- All CLI verbs implemented + unit tested
- `mesh.yml` schema validated with line-numbered error messages
- `list-nodes` output is stable + readable JSON
- Drain state persists across restarts (role-scoped file)
- Doctor integration reports per-node typed exceptions
- Self-dispatch protection: startup fails if `self:` is missing/invalid

**R3.5-lite:**
- Install command tested against mocked node
- VRAM precheck refuses installs that would strand
- Stranded-download metric fires if a download slips past

**R3.6-lite:**
- All admin endpoints implemented + auth-gated
- Admin API rate limiting enforced + tested
- Request-trace ring buffer: run a sim, `GET /v1/mesh/request-trace/<req_id>` returns the entry
- Per-agent filter: `?agent_id=X` returns only matching entries
- Per-agent stats: `GET /v1/mesh/agents/<id>/stats` returns valid JSON
- Per-agent rate limiting: configured cap enforced via `BackendOverloaded` typed exception
- Overflow logging fires when buffer wraps + drops exceed threshold
- Cluster key rotation: tested end-to-end in 2-node fixture, rollback on simulated failure

## Hard testing requirement (checkpoint before plan declared shipped)

**Non-negotiable.**

**Automated:**
```bash
# Full fast suite
python -m pytest tests/ -x -q -m "not slow" --ignore=tests/integration/test_memory_hub.py

# Targeted mesh tests
python -m pytest tests/unit/test_mesh_config.py -v
python -m pytest tests/unit/test_self_dispatch_protection.py -v
python -m pytest tests/unit/test_request_trace_ring_buffer.py -v
python -m pytest tests/unit/test_admin_api.py -v
python -m pytest tests/unit/test_agent_rate_limiting.py -v
python -m pytest tests/unit/test_cluster_key_rotation.py -v

# Integration: 2-node fixture
python -m pytest tests/integration/test_mesh_visibility.py -v

# Lint + format
ruff check src/ tests/
ruff format --check src/ tests/
```

**Manual smoke:**
```bash
# 1. Set up 2 nodes with mesh.yml
# 2. Run sims from each, verify request-trace records
maxim --sim "test" &
sleep 5
curl -H "Authorization: Bearer $CLUSTER_KEY" \
  http://localhost:8100/v1/mesh/request-trace/recent?limit=20 | jq

# 3. Per-agent filtering
curl -H "Authorization: Bearer $CLUSTER_KEY" \
  "http://localhost:8100/v1/mesh/request-trace/recent?agent_id=sim_orchestrator&limit=10" | jq

# 4. Per-agent stats
curl -H "Authorization: Bearer $CLUSTER_KEY" \
  http://localhost:8100/v1/mesh/agents/sim_orchestrator/stats | jq

# 5. Drain + verify exclusion
curl -H "Authorization: Bearer $CLUSTER_KEY" -X POST \
  http://localhost:8100/v1/mesh/drain
maxim --sim "quick test"  # should error out (no peers available)
curl -X POST http://localhost:8100/v1/mesh/resume

# 6. Rate limit test (configure agent_rate_limits, run a burst)
# Expected: BackendOverloaded exception after limit exceeded

# 7. Doctor check
maxim doctor
# Expected: mesh health check runs per-peer

# 8. Cluster key rotation
maxim peer rotate-cluster-key
# Expected: both peers update atomically, or rollback cleanly on failure
```

**Rollback drill:**
```bash
MAXIM_MESH=0 maxim --sim "rollback test"
# Expected: Plan 3 behavior exactly, no admin API, no trace buffer
```

## Documentation & memory update (runs after testing passes)

**Load-bearing.** Plan 4 is the most operator-facing of the four plans.

**1. Update [../architecture/llm_routing.md](../architecture/llm_routing.md):**

Add sections:
- **"Operator visibility layer"** — request-trace ring buffer, per-agent filtering, admin API
- **"Per-agent rate limiting"** — how the router checks bucket, how `mesh.yml` configures, how 429 surfaces
- **"Cluster key rotation"** — the atomic rotation protocol
- **"Drain state lifecycle"** — how drain affects dispatch (reactive rebuild), how it persists, how it interacts with concurrent requests
- **"What Plan 4 does NOT do"** — explicit reference to deferred multi-peer dispatch, explaining the stress-test-driven decision

**2. Create [../architecture/mesh_operations.md](../architecture/mesh_operations.md):**

Operator-focused doc (~400 lines):

- **Setup:** 1-node (solo), 2-node (leader + peer), 3-node (leader + 2 peers) with `mesh.yml` examples
- **Daily operations:** `list-nodes`, `status`, `drain/resume` workflow for maintenance
- **Multi-agent debugging:**
  - "Which agent is slow?" → per-agent stats endpoint
  - "Why did agent X's request fail?" → `request-trace/recent?agent_id=X`
  - "Agent X is hammering the leader" → configure `mesh.yml::agent_rate_limits`
- **Cluster key rotation workflow** with rollback scenarios
- **Failure modes:** per typed exception class, operator action
- **Upgrade paths:** how to drain/upgrade/resume without dropping requests
- **Log correlation:** how to grep JSONL logs by `request_id`, `agent_id`, `session_id`
- **Capacity planning:** what metrics to watch, when to add a peer, when batching beats distribution
- **Troubleshooting checklist:** per "symptom" → first thing to check

**3. Update [../reference.md](../reference.md):**

- **"Mesh configuration"** section: `mesh.yml` schema including `self`, `protocol_version`, `agent_rate_limits`
- **"Admin API surface"** list of `/v1/mesh/*` endpoints with auth + rate limiting
- **"Request trace"** ring buffer location, retention, access pattern
- **"Per-agent stats"** tracking + admin API endpoint

**4. Update [../../CLAUDE.md](../../CLAUDE.md):**

- **Env var table:** `MAXIM_MESH`, `MAXIM_MESH_VERBOSITY`, `MAXIM_CLUSTER_KEY`, `MAXIM_REQUEST_TRACE_SIZE`
- **Key commands:** full `maxim peer --node` verb list
- **Lessons learned:**
  > **`mesh.yml::self` is load-bearing.** Self-dispatch protection depends on it. Startup fails loudly if missing or mismatched. This prevents infinite self-dispatch and silent misconfiguration.
  > 
  > **Request trace ring buffer is in-memory only.** For post-mortem, use the JSONL logs (persisted) + `request_id`. The ring buffer is for active debugging.
  > 
  > **Per-agent rate limits go in `mesh.yml::agent_rate_limits`.** Configured agents appear as metric labels; unconfigured agents aggregate as `agent_id="__other__"`. This prevents cardinality explosion.
- **Architectural invariants:**
  > **Admin API endpoints require cluster key auth + are rate limited.** Adding a new admin endpoint means updating both the auth check and the rate limiter. Both are enforced by decorators on the endpoint handler.
  > 
  > **Per-agent rate limiting happens at the router entry, BEFORE `_inference_lock` is acquired.** This is load-bearing — if the check happens inside the lock, one rate-limited agent holds the lock for nothing, starving others.

**5. Create [../troubleshooting/mesh_debug.md](../troubleshooting/mesh_debug.md):**

Replaces the legacy `mesh.md` (which R0 deprecated). ~250 lines covering:
- "A request went to the wrong node" → `request-trace/<req_id>`
- "An agent is slow" → `agents/<agent_id>/stats` + `request-trace?agent_id=X`
- "Node marked dead but alive" → `refresh/<peer>` + backoff check
- "Cluster key rotation" → workflow + rollback
- "Drain stuck" → check lock state, force refresh
- "Mesh.yml invalid" → schema error cross-reference
- "Per-agent rate limit misfiring" → config + observation
- "Mixed-version cluster" → protocol version header + upgrade order

**6. Update project memory:**

Add `project_llm_path_operator_visibility_shipped.md`:

```markdown
---
name: LLM path operator visibility shipped
description: mesh.yml + admin API + per-agent observability + rate limiting
type: project
---

**Shipped:** <date> as part of [llm_path_operator_visibility.md](docs/plans/llm_path_operator_visibility.md).

**Key decisions:**
- Renamed from "Reactive Mesh" — multi-peer dispatch moved to deferred per stress-test results
- Ships unconditionally (not stress-test-gated) — operator visibility is always valuable
- Per-agent rate limiting landed here (not deferred) — prevents runaway agent starvation
- Cluster key rotation via atomic POST-to-every-peer

**What shipped:**
- `maxim peer` CLI with 11 new verbs + `mesh.yml` config
- `/v1/mesh/*` admin API with per-agent stats + request-trace filtering
- Per-agent token-bucket rate limiter (configurable via `mesh.yml::agent_rate_limits`)
- Self-dispatch protection via `mesh.yml::self` field
- Cluster key rotation verb + atomic multi-peer update
- Doctor check covering all mesh.yml nodes via `_MaximPeerBackend.health_check()`

**What is NOT in this plan (deferred):**
- Multi-peer reactive dispatch → [deferred/llm_path_multi_peer_dispatch.md](docs/plans/deferred/llm_path_multi_peer_dispatch.md)
- Async router (concurrent per-lane agents) → [deferred/llm_path_async_router.md](docs/plans/deferred/llm_path_async_router.md)
- Fair-share scheduling → [deferred/llm_path_fair_scheduling.md](docs/plans/deferred/llm_path_fair_scheduling.md)
- Capability-aware mesh → [deferred/llm_mesh_capability_aware.md](docs/plans/deferred/llm_mesh_capability_aware.md)

**Load-bearing facts for future refinement:**
- NEVER bypass the admin API auth check or rate limiter. Both are decorated.
- NEVER move per-agent rate limiting inside `_inference_lock`. Must be at router entry.
- `mesh.yml::self` must match a node entry. Startup failure is intentional.
- Request-trace ring buffer is in-memory only. Persist via JSONL logs if needed.
- Per-agent metric labels are bounded by config + top-N threshold. Do not add unbounded labels.

**Related:** 
- [docs/architecture/mesh_operations.md](docs/architecture/mesh_operations.md) — operator runbook
- [docs/architecture/llm_routing.md](docs/architecture/llm_routing.md) — full routing architecture
```

Update `MEMORY.md`.

**7. Update meta-plan + stress test report link.**

## Migration notes

- `mesh.yml` is opt-in. Absent → fall back to `peer.yml` (Plan 3 behavior). Zero user action required.
- `MAXIM_MESH=0` (default) preserves Plan 3 behavior exactly. Opt-in via env var for bake-in.
- `MAXIM_CLUSTER_KEY` replaces per-node API keys in mesh mode.
- Drain state file at `~/.maxim/util/drained_nodes.{role}.txt` — auto-created, role-scoped.
- Per-agent rate limits opt-in via `mesh.yml::agent_rate_limits`. Default is unlimited (backwards-compatible).

## Open questions — resolved

1. **Partial Plan 4 ship:** **R3.0 + R3.5-lite + R3.6-lite ships unconditionally.** R3.3-lite + R3.4 moved to deferred.
2. **Rendezvous hash seed:** N/A — no multi-peer dispatch in this plan.
3. **`X-Maxim-Suggested-Peer` population:** deferred (in the multi-peer plan).
4. **Ring buffer 100 entries:** acceptable, configurable via env var, overflow logged at WARN above 1000/min.
5. **Cluster key rotation:** POST-to-every-peer with atomic rollback.
6. **Per-agent rate limiting:** **folded in** — not deferred. Prevents runaway agent scenarios.

## Related docs

- **Previous plan:** [llm_path_fast_failover.md](llm_path_fast_failover.md) — prerequisite
- **Meta plan:** [llm_path_refinement.md](llm_path_refinement.md)
- **Foundation:** [llm_path_foundation.md](llm_path_foundation.md)
- **Architecture:** [../architecture/llm_routing.md](../architecture/llm_routing.md) — extended by this plan
- **Architecture:** [../architecture/mesh_operations.md](../architecture/mesh_operations.md) — created by this plan
- **Deferred:** [deferred/llm_path_multi_peer_dispatch.md](deferred/llm_path_multi_peer_dispatch.md) — multi-peer we chose NOT to build now
- **Deferred:** [deferred/llm_path_async_router.md](deferred/llm_path_async_router.md) — async router refactor
- **Deferred:** [deferred/llm_path_fair_scheduling.md](deferred/llm_path_fair_scheduling.md) — full fair-share (rate limiting is folded into this plan; fair-share is the bigger refactor)
- **Stress test results:** `docs/experiments/results/llm_path_stress_<date>.md`
- **Project guide:** [../../CLAUDE.md](../../CLAUDE.md)
