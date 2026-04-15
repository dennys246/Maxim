# LLM Path Refinement — Plan 4: Operator Visibility

**Status:** Draft v4 — 2026-04-14: split into three sequential stages after Phase D2 surfaced two concrete must-ship items (A+B) that close out the 0.4 stability story before the bigger mesh.yml / admin-API work (C).
**Scope:** ~750 LOC new (A: ~150, B: ~550, C: ~650 as originally scoped — ships across 2-3 sessions)
**Target version:** A+B in 0.4. C spans 0.4/0.5 depending on session cadence.
**Part of:** [llm_path_refinement.md](llm_path_refinement.md)
**Depends on:**
- [archive/llm_path_fast_failover.md](archive/llm_path_fast_failover.md) (Plan 3) — typed-exception router loop ✅ shipped
- [archive/llm_path_cancellation_hygiene.md](archive/llm_path_cancellation_hygiene.md) (Plan 3.5) — "HTTP fires first" contract ✅ shipped
- [llm_path_peer_failover.md](llm_path_peer_failover.md) (Plan 3.6) — VRAM spillover detection + multi-leader precursor ✅ shipped
**Note:** renamed from "Reactive Mesh" in v1. Multi-peer dispatch moved to [deferred/llm_path_multi_peer_dispatch.md](deferred/llm_path_multi_peer_dispatch.md). Capability-aware ranking is [deferred/llm_mesh_capability_aware.md](deferred/llm_mesh_capability_aware.md).
**Bake-in target (2026-04-13):** the user's RTX 5080 + RTX 3070 setup is the concrete two-node deployment for testing `mesh.yml`'s schema validation, drain/resume, per-node admin endpoints, and per-agent rate limiting. Plan 3.6 unblocks failover testing without waiting for the full Plan 4 admin API.

---

## Shipping stages (2026-04-14)

Plan 4 is split into three sequential stages so each ships cleanly in
its own session with its own pre-merge review. **Stage A+B are
Phase-D-surfaced must-ships**; Stage C is the original platform-grade
operator-visibility scope that needs a dedicated multi-session effort.

### Stage A — agent_id observability fix ✅ SHIPPED (2026-04-14)

Fixes the Phase D report's "agent_id=null in peer_backend_call events"
gap via three complementary changes:

1. **Router capability-flag forwarding.** `LLMRouter._invoke_backend`
   now forwards `request_context` through `kwargs` for backends that
   declare `supports_request_context = True`. Matches the existing
   `supports_model_override`/`supports_tool_use`/`supports_streaming`
   capability-flag pattern. Only `_MaximPeerBackend` sets this flag;
   cloud backends are unchanged (no `**kwargs` catch-all →
   `TypeError` if unconditionally forwarded).
2. **Boundary contextvar binding.**
   `LLMWorker._call_llm_with_timeout` calls
   `maxim.utils.http.set_context(normalized)` next to the existing
   `set_cancel_event` binding, BEFORE `contextvars.copy_context()`
   snapshots the context into the worker thread. Resets in `finally`.
   This populates `X-Maxim-Agent-Id` / `X-Maxim-Session-Id` /
   `X-Maxim-Request-Id` on all outbound internal HTTP calls
   automatically — the previously-dead path that `utils/http.py`
   was wired for but nobody called.
3. **Contextvar fallback in the shim.**
   `_normalize_request_context(None)` now reads `current_context()`
   before manufacturing a fresh empty RequestContext. Defense in
   depth for paths that bypass the kwarg threading.

**Regression guards (11 new tests):**
- `TestRequestContext::test_contextvar_fallback_populates_context_when_dict_is_none`
- `TestRequestContext::test_explicit_dict_still_wins_over_contextvar`
- `TestRequestContext::test_supports_request_context_capability_flag_is_declared`
- `TestRequestContextForwarding` in `test_router_typed_exceptions.py` — 3 tests
- `test_set_context_binding_propagates_into_worker_thread`
- `test_set_context_binding_is_reset_between_sequential_calls`
- `test_outbound_headers_populate_from_bound_context`

**Load-bearing invariants locked in:**
- The `supports_request_context` capability flag on `_MaximPeerBackend`
  is load-bearing. Removing it silently drops `agent_id` from
  `peer_backend_call` logs.
- The `set_context` binding in `_call_llm_with_timeout` must live
  BEFORE `copy_context()` so the worker thread inherits it. Moving
  it below breaks contextvar propagation across the ThreadPoolExecutor
  boundary (same mechanism as cancellation).
- `reset_context(context_token)` in `finally` is mandatory — without
  it, sequential calls leak the first call's `agent_id` into later
  calls' outbound HTTP headers.

**~~Out of scope (deferred):~~ ✅ RESOLVED (2026-04-14 follow-up):**
`session_id` plumbing from `SimulationOrchestrator` through
`LLMWorker`. The fix turned out simpler than the original framing
implied — no agent-facing interface change needed. ``LLMWorker``
gained an optional ``session_id`` constructor argument stored on the
instance; both dict-build sites in ``llm_worker.py`` now include
``"session_id": self._session_id``. The simulation orchestrator
pre-generates the ``time.strftime`` timestamp at sim entry (matching
the ``research_orchestrator.py:73`` pattern) and passes the same
value to both ``LLMWorker(session_id=...)`` constructors and to
``build_report(session_id=...)`` so the report directory name
matches the id in the JSONL log trace. Non-sim callers (exec_agent,
api.py, bench, embodied_runtime) keep ``session_id=None`` as the
default and their logs continue to emit ``session_id=null`` — the
correct semantic for non-session contexts. Also folded: the bench
harness now generates a distinct per-run ``bench_YYYYMMDD_HHMMSS``
session_id instead of reusing ``BENCH_AGENT_ID``, so back-to-back
bench runs are distinguishable in the JSONL log. Six new regression
tests across ``test_llm_worker_pool.py::TestSessionIdPlumbing`` (5)
and ``test_bench_recovery_time.py`` (1 per-run + 1 CLI emitter).

### Stage B — recovery-time benchmark harness ✅ SHIPPED (2026-04-14)

New `maxim bench recovery-time` CLI subcommand that fires chat
completions in a tight loop against the peer URL, records per-call
timing + outcome, and extracts a rigorous recovery-time number from
the first `success → failure → success` transition.

**Rationale:** the Phase D report (2026-04-13) flagged the
leader-ready-to-first-success recovery gate as "inconclusive under sim
workload" because the orchestrator does 30+s of local agent work
between LLM calls. The observed 30.7s gap was sim cadence, not
peer-side wedge state. A tight-loop bench eliminates the cadence
artifact and gives a clean number.

**Implementation:**
- `src/maxim/bench/recovery_time.py` — `run_recovery_benchmark()`
  pure function with a `backend_factory` test hook for offline unit
  tests. Uses `_MaximPeerBackend` directly (no router, no
  `_inference_lock` contention). Each attempt binds its own
  `RequestContext` with `BENCH_AGENT_ID = "bench_recovery_time"`.
- `src/maxim/bench/cli.py` — `run_bench_subcommand()` dispatch +
  `_run_recovery_time()` argparse entry point. Emits JSONL events
  matching production `peer_backend_call` / `peer_backend_failed`
  shape so existing `jq 'select(.e=="peer_backend_call")'` queries
  work unchanged on the bench output.
- `src/maxim/cli.py::main` dispatches `maxim bench <subcommand>`.

**Regression guards (21 new tests):**
- `TestClassifyError` — 8 tests covering the typed exception mapping
- `TestAnalyseRecovery` — 5 tests covering the recovery-window analysis
  (simple recovery, no outage, did-not-recover, no-pre-outage-success,
  first-failure-as-denominator)
- `TestRunRecoveryBenchmark` — 5 tests (tight loop, recovery transition,
  SIGINT stop, stable bench agent_id, contextvar cleanup)
- `TestBenchCliOutput` — 3 tests (JSONL shape, unknown subcommand,
  empty argv)

**Real-hardware validation:** first Phase D2 run (2026-04-14) measured
**58.68s recovery window** on a 16 GB RTX 5080 running Qwen-14B Q4_K_M
vs 53s leader-self-reported restart duration. Peer-side overhead ≈ 0s
beyond the leader's intrinsic reload time. All failures fast-failed
within 3.1s (p99 = 614ms). Every one of the 750 JSONL events had
`agent_id=bench_recovery_time` — Stage A validated end-to-end on real
traffic.

**Report:** [../experiments/results/llm_path_stress_plan4_20260414.md](../experiments/results/llm_path_stress_plan4_20260414.md)
**Rerun runbook:** [../experiments/protocols/bench_recovery_time_rerun.md](../experiments/protocols/bench_recovery_time_rerun.md)

### Stage C — mesh.yml + admin API + per-agent rate limiting (C1 ✅ SHIPPED, C2 ✅ SHIPPED, C3.1 ✅ SHIPPED, rest of C3 DEFERRED)

**Stage C1 — mesh.yml + CLI verb foundations — ✅ SHIPPED 2026-04-14** (branch `feat/plan4-c1-mesh-yml`). Delivered as a read-only-verbs slice of the original Stage C after pre-merge review folded drain state + `drain:` schema field to C2:

- `src/maxim/peer/mesh_config.py` — schema parser + line-numbered validation + fallback from `peer.yml` + `classify_probe_outcome()` shared classifier (single source of truth for probe outcome → (status, message, fix) mapping across mesh_cli and doctor)
- `src/maxim/peer/mesh_cli.py` — new verbs `maxim peer list-nodes [--json]` and `maxim peer --node <name> {status|health}`; probes via `_MaximPeerBackend.for_url(...).health_check()` — the canonical Plan 3 R2.6 entry point
- `src/maxim/doctor/checks.py::check_mesh_nodes` — per-node `CheckResult` routed through the shared classifier; returns `[]` (not a sentinel) when no mesh.yml is configured
- `src/maxim/doctor/cli.py` — dynamic `mesh_node_<name>` retry-id registration in the retry loop; re-probes re-read `mesh.yml` each iteration
- 53 new unit tests (31 schema + 14 CLI + 8 doctor), all offline with fresh-per-test fake backend classes (no shared class-level mutable state)

**Parser hardening:** rejects tabs, bare `- ` entries, duplicate node names, inline `# comments` inside values, and unknown top-level keys — all with line-numbered errors. Pre-merge review caught four silent-mis-parse cases that the initial implementation tolerated.

Live smoke (RTX 5080 leader via Cloudflare tunnel) validated list-nodes, `--json`, `--node health`, and unknown-node rejection at 292-380ms stage-2 latency.

**Zero behavior change for existing users:** when `mesh.yml` is absent, `read_or_synthesize_mesh_config()` builds a one-node mesh from `peer.yml`. The new verbs Just Work on existing peer installs.

**Deferred to C2 as a block** (pre-merge review cross-confirmed finding): drain/resume verbs, `mesh.yml::drain` schema field, runtime drain state file. The original C1 design had a two-layer config-vs-runtime drain with no reconciliation contract, no role-detection timing story for the `MAXIM_ROLE` env var, a read/write race, and no orphan validation. Four findings collapse into one deferral until C2 does a proper drain design pass.

**Stage C2 — drain/resume with runtime state layer — ✅ SHIPPED 2026-04-14** (branch `feat/plan4-c2-drain`). The pre-design review round surfaced 3 critical findings that killed the original Option A1 proposal (config-only drain with TOML migration): `tomllib` isn't available on Python 3.10, concurrent drain RMW race was unsolved, and config-only drain constrains C3's admin API into a dead end. Pivoted to **Option B (runtime state layer)** which resolves all 4 CC2 findings explicitly without forcing a format migration or deleting the FROZEN parser invariant. Delivered:

- `src/maxim/peer/drain_state.py` (NEW) — role-scoped drain state at `~/.maxim/util/drained_nodes.{role}.txt`, `filelock.FileLock` serialized RMW cycle, `DrainError` with known-node list for orphan validation, `DrainReadResult` dataclass with active/orphans partition
- `src/maxim/peer/mesh_cli.py` — new verbs `drain` / `resume` / `list-drained`, drain display in `list-nodes` table + JSON (⊝ symbol, `drained` boolean, top-level `orphans` array), self-drain guard with `--force-self` override
- `src/maxim/peer/cli.py` — `detect_and_apply_role(argv)` call at the top of `run_peer_connect_subcommand` so drain state path resolution sees the correct `MAXIM_ROLE` on peer subcommand dispatch (fix for CC2 finding #1)
- `src/maxim/doctor/checks.py::check_mesh_nodes` — drained nodes render as `info` without probing, orphan drain entries surface as `warn` `Drain orphan <name>` CheckResults with resume hints, regression-guarded via counting backend that asserts `call_count == 1` for a 2-node mesh with one drained
- `src/maxim/utils/atomic_io.py` — new `preserve_mode: bool = False` kwarg preserves pre-existing mode bits across rewrites via `os.stat` + `os.chmod`. Fix for pre-design review finding E3 (silent secret leak when a 0600 file gets widened to umask 0644). Drain state writes opt in; future C3 credential-bearing files inherit the flag.
- `pyproject.toml` — `filelock>=3.0,<4.0` added as core dep. Already present transitively via `huggingface_hub` / `torch` on most installs; explicit here so headless peers without either optional extra still get it. POSIX `fcntl` + Windows `msvcrt` wrapped under one API.
- 50+ new unit tests (13 atomic_io `preserve_mode` + 22 drain_state incl. `multiprocessing.Pool(4)` RMW race + 13 mesh_cli drain verb/exit-code/display + 3 doctor drain handling)

**Three-lens pre-design review** (Architecture + Execution + Blast Radius, all run in parallel before any C2 code) caught 31 findings across the original proposal, including 4 criticals that forced the Option A1 → B pivot. See [feedback_cross_confirmed_review_findings.md](../../.claude/projects/-Users-dennyschaedig-Scripts-Maxim/memory/feedback_cross_confirmed_review_findings.md) for the pattern; the C2 session validates it a fifth time (Plan 3 R3, Plan 3.6 R5, Plan 4 A+B, Plan 4 C1 rounds 1+2, Plan 4 C2 design phase).

**Four CC2 findings from C1 pre-merge review each fixed explicitly** (with a regression test):

| CC2 Finding | C2 Fix | Regression Guard |
|---|---|---|
| Role detection timing | `detect_and_apply_role(argv)` at top of `run_peer_connect_subcommand` | `TestRoleIsolation::test_leader_and_peer_have_distinct_files` |
| Read/write race | `filelock.FileLock` on sibling `.lock` file around RMW | `TestConcurrency::test_ten_parallel_drains_all_land` (via `multiprocessing.Pool`) |
| Orphan validation | `DrainReadResult.orphans` + orphan warn in `doctor` + `list-drained` footer | `TestOrphanValidation::test_orphans_surfaced_on_read` + doctor equivalent |
| Permission preservation | `atomic_write_text(preserve_mode=True)` | `TestPreserveMode::test_preserves_0600_on_rewrite` + setuid bits guard |

**Deferred to C3:** `--node install` + VRAM precheck, `--node refresh`, `add-node`, `remove-node`, `/v1/mesh/*` admin API, per-agent rate limiting, request-trace ring buffer, cluster key rotation. The `KeyedRateLimiter` dormant code from Plan 1 R0 lights up in C3.

**Stage C3.1 — `init-mesh` verb — ✅ SHIPPED 2026-04-14** (branch `feat/plan4-c3.1-init-mesh`). The smallest C3 piece, separated to unblock drain/resume on `peer.yml`-only installs without waiting for the bigger C3 surface (admin API, rate limiting). Delivered:

- `src/maxim/peer/init_mesh.py` (NEW, ~190 LOC) — decision-tree driver for `maxim peer init-mesh [--force]`. Reads `peer.yml`, synthesizes a one-node `MeshConfig` via the existing C1 helper, writes `mesh.yml` via the new `write_mesh_config`. Backs up the existing `mesh.yml` to `mesh.yml.bak` (via `shutil.copy2`, preserves mtime + mode) when `--force` is passed.
- `src/maxim/peer/mesh_config.py` — added `MeshConfig.to_yaml()` method (mirrors `peer/config.py::PeerConfig.to_yaml()` shape) + `write_mesh_config(cfg, path=None)` disk-I/O wrapper that routes through `atomic_write_secret` because `mesh.yml::cluster_key` is a secret per the C2 invariant. First write chmods to `0o600`; rewrites preserve existing mode bits.
- `src/maxim/peer/cli.py` — dispatch the new `init-mesh` verb
- `src/maxim/cli.py` — `peer_action` recognition list extended
- `tests/unit/test_init_mesh.py` (NEW, 20 tests) — full decision-tree coverage including: nothing-to-convert exit 1, mesh-already-exists no-op exit 0, happy-path synthesize, `peer.yml` preservation regression guard (load-bearing for role detection), round-trip through parser, `0o600` perm assertion (POSIX), refuse-without-force exit 2, force overwrite + backup byte-equal, malformed peer.yml fails before touching mesh, backup-failure aborts before overwrite, end-to-end drain post-init integration test
- `tests/unit/test_mesh_config.py` — added 11 round-trip + write_mesh_config tests (1-node + multi-node + format stability + no-PyYAML-syntax + protocol_version default + first-write-chmod + rewrite-preserves-mode)

**Decision tree (locked from C2 pre-design review E7):**

| `peer.yml` | `mesh.yml` | `--force` | Action | Exit |
|---|---|---|---|---|
| absent | absent | — | "nothing to convert" | 1 |
| absent | present | — | "already exists, nothing to do" | 0 |
| present | absent | — | synthesize from peer.yml | 0 |
| present | present | no | refuse with `--force` hint | 2 |
| present | present | yes | back up `mesh.yml` → `mesh.yml.bak`, then synthesize | 0 |

**`peer.yml` is left in place by design.** `runtime/role.py` reads `peer.yml` existence as part of the role detection decision order (Plan 2 R2a). Deleting or moving it post-init-mesh would break role detection silently. The two files coexist: `peer.yml` is the role-detection signal + simple-single-leader config; `mesh.yml` is the multi-node topology surface that drain/resume + `list-nodes` consume.

**Architectural note (write path):** `write_mesh_config` is the first caller of `atomic_write_secret` outside of the C2 fold. Validates the C2 invariant ("credential-bearing files use `atomic_write_secret`, not `atomic_write_text(preserve_mode=True)`") in production code. The `peer/config.py::write_peer_config` function currently uses plain `path.write_text` + explicit `os.chmod` — that's a latent inconsistency from before the C2 invariant landed, NOT in C3.1 scope. Filed as a follow-up; cleanup is one-liner replacement to `atomic_write_secret`.

**C3 remaining scope.** The original Plan 4 scope (R3.6-lite, ~300 LOC):

- ~300 LOC for admin API + per-agent rate limiting + ring buffer +
  cluster key rotation (**C3**: `/v1/mesh/*` endpoints, per-agent rate limiting, request-trace ring buffer, cluster key rotation, `init-mesh` verb)
- 6 new doc files (mesh_operations.md, mesh_debug.md, CLAUDE.md updates,
  architecture updates)
- 2-node integration test fixture + a hard-testing manual smoke
  covering drain/resume/rotate-cluster-key

Attempting C in a single session would guarantee a shallow
implementation. It will ship as three sub-stages (C1 = R3.0, C2 =
R3.5-lite, C3 = R3.6-lite) across dedicated sessions, each with its
own pre-merge review round.

The **full scope for C is defined in the "Phases" section below**
(unchanged from v3 of this doc). Read the existing R3.0 / R3.5-lite /
R3.6-lite sections for the concrete deliverables, load-bearing
invariants, logging requirements, success criteria, and hard-testing
gate. Nothing in those sections is obsolete post Plans 2/3/3.5/3.6 —
the typed exception hierarchy, the `_MaximPeerBackend` contract, the
"HTTP fires first" cancellation pattern, and the Plan 3.6 R5 VRAM
spillover detection all compose cleanly with the Stage C admin API.

**Not obsolete after prior plans:** the per-agent rate limiting
(`KeyedRateLimiter` from `runtime/rate_limit.py`) is still cherry-picked
dormant code from Plan 1 R0 and is waiting for Stage C to light it up
at the router entry point.

**Still-open questions for Stage C kickoff:**
- Should Stage C1 (`mesh.yml`) merge the existing `peer.yml` schema
  instead of running alongside it? (Plan 3.6 R5 already added mesh.yml
  as a later canonical successor per its plan doc.)
- Does the dispatch-trace ring buffer need to be persistent (e.g.,
  JSONL rotation) or is in-memory + jq-on-MAXIM_LOG_FILE enough for
  post-mortem?

---

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

- **Previous plan:** [archive/llm_path_fast_failover.md](archive/llm_path_fast_failover.md) — prerequisite
- **Meta plan:** [llm_path_refinement.md](llm_path_refinement.md)
- **Foundation:** [archive/llm_path_foundation.md](archive/llm_path_foundation.md)
- **Architecture:** [../architecture/llm_routing.md](../architecture/llm_routing.md) — extended by this plan
- **Architecture:** [../architecture/mesh_operations.md](../architecture/mesh_operations.md) — created by this plan
- **Deferred:** [deferred/llm_path_multi_peer_dispatch.md](deferred/llm_path_multi_peer_dispatch.md) — multi-peer we chose NOT to build now
- **Deferred:** [deferred/llm_path_async_router.md](deferred/llm_path_async_router.md) — async router refactor
- **Deferred:** [deferred/llm_path_fair_scheduling.md](deferred/llm_path_fair_scheduling.md) — full fair-share (rate limiting is folded into this plan; fair-share is the bigger refactor)
- **Stress test results:** `docs/experiments/results/llm_path_stress_<date>.md`
- **Project guide:** [../../CLAUDE.md](../../CLAUDE.md)
