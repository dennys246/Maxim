# LLM Path Refinement — meta-plan

**Status:** Draft v4 — four sub-plans + three deferred shell plans
**Scope:** ~1,800 LOC new + ~1,330 LOC deleted across four sub-plans
**Target version:** **0.4** (single stability version containing all LLM path sub-plans)
**Last updated:** 2026-04-12
**Deployment model:** single-tenant (one user controls their leader via API key; multi-agent under that key is in scope; multi-tenant user isolation is NOT)

## What this is

A four-part refinement of Maxim's LLM routing path, motivated by two back-to-back peer-leader incidents on 2026-04-12 and an architecture audit that revealed most of the infrastructure we needed was already present but fragmented or wrong-shaped.

**The core insight:** the existing `LLMRouter._complete_text_locked` already implements reactive provider fallback natively. We don't need a new dispatcher class. We need to (1) clean the foundations, (2) add correctness primitives (typed errors, two-stage probe, role detection), (3) replace the retry-bloated `_OpenAIBackend` for self-hosted peers with a purpose-built backend, and (4) give operators visibility into the routing path.

**Multi-agent, single-user context:** concurrent agents (AgentPool, NPC campaigns) running under one user's API key are first-class. Multi-tenant user isolation is out of scope.

## The four sub-plans

All four ship under version **0.4** as a single "major stability" milestone per user decision.

### Plan 1: Foundation Cleanup — [llm_path_foundation.md](llm_path_foundation.md)

**~450 LOC new, ~1,330 LOC deleted. Pure refactoring.**

- **R0** — Delete ~1,250 LOC of dead mesh scaffolding (`peer_registry`, `peer_info`, `peer_channel`, `task_delegation`, `knowledge`, `clock`, `agent_identity`). Cherry-pick `admission.py` rate-limiting into `runtime/rate_limit.py` before deleting.
- **R1** — New `maxim/utils/http.py`: endpoint registry, typed `HTTPError` hierarchy, connection pooling, `RequestContext` dataclass propagated via `contextvars.ContextVar`, automatic `X-Maxim-*` header propagation (including `X-Maxim-Protocol-Version: 1`). Eleven scattered urllib call sites collapse into one registry.

**Ship when:** zero `urllib.request.urlopen` outside `utils/http.py`, fast suite green, manual smoke passes.

### Plan 2: Typed Errors + Role Detection — [llm_path_typed_errors.md](llm_path_typed_errors.md)

**~280 LOC new. Split out of former Plan 1 per user decision — each sub-phase ships and tests independently.**

- **R2a** — Role-scoped persistence (`active_llm_model.{role}.txt`), explicit `detect_role()` called as first line of `cli.py::main()`
- **R2b** — Typed `BackendError` taxonomy with `.fix_hint` on every class; backcompat shim for `request_context["agent"]` → `agent_id`
- **R2c** — Two-stage probe (liveness + readiness), new `inference_broken` outcome with 15s cache TTL
- **R2d** — Move SSRF check from `openai_backend.py` → `maxim/utils/net.py`

**Ship when:** all four sub-phases have independent tests green, first startup log line is `event=role_detected`, all typed exceptions have `.fix_hint`.

### Plan 3: Fast Failover — [llm_path_fast_failover.md](llm_path_fast_failover.md)

**~420 LOC new, ~-80 LOC deleted. The single most important reliability win.**

- **R2.5** — New `_MaximPeerBackend` in `models/language/maxim_peer_backend.py`. Purpose-built for self-hosted peers: single HTTP call per `complete_with_usage`, raises typed exceptions, no retry loop, supports streaming via httpx native. Router integration: `_try_provider` catches typed exceptions and applies per-class backoff. Aggregated failure logging.
- **R2.6** — Delete three parallel probe implementations. All callers route through `_MaximPeerBackend.health_check()`.

**The performance gate:** `backend_call_duration_seconds` p99 < 5s against mocked dead peer. Pre-plan baseline is ~52s due to `_OpenAIBackend`'s hidden retry loop. **This is almost certainly why `maxim peer restart` feels broken today.**

**Ship when:** p99 gate met, zero `retry|backoff|gateway` in `maxim_peer_backend.py` (CI grep), stress test protocol complete.

### Plan 4: Operator Visibility — [llm_path_operator_visibility.md](llm_path_operator_visibility.md)

**~650 LOC new. Ships unconditionally — operator visibility is always valuable.** Renamed from "Reactive Mesh" per stress-test-driven scope decision.

- **R3.0** — `maxim peer list-nodes`, `--node X status/install/drain/resume/health/refresh`, new `mesh.yml` with `self:`, `protocol_version: 1`, `agent_rate_limits:`. Schema validation + doctor integration.
- **R3.5-lite** — `install` command + VRAM precheck (prevents Mac-downloads-14B-can't-load-it).
- **R3.6-lite** — Admin API on `/v1/mesh/*`: state, health, per-agent stats, request-trace with agent/session filtering, drain/resume, refresh, cluster key rotation (atomic POST-to-every-peer). Per-agent token-bucket rate limiting at router entry (~70 LOC — prevents runaway agent starvation). Admin API rate limiting.

**Ship when:** admin endpoints auth-gated + rate limited, per-agent rate limiter enforces caps, cluster key rotation test passes.

## Dependency chain

```
Plan 1 (Foundation, R0 + R1, ~450 LOC new + 1,330 deleted)
      ▼
  [Hard test + doc update + memory update]
      ▼
Plan 2 (Typed Errors, R2a-R2d, ~280 LOC new)
      ▼
  [Hard test + doc update + memory update]
      ▼
Plan 3 (Fast Failover, R2.5 + R2.6, ~420 LOC new)
      ▼
  [Hard test + doc update + memory update]
      ▼
  [STRESS TEST — Phases A-E per protocol doc]
  [Triple duty: Plan 3 validation + substrate P2 + batching PoC]
      ▼
Plan 4 (Operator Visibility, R3.0 + R3.5-lite + R3.6-lite, ~650 LOC new)
      ▼
  [Hard test + doc update + memory update]
      ▼
  [0.4 released]
```

Each plan has three mandatory finish steps:
1. **Hard testing checkpoint** — pytest + manual smoke + rollback drill
2. **Documentation update** — `reference.md`, `architecture/llm_routing.md` (drafted in Plan 1 R0, extended by each plan), `architecture/mesh_operations.md` (Plan 4), `CLAUDE.md`, relevant `troubleshooting/*`
3. **Memory update** — project memory file per plan + `MEMORY.md` pointer

## The stress test protocol (between Plan 3 and Plan 4)

Full protocol in [../experiments/protocols/llm_path_stress_test.md](../experiments/protocols/llm_path_stress_test.md). Triple duty:

1. **Plan 3's 52-second-retry-loop kill validation** (Phase D — leader restart mid-workload)
2. **Substrate P2 reward modulation under multi-agent load** (Phase A baseline + Phase B fan-out) — user decision to run both together
3. **`llama.cpp --parallel` batching PoC** (Phase C sweep) — user decision to measure whether batching obviates multi-peer dispatch

**Decision inputs from Phase C for Plan 4 scope:**
- Batching solves saturation → Plan 4 ships as scoped (R3.0 + R3.5-lite + R3.6-lite), multi-peer dispatch stays deferred
- Batching helps but peer overflow still needed → same Plan 4 scope + revive [deferred/llm_path_multi_peer_dispatch.md](deferred/llm_path_multi_peer_dispatch.md)
- Batching has no effect → investigate deeper (VRAM, context, quantization) before Plan 4 decision

**Stress test report:** `docs/experiments/results/llm_path_stress_<date>.md`. Protocol doc has template.

## The three deferred shell plans

| Deferred plan | Revive when | Scope |
|---|---|---|
| [deferred/llm_path_multi_peer_dispatch.md](deferred/llm_path_multi_peer_dispatch.md) | Post-stress-test shows leader saturation that `--parallel` doesn't solve | ~250 LOC |
| [deferred/llm_path_async_router.md](deferred/llm_path_async_router.md) | Multi-agent wait-time p99 > backend call p99 (lock is bottleneck) | ~800-1,200 LOC |
| [deferred/llm_path_fair_scheduling.md](deferred/llm_path_fair_scheduling.md) | Per-agent rate limits insufficient; need priority classes. **Aim for bio-inspired scheduling, not Kubernetes-style copy.** | ~300-500 LOC |

Shell plans are thin design sketches — not commitments. They preserve context for when triggers fire.

## Why this structure

**1. Plan 1 is pure foundations.** Ships incrementally (file-by-file migration), rolls back via revert.

**2. Plan 2 is split from Plan 1 per user request.** Role detection, typed errors, two-stage probe, SSRF move are each small and independent. Shipping each with its own tests means tighter feedback loops.

**3. Plan 3 is the "fix the 52-second retry loop" win.** Standalone deliverable. Plan 3 alone may be enough for single-user single-peer workloads.

**4. Plan 4 is operator visibility, not distribution.** Multi-peer dispatch moved to deferred. What Plan 4 actually ships: CLI, admin API, per-agent observability, per-agent rate limiting. Distribution becomes a deferred extension that revives if stress tests demand it.

**5. Multi-agent is first-class; multi-user is not.** `agent_id` / `session_id` / `request_id` propagate end-to-end via `RequestContext` + contextvars. Per-agent rate limiting prevents starvation. Per-agent admin API stats. **But** there's no cross-user isolation — one API key, one cluster, one user.

## Multi-agent lens — design decisions (applied across all sub-plans)

**1. `RequestContext` dataclass** replaces untyped `request_context: dict`. Typed fields: `request_id`, `agent_id`, `session_id`, `lane`, `parent_request_id`. Propagated via `contextvars.ContextVar` so callers don't thread it through function signatures.

**2. `X-Maxim-*` header contract** is the versioned wire protocol between nodes. Documented in architecture doc. Changing header names = breaking change + protocol version bump.

**3. Structured logging via existing `utils/structured_logging.py`.** Plans 1-4 use `log_structured()` for new events. Existing ~1,450 `logger.info/warning` calls stay unchanged. Dual-format: human-readable stdout (unchanged), JSONL file output when `MAXIM_LOG_FILE=...` is set.

**4. Header input sanitization at R1 boundary.** Control chars, CR/LF, non-ASCII, >256 char values rejected. Prevents log injection from user-controlled content.

**5. Per-agent rate limiting in Plan 4** — token bucket BEFORE acquiring `_inference_lock`. Prevents runaway agent starvation. ~70 LOC.

**6. Metric cardinality bounded** — no `agent_id` on hot-path metric labels. Per-agent debugging uses JSONL logs + Plan 4 request-trace filtering. Only configured agents get metric labels; rest aggregate as `agent_id="__other__"`.

**7. `_inference_lock` serialization acknowledged.** Plan 3 shortens worst-case lock hold from 52s to ~2s. Full async is [deferred/llm_path_async_router.md](deferred/llm_path_async_router.md).

**8. Data sovereignty documented.** Architecture doc has "What crosses node boundaries" section. Single-tenant assumption explicit.

**9. Protocol version header** — `X-Maxim-Protocol-Version: 1` on every outbound call. Receivers log warnings on unknown versions but process normally (forward compat).

**10. Concurrency safety of drain state** — reader-writer lock primitives introduced in Plan 2, consumed by Plan 4.

## Platform standards (all four sub-plans)

Non-negotiable:

1. **Timeouts layered + named.** `TimeoutPolicy` from `utils/http.py`.
2. **Typed error taxonomy.** `.fix_hint` on every class. Catching `Exception` forbidden in hot paths (safety net counter instead).
3. **Metrics on hot paths.** Via `lane_metrics.metrics_snapshot()`. Bounded cardinality.
4. **Request IDs propagate end-to-end.** Via `RequestContext` + `X-Maxim-*` headers.
5. **Feature flags gate new behavior.** One week default bake-in, unlimited available.
6. **Conftest autouse scrub for every env var** (load-bearing per CLAUDE.md P5 lesson).
7. **Config validated on read.** Startup errors are line-numbered.
8. **No disk persistence of rederivable state.** Three durable files: `mesh.yml`, `active_llm_model.{role}.txt`, `probe_cache.json`.
9. **Every new env var in CLAUDE.md table in the same PR.**
10. **Dual-format logging.** Human stdout (unchanged), JSONL file via existing `StructuredFormatter`. New events use `log_structured()`.
11. **Every sub-plan has a hard testing checkpoint + documentation step.**
12. **Every sub-plan runs a pre-merge review round.** Two parallel review Claudes (Executor + Architecture lens) against the branch tip BEFORE the PR opens/merges. Findings fold into the same branch via a follow-up commit. No `fix/<plan>-loose-ends` split-PR pattern — that was R1's approach and got abandoned after R2 proved pre-merge review timing works. See [feedback_review_before_ship.md](../../.claude/projects/-Users-dennyschaedig-Scripts-Maxim/memory/feedback_review_before_ship.md) for evidence + templates. Skipping the review round is gambling — both R1 and R2 had bugs that passed 4000+ unit tests.

## Env var inventory (six new vars total)

| Var | Default | Purpose | Plan |
|---|---|---|---|
| `MAXIM_ROLE` | auto | Override role detection (leader/peer/solo) | Plan 2 R2a |
| `MAXIM_HTTP_TRACE` | 0 | Verbose HTTP logging (INFO + headers) | Plan 1 R1 |
| `MAXIM_LOG_FILE` | unset | JSONL log file path | Plan 1 R1 |
| `MAXIM_BACKEND_TRACE` | 0 | Verbose backend call logging | Plan 3 R2.5 |
| `MAXIM_MESH` | 0 | Enable Plan 4 mesh features | Plan 4 R3.0 |
| `MAXIM_MESH_VERBOSITY` | 0 | 0=metrics only, 1=decision, 2=verbose | Plan 4 R3.6-lite |

Plus `MAXIM_CLUSTER_KEY` (cluster bearer token) and `MAXIM_REQUEST_TRACE_SIZE` (ring buffer size) from Plan 4.

**Bake-in policy:** one week dogfooding per flag, unlimited available if issues found. Conftest autouse scrub for every flag.

## Current state tracking

| Sub-plan | Status | Checkpoint | Memory updated |
|---|---|---|---|
| Plan 1 R0 (dead mesh delete) | ✅ SHIPPED 2026-04-11 (commit `e811787`) | ✅ Done | ✅ [project_llm_path_r0_shipped.md](../../.claude/projects/-Users-dennyschaedig-Scripts-Maxim/memory/project_llm_path_r0_shipped.md) |
| Plan 1 R1 (utils/http.py + 9 migrations) | ✅ SHIPPED 2026-04-12 (PRs #88, #90, pending cleanup PR for `c8a07e9`) | ✅ Done (4003 passed, CI grep CLEAN, smoke green) | ✅ [project_llm_path_r1_shipped.md](../../.claude/projects/-Users-dennyschaedig-Scripts-Maxim/memory/project_llm_path_r1_shipped.md) |
| Plan 2: Typed Errors (R2a-d) | ✅ SHIPPED 2026-04-12 (branch `feat/llm-path-r2`) | ✅ Done (4073 passed, CI grep CLEAN, smoke green with `role_detected` first event) | ✅ [project_llm_path_r2_shipped.md](../../.claude/projects/-Users-dennyschaedig-Scripts-Maxim/memory/project_llm_path_r2_shipped.md) |
| Plan 3: Fast Failover (R2.5+R2.6) | ✅ SHIPPED 2026-04-12 (PR #94, `ce5f034`) | ✅ Done (4142 passed, 3 CI grep invariants CLEAN, smoke green with `peer_backend_call` multi-agent context propagation) | ✅ [project_llm_path_r3_shipped.md](../../.claude/projects/-Users-dennyschaedig-Scripts-Maxim/memory/project_llm_path_r3_shipped.md) |
| Stress test (A-E) | Draft protocol | ▶ **READY** (Phase D measures post-Plan-3 leader restart delta vs ~63s pre-baseline) | N/A |
| Plan 4: Operator Visibility (R3.x) | Draft v2 | ▶ **READY TO START** (no longer blocked — scoped down per user decision; multi-peer revival waits on Phase D results) | ⏸ Pending |
| Deferred: multi-peer dispatch | Shell plan | ⏸ Revive trigger TBD | N/A |
| Deferred: async router | Shell plan | ⏸ Revive trigger TBD | N/A |
| Deferred: fair scheduling | Shell plan | ⏸ Revive trigger TBD + bio-inspired aspiration | N/A |

Update this table as each plan ships.

### Plan 1 R1 — lessons for future sub-plans

The R1 execution surfaced five load-bearing patterns that Plans 2-4 must follow. See [project_llm_path_r1_shipped.md](../../.claude/projects/-Users-dennyschaedig-Scripts-Maxim/memory/project_llm_path_r1_shipped.md) for the full set, summarized here:

1. **Error-handling idioms stabilize in an early commit.** R1 converged on `e.status` / `e.response.json()` / `e.fix_hint` as the three `HTTPError` access patterns through steps 2-3. Plan 2 R2b's `BackendError` MUST mirror this shape so Plan 3's `except HTTPRateLimited as e: raise BackendOverloaded(e.retry_after_s, ...)` bridge is trivial.
2. **Migration steps are conceptually parallel but depend on idiom stability.** R1's 9 migration steps could have parallelized in principle, but steps 4+ copied the typed-error handling idiom from step 2 (`probe_llm_server`). If two executors had worked on steps 2 and 4 in parallel, they would have converged on different shapes. Future multi-step migration plans should say "the first 1-2 steps stabilize the idiom before later steps copy it."
3. **Subcommand dispatch in `cli.py::main` bypasses logging setup.** Any feature with an "emit at startup" contract (Plan 2 R2a's `event=role_detected` is a hot candidate) needs `configure_logging` called at the TOP of `main()` before subcommand dispatch, not at sim-loop entry. R1 cleanup commit `c8a07e9` already fixed this for MAXIM_LOG_FILE; Plan 2 inherits the fix.
4. **JSONL log format uses single-letter keys** (`t`/`l`/`s`/`e`). Runbook jq examples need `.e`, not `.event`. Every new runbook inherits this.
5. **Shared test helpers.** R1 copy-pasted an `http.Response(...)` construction stub 8 times across test files. Plan 2 should add `tests/conftest.py::make_http_response(status=200, body={})` + `make_backend_error(...)` before starting, so later plans aren't tempted to copy-paste again.

## Motivation — the 2026-04-12 incidents

Two bugs found in one session, both caused by structural fragility:

1. **Stale persisted profile clobbered peer config** — `build_primary_router` restored `MAXIM_LLM_PROFILE=qwen2.5-14b-instruct` from `active_llm_model.txt`, then `_apply_local_llm_override` interpreted that as "user wants local" and cleared the peer's `remote_url`. Commit `d875fb9`.

2. **Cloudflare Bot Fight Mode blocked the probe** — `probe_llm_server` was the one urllib call site that forgot to set `User-Agent: maxim-peer/1.0`. Cloudflare returned HTTP 403 + error 1010. Commit `8b52cbd`.

Both were one-line bugs caused by responsibility scattered across many files without a unified contract. The four sub-plans fix the collective pattern.

## Related docs

- **Project guide:** [../../CLAUDE.md](../../CLAUDE.md)
- **Architecture reference:** [../reference.md](../reference.md)
- **Architecture (drafted in Plan 1 R0, extended by each plan):** [../architecture/llm_routing.md](../architecture/llm_routing.md)
- **Architecture (created by Plan 4):** [../architecture/mesh_operations.md](../architecture/mesh_operations.md)
- **Stress test protocol:** [../experiments/protocols/llm_path_stress_test.md](../experiments/protocols/llm_path_stress_test.md)
- **Deferred plans:** [deferred/llm_path_multi_peer_dispatch.md](deferred/llm_path_multi_peer_dispatch.md), [deferred/llm_path_async_router.md](deferred/llm_path_async_router.md), [deferred/llm_path_fair_scheduling.md](deferred/llm_path_fair_scheduling.md)
- **Troubleshooting (created by sub-plans):** `http_debugging.md` (Plan 1), `peer_backend_debug.md` (Plan 3), `mesh_debug.md` (Plan 4)
- **Related plans:**
  - [substrate_recognition.md](substrate_recognition.md) — P2 validation runs alongside Plan 3's stress test (Phase A)
  - [tool_refinement_plan.md](tool_refinement_plan.md) — tools migrate in Plan 1 R1 step 4
- **Related incident commits:** `d875fb9` + `8b52cbd` on main
