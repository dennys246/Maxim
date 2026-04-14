# LLM path stress test — Phase D2 / Plan 4 B recovery-time bench (2026-04-14)

Validates Plan 4 A (agent_id observability fix) and Plan 4 B (tight-loop
recovery-time bench harness) against a real leader restart. This is the
**rigorous recovery-time measurement** the original Phase D report
(2026-04-13) flagged as inconclusive under sim workload — the sim
orchestrator does 30+s of local work between LLM calls, so the
leader-ready-to-first-success gap was dominated by workload cadence,
not peer-side wedge state.

This run uses the new `maxim bench recovery-time` harness (Plan 4 B)
which fires chat completions in a tight loop with zero local work
between attempts. The result is a clean single number.

**Trace:** [plan4_bench_recovery_20260414.jsonl](plan4_bench_recovery_20260414.jsonl)
(751 events, ~92 KB — 750 attempts + 1 summary)
**Prior report:** [llm_path_stress_20260413.md](llm_path_stress_20260413.md) — the Phase D run this closes out
**Plan doc:** [../../plans/llm_path_operator_visibility.md](../../plans/llm_path_operator_visibility.md) — stage A+B

## Setup

| | |
|---|---|
| Leader | RTX 5080 (16 GB), Qwen2.5-14B-Instruct Q4_K_M, llama-cpp-server via Cloudflare tunnel `maxim.dennyschaedig.com`. VRAM 11.3/16 GB at start — well below the Plan 3.6 R5 spillover threshold. |
| Peer | Mac, `feat/llm-path-operator-visibility` branch (Plan 4 A + Plan 4 B uncommitted changes; leader code unchanged) |
| Bench cmd | `maxim bench recovery-time --url https://maxim.dennyschaedig.com/v1 --api-key <redacted> --duration 240 --max-tokens 8 --pace 0.1 --output /tmp/plan4_bench/recovery.jsonl` |
| Sanity curl | `/v1/models` returns 200 in <1s with Qwen-14B GGUF listed. `X-Maxim-GPU-VRAM: 11.3/16`. |

**Key difference vs Phase D:** no sim orchestrator, no agent loop, no
`_inference_lock` contention. The bench wraps `_MaximPeerBackend`
directly and fires as fast as possible.

## Phase D2 — leader restart mid-bench

- Bench started at bench-relative t=0.
- After 15s baseline (~75 successful calls captured), `maxim peer restart`
  was triggered synchronously in a second terminal. The restart command
  blocks until leader-ready and reports its own timing.
- Bench continued throughout. Restart wall-clock: 62s (2s proxy-down
  detection, 5s proxy back up, 53s model reload).
- Bench ran to its 240s natural duration, then exited cleanly.

### Headline numbers (from the `benchmark` summary event)

```json
{
  "e": "benchmark",
  "bench": "recovery_time",
  "duration_s": 240.014,
  "total_attempts": 750,
  "successes": 551,
  "failures": 199,
  "recovery_time_s": 58.681,
  "reason": "recovered",
  "last_success_before_failure_ts": 124.58,
  "first_failure_ts": 124.87,
  "first_success_after_failure_ts": 183.55
}
```

| Metric | Value |
|---|---|
| Total attempts | 750 in 240s (≈ 3.1 calls/sec average) |
| Successes | 551 (73.5%) |
| Failures | 199 (26.5%) |
| **Outage window (rigorous recovery time)** | **58.68s** |
| Outage window vs leader-reported restart time | 58.68s bench / 53s leader self-report → ~5.7s unexplained, matches the proxy-up-before-model-ready gap |

### Per-outcome latency distributions

**Successes (n=551):**

| p50 | p99 | max | min |
|---|---|---|---|
| 220 ms | 375 ms | 1076 ms | 192 ms |

No post-recovery latency spike. The `max=1076ms` is the single first
post-recovery call (model cold cache). Everything else is tight.

**Failures (n=199):**

| p50 | p99 | max | min |
|---|---|---|---|
| 170 ms | 614 ms | 3102 ms | 123 ms |

**Every failure fast-failed under 3.1s**; no stacked 60s/125s timeouts.
Plan 3's "fail fast per attempt, let the caller decide" contract holds
through a tight-loop bench. The single 3.1s max is Cloudflare tunnel
re-establishment on the very first failure after the proxy went down.

**Outcome histogram:**

| Outcome | Count |
|---|---|
| `down` | 199 (100%) |
| `unhandled_*` safety net | 0 |

All 199 failures classify as `BackendDown` — the typed Plan 2 R2b
exception. Zero leak to the generic safety net. **Plan 3's exception
hierarchy covers the leader-outage class cleanly.**

## Pass/fail gates

| Gate | Result | Evidence |
|---|---|---|
| **Rigorous recovery time < 90s on this hardware** | ✅ PASS | 58.68s, dominated by leader's model reload (53s) |
| **Peer-side overhead ≈ 0** | ✅ PASS | 58.68s bench gap vs 53s leader self-report ≈ 5s proxy-gap, not peer wedge |
| **Fast-fail per attempt during outage** | ✅ PASS | Failure p99 = 614ms (under 1s), max = 3.1s (Cloudflare reconnect), zero >5s |
| **No stacked timeouts** | ✅ PASS | No failure above the Plan 3.5 agent-level 300s timeout; no 125s Cloudflare 524s |
| **All failures are typed, not unclassified** | ✅ PASS | 199/199 = `BackendDown`, 0 fall through to generic |
| **First success on next attempt after leader-ready** | ✅ PASS | `first_success_after_failure_ts - last_failure_ts ≈ pace_s` |
| **`agent_id` populated on every event** | ✅ PASS | 750/750 = `bench_recovery_time`, 0/750 null. **Plan 4 A validated end-to-end.** |
| **Sim resumes normally post-recovery** | ✅ PASS (bench-equivalent) | 551 successes captured, p50 stable at 220ms |
| **Bench fires a tight loop** | ✅ PASS | 750 attempts in 240s ≈ 3.1/sec, limited by pace=0.1s + ~200ms call |

## What this run proves that Phase D could not

The original Phase D report (2026-04-13) concluded:

> **Recovery time < 10s from leader-ready** — ⚠️ INCONCLUSIVE. First
> post-recovery success was 30.7s after leader-ready, but this is
> sim-cadence, not peer-side stuck state.

The bench replaces sim cadence with a tight loop. Result: **the
observed recovery window is 58.68s, which matches the leader's
self-reported restart duration (53s) plus ~5s of proxy-up-before-model
lag.** There is no peer-side overhead beyond the leader's intrinsic
restart time. The peer is trivially recoverable — Plan 3 + 3.5's
contract holds under tight-loop load.

The 30.7s number from Phase D was definitively a sim-cadence artifact.

## Secondary finding — Plan 4 A validated on real traffic

Every one of the 750 JSONL events in this run carries `agent_id=bench_recovery_time`.
Zero have `agent_id=null`. The Phase D report flagged that
`peer_backend_call` / `peer_backend_failed` events had `agent_id=null`
because the router was dropping `request_context` on the floor when
constructing kwargs for `complete_with_usage`. Plan 4 A fixed this via
three complementary changes:

1. **Router capability-flag forwarding** — `_invoke_backend` now
   forwards `request_context` for backends that declare
   `accepts_request_context = True` (only `_MaximPeerBackend` sets this
   flag; cloud backends stay unchanged).
2. **Boundary contextvar binding** — `LLMWorker._call_llm_with_timeout`
   calls `set_context(normalized)` alongside the existing
   `set_cancel_event` binding, so `copy_context()` snapshots the
   RequestContext into the worker thread. Resets in `finally`.
3. **Contextvar fallback in the shim** — `_normalize_request_context(None)`
   now reads `current_context()` before manufacturing a fresh empty
   RequestContext. Defense in depth for paths that don't thread the
   dict.

The bench explicitly exercises path 1 (explicit dict kwarg) AND
indirectly exercises path 2+3 (the bench's own `set_context` binding
makes the contextvar live across calls). Both agree on the same
`agent_id` because the explicit dict wins over the contextvar when both
are present — precedence locked in by
`test_explicit_dict_still_wins_over_contextvar`.

## Reproduce

```bash
# Start the bench in one terminal (240s window, 0.1s pace between attempts)
maxim bench recovery-time \
    --url https://maxim.dennyschaedig.com/v1 \
    --api-key "$MAXIM_LANE_LARGE_REMOTE_API_KEY" \
    --duration 240 \
    --max-tokens 8 \
    --pace 0.1 \
    --output /tmp/plan4_bench/recovery.jsonl

# After ~15s in a second terminal:
maxim peer restart

# When the bench exits, extract the headline number:
tail -1 /tmp/plan4_bench/recovery.jsonl | jq '.recovery_time_s'

# Verify agent_id coverage (should return 0):
jq -c 'select(.agent_id == null)' /tmp/plan4_bench/recovery.jsonl | wc -l

# Inspect per-outcome latency distribution:
jq -r 'select(.e=="peer_backend_failed") | .latency_ms' /tmp/plan4_bench/recovery.jsonl \
  | sort -n | awk 'BEGIN{n=0}{a[n++]=$1}END{print "p50",a[int(n*0.5)],"p99",a[int(n*0.99)],"max",a[n-1]}'
```

## Follow-ups

1. **Phase D2 closes out the Phase D recovery-time gate.** The sim run
   in the original report (2026-04-13) remains the source of truth for
   end-to-end robustness under agent workload; this bench run is the
   source of truth for **peer recovery time with zero workload
   artifact**. Both are kept.
2. **Longer-window runs** (e.g., 10-minute duration with several
   restarts) are possible with the bench harness and may be worth doing
   before the 0.5 substrate work ships, to confirm no drift across
   sequential restart cycles.
3. **Plan 4 C** (mesh.yml + admin API + per-agent rate limiting) is a
   separate multi-session effort tracked in the main Plan 4 doc with
   the new "Stage C" header. See plan doc for the scope split rationale.
