# LLM path stress test — Phase D (2026-04-13)

Validates Plan 3 (Fast Failover, PR #94) + Plan 3.5 (Cancellation Hygiene, PR #96) against a real leader restart under live multi-turn agent load.

**Trace:** [phase_d_20260413.jsonl](phase_d_20260413.jsonl) (3052 events, 244 KB)
**Pre-requisite:** the KV cache spillover on the leader was resolved by lowering `MAXIM_LLM_N_CTX` from ~12380 before this run — see [feedback_kv_cache_shared_gpu_spillover.md](../../../.claude/projects/-Users-dennyschaedig-Scripts-Maxim/memory/feedback_kv_cache_shared_gpu_spillover.md). Without that fix, individual calls were 60-125s and no recovery-time measurement was meaningful.

## Setup

| | |
|---|---|
| Leader | RTX 5080, Qwen-14B-Instruct Q4_K_M, llama-cpp-server via Cloudflare tunnel `maxim.dennyschaedig.com` |
| Peer | Mac, `feat/llm-path-cancellation-hygiene` merged to main (commit `6a4f505`) |
| Sim | `maxim --sim "explore a small cave, describe what you see each turn, and decide what to do next" --seed 42` |
| Env | `MAXIM_BACKEND_TRACE=1 MAXIM_LOG_FILE=/tmp/phase_d_main.jsonl MAXIM_HEARTBEAT=1` |

**Sanity curl before sim:** 50 output tokens in 1.25s (~40 tok/s). Confirms leader is healthy post-spillover-fix.

**Baseline warmup:** 3 successful calls, 3.0-3.5s latency each, zero failures. Healthy floor.

## Phase D — leader restart mid-workload

- Sim ran in background to 47 successful calls (~45s of continuous agent activity).
- `maxim peer restart` fired at **t₀**.
- Leader reported "LLM ready" at t₀+53s (from the restart command's own progress output — proxy up at ~10s, model reload complete at 53s).
- Sim continued; 16 more successful calls observed post-recovery; sim was SIGINT'd after confirming steady state.

### Timeline (dt = seconds after restart trigger)

| dt | Event | Latency | Detail |
|---|---|---|---|
| 2.7 | `peer_backend_call` 200 | 1.9s | In-flight pre-restart request drains cleanly |
| **2.8** | `peer_backend_failed` 502 | **135 ms** | Fast fail — leader proxy is down |
| 2.8 | `dispatch_exhausted` | 138 ms | 1 attempt, `agent_id=llm_worker` |
| **19.5** | `peer_backend_failed` 502 | **468 ms** | Fast fail |
| 19.5 | `dispatch_exhausted` | 469 ms | 1 attempt |
| **49.9** | `peer_backend_failed` 502 | **599 ms** | Fast fail |
| 49.9 | `dispatch_exhausted` | 599 ms | 1 attempt |
| ~53 | Leader "LLM ready" (from restart command stderr) | — | — |
| **83.7** | `peer_backend_call` 200 | 4.2s | First post-restart success (includes model warmup hit) |
| 88.4–99.9 | 8 more `peer_backend_call` 200 | 1-4s each | Steady state restored |

### Pass/fail gates

| Gate | Result | Evidence |
|---|---|---|
| **No stacked agent-level timeouts** | ✅ PASS | Zero `LLM call timed out after 300.0s` warnings in the trace. Plan 3.5's "HTTP fires first" contract holds. |
| **Fast-fail per attempt** | ✅ PASS | All 3 failed attempts completed in ≤ 600 ms. Plan 3 killed the ~63s pre-fix fail-slow. |
| **`dispatch_exhausted` not cascading** | ✅ PASS | Each event is a single-attempt fast fail (`attempts: 1`), not a retry loop. |
| **No provider-state pollution** | ✅ PASS | Grepped the 45-90s window for `provider_silenced` / backoff log events — none present. Router state stayed clean. |
| **Sim continues after recovery** | ✅ PASS | 16 successful post-recovery calls at normal latency. No wedging. |
| **Recovery time < 10s from leader-ready** | ⚠️ INCONCLUSIVE | First post-recovery success was 30.7s after leader-ready, but this is sim-cadence (see below), not peer-side stuck state. |
| **`agent_id` populated throughout** | ⚠️ OBSERVABILITY GAP | `dispatch_exhausted` has correct `agent_id=llm_worker`, but `peer_backend_call` / `peer_backend_failed` have `agent_id=null`. Plan 4 material. |

### On the 30.7s recovery-time ambiguity

Literal measurement: leader-ready at dt≈53, first success at dt=83.7 → 30.7s gap. This appears to miss the <10s target. But investigation shows this is **sim cadence, not peer-side stuck state**:

- Failed attempts were spaced at dt=2.8, 19.5, 49.9 — gaps of 16.7s and 30.4s between attempts. **Neither gap correlates to router backoff** (none observed).
- The gap between dt=49.9 (last failure) and dt=83.7 (first success) was 33.8s. During that window the trace shows 249 `log` events and 146 `percept` events — the orchestrator was busy with local agent work (reasoning, tool execution, memory formation) between LLM calls.
- When the peer did attempt a call after leader-ready (dt=83.7), it **succeeded in one shot** — no warmup retries, no router cooldown, no held locks.

**Interpretation:** the peer is trivially recoverable under this workload. The measurement is muddied because a sim workload doesn't fire LLM calls continuously. A rigorous recovery-time measurement would need a tight benchmark loop that fires a request the instant the previous one completes or fails.

## Verdict

**Plan 3 + Plan 3.5 pass Phase D on the axes they were designed for:** fast-fail, no stacked timeouts, no state pollution, sim continues. The strict recovery-time gate is inconclusive under sim workload, but there is no evidence of peer-side stuck state — the pre-fix symptom (60s stacked timeouts cascading across turns) is gone.

## Follow-ups

1. **Observability gap: `agent_id` on `peer_backend_call` / `peer_backend_failed`.** Router-side `dispatch_exhausted` correctly attributes to `llm_worker`, but backend-side events don't pull from the `RequestContext` contextvar. Likely a one-line fix in `_MaximPeerBackend._log_failure` and the success-path logger. **Plan 4 scope.**
2. **Phase D protocol ambiguity: recovery-time measurement.** Should specify "use a benchmark workload for recovery timing; use a sim for end-to-end robustness under agent workload". Worth a one-paragraph refresh in [docs/plans/llm_path_fast_failover.md](../../plans/llm_path_fast_failover.md) lines 405-470.
3. **Latent streaming bug found during the parallel 125s latency investigation:** [utils/http.py:1028](../../../src/maxim/utils/http.py#L1028) is missing `_stream_ctx=stream_ctx` in the `StreamingResponse` constructor — same class of bug as the `raw_proxy_forward_streaming` fix in commit `627727e`. Dormant today because LLMWorker defaults `stream=False`, but will bite the moment streaming is turned on. Tracked separately.

## Reproduce

```bash
rm -f /tmp/phase_d_main.jsonl
MAXIM_BACKEND_TRACE=1 MAXIM_LOG_FILE=/tmp/phase_d_main.jsonl MAXIM_HEARTBEAT=1 \
  maxim --sim "explore a small cave, describe what you see each turn, and decide what to do next" --seed 42 &
SIM_PID=$!
# wait for ~30+ successful calls to confirm steady state
# then:
date +%s && maxim peer restart
# wait ~60s to observe recovery; then:
kill -INT $SIM_PID
# analyse:
jq -c 'select(.e | test("peer_backend|dispatch"))' /tmp/phase_d_main.jsonl
```
