# `maxim bench recovery-time` rerun guide

Step-by-step rerun of the Plan 4 B tight-loop recovery-time benchmark.
This is the rigorous recovery-time measurement that replaces the
sim-cadence-muddied gate from the original Phase D run (2026-04-13).

**Latest result:** [../results/llm_path_stress_plan4_20260414.md](../results/llm_path_stress_plan4_20260414.md) — recovery window = 58.68s on a 16 GB RTX 5080 running Qwen-14B Q4_K_M, matching the leader's self-reported 53s reload + ~5s proxy gap.

**Companion docs:**
- [llm_path_stress_test_rerun.md](llm_path_stress_test_rerun.md) — the end-to-end sim-workload Phase D runbook
- [llm_path_stress_test.md](llm_path_stress_test.md) — the full 4-6 hour protocol (A + B + C + D)

## What this bench tests

Measures **peer-side recovery time after a leader restart** without any
workload artifact from the sim orchestrator's local-work cadence. Fires
chat completions in a tight loop (configurable pace, default 0.1s
between attempts), records per-call latency + outcome, then analyses
the first `success → failure → success` transition to extract a
single recovery_time_s number.

Answers one specific question: **how long does the peer side take to
notice the leader is back and serve the next successful call?** The
answer is dominated by the leader's intrinsic restart time — there
should be effectively zero peer-side overhead.

### Two different gates, two different denominators

Plan 3's original recovery-time target was **"< 10s from
leader-ready to first-success"** — the denominator is the moment the
leader reports `LLM ready`, and the numerator is the gap between
that and the next successful peer call. That gate measures
**peer-side wedge latency**: how much dead time the peer adds
AFTER the leader is back.

This bench's `recovery_time_s` uses a **different denominator**:
from the first observed failure to the first observed success. The
denominator is when the PEER FIRST NOTICED the leader was down, and
the numerator includes the entire restart window (model reload +
proxy re-establishment + peer notices recovery). A 14B reload takes
~50-55s on its own, so this bench's `recovery_time_s` is NECESSARILY
larger than Plan 3's gate — they're measuring different things.

**Both numbers matter:**
- **Plan 3's gate (peer-side wedge latency):** < 10s. If the peer
  takes longer than this to notice recovery after `LLM ready`, that's
  a peer-side wedge — the thing Plan 3 killed.
- **This bench's gate (end-to-end outage window):** dominated by
  leader reload time. Expected: `recovery_time_s ≈
  leader_self_reported_restart + ~5s` (proxy-up gap). A materially
  larger value indicates peer-side overhead beyond leader reload.

The reference run (2026-04-14) measured 58.68s bench `recovery_time_s`
vs 53s leader self-report → ~5s gap → peer-side overhead ≈ 0s. Both
Plan 3's gate AND this bench's gate pass.

**Pre-fix baseline (Phase D, 2026-04-13):** sim-workload recovery gate
was 30.7s from leader-ready to first-success, but investigation showed
this was a workload artifact (the orchestrator was doing 30s of local
agent work between LLM calls). The true peer-side recovery overhead
was not measurable under sim cadence.

**Post-fix (Plan 4 B, 2026-04-14):** tight-loop bench measures 58.68s
total outage window vs 53s leader self-report — ~5s unexplained gap,
attributed to Cloudflare proxy-up-before-model-ready. Peer-side
overhead is effectively zero.

## Prerequisites

**Code state on peer:**
- `main` at or past PR #99 (Plan 3.6 R5 VRAM spillover detection)
- Plan 4 A+B shipped (this branch)
- `ruff check` clean, `python -m pytest tests/unit/test_bench_recovery_time.py tests/unit/test_llm_worker_cancellation.py tests/unit/test_maxim_peer_backend.py tests/unit/test_router_typed_exceptions.py -q` green

**Hardware:**
- **Leader**: GPU with enough VRAM headroom. Reference config: RTX 5080 (16 GB) running Qwen-14B Q4_K_M at `MAXIM_LLM_N_CTX` sized below the Plan 3.6 R5 spillover threshold. `maxim doctor` must show VRAM pressure ≤ 85% on the leader.
- **Peer**: the machine running `maxim bench`. No GPU requirement — the peer only sends HTTP, doesn't run the model.

**Peer config:**
```bash
# ~/.config/maxim/peer.yml must point at the leader's tunnel URL
cat ~/.config/maxim/peer.yml
# url: https://maxim.<your-domain>/v1
# api_key: <cluster-key>
# is_cloud: false
```

## Pre-flight health gate — CRITICAL

Same gate as the sim runbook. If the leader can't serve a healthy
inference in <5s, bench results are meaningless:

```bash
KEY=$(python3 -c "import yaml; print(yaml.safe_load(open('$HOME/.config/maxim/peer.yml'))['api_key'])")
URL=$(python3 -c "import yaml; print(yaml.safe_load(open('$HOME/.config/maxim/peer.yml'))['url'])")
MODEL=$(curl -s -H "Authorization: Bearer $KEY" "$URL/models" | python3 -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])")
echo "MODEL=$MODEL"

time curl -s -o /tmp/sanity.json -w "http_code=%{http_code}\n" --max-time 10 \
  -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"say hi\"}],\"max_tokens\":8,\"temperature\":0}" \
  "$URL/chat/completions"
```

**Pass:** real time < 3s, http_code=200, `/tmp/sanity.json` contains a
non-empty `choices[0].message.content`. On a healthy 5080 this lands
at ~200-400ms for a short completion.

**Fail:** time > 5s or http_code ≠ 200 → **STOP**. Investigate leader
VRAM pressure via `maxim doctor` first. See
[feedback_kv_cache_shared_gpu_spillover.md](../../../.claude/projects/-Users-dennyschaedig-Scripts-Maxim/memory/feedback_kv_cache_shared_gpu_spillover.md).

## Runbook

### Step 1 — Clean bench state

```bash
mkdir -p /tmp/plan4_bench
rm -f /tmp/plan4_bench/recovery.jsonl /tmp/plan4_bench/bench.log
```

### Step 2 — Start the bench (Terminal A)

```bash
KEY=$(python3 -c "import yaml; print(yaml.safe_load(open('$HOME/.config/maxim/peer.yml'))['api_key'])")
URL=$(python3 -c "import yaml; print(yaml.safe_load(open('$HOME/.config/maxim/peer.yml'))['url'])")

# 240s total window, 0.1s pace, JSONL output to file
maxim bench recovery-time \
    --url "$URL" \
    --api-key "$KEY" \
    --duration 240 \
    --max-tokens 8 \
    --pace 0.1 \
    --output /tmp/plan4_bench/recovery.jsonl
```

The bench prints live `peer_backend_call` / `peer_backend_failed` events
to stderr (one WARNING per failure, one INFO per success). stdout is
reserved for JSONL if you omit `--output`, so leave stdout untouched
while the bench runs.

**Pace choice rationale:** `--pace 0.1` is ~3-4 calls/sec, slow enough
to not hammer a fragile leader but tight enough that every gap in the
trace comes from the leader, not the pace. For stress-testing with no
pace restriction, use `--pace 0.0`.

### Step 3 — Baseline window (15s)

Wait for ~50 successful calls to establish the pre-outage baseline.
Terminal A's stderr will show a stream of `peer_backend_call` INFO
events. **Do not trigger the restart until you see at least 30 successes**
— fewer than that and the "last success before failure" anchor may be
noisy.

### Step 4 — Fire the restart (Terminal B)

```bash
# Times this out at 300s just in case the leader wedges
timeout 300 maxim peer restart
```

The command prints its own progress:
```
Leader is restarting (was up 2077.7s).
  Waiting for leader to come back... 2s[502] 5s
  Proxy is up, waiting for LLM model to load... 5s 10s 15s 20s 26s 31s 36s 42s 47s 52s
  Leader is back online (LLM ready, 53s).
```

Typical 14B reload is 50-55s. Note the wall-clock — you'll cross-check
against the bench's observed outage window.

### Step 5 — Let the bench finish

The bench runs to its `--duration 240` natural end (240s minus elapsed
by the time you trigger restart, so another ~2-3 minutes of post-recovery
observation). Do NOT kill it early — the post-recovery success window
is what validates that the peer resumed normally.

Alternatively, Ctrl+C in Terminal A exits the bench cleanly and writes
the JSONL with whatever was captured so far. The summary event is
always written last.

### Step 6 — Extract the headline number

```bash
# Recovery time in seconds
jq '.recovery_time_s' /tmp/plan4_bench/recovery.jsonl | tail -1

# Full summary event
jq 'select(.e == "benchmark")' /tmp/plan4_bench/recovery.jsonl

# Agent_id coverage check (should be 0 — regression guard for Plan 4 A)
jq -c 'select(.agent_id == null)' /tmp/plan4_bench/recovery.jsonl | wc -l

# Per-outcome latency distributions
jq -r 'select(.e=="peer_backend_call") | .latency_ms' /tmp/plan4_bench/recovery.jsonl \
  | sort -n | awk 'BEGIN{n=0}{a[n++]=$1}END{print "successes p50",a[int(n*0.5)],"p99",a[int(n*0.99)],"max",a[n-1]}'
jq -r 'select(.e=="peer_backend_failed") | .latency_ms' /tmp/plan4_bench/recovery.jsonl \
  | sort -n | awk 'BEGIN{n=0}{a[n++]=$1}END{print "failures p50",a[int(n*0.5)],"p99",a[int(n*0.99)],"max",a[n-1]}'

# Failure outcome histogram (should be 100% typed, zero unhandled_*)
jq -r 'select(.e=="peer_backend_failed") | .outcome' /tmp/plan4_bench/recovery.jsonl | sort | uniq -c
```

## Pass/fail gates

| Gate | Target | Reference (2026-04-14 run) |
|---|---|---|
| `recovery_time_s` not null | `reason == "recovered"` | ✅ 58.68s |
| Peer-side overhead | `recovery_time_s - leader_self_reported_restart ≈ 0-10s` | ✅ ~5s (proxy-up gap) |
| Failure latency p99 | < 2s (Plan 3 fast-fail contract) | ✅ 614ms |
| Failure latency max | < 5s (Cloudflare tunnel reconnect) | ✅ 3102ms |
| Success latency p50 | < 500ms | ✅ 220ms |
| Success latency p99 | < 1500ms | ✅ 375ms |
| Outcome histogram | 100% typed (`down`, `timeout`, `overloaded`, ...) | ✅ 199/199 `down` |
| `agent_id` coverage | 100% of events (Plan 4 A regression guard) | ✅ 750/750 |

## Pitfalls

**1. Bench started AFTER restart.** If Terminal B fires `maxim peer restart`
before Terminal A's bench is running, the first pre-outage baseline
will be missing. The bench's recovery analysis requires a
`success → failure → success` transition; a `failure → success` pattern
returns `reason = "no_pre_outage_success"` with `recovery_time_s = null`.

**2. Bench duration too short.** If the bench exits before the leader
recovers, you get `reason = "did_not_recover"`. For a 14B model
restart, use `--duration 240` minimum to give at least ~60s of
post-recovery observation window.

**3. Pace too tight on a fragile leader.** `--pace 0.0` hammers the
leader at ~5-10 req/sec. If the leader is already near capacity, this
can mask the measurement (the leader looks "degraded" rather than
"restarting"). Use `--pace 0.1` or `--pace 0.2` for baseline runs;
reserve `--pace 0.0` for dedicated stress tests.

**4. VRAM spillover on leader.** If `maxim doctor` shows VRAM > 93%
pressure on the leader, the pre-restart baseline will show degraded
latency (500ms-5s instead of 200-400ms), the restart will take longer
to drain, and the recovery window will be inflated by residual
spillover effects. Fix the leader first — see
[feedback_kv_cache_shared_gpu_spillover.md](../../../.claude/projects/-Users-dennyschaedig-Scripts-Maxim/memory/feedback_kv_cache_shared_gpu_spillover.md).

**5. Cloudflare tunnel dormancy.** If the leader has been idle for a
long time, the tunnel may be cold. Run the pre-flight curl 2-3 times
before starting the bench to warm up the tunnel; the first call can
be ~1-2s even on a healthy leader.

**6. Bench kwarg forwarding.** The bench wraps `_MaximPeerBackend`
directly and passes `request_context` explicitly on every call, so
it exercises both the router's capability-flag kwarg path (Plan 4 A.1)
AND the boundary contextvar path (A.2) simultaneously. If a future
refactor breaks either path, the `agent_id` coverage gate will
catch it — 750/750 drops to something less.

## Typical runtime

- 240s wall-clock for the bench duration
- + ~15s baseline before firing restart
- + ~60s restart wall-clock (blocking, in parallel with bench)
- + ~30s analysis

**Total: ~5-6 minutes for a full rigorous recovery-time measurement.**
Compared to the sim-workload Phase D run (15-20 minutes) this is ~3x
faster AND produces a cleaner number.

## When to run this

1. **Before any PR that touches** `router.py`, `llm_worker.py`,
   `_MaximPeerBackend`, `utils/http.py`, or `utils/cancellation.py`.
   The existing `tests/unit/test_llm_worker_cancellation.py` catches
   some regressions but not all — the real-hardware bench is the
   authoritative recovery-time signal.
2. **Before shipping a new version tag.** Regression catch for any
   subtle router-layer or cancellation-layer drift.
3. **After upgrading the leader's model or changing `MAXIM_LLM_N_CTX`.**
   The reload time is a function of model + n_ctx; the bench confirms
   the peer recovers cleanly at the new reload time.
4. **When debugging a suspected peer-side wedge.** If the sim is
   "stuck" and you want to rule out peer-side state, run the bench —
   if the bench shows clean recovery, the wedge is in the sim
   orchestrator, not the LLM path.
