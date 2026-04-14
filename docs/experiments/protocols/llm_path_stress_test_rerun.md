# LLM path stress test — Phase D rerun guide

Step-by-step rerun of the Phase D leader-restart stress test. Companion to [llm_path_stress_test.md](llm_path_stress_test.md) (the full 4-6 hour protocol) — this doc covers **only Phase D** (the ~15-minute single-restart validation run), with exact commands, timing tricks, and pitfalls that bit the 2026-04-13 run.

**Latest result:** [../results/llm_path_stress_20260413.md](../results/llm_path_stress_20260413.md) — all Plan 3 / Plan 3.5 gates PASS.

## What Phase D tests

One specific claim: **after a mid-workload leader restart, the peer recovers without stacked agent-level timeouts, fast-fails each attempt in < 1s, and resumes on the first post-restart attempt.** This validates Plan 3 (fast failover, PR #94) + Plan 3.5 (cancellation hygiene, PR #96) together on real hardware.

Pre-fix symptom (2026-04-12 incident): peer took ~63 seconds to recover after `maxim peer restart`, driven by `_OpenAIBackend`'s internal gateway-retry loop amplified by `_inference_lock` head-of-line blocking + stacked 60s agent-level timeouts.

## Prerequisites

**Code state:**
- `main` at or past commit `6a4f505` (PR #96 — Plan 3.5 merged).
- `ruff check` clean, `python -m pytest tests/unit/test_http_client.py tests/unit/test_maxim_peer_backend.py tests/unit/test_cancellation.py tests/unit/test_llm_worker_cancellation.py -q` green.

**Hardware:**
- **Leader**: GPU with enough VRAM headroom (see "Pre-flight health gate" below). RTX 5080 + Qwen-14B Q4_K_M + reduced `MAXIM_LLM_N_CTX` is the reference config.
- **Peer**: separate machine running the agent loop, with `~/.config/maxim/peer.yml` pointing at the leader's tunnel URL.

**Env setup on peer:**
```bash
cd /path/to/Maxim
# Kill any stale sim process that could hold GPU/port state
pkill -f "maxim.*sim" 2>/dev/null; sleep 1

# Versions match?
maxim peer version
# If leader is behind, with explicit user OK:
# maxim peer update && maxim peer restart
```

## Pre-flight health gate — CRITICAL

**If the leader can't serve a healthy inference in <5s, Phase D results are meaningless.** The 2026-04-13 run was blocked by a KV cache spillover to shared GPU memory (Qwen-14B at `n_ctx=12380` pushed total VRAM past 16 GB, dropping rate to ~0.9 tok/s). See [feedback_kv_cache_shared_gpu_spillover.md](../../../.claude/projects/-Users-dennyschaedig-Scripts-Maxim/memory/feedback_kv_cache_shared_gpu_spillover.md).

**Gate:**
```bash
KEY=$(python3 -c "import yaml; print(yaml.safe_load(open('$HOME/.config/maxim/peer.yml'))['api_key'])")
URL=$(python3 -c "import yaml; print(yaml.safe_load(open('$HOME/.config/maxim/peer.yml'))['url'])")
MODEL=$(curl -s -H "Authorization: Bearer $KEY" "$URL/models" | python3 -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])")
echo "MODEL=$MODEL"

time curl -s -o /tmp/sanity.json -w "http_code=%{http_code}\n" --max-time 30 \
  -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Count 1 to 20 one per line.\"}],\"max_tokens\":200,\"temperature\":0}" \
  "$URL/chat/completions"
python3 -c "import json; d=json.load(open('/tmp/sanity.json')); print('usage:', d.get('usage'))"
```

**Pass:** real time < 5s, usage shows ~50-70 completion_tokens. On a healthy 5080 this lands at ~1.3s / ~40 tok/s.

**Fail:** time > 10s or http_code=524 → **STOP**. Do not proceed. Investigate leader VRAM (`nvidia-smi` on leader should show at least ~1-2 GB dedicated-VRAM headroom; if pegged near physical max, lower `MAXIM_LLM_N_CTX` and restart llama-cpp-server before continuing).

## Phase D runbook

### Step 1 — Baseline warmup (optional but recommended)

Confirms the sim → backend → leader path is clean before introducing the restart.

```bash
rm -f /tmp/phase_d_warmup.jsonl
MAXIM_BACKEND_TRACE=1 MAXIM_LOG_FILE=/tmp/phase_d_warmup.jsonl MAXIM_HEARTBEAT=1 \
  maxim --sim scenarios/campaigns/heist_v1.yaml --seed 42 --sim-max-turns 5
```

Cap at ~5 min; Ctrl+C if it stalls. After it finishes:

```bash
# Per-call latency + token counts
jq -c 'select(.e=="peer_backend_call") | {t, latency_ms, in: .input_tokens, out: .output_tokens}' /tmp/phase_d_warmup.jsonl

# Any failures?
jq -c 'select(.e=="peer_backend_failed" or .e=="dispatch_exhausted")' /tmp/phase_d_warmup.jsonl
```

**Pass:** 3+ successful calls at 1-5s latency, zero failures. If you see anything else, stop and diagnose — Phase D on a flaky baseline is noise.

### Step 2 — Phase D main run

**Key insight:** `scenarios/campaigns/heist_v1.yaml` only runs ~3 turns before terminating, which is **too short** to inject a restart mid-flow. Use a generative goal instead for a wider window:

```bash
rm -f /tmp/phase_d_main.jsonl /tmp/phase_d_sim.out
MAXIM_BACKEND_TRACE=1 MAXIM_LOG_FILE=/tmp/phase_d_main.jsonl MAXIM_HEARTBEAT=1 \
  maxim --sim "explore a small cave, describe what you see each turn, and decide what to do next" \
  --seed 42 > /tmp/phase_d_sim.out 2>&1 &
SIM_PID=$!
echo "sim_pid=$SIM_PID" | tee /tmp/phase_d_sim.pid
```

**Wait for the sim to reach steady state** — watch for 20+ successful `peer_backend_call` events in the log before triggering the restart:

```bash
# Poll until you see enough activity
while true; do
  COUNT=$(jq -c 'select(.e=="peer_backend_call")' /tmp/phase_d_main.jsonl 2>/dev/null | wc -l | tr -d ' ')
  echo "$(date +%H:%M:%S) calls=$COUNT"
  [ "$COUNT" -ge 20 ] && { echo "READY"; break; }
  sleep 2
done
```

**Trigger the restart. RECORD THE TIMESTAMP.**

```bash
# Open a second terminal (or use a subshell) so the sim keeps running
RESTART_T=$(date +%s)
echo "RESTART_T=$RESTART_T" | tee /tmp/phase_d_markers.txt
maxim peer restart
# The command returns when the leader reports "LLM ready" — note that wall-clock
# time too. On a 16GB GPU loading Qwen-14B Q4 this is ~50-55s.
echo "LEADER_READY=$(date +%s)" | tee -a /tmp/phase_d_markers.txt
```

**Let the sim run ~60s more** to capture post-recovery behavior:

```bash
while true; do
  POST=$(jq -c --argjson t "$RESTART_T" 'select(.e=="peer_backend_call" and .t > $t + 50)' /tmp/phase_d_main.jsonl 2>/dev/null | wc -l | tr -d ' ')
  echo "post-recovery calls=$POST"
  [ "$POST" -ge 5 ] && break
  sleep 3
done
```

**Graceful shutdown:**

```bash
PID=$(cat /tmp/phase_d_sim.pid | grep -oE '[0-9]+')
kill -INT $PID
# Wait a couple seconds; if still alive, kill harder
sleep 3
kill -0 $PID 2>/dev/null && kill -TERM $PID
kill -0 $PID 2>/dev/null && kill -KILL $PID
```

### Step 3 — Extract the gates

```bash
RESTART_T=$(grep RESTART_T /tmp/phase_d_markers.txt | cut -d= -f2)
echo "restart_t=$RESTART_T"

# Timeline in the restart window (just backend + router events, suppress percept/log noise)
jq -c --argjson t "$RESTART_T" 'select(.t >= $t and .t <= $t + 120) |
  select(.e | test("peer_backend|dispatch|http_request")) |
  {dt: ((.t - $t) * 10 | round / 10), e, status, lat: .latency_ms}' /tmp/phase_d_main.jsonl

# Gate 1: agent-level timeouts (must be empty)
jq -c 'select((.message // "") | test("timed out after 300"))' /tmp/phase_d_main.jsonl

# Gate 2: dispatch_exhausted details (each should be attempts=1, not a retry cascade)
jq -c --argjson t "$RESTART_T" 'select(.e=="dispatch_exhausted" and .t >= $t and .t <= $t + 120) |
  {dt: ((.t - $t) * 10 | round / 10), elapsed: .total_elapsed_ms, attempts: (.attempts | length)}' /tmp/phase_d_main.jsonl

# Gate 3: provider backoff / silencing during the gap (should be empty)
jq -c --argjson t "$RESTART_T" 'select(.t >= $t and .t <= $t + 120) |
  select(.e | test("provider|backoff|silenced"))' /tmp/phase_d_main.jsonl

# Gate 4: agent_id coverage on backend events
jq -r --argjson t "$RESTART_T" 'select(.t >= $t and .t <= $t + 120) |
  select(.e | test("peer_backend|dispatch")) | .agent_id // "NULL"' /tmp/phase_d_main.jsonl | sort | uniq -c
```

### Step 4 — Archive the trace

```bash
cp /tmp/phase_d_main.jsonl docs/experiments/results/phase_d_$(date +%Y%m%d).jsonl
ls -la docs/experiments/results/phase_d_*.jsonl
```

## Pass/fail gates

| Gate | Pass | Fail |
|---|---|---|
| **No stacked agent timeouts** | Zero `LLM call timed out after 300.0s` warnings in the entire trace | Any such warning = HTTP layer didn't fire first, Plan 3.5 regression |
| **Fast-fail per attempt** | Every `peer_backend_failed` has `latency_ms < 2000` | Any single attempt > 10s = Plan 3 regression |
| **No `dispatch_exhausted` cascade** | Each `dispatch_exhausted` has `attempts: 1` | `attempts > 1` means the router is still retrying internally |
| **No provider silencing** | Zero `provider_silenced` / backoff log events in window | Any silencing = `_note_provider_failure` pollution |
| **Sim continues after recovery** | ≥ 5 successful `peer_backend_call` events after the first post-recovery success | Sim wedges or every subsequent turn errors |
| **Recovery time** | First post-recovery success ≤ 10s after the peer's next attempt post-leader-ready | Peer attempts a call after leader-ready and it still fails-slow |

### Recovery-time measurement caveat

Under sim workload, the literal "leader-ready → first successful call" gap can look like 30+ seconds even on a passing run. The cause is sim cadence (the orchestrator spends 10-30s between LLM calls doing local agent work), **not** peer-side stuck state. Measure recovery time by asking: *when the peer's next call actually went out, did it succeed?* — not by timer from leader-ready. For a tight recovery-time number, use a benchmark workload that fires a request the instant the previous one completes or fails. That's a separate future experiment.

## Common pitfalls

1. **Stale sims holding state.** `pkill -f "maxim.*sim"` before starting; the sim runner holds GPU/port state that causes confusing failures in the next run.
2. **`scenarios/campaigns/heist_v1.yaml` is too short** (~3 turns, ~25s) — not enough window to inject a restart. Use a generative goal.
3. **Forgetting to wait for leader-ready before checking post-recovery metrics.** `maxim peer restart` blocks until "LLM ready" — use that command's return as a synchronization point, don't guess.
4. **Running on a VRAM-saturated leader.** Always run the pre-flight health gate first. KV cache spillover to shared GPU memory produces ~125s latency per call and poisons the entire run.
5. **Interpreting `peer_backend_call` / `peer_backend_failed` `agent_id=null` as a problem.** Known observability gap (Plan 4 scope) — the events are real, just missing attribution. `dispatch_exhausted` from the router has `agent_id=llm_worker` correctly.
6. **Running two stress test attempts back to back.** Each is a real leader restart with real inference workload — not free. One attempt, analyse, iterate, second attempt if needed.
7. **Trying to run this from the main conversation context of a Claude session** — use a second terminal or background the sim. `maxim peer restart` blocks, and if the sim is foregrounded you can't trigger it.

## Running it from a Claude Code session

Same steps, but use `run_in_background=true` for the sim and poll the log file between tool calls:

1. Start sim with `run_in_background=true`.
2. Poll log in a loop with a small `sleep` cadence; break when call count threshold is reached.
3. Trigger `maxim peer restart` synchronously (it blocks until leader-ready).
4. Poll for post-recovery calls.
5. Send `SIGINT` to the sim pid; fall through to `SIGTERM` / `SIGKILL` if needed.
6. Run the Step 3 jq queries; relay results to the user.

The 2026-04-13 run did this end-to-end from a single session — see session summary in memory (`project_llm_path_cancellation_hygiene_shipped.md`) for the exact timing used.

## What to deliver

After a successful run, write up a report at `docs/experiments/results/llm_path_stress_<date>.md` using the [2026-04-13 report](../results/llm_path_stress_20260413.md) as the template. Include:

1. Hardware + git commit
2. Pre-flight sanity curl result (time + tok/s)
3. Timeline table (dt-ordered backend events in the restart window)
4. Pass/fail for each gate
5. Trace file path for reproducibility

Commit the trace JSONL alongside the report so future sessions can re-analyze without rerunning.
