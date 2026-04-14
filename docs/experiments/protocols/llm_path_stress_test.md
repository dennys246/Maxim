# LLM Path Stress Test Protocol

**Status:** Active protocol — runs after [llm_path_fast_failover.md](../../plans/archive/llm_path_fast_failover.md) ships, before [llm_path_operator_visibility.md](../../plans/llm_path_operator_visibility.md) scope is finalized
**Purpose:** decide what (if any) multi-peer dispatch is needed, verify the 52-second retry loop is dead, validate substrate P2 under realistic load, measure `llama.cpp --parallel` batching as a distribution alternative
**Expected runtime:** 4-6 hours focused execution + 2-3 hours for analysis + write-up

## Purpose

This is the gating experiment that determines Plan 4 (Operator Visibility) scope. Its three goals are independent but share infrastructure:

1. **Validate Plan 3 shipped correctly.** The 52-second `_OpenAIBackend` retry loop should be gone. Measurable via `backend_call_duration_seconds` p99 and subjective `peer restart` timing.
2. **Validate substrate P2 reward modulation** under multi-agent load. This was going to run as its own experiment; running it inside the stress test means one setup serves both needs.
3. **Measure `llama.cpp --parallel` batching PoC.** If increasing leader-side concurrency solves saturation, we don't need multi-peer dispatch (defer [llm_path_multi_peer_dispatch.md](../../plans/deferred/llm_path_multi_peer_dispatch.md) permanently). If it doesn't, we know the ceiling and can scope distribution work to what's actually needed.

## Prerequisites

**Code:**
- [llm_path_foundation.md](../../plans/archive/llm_path_foundation.md) shipped (R0 + R1)
- [llm_path_typed_errors.md](../../plans/archive/llm_path_typed_errors.md) shipped (R2)
- [llm_path_fast_failover.md](../../plans/archive/llm_path_fast_failover.md) shipped (R2.5 + R2.6)
- [llm_path_operator_visibility.md](../../plans/llm_path_operator_visibility.md) **NOT** required — this protocol runs without Plan 4's admin API; observability comes from JSONL logs + `lane_metrics.metrics_snapshot()`
- Substrate P2 code merged (already done)
- P2 validation fixtures available in `tests/substrate/` and `scenarios/substrate/`

**Hardware:**
- **Leader node:** RTX 5080 (16 GB VRAM), Qwen 14B Q4_K_M loaded, llama-cpp-server running
- **Peer node:** Mac (24 GB unified memory), fast failover code installed, cluster key configured
- Both nodes on the same LAN or Cloudflare tunnel; peer → leader RTT < 50ms ideal

**Environment:**
- Clean probe cache: `rm -f ~/.maxim/util/probe_cache.json` on both nodes
- `MAXIM_LOG_FILE=/tmp/maxim-stress-<phase>.jsonl` set for each phase (separate file per phase for clean analysis)
- `MAXIM_BACKEND_TRACE=1` for Phase D (leader restart) to capture full retry-loop-killed traces
- `MAXIM_HTTP_TRACE=0` by default (too verbose); flip to 1 for debugging if a phase goes wrong

**Tools:**
- `jq` for JSONL log parsing
- `curl` for manual endpoint probing
- `htop` or equivalent for leader GPU utilization watching

## Phase A — Baseline + substrate P2 validation (single-user, one agent)

**Goal:** establish the baseline. Does Plan 3 work for the single-user one-agent case? Does substrate P2 pass its mechanistic targets?

**Setup:**
```bash
# On leader (ssh to RTX machine)
systemctl restart maxim-leader  # fresh start
sleep 10  # give it time to load model + probe

# On peer (Mac)
rm -f ~/.maxim/util/probe_cache.json
export MAXIM_LOG_FILE=/tmp/maxim-stress-phase-a.jsonl
export MAXIM_BACKEND_TRACE=1
```

**Run:**
```bash
# Substrate P2 validation fixture (from peer)
python -m pytest tests/substrate/test_p2_reward_modulation.py -v --seed 42
# Expected: all P2 mechanistic targets pass (per substrate_recognition.md)

# Single-user sim (from peer) — no --language-model flag; peer config routes
# to the leader's loaded model automatically. Passing --language-model triggers
# _apply_local_llm_override which clears remote_url and routes locally instead.
maxim --sim "A cyberpunk heist" --seed 42

# Record baseline numbers
jq -c 'select(.event=="peer_backend_call")' /tmp/maxim-stress-phase-a.jsonl | \
  jq -s '[.[] | .latency_ms] | {p50: (sort | .[length/2|floor]), p99: (sort | .[(length*0.99)|floor])}'
```

**Collect:**
- Substrate P2 test results (pass/fail, metrics)
- Per-lane `backend_call_duration_seconds` p50 + p99
- Total sim duration
- Token throughput (tokens/sec)
- Any `backend_unclassified_errors_total` entries (should be 0)

**Expected baseline numbers** (rough, RTX 5080 + Qwen 14B):
- Large lane backend call p50: 2-5s
- Large lane backend call p99: 8-15s
- Token throughput: 20-40 tok/s
- Zero unclassified errors

**Failure mode to watch for:**
- `backend_unclassified_errors_total > 0` → Plan 3 missed an exception type. **STOP**, file bug, fix before proceeding.
- Substrate P2 metrics fail → investigate P2 implementation; may not be related to LLM path.

## Phase B — Multi-agent fan-out (stress the lock)

**Goal:** does `_inference_lock` become a visible bottleneck under concurrent agents? This is the direct test of whether [llm_path_async_router.md](../../plans/deferred/llm_path_async_router.md) needs reviving.

**Setup:**
```bash
# On peer
rm -f ~/.maxim/util/probe_cache.json
export MAXIM_LOG_FILE=/tmp/maxim-stress-phase-b.jsonl
export MAXIM_BACKEND_TRACE=1

# Use a campaign that spawns multiple concurrent NPC agents
# If tests/stress/multi_agent_fan_out.py doesn't exist yet, create it as part of Plan 3 prep
```

**Run:**
```bash
# Multi-agent campaign with 3-5 concurrent NPCs
maxim --sim scenarios/campaigns/heist_v1.yaml --seed 42 --agent-pool-size 3

# Parse logs for wait time analysis
# "wait time" = time from agent submitting request to backend call actually starting
jq -c 'select(.event=="peer_backend_call" or .event=="llm_submit_queued")' \
  /tmp/maxim-stress-phase-b.jsonl > /tmp/phase-b-events.jsonl
```

**Collect:**
- Per-agent `backend_call_duration_seconds` p50/p99 (group by `agent_id` in the JSONL)
- Estimated wait time per agent (time from submit to call start)
- Total campaign duration vs Phase A single-agent duration (should scale roughly linearly if lock is serializing; less than linear if parallelism is happening)
- Substrate P2 reward modulation events per agent — verify that parallel agents correctly update NAc without race conditions

**Decision signal:**
- If wait time p99 > backend call p99 by 2x or more → `_inference_lock` IS the bottleneck. [llm_path_async_router.md](../../plans/deferred/llm_path_async_router.md) has a clear revive trigger.
- If wait time p99 ≈ backend call p99 → lock is not the bottleneck; agents are mostly sequential by nature. Async routing wouldn't help.
- If substrate P2 shows race-condition-style metric corruption (NAc bias values outside expected bounds) → **STOP**, substrate P2 has a concurrency bug, fix before continuing.

**Analysis query:**
```bash
# Per-agent p50/p99
jq -c 'select(.event=="peer_backend_call") | {agent_id, latency_ms}' /tmp/maxim-stress-phase-b.jsonl | \
  jq -s 'group_by(.agent_id) | map({
    agent_id: .[0].agent_id,
    count: length,
    p50: (map(.latency_ms) | sort | .[length/2|floor]),
    p99: (map(.latency_ms) | sort | .[(length*0.99)|floor])
  })'
```

## Phase C — `llama.cpp --parallel` batching sweep

**Goal:** does increasing leader-side concurrency eliminate the need for multi-peer dispatch? This is the decision input for Plan 4's scope.

**Setup:**
```bash
# On leader — restart with --parallel N for each sweep value
# This requires editing the llama-cpp-server launch config (check how your leader starts it)
# Commonly: /etc/systemd/system/maxim-leader.service or similar
```

**Sweep protocol:** for each `N ∈ {1, 2, 4, 8}`:

1. **Restart leader with new `--parallel N`:**
   ```bash
   # On leader
   sudo systemctl stop maxim-leader
   # Edit launch to add --parallel N
   sudo systemctl start maxim-leader
   sleep 15  # model reload + probe
   ```

2. **Reset peer probe cache:**
   ```bash
   # On peer
   rm -f ~/.maxim/util/probe_cache.json
   export MAXIM_LOG_FILE=/tmp/maxim-stress-phase-c-parallel-${N}.jsonl
   ```

3. **Run the Phase B multi-agent scenario:**
   ```bash
   maxim --sim scenarios/campaigns/heist_v1.yaml --seed 42 --agent-pool-size 5
   ```

4. **Collect metrics:**
   ```bash
   # Total campaign duration
   jq -c 'select(.event=="sim_complete")' /tmp/maxim-stress-phase-c-parallel-${N}.jsonl
   
   # Leader throughput (tokens/sec)
   jq -c 'select(.event=="peer_backend_call") | .output_tokens' \
     /tmp/maxim-stress-phase-c-parallel-${N}.jsonl | jq -s 'add / (length * 1)'
   
   # Per-agent p99 latency (to check for slowdown)
   jq -c 'select(.event=="peer_backend_call") | .latency_ms' \
     /tmp/maxim-stress-phase-c-parallel-${N}.jsonl | jq -s 'sort | .[(length*0.99)|floor]'
   ```

**Build a table:**

| `--parallel` | Campaign duration | Leader throughput | p99 latency | VRAM peak |
|---|---|---|---|---|
| 1 (baseline) | ? | ? | ? | ? |
| 2 | ? | ? | ? | ? |
| 4 | ? | ? | ? | ? |
| 8 | ? | ? | ? | ? |

**Decision criteria:**

- **If `--parallel 4` or higher doubles throughput AND p99 latency stays within 2x of baseline:** batching solves saturation cheaply. Commit the `--parallel N` config. **Defer multi-peer dispatch permanently** — update the deferred plan's revive trigger to "only revive if we need multiple physical machines, not just leader concurrency."
- **If batching helps somewhat but throughput saturates at some N < 8 with large p99 degradation:** batching hits a wall. You'll want both batching AND multi-peer eventually. Ship Plan 4 as scoped (R3.0 + R3.5-lite + R3.6-lite), revive [llm_path_multi_peer_dispatch.md](../../plans/deferred/llm_path_multi_peer_dispatch.md) for the cases batching doesn't cover.
- **If batching has no effect (throughput flat across N):** leader bottleneck is not concurrency. Investigate VRAM pressure, context length, KV cache settings, or model quantization. **STOP** the stress protocol and run a root-cause analysis before deciding on Plan 4 scope.

**Edge case — `--parallel N` fails to start:** some llama.cpp builds don't support `--parallel`. If leader won't start, check the build with `llama-server --version` and consult `llama.cpp` docs. If unsupported, document it and skip to Phase D.

## Phase D — Leader restart mid-workload (the definitive Plan 3 test)

**Goal:** prove the 52-second retry loop is dead.

**Setup:**
```bash
# On peer
rm -f ~/.maxim/util/probe_cache.json
export MAXIM_LOG_FILE=/tmp/maxim-stress-phase-d.jsonl
export MAXIM_BACKEND_TRACE=1
```

**Run:**
```bash
# Start a multi-agent sim on the peer
maxim --sim scenarios/campaigns/heist_v1.yaml --seed 42 --agent-pool-size 3 &
SIM_PID=$!

# Wait for a few turns to land
sleep 60

# On leader, restart
ssh leader 'sudo systemctl restart maxim-leader'

# Watch peer's log for recovery
tail -f /tmp/maxim-stress-phase-d.jsonl | jq -c 'select(.event=="peer_backend_call" or .event=="peer_backend_failed")'
```

**Collect:**
- **Time from leader restart to first successful peer backend call after the restart.** This is the key metric.
- Number of failed calls during the restart window (expected: few, < 5)
- Per-failure exception class (expected: mostly `BackendDown`, some `BackendTimeout`)
- Whether the sim crashes, hangs, or recovers gracefully

**Measurement script:**
```bash
# Find the first post-restart failure
RESTART_TIME=$(jq -c 'select(.event=="peer_backend_failed") | .ts' /tmp/maxim-stress-phase-d.jsonl | head -1)

# Find the next success after that
RECOVERY_TIME=$(jq -c 'select(.event=="peer_backend_call" and .status==200) | .ts' /tmp/maxim-stress-phase-d.jsonl | \
  awk -v start="$RESTART_TIME" '$0 > start {print; exit}')

# Compute difference
python -c "
from datetime import datetime
r = datetime.fromisoformat('${RESTART_TIME}'.replace('Z', '+00:00'))
c = datetime.fromisoformat('${RECOVERY_TIME}'.replace('Z', '+00:00'))
print(f'Recovery time: {(c - r).total_seconds():.1f}s')
"
```

**Success criteria:**
- Recovery time < 10 seconds (target: 2-5s)
- Zero `backend_unclassified_errors_total` increments
- Sim doesn't crash or hang
- Log output includes `BackendDown` + `BackendTimeout` typed events during the restart window

**Pre-Plan-3 baseline for comparison:** if you have an older log file from before Plan 3 shipped, compute the same metric. Should be ~52+ seconds. **The delta between pre-Plan-3 and post-Plan-3 is the headline number for the stress test report.**

## Phase E — Fault injection (chaos scenarios)

**Goal:** verify typed exception handling covers every failure mode we care about.

**Setup:** scripted fault injection. For each scenario, run the peer's multi-agent sim and verify the expected typed exception fires.

**Scenarios:**

### E.1 — Kill leader process entirely

```bash
# On leader
ssh leader 'sudo pkill -9 llama-server'  # not just restart — hard kill

# On peer
export MAXIM_LOG_FILE=/tmp/maxim-stress-phase-e-kill.jsonl
maxim --sim "test" --seed 42 --timeout 30
```

**Expected:** `peer_backend_failed error=BackendDown`, then either sim completes via cloud fallback (if configured) or fails cleanly after exhaustion.

**Verify:**
```bash
jq -c 'select(.event=="peer_backend_failed") | .error' /tmp/maxim-stress-phase-e-kill.jsonl | sort -u
# Expected: "BackendDown" (primarily)
```

### E.2 — Network partition (iptables drop)

```bash
# On peer (temporarily block leader IP)
sudo iptables -A OUTPUT -d <leader_ip> -j DROP

# Start sim
maxim --sim "test" --seed 42 --timeout 30 &
SIM_PID=$!
sleep 30

# Restore network
sudo iptables -D OUTPUT -d <leader_ip> -j DROP

wait $SIM_PID
```

**Expected:** `BackendTimeout` or `BackendDown` exceptions during partition, graceful recovery after restore.

### E.3 — Leader returning 429 for 30 seconds (mock)

This requires either modifying the leader's llama-cpp-server config or inserting a local proxy returning 429. Skip if infrastructure isn't available; Plan 3's unit tests cover 429 handling already.

### E.4 — Auth rejection (cluster key rotation)

```bash
# On leader, rotate the cluster key WITHOUT telling the peer
ssh leader 'echo "new-key-$(date +%s)" > /etc/maxim/cluster_key && systemctl reload maxim-leader'

# On peer, start a sim with the OLD key
maxim --sim "test" --seed 42 --timeout 30
```

**Expected:** `peer_backend_failed error=BackendAuthFailed fix_hint="Cluster key rejected..."`. 300-second backoff kicks in (subsequent calls skip this provider).

**Verify:**
```bash
jq -c 'select(.event=="peer_backend_failed" and .error=="BackendAuthFailed")' \
  /tmp/maxim-stress-phase-e-auth.jsonl | head -3
```

### E.5 — Requesting a model the peer doesn't have

```bash
# Start a sim with a model name the leader doesn't have loaded
maxim --sim "test" --language-model nonexistent-model-xyz --timeout 30
```

**Expected:** `BackendModelMissing fix_hint="Run maxim peer --node <name> install nonexistent-model-xyz"`.

### Phase E pass criteria

Each scenario must emit the correct typed exception. `backend_unclassified_errors_total` must stay at 0 across the entire phase. If any scenario falls through to the generic safety net, **STOP** and file a bug for the missing typed class.

## Report — `docs/experiments/results/llm_path_stress_<date>.md`

Template:

```markdown
# LLM Path Stress Test Results — <YYYY-MM-DD>

**Protocol:** [llm_path_stress_test.md](../protocols/llm_path_stress_test.md)
**Environment:** RTX 5080 leader (Qwen 14B Q4_K_M) + Mac 24GB peer
**Total runtime:** <hours>

## Phase A — Baseline

- Substrate P2 mechanistic targets: PASS / FAIL (link to test output)
- Large lane p50 latency: X ms
- Large lane p99 latency: X ms
- Token throughput: X tok/s
- Unclassified errors: 0 / N

## Phase B — Multi-agent fan-out (3-5 agents)

- Wait time p50 / p99: X / Y ms
- Backend call p50 / p99: X / Y ms
- Lock bottleneck verdict: SIGNIFICANT / MARGINAL / NONE
- Substrate P2 under parallel agents: PASS / FAIL (any race conditions?)

## Phase C — Batching sweep

| `--parallel` | Duration | Throughput | p99 latency | VRAM peak |
|---|---|---|---|---|
| 1 | | | | |
| 2 | | | | |
| 4 | | | | |
| 8 | | | | |

**Verdict:** BATCHING SOLVES SATURATION / HELPS BUT NOT ENOUGH / NO EFFECT

## Phase D — Leader restart

- Recovery time: X seconds
- Failed calls during restart window: N
- Exception classes observed: [BackendDown, BackendTimeout, ...]
- Pre-Plan-3 baseline (if available): Y seconds
- **Delta: Y - X = improvement in seconds**

## Phase E — Fault injection

| Scenario | Expected exception | Observed | Verdict |
|---|---|---|---|
| E.1 Kill | BackendDown | | |
| E.2 Partition | BackendTimeout/BackendDown | | |
| E.3 429 mock | BackendOverloaded | SKIP / PASS | |
| E.4 Auth | BackendAuthFailed | | |
| E.5 Model missing | BackendModelMissing | | |

Unclassified errors across all of Phase E: 0 / N

## Decision — Plan 4 scope

- [ ] Ship Plan 4 as scoped (R3.0 + R3.5-lite + R3.6-lite only). Multi-peer dispatch stays deferred.
- [ ] Ship Plan 4 AND revive [multi-peer dispatch](../../plans/deferred/llm_path_multi_peer_dispatch.md). Reason: <stress test showed X>.
- [ ] Defer all of Plan 4. Reason: <unlikely but document if so>.

**Rationale:** <2-3 paragraphs explaining the decision from the data above>

## Follow-up actions

- <Any bugs found that need fixing before Plan 4>
- <Any surprising findings that deserve their own plan>
- <Any refinements to the deferred plans based on what we learned>
```

## Abort conditions

**STOP the stress test immediately if:**

1. **Plan 3's safety net fires** — `backend_unclassified_errors_total > 0` means a typed exception is missing. File a bug, fix, rerun Phase A before continuing.
2. **Substrate P2 metrics corrupt under multi-agent** — indicates a concurrency bug in substrate code. File a substrate bug, DO NOT proceed to later phases (the data would be unreliable).
3. **Leader crash during Phase C batching sweep** — indicates `--parallel N` has a bug in your llama.cpp build. Document the value of N that crashed, skip to Phase D with `--parallel 1`.
4. **Peer hangs indefinitely** — indicates the retry loop is still present (Plan 3 didn't ship correctly). Verify `grep -E "retry|backoff" src/maxim/models/language/maxim_peer_backend.py` returns zero.

## Notes for future stress tests

This protocol is a template. Future stress tests (for the deferred plans' revive conditions) should reuse the phase structure:

- **Phase A** — baseline + mechanistic validation
- **Phase B** — multi-agent parallelism stress
- **Phase C** — alternative-approach PoC (batching, multi-peer, async router, etc.)
- **Phase D** — chaos scenarios
- **Phase E** — fault injection

Adjust scenarios per plan but keep the phase numbering stable for cross-stress-test comparison.
