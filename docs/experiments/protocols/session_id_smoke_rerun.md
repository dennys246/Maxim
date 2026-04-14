# Session_id + agent_id observability smoke test — rerun guide

Four-smoke pass validating the end-to-end observability story across
Plan 4 Stage A (agent_id fix, PR #101), Plan 4 Stage B (bench harness,
PR #101), and PR #104 (session_id plumbing). Takes ~3 minutes
wall-clock against a healthy leader.

**Latest result:** [../results/session_id_smoke_20260414.md](../results/session_id_smoke_20260414.md) — 4/4 PASS against RTX 5080 + Qwen2.5-14B-Instruct Q4_K_M.

**Companion docs:**
- [bench_recovery_time_rerun.md](bench_recovery_time_rerun.md) — the more thorough recovery-time protocol
- [llm_path_stress_test_rerun.md](llm_path_stress_test_rerun.md) — the full Phase D restart protocol

## What this smoke tests

Four independent gates, each one probing a different layer of the
observability story:

1. **Bench per-run session_id distinction** — two back-to-back bench
   runs must produce different `session_id` values in their JSONL
   traces. Guards against the initial Plan 4 B mistake of reusing
   `BENCH_AGENT_ID` as session_id.

2. **Sim end-to-end coverage** — every `peer_backend_call` and
   `peer_backend_failed` event in a real sim's JSONL trace must carry
   non-null `agent_id`, `session_id`, and `request_id`. Guards against
   Plan 4 A regressions AND PR #104 session_id plumbing regressions in
   one shot.

3. **Report / JSONL / report.json triple-match** — the sim report
   directory name must equal the session_id in the JSONL trace and
   the `session_id` field inside `report.json`. Guards against the
   cross-correlation claim from PR #104.

4. **Doctor sanity** — `maxim doctor` against the configured peer
   reports all connectivity green. Doesn't fail-harden any specific
   session_id claim but catches "did my branch silently break peer
   auth" regressions.

Pre-fix symptoms to look for:

- **Pre-Plan-4-A:** `jq -c 'select(.e=="peer_backend_call" and .agent_id==null)' | wc -l` returns >0
- **Pre-PR-#104 (sim):** same jq against `.session_id==null` returns >0
- **Pre-PR-#104 (bench):** `jq -r '.session_id'` on two back-to-back runs returns the same value
- **Pre-PR-#104 (report match):** `~/.maxim/sim_reports/<jsonl_session_id>` doesn't exist

## Prerequisites

**Code state on peer:**

- `main` at or past PR #104 (session_id plumbing)
- `ruff check` + fast suite green:
  ```bash
  pkill -f "maxim.*sim" 2>/dev/null; sleep 1
  python -m pytest tests/unit/test_llm_worker_pool.py::TestSessionIdPlumbing \
    tests/unit/test_bench_recovery_time.py \
    tests/unit/test_maxim_peer_backend.py::TestRequestContext \
    -q
  ```
  Must report >= 37 passed (5 session_id plumbing + 27 bench + 7+ peer
  backend).

**Hardware:**

- **Leader:** GPU running a healthy llama-cpp-server behind a tunnel.
  Reference: RTX 5080 + Qwen2.5-14B-Instruct Q4_K_M + `MAXIM_LLM_N_CTX`
  below the Plan 3.6 R5 spillover threshold. `maxim doctor` must show
  VRAM pressure < 85% on the leader (via the Plan 3.6 R5 check).
- **Peer:** the machine running the smoke. No GPU requirement.

**Peer config:**

```bash
cat ~/.config/maxim/peer.yml
# Must show url + api_key pointing at the leader
```

## Pre-flight health gate

Same gate as the other rerun protocols. If the leader can't serve a
healthy inference in <3 s, smoke results are meaningless:

```bash
KEY=$(python3 -c "import yaml; print(yaml.safe_load(open('$HOME/.config/maxim/peer.yml'))['api_key'])")
URL=$(python3 -c "import yaml; print(yaml.safe_load(open('$HOME/.config/maxim/peer.yml'))['url'])")

time curl -s -o /tmp/sanity.json -w "http_code=%{http_code}\n" --max-time 10 \
  -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d '{"model":"any","messages":[{"role":"user","content":"say hi"}],"max_tokens":8,"temperature":0}' \
  "$URL/chat/completions"
```

**Pass:** real time < 3 s, http_code=200, `/tmp/sanity.json` contains
non-empty `choices[0].message.content`.

**Fail:** time > 5 s or http_code != 200 → STOP. Fix the leader first
(see [bench_recovery_time_rerun.md](bench_recovery_time_rerun.md) for
the spillover diagnostic).

## Runbook

### Step 1 — Clean smoke scratch dir

```bash
mkdir -p /tmp/smoke
rm -f /tmp/smoke/bench_run1.jsonl /tmp/smoke/bench_run2.jsonl \
      /tmp/smoke/bench_run1.log /tmp/smoke/bench_run2.log \
      /tmp/smoke/sim.jsonl /tmp/smoke/sim.log

KEY=$(python3 -c "import yaml; print(yaml.safe_load(open('$HOME/.config/maxim/peer.yml'))['api_key'])")
URL=$(python3 -c "import yaml; print(yaml.safe_load(open('$HOME/.config/maxim/peer.yml'))['url'])")
```

### Step 2 — Smoke 1: back-to-back bench

```bash
maxim bench recovery-time --url "$URL" --api-key "$KEY" \
    --duration 8 --max-tokens 4 --pace 0.2 \
    --output /tmp/smoke/bench_run1.jsonl 2>/tmp/smoke/bench_run1.log

maxim bench recovery-time --url "$URL" --api-key "$KEY" \
    --duration 8 --max-tokens 4 --pace 0.2 \
    --output /tmp/smoke/bench_run2.jsonl 2>/tmp/smoke/bench_run2.log
```

**Wall-clock:** ~17 s (two 8-second runs plus startup).

**Assertions:**

```bash
SID1=$(jq -r 'select(.e=="benchmark") | .session_id' /tmp/smoke/bench_run1.jsonl)
SID2=$(jq -r 'select(.e=="benchmark") | .session_id' /tmp/smoke/bench_run2.jsonl)
echo "Run 1 session_id: $SID1"
echo "Run 2 session_id: $SID2"

# Gate 1: distinct session_ids
test "$SID1" != "$SID2" && echo "✓ distinct session_ids" || echo "✗ FAIL: same session_id"

# Gate 2: both match bench_YYYYMMDD_HHMMSS shape
echo "$SID1" | grep -Eq '^bench_[0-9]{8}_[0-9]{6}$' && echo "✓ run 1 shape" || echo "✗ FAIL: run 1 shape"
echo "$SID2" | grep -Eq '^bench_[0-9]{8}_[0-9]{6}$' && echo "✓ run 2 shape" || echo "✗ FAIL: run 2 shape"

# Gate 3: zero null session_id on per-call events
NULL1=$(jq -c 'select(.e=="peer_backend_call" and .session_id==null)' /tmp/smoke/bench_run1.jsonl | wc -l)
NULL2=$(jq -c 'select(.e=="peer_backend_call" and .session_id==null)' /tmp/smoke/bench_run2.jsonl | wc -l)
test "$NULL1" -eq 0 && echo "✓ run 1 no null session_ids" || echo "✗ FAIL: run 1 has $NULL1 nulls"
test "$NULL2" -eq 0 && echo "✓ run 2 no null session_ids" || echo "✗ FAIL: run 2 has $NULL2 nulls"
```

### Step 3 — Smoke 2: real sim end-to-end

Use `--sim-max-turns 3` so the sim self-terminates via the max-turns
finish path rather than needing SIGINT. SIGINT-exit skips report
generation so Smoke 3 needs a clean exit.

```bash
MAXIM_BACKEND_TRACE=1 MAXIM_LOG_FILE=/tmp/smoke/sim.jsonl \
  maxim --sim "just say hello and call finish_simulation" \
  --seed 42 --sim-max-turns 3 2>/tmp/smoke/sim.log
```

**Wall-clock:** ~40-60 s depending on leader latency (3 turns × ~15 s
each on a healthy 14B setup).

**Assertions:**

```bash
# Gate 1: coverage — zero null on any of agent_id, session_id, request_id
TOTAL=$(jq -c 'select(.e=="peer_backend_call" or .e=="peer_backend_failed")' /tmp/smoke/sim.jsonl | wc -l)
NULL_AGENT=$(jq -c 'select((.e=="peer_backend_call" or .e=="peer_backend_failed") and .agent_id==null)' /tmp/smoke/sim.jsonl | wc -l)
NULL_SESSION=$(jq -c 'select((.e=="peer_backend_call" or .e=="peer_backend_failed") and .session_id==null)' /tmp/smoke/sim.jsonl | wc -l)
NULL_REQUEST=$(jq -c 'select((.e=="peer_backend_call" or .e=="peer_backend_failed") and .request_id==null)' /tmp/smoke/sim.jsonl | wc -l)

echo "Total peer_backend events: $TOTAL"
echo "  null agent_id:   $NULL_AGENT"
echo "  null session_id: $NULL_SESSION"
echo "  null request_id: $NULL_REQUEST"

test "$TOTAL" -gt 0 && echo "✓ events captured" || echo "✗ FAIL: no events — did the sim run?"
test "$NULL_AGENT" -eq 0 && echo "✓ zero null agent_id" || echo "✗ FAIL"
test "$NULL_SESSION" -eq 0 && echo "✓ zero null session_id" || echo "✗ FAIL"
test "$NULL_REQUEST" -eq 0 && echo "✓ zero null request_id" || echo "✗ FAIL"

# Gate 2: single session_id across the sim
SID_COUNT=$(jq -r 'select(.e=="peer_backend_call") | .session_id' /tmp/smoke/sim.jsonl | sort -u | wc -l)
test "$SID_COUNT" -eq 1 && echo "✓ single session_id across sim" || echo "⚠ unexpected session_id count: $SID_COUNT"
```

### Step 4 — Smoke 3: report dir / JSONL / report.json triple-match

```bash
SID=$(jq -r 'select(.e=="peer_backend_call") | .session_id' /tmp/smoke/sim.jsonl | sort -u | head -1)
echo "JSONL session_id: $SID"

# Gate 1: report directory exists at expected path
REPORT_DIR=~/.maxim/sim_reports/"$SID"
test -d "$REPORT_DIR" && echo "✓ report dir exists" || echo "✗ FAIL: no dir at $REPORT_DIR"

# Gate 2: report.json session_id field matches
REPORT_SID=$(jq -r '.session_id' "$REPORT_DIR/report.json" 2>/dev/null)
test "$REPORT_SID" = "$SID" && echo "✓ report.json.session_id matches ($REPORT_SID)" || echo "✗ FAIL: $REPORT_SID != $SID"

# Gate 3: expected artifacts exist
for f in report.json actions.jsonl aut_hippocampus.json aut_nac.json; do
    test -f "$REPORT_DIR/$f" && echo "✓ $f" || echo "⚠ missing: $f"
done
```

### Step 5 — Smoke 4: doctor sanity check

```bash
maxim doctor 2>&1 | tail -40
```

Scan the output for:

- **Environment section:** all ✓ except the CPU-only ⚠ on a GPU-less
  peer machine (expected)
- **Peer Connectivity section:** all ✓, no ✗
- **Peer latency probes:** 5/5 successful
- The `role_divergence` WARNING at the top is pre-existing and NOT a
  Plan 4 regression — if you see it, ignore.

## Pass/fail gates — full matrix

| Smoke | Gate | Target |
|---|---|---|
| 1 | Distinct per-run session_ids | `SID1 != SID2` |
| 1 | Session_id shape | `bench_YYYYMMDD_HHMMSS` regex |
| 1 | Zero null session_id on per-call events | Both runs |
| 2 | Events captured | > 0 `peer_backend_call` + `peer_backend_failed` |
| 2 | Zero null agent_id | 100% coverage |
| 2 | Zero null session_id | 100% coverage |
| 2 | Zero null request_id | 100% coverage |
| 2 | Single session_id across sim | `jq -u .session_id | wc -l == 1` |
| 3 | Report dir exists at expected path | `~/.maxim/sim_reports/{sid}/` |
| 3 | `report.json.session_id` matches JSONL | String equality |
| 3 | Core artifacts present | `report.json`, `actions.jsonl`, `aut_hippocampus.json`, `aut_nac.json` |
| 4 | Doctor peer connectivity | All ✓ |

**Typical pass latencies** (reference: 2026-04-14 run against RTX 5080
+ Qwen2.5-14B-Instruct):

| Stage | Wall-clock |
|---|---|
| Pre-flight curl | < 1 s |
| Smoke 1 (bench x2) | ~17 s |
| Smoke 2 (sim 3 turns) | ~40 s |
| Smoke 3 (assertions) | < 1 s |
| Smoke 4 (doctor) | ~5 s |
| **Total** | **~65 s** (plus bench startup jitter) |

## Pitfalls

**1. SIGINT-exit sim produces no report directory.** The sim must
self-terminate via `--sim-max-turns` or an LLM-initiated
`finish_simulation` call — if you Ctrl+C it, `build_report` never
runs, no session directory is created, and Smoke 3 fails with "no
dir at ~/.maxim/sim_reports/...". The 2026-04-14 run caught this the
hard way: first attempt used SIGINT, had to re-run with
`--sim-max-turns 3`. Protocol above uses the self-terminating form.

**2. Aggregate `benchmark` summary event has no `agent_id`.** The
bench JSONL contains one final event with `e=benchmark` summarizing
the whole run. This event does NOT have an agent_id field — it's a
run-level aggregate, not a per-call event. If you `jq 'select(.agent_id==null)' | wc -l`
on a bench trace, you'll see 1 (the summary). That's expected.
Filter with `jq 'select((.e=="peer_backend_call" or .e=="peer_backend_failed") and .agent_id==null)'`
if you want per-call-only coverage.

**3. Single `agent_id=llm_worker` in sim trace.** The queue-path
`LLMWorker._process_request` hard-codes `"agent": "llm_worker"` at
[llm_worker.py:964](../../../src/maxim/agents/llm_worker.py#L964).
A sim with multiple named agents (via `generate_json_direct` — used
by ExecAgent) will show distinct agent_ids; a trivial sim that only
exercises the queue path will show one. This is NOT a regression —
it's the expected shape for non-multi-agent sims.

**4. Pre-existing `role_divergence` WARNING** at doctor startup. This
is logged by both `leader_mode.detect_role()` and `role.detect_role()`
producing consistent but separately-logged results. Filed against
the Plan 4 C1 followup list. Ignore for smoke purposes.

**5. VRAM pressure from an unrelated process.** If `maxim doctor`
reports VRAM > 85%, the sim calls may be slow enough to time out.
Check `nvidia-smi` on the leader and close any GPU consumers
(browser hw accel, other models).

## When to run this smoke

- **Before any PR that touches** `llm_worker.py`, `router.py`,
  `maxim_peer_backend.py`, `utils/http.py::_build_headers`, the bench
  harness, or the simulation orchestrator's LLMWorker construction.
- **After shipping a new release tag** to confirm the observability
  story didn't silently regress.
- **As a quick "is my branch still sane" check** before starting a
  longer stress test or benchmark — you get the full observability
  contract validation in ~65 seconds.
- **Not** as a replacement for [bench_recovery_time_rerun.md](bench_recovery_time_rerun.md)
  — that protocol measures recovery-time under leader restart, which
  this smoke doesn't probe at all.

## Cost

~3 minutes wall-clock total. ~50-80 LLM calls to the leader (18 × 2
bench + 15-30 sim + 5 doctor probes). Negligible tokens because the
bench uses `max_tokens=4` and the sim ends at turn 3.
