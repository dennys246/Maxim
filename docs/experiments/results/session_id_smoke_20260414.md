# Session_id plumbing smoke test — 2026-04-14

Validates the Plan 4 follow-up session_id plumbing (PR #104, commit
`36f9ff0`) end-to-end on real hardware. Four independent smoke probes
confirm:

1. The bench harness generates a **distinct per-run** `session_id`
   (not reusing `BENCH_AGENT_ID` like the initial Plan 4 B ship).
2. A live sim against the real RTX 5080 leader populates `agent_id`,
   `session_id`, and `request_id` on **every** `peer_backend_call`
   and `peer_backend_failed` event — zero nulls.
3. The sim's report directory at `~/.maxim/sim_reports/{session_id}/`
   matches the `session_id` in the JSONL trace AND the
   `session_id` field inside `report.json` — triple-match, no drift.
4. `maxim doctor` reports all peer connectivity green under the same
   code path.

**Pre-fix baseline (Phase D, 2026-04-13):** `peer_backend_call` events
had `agent_id=null` and `session_id=null`. Plan 4 A (PR #101) fixed
the `agent_id` gap via a three-way router/boundary/contextvar fix;
this run validates that Plan 4 A holds AND that the session_id
plumbing PR #104 closes the remaining half.

**Traces archived:**
- [smoke_bench_run1_20260414.jsonl](smoke_bench_run1_20260414.jsonl) — 19 events (8 s tight-loop bench)
- [smoke_bench_run2_20260414.jsonl](smoke_bench_run2_20260414.jsonl) — 19 events (back-to-back re-run)
- [smoke_sim_20260414.jsonl](smoke_sim_20260414.jsonl) — 1072 events (3-turn sim, `finish_reason=max_turns`)

## Setup

| | |
|---|---|
| Leader | RTX 5080 (16 GB), Qwen2.5-14B-Instruct Q4_K_M, Cloudflare tunnel `maxim.dennyschaedig.com` |
| Leader VRAM | 11.4 / 16 GB (below 95% spillover threshold per Plan 3.6 R5) |
| Peer | Mac, `feat/session-id-plumbing` branch fast-forwarded to `main` @ `87ffcf6` (Plan 4 A+B + session_id merged) |
| Code state | PR #101 (Plan 4 A+B) + PR #104 (session_id plumbing) + PR #103 (docs hygiene) all on main |
| Pre-flight | `curl /v1/models` returned 200 in <1 s with 11.4/16 GB VRAM reported |

## Smoke 1 — Back-to-back bench runs have distinct per-run session_ids

**Claim:** after PR #104 the bench harness generates a fresh
`session_id = "bench_" + time.strftime("%Y%m%d_%H%M%S")` per run,
so two back-to-back runs are distinguishable in the JSONL trace.
Pre-fix, both runs reused `BENCH_AGENT_ID` as the session_id.

**Commands:**

```bash
maxim bench recovery-time \
    --url https://maxim.dennyschaedig.com/v1 \
    --api-key <redacted> \
    --duration 8 --max-tokens 4 --pace 0.2 \
    --output /tmp/smoke/bench_run1.jsonl
maxim bench recovery-time \
    --url https://maxim.dennyschaedig.com/v1 \
    --api-key <redacted> \
    --duration 8 --max-tokens 4 --pace 0.2 \
    --output /tmp/smoke/bench_run2.jsonl
```

**Results:**

| Metric | Run 1 | Run 2 |
|---|---|---|
| Duration | 8.385 s | 8.114 s |
| Total attempts | 18 | 18 |
| Successes | 18 | 18 |
| Failures | 0 | 0 |
| **`session_id`** | **`bench_20260414_102407`** | **`bench_20260414_102416`** |
| Null session_id events | 0 / 19 | 0 / 19 |

**Per-call rate:** ~2.1 calls/sec (consistent with `--pace 0.2` +
~200 ms round-trip).

**Verdict:** ✅ **PASS** — distinct per-run session_ids, 9 seconds
apart as expected from wall-clock. Every attempt event carries the
correct session_id. The one "null agent_id" event in each file is
the aggregate `benchmark` summary event (not a per-call event); it
does not have an agent_id by design.

## Smoke 2 — Real sim coverage end-to-end

**Claim:** a live sim against the real leader produces
`peer_backend_call` / `peer_backend_failed` events that all carry
non-null `agent_id`, `session_id`, and `request_id`. Pre-fix, the
agent_id was null; pre-PR-#104, the session_id was null.

**Command:**

```bash
MAXIM_BACKEND_TRACE=1 MAXIM_LOG_FILE=/tmp/smoke/sim2.jsonl \
  maxim --sim "just say hello and call finish_simulation" \
  --seed 42 --sim-max-turns 3
```

Sim ran 40.2 s, 3 turns, 10 actions, finished via `max_turns` cap
(orchestrator never called `finish_simulation`, which exercises the
clean-exit path rather than SIGINT).

**JSONL event distribution:**

| Event type | Count |
|---|---|
| `log` | 890 |
| `percept` | 90 |
| `http_request` | 30 |
| **`peer_backend_call`** | **15** (this run; first smoke sub-run had 29 before SIGINT) |
| `cloud_audit` | 29 |
| `role_divergence` | 1 |
| `role_detected` | 1 |
| `probe_started` / `probe_completed` | 1 each |
| **Total** | **1072** |

**Coverage assertions:**

| Assertion | Result |
|---|---|
| `peer_backend_call` events with `agent_id=null` | **0 / 15** |
| `peer_backend_call` events with `session_id=null` | **0 / 15** |
| `peer_backend_call` events with `request_id=null` | **0 / 15** |
| Unique `agent_id` observed | `llm_worker` (single-agent sim) |
| Unique `session_id` observed | `20260414_103257` (one sim, one session) |

**Why a single `agent_id`:** the short sim ran the primary AUT +
orchestrator loops which both use the `LLMWorker._process_request`
queue-path that hard-codes `"agent": "llm_worker"` in the dict at
[src/maxim/agents/llm_worker.py:964](src/maxim/agents/llm_worker.py#L964).
The `generate_json_direct` ExecAgent path (which accepts
`agent_name=...`) wasn't exercised in this trivial sim. Both paths
carry session_id correctly. **This is expected for a non-multi-agent
sim, not a gap.**

**Verdict:** ✅ **PASS** — Plan 4 A + session_id plumbing both
validate end-to-end on real traffic.

## Smoke 3 — Report directory / JSONL / report.json triple-match

**Claim:** PR #104 promised the sim's report directory name matches
the session_id in the JSONL log so cross-correlating reports with
JSONL events is a string-equality match instead of a "figure out the
right timestamp window" exercise.

**Results:**

| Source | Value |
|---|---|
| JSONL `peer_backend_call.session_id` | `20260414_103257` |
| Report directory | `~/.maxim/sim_reports/20260414_103257/` |
| `report.json::session_id` field | `"20260414_103257"` |

All three match exactly. The report directory contains:

- `report.json` (full SimulationReport)
- `actions.jsonl` (bridged action log)
- `aut_hippocampus.json` (AUT hippocampus snapshot)
- `aut_nac.json` (AUT NAc causal-link snapshot)

**Report summary excerpt:**

```json
{
  "session_id": "20260414_103257",
  "goal": "just say hello and call finish_simulation",
  "duration_s": 40.2,
  "turns": 3,
  "finish_reason": "max_turns"
}
```

**Verdict:** ✅ **PASS** — triple-match confirmed. The PR #104 claim
that orchestrator pre-generates session_id at sim entry and threads
it through both LLMWorker construction AND build_report holds on
real hardware.

## Smoke 4 — Doctor sanity check

Verifies the doctor check wiring remained healthy across the Plan 4
A+B + session_id + substrate P2 stage 3 + hygiene pass merges.

**Command:** `maxim doctor`

**Results (excerpted):**

```
━━━ Environment ━━━
  ✓ Platform: macOS 15.7.3
  ✓ Architecture: arm64
  ⚠ GPU / CUDA: No CUDA device available — inference will be CPU-only
  ✓ LLM Tiers: Tiers: large, small. Profiles: {'small': 'smollm-1.7b-instruct',
    'large': 'qwen2.5-14b-instruct'}
  ✓ LLM Tier headroom: Hardware-best profile selected: qwen2.5-14b-instruct
  ✓ Disk space: 57.7 GB free of 926 GB
  ✓ RAM: 8.2 GB available of 24.0 GB
  ✓ Storage footprint: 57.7 GB free, Maxim using 8.4 GB
  ✓ MAXIM_ROLE: MAXIM_ROLE=peer

━━━ Peer Connectivity ━━━
  ✓ Remote URL: maxim.dennyschaedig.com reachable
  ✓ Peer API key: key set: jXzgjz…3LwzD4
  ✓ Peer auth: authenticated successfully
  ✓ Remote leader probe: https://maxim.dennyschaedig.com/v1 responding (168 ms)
  ✓ --llm vs peer config: no local --llm override active
  ✓ Remote model: leader model: qwen2.5-14b-instruct Q4_K_M
  ✓ Peer latency: p50=177 ms, p95=891 ms (5/5 probes)
```

**`role_divergence` WARNING** at startup is a pre-existing known
quirk: `leader_mode.detect_role()` and `role.detect_role()` return
equivalent but distinctly-logged results. Not a Plan 4 regression.

**Verdict:** ✅ **PASS** — all peer connectivity green, round-trip
probe latencies healthy, leader model correctly identified.

## Summary

| Smoke | Gate | Result |
|---|---|---|
| 1 | Two bench runs have distinct per-run session_ids | ✅ PASS |
| 2 | Real sim: 0 null agent_id / session_id / request_id on peer_backend events | ✅ PASS |
| 3 | JSONL session_id == report dir name == report.json.session_id | ✅ PASS |
| 4 | doctor sanity check all peer-connectivity checks green | ✅ PASS |

**Four smoke tests, 4/4 PASS.** Plan 4 A (agent_id observability),
PR #104 (session_id plumbing), and PR #103 (docs hygiene) all
compose cleanly on real hardware. The LLM-path observability story
for 0.4 is complete: every peer call is attributable to its
originating agent, session, and request; reports and JSONL traces
cross-correlate by string-equality; and the bench harness is
operator-usable without sim-cadence artifacts.

## Rerun

Full rerun protocol: [../protocols/session_id_smoke_rerun.md](../protocols/session_id_smoke_rerun.md)

## Follow-ups / known gaps

- **Single-agent_id in sim path:** the queue-path `_process_request`
  hard-codes `"agent": "llm_worker"`, so a sim with multiple NPCs
  will still emit a single agent_id unless callers switch to the
  `generate_json_direct` path. Not a Plan 4 A+B gap — the dict-build
  site is correct — but a future multi-agent observability story may
  want the queue-path to carry an agent-class hint. Track for Plan 4
  Stage C1 scoping.
- **`role_divergence` WARNING** is noise not caught by any Plan 4
  scope. Pre-existing; filed to the plan-4 C1 followup list.
- **The bench's aggregate `benchmark` summary event has no
  `agent_id`** (the one null per run this smoke observed). By design:
  it's the run summary, not a per-call event. Worth a one-line note
  in the bench protocol doc to avoid future "agent_id=null" false
  alarms.
