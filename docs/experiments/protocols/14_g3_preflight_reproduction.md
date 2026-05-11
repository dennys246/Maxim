# Reproduction — G3 Roy fail-fast LLM pre-flight probe

**Companion:** [14_g3_roy_preflight_probe.md](../14_g3_roy_preflight_probe.md)

## Prerequisites

- Maxim repo checked out, `pip install -e .` (or `PYTHONPATH=src`).
- For the live check (B): a Roy iteration spec on disk (the bundled `docs/plans/roy/roy_0_smoke.yaml` works).
- For the live check (B): no real leader running — we WANT the probe to fail.

## A. Unit-level reproduction (offline, deterministic)

The 8 regression tests cover the four runner-level paths and the four helper-level paths:

```bash
python -m pytest tests/integration/test_roy_runner.py::TestRoyPreflight \
                 tests/integration/test_roy_runner.py::TestPreflightHelper \
                 -v
```

Expected: 8 passed in <1s.

Each test exercises one branch:

| Test | What it pins |
|---|---|
| `TestRoyPreflight::test_preflight_failure_aborts_before_priming` | Failing probe → `aborted_at="preflight"`, no arms run, `result.json` persisted, `preflight.outcome="connection_refused"` recorded. |
| `TestRoyPreflight::test_preflight_pass_runs_full_iteration` | Passing probe → all 3 arms run, `aborted_at=None`, `preflight.outcome="ok"` + `latency_ms` recorded. |
| `TestRoyPreflight::test_preflight_skipped_when_fake_sim_runner_injected` | Fake `sim_runner` + no explicit `preflight_fn` → probe skipped, `result.preflight == {}`, iteration completes. |
| `TestRoyPreflight::test_preflight_raising_treated_as_failure` | `preflight_fn` raises → treated as preflight failure (no crash), `preflight.outcome="preflight_raised"`. |
| `TestPreflightHelper::test_skips_when_no_remote_url_configured` | No `MAXIM_LANE_LARGE_REMOTE_URL` → returns `(True, {skipped: True})`. |
| `TestPreflightHelper::test_probes_when_remote_url_configured` | URL + key + model env vars → helper calls `_MaximPeerBackend.for_url(url, api_key=k, model=m).health_check()` and surfaces the outcome. |
| `TestPreflightHelper::test_auth_rejected_is_soft_pass` | `auth_rejected` outcome → returns `(True, {soft_pass: True})`. |
| `TestPreflightHelper::test_health_check_exception_treated_as_failure` | `health_check` raises → returns `(False, {outcome: "probe_error"})`. |

## B. Live reproduction (unreachable URL, ~3s)

This is the empirical check the unit tests don't cover: a real `maxim roy run` against a deliberately-unreachable URL should abort in ≤3.3s with a useful operator message.

```bash
# 1. Make sure no leader is running on the configured URL (or pick a bogus one).
unset MAXIM_LANE_LARGE_REMOTE_URL  # clear any env config first
export MAXIM_LANE_LARGE_REMOTE_URL=https://leader.unreachable.example.com
export MAXIM_LANE_LARGE_REMOTE_API_KEY=sk-placeholder
export MAXIM_LANE_LARGE_REMOTE_MODEL=qwen2.5-14b-instruct

# 2. Time the run. Expect <10s wall total (probe is ≤3.3s; rest is setup + persistence).
time maxim roy run docs/plans/roy/roy_0_smoke.yaml
```

Expected stdout (abbreviated):

```
Roy iteration 'roy-0-smoke' aborted at preflight:
  outcome=connection_refused
  detail=<connection error from urllib>
  url=https://leader.unreachable.example.com
  fix=Leader not accepting connections — start `maxim` on the leader
```

Expected `~/.maxim/roy/roy-0-smoke/result.json` fields:

```json
{
  "aborted_at": "preflight",
  "preflight": {
    "url": "https://leader.unreachable.example.com",
    "outcome": "connection_refused",
    "detail": "...",
    "fix": "..."
  },
  "priming": {"error": "preflight: ..."},
  "arms": {},
  "pairwise_diffs": {}
}
```

Expected `~/.maxim/roy/roy-0-smoke/summary.md` has a `## Pre-flight` section with the same outcome + fix.

Pass criterion: total wall < 10s, `aborted_at == "preflight"`, no arms invoked.

## C. Auth-soft-pass check (optional)

To confirm `auth_rejected` doesn't block the iteration (the listener-is-alive soft-pass):

```bash
# Point at a real leader (or any HTTP endpoint that returns 401)
# with a bad key.
export MAXIM_LANE_LARGE_REMOTE_URL=https://your-real-leader.example.com
export MAXIM_LANE_LARGE_REMOTE_API_KEY=sk-deliberately-wrong
maxim roy run docs/plans/roy/roy_0_smoke.yaml
```

Expected: the iteration proceeds past preflight (`result.preflight.outcome == "auth_rejected"`, `result.preflight.soft_pass == True`), then the actual priming sim calls fail loudly with `BackendAuthFailed.fix_hint`. This confirms preflight isn't over-zealous.

## D. Cleanup

```bash
unset MAXIM_LANE_LARGE_REMOTE_URL MAXIM_LANE_LARGE_REMOTE_API_KEY MAXIM_LANE_LARGE_REMOTE_MODEL
rm -rf ~/.maxim/roy/roy-0-smoke  # only if you don't want to keep the failed-preflight artifact
```
