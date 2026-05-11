# G3 — Roy fail-fast LLM pre-flight probe

**Date:** 2026-05-11
**Plan:** [persona_convergence_crucible.md](../plans/persona_convergence_crucible.md) (Roy harness § Roy-0 iteration log)
**Status:** Shipped; unit-verified end-to-end. Empirical "abort in <2s on unreachable URL" check still recommended on first user-driven run.
**Companion:** [G4 — substrate-primary cluster_id reward wire](15_g4_cluster_reward_wire.md) (the substrate-primary closure that motivated splitting these as paired PRs).

## What was caught

Roy-0 (2026-05-10, PRs #233/#234) ran end-to-end against a healthy leader for 15 min. The pre-fix dev-box failure mode — broken local 14B + no cloud key — was that `run_roy_iteration` would grind out ~10 min of `dispatch_exhausted` narration on every priming/test call before the iteration finished with empty results. Roy-1's cost ceiling (~$5-15 Claude or ~1500 local LLM calls) on that pattern was not acceptable.

## What shipped

[src/maxim/simulation/roy_runner.py](../../src/maxim/simulation/roy_runner.py) `run_roy_iteration` now invokes `_preflight_llm()` between the interactive-off block and the priming curriculum. The probe:

1. Resolves the configured large lane from `MAXIM_LANE_LARGE_REMOTE_URL` / `_API_KEY` / `_MODEL`.
2. Probes via the canonical entry point `_MaximPeerBackend.for_url(url, api_key=k, model=m).health_check()` (Plan 3 R2.5 — "exactly one HTTP call" invariant; no retry loop in the runner).
3. On non-`ok` and non-`auth_rejected` outcome: sets `result.aborted_at = "preflight"`, populates `result.preflight = {url, outcome, detail, fix, latency_ms}`, persists `result.json` for the operator, returns.
4. `auth_rejected` is a soft-pass — listener is alive; auth errors will surface fast and loud during the actual sim with typed `BackendAuthFailed.fix_hint`.
5. Local-LLM and cloud-only configurations (no `MAXIM_LANE_LARGE_REMOTE_URL`) skip the probe with a documented reason — those failure modes don't have the 10-min grind.

Test seam: production path (no fake `sim_runner`) defaults to `_preflight_llm`. Tests with a fake `sim_runner` skip the probe unless they pass `preflight_fn=` explicitly. Preserves R3's R3-era fake-runner test seam cleanly.

## Result

| Metric | Value |
|---|---|
| New tests | 8 |
| Total roy_runner tests | 26 (all passing) |
| Full fast suite | 6479 passed, 15 skipped, 0 failures attributable to G3 |
| Pre-existing flake | `test_context_index.py::test_similar_text_found` (unrelated; documented as load-order-dependent) |
| Failure window | Probe budget ≤ ~3.3s on standard health_check timeouts (first 0.8s + retry 2.5s) |

## What this does NOT prove

- That a real `maxim roy run` against an intentionally unreachable URL actually aborts in <3s. Unit-verified, not empirically verified. **Recommended first check:** `MAXIM_LANE_LARGE_REMOTE_URL=https://wrong.example.com maxim roy run docs/plans/roy/roy_0_smoke.yaml` against a known-bad URL — should print `aborted_at="preflight"` and exit quickly.
- That the probe correctly handles every backend type. Local-LLM and cloud-only paths skip the probe by design; they retain their pre-G3 failure modes (which surface at first dispatch, no 10-min grind to prevent).

## Reproduction

See [protocols/14_g3_preflight_reproduction.md](protocols/14_g3_preflight_reproduction.md) for the runbook (synthetic failing-URL test + the live `roy run` validation).

## PR

[fix(roy): G3 — fail-fast LLM pre-flight before priming](https://github.com/dennys246/Maxim/pull/235)
