# G3 — Roy fail-fast LLM pre-flight probe

**Date:** 2026-05-11
**Plan:** [persona_convergence_crucible.md](../plans/deferred/persona_convergence_crucible.md) (Roy harness § Roy-0 iteration log)
**Status:** Shipped + follow-up fold for peer.yml fallback. The G4 Roy-0 re-run on 2026-05-11 surfaced that the original probe was a no-op for the standard peer-with-peer.yml setup (env vars are exported by `apply_peer_config_to_env` only at lane resolution, which happens AFTER `_preflight_llm`); the follow-up reads `~/.config/maxim/peer.yml` directly when env vars are absent.
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

## Follow-up fold: peer.yml fallback (post-Roy-0)

The 2026-05-11 Roy-0 re-run (G4 empirical validation) revealed that the original probe was silently skipping under the canonical peer-leader setup. `apply_peer_config_to_env` in [runtime/lane_backends.py:1073](../../src/maxim/runtime/lane_backends.py) reads `~/.config/maxim/peer.yml` and exports `MAXIM_LANE_LARGE_REMOTE_*` env vars — but only at lane resolution, which fires AFTER `_preflight_llm`. Operator who runs `maxim roy run` with no env vars exported but a valid `peer.yml` got `result.preflight = {"skipped": True, "reason": "MAXIM_LANE_LARGE_REMOTE_URL not set"}`, leaving the broken-leader failure mode uncaught.

The fix: `_preflight_llm` now reads `peer.yml` directly when env vars are absent, falling back to that config source before deciding to skip. Resolution order:

1. `MAXIM_LANE_LARGE_REMOTE_URL` / `_API_KEY` / `_MODEL` env vars (explicit per-session override).
2. `~/.config/maxim/peer.yml` via `read_peer_config()` (the canonical peer-leader setup).
3. Otherwise: skip the probe (local-LLM / cloud-only setups don't have the 10-min grind failure mode).

`result.preflight.source` field records which path was used (`"env"` or `"peer.yml"`) so operators can verify their config was picked up. Env always wins when both are present.

Regression guards: `TestPreflightHelper::test_peer_yml_fallback_when_env_not_set` (asserts URL/key/model are read from peer.yml when env is absent) + `TestPreflightHelper::test_env_takes_precedence_over_peer_yml` (env wins when both present).

## What this does NOT prove

- That a real `maxim roy run` against an intentionally unreachable URL actually aborts in <3s. Unit-verified, not empirically verified. **Recommended first check:** `MAXIM_LANE_LARGE_REMOTE_URL=https://wrong.example.com maxim roy run docs/plans/roy/roy_0_smoke.yaml` against a known-bad URL — should print `aborted_at="preflight"` and exit quickly.
- That the probe correctly handles every backend type. Local-LLM and cloud-only paths skip the probe by design; they retain their pre-G3 failure modes (which surface at first dispatch, no 10-min grind to prevent).

## Reproduction

See [protocols/14_g3_preflight_reproduction.md](protocols/14_g3_preflight_reproduction.md) for the runbook (synthetic failing-URL test + the live `roy run` validation).

## PR

[fix(roy): G3 — fail-fast LLM pre-flight before priming](https://github.com/dennys246/Maxim/pull/235)
