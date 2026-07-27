# Long-lived-server config guards — `_lane_env_applied` + role export

**Status:** deferred shell plan (post-merge review round, 2026-07-26).
**Trigger to revive:** the console wizard needs *run → re-setup → run* to fully
re-route lanes in one `maxim serve` process, OR Phase 3 talk/rest modes ship
(which make long-lived reconfiguration routine).

## The problem class

Every once-per-startup idempotency guard in the config/lane layer was designed
for **one-shot CLI processes**. `maxim serve` is the first process that both
WRITES config (`/api/setup/*`) and REBUILDS routers in-process (`/api/run`),
so those guards now have a second, unanticipated failure mode: *stale by
design* within a server's lifetime.

Fixed already (the same review round): the `get_config()` singleton —
`write_config`/`mutate_config` now call `invalidate_config_cache()`, so
post-setup lane builds read fresh config. Two guards remain, deliberately NOT
band-aided here because each couples to a reviewed invariant:

1. **`lane_backends._lane_env_applied`** (once-per-process): after the first
   lane build, a later `apply_mesh_setup` write does not repopulate the
   `MAXIM_LANE_*` env vars, so *run → mesh-setup → run* keeps the old routing
   until restart. The guard exists because a second `_apply_lane_config_to_env`
   call would re-attribute the doctor's source column from "config" to "env"
   (post-implementation Executor C1 fold) — clearing it naively re-breaks C5
   source attribution.
2. **`MAXIM_ROLE` export at startup** (`detect_and_apply_role`): a
   `config.json::role` written by setup after startup is shadowed by the env
   export for the process lifetime. Load-bearing: role detection is
   FIRST-runtime-action by invariant, and downstream reads env only
   (never re-detects) — per the role.py invariant, re-detection mid-process is
   currently forbidden, so this needs a deliberate contract change, not a hack.

## Options to evaluate at revival

- (a) A `reload_lane_env()` verb that clears `_lane_env_applied` AND re-runs
  source attribution coherently (teach the decision log a "reloaded" source).
- (b) Restart-required semantics made LOUD: `/api/setup/*` responses carry
  `restart_required: true` when a guard would shadow the write; the wizard
  surfaces it. (Cheapest honest option; no invariant changes.)
- (c) Scoped re-exec: the server relaunches its worker process after setup
  writes (systemd-style), keeping one-shot semantics everywhere.

Option (b) is the recommended near-term shape; (a) is the real fix if
in-process re-routing becomes a product requirement.

## Regression guard (when revived)

An integration test driving one process through *lane build → mesh setup →
lane build* asserting either fresh routing (a) or a surfaced
`restart_required` (b).