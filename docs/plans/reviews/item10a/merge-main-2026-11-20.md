# Item 10a, delta round on the merge with main (2026-11-20)

Scope: the merge of origin/main (331 commits since 4b773ef8) into
feat/1315-item10a-offline-router: the conflict resolutions in router.py,
lane_backends.py, api.py, cli.py and research_orchestrator.py, plus the two
re-raises the merge needed (runtime/leader_mode.py::start_claimed_leader_services,
simulation/orchestrator.py::_build_aut_router). Two readers on a frozen copy of
the staged tree.

## Executor: STOP

No resolution drops behaviour from either side or runs a step out of order:
build_primary_router engages, then #1610's process services, then validates
profiles. cli.main engages before _early_leader_boot, and every api verb that
builds a run engages first. The leader-tunnel re-raise is correct and, today,
unreachable: offline, the leader's HTTPServer bind is refused, so an offline
process never claims the services (V8: offline means solo). 314 tests passed in
a copy; the 4 failures were the copy's missing .git.

NITs, outside the merge delta:
- models/download.py::ensure_available asks, then reports "not downloaded",
  offline. 10b's ensure_available refusal fixes this; 10b's re-sync must keep it
  ahead of main's new ask sites (build_run_router's router_model_problem,
  cli._cli_local_router).
- cli.py::_early_leader_boot reports an offline refusal as "port conflict or
  missing CUDA". Folded into 10b.

## Wire integrity: FINDINGS (both folded into 10b)

- SHOULD-FIX cli.py::_curate_router_factory.build (and the outer
  _auto_curate_before_sim handler): an offline refusal from the consented curate
  router is logged at DEBUG, and auto-curate's template fallback can write the
  registry before the sim refuses. Proved on a copy.
- SHOULD-FIX cli.py::_handle_foundry: the refusal reads "LLM init failed — using
  template fallback", the foundry writes template candidates and exits 0. Proved.
- NIT simulation/benchmark.py paper writer: the paper is skipped with "No LLM
  available"; the models are already recorded as refused in 10b.
- NIT cli.py::_early_leader_boot (as above); cli.py::_run_stopped_on (already
  fixed in 10b); the ExecAgent goal-proposal handler (to check in 10b).

All sites checked and cleared are listed in the reviewer's report. Because the
findings are call-site refusals (10b's layer), and 10a merges only with 10b (V22),
they are fixed in 10b's re-sync. That re-sync gets its own delta round.
