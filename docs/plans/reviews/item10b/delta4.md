# Item 10b, delta round 4: the re-sync with main (2026-11-21)

Scope: commit bf0955a5, the merge of the updated 10a (f826e33c, 10a + origin/main
043eab8f) into 10b (85ba5751, delta round 3 STOP), and its folds from 10a's merge
review (cli.py's curate/auto-curate/foundry/early-leader-boot re-raises, the
benchmark paper writer). Two readers, each on an exported tree.

## Executor: STOP

`_ensure_lane_profiles_available` walked branch by branch. Online, main's
OD2/#1658 path is unchanged. Offline and pinned (the build's own `profile`, or
`_profile_is_pinned`), `ensure_available` refuses by name before consent. Offline
and auto, nothing asks, and the degraded records name `lane_profile`. The warmup
thread's refusal clause keeps #1657's deferred release and the ready event.
Online decision records keep their shape. Each cli.py re-raise reaches
`cli.main`'s exit 2, and the curation routers are unloaded in a `finally`.

NIT, folded: `write_paper` booked an offline "degraded" entry for any security
refusal; it now does so only for an `OfflineRefusal` (test
`test_a_leader_refused_paper_router_is_not_recorded_as_offline`).

## Wire integrity: FINDINGS

- SHOULD-FIX `agents/exec_agent.py::ExecAgent._propose_goal`: its catch-all turns
  an OfflineRefusal from the agent's own lazy router build into "Language model
  error" and no goal. Proved by calling `_propose_goal` directly. On a closer
  read the path has no caller in `src/` or `scripts/`: the proposal worker starts
  only from `MaximAgent.on_start`, which nothing calls, and `deliberate()` has
  none. The one live path, `maxim --agentic`, builds the router eagerly
  (`_ensure_router(consent="ask")`) and exits 2. The owner chose to fix the root
  cause anyway (**V29**): every `MaximAgent` takes the run's router, and an
  ExecAgent never builds its own. See the notes §6 status and the next delta
  round.
- NIT `models/download.py::download_file` (`api.download_model`, the download
  CLI): a refused download reads as a generic failure, not typed. Fail-closed.
  For 10c.
- NIT `models/audio/tts.py::_VoiceProvisioner._run` retries a refused voice
  with backoff, booking a ledger entry each try. Fail-closed. For 10c.

All other sites checked were cleared: lane preflight, `build_run_router`, the
AUT and research builders, agentic_runtime, leader_mode, console talk,
`_cli_local_router`, the benchmark paper, `llama_backend`, router warmup, rtm,
transformers and the hivemind client, the spawn records. No 10b refusal is dead
or bypassed after the merge.
