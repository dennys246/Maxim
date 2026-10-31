- **A stopped simulation stops too** ([#1438](https://github.com/dennys246/Maxim/issues/1438) PR 1c,
  docs/plans/bounded_stop.md §5). PR 1b's process stop request ended a campaign's agent loop, but the rest of the
  simulation kept going: the narrator and DM looped on, a direct LLM call ran to its backend's timeout, and the sim's
  own reset re-armed the backend mid-stop.
  - A direct router call (the narrator, the DM, the roundup, research) is refused under a stop request and abandoned
    within about half a second when one lands mid-call; a Ctrl+C during the call now cancels it too. An `LLMWorker`
    call still abandons on the worker's own stop event, as before. `LLMRouter.generate` raises the new
    `LLMCallStopped` (a `ModelInferenceError`) for a stopped call, so a stop no longer reads as a model failure.
  - `reset_shutdown()` no longer re-arms the backend while a stop is requested.
  - `LLMRouter.wait_ready` returns False on a stop request instead of waiting out the model load.
  - The generative and DM runners stop at the next turn or encounter, never deliver or record a fallback turn or
    choice made from an abandoned call, and report `yielded`, which the run's finish context now reads as
    `yielded`, not `error`.
  - A simulation started under a stop request raises `StopStillPending` before building anything (its start
    provenance included), and again, releasing its serving registration, if the stop lands while the model loads.
  - A loop's session end under a stop keeps a reserve for whoever waits on it, so a slow end no longer overruns
    the console's stop. It persists whatever the bound, and a spent bound still flushes the session's own queued
    captures for 0.5 s. An exit with no stop keeps its whole bound.
  - A stopped sim's container is removed in the background. `run_provenance` / `serving_provenance` gain a
    `deadline=` that bounds every report-time wait and fails skipped checks closed (`unknown: stopped`,
    `model_changed_during_run` unknown); the sim's stop tail passes it.
  - A stopped sim's end fits inside the console's stop: it saves an early, partial report.json and action log
    (`provenance.stop_stage: "early"`) before any wait, bounds its waits for the agent and DM threads and the
    orchestrator's own shutdown by a share of the stop's deadline (an adopted agent's join no longer waits 300 s),
    saves the full report (`stop_stage: "late"`) only if time is left, skips the LLM roundup, and never shows the
    interactive post-sim prompt. A failed save never skips the shutdown and release that follow it. While the agent's
    loop is still ending its session, the end leaves its state alone: no tool lease restore (by the sim or by
    `MaximHandle`'s safety net), no pain-bridge unsubscribe, no imagination cleanup, no subsystem snapshot. With no stop, the end of a run is unchanged; its code moved to
    `simulation/sim_teardown.py`.
