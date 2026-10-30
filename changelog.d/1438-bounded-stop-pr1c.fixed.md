- **A stopped simulation stops too** ([#1438](https://github.com/dennys246/Maxim/issues/1438) PR 1c,
  docs/plans/bounded_stop.md §5). PR 1b's process stop request ended a campaign's agent loop, but the rest of the
  simulation kept going: the narrator and DM looped on, a direct LLM call ran to its backend's timeout, and the sim's
  own reset re-armed the backend mid-stop.
  - A direct router call (the narrator, the DM, the roundup, research) is refused under a stop request and abandoned
    within about half a second when one lands mid-call. Calls made through an `LLMWorker` are unchanged.
  - `reset_shutdown()` no longer re-arms the backend while a stop is requested.
  - `LLMRouter.wait_ready` returns False on a stop request instead of waiting out the model load.
  - The generative and DM runners stop at the next turn or encounter, never deliver or record a fallback turn made
    from an abandoned call, and report `yielded`, which the run's finish context now reads as `yielded`, not `error`.
  - A simulation started under a stop request raises `StopStillPending` before building anything, and again if the
    stop lands while the model loads.
  - A loop's session end under a stop keeps a reserve for whoever waits on it, so a slow end no longer overruns
    the console's stop. It persists whatever the bound, and a spent bound still flushes queued captures for 0.5 s.
  - A stopped sim's report bounds its provenance waits and fails its skipped checks closed (`unknown: stopped`,
    `model_changed_during_run` unknown), and its container is removed in the background.
