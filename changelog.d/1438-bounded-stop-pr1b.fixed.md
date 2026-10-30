- **A console stop now stops the work it waits for, within a budget** ([#1438](https://github.com/dennys246/Maxim/issues/1438)
  PR 1b, docs/plans/bounded_stop.md §5). A stop used to tell nothing to stop: it waited out a run (30 s), a campaign
  (60 s) and Talk (20 s), then ended the agent's session unbounded, past pulse's ~20 s kill.
  - New config `console.stop_budget_s` (default 15 s; `MAXIM_CONSOLE_STOP_BUDGET_S`; `build_app(stop_budget_s=)` for
    an embedder). `CONFIG_FORMAT_VERSION` is now 1.4.
  - The console's shutdown (and `maxim serve` on SIGINT/SIGTERM) makes a process stop request
    (`utils.cancellation.request_stop`): running agent loops end at their next tick, in-flight LLM calls through
    `LLMWorker` are abandoned within half a second, and backend retries stop (until a campaign's own tail re-arms
    them; PR 1c). Each wait takes a share of the budget, and a loop that ends this way saves every store and
    consolidates only within the budget. The request stops every loop and LLM worker in the process, not only the
    console's.
  - The request is cleared once the stop finished. If anything was still running at the budget (a campaign, a session
    end), it stays set, and a console started again in the same process refuses with `StopStillPending` rather than
    building loops that are born stopped. Restart the process to recover.
  - `MaximHandle.stop(deadline=)` bounds its waits and the agent's own end without stopping anything else in the
    process, and returns whether everything it waited for finished. It does NOT bound the session end of a Talk loop
    that owns the session, which only a process stop request bounds (#1536). `AgentInstance.shutdown` returns
    whether the session end was reached.
  - A simulation that ends because of a stop request reports `yielded`, a typed abort (exit 4), never a completed
    run, even when the stop lands after its last turn. Running campaigns themselves notice the stop in PR 1c.
  - `maxim serve` runs uvicorn through `console/server.py::_MaximServer`; uvicorn's own wait for open requests is
    bounded to a quarter of the budget (it was unbounded).
