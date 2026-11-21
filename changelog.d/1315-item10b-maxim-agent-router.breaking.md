- **`MaximAgent` requires `llm_router=`** (#1315 item 10b, owner decision V29): the constructor takes the run's router
  as a required keyword-only argument (and an optional `llm_worker=`), and its ExecAgent never builds a router, a
  worker or a model of its own. `MaximAgent()` with no `llm_router` raises `TypeError`. Migrate:
  - `MaximAgent(...)` in a run that has a router → `MaximAgent(llm_router=router, llm_worker=worker, ...)`;
  - an agent whose router is built after it → `MaximAgent(llm_router=None, ...)`, then
    `agent.share_run_llm(router, worker)`;
  - an agent with no LLM → `MaximAgent(llm_router=None, ...)`.

  `MaximAgent` (and its alias `AgenticMaximAgent`) is exported from `maxim.agents` but is not in
  `docs/user/stable_api.md`; the stable verbs (`maxim.run`, `create.*`, `load.*`) pass the router themselves, so
  their signatures are unchanged.
