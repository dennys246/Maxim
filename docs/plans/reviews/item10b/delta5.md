# Item 10b, delta round 5: V29, every MaximAgent takes the run's router (2026-11-21)

Scope: commit 4293b592 (parent bf0955a5): `MaximAgent(llm_router=)` (required,
keyword-only) and `share_run_llm`; ExecAgent builds no LLM of its own on the
proposal path; `_propose_goal` re-raises `SECURITY_REFUSALS` and the proposal
worker ends on one; the eight construction sites; the `write_paper` NIT fold.
Three readers, each on an exported tree.

## Architecture: STOP

There is no layering violation: `agents/` imports only `maxim.exceptions`, plus
TYPE_CHECKING imports. There is no duplicate mechanism either. `llm_router=` feeds
ExecAgent's existing `shared_router`/`shared_llm_worker`, and the eight sites need
different routers, so a builder would only carry the same argument. The helpers sit
next to `_build_aut_router` and `_cli_agentic_llm_worker`.

Folded:
- `ExecAgent.share_run_llm` owns the rule, and `MaximAgent.share_run_llm`
  delegates to it, instead of writing ExecAgent's ownership fields.
- The docstring says that a router passed later is ignored, not swapped in.
- The worker-loop comment says that no caller starts that worker today.

Recorded as triggers, not changed now:
- `ExecAgent.__init__`'s `shared_router` still defaults to None. That is harmless
  while `MaximAgent` is its only constructor.
- `llm_router=None` means three things: shared later, built by the owner, or no
  LLM. A builder that passes None and forgets to share would get an ExecAgent that
  proposes nothing. That cannot happen today, because nothing starts the proposal
  worker. **Trigger:** when a caller first starts it (`ExecAgent.on_start`,
  `deliberate`), replace the bare None with a typed sentinel, or make the worker
  warn once on a None router.

## Executor: STOP

Online, production never used ExecAgent's lazy builds. The proposal path is
dormant: `_worker_loop` starts only from `MaximAgent.on_start`, which nothing
calls, and `deliberate()` has no callers. The CLI's order of operations is intact:
`_build_cli_agentic_agent`, then `_cli_agentic_llm_worker`'s eager
`_ensure_llm`/`_ensure_router(consent="ask")`. api.run's and console's workers
are still stopped by their owners, and ExecAgent stops nothing it does not own.
`share_run_llm` is equivalent to the robot runtime's old block. The worker
loop's return is safe for `on_stop`. 53 and 51 tests passed in the copy.

NITs:
- Folded: `_resolve_profile_override`'s docstring says the
  `agent_profiles.exec_agent` override is read only where the agent builds its own
  router (the CLI agentic path).
- Recorded as a trigger: in the robot runtime, a router that builds but whose
  warmup or `LLMWorker` raises leaves `llm_worker` None, so it is not shared. The
  fix is to call `share_run_llm` whenever `llm_router` is not None. It is dormant
  today, and the line sits inside `_start_agentic_runtime`, which is pinned. Fold
  it with the trigger above, when something first starts the proposal worker.

## Wire integrity: one SHOULD-FIX, then STOP

- SHOULD-FIX, folded: `MaximAgent` (with its alias `AgenticMaximAgent`) is exported
  from `maxim.agents`, so the required `llm_router=` is a public break. It is now in
  `changelog.d/1315-item10b-maxim-agent-router.breaking.md`, with the migration,
  following the item-3 precedent. It is not in `docs/user/stable_api.md`.
- NITs, folded:
  - `ExecAgent._ensure_router`'s docstring and `docs/agents/llm-routing.md` no
    longer describe a lazy `consent="never"` library build.
  - `tests/unit/test_contemplation.py`'s dead `_ensure_llm` mocks are gone; one of
    them now fails if it is called.

Every `MaximAgent(` construction passes `llm_router=`: the 8 in src, the 6 in
tests, and none in scripts, docs or README. There are no subclasses. Cleared
outside the diff: `_refuse_if_named_model_missing`, the orchestrator's warmed AUT
router (the same object), console talk's `worker.router`, the robot runtime's
share, the sub-sim tool, minecraft, the `ExecAgent.__new__` tests,
`test_direct_router_consent_1396b2`, `maxim.create`, AgentFactory, personas,
hivemind and research. 273 narrow tests passed in a copy.

## After the round

The folds after this round are: the `ExecAgent.share_run_llm` move (a
behaviour-neutral relocation of `MaximAgent.share_run_llm`'s body), docstrings,
the changelog and the tests. They get a delta-scoped round 6 on the fold commit.
