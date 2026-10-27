# #1338 (advisor item 6): Executor lens, DELTA round

Scope: the staged fold (`git diff --cached` on top of d362b481), answering `executor.md`, `architecture.md` and
`wire.md`. Tests were run with `PYTHONPATH=<worktree>/src`. The two new test files, `test_atonia_open_items_other.py`
and `test_agent_factory.py` together give 77 passed and 4 xfailed (O4/O6/O9, which belong to other items). All three
O8 gates PASS, and their file is not in the diff.

## Real-path checks

- **Does `LLMRouter.generate` use the shared dispatch?** Yes: it calls `_complete_text`, the same entry point
  `generate_json` uses. That one path covers:
  - the session cost ceiling
  - the bounded `_inference_lock` (D12)
  - `_complete_text_locked`, which records `call_accounting` start and outcome
  - `_dispatch_locked`, which wraps every provider in `llm_call_registry` `register_call_start`/`register_call_end`
    (the lane comes from `request_context` via `_normalize_request_context`; the pool's `{"agent_id": ...}` dict is
    accepted)
  - the provider loop with `_is_cancelled`
  - `_invoke_backend`, which does the prompt formatting for llama_cpp and the `complete_with_usage` call for cloud
    and maxim_peer

  Nothing in that dispatch is backend-specific.
- **How a failing backend surfaces.** `_try_provider` turns BackendError-class failures into `"failed"`. When every
  provider fails, the result is `"exhausted"` and `_emit_dispatch_exhausted` logs an aggregated WARN with the reason
  for each attempt. `generate` then raises `ModelInferenceError("no text")`. The cause is not masked: it is in the
  WARN, though not in the exception (see N1). An exception the dispatch raises is wrapped `from e`. `LeaderRefusal`
  passes through unwrapped.
- **Stay-up branch.**
  - The main thread stays alive in `threading.Event().wait()`. The wait has no timeout, so the thread blocks; it does
    not busy-wait.
  - Ctrl+C still stops the process: the unchanged `_force_exit_handler` raises `KeyboardInterrupt`, the wait is
    interruptible on POSIX, and the `finally` stops the spawner and returns 0. A second Ctrl+C still calls
    `os._exit(1)`.
  - A Ctrl+C during the 180 s `wait_ready` propagates as before, and the spawner is stopped.
  - A peer can recover a router-`None` leader. `/v1/admin/llm-swap` → `swap_llm_server` needs no active spawner.
- **Is `leader_proxy_running()` reliable?** It is reliable for its documented meaning: this process bound the
  default-port proxy. See N2 for the edges.
- **Do the tests use real objects?** `test_router_generate_1338.py` drives a real `LLMRouter` through the full
  dispatch over a stub backend, and a real `create.pool` / `AgentPool` and `ToolPainBridge`.
  `test_server_mode_1338.py` calls `cli._run_server_mode` in process. It patches only the builder, the spawner stop,
  `threading.Event` and the proxy global.
- **Deletion probes, run on a scratch copy.** Each mechanism was deleted on its own, and each deletion makes at least
  one new test fail (1 to 3 failures each):
  - P1: the empty-text raise in `generate`
  - P2: the stay-up gate
  - P3: the not-ready status write
  - P4: the ping's `server_llm` key
  - P5: the `create.router` raise
  - P6: the personality early return reordered back
  - P7: `error_type` in `run_turn`
  - P8: the spawner-stop log
  - P9: `generate`'s exception wrap
  - P10: the ready status write
- **Prior B1 (diff coverage) is RESOLVED.** I measured this with pytest-cov over the 4 files, against d362b481~1:

  | File | Changed lines covered | Uncovered |
  |---|---|---|
  | cli.py | 59/60 | the pre-existing dispatch line |
  | create.py | 16/16 | |
  | router.py | 14/14 | |
  | agent_pool.py | 21/23 | the concurrent `future.result()` except, and the non-str guard |
  | leader_proxy.py | 7/7 | |

  Overall about 97%.
- **Prior S1 is answered.** Behind a proxy, all three not-ready causes stay up, which matches the owner decision.

## BLOCKER

None.

## SHOULD-FIX

**S1 (fail-closed). Location: `router.py::LLMRouter.generate`, `agent_pool.py::_generate_llm_response`.**
`generate` re-types an `OptionalDependencyError` as `ModelInferenceError`. `_try_provider` deliberately re-raises
`OptionalDependencyError`: the setup error that names the missing extra must surface loudly. `generate_json` lets it
propagate, but `generate`'s `except Exception` wraps it.

I confirmed this with a probe. A stub backend that raises `OptionalDependencyError("anthropic", extra="claude")` makes
`generate` raise `ModelInferenceError: LLM inference failed: required dependency 'anthropic' is not installed. Fix:
pip install 'pymaxim[claude]'`. The fix hint survives in the message and in `__cause__`. However, a caller that
catches `OptionalDependencyError`, or reads `.fix_hint` / `.extra`, misses it. `TurnResult.error_type` then says
`ModelInferenceError` when the cause is a setup error.

Fix:
- add `except (LeaderRefusal, OptionalDependencyError): raise` in `generate`
- let `OptionalDependencyError` through in `_generate_llm_response`, whose `except Exception` would otherwise re-wrap
  it
- add a test with a stub backend that raises it

The failure is still loud, so this is not fail-open.

## NIT

- **N1. `generate`'s "no text" error does not say which cause applied.** Its message lists every possible cause
  (every provider failed, cost ceiling, lock, and also cancellation, which it does not mention). The dispatch outcome
  (`exhausted` / `budget_reject`, the lock timeout) is known but not carried into the exception. The only diagnostic is
  the WARN log. Optional: return or record the outcome so the error can name it.
- **N2. Edges of `leader_proxy_running()` (fail-closed).**
  - It is False when the leader's proxy runs on a non-default `MAXIM_LEADER_PROXY_PORT`, because the singleton is only
    recorded for the default port. That leader exits 1 although a proxy is up.
  - It is False when this process shares another process's proxy (`start_leader_proxy_for_role` returns None). Exit 1
    is right here, since that proxy cannot serve this process's `server_llm`.
  - It stays True if the serve thread dies. This is the same as the existing ready path.

  The docstring states "default-port", so this is consistent.
- **N3. `server_llm` keeps the startup verdict after a recovery.** After a successful `peer llm <model>` swap,
  `server_llm.ready` stays `false` while `llm_ready` turns true, because `swap_llm_server` does not update it. This is
  documented as the startup verdict, and it carries `at`. The pessimistic direction is fail-closed. Consider having a
  successful swap call `set_server_llm_status(ready=True, reason="swapped")`, or say "startup" in the docs.
- **N4. A stayed-up, never-ready leader stopped by Ctrl+C exits 0.** This is as documented ("0 stopped by Ctrl+C").
  A supervisor cannot tell that exit from a healthy run's.
- **N5. `test_tool_pain_bridge_reflection_uses_the_real_router` tests a path no production code reaches.**
  `bootstrap.py` builds `ToolPainBridge` without `llm=`, and no other `.generate(` caller of an `LLMRouter` exists in
  `src/` or `scripts/`. The test shows the path works. It does not show that the path is live.

STOP: no further round owed.
