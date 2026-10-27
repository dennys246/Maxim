# Item 6 — delta round 2 (Executor + Wire), scope `git show 1b07e050`

Scope: the `router is None` guard in `runtime/agent_pool.py::AgentPool._generate_llm_response` and
`tests/unit/test_router_generate_1338.py::test_generate_llm_response_without_a_router_is_a_typed_error`.

## Findings

**Reachability (Wire).** The only caller of `_generate_llm_response` in `src/` + `scripts/` is
`AgentPool._generate_npc_response`, which calls it only under `if self._llm_router is not None:`. The guard is
therefore unreachable on every shipped path; it exists for mypy narrowing and as a fail-loud tripwire. No
behaviour change on any existing path.

**Pre-guard equivalence.** Even without the guard, a None router would have raised `AttributeError` inside the
`try`, which the broad `except Exception` already converted into `ModelInferenceError("... LLM inference failed:
'NoneType' ...")`. The guard changes only the message (and skips the `from e` chain). Fail-closed both ways.

**`run_turn` contract.** `ModelInferenceError` is an `Exception`; `run_turn`'s outer `except Exception` turns it into a
`TurnResult` with `error`/`error_type`, never raised. Unchanged.

**Bare `AgentPool()` (#1498 follow-up).** `_generate_npc_response` still takes the persona-stub branch when
`_llm_router is None`; the guard is downstream of that branch and never reached. Unchanged. `create.pool`
(create.py) remains the only production constructor and passes a router.

**Test meaningfulness.** Deletion probe (reasoned + confirmed the `AttributeError` the unguarded line raises): with the
guard removed, the call yields `ModelInferenceError("... LLM inference failed: 'NoneType' object has no attribute
'generate'")`, which does not match `"no LLM router"`, so the test fails. Meaningful, though it discriminates by
message only (type is the same either way). Narrow run: 13 passed.

- NIT (fail-closed): the test calls the private method directly and asserts a state the public path cannot produce
  (a router-less pool reaches the stub, not this method). Acceptable as a guard pin; no change needed.
- NIT (fail-closed): the inline comment says "callers route here only with a router" — accurate today; it is the
  only thing tying the guard to `_generate_npc_response`'s check. No change needed.

No BLOCKER, no SHOULD-FIX.

**STOP: no further round owed.**
