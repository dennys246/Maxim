# #1315 item 3: delta round 3 (76ec17aa, prepr's mypy ratchet fold)

Scope: `git show 76ec17aa` only. `create_full_agent` passes `require_policy(config.policy, where="create_full_agent(with_executor=True)")` to `build_executor`; one test in `test_tool_policy.py::TestReviewFolds`.

1. **Narrowing.** It is the same `type(policy) is ToolPolicy` check that `build_executor` runs. Only the error prefix changes. A config invalid in two ways now raises the policy TypeError before the ValueError from `_check_build_executor_args`; both fail fast before any construction, and no test pins the old order.
2. **Cleanup.** The raise sits inside the same `try:` as before, so it reaches the same `except BaseException: instance.shutdown(); raise`. Nothing was built between the old raise point and the new one. No new swallow.
3. **Test.** It fails on revert: `build_executor`'s message does not match `create_full_agent\(with_executor=True\)`.
4. **Callers.** Every `with_executor=True` AgentConfig in `src/` passes a real policy (cli, api, the orchestrator x2, console).

NIT: the test runs with `with_bio_stack=False`, so it skips the bio-stack shutdown path on this raise. That code is unchanged; nothing owed.

**Verdict: STOP.**
