# Item 10b, delta round 6: the delta-5 folds (d628e422, 2026-11-21)

Scope: d628e422 only. `ExecAgent.share_run_llm` takes over `MaximAgent.share_run_llm`'s
body (MaximAgent now delegates). The commit also adds docstrings, the breaking
fragment, the llm-routing brief line and two contemplation-test edits. One reader.

**Verdict: STOP.**
- The moved share rule is identical. The router is set only when it is None. The
  worker is set only when it is None and one is given, and then
  `_owns_llm_worker=False`.
- The docstring claims hold at the SHA. `_ensure_router`'s one caller in `src/` or
  `scripts/` is `cli.py` with `consent="ask"`. Nothing calls
  `MaximAgent.on_start`/`ExecAgent.on_start`.
- The breaking fragment follows `changelog.d/README.md`, and its migration matches
  the constructor. `MaximAgent` is not in `docs/user/stable_api.md` (checked by the
  author).
- Both contemplation tests still test what their names say.
- 146 tests passed in a copy.
