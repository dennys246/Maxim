# v6.8 narrow delta check (Executor + Wire)

Scope: `git diff 221ad88c 800f4e4c -- tests/` and the matching §1A.8 text. Read-only. The four touched gates XFAIL as expected
(`pytest -k "cursor_carries or default_context_is_clamped or session_stop_abandons or import_closure"`: 6 xfailed).

## Fold 1: `maxim.utils.cancellation` in FORBIDDEN_MODULES

**Can the gate be met today?** Yes. Every current importer of `maxim.utils.cancellation` imports it lazily, inside a
function body: `router._is_cancelled`, `maxim_peer_backend`, and `llm_worker`. The observer reaches none of those
functions. A probe in a child process shows that none of `maxim.models.language`, `maxim.models.language.llama_backend`
(the home of the shared `Llama` builder in §1A.3) or `maxim.runtime.executor` puts it in `sys.modules`.
`utils/deadline.py` does not exist yet. bounded_stop.md §5 makes it a module separate from `cancellation.py`, so it is
not in conflict.

**S1. SHOULD-FIX, fail-closed (the gate could become impossible).** bounded_stop.md §5 says "memory, runtime, console
and the CLI import" StopRequest from `utils/cancellation.py`. The observer needs `runtime.executor`. If PR 1 adds a
module-level `from maxim.utils.cancellation import ...` to `executor.py`, or to anything still in its closure after
step 0a, `test_import_closure` can no longer flip. The same happens if the shared `Llama` builder or `closed_local`
ever honours the per-call cancellation ContextVar, as router and the peer backend already do, because that ContextVar
lives in the same module. The fix is one sentence, in §1A.8 or in bounded_stop §5: a module in the observer's closure
(`runtime.executor`, `models/language/llama_backend.py`, `closed_local`) must not import `utils.cancellation`, even
lazily on the ingest path. Or StopRequest gets its own module, and only that module is forbidden.

**S2. SHOULD-FIX, fail-open (the proof has a hole).** Module absence does not prove the process stop goes unread.
bounded_stop §5 has `request_stop()` set the backend's process-wide event, `models/language/cancellation.py::_SHUTDOWN_EVENT`
(via `request_shutdown()`). Every other backend consults that event (`is_shutdown_requested` / `shutdown_wait` in
anthropic, openai and maxim_peer). If `closed_local` or the shared builder picks up that idiom, `MaximHandle.stop()`
abandons observer sessions without `utils.cancellation` ever loading. That is exactly the cross-session leak that
§1A.8 rules out. Two fixes, best done together:
- (a) Add `"maxim.models.language.cancellation"` to FORBIDDEN_MODULES. Today it is not loaded by `maxim.models.language`
  or `llama_backend`, so this costs nothing.
- (b) A better proof is behavioural, not by module: with the real backend, set the process stop
  (`request_shutdown()` now, and `request_stop()` once PR 1 lands) before `ingest()`, then assert the report is `ok`
  and `.stopped` is False. A runtime spy on `current_stop()` only covers the one accessor; the sticky event can be
  read many ways. Note that the module gate's `fake` parametrization never imports `closed_local`. Only the `real`
  cell exercises it.

## Fold 2: `test_cursor_carries_across_ingests`

The test matches the v6.8 text (the session holds the cursor; the first `cursor_in` is null with no `open(cursor=)`;
on `ok`, `cursor_out` is the last event's cursor). The fixtures are consistent (feed:101/102, then feed:103). Reusing
one FakeBackend across two turns is safe.

**S3. SHOULD-FIX, fail-open (small).** Only the `ok` advance is pinned. The rule that an `error` or `abandoned` turn
must NOT advance the session cursor is untested across ingests. An implementation that advances to the last event's
cursor on every status passes every gate, yet silently skips the failed events on the next `ingest()`, which is data
loss for AvServ. Add a case: `backend.mode = "raise"` for ingest 1, back to `"ok"` for ingest 2, then assert
`r2.covered.cursor_in is None`. With `open(cursor="feed:100")`, assert it stays `"feed:100"`.

## Fold 3: `test_closed_local_default_context_is_clamped`

**S4. SHOULD-FIX, fail-open.** The tiny runnable GGUF is not in the tree yet. A "tiny" model almost certainly has a
trained context ≤ 8192, so `min(trained, 8192) == trained`, and the clamp is never exercised. An implementation that
passes `n_ctx=0` (unbounded, which resolves to the trained context), or that skips the clamp, passes. Fix: monkeypatch
`closed_local.trained_context` to return 131072, and capture the kwargs given to `llama_cpp.Llama` (or the shared
builder). Then assert `n_ctx == 8192` and `n_ctx != 0` in what was passed, not only in `backend.n_ctx`, which an
implementation could echo from its own request. Optionally assert that `trained_context` was called before `Llama`
(the "before the full load" clause).

**N1. NIT, fail-closed.** The test does not call `_require_llama_cpp()`. On a box without llama_cpp, after item 1
lands, it fails with OptionalDependencyError and not the clear "must FAIL, never skip" assertion message.

## The stopped flag and the reason

The FakeBackend matches v6.8: a stop returns `stopped=True, deadline_hit=False`, and the deadline modes return
`stopped=False`. The child FakeBackend gained `n_ctx` and `stopped=False`. `test_session_stop_abandons_at_the_next_token`
asserts `reason == "stopped"`. The docstring seam in the module header (`complete_tokens(..., deadline, should_stop)`,
`.stopped`, no `max_prompt_tokens`, an `n_ctx` attribute) matches §1A.8 A, B and D.

**N2. NIT, fail-open.** The symmetric half is unpinned. `test_report_status_rows[deadline*]` only asserts that `reason`
is truthy, so a deadline reported as `"stopped"` would pass. Assert `reason == "deadline"` there.

## Verdict

No BLOCKER. There are fail-open SHOULD-FIXes (S2, S3, S4). All are test-only and narrow (forbid one more module or add
a behavioural stop probe, add an error-then-ok cursor case, monkeypatch the trained context). Fold them, then run one
delta-scoped check on those test hunks. **A further round IS owed** under the stop rule.
