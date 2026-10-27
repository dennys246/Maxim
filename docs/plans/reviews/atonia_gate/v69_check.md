# v6.9 narrow delta check (Executor + Wire)

Scope: `git diff 800f4e4c 136b9974`, the folds of `v68_check.md` S1–S4. Read-only. All three touched gates XFAIL as
expected (`pytest -k "process_stop_signals or cursor_carries or huge_trained"`: 3 xfailed).

## Fold 1: `test_process_stop_signals_do_not_stop_the_observer`

**Does it flip only when item 1 is built per v6.9?** Yes, for the session half. Before item 1, `_require_advisor()`
raises AssertionError before anything is set. After item 1, a session whose `should_stop` reads `_SHUTDOWN_EVENT` (or a
future `StopRequest`) makes the FakeBackend return `stopped=True` at `tokens_sampled == 1`. That breaks both asserts, so
the gate stays red. A session that checks a process signal at ingest entry gets a non-`ok` status, which also fails.

**Vacuity.** A session that ignores ALL stops passes this gate. That is acceptable, because the gate's claim is only the
negative half. The positive half is `test_session_stop_abandons_at_the_next_token`, and that test fails for a session
that ignores its own stop. Together the two gates pin "reads the session stop and nothing else". `tokens_sampled == 8`
matches the fake's loop, which increments before it consults `should_stop`, so an early stop shows as a value below 8.

**Leakage.** `request_shutdown()` is set inside `try`, and the `finally` calls `reset_shutdown()`. That reset is the
right one: it clears exactly `_SHUTDOWN_EVENT`, and its docstring names test isolation. `tests/conftest.py` has no
autouse scrub for that event today, so this `finally` is the only reset, and it is sufficient. `maxim.utils.cancellation`
exists today without `request_stop`, so the `getattr` path is a no-op. When bounded_stop PR 0 lands, `_reset_for_tests`
is a guessed name. PR 0 adds its own conftest autouse scrub (bounded_stop §5), so a wrong guess does not leak.

**S1. SHOULD-FIX, fail-open (v68 S2 not fully closed).** The gate runs only against the FakeBackend, and the fake never
consults `_SHUTDOWN_EVENT` itself. v6.9 now states "**`closed_local` must not poll the shutdown event**", and no gate
proves it. A `closed_local` or shared builder that adopts the other backends' idiom, `if is_shutdown_requested(): break`
inside its token loop or its `stopping_criteria`, passes this gate, because the fake replaces it. Once the module entry
was removed (correctly, per v68 S1), nothing covers that half. v68 S2(b) asked for the proof "with the real backend".
The fix is test-only. Do one of the following:
- (a) Parametrize this gate over `["fake", "real"]`, as the gates at lines 821 and 837 are; the real cell uses
  `_require_tiny_gguf()`. Then assert `ok` and `.stopped is False` in the real cell too.
- (b) Add a `closed_local`-level unit gate: set `request_shutdown()`, call `complete_tokens` on a backend whose `Llama`
  is a token-emitting spy, and assert all tokens are sampled.

**N1. NIT.** Assert `report.get("stopped") is False` (or the field §1A.8 names) next to `status == "ok"`.

## Fold 2: the extended `test_cursor_carries_across_ingests`

The test is sound. `_install_fake` puts in place a closure that returns the one `FakeBackend`, so calling
`_session_mod()._backend_factory("", n_ctx=None, ...)` returns that same instance. Because `n_ctx` is None, the call
does not change `backend.n_ctx`. The session gets the same object whether it calls the factory once at `open()` or once
per `ingest()`, so setting `mode = "raise"` reaches it. The asserts match §1A.8. With `error`, `cursor_out == cursor_in`
(feed:102, then feed:102). The next `ok` turn's `cursor_in` is feed:102, so the error turn did not advance it.

**N2. NIT.** Calling the factory adds a spurious entry to the call list that `_install_fake` returns. This test ignores
that list, but `backend = FakeBackend(); _install_fake(monkeypatch, backend)` is simpler and leaves no side effect.

## Fold 3: `test_closed_local_clamps_a_huge_trained_context`

**Spy reach.** `_LlamaCppBackend` resolves `Llama` at call time (`llama_mod = require_optional_dependency("llama_cpp")`,
then `llama_mod.Llama`). That is an attribute lookup on the module object that `_require_llama_cpp()` returns, so
patching `llama_cpp.Llama` is seen there. `closed_local.Llama` is also patched if it exists. Only a builder that did
`from llama_cpp import Llama` at the top of `llama_backend.py` would escape the spy, and it does not do that today.

**S2. SHOULD-FIX, fail-closed (the gate may never flip).** The model path is `"/nonexistent/model.gguf"`. Today the
shared path checks `os.path.exists(model_path)` before `Llama(...)`, and if the file is missing it calls
`_fetch_missing_model`, which can reach `ensure_available` and then returns False. A correct `closed_local` that
validates its path first would also stop before `Llama`. That validation is a reasonable choice, given `open()`'s
regular-file checks, or a builder shared with `_LlamaCppBackend`. Then `except Exception: pass` swallows the refusal,
`seen` stays empty, and the gate fails forever for a correct implementation. The fix is to use an existing file:
`model = tmp_path / "m.gguf"; model.write_bytes(b"GGUF")`. `trained_context` is already patched, so the bytes are never
parsed. Also narrow the `except` to `ScenarioBroken` plus closed_local's own typed error, so that an unrelated
refusal is not swallowed.

**N3. NIT, fail-closed.** Also patch `llama_backend.Llama` when it exists (`monkeypatch.setattr(..., raising=False)`),
for the case where a future builder caches `Llama` at its module top.

## Verdict

No BLOCKER. One fail-open SHOULD-FIX remains, S1: "`closed_local` must not poll the shutdown event" is stated, and only
the session half is proven. S2 is fail-closed. Both are test-only and narrow. Fold S1, using (a) or (b), and S2, then run
one more delta-scoped check on those hunks. **A further round IS owed** under the stop rule, because of S1.
