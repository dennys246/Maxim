# v6.7 delta, Executor lens: §1A.8 and the rewritten 7b gates

Scope: `git diff e7a20742 221ad88c`. That covers notes §1A.8 (A–D), the §1A.10 `complete_tokens` line, the
`test_advisor_gate.py` 7b gates, `FakeBackend`'s `n_ctx`/`on_token`/`should_stop`, and `_install_fake`'s n_ctx
override. Read-only. All 8 changed gates XFAIL today on `_require_advisor()`'s AssertionError
(`-k "limits or token_counts or cursor or stop"`: 8 xfailed). The question: once item 1 is built exactly to v6.7,
does each gate flip only on correct behaviour?

Verdict: **no BLOCKER, 1 fail-open SHOULD-FIX (S1), 3 fail-closed SHOULD-FIX, 5 NIT.** The v6.6 findings S1, S2,
S3 and N1 are resolved by the v6.7 design and the test rewrite. S4 is resolved for the in-process fake only (S3 below).

## Checks that hold

- **Per-token loop and `tokens_sampled <= 4`.** `request_stop` fires in `on_token(2)`, before the increment. The
  counter reaches 3, `should_stop()` is True, and the fake returns. A build that checks the stop only after
  `complete_tokens` returns samples 8, so the assert catches it. A build that never passes `should_stop` also samples
  8. There is no vacuous pass. Every non-`raise` mode now runs the 8-token loop first, which is harmless to the older
  gates because `on_token` is None and no stop is set.
- **Per-call stop.** The `threading.Event` is set at token 1 and reaches the fake only through the session's
  `should_stop`, so a build that ignores `stop=` reports `ok` and fails `first == "abandoned"`. A pre-turn-only
  `is_set()` check misses the mid-turn set, so it also fails. For the second call, `on_token` is cleared and no `stop=`
  is passed, so it runs clean even though `flag` stays set. If a per-call stop wrongly marked the whole session as
  stopped, the second call would raise `AdvisorStopped`. That is not an AssertionError, so the strict xfail fails
  loudly: fail-closed, and the notes say plainly "abandons only that call".
- **`_install_fake` across the three `open()` calls in `test_open_exposes_limits`.** Each call re-installs a FRESH
  `FakeBackend`, so call 1's `n_ctx=2048` cannot leak into call 2's default. Call 2 passes no `n_ctx`, and the fake
  stays at 4096 whether the builder passes `None`, `0` or omits the argument, because the override is guarded by
  `if kwargs.get("n_ctx")`. Call 2 expects `(4096, 3072)`. Call 3 (`2048`/`1500`, so 2524 > 2048) must raise. Call 1
  (1000 + 1024 = 2024 ≤ 2048) is valid. The arithmetic is right.
- **`prompt.tokens == ids`.** §1A.10 (line ~784) pins two untrusted segments per event (`text` and canonical
  `payload`). `FakeBackend.build_prompt` returns `len(segs) + 2` ids, which is 6 for the 2-event `EVENTS`. Exact
  equality is justified. The cap of 2 trips, and `n_ctx=2048` keeps 2 + 1024 within the refusal rule.
- **Existing gates that call `open()` without `n_ctx`.** In process, they get 4096 and a cap of 3072. Their prompts are
  about 6 ids, so nothing trips. `_ingest`'s `turn_timeout_s=5` does not interact with the stop.

## SHOULD-FIX

**S1 (fail-open). "The process-wide StopRequest is never consulted" is asserted by nothing.** Both D docstrings
claim it ("never consulted", and "a process-wide stop request set meanwhile changes nothing"), and the v6.7 fold
table calls the old S3 "moot: no process stop is used". No line checks it. Consider a build that wires
`current_stop()` (bounded_stop PR 0) into `should_stop` as well as the session flag. It passes every 7b gate, because
no test sets the process stop. It then reintroduces exactly the defect v6.7 D was folded to remove: `MaximHandle.stop()`
abandons unrelated embedded sessions. The fix costs one line, with no state to leak. Add `"maxim.utils.cancellation"`
to `FORBIDDEN_MODULES` (the closure ceiling; verified: `import maxim.runtime.executor` does not load it today, and
the only importers are `agents.llm_worker`, `models.language.router` and `maxim_peer_backend`, all outside the
observer's closure). Or drop the two docstring sentences and say the claim is unguarded. A setting-based test would
bring back the old S3 leak.

**S2 (fail-closed). The module's SEAMS docstring still pins the v6.6 backend call.** It still says
`complete_tokens(prompt_ids, *, grammar, max_tokens, max_prompt_tokens, deadline)`, with no `should_stop` and with
the cap that v6.7 moved to the session. It also says nothing about `n_ctx` readback. The docstring says "change them
only with the implementing PR", so a builder who follows it never passes `should_stop`. Both D gates then stay red
on a build that is otherwise correct. Fix: update the SEAMS bullet to §1A.10's new signature, plus
`_backend_factory(gguf_path, *, n_ctx=None, n_gpu_layers, seed)` and "the backend exposes the resolved `n_ctx` as an
int attribute".

**S3 (fail-closed). The `n_ctx` readback is still unpinned, and the child-script fake lacks it (v6.6 S4, half
folded).** §1A.10 still reads `open_closed_local_backend(gguf_path, *, n_ctx, n_gpu_layers, seed)`, which has no
`None` default and no readback. llama-cpp's own `Llama.n_ctx` is a METHOD. A closed backend that mirrors it makes
`session.limits` call `backend.n_ctx()`, which raises TypeError on the fake's int. That is an ERROR, not a flip.
Separately, the child `_CHILD` script's `FakeBackend` (`FakeBackend(completion)`, used by the closure, no-effect and
framing child gates) has no `n_ctx`. Every `open()` in the child then raises AttributeError, which lands in
`OUT["error"]`. The parent's assertion fails, the strict xfail stays XFAIL, and those child gates can never flip on a
correct build. They do not ERROR; they stay masked as "expected". Fix: pin "attribute, int" in §1A.10 and add
`self.n_ctx = 4096` to the child fake.

**S4 (fail-closed, notes). §1A.10's "two-part bound" bullet contradicts the new line above it.** It still says
"`max_prompt_tokens` is checked before `eval`, and an over-long prompt returns `status: "error"` with
`reason: "prompt_too_long"`" inside the backend's paragraph, directly after "The prompt cap is checked by the session
(§1A.8), not here". This is the old two-producer seam that v6.6 S1 flagged. A builder who follows the bullet puts
the check in `closed_local`, the fake ignores it, and `test_prompt_too_long_reports_token_counts` stays red. Fix:
reword the bullet as "the session's cap (§1A.8) bounds prefill; after that, `stopping_criteria` enforces `deadline`
and `should_stop`". Also: the §1A.8 summary table rows for "Context size and prompt cap" ("token counts on a
`prompt_too_long` report") and "Is the cursor opaque…" ("the last judged event's cursor") still state the v6.6
answers. Mark them superseded, because AvServ reads that table.

## NIT

- **N1.** `test_open_exposes_limits`: `err = getattr(advisor, "AdvisorCompositionError", Exception)`. If the name is
  missing, ANY exception satisfies the refusal check (e.g. a TypeError from a bad kwarg). Earlier lines constrain
  most of that, but the name is already pinned in SEAMS, so use `advisor.AdvisorCompositionError` behind an assert.
- **N2.** `test_cursor_required_on_every_event_when_used`: `EVENTS` also lack `occurred_at`, which §1A.8 lists as a
  field. A build that requires `occurred_at` but never checks cursors raises `AdvisorInputError` for the wrong reason,
  and the first half passes. The second half (`_ingest(EVENTS)`) then ERRORs, so this is not fail-open overall. Still,
  add a positive control: a CURSORED batch is accepted in the same session.
  Also `covered = report.get("covered") or {}` makes the null-cursor half pass vacuously when `covered` is absent.
  `REPORT_KEYS` catches that in the schema gate, but assert `"cursor_in" in covered` here.
- **N3.** `test_session_stop_abandons_at_the_next_token` calls `session.request_stop` re-entrantly, from inside
  `ingest` on the same thread. That is faithful to the signal-handler case, and it is correct to test it. But a build
  that takes a non-reentrant ingest lock in `request_stop` hangs rather than failing, because there is no global
  pytest timeout. Add `@pytest.mark.timeout(60)` (pytest-timeout is a dev dependency).
- **N4.** Same test: `assert callable(session.request_stop)` sits before the `try`, so a miss leaks an unclosed
  session. Move it inside the `try`.
- **N5.** Per-call stop: `tokens_sampled` is not asserted, so a build that checks `stop` only after
  `complete_tokens` returns (8 tokens) passes. The notes require per-token checking for `should_stop` generally. Add
  `assert backend.tokens_sampled <= 3` after the first call (the stop is set at token 1, so it should stop at 2).

## STOP rule

No BLOCKER. One fail-open SHOULD-FIX (S1), whose fold is a single tuple entry or a docstring deletion. **A further
round is owed, scoped to the S1 fold's delta only.** S2–S4 and the NITs are fail-closed (gates stay red, never green
wrongly) and can go to the fold batch or a follow-up.
