# v6.6 Executor lens: notes §1A.8 (owner decisions A–D) and the 7b red gates

Scope: commit `e7a20742` only. Read-only. All seven new or widened gates XFAIL today
(`pytest -k "limits or token_counts or cursor_out or stop_request or refusals_signature"` → 7 xfailed), each on
`_require_advisor()`'s AssertionError. The question here is the day after item 1 lands: does each gate flip only
on correct behaviour?

Verdict: **no BLOCKER, 5 SHOULD-FIX, 5 NIT.** Two of the SHOULD-FIXes (S1, S2) are a seam the notes leave open. If
it is filled the other reasonable way, a correct build never flips the gate, which is fail-closed but leaves it red
forever. One (S3) is a fail-open leak of process-global state into later tests.

## SHOULD-FIX

**S1. `test_prompt_too_long_reports_token_counts`: §4.2 has the BACKEND enforce the cap, but the gate needs the
SESSION to enforce it.** §4.2 ("The bound is two-part") pins `complete_tokens(..., max_prompt_tokens, ...)` and
"`max_prompt_tokens` is checked before `eval`", which places the check inside the closed backend. `FakeBackend.complete_tokens`
ignores every kwarg. Scenario: item 1 is built to §4.2, with the check in `closed_local.complete_tokens` raising
or flagging. The fake completes, so `backend.completions == 1` and the status is `ok`. The gate raises
AssertionError forever on a correct build (fail-closed, never flips). v6.6 B also does not say how `prompt_tokens`
reaches the report if the backend owns the check (a typed exception carrying the counts? a flag on the
completion?). Fix: pin ONE owner in §1A.8/§4.2. The recommended owner is the session: compare
`len(prompt_ids.ids) > max_prompt_tokens` BEFORE calling `complete_tokens`, with the backend's own check kept as
defence in depth. Name it in the test module's SEAMS list. The other option is to make the fake honour
`max_prompt_tokens` the way the real backend would, and pin the typed error.

**S2. `test_ingest_honours_the_process_stop_request`: the gate needs a check AFTER `complete_tokens` returns, and
the notes never pin one.** `_StopDuringCompletion` calls `request_stop` and then returns a valid completion with
`deadline_hit=False`. That is right: it tests the stop alone, not the deadline row. But v6.6 D says "a stop that lands
during sampling ends the turn at the next token", which a builder will naturally wire into the per-token
`stopping_criteria`, setting the same flag the deadline uses. The fake never runs a criterion. Scenario: a correct
criterion-only build reports `ok`, and the gate never flips (fail-closed). Fix: add one line to §1A.8 D, mirrored
in §1A.8's precedence rule: "after `complete_tokens` returns, the turn re-checks `current_stop()` and the per-call
`stop`; a set stop makes the turn `abandoned` and the output is not parsed (same precedence exception as the
deadline flag)". "A stop that lands during prefill takes effect when prefill returns" already implies this check,
so pin it explicitly.

**S3. `test_ingest_honours_the_process_stop_request`: the reset is conditional on an unpinned name. Fail-open
leak.** The `finally` calls `cancellation._reset_for_tests` only `if callable(...)`. bounded_stop.md pins no such
name; it says only "a conftest autouse scrub clears the stop request", and it places that in **PR 1** §5, while
`request_stop`/`current_stop`/`Deadline` land in **PR 0** (§3 item 1). In the PR 0 → PR 1 window, or if the scrub
gets another name, the sticky stop survives the test. Per bounded_stop §5, `request_stop` ALSO sets the backend's
process-wide `request_shutdown()` and every registered stop event. Per §6, it also flips the CLI gate's
"a stop was ever requested" monotonic state. Scenario: every later in-process advisor gate in the same worker
returns `abandoned`, fails its `status == "ok"`/`"error"` assert with an AssertionError, and is
reported **XFAIL**. The gates are masked, not red. After they flip they fail for the wrong reason. Later
backend/CLI tests also see a set shutdown. Fix: name the reset in bounded_stop.md §4/§5 (it is PR 0's contract,
since PR 0 introduces the state). In the test, `assert callable(reset)` BEFORE `session.open`, so a missing reset
is an AssertionError and not a leak, and call it unconditionally. Ideally the conftest scrub lands in PR 0 with
the state, and also re-arms `reset_shutdown()`. Also put the reset in its own `finally`, so a raising
`session.close()` cannot skip it.

**S4. `n_ctx` default resolution versus the fake model and the factory seam. Every in-process gate ERRORs.**
v6.6 A: "`n_ctx` defaults to the GGUF's trained context" and "`max_prompt_tokens` defaults to `n_ctx` minus the
completion budget". The factory seam `_backend_factory(gguf_path, *, n_ctx, ...)` needs `n_ctx` BEFORE the backend
exists, so a builder reads the GGUF header, and `_dirs()` writes `b"GGUF-fake"`. Or the builder passes
`n_ctx=0` and reads it back from the backend, and `FakeBackend` (and the child script's `FakeBackend`) has no such
attribute. Scenario: every gate that omits `n_ctx` (schema, status rows, cursor, stop, prompt-too-long, the child
gates) raises a GGUF-parse error or an AttributeError, not an AssertionError. Under `raises=AssertionError` that is
a loud FAIL (fail-closed), but no gate can flip until the tests change. Fix: pin the resolution seam. Recommended:
the factory takes `n_ctx=None`, and the backend exposes `n_ctx` (resolved); `session.limits` reads it from there.
Give both fakes an `n_ctx` attribute now. Also pin the completion budget (`max_tokens`): see N1.

**S5. Notes consistency: the status table and `degraded`.**
- §4.2's status table still has "Item 10 marked an input degraded | `degraded`". §6 (item 10, "The fallbacks become
  loud") still says "records a `degraded` reason, and the observer's report says `degraded`". Both contradict
  v6.6's "no `degraded` producer in this release". AvServ designs against these rows (plan 50), so leaving them
  invites a consumer branch for a status that never comes. Fix: strike or footnote both rows as "no producer in
  this release (v6.6)".
- §4.2's status table has no `prompt_too_long` → `error` row and no stop → `abandoned` row. §1A.8's precedence
  exception names only "the deadline flag" (see S2).
- §4.2 still writes `session.ingest(events) -> AdvisorReport`. `open` got a pointer to §1A.8, but `ingest` did
  not, and it lacks `*, stop=None`.

## NIT

**N1. `test_open_exposes_limits::max_prompt_tokens=1500` with `n_ctx=2048`.** If `open()` validates
`max_prompt_tokens + completion budget <= n_ctx`, which is a sensible refusal and the converse of v6.6's default
rule, a budget over 548 makes `open()` raise `AdvisorCompositionError` (not AssertionError). Use `n_ctx=4096`, or
pin the budget and the validation. Also pin `limits` as attribute-bearing: a Mapping `limits` reads as all-None
through `getattr` (fail-closed).

**N2. `test_prompt_too_long_reports_token_counts`: the count assertion is loose.** `3 <= prompt_tokens <= 70`
accepts wrong counts such as `len(segments)` (4) or `n_events + 1`. `FakeBackend.build_prompt`'s ids length does
not depend on the template, so "the template may add ids" does not apply to the fake. The arithmetic itself is
right: §4.2 makes each event's `text` and its `payload` separate untrusted segments, so 2 per event, which is
`1 + 4 + 1 = 6`. Record `len(ids)` in the fake and assert `extra["prompt_tokens"] == that` (fail-open today on
miscounts).

**N3. `occurred_at`.** v6.6 lists each event as `{event_id, kind, occurred_at, payload, text, cursor?}`, with
`occurred_at` not marked optional. `EVENTS`, `CURSORED` and the module docstring's seam ("Events are dicts
`{event_id, kind, text, payload}`") lack it. A builder that validates it rejects or raises on every gate's input.
Either mark it `occurred_at?` or add it to the fixtures and the seam line.

**N4. The stop gate's per-call `stop=` is only signature-checked.** An accepted-and-ignored `stop=` passes. Either
add a parametrised case that passes `stop=` with no process request, or say in the docstring that only the
signature is pinned. Separately, `Deadline(5.0)` is constructed inside the fake's `complete_tokens`. If PR 0 picks
another constructor, the TypeError becomes a completion `error` and the gate stays red for the wrong reason.
Construct the `Deadline` in the test body before `open()`, so drift is loud. The `_StopDuringCompletion` docstring
("answers as the deadline would") is wrong: it answers `ok` with `deadline_hit=False`, which is the correct design.

**N5. The test module docstring** still says "DRAFT v6.4" and does not list the 7b gates in "THE GATES, AND WHAT
FLIPS EACH".

## Checked and sound

- **The import order in the stop gate.** `maxim.utils.cancellation` exists today, so `import_module` succeeds, and
  a missing `request_stop`/`current_stop` is the `hasattr` AssertionError, not an ImportError. `maxim.utils.deadline`
  (absent today) is imported only inside the fake, after that assert passes. `_require_advisor()` runs first.
- **The cursor gate.** `ok` → `"e2"` (e2 carries no cursor, so its `event_id`). The second session over `[e1]` →
  `"feed:101"`. `error`/`abandoned` → `"feed:100"`. All match v6.6 C. A second `open()` in one test is fine: the
  latch's `engage()` is idempotent, and session ids give distinct `O_EXCL` log names.
- **The widened `test_builder_refusals_signature`** matches §1A.8's list exactly. **`test_open_exposes_limits`**
  checks the factory kwargs against the pinned seam names.
- **The AvServ table and the status line** are consistent with §1A.8.
