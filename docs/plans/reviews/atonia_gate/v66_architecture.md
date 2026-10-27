# v6.6 review: Architecture lens (commit e7a20742)

Scope: notes §1A.8 (A–D, `degraded`), the §4.2 pointer, the §9 AvServ table, and `tests/integration/test_advisor_gate.py` §7b.
Read against bounded_stop.md §5/§6, advisor_mode.md item 12 and report half, notes §0, §1A.3, §4.2 and §6.

**Verdict: no BLOCKER. 6 SHOULD-FIX, 5 NIT.** D is the right seam, but the per-token half is unwired and untested, and its semantics in an embedder are not stated. C needs "judged" defined structurally and needs the namespaces kept apart. A's defaults undercut D's bound.

## SHOULD-FIX

**S1. D is not wired to the per-token criterion, and the test cannot tell.** `atonia_gate_approach_notes.md` §4.2::`complete_tokens(prompt_ids, *, grammar, max_tokens, max_prompt_tokens, deadline)` is the pinned three-call surface, and it has no stop input. §1A.8 says a stop "ends the turn at the next token", but only the `deadline` reaches `stopping_criteria`. `test_advisor_gate.py::_StopDuringCompletion` raises the stop inside `complete_tokens` and then returns a normal completion with `deadline_hit=False`. So an implementation that checks `current_stop()` once, after `complete_tokens` returns, flips the gate.
- Failure scenario: SIGTERM lands at token 10 of a 512-token completion. The turn samples all 502 remaining tokens, then reports `abandoned`. The embedder's ~15–20 s kill budget (bounded_stop §5) is spent on tokens nobody uses.
- Fold: give `complete_tokens` a `stop` input (a `should_stop()` callable or the `StopRequest`) that the criterion polls next to `deadline`, and pin that in §4.2.
- Test: the fake asserts it received the stop and sees it set, then returns `stopped=True` without finishing. Also add the status-from-flag rule for a stop, like the deadline's: the stop flag, not `finish_reason`, decides the status.

**S2. D's embedder semantics are unstated.** These hold for a process-global, sticky stop shared with other Maxim users in the same process.
- (a) **Precedence.** No precedence is needed. Either stop is enough to abandon the turn (process OR per-call), and a per-call stop can never un-set a process stop. Say so: "honours both" reads as if one might override the other.
- (b) **Sticky at entry.** After a process stop, what do later `ingest()` calls do, and does `open()` refuse? The sim and campaign paths refuse to start on `yield_requested()` (bounded_stop §5). The notes should match: `ingest()` returns `abandoned` with `completions == 0` and no prefill, and `open()` raises `AdvisorCompositionError`, because loading a GGUF after SIGTERM wastes the kill budget.
- (c) **Cross-coupling.** `MaximHandle.stop(deadline=)` and `_drain_and_stop_handle` call `request_stop` FIRST (bounded_stop §5). An embedder that stops a console handle permanently abandons every advisor session in that process. That is correct for SIGTERM, surprising for "stop one component", and has no reset (`reset_shutdown()` re-arms only the backend event). State it, and point embedders that need a scoped stop at `stop=`.
- Process-global is the right default only because AD5 Profile B gives the observer a process the supervisor owns. Say that is why.
- Tests: add a stop-already-set-at-entry case (`backend.completions == 0`) and a per-call `stop=` case. Today `stop=` is checked only by signature, so an implementation that accepts and ignores it flips the gate.

**S3. "Judged" must be structural, and the rejected-last-event case must be decided.** In §1A.8 Cursors (C), `cursor_out` is "the last event the turn judged". Two gaps:
- (a) Nothing says the model's output cannot pick that event. The model is adversary 1 (§0). If "judged" is ever read from the completion (for example "events I considered"), the model chooses the caller's resume point.
- (b) A rejected event (`inputs[].rejected`, report stays `ok`) is not judged. If it is last in the batch, `cursor_out` stops before it, and the caller re-sends it forever: a poison-event livelock. If `cursor_out` moves past it instead, it breaks advisor_mode.md's report-half rule ("the caller never advances past events nobody judged").
- Fold: define judged as "in input order, every event whose segments entered the prompt, decided by Maxim before completion". Then pick the rejected-event rule explicitly (the owner's call). The strict option, never advancing past a rejected event, needs the caller to drop it, and `inputs[].rejected` lets it.

**S4. The `event_id` fallback mixes two namespaces in `cursor_out`.** Item 12 (advisor_mode.md) makes `cursor` the caller's *feed* cursor. With the fallback, `cursor_out` is sometimes a feed cursor and sometimes an `event_id`, and the consumer cannot tell which.
- Failure scenario: in `test_cursor_out_follows_the_last_judged_event[ok]`, `cursor_in="feed:100"` and `cursor_out="e2"`. AvServ resumes its feed with `since="e2"`, and the feed either errors or rewinds to the start.
- Fold (either works):
  - when the last judged event carries no `cursor`, carry forward the most recent `cursor` among the judged events, else `cursor_in`, and add `covered.last_event_id` as a separate field;
  - or require `cursor` on every event once any event has one.

  This does not contradict item 12 otherwise: Maxim still keeps no cursor between sessions.
- Also state what `cursor_in` is on the second `ingest()` of a session. Is it `open(cursor=)` or the previous `cursor_out`? The latter is a cursor kept within the session. That is allowed by item 7, but it needs saying.

**S5. A's defaults defeat D's bound, and `limits` must read back the actual values.** `n_ctx` defaults to the GGUF's trained context (32k–128k on current models), and `max_prompt_tokens` defaults to `n_ctx` minus the completion budget.
- (a) **The prefill bound.** §1A.8 says a stop during prefill "takes effect when prefill returns, which the token cap bounds". That cap is in tokens, not seconds. On a Mac or CPU, a ~120k-token prefill runs for minutes, far past the 15 s console budget. The embedder SIGKILLs mid-turn, and the log has no close record.
- (b) **Memory at open().** KV allocation at the trained context can OOM at `open()`.
- (c) **The clamp.** §1A.3 says the shared `Llama(**kwargs)` builder applies an "n_ctx clamp". So the trained context is not what loads, and an echoed `limits.n_ctx` would differ from the real value. That is the n_ctx-drift class (CLAUDE.md, sim-n-ctx-config-drift).
- Fold:
  - `session.limits` reads back from the constructed `Llama` (`n_ctx()`), after the clamp;
  - `open()` refuses `max_prompt_tokens + completion budget > n_ctx`. Otherwise a prompt under the cap overflows the context inside `eval`, which surfaces as a backend error rather than `prompt_too_long`;
  - the docs say plainly that the stop's deadline is not honoured during prefill, so the embedder sizes `max_prompt_tokens` to its kill budget;
  - consider a conservative fixed default for `max_prompt_tokens` instead of one derived from `n_ctx`.

**S6. `degraded` has no producer, but §4.2 and §6 still describe one.** §1A.8 says "no `degraded` producer". But two places still produce it:
- §4.2's status table keeps the row "Item 10 marked an input degraded → `degraded`";
- §6 says a missing encoder/semantic weight "records a `degraded` reason, and the observer's report says `degraded`".

Both describe a producer. A builder following §4.2 builds one, and a red gate written from §4.2 would stay red forever. Strike or annotate both ("not in the observer's closure; no producer this release, §1A.8"). Otherwise the §1A.8 statement is honest. Keeping the status in the schema with a stated precedence is fine.

## NIT

- **N1. B puts contracted fields in `extra`.** In this codebase `extra` is the CC3 forward-compat escape hatch (persistence-config §3). Fields AvServ designs against belong in declared keys. Cheaper and stronger: a root `limits: {n_ctx, max_prompt_tokens, n_gpu_layers}` on every report, plus a nullable `prompt_tokens`. That also serves the weak-evidence rule: a record whose context size is only operator-attested cannot be sole evidence, and the report would stamp it. Then `prompt_too_long` needs no `extra`.
- **N2. `session.limits` as an attribute object is right**: it is read-only, typed, and the test uses `getattr`. Pin it as a frozen dataclass (`AdvisorLimits`). It is in-process only, so CC3 does not apply, but say so in its docstring. Its definition should name the read-back source (S5).
- **N3. Imports and layers fit.** `utils/cancellation.py` stays a stdlib leaf (bounded_stop §5: the backend event registers itself, so `utils` never imports `models`), and `utils/deadline.py` is a new leaf. `advisor` may import `utils`, so §1A.3 LAYER_RULES is unaffected and `FORBIDDEN_MODULES/PREFIXES` gain nothing. One dependency: both modules sit behind `utils/__init__.py`'s eager imports until V9 step 0 lands. List D's build after step 0 and bounded_stop PR 1 in §7's order.
- **N4. Two stale lines.** §4.2's bullet still reads `session.ingest(events) -> AdvisorReport`, so add `*, stop=None`. In D, state the stop that lands during dispatch: once dispatch starts, does the turn complete (bounded by `max_calls`), or is it abandoned with `proposals: []` while the log already holds the `propose_action` records (a `log_last_seq` mismatch)? The test asserts `proposals == []` only for a stop raised before dispatch.
- **N5. The test leaks its stop.** `test_ingest_honours_the_process_stop_request` cleans up only `cancellation._reset_for_tests`. But `request_stop` also sets the backend `request_shutdown()` event, which `utils` does not reset. Until bounded_stop's conftest autouse scrub exists, a flipped gate leaks a set shutdown event into later tests. Also reset `models.language.cancellation`, or rely on the scrub and say so. A further point: the per-event `cursor` is inbound text (§0 adversary 2). State that it never enters the prompt, and cap its type and length (str, ≤ N bytes) so a hostile feed cannot bloat the atomic report.

## §0 overclaim check

Nothing claims more than §0. Two edges:
- **The prefill wording.** "Bounded" is true in tokens only (S5).
- **The per-event cursor.** It is caller- or feed-supplied and echoed verbatim. Maxim does not vouch for it, and the notes should say so (N5), so nobody reads `cursor_out` as validated.
