# v6.7 delta review: Architecture lens (`git diff e7a20742 221ad88c`)

Scope: notes §1A.8 (A–D, `degraded`), §4.2's closed-backend bullet, the §9 AvServ table and the v6.6-round fold table, and
`test_advisor_gate.py` §7b. Read against bounded_stop.md §2/§5/§6, advisor_mode.md items 9 and 12, the v6.6 architecture
findings, and at HEAD `src/maxim/tools/http_fetch.py`, `src/maxim/utils/http.py` and `src/maxim/models/language/llama_backend.py`.

**Verdict: no BLOCKER. 4 SHOULD-FIX, all fail-closed. 6 NIT.** The session-scoped stop is the right fold and is coherent
with bounded_stop.md. The #824 correction is accurate. The open items are spec gaps a builder would otherwise guess at.

**STOP: no further round owed.**

## What holds

- **D fits bounded_stop.md's principle.** The observer has no CLI entry, so the embedder owns the handler, and the library
  installs none and never exits. Not consulting the sticky process `StopRequest` removes v6.6 S2(c), the cross-coupling
  where `MaximHandle.stop` abandoned unrelated sessions. `deadline_s` is a float, not `utils/deadline.py::Deadline`, so
  "does not wait for PR 0" is true. v6.6 S1 is closed: `complete_tokens(..., should_stop)` is pinned in §4.2, and
  `test_session_stop_abandons_at_the_next_token` asserts `tokens_sampled <= 4`, so a check made only after completion
  stays red. v6.6 S2's per-call and entry cases are covered: `test_per_call_stop_abandons_only_that_call` and the
  `AdvisorStopped` assertion.
- **D is stated without contradiction** in §1A.8, §4.2 (the `complete_tokens` signature) and the §9 table.
- **The #824 correction is accurate at HEAD.**
  - `http_fetch.py` passes `public_only=policy.block_private_ips` to both `fetch_url` and `check_robots_txt`.
  - `InternetPolicy.block_private_ips` defaults to `True` (`utils/internet_access.py`).
  - `fetch_url(public_only=True)` selects `_EXTERNAL_PUBLIC_ENDPOINT`, which is registered with
    `strict_address_check=True`. Its client therefore has `mounts=None` and `trust_env=False`, so there is no env or
    system proxy.
  - With `block_private_ips=False`, the `"any"` endpoint honours a proxy, loudly. "At defaults" is the right qualifier.
- **C closes v6.6 S3 and S4:** "Maxim decides" is structural, there is no mixed namespace, and cursors are required when
  used. **B closes N1**, because the counts are now a named field. **A closes S5(b)/(c) and the refusal rule.**

## SHOULD-FIX (all fail-closed)

**S1. "Callable from a signal handler" needs its contract pinned.** CPython runs the handler on the main thread, between
bytecodes. If the embedder calls `ingest()` on the main thread, `request_stop` runs *inside* `ingest()`, at an
arbitrary point.
- If it raises (for example on a bad `deadline_s` or `reason`), the exception is injected into `ingest()`. That could
  happen between a log line's `write` and its `fsync`.
- If it takes a lock that the interrupted frame holds, it self-deadlocks. `threading.Event.set()` takes a
  non-reentrant `Condition` lock, and so does `logging`.
- Fold: state that `request_stop` only stores the reason and a monotonic deadline and then sets a plain bool, in that
  order. It takes no lock, does no logging and no I/O, and **never raises**: it coerces or ignores bad arguments. The
  stop's log record is written later by the `ingest`/`close` thread.
- Also state that during prefill a main-thread handler itself does not run until `eval` returns. That is consistent with
  "takes effect when prefill returns", but it means a stop delivered by signal cannot even be *recorded* until then.
- Test: call `request_stop` with garbage arguments and expect no raise.

**S2. The default `n_ctx` clamp is new behaviour, but the notes cite it as existing.** §1A.8 says the clamp resolves
`min(trained context, 8192)` and attributes it to "the shared builder's clamp (§1A.3)". At HEAD,
`llama_backend.py::_LlamaCppBackend` clamps only *downward* by filename hint (`"2k"`/`"4k"`/`"8k"`/`"16k"`), and runs
`int(self.cfg.n_ctx)`, which raises on `None`.
- Fold: state that the `None` branch is new, belongs to `closed_local`'s call into the builder, and leaves
  `_LlamaCppBackend`'s explicit `n_ctx` unchanged. Say how the trained context is read before construction (GGUF
  metadata, or a `vocab_only` probe), what an explicit `n_ctx` above the trained context does, and that
  `limits.n_ctx` comes from `Llama.n_ctx()` after construction.
- Gap: the red gate cannot see a regression to the trained context. `FakeBackend` hard-codes 4096, so an implementation
  that passes `n_ctx=0` (llama-cpp then uses the full trained context) flips `test_open_exposes_limits`. Add a builder
  unit test with a stub `Llama` that records the `n_ctx` kwarg.

**S3. The report schema block was not updated.** §1A.8's JSON lacks the new `"prompt": {"tokens", "max_tokens"}` field.
It also still shows the `covered` cursors as strings, though C now makes them `null` when unused. The schema is what the
owner relays to AvServ (the hand-off in §1A.8). Fold:
- add `prompt` to the block;
- mark `cursor_in`/`cursor_out` nullable;
- state when `prompt.tokens` is `null` (for example when `build_prompt` itself fails).

**S4. Two cursor cases are still undefined.**
- (a) **Bootstrap.** `open(cursor=None)` with events that carry cursors. AvServ's first run has no prior cursor, but its
  feed does. "With no `cursor` given, both cursor fields are `null`" reads as `cursor_out = null`, so AvServ could never
  obtain a first cursor. Decide: cursor mode follows the events (all or none per batch; `cursor_in` null, `cursor_out` =
  the last event's), or `open()` requires a cursor whenever the events have them.
- (b) **The second `ingest()`.** Is `cursor_in` the `open()` cursor or the previous `cursor_out`? Carried over from
  v6.6 S4. If it stays the `open()` cursor, an `error` on the second batch rewinds the caller to the start of the
  session, and the events are judged twice. That is duplicate advice, not an unsafe outcome. The other answer is a
  cursor kept within the session, which item 7 allows; say which.

## NIT

- **N1. §6 still contradicts §1A.8, and §4.2's table is stale.**
  - §6 still says "the **observer's** report says `degraded`". The v6.7 sentence in §1A.8 says §6 applies to
    non-observer runs, but §6's own text says the opposite. Edit §6 itself.
  - §4.2's status table still lacks the rows "stop requested → `abandoned`" and "over the cap → `error`
    `prompt_too_long`".
  - §4.2's `session.ingest(events)` bullet still lacks `*, stop=None` (v6.6 N4).
  - §4.2's "two-part bound" bullet should say that "`max_prompt_tokens` is checked before `eval`" happens in the
    session.
- **N2. The status precedence sentence names only the deadline flag.** Extend its exception to the stop flag: a stop
  sets `abandoned`, and truncated output is never parsed.
- **N3. `deadline_s` has no stated use.** In-turn stops are immediate (the next token), so say what it bounds:
  dispatch, the report write, or `close()`. Otherwise drop it from the signature.
- **N4. Two stop states are unstated.**
  - A stop that lands during dispatch, after completion (v6.6 N4, still open). Is the turn completed, or abandoned
    with `proposals: []` while the log already holds `propose_action` records?
  - A per-call `stop=` that is already set at entry. Does that call return `abandoned` with no completions, or raise?
- **N5. advisor_mode.md still says the old rule.** Its item 12 and report half (lines ~321, 554 and 602) still say
  "last event *judged*" and "the caller never advances past events nobody judged". C now advances past rejected events
  by decision. Annotate them as superseded, and add a pointer from bounded_stop.md §6's "amends advisor item 9" to the
  observer's session-scoped stop.
- **N6. The transparent-proxy wording should be narrower.** The proxy must redirect at the IP level (TPROXY/REDIRECT),
  not by DNS: a resolver that answers with the proxy's private address is refused by `public_only`. Also, with
  `trust_env=False`, a TLS-intercepting proxy's CA cannot come from `SSL_CERT_FILE`. Also carried from v6.6 N5: cap
  the `cursor` type and length, and state that it never enters the prompt.

## Fail-open check

None found.
- Advancing `cursor_out` past rejected events is an owner decision, and `inputs[].rejected` lists those events.
- Every stop path ends in `abandoned` or `AdvisorStopped`, never `ok`.
- The proxy correction narrows the deployment. It does not widen egress.
