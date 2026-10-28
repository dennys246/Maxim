# Atonia notes v6.7: wire-integrity delta review (`e7a20742..221ad88c`)

Scope: exactly `git diff e7a20742 221ad88c`: notes §1A.8 (limits A, prompt cap B, cursors C, session-scoped stop D),
the §4.2 `complete_tokens` line, the AvServ answer table's corrections, the v6.6 fold table, and the §7b red gates in
`tests/integration/test_advisor_gate.py`. Read-only. Lens: docs/CODE_REVIEW.md wire integrity, each touched contract's
producers and consumers outside the diff, checked at HEAD and against llama-cpp-python 0.3.8 as installed.

## Producer → consumer map

| Contract | Producer | Consumers outside the diff | Agrees? |
|---|---|---|---|
| Session-scoped stop (`session.request_stop`, per-call `stop=`) | the embedder's handler (AvServ) | none in maxim. bounded_stop.md's `request_stop` producers (CLI handler §6, `MaximHandle.stop(deadline=)`, `_drain_and_stop_handle`, §9 reroutes) target the process `StopRequest`, which the advisor now never reads; bounded_stop line 309's backlog grep ("only cli.py/server call `request_stop`") is unaffected because the advisor never calls it; no Maxim-owned process hosts an advisor `ingest()` in this release (`doctor --advisor` is a preflight, not an ingest) | Yes. The v6.6 conflict (sticky stop, `MaximHandle.stop` abandoning unrelated sessions, the PR 0 ordering) is gone |
| `should_stop` → llama-cpp `stopping_criteria` | closed backend | llama-cpp-python 0.3.8 `StoppingCriteria = Callable[[NDArray[intc], NDArray[single]], bool]`, called once per sampled token inside `generate` (after `eval`) and once more at the end of `_create_completion`; a criterion stop yields `finish_reason "stop"` | Compatible if wrapped (`lambda ids, logits: should_stop() or deadline_passed()`); the zero-arg predicate cannot be passed directly. Status attribution: S2 |
| Shared `Llama(**kwargs)` builder clamp | `models/language/llama_backend.py::_LlamaCppBackend._ensure` | the existing local-llama path (profiles, PromptBudgeter belief) and `closed_local` | **No clamp of the described kind exists at HEAD** (S1) |
| `prompt_too_long` producer | the session (§1A.8 B) | §4.2's backend bullet; the test module's SEAMS docstring | Stale text left in two places (S3) |
| Named `prompt` field | every report | AvServ batch sizing; `advisor_report/1` freeze (AD4) | Yes: v66 S6 folded |
| Per-event `cursor` / `cursor_out` | AvServ | advisor_mode.md item 12 + "The report half" ("last event *judged*"; error/abandoned keep `cursor_in`) | Compatible once "judged" = "judged or listed as rejected"; advisor_mode's wording is not pointed at it (N1) |
| AvServ answer: `http_fetch` and proxies | the notes' answer table | `tools/http_fetch.py` (`fetch_url(..., public_only=policy.block_private_ips)`, robots too), `utils/internet_access.py::InternetAccessPolicy.block_private_ips = True`, `utils/http.py::fetch_url` docstring "Never uses a proxy" for `public_only` (#824) | **Correct now** (v66 B1 closed). Residual constraints: N2 |

## Findings

### BLOCKER

None.

### SHOULD-FIX

**S1 (fail-closed). The "shared builder's clamp to `min(trained context, 8192)`" does not exist at HEAD, and the notes
do not scope the new default away from the existing consumer.** `llama_backend.py`'s builder block does
`n_ctx = int(self.cfg.n_ctx)` (so `None` is a `TypeError`), then clamps only by filename hint
(`{"2k": 2048, "4k": 4096, "8k": 8192, "16k": 16384}`); nothing reads `n_ctx_train`, and nothing caps at 8192. 8192 is
consistent with that table (the hints can only lower it). Two things the notes must say:
- **Mechanism.** `min(trained, 8192)` needs the trained context *before* `Llama(...)` is constructed: in 0.3.8,
  `n_ctx=0` means `n_ctx_train()` (unbounded, the very failure the default avoids), so the builder must read it first (a
  `vocab_only=True` load or the GGUF metadata), then construct. `session.limits.n_ctx` reads back `llm.n_ctx()`.
- **Scope.** The 8192 default applies only when the caller passes `n_ctx=None`. If an implementer makes it the shared
  builder's universal clamp, every `_LlamaCppBackend` profile above 8K (the 32K profiles) is silently cut while
  PromptBudgeter still believes the profile value: the n_ctx-drift class in CLAUDE.md
  (docs/lessons/sim-n-ctx-config-drift.md). State "explicit n_ctx and profile n_ctx pass through the existing hint clamp
  unchanged". For the advisor this fails closed (a smaller context yields `prompt_too_long`); the hazard is to the other
  consumer.

**S2 (fail-closed). A per-token stop is reported as `abandoned` only by convention of the fake.** §4.2 says "a deadline
stop sets a flag on the criterion, and the status comes from that flag" (llama-cpp reports any criterion stop as
`finish_reason "stop"`). The new `FakeBackend.complete_tokens` returns `deadline_hit=True` when `should_stop()` fires.
The notes nowhere say the real criterion sets that flag for a `should_stop` stop. A builder who sets it only on the
clock gets a truncated, grammar-incomplete completion parsed as `error`/schema-invalid instead of `abandoned`, while
gate D still flips against the fake. Say that the criterion's flag records *which* bound fired (deadline or stop) and
that the session checks its own stop flag before parsing. Also add the stop row to §4.2's status table and name the two
`abandoned` reasons (e.g. `deadline` / `stopped`), the v66 S2 residue. Cursor and proposals are safe either way, which
is why this is fail-closed.

**S3 (fail-closed). The v66 S5 fold is incomplete: `prompt_too_long` still has a backend producer in the text.**
§4.2's "The bound is two-part" bullet, right after "The prompt cap is checked by the session (§1A.8), not here", still
says "`max_prompt_tokens` is checked before `eval`, and an over-long prompt returns `status: "error"` with
`reason: "prompt_too_long"`". And the test module's SEAMS docstring still pins
`complete_tokens(prompt_ids, *, grammar, max_tokens, max_prompt_tokens, deadline)`, with no `should_stop`. Reword the
bullet ("the session's cap, checked before `complete_tokens`, bounds `eval`") and update the SEAMS line to the §4.2
signature. A builder following the stale text gets a `TypeError` or a redundant check, which is loud.

### NIT

- **N1.** advisor_mode.md item 12 and "The report half" still say "`cursor_out` is the last event *judged*", and item
  12 says "a yield prints `cursor_out`" (there is no process yield in the advisor now). Add a one-line pointer to
  notes §1A.8 C/D ("judged" = judged or listed as rejected; the stop is session-scoped).
- **N2.** "The egress proxy must be transparent" is correct but incomplete for `public_only`: the address dialled is
  checked as globally routable, so the sandbox's DNS must return public addresses (a resolver answering with a private
  proxy IP is refused), and HTTPS must be passed through by SNI or intercepted with a CA httpx's verify store trusts.
  Say so in the answer, so AvServ does not build a redirect that every fetch fails.
- **N3.** `session.request_stop` "callable from a signal handler": the handler runs on the main thread between
  bytecodes, possibly *inside* `ingest()` on that same thread (llama-cpp's `generate` returns to Python per token). It
  must only set an Event and take no lock that `ingest()` holds and write nothing; say so.
- **N4.** A per-call `stop=` already set at entry: does the call raise, return `abandoned` with no completion, or run
  prefill first? Unspecified; the session-stop case is specified (`AdvisorStopped`).
- **N5.** An explicit `n_ctx` above the trained context is passed through (the fake honours it); llama.cpp then
  runs past training with degraded quality. Either refuse it in `open()` or document it. And when `open()` refuses
  (`AdvisorCompositionError`) after the eager load, it must free the `Llama`.
- **N6.** `test_per_call_stop_abandons_only_that_call` and `test_session_stop_abandons_at_the_next_token` claim in
  their docstrings that a process-wide stop changes nothing / is never consulted, but neither sets one (it cannot until
  bounded_stop PR 0). Either drop the claim or add the case gated on `hasattr(cancellation, "request_stop")`.

## Verdict

No BLOCKER; three SHOULD-FIXes, all fail-closed (S1's hazard is to the existing `_LlamaCppBackend` consumer if the
default is mis-scoped, not to the gate); six NITs. The v6.6 wire findings B1, S2 (process stop), S3, S4, S6 and S7 are
closed by the folds; S1 and S5 are partially closed (above).

**STOP: no further round owed.** Fold S1–S3 as paper edits (S3 includes the test's SEAMS docstring).
