# Atonia notes v6.6: wire-integrity review (commit e7a20742)

Scope: exactly `e7a20742`, the notes' §1A.8 additions A–D, the §4.2 pointer, the AvServ answer table and the five new
red gates in `tests/integration/test_advisor_gate.py` (§7b). Read-only; the lens is docs/CODE_REVIEW.md's wire
integrity: each contract's producers and consumers outside the diff.

## Producer → consumer map

| Contract | Producer(s) | Consumer(s) outside the diff | Agrees? |
|---|---|---|---|
| Process `StopRequest` (`request_stop` / `current_stop`, sticky) | bounded_stop.md §5/§6: CLI's one SIGINT/SIGTERM handler; `maxim serve` flag handlers; `MaximHandle.stop(deadline=)` and `_drain_and_stop_handle` (call `request_stop` FIRST); the embedder's (pulse/AvServ) own SIGTERM handler; rerouted `orchestrator` stall abort + `transcribe_audio` (§9) | the hub's `end_session`, the sim tail (`yield_requested()` refusal), LLMWorkers (registered events), campaign `stop_event`, and now `AdvisorSession.ingest` | Partially: stickiness and non-signal producers not addressed (S2) |
| `StopRequest` in-flight delivery | the stop request | `closed_local.complete_tokens(..., deadline)` stopping criterion | No path for the per-call `stop=` to reach the criterion (S3) |
| `utils/cancellation.py` | today: only the per-call ContextVar (`current_cancel_event` etc.); `request_stop`/`current_stop`/`_reset_for_tests` do not exist yet (PR 0 of bounded_stop) | gate D (`hasattr` check) | Ordering dependency unstated (S4) |
| `open(n_ctx=, n_gpu_layers=)` → closed backend | `maxim.advisor.open` | `closed_local.open_closed_local_backend(path, *, n_ctx, n_gpu_layers, seed)` → the ONE shared `Llama(**kwargs)` builder in `llama_backend.py` (§1A.3) | Names agree; `n_gpu_layers=-1` agrees with `_LlamaCppBackend` (`getattr(cfg, "n_gpu_layers", -1)`). n_ctx default does NOT (S1) |
| `max_prompt_tokens` / `prompt_too_long` | §4.2: the backend checks before `eval` (`complete_tokens(..., max_prompt_tokens)`) | gate B: the SESSION must refuse with `backend.completions == 0` and report counts | Two homes for one check (S5) |
| report `extra.prompt_tokens` / `extra.max_prompt_tokens` | the session | AvServ (batch sizing); AD4 freeze at 1.4.0; persistence-config.md §3 `extra` rules | Contract data in the escape hatch (S6) |
| per-event `cursor` / `covered.cursor_out` | AvServ (opaque feed cursor) | advisor_mode.md item 12 + §"The report half" ("last event *judged*"; error/abandoned keep `cursor_in`); the superseded draft `AdvisorEvent` (no `cursor` field; had `source`, `extra`) | Consistent with item 12; "judged" and multi-ingest undefined (S7) |
| `session.limits` | `open()` | AvServ | Fine once S1 defines the resolved values |
| AvServ answer: "`http_fetch` honours `HTTPS_PROXY`" | the notes' answer table | AvServ's Phase 1 sandbox design (1.3.2 harness) | **False at defaults** (B1) |

## Findings

### BLOCKER

**B1. `docs/plans/atonia_gate_approach_notes.md` (AvServ answer table, row "Phase 1 web research") tells AvServ that
"`http_fetch` honours `HTTPS_PROXY`"; at defaults it does not, on 1.3.2 or on HEAD.**
`tools/http_fetch.py` calls `_http.fetch_url(..., public_only=policy.block_private_ips)`;
`utils/internet_access.py::InternetAccessPolicy.block_private_ips` defaults to `True` (same at `v1.3.2`); and
`utils/http.py::fetch_url`'s docstring for `public_only` says "**Never uses a proxy**" (the strict-address endpoint,
`HTTPEndpoint.strict_address_check`, #824). Only operator-configured URLs (`address_policy`, #921) let an env proxy win.
Failure scenario: AvServ builds a Phase 1 sandbox whose only egress is an HTTPS proxy, trusting this row; every
model-chosen fetch dials directly and fails (or AvServ is pushed to set `block_private_ips: false`, dropping the SSRF
check, to make it work). The owner relays this table, so it must be corrected before the relay: "the model-chosen
`http_fetch` never uses a proxy (#824); a proxy-only sandbox needs the robots/paywall checks off-path or an
allow-listed direct route", or whatever the owner decides. Doc-only fix.

### SHOULD-FIX

**S1. `n_ctx` "defaults to the GGUF's trained context" contradicts the shared builder and the repo's own context
practice.** `models/language/llama_backend.py::_LlamaCppBackend` (the `Llama(**kwargs)` block §1A.3 says
`closed_local` shares) takes `int(cfg.n_ctx)` from the profile, then clamps by filename hint; nothing reads
`n_ctx_train`. `models/language/config.py` deliberately caps profiles below training (e.g. the "training-time ceiling
is 131072 (128K); the 32K cap" comment; P4c dynamic sizing). Failure scenario: AvServ omits `n_ctx` on a 128K-trained
GGUF on a CM4 or small GPU; `open()` allocates a 128K KV cache and OOMs or GGML_ASSERTs (exactly what the builder's
clamp comment guards against). Also the seam `_backend_factory(gguf_path, *, n_ctx, ...)` takes n_ctx as required, so
"defaults to trained" needs a stated mechanism (`n_ctx=0` → llama.cpp's trained value read back via `llm.n_ctx()`, or a
GGUF metadata read) and the hint clamp must not be bypassed by `0`. Pick a bounded default (e.g. min(trained, a
documented cap)) and say how `limits.n_ctx` is read back.

**S1b (same row). `max_prompt_tokens` "defaults to n_ctx minus the completion budget": the completion budget is
defined nowhere.** `open()` takes no `max_tokens`; §4.2's `complete_tokens(..., max_tokens, ...)` has no stated value.
And an explicit `max_prompt_tokens > n_ctx - max_tokens` (or `> n_ctx`) is not refused, so an over-cap prompt passes
the check and fails inside `eval` as a generic completion `error` instead of `prompt_too_long`. Name the budget (and
expose it in `limits`) and make `open()` refuse an inconsistent triple.

**S2. The process `StopRequest` is sticky and has non-signal producers; v6.6 does not say what `ingest()` does when it
is already set.** bounded_stop.md §5: "a sticky event ... set by `request_stop()`"; `reset_shutdown()` re-arms only
the backend event; nothing clears StopRequest outside `_reset_for_tests`. Producers include `MaximHandle.stop` and
`_drain_and_stop_handle`, not just SIGTERM. Failure scenarios: (a) an embedder that also hosts a `MaximHandle` stops
it for a routine restart; every later `ingest()` in every advisor session in that process returns `abandoned` forever,
with no reason distinguishing it from a deadline; (b) after a yield the embedder (or a test) opens a second session and
ingests: per D it abandons immediately. That may be intended (the process is ending), but the notes must say it, and
say whether `ingest()` refuses at ENTRY (before prefill, no backend call: a stated `reason`, e.g. `"stop_requested"`)
or runs prefill first. bounded_stop.md's own sim tail "refuses to start when `yield_requested()` is set"; the advisor
should match and document it. Also give the abandoned report a `reason` that separates stop from deadline (the §4.2
status table has only "Deadline hit | abandoned"; add the stop row). Conversely, the advisor only reads the stop, so it
does not interfere with the CLI/serve/handle producers, and an advisor `open()` is not a hub session, so bounded_stop
§6's "no session was ever opened" CLI gate would still raise `KeyboardInterrupt` in a CLI-hosted advisor: state
whether `open()` counts as a session for that gate.

**S3. The per-call `stop: StopRequest` has no type or wire.** bounded_stop.md defines `StopRequest` as the process
singleton set by `request_stop()`; it names no public constructor or per-instance setter. And the closed backend's
seam (`complete_tokens(prompt_ids, *, grammar, max_tokens, max_prompt_tokens, deadline)`, notes §4.2 and the test
module's SEAMS block) has no stop parameter, so "ends the turn at the next token" can only come from the backend
polling `current_stop()` itself, which the per-call stop can never reach. Failure scenario: AvServ passes `stop=` to
bound one ingest; it is honoured only after `complete_tokens` returns, i.e. after a full `turn_timeout_s`. Fix: either
add `stop=` (or a combined `should_stop` predicate) to `complete_tokens` and the pinned seam, or type `stop=` as a
`threading.Event` the criterion polls alongside `current_stop()`. Gate D checks only that the parameter exists; add a
case that sets only the per-call stop.

**S4. Release ordering with bounded_stop PR 0 is unstated.** `utils/cancellation.py` today has no
`request_stop`/`current_stop`/`_reset_for_tests`; they land in bounded_stop PR 0 (§3), and `utils/deadline.py` does not
exist. The atonia gate is "on its own clock" (AD7, MEMORY). If the gate ships first, D cannot be built and gate D stays
red, or the builder invents a local stop that PR 0 then has to reconcile. Say "D's build waits for PR 0 on main" (or
the fallback), and relay that dependency to AvServ with the answer.

**S5. `prompt_too_long` has two producers.** §4.2 places the check in the backend (`complete_tokens` takes
`max_prompt_tokens`, "checked before `eval`"); gate B injects a fake whose `complete_tokens` never refuses and
asserts `backend.completions == 0`, i.e. the SESSION must count `len(PromptIds.ids)` and refuse. Failure scenario:
the builder follows §4.2, the session never pre-checks, and gate B can never flip (or the builder adds a second check
and the two disagree on the count when the template adds ids). Pick one producer (the session, which has the ids and
needs the count for `extra`), and keep the backend check as a defence that raises a typed error the session maps to the
same reason.

**S6. Load-bearing values in the `extra` escape hatch.** `extra` is the forward-compat hatch (CC3 / persistence-config
§3: unknown data rides there; it is not where a documented contract lives). B makes `extra.prompt_tokens` and
`extra.max_prompt_tokens` a documented API AvServ sizes batches by, and AD4 freezes `advisor_report/1` at 1.4.0. At the
freeze they either stay as a contract inside `extra` forever or move to a named field (a breaking change for AvServ).
Promote them now to a declared, nullable field (e.g. `"tokens": {"prompt": n, "max_prompt": m}` set on every report,
or at least on `prompt_too_long`) in §1A.8's report shape and the test's key allowlist; keep `extra: {}` as the hatch.

**S7. `cursor_out`: "judged" and multi-ingest are undefined.** (a) Is a `rejected: "special_token"` event "judged"? If
the last event of a page is rejected and not judged, `cursor_out` stops before it and AvServ re-feeds the same poison
event on every page, forever; if it is judged, say so. (b) The event_id fallback puts an event_id into the opaque-cursor
slot; a caller that resumes with `open(cursor=cursor_out)` gets an event_id back as `cursor_in` and cannot tell which
it is. Acceptable if documented, but say it. (c) `cursor_in` for a SECOND `ingest()` in the same session: the open()
cursor, or the previous report's `cursor_out`? advisor_mode.md item 12 implies one cursor per session; §1A.8 allows
many ingests. (d) "last" means list order, not `occurred_at`; state it. Item 12 itself agrees with C.

### NIT

- **N1.** Gate D's `_StopDuringCompletion` returns a full valid completion with `deadline_hit=False`, so the gate
  pins a post-completion `current_stop()` check (the status comes from the stop, not a criterion flag). Fine, but say
  so in the notes' status rules, next to "the status comes from that flag".
- **N2.** Gate D constructs `Deadline(5.0)`; bounded_stop.md §5 does not pin `Deadline`'s constructor (only
  `remaining()`, `share`, `sub`, `none()`). The gate pins a seam the plan left open: list it in the SEAMS block.
- **N3.** §1A.8's event shape `{event_id, kind, occurred_at, payload, text, cursor?}` drops the superseded draft's
  `source` and `extra`; `CURSORED`/`EVENTS` omit `occurred_at`. Say whether `occurred_at` is required and whether
  unknown event keys are refused (closed) or ignored, since AvServ will send `source`.
- **N4.** `test_cursor_out_follows_the_last_judged_event[abandoned]` uses the `deadline` fake, so it pins the
  deadline path only; the stop path's cursor is pinned by gate D. Fine; noted for coverage accounting.

## Verdict

**Not ready to relay.** One BLOCKER (B1, a false claim to AvServ about `http_fetch` and proxies, doc-only) and seven
SHOULD-FIXes. The worst are S1 (an unbounded trained-context default that the shared builder's clamp exists to
prevent), S2/S3 (the sticky process stop and the unwired per-call stop) and S5 (two producers for `prompt_too_long`,
which the gate cannot flip against). The `n_gpu_layers=-1` name and default agree with `llama_backend.py`, and the
cursor rule agrees with advisor_mode.md item 12. All folds are paper or test-docstring changes; no `src/` change is
implied.
