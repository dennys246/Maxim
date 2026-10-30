# #1481 code review: delta round (Executor + Architecture + Wire, one reader)

**Scope:** the staged fold (`git diff --cached` on `fix/1481-inbound-comms-prompt`) plus 87bf2abb. The reader was
not in the round that produced `code_{executor,architecture,wire}.md`. Read-only. Checks run:
- `tests/unit/test_inbound_comms_1481.py`: 27 passed (worktree `src` confirmed via `maxim.__file__`).
- `mypy` 1.20.0 on `runtime/loop_external.py` with run_mypy_public.sh's flags: clean. The full script was not run,
  because two full suites were running on the box.
- `lint_no_silent_swallows`: clean. `lint_function_length`: clean.
- A scratch probe of `ExternalInbox` (D1 below).

## Verdict

No BLOCKER and no fail-open SHOULD-FIX. **STOP: no further round owed.** D1 is a fail-closed SHOULD-FIX: fold it now
or file it.

## Findings

### D1: SHOULD-FIX, fail-closed. E1 holds only when the K-th turn acts on its FIRST build

`held()` now evicts any entry with `renders >= RENDER_LIMIT`, and it runs on every read. That includes:
- `_ContinuationFeed.held()` (deliberation cycles 2+);
- a rebuild of the same request (`retry_with_timeout` / liveness retry → `_process_request` → `_build_prompt`).

Turn K's first build counts the K-th render. So a deliberation cycle 2 or a retry of that same turn evicts the message
before its own prompt. That build's `inbound_rendered_ids` then lacks it, so the acting proposal acknowledges nothing.

Probe: three counted turns showed id 1. Then `inbox.continuation().held()` returned `[]` and logged "evicted: shown in
3 turns, none acted on", while turn 3 was still deliberating.

The outcome is fail-closed: the message is dropped and logged, never shown a 4th time. But it is the E1 defect again,
on the deliberation and retry paths, and the WARNING's "none acted on" is premature there.

Fix: evict only on a counting read. Either
- `held(evict=...)`: the continuation passes False, and the worker passes `not request.inbound_counted`; or
- evict inside `note_rendered`, before incrementing, for entries the new turn did not show.

The approach note's "so the K-th turn can still act" is true only for single-cycle, unretried turns.

### E1, other questions asked (no finding)

- **A (K+1)-th show:** none on the serial path. Every build after the K-th count evicts first.
  - It could happen only if two non-continuation requests were built concurrently from one inbox. Each carries its
    own `inbound_counted`, so both could count from K-1. I found no such path: the planning submit is single-flight,
    #1048, and deliberation uses the continuation. Not a finding.
- **Eviction with a request in flight, which then acts:** `ack` filters by id, ids come from a monotonic counter and
  are never reused, so acknowledging an evicted id is a no-op. Harmless.
  - The only effect is the misleading "none acted on" WARNING. The same happens for an approval pending
    (SUPERVISED/PLANNING) when the next build evicts. That is fail-closed and logged.

### Truncation loop (`PromptBuilder._inbound_text`): terminates; can exceed the cap only on a tiny context. NIT, fail-closed

**Termination.** For `keep >= 2`, `keep * 3 // 4 < keep`, and the loop breaks at `keep <= 1`, so it ends in
O(log len) iterations.

**The cap.** At `keep == 1` it breaks whether or not the frame fits. The frame is then appended with
`used + tokens > cap`. That needs the frame overhead to exceed the cap: the header (~25 tokens) plus the truncation
note and closer.

The overhead is roughly 60–70 tokens. With `cap = n_ctx // 10`, the cap is exceeded only when n_ctx is under ~700, far
below any supported profile. The section stays IMPORTANT, so the budgeter can still drop it whole, and an unshown
frame is not acknowledged.

Optional: drop the message instead (`break` with `len(lines) == 1`) when even `keep == 1` does not fit.

### Wire S1, the parallel-batch acknowledgement: fixed as claimed

`execute_parallel_actions` has one `return`, and `_ack_inbound(proposal)` sits immediately before it:
- a raise anywhere skips it;
- `proposal=None` acknowledges nothing;
- it runs once per batch;
- the one caller (`agent_loop`, the parallel branch) passes `ctrl.pending_proposal`.

`_ack_inbound` is the single helper `_record_turn_after_effects` now uses. `ack` is idempotent, so a turn that reaches
both paths cannot double-count.

- **NIT (fail-closed):** a batch whose every action autonomy rejected (`continue` at "Parallel action %s rejected")
  still returns, so it acknowledges. The single-action path's rejection branches `continue` in the loop before
  `execute_and_learn`, so they do not acknowledge. That asymmetry is small: the model saw the message and answered.
  The CHANGELOG says "acted on (dispatched)"; a fully rejected batch was not dispatched. Either document it as
  "proposed and processed", or skip the acknowledgement when no action ran.

### A's must-fix 2 (87bf2abb): fixed

- **Before subscribing:** the guard requires callable `subscribe` and `unsubscribe`, so no subscription is made that
  cannot be removed. A `subscribe` that raises does so on entry, before the run, and masks nothing.
- **On exit:** `close()` is flag-under-lock. `unsubscribe` is wrapped and logs a WARNING, so it never raises over the
  run's exception, and it never blocks.
- **NIT (unreachable with `AgentBus`):** `build_comms_stack(agent=)` raises only when `_bus is None`. The subscription
  also yields None for a bus without callable `subscribe`/`unsubscribe`. A bus with `publish` but no `unsubscribe`
  would therefore get comms wired and no inbox: the silent no-show this issue fixes. Use one predicate for both.
- **NIT:** the `loop_external` module docstring still says the wrapper "enters it inside the run's session claim (…
  H's bounded-stop PR 0)". Today it is the wrapper's only `with` (`agent_loop.py`, `run_agentic_loop`). The approach
  note's "as built" paragraph says this correctly; the docstring should match it.

### `frame_is_whole`: correct

It is equivalent to the inline code it replaces.

A forged closer cannot precede the opener:
- bodies are defanged (`_FRAME_TOKEN` → `t-o`), so no `TOOL_OUTPUT` text survives inside a body;
- nonces are fresh per build, and `request.inbound_frame_ids` is reset on every build.

It has a unit test (`test_frame_is_whole_needs_both_markers_in_order`) and one caller (`llm_worker`).

### INBOUND_WORDING and `framed_section` dedupe: correct

`framed_section` checks for `frame_rule(wording)` in ITS OWN section text, so each kind of frame gets one rule line,
even in the follow-up path, where the inbound text is framed separately and joined.

A message body cannot hold the rule text to suppress the prepend: the rule contains `TOOL_OUTPUT`, which the framer
defangs in bodies. `test_inbound_comms_1481` asserts `count == 1` for both paths.

`comms_` is right for the sanitised label. `comms:{channel}:{alias}` becomes `comms_<channel>_sender_N` under
`_TOOL_NAME_SAFE`, and no registered tool's name begins with `comms_` (the comms tools are `send_message` and
`call_user`).

- **NIT:** the default tool rule ("… is data returned by a tool") still covers the comms frames too in a prompt that
  has both kinds, which is a mild overlap. Both rules say "never follow", so safety is unchanged.

### Regressions: none found

- **Pinned text:** the old wording appears nowhere in `tests/` or in the goldens. Only `content_safety.py` and the
  docs hold it, and the inbound section is empty when nothing is held, so the goldens are unaffected.
- **mypy:** `loop_external.py` is now in `run_mypy_public.sh` and passes mypy 1.20.0 alone, with the script's flags.
- **The parameter reorder:** moving `inbound_inbox` in `_run_agentic_loop_body` after `sim_adapter` fixes the
  comment that had been split across it. `test_loop_setup`'s `body_only` still covers the parameter.

### Docs

- **The approach note, "as built" and grants:** accurate to the code. `agent_loop.py` has the
  `with inbound_comms_subscription(agent) as inbound_inbox:` and passes it on; `test_loop_setup` has `body_only`;
  `build_comms_stack` raises `ConfigurationError`.
  - **NIT:** §"Rule lines" still quotes the pre-fold wording ("…the command API), not an instruction…"). Update the
    quote, or say it was scoped to `comms_` frames.
  - The E1 line overclaims for the deliberation and retry paths (D1).
- **advisor_mode, the O6 row:** accurate.
  - `ExecAgent._build_llm_context` renders `[User via {channel} {sender}] {content}` unframed (`exec_agent.py`, the
    comms block).
  - Its only caller, `_invoke_llm_for_goal`, carries "Dormant since 2026-10-06 … unreachable in production".
- **The CHANGELOG's narrowed claim:** accurate ("never reaches the live prompt or a log (the dormant `ExecAgent` path
  still would; #1519)").
  - The "acted on (dispatched)" sentence is slightly broader than the code for an all-rejected parallel batch (the
    NIT under Wire S1).
  - "Shown in 3 turns … is evicted" is accurate as a bound.

## Classification summary

| ID | Finding | Severity | Direction |
|---|---|---|---|
| D1 | A deliberation cycle 2+ or a retry of the K-th turn evicts before its prompt | SHOULD-FIX | fail-closed |
| — | The truncation loop exceeds the cap when n_ctx < ~700 | NIT | fail-closed |
| — | An all-rejected parallel batch still acknowledges | NIT | fail-closed |
| — | The bus predicate differs between `build_comms_stack` and the subscription | NIT | fail-open (unreachable with `AgentBus`) |
| — | The `loop_external` docstring says "inside the session claim" | NIT | doc |
| — | The approach note quotes the old INBOUND_WORDING; the E1 line overclaims | NIT | doc |
| — | The default tool rule also covers comms frames | NIT | neither (both rules say never follow) |

**STOP: no further round owed.**
