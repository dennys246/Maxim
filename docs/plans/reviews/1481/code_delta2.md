# #1481 code delta 2: the D1 fold and the merge seam (Executor + Wire integrity)

Reviewer: a different reader, delta-scoped. Scope: commit 576c71a0's D1 fold (`loop_external.ExternalInbox.held`,
`_ContinuationFeed.held`, `llm_types.InboundFeed.held`, `prompt_builder.PromptBuilder._inbound_text`, the D1 tests) and
the merge seam in `agent_loop.run_agentic_loop` (`git diff 576c71a0 -- src/maxim/runtime/agent_loop.py
tests/unit/test_loop_setup.py`). Narrow tests run: `tests/unit/test_inbound_comms_1481.py` +
`tests/unit/test_loop_setup.py`, 85 passed.

## 1. D1: eviction only on a new turn's first build

How `inbound_counted` is set: `llm_worker._record_inbound_rendered` runs after every `PromptBuilder.build_prompt`
(via `LLMWorker._build_prompt`, the only caller). It sets `inbound_counted = True` and calls `note_rendered` once per
request, and only when this build showed at least one frame whole. `note_rendered` is now the only writer of
`renders`; `held(evict=True)` is the only place that evicts.

Every build path:

| Path | `held` call | Evicts? | Counts? |
|---|---|---|---|
| New turn, first build (tool-aware or follow-up branch; each calls `_inbound_text` exactly once) | `evict=not counted` = True | yes, entries already at `RENDER_LIMIT` | yes, if a frame survives whole |
| Retry of the same request (`retry_with_timeout`, `requeue_request`/`requeue_last_request` → `_resubmit` → `_process_request` on the SAME `LLMRequest` object) after a counted build | `evict=False` | no | no (`inbound_counted` already True) |
| Deliberation cycle 2+ (`inbound_inbox.continuation()`, `agent_loop` deliberation submit) | `_ContinuationFeed.held` → `evict=False` whatever is passed | no | no (`note_rendered` is a no-op) |
| Simple-answer short-circuit / empty prompt (return before `_inbound_text`) | not called | no | no |

(a) **Eviction on a non-first build.** Possible in one shape: the first build showed nothing whole (the budgeter
dropped the section), so `inbound_counted` stays False and a retry reads with `evict=True` again. That is harmless.
`renders` changes only through `note_rendered` from a counted request, and the first build already evicted every entry
at the limit, so the retry's sweep finds a new victim only if ANOTHER request counted in between. The planning lane
has one worker per `LLMWorker`, and the continuation never counts. Even then, the turn being retried never showed that
message whole, so no "turn still working on it" loses it. Not a finding.

(b) **A message that is never evicted.** An entry at `renders >= RENDER_LIMIT` waits for the next new turn's first
build, which the D1 tests pin (`test_a_turn_still_working_on_a_message_never_loses_it`,
`test_the_builder_evicts_only_on_a_new_turns_first_build`). An entry whose frame is never whole (the budgeter drops
the section every turn) never counts and is never evicted. It is still bounded by `MAX_HELD` (oldest dropped, with a
WARNING), and this was the same before D1, since counting was always "whole only". Not new.
- **NIT (N1), fail-closed:** a turn whose cycle 1 showed nothing whole but whose cycle 2+ did is never counted. The
  continuation does not count, and the first build left `inbound_counted` False. That turn does not consume one of the
  K renders, so the message lives one turn longer. The cost is only a longer life, inside the `MAX_HELD` bound.
  Already-covered class (the render count is a liveness bound, not a correctness gate).

(c) **Double counting.** None found. One count per `LLMRequest` (the flag lives on the request object, which a retry
reuses rather than copies), zero per continuation, and `ack` is idempotent across the two dispatch paths
(`tool_dispatch._ack_inbound` from `_record_turn_after_effects` and `execute_parallel_actions`).

Protocol: `InboundFeed.held(self, *, evict: bool = True)` matches both implementers. `ExternalInbox.held` and
`_ContinuationFeed.held` both accept and default `evict`; the continuation deliberately ignores it. No other `held(`
caller exists in `src/` or `scripts/`.

## 2. The merge seam: `run_agentic_loop`

```python
with (
    bio_session(memory_hub=..., hippocampus=..., owner=_LoopOwner(run_id), consolidation=consolidation) as session,
    inbound_comms_subscription(agent) as inbound_inbox,
):
    deferred = _run_agentic_loop_body(..., session=session, inbound_inbox=inbound_inbox)
if deferred is not None:
    raise deferred
```

- **Matches main except the two inbound lines.** I checked `git diff origin/main -- src/maxim/runtime/agent_loop.py`
  at the wrapper. The rest of that diff is #1481's own rule-T extractions.
- **Enter order:** the session is claimed first, then the subscription. If `bus.subscribe` raises, `bio_session` exits
  with that exception, so the session still ends. If `bio_session.__enter__` raises, the subscription is never
  entered. Correct.
- **Exit order:** the subscription closes first (inner), then the session ends. `inbound_comms_subscription`'s
  `finally` cannot raise: `inbox.close()` only sets a flag under the lock, and `bus.unsubscribe` is wrapped in a
  logged `except Exception`. A body exception therefore passes to `bio_session.__exit__` unchanged. The
  `@contextmanager` generator re-raises the in-flight exception and never suppresses it, which
  `test_the_runs_own_exception_passes_through_unchanged` pins. A D13 abort is returned, not raised, so it is raised
  only after both exits, as main intends.
  - **NIT (N2), observability only:** before the merge, `end_bio_session` ran inside the body, so inside the
    subscription. A message arriving during session-end consolidation (can be long under `"full"`) hit the closed
    inbox and logged "arrived after its loop ended; not shown". Now the subscription is already removed, and such a
    message reaches no handler with no line logged. The message was never going to be shown either way (the loop is
    over), and messages after the run were already unlogged. Nesting the subscription OUTSIDE the session would
    contradict the approach note ("inside the run's session claim"). Accept as is, or note it in the approach note.
- **Signature and callers:** `_run_agentic_loop_body` is keyword-only from `*` on, so the required `session:
  BioSession` after defaulted parameters is legal. `inbound_inbox: ExternalInbox | None = None` follows it, and the
  return type is `Exception | None`.
  - The only caller is the wrapper. `test_run_agentic_loop_is_a_wrapper_that_forwards_every_argument_to_the_body`
    asserts exactly one call and no caller outside `agent_loop.py`, with `body_only = {"session", "inbound_inbox"}`.
  - `inbound_comms_subscription` has one `src/` caller (the wrapper) and none in `scripts/`. `bootstrap.py` only
    names it in a docstring.
- **Parenthesized multi-item `with`:** it is official grammar from Python 3.10 (`requires-python = ">=3.10"`). Fine.
- **NIT (N3), doc:** the `loop_external` module docstring, as edited by 576c71a0, says the wrapper enters the
  subscription "around the body today, and inside the run's session claim once H's bounded-stop PR 0 lands". PR 0
  has landed and this merge nests it inside the claim, so that sentence is stale again. Say "inside the run's session
  claim (`bio_session`)".

## Out of the delta (for the record, not owed here)

`PromptBuilder.build_prompt` returns `""` before `_inbound_text` when there is no user question and
`use_tool_prompting` is False. The loop passes `use_tool_prompting and bool(available_tools)`, so an agent with an
empty tool roster never shows held comms. Earlier rounds did not flag this. The messages stay held, bounded by
`MAX_HELD`, so it fails closed.

## Classification summary

| ID | Finding | Severity | Direction |
|---|---|---|---|
| N1 | A turn that shows a message only in cycles 2+ is never counted (the message lives one turn longer) | NIT | fail-closed, already-covered class (liveness bound) |
| N2 | A message arriving during session-end consolidation is no longer logged as late | NIT | fail-closed (never shown either way); observability only |
| N3 | The `loop_external` docstring still says "once PR 0 lands" | NIT | doc |

No BLOCKER and no SHOULD-FIX. The D1 fold does what it claims on every rebuild path. The merge seam keeps main's
exception and exit semantics.

**STOP: no further round owed.**
