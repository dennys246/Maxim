# #1481 approach note v2: delta adversarial pass 1

> Reviewer: a different reader from `adversarial.md`. Read-only. Material: `docs/plans/inbound_comms_1481_approach.md`
> v2 (with A's grant terms), checked against the code on `f177ab45`.

## Verdict

**NOT STOP.** No DO-NOT-BUILD finding. Four SHOULD-FIX findings are fail-open (N1, N2, N3, N6). Each is a sentence or
two of specification in the note, plus a test. The A1 fix (acknowledge on render) holds in principle. Its mechanism is
under-specified at exactly the seams where the existing code would let it ack an unrendered message, or never ack one.

## 1. Acknowledge on render

### N1: "ids emitted in the final, budget-fitted prompt" has no mechanism [SHOULD-FIX, fail-open]
- `PromptBudgeter.build_segmented` returns `(stable, dynamic, dropped_names)`.
  - A **truncatable** section that `_fit` shortens is NOT in `dropped`.
  - `_fit` then runs `drop_unbalanced_frames`, which deletes every frame from the first orphan opener to the end.
  - So "the section is not in `dropped`" does not mean "every frame in it rendered". Acking on that basis loses
    messages silently, which is the A1 class again.
- `PromptBuilder.build_prompt` returns `str`, and the worker holds one `PromptBuilder` (`LLMWorker._prompt_builder`).
  Ids recorded on `self` have the wrong scope: a later build overwrites them before the earlier call returns.
- **Fix:** say which mechanism is used. Prefer deriving the ids from the returned prompt text: scan it for
  complete frames (`content_safety._FRAMED_REGION`) and map nonce to id. This holds whatever truncation exists now
  or is added later. Otherwise add the section with `truncatable=False` and check `dropped`. Either way, store the ids
  on the `LLMRequest`, never on the builder.

### N2: "after the LLM call returns" includes the timeout and failure returns [SHOULD-FIX, fail-open]
`LLMWorker._process_request` gets a dict back from `_call_llm_with_timeout` even on timeout:
`{"_timeout": True, ...}`, which is truthy. Other returns after the call:
- a shutdown-after-call proposal;
- `None`, a non-dict or `{}`, all of which go to `_generate_llm_fallback`;
- an exception in the proposal build.

**Fix:** pin the ack to the single point where a valid-response `LLMProposal` is built. That is after the
`not response or not isinstance(response, dict)` check and after the stop check. Never ack on `_timeout`. An empty
dict is a failure: no ack. Name it in a test.

The rest checks out:
- **Retries.** `retry_with_timeout`, `requeue_request` and `requeue_last_request` re-run `_process_request` on the
  same `LLMRequest`. So a callable carried on the request acks correctly on the retry's success.
- **Streaming and the cloud/local routers.** Both go through `self._llm.generate_json`, non-streaming. No
  separate path.
- **Threads.** The callable runs on the worker thread under the inbox lock, so it is thread-safe.

### N3: A poison message is re-rendered forever [SHOULD-FIX, fail-open: availability]
"Held until a model saw it" has no attempt bound. Take a message whose content reliably breaks the reply, for example
"answer only in prose", or one that pushes a drifted server n_ctx over (the 500 class in CLAUDE.md):
- it is never acked;
- it renders in every later prompt, and D13 requeues it too;
- this lasts until 20 newer messages evict it.

So one SMS can degrade every operator turn for as long as the attacker sends nothing else. A small local model does
the same with no attacker at all.

**Fix:** a per-entry render-attempt counter, kept in the inbox and bumped when an entry is handed to a rendering
prompt. After K attempts (say 3) with no ack, evict the entry with a WARNING (channel, alias, attempts; no content).
Add an age bound as well, if wanted.

### N4: Stale snapshots render duplicates [SHOULD-FIX, fail-closed]
- The snapshot is taken at submit. The `large` lane has `max_workers=1` (`runtime/worker_pool.py`), so a job queued
  behind an in-flight job renders the messages that job has already acked.
- The AUT has no in-flight hold: `_planning_submit_in_flight` only holds when liveness is on.
- **Fix:** at build time, filter the snapshot against the inbox's current held set. Do this by passing a small
  inbox view (`held(ids)`, `ack(ids)`) instead of a bare callable.
- **Deliberation.** Cycles 2+ re-submit the same mutated `context`
  (`agent_loop.py::_run_deliberation_cycles`), so they re-render the snapshot. `_submit_kwargs` carries no
  callable. State whether this is intended: it is the same turn, so rendering is defensible.
- **Type the invariant.** Make it a `TypeError`/WARNING when `comms_messages` is non-empty and no callable rides
  along, so a forgotten pass is not a silent re-render loop.

### N5: Head-of-line blocking under the token cap [SHOULD-FIX, fail-closed: availability]
- The 2,000-char cap does not bound tokens. 2,000 chars of CJK or emoji can exceed 800 tokens.
- At n_ctx 4k the cap is about 400 tokens, which is less than 2,000 ASCII chars.
- With strict oldest-first, one oversize message blocks every message behind it until eviction.
- **Fix:** cut a single oversize message to fit the cap, with the marker.
- **Also specify:**
  - the section's priority (not NICE_TO_HAVE, which is dropped first on most tight prompts);
  - whether the rule line and headers count toward the cap.

## 3. `framed_section` wording and the cap, against the real API

### N7: "One rule line" is not achievable as stated [NIT, fail-closed]
- `PromptBudgeter.add` has no wording parameter. It calls `framed_section(content)` on any text containing
  `TOOL_OUTPUT`.
- `framed_section`'s idempotence test is the literal `TOOL_OUTPUT_RULE not in text`. So unless that test becomes
  wording-aware, the comms section gets the default rule prepended above its own.
- Even when fixed, the follow-up prompt always holds a tool frame with the default rule, and the tool-aware prompt
  often holds bio_enrichment frames. The default rule says text between any `TOOL_OUTPUT` markers is "data returned
  by a tool", which covers the comms frames too.
- **Fix:** say "one rule line per wording". The source label then lives in each frame's header.

### N8: `frame_tool_output` has no alias parameter [NIT]
It needs a keyword. `_TOOL_NAME_SAFE` already maps a space to `_` in the tool name.

## 4. D1: `build_comms_stack(agent=)`

### N10: D1 call sites [NIT]
- **Callers.** Only `cli.py` and `embodied_runtime/agentic_runtime.py` call it. No test or script does:
  `tests/unit/test_comms_inbound_1466.py` names it in a docstring only.
- **A `None` bus.** `agentic_runtime` passes `getattr(agent, "_bus", None)`, which can be `None`. D1 must raise on
  `agent._bus is None`, not build a bus-less gateway.
- **An option.** With `agent=` in hand, `build_comms_stack` could create and subscribe the inbox itself. That closes
  the startup-window residual (§6) instead of accepting it.

## 5. D2: the memory renderers (all in `agents/prompt_builder.py`)

### N12: Each named renderer is unframed [NIT, follow-up as the note plans]
- **The three memory sections** in `PromptBuilder._add_memory_sections`:
  - `relevant_memories` renders `content["transcript"][:80]` raw;
  - `concept_context` renders names and relationship targets raw;
  - `knowledge_context` renders `definition` raw.

  All three are **unframed but unfed**. The method is "Dormant since 2026-10-06" (#845), and all three live callers
  pass `build_memory()` (`InMemoryMemory`), which fills none of them.
- **`working_memory_thoughts`** (`PromptBuilder._add_working_memory_section`) is **live and unframed**. `_whole_frames`
  only drops partial frames. It is filled from THOUGHT entries in `agent_loop.py::_run_deliberation_cycles`. The
  enrichment text is R-framed, but the model's own reasoning, which can quote an SMS, is laundered into it. This is
  the #823 class shared by every tool output, not new.
- **The WorkingMemorySet.** `MemoryAgent._on_percept` writes the raw comms content into the shared WorkingMemorySet
  as a PERCEPT (`maxim_agent.py` shares `exec_agent.working_memory`). Only `ThoughtGate` scoring reads it; it is not
  rendered.
- **The raw number.** `MemoryAgent._comms_messages` stores the raw sender. Its only renderer,
  `ExecAgent._build_llm_context`, is Dormant: unframed, raw number, labelled "User".
  `percept.metadata["conversation_context"]` also carries the raw number ("with +1…"); no renderer found.

### N11: `context.comms_messages` schema collision [NIT]
The documented shape is `{direction, content, channel, sender, timestamp}`. Two existing users expect it:
`MemoryAgent.build_context`, which writes raw senders, and the dormant ExecAgent renderer above. The inbox entry
shape differs from it.

**Fix:** use a new field, or redocument this one and state overwrite semantics.

## 6. The gate

### N6: The cited harness cannot observe the A1 fix [SHOULD-FIX, fail-open: the gate]
In `test_talk_rejection_1413_1414.py::_run_llm_loop`:
- `_Bus.subscribe` is a no-op, so the inbox never receives the publish;
- `_Worker.submit_context` only records kwargs, so no prompt is built and nothing is acked.

So (b), the budget-pressure case and the "removing the acknowledgement turns it red" probe cannot be observed. The
easy path is to hand-compose, which is G1 again.

**Fix:** state that the gate uses:
- a real `AgentBus`;
- a real `LLMWorker` with a `WorkerPool` and a fake `_llm.generate_json` that captures the prompt (with
  `_pool is None`, `submit_context` returns `None`);
- `_Env.observe` delivering the CLI input at tick 3 or later.

Twilio is not installed locally: use `TwilioChannel.__new__`, as `test_comms_inbound_1466.py` does.

On the assertions:
- **(a) is feasible.** `has_meaningful_input` gates the submit. The harness patches `llm_submit_interval=0.0`, so
  "past the interval" holds trivially.
- **(e) is feasible for `caplog` only.** `sim_logger` and `log_agentic` sinks are not captured. Narrow the claim, or
  capture those sinks.

## 7. New in v2

- **N9 [NIT].** MessageSid de-dup: `comms/api.py` reads only `From`/`Body`, and `TwilioChannel.receive_sms` takes no
  sid. The diff must include both. (Today `external` comes first in the dict and can be overridden; the note's
  set-last fixes it.)
- **A's grant terms read as consistent with the code.** The submit block is one `try`/`finally` over
  `submit_context` → `_release_unsent_deferred_inputs`, about 29 lines.
