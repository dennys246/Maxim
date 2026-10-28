# #1481 approach note v3: delta adversarial pass 2

> Reviewer: a third reader (not `adversarial.md`, not `delta1.md`). Read-only. Material:
> `docs/plans/inbound_comms_1481_approach.md` v3, checked against the code on `f177ab45`.

## Verdict

**NOT STOP.** No DO-NOT-BUILD. Two SHOULD-FIX findings are fail-open (P1, P2); each is a sentence of specification
plus a test case. The rest are fail-closed or NITs.

## 1. Delta1 findings: fixed as stated?

| delta1 | v3 | Status |
|---|---|---|
| N1 complete-frame scan, ids on the request | §2 "Which ids rendered" | Fixed in principle; see P2 (replace-per-build) and P4 (where the scan runs). |
| N2 ack only on a valid proposal | §2 "When to acknowledge" | Seam is right; "valid" is under-defined (P1). |
| N3 K=3 eviction | §2 | Fixed. Abuse checked in §2 below (P5). |
| N4 `held()` at build time | §2 | Fixed. Lock is adequate (P6). |
| N5 truncate, not skip | §4 Caps | Fixed. |
| N6 real bus + worker | §5 | Fixed as a requirement; feasibility gaps in P3. |
| N7 one rule line per kind | §4 Rule lines | Fixed (`framed_section(wording=)`, `PromptBudgeter.add(frame_wording=)`). |
| N8 alias in `tool=` header | §4 | Fixed. |
| N9 MessageSid dedup | §4 Dedup | Fixed (both `comms/api.py` and `receive_sms` are in the diff). |
| N10 D1 raises on a None bus | §6 | Fixed. |
| N11 not `context.comms_messages` | §2 | Fixed (feed on the request). |
| N12 #1519 | Inputs 7 | Fixed: #1519 is OPEN with the right title. |

## 2. New holes in v3

### P1: "valid proposal" acks deliberation cycle 1 and degenerate replies; the §2 Deliberation bullet is false [SHOULD-FIX, fail-open]
- **The seam is findable:** `src/maxim/agents/llm_worker.py::LLMWorker._process_request`, the success branch that builds
  `proposal = LLMProposal(...)` after the `not response or not isinstance(response, dict)` check and the
  shutdown-after-call check. It is the only return that carries a model's parsed reply.
- **But it fires on proposals the loop never acts on:**
  - **Deliberation cycle 1** returns `ready_to_act=False`. That is a "valid" proposal, so the message is acked. Cycles
    2+ are submitted through `agent_loop.py`'s `_submit_fn` / `_submit_kwargs` (in `_run_agentic_loop_body`), which
    calls `llm_worker.submit_context` directly, **outside** A's helper, so no feed rides on them. The v3 sentence
    "Cycles 2+ of one turn re-read `held()`" is therefore false twice: no feed reaches them, and the message is
    already acked. The proposal that is acted on (`ctrl.pending_proposal = _delib`) was built from a prompt that
    never showed the message, and nothing logs this.
  - **Degenerate replies.** `ModelRouter._generate_tool_response` turns non-JSON prose into a `respond` action
    (`reasoning="fallback_parse"` / `"salvaged_from_json"`), and fills `confidence`/`reasoning`/`mode_goal_achieved`
    into any dict, so `{}` arrives as a truthy dict. `generate_json`'s standard path adds `usage` the same way. So
    "an empty response or a parse failure acknowledges nothing" holds only for a `None` return. Action-less
    proposals (`agent_loop.py::_proposal_without_action_reason`) are treated by the loop as planning failures, yet
    they ack.
  - **Stale drop.** `agent_loop.py::_drop_stale_proposal` discards a completed proposal after the worker acked; the
    D13 requeue rebuilds the same request, and `held()` no longer has the message. Low reachability (its docstring
    says "close to unreachable"), same class.
- **Fix (pick one, state it):** ack only when `isinstance(proposal.action, dict) and proposal.ready_to_act`, and
  either carry the feed in `_submit_kwargs` (a body edit outside A's term-1 boundary, so A must agree), or state
  that cycles 2+ do not render and the message stays held for the next turn (fail-closed duplicate, bounded by K).
  Delete or correct the Deliberation bullet. Add a gate case with deliberation on (`bio_enrichment_pipeline` set and
  a cycle-1 `ready_to_act=False` reply).

### P2: ids on a reused `LLMRequest` must be REPLACED per build, never accumulated [SHOULD-FIX, fail-open]
`retry_with_timeout`, `requeue_request` and `requeue_last_request` re-run `_process_request` on the same request, and
`_add_planning_correction` / `_add_failed_tool_feedback` lengthen the next prompt. A comms frame that survived the
fit on attempt 1 can be cut on attempt 2. If `inbound_rendered_ids` is a union, or is set only when non-empty, a valid
attempt 2 acks a message its prompt never showed. **Fix:** say "assigned (replaced) on every build, including an
empty list", and test it with a correction that pushes the frame out on the retry.

### P3: the gate has no existing precedent and three concrete traps [SHOULD-FIX, fail-closed]
No test today drives `run_agentic_loop` with a real `LLMWorker`. The parts exist separately:
`tests/unit/test_llm_worker_pool.py::FakeLLM` + `_submit_test_context` (a real worker and its own `WorkerPool`
from `LLMWorker.start()`, with no loop), and `tests/unit/test_talk_rejection_1413_1414.py::_run_llm_loop` (the loop,
with a stub worker). Composing them is feasible, but the note must name:
- **Tools.** `_run_llm_loop` passes an empty `ToolRegistry()`. With no `available_tools`, `PromptBuilder.build_prompt`
  returns `ANSWER_ONLY|...`, a path that renders no comms by design. The gate would go red for the wrong reason.
  Register at least one tool.
- **Synchronisation.** The harness runs 200 Hz for a fixed `max_steps` on wall time, while the real worker answers on
  a pool thread. `_Env.observe` must block on the fake LLM's call or poll `worker.latest_attempt_state()`, or the
  loop ends before the proposal lands (a flake).
- **The manual `pending_user_input` pop** at `answered_at_tick` must go: the real proposal clears it.
- The fake `generate_json` bypasses the router, so the router's prefix strip and `\x1e` split (P4) are not exercised.
  This is acceptable, but state it.

### P4: run the scan once, on the exact string sent [NIT, fail-closed]
The follow-up path has no budgeter fit, and `build_prompt` appends `deferred` and `correction` after
`_build_followup_prompt` returns. "After the budgeter's fit" is ambiguous there. The single exact-text point is
`LLMWorker._process_request` immediately after `self._build_prompt(request)`: that string goes unchanged to
`_call_llm_with_timeout`. Downstream, the router only strips `TOOL_PROMPT|`, splits at `PROMPT_SEGMENT_DELIMITER`
into system/user, and appends an instruction, so complete frames survive, **provided the comms section sits wholly in
the dynamic segment** (state that). Nonces are fresh per build, so content cannot forge a complete frame with a
live nonce.

### P5: can K-eviction be weaponised? [NIT, fail-closed]
- Co-rendered messages share a fate: if the call fails, every rendered id counts a render. An attacker would need
  three calls in a row that return `None` or time out. The router salvages prose into a `respond`, the section is capped at 10% of
  context, and `max_tokens` bounds the output, so content-driven failure is hard. The cheaper lever is the
  owner-accepted `MAX_HELD = 20` drop-oldest, which K does not widen. Both log a WARNING.
- **Non-adversarial:** three timeouts on a slow local model evict every held message (each retry rebuilds and
  counts). It is logged, but say so in Residuals.

### P6: `held()` at build time vs the webhook threads [NIT]
A copy under the lock is enough: a message that arrives after `held()` is simply not in this prompt and not in its
ids, and `ack` of an id already evicted is a no-op. One gap: the K-eviction WARNING fires inside
`note_rendered` on the worker thread. Emit it **after** releasing the lock (as the handler does), or use an `RLock`,
since a log handler that publishes on the bus would re-enter the inbox handler.

### P7: Rule T reaches `LLMWorker._process_request` [SHOULD-FIX, fail-closed]
It is pinned at 259 lines in `src/maxim/utils/function_length_baseline.json`. Adding the ack there must shrink it by
at least the lines touched. Extract the success branch (the `next_actions` → `LLMProposal` → `sim_log` block) into a
helper that carries the ack. The note's Rule T line covers only the loop helper. `_run_deliberation_cycles` (235) is
pinned too, if P1's fix touches it. `lint_function_length.py` catches this, so it is fail-closed.

## STOP rule

P1 and P2 are fail-open SHOULD-FIX. **Not STOP.** After folding them (and P3/P7 as specification), a delta round
scoped to §2 and §5 is enough.
