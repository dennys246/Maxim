# #1481 approach note v3.1: delta adversarial pass 3

> Reviewer: a fourth reader (not `adversarial.md`, `delta1.md` or `delta2.md`). Read-only. Scope: §2's P1–P7 folds,
> and any new fail-open hole from moving the acknowledgement to dispatch. Checked against the code on `f177ab45`.

## Verdict

**NOT STOP, narrowly.** No DO-NOT-BUILD. One SHOULD-FIX is fail-open (Q1). It needs one sentence of specification
and one gate case. Everything else is fail-closed or a NIT. The seam (P1) and replace-per-build (P2) are right.

## 1. The folds against the code

| Fold | Status |
|---|---|
| P1 seam | Right. `execute_and_learn` is reached by §4 (`agent_loop.py`, the `can_execute` branch), `loop_planning._drain_one` (passes `proposal.source`, the `LLMProposal`), and `LoopController.handle_confirmation` (passes `pc.source`). The plan-approval path re-enters §4 with the same object (`loop_controller.py`, `pending_proposal = pending_plan_proposal`). The respond action and the `next_actions` head take the same §4 path. Substrate and agent-fallback proposals carry no feed. |
| Proposal identity | Kept. `LLMProposal` is `@dataclass(frozen=True)`, so the new members must be constructor **fields**, not attributes set afterwards. The only rebuild between the worker and dispatch is `substrate_proposal._attach_live_situation`'s `dataclasses.replace` (the llm-primary embodied path), and that keeps fields. Deliberation returns the worker's object unchanged. `agent_loop.py`'s `multi_step_continuation` builds a fresh proposal with no feed, which is correct: the head already acknowledged. No deepcopy, pickle or `asdict` on the path. |
| Deliberation | Right. Cycles 2+ go through `_submit_fn` → `submit_context` (only two submit sites exist: the main one and the deliberation closure). The acting proposal is cycle N's own object. On a `break` or at max cycles, `_run_deliberation_cycles` returns cycle 1's **not-ready** proposal when it has an action, and that proposal is executed. It then acknowledges what cycle 1 showed. That is correct behaviour, but §2's list "deliberation's `ready_to_act=False` cycle acknowledges nothing" is wrong as worded (Q4). A no-op `note_rendered` on the continuation keeps the count at one per turn. |
| P2 | Feasible. Each retry (`retry_with_timeout`, `requeue_request`, `requeue_last_request`) and each correction re-enters `_process_request` → `_build_prompt`, so one assignment right after `_build_prompt` covers every build. |
| P6, P7 | Fine. |

## 2. Findings

### Q1: the timeout and llm_fallback proposals carry an EXECUTED action [SHOULD-FIX, fail-open]
`_process_request` has six other `LLMProposal(...)` returns besides the success branch. Every one sets
`original_request=request`. Two of them carry an action that §4 executes through `execute_and_learn`:
- the `_timeout` branch, a `respond` with `_timeout_retry`;
- the `not response` branch, `action=self._generate_llm_fallback(request)`.

Both run after the scan has already stored ids on the request. So the ids reach the ack in two cases:
- the implementer threads ids or the feed through every constructor;
- the ack helper reads `proposal.original_request.inbound_rendered_ids`, the natural shortcut, because `original_request` is on every branch.

In either case a message the model never answered is acknowledged.

§5 tests the timeout case, but not the `None`-response fallback.

**Fix:**
- State that only the success-branch proposal carries `inbound_feed` and `inbound_rendered_ids`, copied by value.
- State that the ack reads those two proposal fields, never `original_request`.
- Add a gate case: the fake `generate_json` returns `None`, the fallback `respond` executes, and nothing is acknowledged.

### Q2: the parallel batch executes a model turn without acknowledging [SHOULD-FIX, fail-closed]
§4's `parallel_actions` branch dispatches through `_execute_parallel`, not `execute_and_learn`. The
`execute_and_learn` docstring names it as a tracked exception. The message then re-renders in the follow-up prompt
and counts toward K, even though the model acted on it. A run of batched exploration can evict a message that was
acted on. Either call the same ack helper in that branch (the proposal is in hand there), or list it under "acknowledge
nothing" and in Residuals.

### Q3: late acknowledgements are not fail-open [NIT]
A PLANNING proposal expires from `ProposalQueue` after 300 s (`expiry_seconds`). A `PendingConfirmation` lives only
in the run. Each one acknowledges only ids its own prompt showed whole, and `ack` is idempotent, a no-op once a message
is evicted or already acknowledged. One effect is worth a line in Residuals: renders made while an action waits for
approval count toward K, so a slow approver can see the message evicted (WARNING) before the "yes".

### Q4: deliberation wording [NIT]
Replace "deliberation's `ready_to_act=False` cycle" with "a cycle's proposal that is not executed". The rule is
"executed ⇒ acknowledge", whatever `ready_to_act` says. Also name where the loop body gets `feed` for
`feed.continuation()`, since the main feed is created inside A's helper.

### Q5: field hygiene [NIT]
- On `LLMRequest`, which is `order=True`, declare the new fields `compare=False`, as every other field is.
- On `LLMProposal`, use `compare=False` too, so `==` never touches the lock-holding feed.
- Place the ack after `executor.execute` returns or raises, not before the dormant pre-execution snapshot.

## STOP rule
Q1 is a fail-open SHOULD-FIX. **Not STOP.** It is closed by the two sentences and one gate case above. A re-check
limited to Q1 is enough, or a reviewer can accept the fold by inspection.
