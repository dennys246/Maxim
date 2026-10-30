# #1481 approach note v3.2: delta check 4

> Reviewer: a fifth reader (not `adversarial.md`, `delta1.md`, `delta2.md` or `delta3.md`). Read-only. Scope: the v3.2
> folds of delta3 Q1–Q5, checked against `src/maxim/agents/llm_worker.py::LLMWorker._process_request`,
> `agents/llm_types.py::LLMProposal`/`LLMRequest`, `models/language/router.py::_generate_tool_response` (salvage),
> `runtime/substrate_proposal.py::_attach_live_situation`.

## Verdict

**STOP: build.** No DO-NOT-BUILD, no fail-open SHOULD-FIX. Two wording SHOULD-FIXes (fail-closed) and three NITs.

## 1. Q1: every return of `_process_request`

| # | Branch | Action | Executed? | Model's reply to the rendered prompt? | v3.2 handling |
|---|---|---|---|---|---|
| 1 | empty prompt | returns `None` | no | no | n/a |
| 2 | `prompt.startswith('{"action":')`, `direct_answer` | `respond` built by `generate_simple_answer` | **yes** | **no** (no LLM call) | not enumerated (D4-2) |
| 3 | shutdown before the call | `None` | no | no | fine |
| 4 | `_timeout` | `respond` + `_timeout_retry` | yes | no | defaults, named (Q1) |
| 5 | shutdown after the call | `None` | no | no | fine |
| 6a | `not response` + `_generate_llm_fallback` | fallback `respond` | yes | no | defaults, named, gate case |
| 6b | `not response`, no fallback | `None`, `error` | no | no | fine |
| 7 | **success**: `response` is a dict | `response["action"]` | yes when dispatched | **yes** | carries ids + feed |
| 8 | outer `except` | `None`, `error` | no | no | fine |

Row 6a also covers the exception, cancellation and missing-dependency paths of `_call_llm_with_timeout`, since each
returns `None`. Row 7 is the only "successfully parsed model reply", and the rule holds against the code.

**Salvage lands in row 7 (D4-1, SHOULD-FIX wording, not fail-open).** `router._generate_tool_response` turns a
non-JSON reply into a dict. It does so as `salvaged_from_json` (the `"message"` regex) or `fallback_parse` (prose
wrapped in `respond`, up to 500 chars). Either way the dict reaches the success branch, carries the feed, and is
**executed and acknowledged**. That is ack-worthy: it is the model's own text, written after it saw the frames. But
§2's "acknowledge nothing" list says "router salvage or default output that is never executed". A reader can take
that to mean salvage never acknowledges. Reword it: "router salvage (`salvaged_from_json`, `fallback_parse`) is the
model's reply and acknowledges when executed. A salvaged or defaulted output that is never executed acknowledges
nothing." The planning-mode synthetic `_plan_text` is also row 7. It acknowledges only on the confirmed execute,
which is correct.

**Correction retries.** A planning correction (an unavailable tool) or a parse-failure resubmit re-enters
`_process_request`. P2's per-build replace means only the reply that is finally executed carries the ids for its own
prompt. Correct.

**Fail-open check.** Rows 2, 4 and 6a execute without a model reply, and none of them carries the fields. The ack
reads only `proposal.inbound_*`, so `original_request` (set on every row) cannot leak ids. No fail-open path found.

## 2. Q5: `LLMProposal` fields

Safe. Every in-repo constructor is keyword-only:
- the 7 returns in `llm_worker.py`;
- `llm_fallback.py`;
- `substrate_proposal.py` (the substrate builder);
- `agent_loop.py` (`multi_step_continuation`);
- a grep found no positional `LLMProposal(` call in `src/`, `tests/` or `scripts/`.

Appending two defaulted fields at the end of the class is non-breaking. `compare=False` drops them from the generated
`__eq__` and `__hash__`. The class is already effectively unhashable, because of its list fields, and nothing hashes
or `astuple`s it. `dataclasses.replace` in `_attach_live_situation` copies every `init=True` field, so it keeps both.
Use an immutable default for the ids (`()` or `frozenset()`) or a `default_factory`, never a bare mutable default.

## 3. Q2, Q3, Q4

- **Q2:** accurate. It is in Residuals as fail-closed, with the closing condition.
- **Q4:** the list wording is fixed. The second half of delta3 Q4, "name where the loop body gets `feed` for
  `feed.continuation()`", is **not folded**: the note still writes `feed.continuation()` with no source for `feed`.
  NIT.
- **Q3:** the Residual says only "bounded, idempotent". It drops delta3's one-line effect: renders made while an
  action waits for approval count toward K, so a slow approver can see the message evicted (WARNING) before the
  "yes". NIT.

## 4. Other findings

- **D4-2 [SHOULD-FIX wording, fail-closed]:** add row 2 (`direct_answer`) to the "acknowledge nothing" list. Its
  prebuilt JSON holds no frames, so the scan finds nothing and the result is benign either way. But it is a seventh
  executed, non-model return, and the enumeration should be complete.
- **D4-3 [NIT, fail-loud]:** delta3 Q5's first bullet (declare the new `LLMRequest` fields `compare=False`) is not in
  v3.2. `LLMRequest` is `order=True`. A comparing feed field would raise `TypeError` in the priority queue on a
  `sort_index` tie: loud, not silent. Every existing field already uses `compare=False`, so state it anyway.

## STOP rule

No DO-NOT-BUILD and no fail-open SHOULD-FIX. **STOP: build.** Fold D4-1, D4-2 and the NITs by inspection. They do not
need another round.
