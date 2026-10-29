# #1481 pre-merge code review: Wire integrity lens

Commit reviewed: `fc69e312`, plus the uncommitted working-tree folds present at review time (`cli.py::_build_cli_comms`,
the guarded `inbound_comms_subscription` exit, a `tool_dispatch` comment, the baseline line, new tests). Read-only;
narrow pytest on the 15 affected test files: 439 passed, 35 xfailed. `lint_no_silent_swallows` and
`lint_orphan_modules`: clean.

## Verdict

No BLOCKER. One SHOULD-FIX on a consumer outside the diff (the parallel-batch dispatch never acknowledges), one
SHOULD-FIX on a stale register row, and NITs.

## Contract map

### 1. `build_comms_stack(bus=)` -> `agent=`
| Caller | Holds? |
|---|---|
| `cli.py` (now `_build_cli_comms`) with `agentic_agent` (`MaximAgent`, always `_bus = AgentBus()`), the same object passed to `run_agentic_loop` | Yes |
| `embodied_runtime/agentic_runtime.py::...` with `agent`, the same object `_worker` passes to `run_agentic_loop`; a raise is caught by its `except Exception` and warned | Yes |
| tests: only `test_inbound_comms_1481.py` (new); `test_comms_inbound_1466.py` mentions it in a docstring | Yes |
| docs/cli-reference, user docs, examples, notebooks: no caller | n/a |
| `docs/plans/reviews/atonia_gate/redgates_delta2_wire.md` and the 2026-09-27 mypy evidence cite `bus=` | Historical records; leave |

### 2. `run_agentic_loop` now inside `inbound_comms_subscription(agent)`
Every production caller passes a `MaximAgent` (real `AgentBus` with `subscribe`/`unsubscribe`): `cli.py`,
`api.py`, `console/handle.py`, `embodied_runtime/agentic_runtime.py`, `simulation/orchestrator.py` (AUT
`aut_agent` and narrator `orch_agent`), `simulation/tools.py` (`sub_agent`), `simulation/minecraft_harness.py`,
`simulation/interactive.py` (agent forwarded from its caller). `scripts/`: no direct caller.
Tests: the only stub bus on a loop path, `test_talk_rejection_1413_1414.py::_Bus`, gained `unsubscribe`; no other
`tests/` class defines `subscribe` without `unsubscribe` and reaches the loop (`test_api_core.py`'s `subscribe_many` is
a function). MagicMock agents are fine. At `fc69e312` the exit's `bus.unsubscribe` was unguarded; the working-tree
fold checks both methods before subscribing and logs (not raises) an exit failure. Holds once the fold is committed.

### 3. `submit_context(inbound_feed=)`, `LLMRequest`/`LLMProposal` fields
- `submit_context` callers: only `loop_external.submit_tick_context` and the deliberation `_submit_fn`. Test fakes
  (`test_loop_gates_characterization`, `test_loop_perception_characterization`, `test_talk_rejection_1413_1414`)
  take `**kw`. Holds.
- Serialization: no `asdict`/`deepcopy`/`pickle`/`json.dumps`/`vars()` of an `LLMRequest` or `LLMProposal` anywhere
  in `src/` (sim_log logs only the tool name; `state.data` gets `plan_text`, not the proposal). `WorkerPool` is
  in-process; `requeue_request` reuses the same object (so `inbound_counted` correctly survives a retry). Holds.
- `dataclasses.replace` (`substrate_proposal._attach_live_situation`): copies `inbound_feed` and ids by reference. Safe.
- PLANNING (`Proposal(source=ctrl.pending_proposal)` -> `drain_approved` -> `proposal.source`) and SUPERVISED
  (`handle_confirmation` -> `pc.source`) both hand the original `LLMProposal` to `execute_and_learn`, so they acknowledge.

### 4. `framed_section(wording=)` / `PromptBudgeter.add(frame_wording=)` / `TOOL_OUTPUT_RULE`
`TOOL_OUTPUT_RULE` is byte-identical to the old literal (checked by import). Consumers:
`prompt_builder._...` (follow-up rule prefix), `test_recall_frame_boundary_1444`, `test_tool_output_framing`: hold.
`framed_section` callers: the budgeter (default path unchanged) and the follow-up inbound branch. The rule line's
`id=...` placeholder does not match the frame-marker regex (inbound rule shares the format), so
`router._neutralize_stray_frame_markers` and `outside_tool_output` (router planning-mode detection, `memory/text.py`,
`bio_enrichment`) behave as before; inbound text sits inside frames, so it cannot flip PLANNING detection. Holds.

### 5. `_release_unsent_deferred_inputs` moved to `loop_external`
Only importer left: `test_narrator_reliability.py`, updated to `loop_external` (and its AST check of the `finally`).
`docs/plans/roadmap_1_3_x.md` names it without a path. Holds.

### 6. Wire 1 constants moved to `loop_planning`
`test_wire_1_risk_annotation.py` imports from `loop_planning`. No `scripts/exp*` or other test imports them from
`agent_loop`, and no test monkeypatches `agent_loop` names the moved block used. One stale doc path (NIT-2).

### 7. `tool_dispatch._record_turn_after_effects`
`execute_and_learn` callers: `agent_loop` §4 (via the `LoopRun` partial), `loop_controller.handle_confirmation`,
`loop_planning._drain_one`: all reach the helper on the normal return. No test patched the moved conversation-turn
block (source-scanning tests in the affected set pass). But see SHOULD-FIX-1.

### 8. `TwilioChannel.receive_sms(message_sid="")` and gateway `external` ordering
Callers: `comms/api.py` (passes the sid), `test_comms_inbound_1466.py` (two positional args; default holds),
`test_inbound_comms_1481.py`. No `src/` reader of `metadata["external"]`; the only readers are the two tests.
`ExternalInbox` filters on `source.startswith("comms:")`; the other `comms:` producers are `conversation.py`
(dict entries, not Percepts) and `concurrent_executor` (resource keys), so nothing else is captured. Holds.

### 9. The removed O6 gate
No `release_gates.json`, workflow, or test still names `test_o6_inbound_comms_reach_the_prompt_framed`; only historical
review records do. But see SHOULD-FIX-2.

## Findings

**SHOULD-FIX-1 (consumer outside the diff): the parallel batch acts on a turn but never acknowledges.**
`agent_loop` dispatches a proposal with `parallel_actions` through `tool_dispatch.execute_parallel_actions`, a
tracked exception that does not go through `execute_and_learn`, so `_record_turn_after_effects` never runs. A message
the model saw and acted on in a parallel turn stays held, is shown again (risking a second answer), and after 3 turns is
evicted with the WARNING "shown in 3 turns, none acted on", which is false. The CHANGELOG claims "leaves the inbox
only when a turn whose prompt showed it whole is acted on (dispatched)". Fix: acknowledge in the parallel branch
(`feed.ack(proposal.inbound_rendered_ids)` after `_execute_parallel` returns, ideally by calling the same
after-effects seam), and add a case to `test_inbound_comms_1481.py`. Or, if deliberately out of scope, state it in the
CHANGELOG and the runtime-tools invariant.

**SHOULD-FIX-2 (stale register): `docs/plans/advisor_mode.md` O6 row** still reads PARTIAL with "Inbound comms reach
the prompt unframed, as '[User via …]'". After this PR the Twilio/voice/`/api/command` paths are framed on the live
prompt. Fix: update the row (state what remains, for example #1519's working-memory thoughts and the other input paths
it lists), citing `tests/unit/test_inbound_comms_1481.py` as the guard.

**NIT-1: mangled comment in `_run_agentic_loop_body`'s signature.** `inbound_inbox` was inserted between `sim_adapter`
and its trailing comment, so the line reads `| None = None,  # #1481: ...  # Pre-built NullSimulationAdapter ...` and
the `sim_adapter` explanation now hangs under `inbound_inbox`. Fix: put `inbound_inbox` after the comment block.

**NIT-2: `docs/experiments/rationale/b4-exp37-floors/bio-faithful.md`** cites `_WIRE1_MIN_OBSERVATIONS` in
`runtime/agent_loop.py`; it now lives in `runtime/loop_planning.py`.

**NIT-3: D1's claim has one hole outside the diff.** `maxim --sim interactive --comms` builds the comms stack on the
CLI's `agentic_agent`, then runs `start_simulation_mode`, whose agents are new `MaximAgent`s. No loop watches that bus,
so the messages are dropped with no WARNING. This predates the PR, but the docstring says comms "can never be wired to a
bus no loop reads". Fix: skip or warn when comms are enabled with `--sim interactive`, or qualify the claim.

**NIT-4: two rule lines claim the same markers.** On a prompt that holds both tool output and inbound messages, the
tool rule says every `<<TOOL_OUTPUT id=...>>` frame is "data returned by a tool" and the inbound rule says every such
frame is "a message received from outside". Both say not to follow instructions inside, so safety holds, but the
scoping contradicts. Fix (optional): word the inbound rule per header ("frames whose tool= starts with comms: are...").

**Note (pre-existing, outside scope):** `scripts/release_gates.json` gates `agent_loop.run_agentic_loop` at
`max_lines 200`. Since bounded_stop step 0 that qualname is the thin wrapper, so the gate no longer measures the body
(`_run_agentic_loop_body`). Worth a follow-up issue, not this PR.
