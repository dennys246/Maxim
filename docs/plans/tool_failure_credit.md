# Tool-failure credit: wiring a failed tool into learning ([#1200](https://github.com/dennys246/Maxim/issues/1200))

> **ACTIVE 2026-10-10.** Owner decisions TF1–TF3 (DECISIONS.md 2026-10-10). Built from a five-angle
> read-only dive (producer, consumers, credit ownership, ledger blast radius, history) with probes against
> `origin/main` 45f1c039. Companion plans: [deferred/pain_bus_bridge_subscriber_unification.md](deferred/pain_bus_bridge_subscriber_unification.md)
> (the trap Stage 3 must not walk into) and [deferred/nociception_layer.md](deferred/nociception_layer.md),
> revived into [autonomic_layer.md](autonomic_layer.md) (its step 2, kind on `Reaction`, gates Stage 3).

## Why this plan exists

`PainDetector.record_tool_error` is reached only through `Executor._report_failure` when the executor has a
`pain_detector`, and no production builder has ever given it one. The detector was the intended producer
(DECISIONS #15, 2026-03-31: "tool errors route through PainDetector → NAc → FearAgent"); PR #114 wired it
(228135a4) and its own review fold removed it (f5f9df4e, aimed at a dead subscription). So tool failure was
**never** wired on `main` in effect: f5f9df4e removed the orchestrator wiring, and #114's
`agentic_runtime` branch survived until c3ddecd1 (2026-04-19) but was dead, because nothing ever set
`self._pain_detector`. Wiring it is a new condition for every earned row, not a restoration.

## What the dive found (code-read and probed)

1. **The builder cannot express it.** `runtime/bootstrap.py::build_executor` raises when `pain_bus` and
   `pain_detector` are both passed (the detector is modelled as a legacy subscription source), and
   `PainDetector._emit_pain` publishes to its bus *or* calls its callbacks, never both.
2. **On the PainBus as it stands, tool failure makes learning worse** (probed):
   `create_pain_nac_subscriber` runs before the bridge, links the failure by context similarity (the pain
   context carries the same `params` dict as the bridge's pending event: similarity 1.0) under
   `pain:unknown:unknown:TOOL_FAILURE`, and consumes the pending event, so the bridge's attributed outcome,
   per-invocation RPE and reflexion are lost; the per-type cooldown (shared across tools) drops a second
   tool's pain and leaves its pending event to be booked against a later tool's pain; on Reachy a failure
   within 5 s of a head move is booked against the movement (`PainCircuitBridge`); the distributor, episode
   valence and the REACTION display line treat frustration as harm. This is the unification plan's trap in
   a worse form. Its trip-wire (`tests/unit/test_pain_bus.py::test_subscriber_does_not_link_pending_tool_event`)
   covers only embodiment-shaped context.
3. **Today a failed tool books two NEGATIVE links and no surprise:** `runtime/tool_dispatch.py::record_outcome`
   (`nac.observe`, keyed on the error text and goal) and the plan bridge
   (`bridges/planning_bridge.py::PlanHistoryBridge.record_plan_outcome`); the executor's failure branch never
   calls the tool-pain bridge, so `ToolOutput.rpe` is `None` for every failure. The memory-strength design
   says failure enters through `|RPE|`.
4. **A separate live defect** (probed): a failure leaves the NAc pending event `tool:X`
   (`ToolPainBridge.finish_invocation` retires only the bridge's own entry), and the next SUCCESS of `X`
   matches every pending event with that signature inside 300 s, so the failure is booked POSITIVE
   (one failure then one success → `tool:X:positive n=2`).
5. **The detector itself:** its "escalating" intensity is a lifetime per-tool counter that never resets;
   the cooldown is per pain type, shared across tools; the context has no `source` / `entity` /
   `failure_mode`; the cooldown and counter read-modify-write are unlocked; it ignores interactive mode;
   `record_tool_running` (TOOL_SUSTAINED) has no caller.
6. **Blast radius:** in every committed survival run (Exp 60/61/62, R3) the AUT's first fear action is
   `flee`, which always fails underwater, inside the measured window. Selection reads the MAX negative-link
   confidence, which an existing failure link already carries, so the mechanics predict no selection
   change; no current offline test records what would change. T1-4 and T3-9 fire by wording; T1-13/14/15
   only on a broad reading of "credit-path change". The `use` tool books `tool:use:<action>` in
   `tool_dispatch` but the bridge keys `tool:use`, so for `use` a bridge link is a new term.

## Owner decisions (2026-10-10)

- **TF1. Staged design B:** the tool-pain bridge owns invocation FAILURE credit through a direct call,
  not the PainBus (clarified at the plan review: the cut is failure-only; success links are unchanged);
  a felt-only FRUSTRATION PainBus signal comes later, once `Reaction` carries its kind.
- **TF2. Only tools that ran:** failure credit only when the invocation reached `tool.run`, returned or
  raised (broader than GL2a's record-iff, which mints nothing on the raised path). A hallucinated or
  inactive tool name is a cognitive error (`_tools_hallucinated`), not a tool failure.
- **TF3. Suppressed while a human drives:** the existing one-predicate interactive learning gate applies.

## Stages

### Stage 1: the stale-success defect ([#1207](https://github.com/dennys246/Maxim/issues/1207))

The bridge attributes by the event id `NAc.record_event` returns (not the signature), and the invocation's
NAc pending event is retired when the invocation ends, so a failure can never be booked POSITIVE by a
later success.

- **API trap (plan review S1):** `NAc.record_outcome(event_id=...)` treats its argument as a SIGNATURE
  and builds `outcome_signature=f"{event_id}:{valence}"`; passing the real id (`tool:X:<time_ns>`) through
  it would mint a new link per invocation and confidence would never accrue (a silent selection change).
  Use `record_outcome_full(attributed_event_id=id, outcome_signature=f"{sig}:{valence}")`, and add an NAc
  retire API (none exists).
- Retiring at invocation end removes the 300 s pending window for tool events. Its other consumers are
  context-similarity matches: `create_pain_nac_subscriber` and `ToolPainBridge._on_embodiment_pain`'s
  unattributed fall-through (neither matches a tool event in practice: pain contexts carry no `params`).
  R4 delayed credit must not assume the window.
- `record_tool_start` runs in `Executor.execute` BEFORE `_run_started`'s inactive-scene and unregistered
  gates, so those never-run invocations also queue `tool:X` (an inactive tool that later activates and
  succeeds books a POSITIVE for the call that never ran). Move `record_tool_start` after the gates (TF2).
  Side effect to state and fingerprint: `record_tool_start` also seeds the invocation's pain entry, so
  never-run calls then stamp `ToolOutput.pain = None` ("not measured") instead of `0.0` ("watched, nothing
  fired"), a memory-strength 2S-c capture signal (arguably more correct; the 2S-c tests may pin it).
- Red gate first, three arms: one failure then one success books exactly one positive; a never-run
  (inactive / unregistered) call queues no NAc pending event and the bridge books nothing for it
  (`tool_dispatch` still books its NEGATIVE, as today); confidence still accrues across invocations. Ledger walk:
  T1-4, T1-6, T1-8 (a `record_outcome` attribution change: it removes spurious positives).

### Stage 2: failure credit through the bridge (#1200)

- **Producer.** `Executor`'s returned-failure and raised branches (the two paths that reached `tool.run`:
  TF2; the interactive gate already suppresses `record_tool_start`, so the call no-ops there: TF3) call a new
  `ToolPainBridge.record_tool_failure(tool_name, invocation_id, error_kind)`, wrapped in the same
  try/except as the success branch. It REPLACES the body of `_on_pain`'s dead TOOL_FAILURE / TIMEOUT /
  INVALID_INPUT branch (one implementation; Stage 3 deletes the branch, or a bus signal arriving first would
  pop the pending entry and book, and the signal would not be felt-only).
- **What it books.** A NEGATIVE attributed outcome (through Stage 1's id attribution) and its RPE noted on
  the invocation, so `ToolOutput.rpe` carries the failure's surprise. NOT the embodiment-failure path's
  harm semantics: its temporal event is declared as a frustration event, not `"pain"` (or omitted), so the
  TemporalCreditDistributor never receives tool frustration as pain. Reflexion (`rpe > 0.3`, a second
  Hippocampus capture of the same action beside the loop capture) is decided in the stage's design pass,
  against `memory_strength_and_forgetting.md`'s "one event once" rule. Whether its context carries
  `agent_id` (which would newly feed the Wire-1 Welford risk profile and the bundle's
  `event_outcome_welford`) is decided there too.
- **The cut (failure-only; plan review B1, B2).** `tool_dispatch.record_outcome` skips its NEGATIVE `tool:X`
  `observe` ONLY for an invocation whose failure the bridge booked. The signal is a declared executor
  stamp on `ToolOutput` (like `rpe`; CC3 status stated in the stage), never inferred from "the executor
  ran it" or from `rpe is None`: executors with no NAc have no bridge (`api.py`'s learning=False path, the
  sandbox sub-executor), the interactive gate suppresses the bridge, and a bridge exception is caught; on
  all of those `record_outcome` books as today. SUCCESS outcomes, and the D53 case (`success=True` with
  `outcome_valence="negative"`, which the bridge's `record_tool_complete` does not book), are unchanged.
  Whether the bridge should own ALL invocation outcomes (success is triple-booked today) is a separate,
  later decision.
- **Interactive behaviour changes (state it in the PR).** `record_outcome` has no interactive check, so in
  interactive mode it is an ungated NAc writer the embodiment brief's "FOUR sites" invariant does not list.
  Under the cut's key, interactive failures also stay on the ungated dispatch path (the bridge does not book
  them), so BOTH its success and failure paths stay ungated; Stage 2 must add `tool_dispatch` to that
  invariant's site list (or gate it, an owner call at the stage's start).
- **Every consumer of the links being cut** (dispatch links: `outcome_type="tool_result"`, outcome text
  `failure:<error[:50]>`, context `{agent_id, goal}`; bridge links: `outcome_type="result"`,
  `tool:X:negative`, context `{"params"}`), walked in the stage's wire-integrity round:
  `NAc.recommend_action` (max negative confidence), `tools/introspection.py::PredictOutcomeTool`
  (`predict_all_outcomes` text to the LLM: the error text is lost), `tools/discovery.py::_nac_annotation` /
  `_apply_nac_ranking`, `harm/tool_predictor.py`, `agents/exec_agent.py` (observation counts),
  `agents/memory_agent.py` (`predict`), `integration/bio_enrichment.py::scan_links_for_keywords`,
  `fear_agent` → `should_gate_tool` (it keys `build_tool_signature`, so for `use` it reads
  `tool:use:<action>`, dispatch-only links: gating for `use` goes blind unless the bridge keys the same way),
  and the hivemind export (`_scrub_link_for_bundle` emits `f"{outcome_type}:{valence}"`, so
  `tool_result:negative` becomes `result:negative`: public-format content, a format-freeze question).
- **Every `record_outcome` caller, classified ran / not-ran:** `execute_and_learn` (both call sites), the
  parallel batch (rejected and ran actions share one call), the loop's hard rejection, the agent-fallback
  exception, `book_refusal` and `book_machine_refusal`.
- **`build_executor`** drops its legacy detector-subscription mode (only `tests/unit/test_build_executor.py`
  passes `pain_detector=`). `Executor(pain_detector=)` and `ToolPainBridge(pain_detector=)` stay for Stage 3.
  The receptor census row for `record_tool_error` is updated when it changes.
- **Before any `src/`:** a loop-level strict red gate on `_loop_harness`'s `fear_water` arm (the failed
  `flee` books exactly one attributed NEGATIVE with a real RPE; Wire-4 cluster fear and percept valences
  byte-identical; `escape_water` gains no negative link), and before/after fingerprints of
  `recommend_action` SCORES (not only decisions: which link family accrues confidence differs), the
  selection golden, the GL2a trio golden, a Hippocampus trace-count / tag fingerprint (reflexion), and the
  slow R3 offline campaign (`tests/unit/test_r3_run.py::test_offline_campaign_apparatus_and_one_event_per_in_process_arm`).
  Rig re-runs of Exp 60/61/62 are owed only if a fingerprint shows a decision or `t_surface` change;
  otherwise T1-4 and T3-9 are discharged structurally in the PR, with the walk written down. No ledger row
  names a new capture site; the nearest is T1-16 (carried recall, Exp 63), whose triggers (the ranking and
  query paths, Hippocampus save/restore, the memory record shape) do not fire by wording, but a kept
  reflexion capture adds recall candidates, so the reflexion decision carries a recall fingerprint.

#### Stage 2 design (2026-10-10; owner decisions TF4–TF9; adversarial design pass folded)

A read-only code map at `origin/main` bfed85bf (after #1207) settled the open questions above, and an adversarial
design pass on this note (DO-NOT-BUILD as first drafted: script callers double-count, aliases defeat the cut, a
ledger row missed) is folded in. Two premises above changed: `FearAgent.should_gate_tool` is dead (nothing sets
`FearAgent._tool_pain_bridge`), so the `use` keying blinds nothing live; and `TemporalCreditDistributor.record_event`
branches only on `"deliberation"`, so a `"pain"` vs `"frustration"` label changes nothing by itself: what an
emitted event adds is an eligibility trace. **This design supersedes the bullets above where they differ**
(notably the red gate's location, below).

- **The stamp.** `ToolOutput.nac_failure_booked: str | None = field(default=None, repr=False)`: the event
  signature the bridge booked the failure under, or `None`. A signature, not a bool, so the cut can check it
  booked the SAME key dispatch would write (an alias such as `recall` → `memory_recall` books
  `tool:memory_recall` while dispatch computes `tool:recall`). Tools never set it; `Executor._stamp_invocation`
  is its only writer (a `failure_booked=` keyword; the field joins the short-circuit tuple, or an otherwise
  unchanged output would drop it, and an outer executor re-stamps an inner one's value). `repr=False` because
  `str(ToolOutput)` is persisted as `Outcome.result`. `ToolOutput` is runtime-ephemeral (not in the CC3
  roster): the bus copies six fields, `actions.jsonl` three, the hivemind none, and no wrapper copies a stamped
  output (the prefetch cache keeps only a success's `output`).
- **The producer.** `ToolPainBridge.record_tool_failure(tool_name, invocation_id) -> str | None` pops the
  invocation's `_pending_tools` / `_pending_contexts` entries (NOT `_pending_event_ids`, which `_book_invocation`
  reads and `finish_invocation` retires); an absent entry returns `None` silently (the interactive and
  speculative gates, never-started calls). It books ONE NEGATIVE through Stage 1's `_book_invocation` with
  `context=None`, notes the RPE on the invocation, and returns the event signature only when a link was
  written (a run longer than NAc's 300 s window, or an event consumed early by an unattributed match, books
  nothing and returns `None`, so dispatch books as today). It does **not** reflect (**TF4**), emit a temporal
  event or register with the SCN (**TF5**), create CAUSES edges, touch the learned tool index, or tag
  `agent_id` (a failures-only Welford would read all-zero rewards as a reliable action). `_on_pain`'s dead
  TOOL_* branch is left untouched (Stage 3 deletes it; delegating now would only rewrite two tests that pin its
  reflexion and SCN calls); since the executor calls `record_tool_failure` before `_report_failure`, a
  detector signal would find no pending entry either way.
- **The executor.** In `_run_started`'s raised branch and its returned-failure branch (the TF2 "ran" paths),
  `if bridge and not suppress_nac:` call `record_tool_failure` inside a try/except reported in the Stage-1 form,
  BEFORE `_report_failure`, and pass the result to `_stamp_invocation(failure_booked=...)`. `suppress_nac` is
  now "a human drives OR the action is speculative" (**TF9**, below). The inactive-scene and unregistered paths
  are unchanged (never ran; no invocation was started since #1207).
- **The cut (TF7, made structural).** `tool_dispatch.record_outcome(..., bridge_booked: str | None)` becomes a
  REQUIRED keyword (no default: a caller that forgets it is a `TypeError`, not a silent double count) once
  Session A's two `agent_loop.py` sites pass it (TF8). Until then it carries a temporary `None` default and
  `tests/unit/test_tool_failure_credit_1200.py::test_every_record_outcome_caller_passes_the_failure_stamp`
  (an AST scan of `src/` and `scripts/`) holds every other caller to it, with exactly those two sites allowed.
  **Done 2026-10-11:** #1231 landed the two sites and #1221 dropped the default, so the keyword is required. It skips
  ONLY its `nac.observe(event_type="tool", outcome_type="tool_result", ...)` and only when `bridge_booked` equals
  its own `build_tool_signature(tool_name, tool_params)`; a compound or aliased signature, or a `None`, books as
  today. Goal credit, cluster reward, the operant pending action, the energy observe and the plan bridge are
  unchanged. Every caller passes it explicitly:
  - ran, with the output in hand: `execute_and_learn`'s try branch AND its except branch (a step between
    `execute` and `rec_outcome` can raise after the bridge booked; #1145's overlap), `execute_parallel_actions`
    (threaded through its per-action dict as an `_`-prefixed key, already stripped), and the script callers
    that replay the loop's sequence through the real executor: `scripts/exp56/common.py::execute_and_record`
    (the T1-11 harness), `scripts/survival_world/r2_learned_bias.py` and `dark_danger_probe.py`;
  - not ran, or no executor (`None`): `book_refusal`, `book_machine_refusal`, the `scripts/orient_substrate/`
    probes that call `tools[action].execute()` directly (`2_full_path_probe.py`, `9_tool_competition_repro.py`),
    and in `runtime/agent_loop.py` the hard rejection and the agent-fallback except (the stamp of a result the
    fallback's own `execute` returned, else `None`: `result` must be bound to `None` before that `try`, or an
    exception before `execute` returns would read an earlier action's output);
  - tests that call `tool_dispatch.record_outcome` directly gain the keyword (`test_tool_dispatch.py`,
    `test_modality_seam.py`, `test_sem_motor_binding.py`, `test_substrate_primary_scene_harm.py`, and the
    integration `test_multi_agent_attribution{,_scale}.py`, `test_substrate_primary_aut.py`);
    `test_approved_proposal_situation_1083.py` wraps the real function. **`agent_loop.py` is Session A's fenced
    file (owner, 2026-10-10: coordinate):** Session A lands those two call-site lines; this PR waits for them or
    merges with Session A's go-ahead.
- **Speculative calls (TF9).** `runtime/executor.py` gains a `speculative_calls()` context manager over a
  module `ContextVar`; `prefetch.py::_execute_cached` wraps its `execute` in it, and the executor treats a
  speculative call like an interactive one for the bridge (no invocation is started, so it books neither
  valence). The agent did not choose these calls. NOT an action-dict key: the action dict is the model's own
  JSON, forwarded unchanged by every wrapper, so a model-written flag would let a chosen call escape credit
  (a red-gate case pins that an action carrying `"_speculative"` books as normal). Prefetch calls the loop's
  executor on its own thread; a wrapper that hopped threads would lose the variable and book as today, the
  safe direction. This also removes today's speculative
  POSITIVE `tool:read_file` / `tool:glob` links: a stated success-side change.
- **Interactive (TF6).** No new gate: `tool_dispatch.record_outcome` and `PlanHistoryBridge` are added to the
  embodiment brief's interactive-gate invariant as named UNGATED writers; gating them is filed separately (it
  would also remove interactive SUCCESS links, which TF1 keeps unchanged).
- **`build_executor`** keeps rejecting `pain_bus` + `pain_detector` together; dropping its legacy detector mode is
  Stage 3's (the detector is that stage's producer), so Stage 2 changes no builder signature.
- **What changes, to state in the PR.** A booked failure carries `ToolOutput.rpe` (≈0.5 for a fresh link) instead
  of `None`: the persisted `str(ToolOutput)` gains the value and `capture_loop_action`'s salience gains
  `rpe * 0.5`. Its NAc trace moves from the dispatch family (`tool_result`, `failure:<error[:50]>`, context
  `{agent_id, goal}`) to the bridge family (`result`, `tool:X:negative`, context `{params}`):
  `PredictOutcomeTool` shows the outcome signature instead of the error text, and hivemind bundles carry
  `result:negative` rather than `tool_result:negative` (content, not format: no frozen field, allowlist or
  fixture changes, `public_format_freeze.md`). Bridge NEGATIVEs with no dispatch twin appear for: the
  `write_file` overwrite retry's first attempt (a real refused invocation, #1146; accepted), returned failures
  on the agent-fallback path, the classic `run_agent_loop`, `foundry.py` and `tools/sandbox.py`. An aliased
  failure books both the bridge's canonical key and dispatch's alias key, as aliased successes already do. In
  the survival world the main action tool is `use`, whose dispatch signature is compound (`tool:use:<action>`),
  so the cut almost never fires there: `use` failures keep their dispatch link AND gain the bridge's `tool:use`
  NEGATIVE.
- **Known residuals (named, not fixed here).** If NAc's activity collector raises after `record_outcome_full`
  wrote the link, the executor's except leaves the stamp `None` and dispatch books a second link (the collector
  call is unwrapped inside NAc); and in the parallel batch a raise after an action's dict was appended adds a
  second, unstamped failure dict for it. Both go to a follow-up issue.
- **Red gate first** (strict xfail, through the public paths; it supersedes the `_loop_harness` red gate above:
  that harness is Session A's and gets no new pin): a returned failure and a raised failure each book exactly
  one attributed NEGATIVE with a real RPE and stamp the booked signature; `execute_and_learn` then writes no
  `tool_result` NEGATIVE for that tool; an interactive, speculative, bridge-less (no NAc), aliased or
  compound-signature failure books through dispatch exactly as today; a model action carrying
  `"_speculative"` books as normal; a success books exactly as today; a
  `record_outcome` call without `bridge_booked` is a `TypeError`. **Fingerprints before/after** (existing tests on
  both commits): `recommend_action` scores on the `fear_water` arm, the selection golden, the GL2a trio golden,
  a Hippocampus trace-count/tag count, and the slow R3 offline campaign. Any decision or `t_surface` change →
  the Exp 60/61/62 rig re-runs are owed; otherwise T1-4, T1-6 ("`record_outcome` credit-rule change", the cut is
  one) and T3-9 are discharged structurally in the PR, with the walk written down, and T1-11's harness
  (`exp56/common.py`) is walked as a caller.

### Stage 3: a felt-only FRUSTRATION signal on the PainBus (later)

Only after `Reaction` carries its kind (nociception step 2), so the distributor and episode valence can
skip FRUSTRATION, after the context-similarity NAc subscriber declares a kind rule, and after `_on_pain`'s
TOOL_* branch is deleted (Stage 2 moved its body into `record_tool_failure`). The detector's
defects (item 5) are fixed then. Until Stage 3, `PainDetector.record_tool_error` stays without a live
caller and the receptor census says so.
