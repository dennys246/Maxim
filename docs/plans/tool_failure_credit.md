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
**never** wired on `main`: wiring it is a new condition for every earned row, not a restoration.

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

- **TF1. Staged design B:** the tool-pain bridge owns invocation failure credit through a direct call, not
  the PainBus; a felt-only FRUSTRATION PainBus signal comes later, once `Reaction` carries its kind.
- **TF2. Only tools that ran:** failure credit only when the invocation reached `tool.run` (the GL2a
  record-iff rule). A hallucinated or inactive tool name is a cognitive error (`_tools_hallucinated`), not
  a tool failure.
- **TF3. Suppressed while a human drives:** the existing one-predicate interactive learning gate applies.

## Stages

### Stage 1: the stale-success defect ([#1207](https://github.com/dennys246/Maxim/issues/1207))

The bridge attributes by the event id `NAc.record_event` returns (not the signature), and the invocation's
NAc pending event is retired when the invocation ends, so a failure can never be booked POSITIVE by a
later success. Red gate first (one failure then one success books no positive), then the fix; ledger walk
(it changes `record_outcome` attribution: removes spurious positives).

### Stage 2: failure credit through the bridge (#1200)

- `Executor`'s returned-failure and raised branches (only where the tool ran: TF2; not under the
  interactive gate: TF3) call a new `ToolPainBridge.record_tool_failure(tool_name, invocation_id,
  error_kind)`, mirroring `record_tool_embodiment_failure`: a NEGATIVE attributed outcome, its RPE noted on
  the invocation, and the bridge's reflexion and SCN behaviour.
- `tool_dispatch.record_outcome` stops its `tool:X` `observe` for invocations the executor ran (it keeps the
  cluster, goal, energy and recent-outcome bookings, and books refusals and invocation-less paths as
  today); `PlanHistoryBridge` is declared a separate owner or moved off `tool:X` (decided in the stage's
  design pass).
- `build_executor` drops the legacy detector-subscription mode (one coherent signature; the canonical
  builders stay required-keyword).
- **Before any `src/`:** a loop-level strict red gate on `_loop_harness`'s `fear_water` arm (the failed
  `flee` books exactly one attributed NEGATIVE with a real RPE; Wire-4 cluster fear and percept valences
  byte-identical; `escape_water` gains no negative link; `recommend_action` records identical), and
  before/after fingerprints from the selection golden, the GL2a trio golden and the slow R3 offline
  campaign (`tests/unit/test_r3_run.py::test_offline_campaign_apparatus_and_one_event_per_in_process_arm`).
  Rig re-runs of Exp 60/61/62 are owed only if a fingerprint shows a decision or `t_surface` change;
  otherwise T1-4 and T3-9 are discharged structurally in the PR, with the walk written down.

### Stage 3: a felt-only FRUSTRATION signal on the PainBus (later)

Only after `Reaction` carries its kind (nociception step 2), so the distributor and episode valence can
skip FRUSTRATION, and after the context-similarity NAc subscriber declares a kind rule. The detector's
defects (item 5) are fixed then. Until Stage 3, `PainDetector.record_tool_error` stays without a live
caller and the receptor census says so.
