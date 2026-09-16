# A harness loop must be PROVEN able to tick and act before a window can measure anything

**Established:** 2026-09-15/16, Exp 60 (learned drowning-avoidance) trial harness — three instrument
causes found in sequence behind one symptom, each by measurement, each masked offline
(PRs #730, #731, #732, #733; prereg Amendments 3–7 in
`docs/experiments/exp60_drowning_avoidance_prereg.md`).

## The symptom, and why it was unreadable

Two full runs of the FEAR-vs-ABLATED harness reported `actions=0` in every one of 120 probe windows,
in both arms, before and after training, while the FEAR seeds' fear read −1.0 and passed the live G2
gate through the production read. The verdict was INCOMPLETE, not NULL — but nothing in the record
could say whether the loop never proposed, proposed and failed, or never ran. **A mechanism that does
not run looks exactly like one that ran and found nothing.** A harness that records only tool
SUCCESSES cannot distinguish them; one that records every executor CALL and every loop TICK can.

## The three causes (in the order the measurements named them)

| # | What the window telemetry read | What the measurement named | Fix | Masked offline by |
|---|---|---|---|---|
| 1 | `ticks=1` per 4.3 s window, no proposals | Live thread profile: the loop iterated ~50×/8 s (socket-recv-bound, not slow) but reached its substrate branch ONCE; no proposal ever installed. Reading the idle gate: it wakes on pending input/work, a sim percept, a carried percept, the first step or an awaited LLM — the harness passes no LLM worker, and `MinecraftPerceptSource.has_pending` is the chat/death EVENT queue, never state. A bridge emitting state but no events idles the loop after step 0. | The substrate submit cadence is a wake source in its own right (`agent_loop._substrate_tick_due`, also the branch's own predicate). #732 | `FakeBridgeServer` emitted a text event every 5th snapshot ("wind shifts") — which ALSO produced the "one tick per five snapshots" reading first blamed on bridge cadence. Default is now `events=False`. |
| 2 | (same, at 100 ms) | The bridge's 500 ms default state cadence does bound sensor FRESHNESS for a 4 Hz-sampled window even after #732 | Bridge at `--state_interval_ms=100`; harness measures the cadence at preflight and refuses > 0.15 s. #731 | — (a real requirement whose first causal story was wrong; superseded, kept) |
| 3 | `ticks≈10`, `proposed=[flee, flee, …]`, `calls=[]`; "consecutive same-tool cap" on unexecuted proposals | Offline: `AutonomyController()` defaults to PLANNING → `can_execute_action(body affordance)` = "requires human approval for all actions"; the loop's "non-interactive auto-approve" branch is dead code (`should_prompt("plan_approval")` is unconditionally True); the queued proposal expires. The orchestrator hands its sim AUTs an AUTONOMOUS controller; `run_minecraft_aut` handed none — no harness on that path had EVER executed a body affordance through the loop. | `_loop_kwargs` passes an AUTONOMOUS controller (orchestrator parity). #733 | Nothing showed it: no harness recorded executor calls, only successes. Exp 58's "loop fired end to end" sentences were the preflight's direct `executor.execute` (dated correction notes appended). |

Same family throughout: **the harness passed nothing the orchestrator passes** (worker → wake, autonomy
→ execute). A harness AUT is a sim AUT; give it what the orchestrator gives one, or prove it does not
need it.

## The measurement ladder (reuse it; do not reason past a rung)

1. **Per-window telemetry** — every loop tick (proposal, active clusters, drives) and every executor
   call (tool, success, error, time) per placement: `exp60_run.py::_loop_window` +
   `SubstrateTelemetry` through `run_minecraft_aut(substrate_telemetry=…)` + a per-seed spy on
   `executor.execute`. Reads: did it tick? propose? execute? fail?
2. **Live thread profile** — `scripts/survival_world/loop_tick_probe.py`: cProfile on the loop
   thread + tick times. Reads: compute-bound, I/O-bound, or idle?
3. **Pending-proposal timeline** — the same probe logs every install/clear of the controller's
   proposal slot with its source (strategy, tool, approval flag). Reads: is the branch blocked by a
   stuck proposal, or never reached?
4. **Read the gate** the measurement points at (idle gate → wake sources; exec stage → autonomy
   decision), then **measure its decision offline** (`can_execute_action`, `should_prompt`) before
   changing anything.
5. **Reproduce the LIVE condition offline** — `FakeBridgeServer(events=False)`; a fake that emits what
   the live world does not (periodic text events) hides the defect it should expose
   (verify-with-the-real-consumer, one level down).

## Preflights every substrate-primary harness now carries (and must keep)

- **Raw bridge roster**: every gated sensor is in `client.latest_state()`; the body carries every
  declared sensor at its initial value whether or not the bridge writes it, so a body-side check is
  vacuous (`exp60_water_check.missing_bridge_sensors`).
- **Bridge cadence**: median snapshot interval ≤ the DV sampling period (`_measure_bridge_cadence`).
- **Loop liveness**: the full loop on the rest situation reaches its substrate branch ≥ 4× in 3 s, or
  refuse (`FROZEN["loop_liveness_min_ticks"]`) — would have refused seed 1 before any window.
- **Actuation through the BACKEND, never the executor**: an executor success books a POSITIVE causal
  link that makes the action selectable without the mechanism under test; assert
  `get_positive_outcomes(tool) == []` before the first probe.
- **US-free probes**: cap test windows below the FIRST pain the mechanism books on (the air-hunger
  edge, ~5 s), not the visible damage (~16 s); a bus subscriber for the whole seed marks any pain
  inside a window DIRTY.

Regression guards: `tests/unit/test_substrate_primary_wake.py` (wake source RED-gated on the events-off
fake; execution RED-gated; PLANNING negative control), `tests/unit/test_minecraft_harness.py`
(loop-kwargs pins: AUTONOMOUS, telemetry passthrough), `tests/unit/test_exp60_run.py` (liveness and
cadence contracts). Owning invariants: [runtime-tools.md](../agents/runtime-tools.md) §3 (wake
source), [simulation-experiments.md](../agents/simulation-experiments.md) §3 (autonomy parity).

## Follow-ups filed, not fixed

- The loop's PLANNING "non-interactive auto-approve" branch is dead code (its comment is wrong).
- The consecutive-same-tool cap (5 identical params, targets LLM hallucination loops) has no
  substrate-primary exemption: a sustained identical fear response executes at a 5/6 duty cycle
  (visible in the per-window record as a proposal tick with no call).
- `check_hard_stop` pauses the controller for the rest of a run on any percept transcript containing
  "stop"/"halt" (chat only on the Minecraft bridge).
- The orchestrator's loop is kept awake by an accidental source (`_submitted_recently`, the legacy
  120 s LLM-await window); changing it would change per-tick NAc decay cadence under Exp 56/57.
- Why the agent loop needs a wake source per percept at all in a sensor-driven mode (runtime brief).

See also: [cosine-separation-is-directional.md](cosine-separation-is-directional.md) corollaries 6–7
(the apparatus-side lessons from the same arc: capped distance sensors, replay with the interoceptive
state the protocol sets).
