# Idle-time default mode network coordinator — cognition while the agent rests

> **PROPOSED 2026-11-12 — plan stub only, no code. Split out of [hybrid_arbitration.md](hybrid_arbitration.md) (its
> former Part B) by owner decision on 2026-11-12.**
> - This is [bio_faithful_roadmap.md](bio_faithful_roadmap.md) "Default-Network reorganization" **Stage B**, which
>   that roadmap says "needs its own plan + two-lens round". This file is that plan.
> - It is gated separately from the hybrid's arbitration, so a null in one does not take down the other.
> - **No `src/` before the 1.4.0 cut** (mechanization backlog M89, shared with the hybrid line's family of plans).
> - The name "default mode network" is earned only through this plan's experiment. Stage C (renaming
>   `default_network/`) stays deferred (hybrid HA-O5).

## What it is (Stage B as written, plus one owner addition)

When the agent is idle, the coordinator spends budgeted slices on:
- (a) **Incremental replay** through `memory/sleep_replay.py`. That module has been Dormant since 2026-07-14, and only
  tests import it.
- (b) **Spontaneous prospection.** `imagination/` invoked from internal state. Today it is wired only in
  `simulation/orchestrator.py`, so this slice is sim-only until a loop or robot caller exists.
- (c) **Anticipatory pre-activation**, once SCN producers exist (Stage B's slice (c), unchanged).
- (d) **Owner addition (2026-11-11): a body-state / drive self-model update.** This is the self-referential part of
  the owner's "contemplating the state of self" idea. It is recorded as an addition to Stage B; it does not replace
  slice (c).

## Constraints carried from the hybrid plan's reviews

1. **A consumer, or it has not shipped.**
   - Every slice writes to something that behaviour reads: NAc values, hippocampal strength, or the anticipatory
     pre-activation `recommend_action` consumes.
   - A self-narrative that only an LLM reads in its prompt is not a consumer. It would re-open the narrator-state
     confound (#1260/#1261).
   - **The first build has no LLM in the lane** (hybrid HA-O4).
2. **Dormancy is respected.** The coordinator exists only as an experiment arm until `sleep_replay` earns
   resurrection. Stage B's bar is an idle-replay arm against a session-end-only arm on cross-session recall, entered
   as `[engineering]`.
3. **Behaviour tier:** each slice is an innate prior, with a fixed budget and schedule.
4. **The lane never competes with the foreground.** If an LLM is ever added, it uses the `small` lane and is never a
   second consumer of the :8100 server while a foreground proposal is in flight. Until a check exists, that rule is
   mechanization backlog M89.
5. **Idle is rare in a run that uses the LLM.**
   - `_submitted_recently` keeps the loop awake for 120 s after each submit.
   - `_substrate_tick_due` wakes the loop every interval.
   - So `_loop_is_idle` (one caller, in `loop_gates.py`) leaves only short sleeps.

   This plan must define idle as "no task pressure" rather than "no wake source", or else add rest phases to the
   world. Doing so costs a touch-rule edit.
6. **Known integration debt** (from Stage B): `network.py` builds an internal `PainBus()`. Close it here rather than
   add a third wiring path.

**Language.** No public surface calls this "consciousness" (framing strategy). The claim is "idle-time cognition
pays", measured on a named metric.
