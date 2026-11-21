# SEM interruptibility — which actions can be cancelled mid-execution, by whom, and how safely

> **PROPOSED 2026-11-12 — plan stub only, no code. Split out of [hybrid_arbitration.md](hybrid_arbitration.md)
> HA-O12 by owner decision on 2026-11-12.**
>
> **Owners**, one per device contract ([embodied_device.md](embodied_device.md)):
> - the robot's declaration belongs to `ReachyMiniController` and [motion_path_model.md](motion_path_model.md),
>   through a new **MP13** owner decision in DECISIONS.md. MP13 is **proposed: the motion-path owner has not yet
>   accepted it**;
> - Minecraft's belongs to its bridge and backend;
> - the contract's shape belongs to the embodiment brief.
>
> **Read [docs/agents/embodiment.md](../agents/embodiment.md) §1 before any part of this plan touches motion.**
> **No `src/` before the 1.4.0 cut** (mechanization backlog M89). The hybrid's first evidence slice is boundary-only
> and does not wait on this plan.

## Why

Nothing can cancel a running action today:
- Minecraft's `simulation/minecraft.py::MinecraftClient.call_action` blocks the loop thread for up to 15 s, and
  `escape_water` holds for up to 8 s.
- A Reachy move passes its point of no return at the first SDK call.
  - `hardware/controller.py::MotionTicket.cancel` works only until the ticket is `accepted`.
  - After that, the in-flight dispatch holds `_motion_lock` for the whole chunk.
- `runtime/preemption.py` has been Dormant since 2026-10-07.

At local LLM latency, a mid-action cancel mostly serves the danger interrupt, not the LLM.

## The contract

**Where it is declared: in code only, on the backend class, by its owner.**
- It is declared per affordance, as a mapping inside the backend class: `dig`, `consume` and `escape_water` live on
  one modulator.
- It is never a field in any loadable spec (YAML, `AffordanceSchema`, a bundled or LLM-generated entity spec),
  because a loadable `interruptible: anytime` would be a lie surface for a safety capability.
- The spec loader refuses any spec that tries to declare interruptibility. That covers specs a user writes, specs
  shipped in a bundle, and LLM-generated specs that arrive through `normalize_llm_entity_spec`.
- An affordance the backend does not declare is `none`.

**What each affordance declares:**
- `interruptible`: `none` (the default), `between_steps`, or `anytime`.
- Its point of no return, reusing `MotionTicket`'s `accepted` where one exists.
- Who may cancel it: the LLM, the danger interrupt, or both.

**The `protective` class, defined once here and cited by [hybrid_arbitration.md](hybrid_arbitration.md) §A.1d.**
Arbitration can never cancel a protective action. The class has two parts:
- **Static.** Affordances that the device owner declares protective **in code, on the backend class**, in the same
  mapping as `interruptible`. They are escapes (including `escape_water`), wake, sleep and the stop path. No loaded,
  bundled or LLM-generated spec can grant or claim membership, because membership confers immunity.
- **Dynamic.** Any selection a danger trigger makes or re-arms, whatever its tool, for as long as that danger
  episode lasts. The LLM can never cancel a protective action. A danger may cancel one only when that danger has a *different*
typed source and a higher priority. Drowning pain must never cancel the escape that relieves it.

**How it is enforced.** The type requires the declaration, and a deletion-proven test covers it.

**Scope: arbitration cancels only.** The contract governs only the cancels that come from arbitration, by the LLM or
the danger interrupt. It never limits the stop path (`_stop`, the stop latch, `goto_sleep`, `disconnect`). The stop
path can always end any sequence, a protective one included. MP10 has one stop generation per wake or sleep, checked
before every step and chunk. MP9 has a wake/sleep race end asleep and latched. Neither changes.

## Robot rules (hardware-safety review, 2026-11-12)

1. **`between_steps` at most** on Reachy. This works through a new non-latching abort, kept **per ticket**, checked
   between chunks.
   - It must never bump the shared `_abort_gen`, which only the stop path changes.
   - A "hold here" `goto_target` cannot cancel a move: it waits for `_motion_lock` and runs after the move has ended.
   - **Scope:** only moves that MP1 splits into chunks can be cancelled this way. A single-chunk move, which is most
     neck moves, stays uncancellable once `accepted`.
   - `anytime` stays ruled out until an MP decision allows it.
2. **A cancel is an operation on one `MotionTicket`, tied to its `basis_seq`.** A cancel that arrives after the
   ticket completed does nothing, and it never becomes a fresh motion command. In v1, only the danger interrupt may
   cancel on the robot. **No robot danger interrupt is defined yet:** MP9 is excluded (rule 5), and DN reflexes are
   not a cancel path. So v1 robot cancel has no caller until MP13 defines one.
3. **After a between-chunk cancel, nothing reaches for the cancelled move's target.** The daemon keeps the last
   completed chunk's target, and the stash already describes that chunk (motion_path_model §5.5's mid-split rule).
   No new rewriting is needed. A red-gate test checks that, after the cancel, the stash and the next `head=None` or
   partial-axes fill both come from the last completed chunk, never from the cancelled move's final target. This
   follows the style of `test_reachy_retained_axes.py`.
4. **Deceleration.** MP2 caps speed, not acceleration. Stopping from mid-trajectory needs its own MP decision: a
   bounded deceleration, or chunk boundaries only.
5. **The MP9 stop latch stays out of arbitration.** It runs a sleep sequence, and on Selfy's path it clears only on a
   wake that never comes.
6. **Every cancel path reaches motors only through `goto_target`**, with readback, rate, path sampling and the stop
   latch.
7. **Wear.** Cap direction reversals per window, or report the reversal rate.

## Cancel semantics (learning review, 2026-11-12)

- A cancelled action has partial effects, such as a half-dug block or a half-eaten item. The world reports what
  actually happened.
- A cancel is never credited as a failure, following the MP4 precedent: no refusal kind teaches a learner.

## Order

1. The contract.
2. Minecraft's long actions (`dig`, `consume`, `escape_water`), measured against Exp 60's window. This needs a
   non-blocking `call_action` and a bridge cancel.
3. The robot last, after MP13 and its own hardware review.
