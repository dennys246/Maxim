# R2 — Premise check: do the world-owned drives move behaviour? (a null)

> **1.3 UPDATE (2026-09-11): break 1 closes — the PRIOR half only; the survival premise is still NULL.**
> The 1.3 survival-loop build fixed R2 break 1 (the drive→corrective-action prior): `_read_drive_states`
> now derives a normalized corrective NEED on deficit (`food→hunger`, `health→threat`), and the
> drive-affinity heuristic no longer feeds it raw sensor values or lands on passive `read_` tools.
> Re-running `r2_drive_premise_probe.py` (prior-only) now reports `moves_behaviour: true` — hungry/hurt →
> `eat`, satiated → nothing. **But that probe instruments the prior ONLY**; the survival PREMISE this doc
> names needs all three breaks. **Break 2 landed the measured-relief credit PATH**
> (`tests/unit/test_survival_learns_break2.py`) — a hungry agent's real `eat` relief is credited to
> the interoception cluster instead of being withheld. **Break 3 (a world that affords the acts) now
> has a live apparatus** (`scripts/survival_world/setup_world.py` — a Paper 1.16.5 survival world
> where hunger drains and food is seeded) and a **live smoke** (`scripts/survival_world/break3_smoke.py`)
> that drove the loop against the real bridge: under a real hunger deficit the substrate prior
> selected `eat`, `eat` executed, food rose, and break-2 produced the measured-relief SIGNAL on
> the interoception channel (`drive_relief_channel = "interoceptive"`, positive measured
> food-delta) — **all three breaks COMPOSE on the live path**. (The smoke also caught a one-action
> credit-lag — the bridge's `eat` snapshot predated mineflayer's food-update packet — fixed
> eat-local in `scripts/minecraft_bridge/index.js`.)
>
> **This is composition validation, NOT the measurement — and it validates the credit path's
> INPUT, not the booking.** The smoke reads the relief SIGNAL break-2 emits (`side_effects`); it
> does NOT call `record_outcome`, so no cluster reward is booked and no bias forms — that (and the
> LEARNED-bias claim R2's premise actually asks, "do the world drives measurably MOVE behaviour
> across many trials") is the pre-registered measurement's job, run through the real agent loop.
> The smoke is print-only (no gated results). So the status below stays PREMISE-NULL and the gated
> `data/r2_drive_premise.json` 1.2 record is unchanged until a pre-registered, two-lens-reviewed,
> provenance-stamped learned-bias run flips it. See `tests/unit/test_survival_drive_prior.py`,
> `tests/unit/test_survival_learns_break2.py`, `scripts/survival_world/`, and CHANGELOG `[Unreleased]`.

**Status: PREMISE-NULL, 2026-09-07.** The Minecraft survival ladder's R2 rung
([minecraft_benchmark.md](../plans/minecraft_benchmark.md) Part II) asks whether the
`minecraft_player` world-owned drives (`health` homeostatic, `food` entropic, both
`drift_rate: 0.0` — "the game drains it, not the model") measurably move behaviour
toward corrective affordances (`eat` when hungry, `attack_nearest` when threatened).
They do not. **Per the rung's own stop rule, R3 and R4 do not run in 1.2** — and this is
the finding, not an apparatus bug to tune away. A null ships as a null (the Exp 53 shape).

## The measurement

- **Instrument:** [scripts/r2_drive_premise_probe.py](../../scripts/r2_drive_premise_probe.py).
  On a substrate-primary agent, first-contact action selection has exactly one active
  signal on a *fresh* substrate — the NAc cold-start drive prior (learned cluster bias,
  causal links and reward bias are all zero before any experience; the explore bonus is
  default-0). `recommend_action` is a pure function of `(available_tools,
  current_drives)`, so a fresh-NAc probe over the real `minecraft_player` tool roster
  isolates the prior **exactly** — a live server would add noise, not signal, on this
  channel, so the offline probe is the exact instrument, not an approximation. It mirrors
  `runtime/agent_loop.propose_via_substrate` (`registry.list()` minus
  `INTROSPECTION_TOOL_NAMES`) and reads drives through the production
  `_read_drive_states`.
- **Record (gated, clean tree):** [data/r2_drive_premise.json](data/r2_drive_premise.json).

| state | drives read | drive prior selects | reasoning |
|---|---|---|---|
| STARVING+HURT (food 2, health 5) | `{health: 5.0, food: 2.0}` | `read_minecraft_player_health` | `drive:health(5.00) name-match` |
| SATIATED+HEALTHY (food 20, health 20) | `{health: 20.0, food: 20.0}` | `read_minecraft_player_health` | `drive:health(20.00) name-match` |

`verdict: PREMISE-NULL`, `moves_behaviour: false`. The prior selects the same passive
sensor-**read** tool in both states, and — because the drive score is monotone in the raw
sensor value — it scores that read *more strongly when healthy* (20.00) than when hurt
(5.00). Behaviour does not move with need; it moves backwards.

## Why — three independent structural breaks (any one sinks the rung)

1. **The drive prior has no corrective affinity for these drives, and name-matches the
   wrong tool.** `_DRIVE_TOOL_AFFINITIES` (`decisions/nac.py`) is keyed `hunger`,
   `thirst`, `fatigue`, `cold`… — there is **no `food` and no `health` key**. The drive
   component's fallback is a substring match of the *drive name* against the *tool name*:
   `"food"`/`"health"` appear in the **sensor-read** tool names
   (`read_minecraft_player_food`, `read_minecraft_player_health`) but in neither `eat` nor
   `attack_nearest`. So the drive term lands on the read tools and never on a corrective
   affordance. `_read_drive_states` returns **raw** sensor values (no deficit derivation
   for `food`/`health`; only thermal drives get a derived corrective "cold" need), so the
   term is also polarity-inverted: largest when satiated.
2. **Corrective affordances cannot self-learn on the live body.** `eat`'s
   `self_effect {food: +4.0}` targets a sensor a live measurement stream owns, so
   `ModulatorAffordanceTool.execute` strips the modeled write and sets
   `drive_credit_withheld = True` (`embodiment/tool_bridge.py`), which also suppresses the
   flat +1 tool-success cluster floor. Measured-relief credit for world-owned
   *interoceptive* drives is unimplemented (only the exteroceptive/azimuth measured path
   exists — "Phase 2"). So even if `eat` were tried, it accrues **no** positive credit on
   the live path, and no learned bias can form to rescue the dead prior.
3. **The bridge cannot supply the means to survive — on this world.** `scripts/minecraft_bridge/index.js`
   `eat` throws `"no food in inventory"` on an empty inventory and `attack_nearest` throws
   `"no hostile nearby"`; there is no `give`/`craft`/`pick_up` affordance. On the
   superflat/void contingency world (no mobs, no food), both corrective affordances are
   *unexecutable* — a perfectly-motivated agent still could not eat. Unlike breaks 1–2,
   this one is **world-config-contingent**: a resource-rich world (mobs present, inventory
   seeded) would make `eat`/`attack_nearest` executable, and the deeper obstacle there —
   multi-step acquisition, since only `mine_block` is offered and there is no direct food
   path — is precisely R4's "multi-step delayed credit" thesis, not an absolute
   unexecutability.

Breaks 1 and 2 are **substrate-mechanism** facts that flat-line the intrinsic path on ANY
world: the drive→action prior is dead (1) and the learning path that could repair it is
withheld (2). Break 3 is the world-config layer on top. The deferral rests on 1–2 alone; 3
is why even the void-world apparatus this rung would have used cannot rescue it.

**Scope: this is the substrate-primary intrinsic path.** The null is a fact about the
*substrate-primary* action channel — the channel the survival benchmark's "isolated" arm
runs on (substrate-primary by design, no LLM in the action path, as Exp 56/57), and the
one Oasis's substrate-learning thesis rests on. An LLM-primary agent reading `food=2` in
its `body_state` prose could in principle choose `eat` — but that is not the channel under
test, and break 3 blocks the LLM path too on the void world.

## What this does — and does not — say

- **It does NOT touch the 1.2 headline (Exp 56) or the sharing/scaling line.** Those ride
  a *taught* want on the `minecraft_bench` body: the Exp 52/56 teacher mints credit
  directly via `NAc.credit_operant_reward`, bypassing breaks 1–2 entirely, on a body whose
  affordances are always-executable. Exp 57 (dose–response), R0 (multi-seed) and R1
  (cross-layout) all ride that proven apparatus and are **unaffected**.
- **It removes R3 (survival benchmark) and R4 (structure formation) from 1.2.** R3's
  "isolated" arm depends on intrinsic survival pressure, which is absent; its "taught" arm
  would reduce to "install and transfer an arbitrary contingency" (Exp 56 again), not
  survival; and the benchmark's dynamic range collapses (isolated = flat-dead) so no
  Goldilocks calibration exists to land. Building the survival loop — a corrective-need
  derivation for `food`/`health`, a measured-relief credit path for interoceptive
  world-owned drives, and a food-acquisition affordance in the bridge — is **new
  engineering that engineers the survival outcome**, exactly what D1 (no synthetic sensor
  to make a behaviour rewarding) warns against doing casually. It is 1.3-line work, gated
  on a deliberate design decision, not a 1.2 verification.

## Consequence recorded

1.2 scope is: the Exp 56 headline (EARNED) + Exp 57 dose–response + R0 multi-seed + R1
cross-layout + gates + release. R3/R4 move to the 1.3 line beside the perception fabric,
where the survival loop can be *designed* (not back-fitted) and the three breaks above are
the explicit build list. See [minecraft_benchmark.md](../plans/minecraft_benchmark.md)
Part II, R2/R3/R4.

## Regression guard

**Re-run on:** `_DRIVE_TOOL_AFFINITIES` / `recommend_action` drive-component change,
`_read_drive_states` change, `minecraft_player.yaml` drive/sensor/affordance change,
`ModulatorAffordanceTool` `drive_credit_withheld` change, Minecraft bridge affordance-set
change. **Guard:** [scripts/r2_drive_premise_probe.py](../../scripts/r2_drive_premise_probe.py)
+ [data/r2_drive_premise.json](data/r2_drive_premise.json). If a future 1.3 build closes
the three breaks, this probe flips to `PREMISE-HELD` and R3 is unblocked.
