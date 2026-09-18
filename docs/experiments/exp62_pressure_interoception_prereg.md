# Exp 62 (DRAFT v2.1, 2026-09-18 — owner decisions D1–D4 TAKEN; four-lens design review FOLDED) — cross-context transfer of the drowning-fear BY THE BODY: what the shipped survival body already carries across pools, the context wall it does not cross, and why a pressure sensor is not the lever

> **STATUS: DRAFT v2.1 (owner decisions D1–D4 TAKEN 2026-09-18: pool 2 at y 95; n 12/12/3; rung A RUNS; the pressure drive deferred with its 1.4 path recorded) — v1 (the pressure-interoception claim) went to the full four-lens design review
> on 2026-09-17; all four lenses returned DO-NOT-BUILD as drafted, and all four ran the SAME offline
> replay independently on the committed pool-1 vectors and got the same grid (reports preserved under
> `docs/experiments/rationale/exp62-pressure-interoception/`; the replay is now a committed check,
> `docs/experiments/data/exp62_cross_pool_replay.py`, which refuses unless it reproduces the live
> gate-(ii) cosine 0.7874). v2 records the design RESULT — the pressure sensor has a real consumer and
> no measurable job on this encoder — and reshapes the experiment around the owner's actual question
> ("how the system learns with cross-context"): rung A measures the cross-pool transfer the shipped
> body already affords and closes Exp 60/61's "fears water anywhere" line; rung B names the context
> wall it does not cross and what crossing it would take. Nothing is built; runs after R3-cal has
> frozen its gauntlet, on the same rig. Owner decisions D1–D4 at the end.**

## The design result from the replay (recorded, not built around)

`docs/experiments/data/exp62_cross_pool_replay.py`, on the live Exp 60 gate-(ii) vectors:

- **There is no cache wall between two dark pools.** For the shipped roster, cos(pool-1 submerged,
  pool-2 submerged) stays in **0.92–1.00** over the whole buildable band (floor altitude 5–123, spawn
  distance 45–90; threshold 0.85). The two absolute features carry gain weight 0.09 (`y_altitude`) and
  0.16 (`distance_from_spawn`) against three shared full-swing cues at 1.0, 1.0 and 0.77
  (`is_in_water`, `light_level`, `time_of_day`). Corollary 7 in the flesh: a same-side move of a
  low-mass sensor never opens the ≈ 32° the threshold needs. v1's "floor" arm was predicted at the
  ceiling; the only placements that miss (both absolutes at their extremes) make pool 2's OWN
  shore/water separation marginal and fail Exp 60's gate (ii). v1's D1 had no solution.
- **The `pressure` sensor cannot move that number.** Declared `[-12, 12]` at the only reachable depth
  (4 blocks of water above the eye; depth is pinned by the US-free window) it carries weight **0.037**
  and moves every cosine by < 0.001. At any range it is a SHARED component of both pools and can only
  RAISE cross-pool similarity (0.9991 → 0.9993 at full weight). The roster already carries a
  surface-relative full-swing cue: `is_in_water` is computed from the EYE block. One honest side
  effect, recorded: at full weight (`[-4, 4]`) it sharpens pool 1's own shore/water separation (0.787
  → 0.679) — a within-pool gain, not a cross-context one. "Absolutes at gain 0" (v1 arm 4) is an
  identity (cos 1.0000) and is not declarable (gain is per modality; a child body cannot delete a
  parent's sensor).
- **The levers with mass are not place.** A lit surface pond reads **0.588** to the cave pool for every
  body (pressure included); a night pool at time 0.99 reads **0.799**. Light and time are the context
  axis this body places by. Crossing them is graded cue similarity — the exact-key read's limit (R1),
  Phase 4's job — and a sensor is not the remedy.

**What this means for the pressure idea (owner, 2026-09-17).** The categorization stands (a
transform of an exposed quantity is the eye-height class, not a synthetic world fact). What the
review adds is the front-gate answer: a sensor enters when an experiment names the contrast it must
move, and no cross-context contrast exists for it to move. The DRIVE version (a homeostatic
`pressure` with its set-point at the surface) stays deferred with a sharpened bar: Minecraft deals no
pressure cost, so its pain would be hand-coded (D1's pain half); its failure mode `drive:pressure` is
OUTSIDE the Wire-4 allowlist, so it would write no fear without an allowlist change that re-fires Exp
60 and 61; and biologically air hunger, not baroreception, drives surfacing — which `oxygen` already
senses. If it is ever built, it is a variant body (`extends: bodies/minecraft_player`), never the
shipped one (re-tagging every world node and firing two ledger triggers by their letter while the
fingerprint shows no drift), with its range declared on the reachable depth and NO golden regen (no
encoder function changes).

## Rung A (build) — the shipped body carries the drowning-fear across pools

**Claim (bounded).** An agent that learned the drowning-fear in pool 1 (Exp 60's protocol) leaves the
water on its first loop-live submersion in pool 2 — a pool at a different altitude and spawn
distance, same sealed-shell class, same frozen day, same depth — with the shipped `minecraft_player`
body and NO mechanism change, because the survival body's feature-first world channel makes the
water situation a FEATURE (`is_in_water` at full weight) and not a PLACE (the absolutes at ≤ 0.16).
The claim is "invariant across altitude and spawn distance inside the sealed-shell, frozen-day
apparatus class", never "anywhere". It closes the line Exp 60 and Exp 61 both left under §Not
claimed, and it is a property of the SHIPPED body — the 1.3 §World topology thesis ("portability
comes from the body") measured.

**Mechanism read.** The loop-OFF representation gate at pool 2 (Exp 61 step 4 verbatim): the pool-2
submerged reading must resolve to the SAME node id the training booked fear on at pool 1, with
water fear at the cap and shore fear 0 on pool 2's shore node. That row is the result; the first
contact is its behavioural consequence.

**Apparatus.** Pool 2 STACKED under or over pool 1 at the same x/z (recommended floor y 95: cross
0.999, own shore/water 0.79, inside the ≤ 90 spawn bound, inherits pool 1's Exp 58 clearance, no
chunk-load window on the teleport). Both pools depth 5. Builder and check made two-pool: `--shore-y`
and `--anchor-file` on `water_classroom` and on the water check (today one fixed anchor path is
OVERWRITTEN on build, destroying pool 1's `measured` block the harnesses refuse without); a
pool-vs-pool shell clearance guard; world spawn stamped into both records (the bridge emits
`bot.spawnPoint` once; nothing exposes it today); the flee anchor recorded per pool; both apparatus
checks and both `measured` blocks frozen. `WaterTrial` per pool (two `geom`s; ONE instrument attach
— the executor spy must not double-wrap). Light and time GATED equal at pool 2's shore and floor (the
full-weight constants; a stale-light read at a freshly filled box is the known failure), and pool 2's
own gate (ii) run live before any row.

**Arms.** One fresh agent per row; the fear learned in pool 1 by Exp 60's training (propose-only,
rescued, no post probe: zero escape links at the boundary, asserted):

| Arm | Trained | Read + first contact | n | Predicted |
|---|---|---|---|---|
| 1 **cross** | pool 1 | pool 2 | 12 | fires; representation gate on the trained node |
| 2 **same** | pool 1 | pool 1 | 12 | fires (the ceiling; Exp 60's result on this apparatus) |
| 3 **cross, fear-ablated** | pool 1, subscriber detached | pool 2 | 3 | censored, zero calls (anti-vacuity: the apparatus does not surface an agent by itself) |

**DVs and gates** (Exp 60/61's, in the US-free rescued window — nothing learns inside the read):
decision DV + behavioural DV at first contact; NODE gate (arm 1's pool-2 reading resolves to the
trained node id, loop OFF, ≥ 11/12 — the mechanism read, and the one gate that can fail for a reason
the replay could not see: live light/time, the EC's live thresholds); CROSS ≥ 0.70; SAME ≥ 0.70;
ANTI-VACUITY (arm 3 at 0/3 with zero calls); REPLAY row present and the live NODE outcome consistent
with it (offline-hit / live-miss is INCOMPLETE with the disagreement named; a predicted hit that hits
is the RESULT). Fisher exact where a contrast exists; latencies with intervals as the informative
number (both fear arms are predicted at the ceiling, so the statistic is the interval, not p).

**Live pre-check before the harness (four rows, ≈ 20 min):** world spawn coordinates; light and
time at pool 2's shore and floor after build and after a client reconnect; pool 2's own gate (ii);
ONE shipped agent trained at pool 1 and read loop-OFF at pool 2. That last row decides the rung
before a harness exists — if the reading does not resolve to the trained node, the replay was wrong
about something live, and that is the finding.

**Budget.** 27 trainings × ≈ 3 min + 27 probes + two apparatus checks ≈ 2 h, after the two-pool
plumbing PR.

## Rung B (design, not built here) — the context wall and what crossing it takes

The measured wall is lighting and time, not place: the same fear at a lit surface pond (0.588) or a
night pool (0.799) is a cache miss for every body. That is a genuinely different situation to a
feature-first body, and crossing it is graded cue similarity: a similarity-weighted or hierarchical
read at the cluster boundary (Phase 4, `roadmap_1_3.md`; R1's designed remedy), or a decision about
WHICH constants belong in the world channel (`light_level` and `time_of_day` carry `rest: null` by
design — the roster the rest lint prints every run). Rung B's prereg is written after rung A lands,
with its own front-gate: whether a generalization channel is a new mechanism (it is) and what
experiment earns it. A pressure sensor is orthogonal to this axis (it adds shared mass; light rotates
the vector) and is not proposed for it.

## What this experiment does NOT claim

- Nothing about a pressure sensor or drive (the design result above; the drive's bar recorded).
- Nothing "anywhere": not lit water, not night, not another depth, not another body.
- Nothing about a generalization channel (rung B); the read stays exact-key throughout.
- Nothing about cross-AGENT transfer of a cross-pool fear (Exp 61's path is unchanged and would
  carry it; a one-row check is a secondary).
- Nothing about extinction (Exp 61's recorded gap, sharpened: a fear reached by shared mass is
  reached at every same-class pool and nothing discounts it).

## Owner decisions (TAKEN 2026-09-18 — D1 y 95; D2 as tabled; D3 YES; D4 deferred, its 1.4 path recorded)

- **D1 — pool 2's placement:** stacked, floor y 95 (recommended) or y 10; both replayed as hits;
  y 95 keeps pool 2's own separation comfortably clear (0.79 vs the deep band's marginal 0.83). **TAKEN: y 95.**
- **D2 — n:** 12 / 12 / 3 as tabled. Recommended. **TAKEN: as tabled.**
- **D3 — run rung A at all?** It is predicted at the ceiling. Recommended YES: it turns a §Not-claimed
  line in two EARNED experiments into a measured, bounded result with a live representation read, it
  measures the §World topology thesis, and it costs ≈ 2 h plus plumbing the two-pool world needs
  anyway (rung B and any future cross-context work stand on it). Its `docs/wiring/` entry: *a place
  sensor only place-keys a fear near an extreme; near its rest it is silent by design.* **TAKEN: YES — rung A runs.**
- **D4 — the pressure drive:** deferred with the bar above (a game-native pressure cost must exist;
  the allowlist change re-fires two ledger rows). Recommended: deferred, not scheduled. **TAKEN: deferred; its 1.4 path is recorded in [roadmap_1_4.md](../plans/roadmap_1_4.md) §Pressure — no sensor in E1 (the depths must stay one cluster), and pressure/depth is a candidate INPUT to the graded predictor if E3 names one, entering only through that plan with the replay as its evidence.**

## Build order

1. Fold recorded (this v2); owner D1–D4.
2. Two-pool plumbing PR (builder + check + anchor records + spawn stamp + flee anchor + `WaterTrial`
   per pool); two-lens code review.
3. The live pre-check (four rows) on big-mac-mini, recorded as a diagnostic; v3 replaces
   "predicted" with measured.
4. Harness PR (`exp62_run.py` on `WaterTrial`, three arms, the NODE gate, the replay row); two-lens
   code review; one-pair dry run; freeze; campaign after R3-cal; merge-commit data PR; §Outcome.
