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

**Protocol clauses frozen with the harness (v3, 2026-09-20).** Five things the built harness does
that v2 did not authorize. They are recorded here BEFORE the first campaign row, and the harness
refuses rather than scores on each:

1. **The NODE gate has FOUR clauses, not three.** v2's mechanism-read sentence names same-node,
   water fear at the cap, and shore fear 0; it also says "Exp 61 step 4 verbatim", and that step's
   fourth clause is the PRODUCTION read — `anticipatory_threat_need` on the read pool's node must
   exceed the consumer's STRICT activation floor (> 0.5, not ≥). A node that resolves correctly but
   reads at or below the floor is dead at recall, so it is a gate failure, not a pass.
2. **ONE probe cap for both pools:** `min(pool 1, pool 2 pain edge) − margin`, not a per-pool cap.
   A per-pool cap would make the arms' first-contact latencies incomparable and put the cap
   difference inside the very contrast the rung measures. **Refusal condition:** if the two pools'
   pain edges ever diverge far enough that `min − margin` leaves no usable window at the slower pool
   (no arm can surface before the cap), that is an APPARATUS refusal, not a behavioural null.
3. **A SAME-arm campaign-drift gate** (last-quartile minus first-quartile median first-contact
   latency > 0.5 s → INCOMPLETE). The SAME arm is the within-pool ceiling: if it drifts across the
   campaign the apparatus drifted, and it would have moved the cross arm the same way. Arms are
   interleaved seed by seed so drift hits them equally.
4. **Gate (ii) is CITED, not re-run** — it is its own script, and citing a frozen artifact is
   stronger provenance than a fresh run. The citation is gated: each record is bound to its pool by
   the probe's own `y_altitude`, its `run_gate` must PASS, and **the two records must agree on
   `light_level` and `time_of_day` at both shore and floor**. A `run_gate` PASS alone is NOT that
   check: pool 2's first probe read `light_level` 1.0 and its reconnect read 0.0, **both records
   pass their own gate**, and citing the stale one puts the real cross-pool cosine at 0.5878 (a MISS)
   while the synthetic replay still says 0.9995 — the campaign would have measured the LIGHT
   contrast and published it as the pool contrast. The replay row additionally computes the cosine
   the two records' OWN probe vectors imply and refuses when it and the synthetic prediction fall on
   opposite sides of the threshold.
5. **The live context check is RECORDED, not gated.** Every row stamps the full-weight constants as
   the BODY sensed them at the training pool and at the submerged read at the read pool
   (`live_state.context_matches_training_pool`). The committed-record gate in (4) can be satisfied by
   a record that no longer describes the world; this is the same question asked of the world. It is
   recorded rather than gated because the record-side gate already refuses the known failure, and a
   second gate on a sensor this world reports unreliably would refuse rows for the instrument's
   noise. A mismatch is a finding for §Outcome.

**Also recorded per row (v3):** the MARGIN each read resolved by — `node_gate.read_margin` — beside
the node id. The gate turns on a threshold comparison and this apparatus runs ≈ 0.06 from that
threshold, so a row carrying only the id cannot say whether it resolved comfortably or by a hair.
Read-only; no gate reads it.

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

## Outcome (2026-09-20) — rung A EARNED, campaign `exp62-rungA-1`

**The shipped `minecraft_player` body carries a learned drowning-fear from pool 1 to pool 2** — a
pool at a different altitude (floor y 90 v 35) and a different spawn distance (62.13 v 70.04),
same sealed-shell class, same frozen day. No mechanism change, no new sensor, no ingest. This is the
1.3 §World topology thesis ("portability comes from the body") measured, and it closes the line
Exp 60 and Exp 61 both left under §Not claimed.

27 rows, **zero refusals**, one code hash.

| arm | n | first-contact | Wilson 95 % | NODE gate | t_first_air median (95 % CI) |
|---|---|---|---|---|---|
| 1 **cross** (train pool 1, read pool 2) | 12 | **12/12** | [0.758, 1.000] | **12/12** | 3.238 s [2.876, 3.327] |
| 2 **same** (train pool 1, read pool 1) | 12 | **12/12** | [0.758, 1.000] | 12/12 | 3.070 s [3.037, 3.155] |
| 3 **cross, fear-ablated** | 3 | **0/3** | [0.000, 0.561] | n/a — ablation check | — |

All five frozen gates PASS: NODE 12/12 (≥ 11/12), CROSS 1.0 (≥ 0.70), SAME 1.0 (≥ 0.70),
ANTI-VACUITY (0/3 surfaces, **zero** executor calls, ablation held 3/3), REPLAY (the committed
offline prediction — cross-pool cosine 0.9992, HIT — and the live NODE outcome agree). Fisher exact
one-sided, cross v ablated: **p = 0.0022**.

**The interval is the number, not p.** Both fear arms sit at the ceiling by design, so the
informative statement is the Wilson lower bound: **≥ 0.758** for the cross arm, not "100 %". The two
arms' latency intervals OVERLAP — the cross arm is nominally 0.17 s slower and there is no evidence
it is actually slower.

### What was measured along the way

- **Arm 3 is a clean VALENCE ablation, not a representation ablation.** The ablated agents' pool-2
  readings resolve to the SAME node as their training, with fear 0.0 and need 0.0. So the arm-1/arm-3
  contrast is the fear alone, with the representation held identical — stronger than the prereg
  claimed for it.
- **The apparatus ran ≈ 25 % under its cap** (median 3.24 s against 4.29 s) and the SAME-arm drift
  gate held across two hours.
- **The match margin (#786) has ZERO variance**: every cross row read 0.999209493831443, identical
  to 15 decimal places across 12 independent agents. `world` is frozen-centroid, so the first
  embedding to reach a node is the prototype forever; with a deterministic apparatus every agent's
  pool-1 prototype and pool-2 probe are the same vectors. A useful determinism check — and a
  confirmation that this campaign contributes **nothing** to the shape-of-the-landscape question
  (zero variance is zero information), which is why that is measured offline instead.

### What this does NOT say

- **It is bounded to the sealed-shell, frozen-day apparatus class**, exactly as pre-registered. It is
  not "fears water anywhere".
- **The context wall is untouched and still stands.** A lit surface pond reads 0.588 and a NIGHT POOL
  reads 0.799 against the 0.85 threshold — so the earned drowning fear **misses at night**, for
  representational reasons, and the frozen-day protocol is what hides it. Rung B owns that, and its
  entry condition no longer depends on this rung failing (`docs/plans/roadmap_1_4.md` Phase 5).
- **This is not a general generalization result.** The apparatus has ONE discriminating world sensor
  (`live_contributors: ["is_in_water"]`, a binary flip), so its situation space is two points. What
  transferred is invariance to the two low-gain place absolutes, which is what "portability comes
  from the body" predicts — not invariance to a changed situation.
- **Nothing about a pressure sensor.** It was refused for this rung at the design review and is not
  implicated either way.

### Provenance

Prereg v3 (the freeze, #787) was on `main` before the first data timestamp; the harness (#780) plus
its two follow-ups (#781, #782) and the margin (#788) were merged first; the campaign ran at one code
hash with a clean tree, `--write-experiment-results`. Records: `docs/experiments/data/exp62_rows.jsonl`
and `docs/experiments/data/exp62_verdict.json`.

## Rung B (design, not built here) — the context wall and what crossing it takes

The measured wall is lighting and time, not place: the same fear at a lit surface pond (0.588) or a
night pool (0.799) is a cache miss for every body. That is a genuinely different situation to a
feature-first body, and crossing it is graded cue similarity: a similarity-weighted or hierarchical
read at the cluster boundary (Phase 4, `archive/roadmap_1_3.md`; R1's designed remedy), or a decision about
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

**Status 2026-09-20 — steps 1–4 DONE through the dry run; this v3 IS the freeze.**

- Harness merged (#780) after a two-lens round that cross-confirmed a blocking defect: the light/time
  gate of clause (4) was unimplemented. Folded before merge, with its red gate running against the
  committed stale record.
- Two follow-ups: #781 (the replay takes its place absolutes from the gate records — `world_spawn` is
  stamped only when a pool is BUILT with `--spawn-x/y/z`, and both live pools were not) and #782 (the
  read-pool snapshot was taken before the bot moved, so it read the OLD pool — mutation-tested).
- **Dry run `exp62-dry-2`, n=1 per arm, all three CLEAN.** `cross`: NODE pass, same node, need 1.0,
  fear −1.0, surfaced drive-decisive at 2.952 s. `same`: NODE pass, 3.172 s. `cross_ablated`: no node
  gate (it has an ablation check), censored, ZERO executor calls. Light 0.0 / time 0.0417 at both
  pools live, `match: True` on every row, and the sensed distances (70.04 / 62.13) match the cited
  gate records to the digit. **n=1 earns nothing** — this is instrument validation, and the frozen
  arm sizes are 12/12/3.
- Measured property worth carrying to §Outcome: `cross_ablated` resolves to the SAME node with
  need 0.0 and fear 0.0. Arm 3 is therefore a clean **valence** ablation, not a representation
  ablation — the arm-1/arm-3 contrast is the fear alone, with the representation held identical.
  That is stronger than v2 claims for it.
- Operating margin to carry into the campaign: first-contact latencies ran 2.95–3.17 s against a
  4.29 s cap (≈ 25 % headroom, against 1.78 s offline). The clause-(3) drift gate is load-bearing
  over a two-hour run.
