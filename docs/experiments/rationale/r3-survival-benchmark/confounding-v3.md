# R3 survival benchmark — CONFOUNDING lens, v3 DELTA re-run (2026-09-17)

Reviewed: `docs/experiments/r3_survival_benchmark_prereg.md` DRAFT v3 (`e78dc2e9`, branch `r3/v3`),
against the pilot rows `docs/experiments/data/r3_pilot_2026-09-17.jsonl` (five rows, hash `5d1e6a62`,
clean tree, big-mac-mini) and the pilot script `scripts/survival_world/r3_pilot.py`. This is the
re-run the v1 review asked for; the predecessor report (`confounding.md`, same dir) is NOT
re-derived — its V1–V6 and V8–V9 stand, and the pilot CONFIRMED V2 (in-situ Wire-4 escape at 8.4 s)
and V3 (innate `health → threat` escape after damage, alive) on live rows. Its V7 is now OBSOLETE
(see V17). Charter: `docs/experiments/DESIGN_REVIEW.md` — does the metric isolate the claimed cause;
could a positive or a null arise for another reason; controls; a statistic matched to the baseline;
floors/ceilings. Fed by `docs/wiring/sensor-range-clamps.md` (the saturation clamp is the crux of the
one new confound below) and `docs/wiring/substrate-learning-channels.md` (the state-blind link).

Short version: **v3's numbers hold up where the pilot measured them, and the pilot reveals ONE
confound v2 could not see — the regeneration-on death edge and the floor arm's margin are funded by
a saturation reservoir that D4's `heal` injects (true value ≈ 20) and that the instrument cannot
observe (the bridge clamps the sensed value at 10).** "Regeneration is the coarse lever" is the
binary form of "the injected reservoir is the lever". Beyond that: v3 quotes one carried-fear
latency (≈ 2.5 s) that its own cited data contradict (3.15 ± 0.20 s, n = 12), states a regen-off
margin (≈ 2 s) from an unmeasured row whose measured neighbour derives to ≈ 3.3 s, calls a per-arm
CONSTANT (`escaped_before_damage`) "unsaturated", and would spend a 36-agent sweep confirming a result
the pilot arithmetic already gives to ≈ 5 SD. And the pilot's `threat` column is `None` on every
proposal by a harness key-path bug — arm A's route is attributed by ELIMINATION, not by a read.
None of this is a design-killer; all of it is fix-before-build. Verdict line at the bottom.

## Verified first (from the rows, not from the prereg's summary of them)

**V10 — The regen-on margin is the saturation reservoir's, and the instrument cannot see it.**
`lethal_A` samples: health holds 18–20 from the damage onset (16.15 s) until ≈ 25 s while the SENSED
saturation falls 10 (20.0 s) → 9 → 8 → 5 → 3 → 1 → 0 (25.11 s); the moment saturation reads 0, health
goes 14.8 → 13.0 → 11 → 9 (fast saturated regeneration stops; the slow food-funded regeneration
≈ 0.25 hp/s remains — `drown_on` loses 12.8 hp over the next 7.0 s, 1.83 hp/s net of the game's
2 hp/s). `drown_on` shows the same shape (health 20.0 at 19.96 s, then the same descent, death
32.99 s). The reservoir is D4's `WaterTrial.heal`: `effect give … saturation 1 10` for 10 s → food 20,
TRUE saturation 20 (the effect adds 2 saturation/tick for 200 ticks; capped at food). The body
senses it at the bridge clamp 10 (`minecraft_player.yaml` saturation comment: "a fed bot holds
10–20 (the bridge clamps to 10)"), so every sample between the heal and 20.0 s reads 10 while the
game spends 20 → 10 underneath. **Consequences:** (i) the "5.2 s margin" is `death(reservoir) −
surface(reservoir)`; both edges move with the injected value — a game-native start (respawn
saturation 5, which the pilot's respawn sample shows) would shorten the hold by ≈ 6–7 s and put
arm A's escape near ≈ 21 s and death near ≈ 26–27 s; (ii) `health lost` for arm A (11) is NET of
regeneration and so is also a function of the reservoir; (iii) the fingerprint/roster cannot
verify the quantity that sets the floor arm's number — the pilot row records `saturation: 10.0`
at t = 0.4 s, which is the clamp, not the state. Note also that the heal's 10 s effect is still
ACTIVE for the first ≈ 8 s of the event (heal → `loop_warm_s` 1.0 → teleport); harmless here
(damage starts at 16 s) but it means the start state is "under a running effect", which D4 should
say. With regeneration OFF the reservoir is inert (no fast-regen drain; `drown_off` needs no
saturation to explain its 2 hp/s line), which makes the regen-off floor the CLEANER one.

**V11 — The pilot did not read the drive that fired; arm A's route is attributed by elimination.**
`first_proposals` carries `None` in the `threat` slot on all 10 proposals across both lethal rows.
Cause: `water_trial._telemetry_ticks` filters `r.get("drives")` for keys in `("threat", "oxygen",
"health")`, but `substrate_telemetry._drive_snapshot` writes the envelope `{"available", "drives",
"sensors"}` under that key — the values sit one level down — and even the inner dict holds RAW
sensor values (`vital_metrics`), never the derived corrective need that `_read_drive_states`
emits. So no row in Exp 60/61/pilot has ever recorded the need that scored a proposal through this
path; Exp 61 attributes through `RecommendCapture`/`decision_events` instead (its transferred rows
read `decisive: True`), which the pilot did not attach. The elimination for arm A is sound —
subscriber detached (`detached_count` 1, `fear_after {}`), `links_before` 0, `_DRIVE_CORRECTIVE_NEEDS`
has NO oxygen needle (verified: `temp`, `thermal`, `food`, `health` only — which is exactly why
17 oxygen-pain publishes at intensity 1.0 moved nothing for 25 s), food 20 → no hunger need; the
only mapped drive in deficit at 25.11 s is health (13.0 → deviation −7 < −6 → need `min(1, 7)` =
1.0) — but it is an inference, and the one coincidence it cannot rule out by itself is V10's:
saturation reached 0 in the SAME sample the health crossing appeared, because both are the
reservoir's exhaustion. Saturation carries no drive block (verified in the YAML), so the inference
survives; the R3 harness must not rely on it.

**V12 — The carried-fear latency is 3.15 ± 0.20 s (n = 12), not "≈ 2.5 s".** Exp 61 transferred
receivers (`exp61_pairs.jsonl`, fresh receiver, ingested −0.75, zero links asserted, loop ON):
`latency_s` 2.70–3.53, mean 3.15, SD 0.20; decomposed: flee call 0.97 ± 0.09 s (the first tick
after the teleport), escape call 1.69 ± 0.12 s (the flee costs 0.72 s to fail through the
executor), escape → air 1.46 ± 0.12 s. Exp 60 FEAR placement 1 (n = 5, fear −1.0 + shore-roam
links): 3.19 ± 0.16 s. Both v3 sentences that say "≈ 2.5 s" (§Pilot consequence 2, §Goldilocks)
are 3 SD below the measured value; the "1.3–3.3 s" range in the arms table spans the link-only
placements 2–6 of Exp 60 (1.3–2.3 s, where `escape_water` was already being called at t = −0.7 s
before the teleport) — those are not carried-fear numbers.

**V13 — The same-tool cap is `count > 5`: five identical proposals execute, the SIXTH is DROPPED
(`ctrl.pending_proposal = None; continue`) and the counter resets.** `agent_loop._MAX_CONSECUTIVE_SAME_TOOL
= 5`. Both drown rows show exactly this: `drown_on` escape calls at 7.16 / 15.28 / 23.38 / 31.50 /
33.67 (five), a silent tick, then 35.47 / 36.23 / 37.02 / 37.76 / 38.48 (five); `drown_off` 7.49 /
15.70 / 23.81 / 25.63 / 26.65, gap, 28.17 / 28.96 / 29.75 / 30.55 / 31.32. The counter is
content-keyed and resets on a different tool (the `flee` before every first escape). The cap
counts proposals, not outcomes — a capped "still submerged (capped)" with `ok: true` counts the
same as "surfaced".

**V14 — "Surfaced" is a ≈ 1.1 s transient, and the 2 s linger books links the event did not.**
`escape_water` (`index.js`) holds jump for `SURFACE_HOLD_MS` 600 after the head clears, then
releases; the bot sinks back. `lethal_A`: head clear 27.84 s, back in water 28.98 s (1.14 s);
`lethal_B`: 8.38 → 9.50 s (1.12 s); apparatus `w5_sinkback_s` 2.06–2.23 s from the escape CALL.
Inside the linger each row logged two more `escape_water` successes (A: 28.44 "already at surface"
— immediate, no jump hold; 29.36 in water again), and `links_after` = 7 in both rows versus 3 in
every Exp 61 transferred receiver after its single successful escape. So ≥ 4 of the 7 links are the
linger's, not the event's.

**V15 — The loop runs at 1.4–1.8 Hz against `FROZEN["loop_hz"] = 4.0`.** 49 ticks in ≈ 31.8 s
(`lethal_A`), 17 in ≈ 12.2 s (`lethal_B`), 107 in 60 s at rest; Exp 61 tick rows are 0.54–0.67 s
apart. A blocked executor call (flee 0.70–0.77 s; escape 1.2–1.8 s; a capped escape 8.0 s) freezes
the loop for its duration. The tick phase — uniform on ≈ 0–0.65 s, SD ≈ 0.19 s — is the largest
single variance term in every arm's latency, and it is the INSTRUMENT's, not the agent's.

**V16 — Two ascent rates are in play.** Bridge-side difference method: 0.428 s/block (1.809 s from
the floor, 0.953 s from floor + 2). Executor-side escape → head clear from the same floor: 1.23 s
(A), 1.35 s (B), 1.46 ± 0.12 s (Exp 61), 1.65 ± 0.19 s (apparatus w4) — i.e. 0.31–0.41 s/block over
the 4 blocks above the eye. v3's depth arithmetic uses 0.43. For the regen-on question the two
agree (all-survive either way); for a regen-off sweep they differ by ≈ 0.8 s at depth 12, which
is the whole margin (V18 below).

**V17 — Exp 61 is EARNED.** `exp61_verdict.json` (hash `4e25b475`, campaign `exp61-campaign-1`):
transferred 12/12, isolated 0/24, cluster_not_fear 0/12, dangling 0/24, Fisher p = 8e-10, all six
checks true, `refused: []`. The predecessor's F4 ("arm 3 must be conditional on its verdict") is
satisfied; v3's status text does not yet say so.

**V18 — The regen-off margin derives to ≈ 3.3 s, not ≈ 2 s, and it is unmeasured.** `drown_off`
health: 14 at 18.09 s (deviation −6, NOT < −6 → no need), 12 at 19.27 s (first sample in deficit,
need 1.0). The pilot's measured crossing → air lag is 2.73 s (A: crossing 25.11 → flee call 25.85
[0.74, one tick] → escape 26.61 [0.76, the flee's failure] → head clear 27.84 [1.23]). 19.27 +
2.73 = 22.0 s against death at 25.32 s: margin ≈ 3.3 s at depth 5. Extra ascent for depth 8 / 10
/ 12 (+3 / +5 / +7 blocks) at 0.31–0.43 s/block leaves ≈ 2.0–2.4 / 1.2–1.8 / 0.3–1.2 s: the
regen-off sweep lands near depth 11–12 IF the lag distribution is as narrow as n = 1 suggests and
IF the deeper ascent rate is the shallow one. With regeneration on: 33.0 − 27.8 = 5.15 s at depth
5, 2.1–3.0 s at depth 12, against a lag SD of ≈ 0.35 s (tick phase 0.19, flee 0.03, ascent
0.12–0.19, crossing 0.1 — A 25.11 vs `drown_on` 25.24 — added in quadrature) → the floor arm
survives depth 12 by ≈ 6 SD.

## Findings

### DO-NOT-BUILD

None as v3 stands. The two premises v1 got wrong are now measured, and the measured shape is the one
the predecessor's salvage asked for. Everything below is fix-before-build.

### SHOULD-FIX

**F17 — Pin the reservoir: the floor arm's frozen number depends on an injected quantity the
instrument cannot see.** Failure scenario: R3-bench freezes arm A at "27.8 ± 0.4 s, 11 hp lost,
survives"; a later `heal` change, a bridge clamp change, or a game-native start (saturation 5)
moves the number by ≈ 6 s and the drift is read as an agent/loop regression, or a future arm that
happens to start with less saturation shows a "worse floor" that is the reservoir's. Evidence: V10.
Fix: read the true `foodSaturationLevel` (and `foodLevel`, `foodExhaustionLevel`) over RCON at the
event teleport with the `data get entity <bot> …` pattern `common.bot_pos` already uses; record it
per row; put its accepted value in the gauntlet file beside `naturalRegeneration`; state in D4/D5
that the regen-on margin and arm A's `health lost` are net of a reservoir the heal sets, and that
regen-off is reservoir-free. Add the sensed-vs-true saturation gap to `docs/wiring/sensor-range-clamps.md`
(it is the same lesson: a clamped world sensor cannot verify a world state above the clamp).

**F18 — Rewrite the calibration statement: R3 is calibrated on `t_surface`; `survived` is a ceiling
with regeneration on at every buildable depth; `escaped_before_damage` is a per-arm CONSTANT.**
Failure scenario: v3 says "the PRIMARY DVs are already unsaturated across the arms" and cites
`escaped_before_damage` 0 / 1 / 1 — but A/E can NEVER escape before damage (the route fires after
it) and B/C/D ALWAYS do at every depth ≤ 12 (B's 8.4 s + ≤ 3 s < 16.2 s), so the binary is
all-0 or all-1 in every arm at every cell; Fisher on it (12/12 vs 0/12, p = 4e-7) tests the ROUTE
ORDER, which the mechanism fixes, and a reader takes the p-value as a measured effect. Meanwhile
R3-cal as written walks depth 5 / 8 / 12 on arm A (36 agents, two column rebuilds, two re-run
apparatus checks) to learn that 36/36 survive — V18 gives that to ≈ 6 SD from the pilot alone —
and the accept rule cannot land by construction; v3 says so ("likely") and runs it anyway. Also
`minecraft_benchmark.md` §R3 forbids a saturated instrument on "the effect it exists to measure";
if that effect is survival, R3 with regeneration on is structurally the ceiling §R3 names, and
naming the arm-A ceiling "not a failed instrument" does not change that. The honest statement,
which I recommend v3 adopt verbatim in spirit:

> R3 measures the COST of one lethal-window event, calibrated on `t_surface`. The instrument's
> range is the window from the first loop tick (≈ 1 s) to death (25.3 s regen-off / 33.0 s
> regen-on at the D4 reservoir); each arm's route fires at a mechanism-fixed onset inside it
> (C/D ≈ 3.15 ± 0.20 s, n = 12; B ≈ 8.4 s, n = 1, predicted SD ≈ 0.35 s; A ≈ 27.8 s regen-on /
> ≈ 22 s regen-off derived, n = 1) and depth adds a common ≈ 0.3–0.43 s per block to EVERY arm;
> no contrast is a function of depth. `survived` is a ceiling in every arm at every buildable
> depth with regeneration on (floor margin ≥ 2.1 s vs SD ≈ 0.35 s) and is reported as such;
> `escaped_before_damage` is a route-order flag (A/E ≡ 0, B/C/D ≡ 1) and is reported, never
> tested. The gauntlet is the Exp 60 depth; R3-cal is ONE verification cell (depth 5, arm A,
> n = 12) whose product is the floor arm's latency and margin DISTRIBUTION and the loop's
> tick-period distribution — the instrument's own variance, which the frozen file needs and the
> pilot's n = 1 cannot give. A lethal floor is a declared regen-off gauntlet, reservoir-free,
> whose margin (≈ 3.3 s derived, V18) is MEASURED before its cells are declared (F20).

And say plainly what "what the drive buys" then means on this world: ≈ 24 s of latency, 11 hp and
≈ 22 s of oxygen pain at intensity 1.0 — not life; the innate reflex plus regeneration keep the
floor alive.

**F19 — Read the drive that fires; do not infer it.** Failure scenario: an arm-A row escapes at
27 s and is booked as "innate health reflex" while the proposal was in fact scored by something
else (a leaked link, an explore term, a future need mapping) — indistinguishable by elimination,
which is all the pilot could do (V11). Fix: fix `_telemetry_ticks` (read `r["drives"]["drives"]`)
AND add the derived corrective needs to `_drive_snapshot` or attach Exp 61's `RecommendCapture` to
every R3 event; make "the executed escape is `drive`-decisive with a `threat` component read from
the capture" a REFUSAL condition for the anti-vacuity arm-A and arm-B rows (v3 already asks the
arm-C row for the object read paired with the behaviour; A and B need it more, their attribution
being the one the pilot could not make). The pilot rows should be re-labelled "attributed by
elimination" in §Pilot.

**F20 — Correct the stated numbers, and name what a regen-off sweep must measure before it is
declared.** Failure scenario: the gauntlet file inherits "carried fear ≈ 2.5 s" and "regen-off
margin ≈ 2 s"; the first is refuted by the campaign's own arm C (3.15 s) and read as a slow-down;
the second declares depth cells on a margin 1.3 s smaller than the derived one and either
over-shoots the middle or misses it. Evidence: V12, V18, and §Pilot's "health crosses 14 at
≈ 25.5 s" (the row: 13.0 first sampled at 25.11 s) plus "first proposal at 26.6 s (tick clock)"
beside calls on the window clock (the proposal is 25.85 s on the window clock; the tick clock is
offset ≈ 0.75 s — `loop_warm_s` 1.0 minus the loop's start latency). Fix the four numbers. Before
ANY regen-off cell is declared, measure: ≥ 3 detached regen-off lethal events at depth 5 (crossing
→ air lag and its spread; the margin), the executor-side ascent from depth 8 and 12 (V16 — the
bridge rate and the executor rate differ by the whole depth-12 margin), and confirm the reservoir
is inert with regen off (`foodSaturationLevel` flat through the event).

**F21 — End the loop window at head-clear; account links by call time; define `survived`.** Failure
scenario: R3 reports `positive_escape_links` after the event as "the snowball" and the number is
2–3× the event's because the harness lingered, or varies with how long `loop.join` took. Evidence:
V14. Fix: `stop_event.set()` at the first sample with `is_in_water == 0`; record the call list;
report links from calls with `t ≤ t_surface` as the event's and any later call as post-event
(a tick in flight may still start one — record it, do not refuse); state that `survived` = "took
one breath by its own act" (the head clears for ≈ 1.1 s, then the bot sinks back; staying afloat
thereafter is the LINK's work, not the drive's, and R3 does not measure it).

**F22 — Freeze the loop cadence as an instrument constant.** Failure scenario: a later loop
change (the `_substrate_tick_due` family, Exp 60 Amendment 5/6) lifts the cadence from 1.6 Hz
toward the 4 Hz target; every arm's `t_surface` drops ≈ 0.3 s and the C arm's "improvement" over the
frozen baseline is read as a drive result. Evidence: V15. Fix: record the tick-period distribution
per event; put its median/IQR in the gauntlet file; refuse a bench row whose median tick period is
outside the calibration band (the Exp 61 drift-refusal shape, applied to the loop rather than the
apparatus).

### NIT

**F23 — Same-tool cap wording.** v3: "the loop's same-tool cap (6 identical calls) broke the chain".
Actual: five execute, the sixth identical proposal is dropped, counter resets (V13). Cannot bite at
depth ≤ 12: the first `escape_water` after the `flee` (which resets the counter) surfaces the agent
in ≤ 11 × 0.43 = 4.7 s < the 8 s hold, so the counter never exceeds 1 in a lethal event; it could
bite only on an obstructed column (the stone cap) or a non-ascending jump hold, both instrument
faults. Also note for the record that a capped escape freezes the loop for 8.0 s (call spacing
8.1 s in both drown rows) — in a capped-column fault the innate reflex gets one proposal per 8 s.

**F24 — Update the Exp 61 status.** EARNED (V17); cite `exp61_verdict.json` and its hash in the
arm-D row; the predecessor's F4 is closed.

**F25 — The in-water `flee` costs 0.70–0.77 s of executor round trip in EVERY arm** (A 0.76, B 0.77,
`drown_on` 0.77, `drown_off` 0.70; Exp 61 0.72 ± 0.05) — 23 % of arm C's latency — while the
shore preflight (0.015 s bridge-side) measures a different thing. Report the tie-break's cost as
a named component of `t_surface` (it is the largest arm-invariant constant after the tick phase),
and say the preflight bounds the ANCHOR path, not the event cost.

**F26 — Q1's n.** With B − C ≥ ≈ 4.7 s structurally (B cannot start before the oxygen-12 publish,
5.77–6.06 s sampled across the five rows; C starts at the first tick, 0.97 ± 0.09 s) and pooled
SD ≈ 0.3 s, a one-sided Mann–Whitney separates the arms completely at n = 4 per arm (p = 0.014);
n = 12 buys nothing for C − B — its value is the arm-A tail (instrument-fault rate: stale bridge,
a multi-second flee, an executor stall) and the arm-B acquisition-curve spread, which the campaign
should say is why 12.

## Answers to the brief's questions

1. *B vs C real?* Yes, on the mechanism: B's route needs the second oxygen-pain publish (oxygen 12,
   ≈ 5.8–6.1 s, a game constant × `pain_scale`; the first publish leaves fear −0.25 < θ), C's fires
   at the first tick (0.97 ± 0.09 s). Measured C = 3.15 ± 0.20 s (Exp 61, n = 12), not ≈ 2.5 (V12);
   B = 8.38 s (n = 1; decomposition 5.92 + 0.34 + 0.77 + 1.35). Structural gap ≥ 4.7 s vs SD ≈ 0.3;
   n = 4 per arm suffices (F26).
2. *Regen-off margin honest?* No — "≈ 2 s" is unmeasured AND does not follow from the measured
   neighbour: `drown_off`'s crossing (12 at 19.27 s) + the pilot's lag (2.73 s) → 22.0 s vs death
   25.32 s → ≈ 3.3 s (V18). Before a regen-off sweep is declared: ≥ 3 detached regen-off events at
   depth 5, the deeper ascent rate both ways, the reservoir flat (F20).
3. *Still Goldilocks per §R3?* Not on survival — with regeneration on it is the ceiling §R3 names,
   at every buildable depth, by ≈ 6 SD. The honest statement is that R3 is calibrated on `t_surface`
   (range: first tick → death; onsets mechanism-fixed; depth a common offset), `survived` a declared
   ceiling, `escaped_before_damage` a route-order constant; R3-cal becomes one verification cell on
   the floor arm's distribution; a lethal floor is the reservoir-free regen-off gauntlet with a
   measured margin (F18, text supplied).
4. *The 2.7 s lag and its spread?* One tick (0.74; phase U(0, 0.65) at the measured 1.4–1.8 Hz, not
   4 Hz) + the flee's executor failure (0.76, SD 0.03) + the ascent (1.23–1.65, SD ≈ 0.15); with the
   crossing's own jitter (SD ≈ 0.1, reservoir-driven) → SD ≈ 0.35 s. Regen-on margin at depth 12 is
   2.1–3.0 s → all-survive, predictably; the sweep is uninformative on survival and informative only
   on the instrument's fault tail (V15, V18).
5. *Linger links?* Yes: two extra successes in the 2 s, `links_after` 7 vs Exp 61's 3 per single
   escape; end the window at head-clear, split links by call time, and define `survived` as one
   breath — the head is clear for ≈ 1.1 s and then the bot sinks back in both rows (F21, V14).
6. *Cap at ≤ 12?* Cannot bite: `> 5` drops the sixth identical proposal; a lethal event's first
   escape follows a `flee` (counter 1) and surfaces in ≤ 4.7 s < 8 s (F23, V13).
7. *Else:* the saturation reservoir the instrument cannot see (V10/F17 — the delta finding); the
   `threat` column is `None` by a key-path bug, attribution by elimination (V11/F19); the loop runs
   at 40 % of `FROZEN["loop_hz"]` (V15/F22); two ascent rates (V16); Exp 61 EARNED (V17/F24); the
   in-water flee cost (F25); four wrong numbers in §Pilot/§Goldilocks (F20).

## What I verified

- Rows: all five `r3_pilot_2026-09-17.jsonl` rows in full — every 4 Hz sample of `lethal_A`/`lethal_B`
  (t, in_water, health, oxygen, food, saturation, y, deaths), the call lists with timestamps and
  errors, `first_proposals` (all `threat` = None), `first_pain`, `fear_before/after`,
  `links_before/after`, both `health_series`, the respawn samples, the apparatus row (rules incl.
  the `doDrowningDamage` parse error, cadence 0.1009 s, liveness 6, clusters distinct, flee
  preflight 0.015 s, both ascent timings, the 60 s rest window, provenance hash `5d1e6a62`, clean).
- Script: `r3_pilot.py` end to end — `live_window` (no rescue, `linger_after_stop_s` 2.0 on lethal
  rows / 6.0 post-death, teleport after the window), `sample_full`, `submerge_to`,
  `bridge_escape_timing`, `flee_preflight`, `cap_pool`, `set_regen`, the row builders and the
  `first_proposals` extraction.
- Code: `agent_loop._DRIVE_CORRECTIVE_NEEDS` (no oxygen needle), `_corrective_need_for`,
  `_read_drive_states` (derived-need max fold), `_MAX_CONSECUTIVE_SAME_TOOL = 5` and the `> 5`
  drop/reset block; `sem.corrective_need_intensity` (strict `< −comfort_band`, `min(1, |Δ|)`);
  `nac.recommend_action` components 1–3 and the `<= 0.5` floor, `anticipatory_threat_need` (θ 0.5,
  `>=`), `_DRIVE_TOOL_AFFINITIES["threat"]`; `substrate_telemetry._drive_snapshot` (envelope shape,
  raw values only) vs `water_trial._telemetry_ticks` (reads the envelope as the values);
  `water_trial.heal` (instant_health + saturation, 10 s), `rescue` settle (`SATURATION_REST` 10 =
  the clamp), `positive_escape_links`; `exp60_run.FROZEN` (`loop_hz` 4.0, `loop_warm_s` 1.0);
  `minecraft_player.yaml` (health/oxygen homeostatic set_point 20 band 6; saturation range [0, 20]
  rest 10 "bridge clamps to 10", no drive; food entropic 16/6); `index.js::escape_water`
  (`SURFACE_HOLD_MS` 600, 8000 ms cap, "already at surface" returns before any jump hold);
  `common.bot_pos` (`data get entity` pattern); `docs/wiring/sensor-range-clamps.md`.
- Data: `exp61_verdict.json` (EARNED, all checks), all 12 Exp 61 transferred `first_contact`
  rows (`t_flee_call`, `t_escape_call`, `latency_s`, `decisive`), Exp 60 FEAR placement-1 latencies
  (n = 5) and placements 2–6, `exp60_water_apparatus.json` (three cycles: pain edge, oxygen zero,
  damage onset, w4 escape, w5 sinkback); `minecraft_benchmark.md` §R3 ("structurally incapable").

**Not verified (other lenses / owner):** the exact Minecraft regeneration accounting (fast
saturated regen: 1 hp/0.5 s at 6 exhaustion per hp, requiring food 20 and saturation > 0; slow:
1 hp/4 s at food ≥ 18 — from game knowledge; the ROWS show the two regimes and their boundary at
sensed saturation 0, which is what the finding rests on); whether the Saturation effect's true
value is 20 or capped lower (the RCON read in F17 makes it a recorded fact); the swim-up rate at
depth 8/12 (environment lens — V16 says the two shallow measurements already disagree); whether a
deeper column's `y_altitude` keeps the water cluster id (the corollary-3 replay v3 already
schedules); anything bio-faithful about whether "one breath" is the right unit.

## Verdict

**FIX-THEN-BUILD.** v3's measured claims hold where they were measured (V2/V3 confirmed live, the
death/respawn seam, the cap, the ascent). The delta the pilot reveals is that the regeneration-on
floor is funded by an injected saturation reservoir the instrument cannot see (F17) — pin it with a
game-native read before anything is frozen. Then: state the calibration on `t_surface`, name
`survived` a ceiling and `escaped_before_damage` a route-order constant, collapse R3-cal to one
verification cell, and make a lethal floor a measured regen-off gauntlet (F18, F20); read the
firing drive instead of inferring it (F19); end the window at head-clear and account links by call
time (F21); freeze the loop cadence (F22); fix the four numbers (F20) and the Exp 61 status (F24).
None of these change the arms, the unit, or the purpose; all of them change what the frozen file
would otherwise silently encode.
