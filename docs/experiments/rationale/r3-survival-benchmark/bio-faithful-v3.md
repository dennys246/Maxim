# R3 survival benchmark — bio-faithful lens, DELTA review of v3 (2026-09-17, after the pilot)

Charter: does the design test the mechanism's REAL job, not a caricature? This is the re-run the
v1 review asked for. The predecessor's report (`bio-faithful.md`) stands; nothing below re-derives
it. The delta question is whether the PILOT's measurements change what the mechanism's job looks
like, and whether v3 describes the measured mechanism faithfully. Read against
`docs/agents/bio-memory.md` §Wire 4, the v3 prereg (§Pilot, §Arms, §Dependent measures,
§Goldilocks, D5), the five pilot rows, `scripts/survival_world/r3_pilot.py`, and the code the
questions name (`sem.py`, `agent_loop.py`, `nac.py`, `body.py`, the bridge's `escape_water`,
`tool_dispatch.py`, `minecraft_player.yaml`).

The short version: **the pilot confirms the mechanism the v1 review described, in every row, to the
tick.** Both routes fire where the code says they fire; the acquisition curve is the Wire-4
arithmetic; the flee tax is one tick; the ascent is the ascent. v3 folds the measurements
honestly. The remaining faults are in how v3 STATES the measured mechanism: one number that is
wrong (the regen-off margin, stated two different ways, neither from the rows), a primary DV
that is a route-identity bit rather than a graded cost, and a handful of wording slips (the
crossing time, the snowball count, the same-tool cap). Nothing here is a DO-NOT-BUILD.

---

## DO-NOT-BUILD

None. The pilot removed the v1 premises and v3 does not reintroduce them.

---

## SHOULD-FIX

### SF-1 (Q4) — The regen-off margin in v3 is unmeasured AND self-inconsistent; the rows give ≈ 3.3 s, not ≈ 2 s, and the "lethal floor" sits at the builder's maximum depth

**Failure scenario.** The owner declares the regen-off sweep expecting depth to consume a 2 s margin
with 3 s of buildable ascent, runs it, and arm A survives every cell (or coin-flips only at depth 12).
The sweep is then read as "the instrument is broken" or, worse, someone reaches for a THIRD lever.

**Evidence.** v3 says three different things: §Pilot item 1 "≈ 2 s with it off (escape ≈ 23 s
predicted)"; §Goldilocks "innate health reflex … ≈ 21 s off"; D5 "≈ 2 s". None comes from the rows.
The rows fix the innate route's own cost precisely: in `lethal_A` the first sub-14 read was 13.0 hp
at 25.11 s, `flee` dispatched at 25.85 s (+0.74 s, one loop tick), `escape_water` at 26.61 s
(+0.76 s, the tie-break), head clear at 27.84 s (+1.23 s ascent from the floor) — **2.7 s from the
first sub-14 read to air**. `drown_off`'s curve (regen off, loop attached but capped) reads
< 14 hp first at **19.27 s (12.0 hp; the 2 hp/s clock skips 13)**. Applying the measured 2.7 s:
air ≈ 22.0 s, death 25.32 s → **margin ≈ 3.3 s**, health at air ≈ 6–7 hp. Consuming 3.3 s at
0.43 s/block is ≈ 7.7 blocks → depth ≈ 12.7, i.e. the builder's maximum (12) is a coin-flip cell,
not a mid-range one. So "regeneration is the coarse lever and depth the fine one" is right
qualitatively, but v3's claim that regen-off makes depth ABLE to cross the margin is at the edge
of the buildable range, not inside it. **Fix:** state the regen-off route timeline as PREDICTED
from `drown_off` + `lethal_A`'s measured 2.7 s (crossing 19.3 s → air ≈ 22.0 s → margin ≈ 3.3 s →
lethal cell at depth ≈ 12–13), give one number in all three places, and say the regen-off fallback
is expected to land only at the builder's limit or not at all. Also correct the regen-on crossing:
v3's "≈ 25.5 s" is between the 14.8 hp sample at 24.86 s and the 13.0 hp sample at 25.11 s —
**≈ 25.0 s** (`drown_on` reads 13.17 hp at 25.24 s, same edge).

### SF-2 (Q5) — `escaped_before_damage` is a route-identity bit, not a graded cost; the drives' currency (pain-seconds) is the DV that separates the arms, and v3 has it as secondary

**Failure scenario.** R3-bench reports its primary binary as 0/12 (A, E) vs 12/12 (B, C, D) in every
cell with a Fisher p that is trivially tiny and carries no information beyond "which route fired";
C − B and C − D on the binary are 0 vs 0; the graded quantity the benchmark exists to report lives
in a secondary nobody gates on.

**Evidence.** Arm A's route IS the damage (`health → threat`), so `escaped_before_damage` = 0 for A
by construction (v3 says so in R3-cal). Arms B/C/D at depth 5 have ≥ 7.7 s of slack to the damage
onset (B: air 8.38 s vs damage 16.15 s), which depth cannot consume (18 blocks). So the binary is
structurally determined by the arm, not measured. `health lost` is likewise 0 by construction for
B/C/D. What the rows DO grade is the integral of `drive_pain_for_value` — computed from the pilot's
own samples, left-Riemann on the actual stamps, cut at `t_surface`: **oxygen-pain seconds A 22.15 /
B 2.94 / C ≈ 0 (predicted: escape at ≈ 2.5 s precedes the 13-bubble edge at 5.3 s); health-pain
seconds A 2.15 / B 0 / C 0.** This is the mechanism's own currency: `drive_pain_for_value` is the
SAME function the relief credit runs on (`sem.py` docstring; `tool_dispatch` `drive_potential_diff`),
so pain-seconds is literally "the negative-reinforcement signal not paid" — the job Wire-4 exists
to do. Note also what `t_surface` measures for B: 5.3 of its 8.4 s is the pain-FREE descent (the
world's oxygen clock to the 13-bubble edge, a body parameter: comfort_band 6). So C − B on
`t_surface` (≈ 6 s) is mostly "not waiting for the pain edge", while C − B on oxygen-pain seconds
(≈ 2.9 s) is "the pain of learning it there" — both honest, different questions. **Fix:** keep
`t_surface` primary; demote `escaped_before_damage` to the route-identity/anti-vacuity check it is;
promote per-drive pain-seconds to primary beside `t_surface` (it is unsaturated, monotone across
routes, and bounded by the event); and write the DV → arm information map explicitly (binary and
health-lost separate A/E from B/C/D only; `t_surface` + pain-seconds carry C − B and B − A).

### SF-3 (Q5, harness) — Pain-seconds are NOT the pain publishes; the harness must integrate the sample series, and the pilot's series is 2.6 Hz with the linger inside `min_health`

**Failure scenario.** The harness reports Σ published intensities (or n_pain) as "pain-seconds":
B's 3 publishes vs A's 17 would read as a 5.7× ratio where the integral is 7.5×; a C-arm agent
with one publish at the edge would read 0.5 "pain-seconds" for what may be 0.1 s of pain.

**Evidence.** `body.py::evaluate_failures` re-publishes only when the breach severity grows by
> eps = 0.05 × band = 0.3 sensor units (`_BREACH_DEEPEN_FRACTION`), under PainBus's 0.5 s
refractory; a standing breach is silent by design. So publishes count DEEPENINGS: A's 17 = 14
oxygen steps (13 → 0 bubbles) + 3 health steps (13, 11, 9 hp); B's 3 = 13, 12, 11 bubbles. The
integral needs the sensor series. The pilot records it (`samples`: health, oxygen at each stamp),
but (a) the cadence is ≈ 0.38 s (`sample_full` does a synchronous RCON `deaths()` read per sample
on top of the 0.25 s sleep), not the "4 Hz" the prereg states; (b) `min_health` is taken over ALL
samples including the 2 s linger (in A the minimum happened to precede the surface, but a
re-submersion in the linger can take damage). **Fix:** the harness records the sensor series to
`t_surface`, integrates `drive_pain_for_value(spec, value)` per drive with Δt from the stamps, cuts
`min_health`/health-lost at `t_surface`, and either moves the `deaths()` read off the sample thread
or states the measured cadence (≈ 0.38 s → ≤ one-sample edge error, fine for 22 / 2.9 / 0, and
exactly 0 for C since the fear fires before the edge).

### SF-4 (Q2, Q6) — "Survived" must be defined as "alive at the first breath": the actuator is a breath, not an exit, and the bot re-submerges within ≈ 1 s of every surface

**Failure scenario.** "Survived 12/12" is read as "the agent got out of the water", when in this
world no agent ever leaves the column by its own act; the post-event teleport to shore is the only
exit.

**Evidence.** The bridge's `escape_water` holds jump until the eye block is air, then 600 ms more,
then releases (`SURFACE_HOLD_MS`; "A surface is a BREATH, not a tick"). The samples show exactly
that: A clears at 27.84 s (y 38.7), peaks y 39.7, and is back in water at 28.98 s (y 37.75, eyes
39.4 < the y = 40 water top); B clears at 8.38 s, back in water at 9.50 s. The breath is partial
(A: 0 → 3 bubbles; B: 9 → 13) and oxygen is already falling again by the last sample (A: 3 → 2).
The `is_in_water` "flicker" is REAL re-submersion, consistent with `y` — not sensor noise. The
pathfinder is dead in water, so there is no route to the shore; a bot that keeps proposing
`escape_water` bobs indefinitely (period ≈ 1.8–2 s, the call's duration + a tick), which is what
the linger shows. This is the world's affordance limit, not the drive's failure, and the prereg's
choice to END the event at the head clearing is exactly right for the mechanism's job (the fear's
job is to get the head to air before the US; that is what happened). **Fix:** define `survived` as
"alive at the first head clearing" in §Dependent measures, and say in §The event that surfaced ≠
out — the teleport after the window is the exit.

### SF-5 (Q6) — The snowball has three components; two are not the mechanism's job, and the harness must record the bridge `detail` per call to tell them apart

**Failure scenario.** `links_after` = 7 is quoted as "a successful escape was reinforced 7 times"
when some of those links were booked for a no-op, and the count is per context-hash, not per call.

**Evidence.** (i) Negative reinforcement of the escape that ended the pain IS the job. (ii) But
`escape_water` returns `"already at surface"` as a non-throwing success when the head is already
clear — the same hole the flee-anchor review closed for `flee` ("a goto-to-where-you-stand …
would book flight SUCCESS for doing nothing") — and A's second call at 28.44 s was dispatched
while the samples at 28.23/28.60 s read `in_water` 0: a success for doing nothing, booked as a
positive link. (iii) In A the innate need is state-blind: health 9 < 14 keeps `threat` = 1.0 for
≈ 20 s after surfacing (slow regen at saturation 0), proposing `escape_water` on land or at the
surface regardless of water. (iv) `NAc` links are keyed by (event, outcome, context-hash)
(`record_outcome_full`), so 7 links from 3 successful calls means differing contexts, not 7
reinforcements. **What is structurally sound and should be STATED in v3:** the first positive link
is booked at call RETURN, ≥ 600 ms after the head clears, so the event window (teleport → head
clear) contains zero positive escape links by construction — the "fear-only read" for the primary
DVs is guaranteed by the actuator's contract, not only by the boundary assertion. **Fix:** record
per call the bridge `detail` (`surfaced` / `already at surface` / `capped`) and dump
`links_after` with contexts and observation counts rather than a count; note that a linger-time
"already at surface" success is a no-op link. (Plugging the no-op hole in the bridge is a
mechanism change and out of R3's front gate; record it as a known bias of the post-event
snowball.)

---

## NIT

### N-1 (Q1) — State the innate route as a STEP at health < 14, with the measured tick, on one clock
`corrective_need_intensity` (homeostatic) returns `min(1, |dev|)` once `dev < −comfort_band`, so
the need is 0 at 14.8 hp and **1.0** at 13.x — a step, not the graded `drive_pain_for_value`
(0.5 at 13 hp). "< ~11" is refuted: 11 hp was first sampled at 26.27 s, after the route's first
act (`flee` at 25.85 s). Faithful sentence: "the innate route fires on the first read below 14 hp
(measured 13.0 hp at 25.11 s; `flee` one tick later at 25.85 s; `escape_water` 26.61 s; air
27.84 s), independent of `pain_scale`". Name the clock: the pilot's `first_proposals` are on the
tick clock (first loop tick ≈ 0.75 s BEFORE the teleport), `calls` and `samples` on the teleport
clock — v3's "26.6 s" for the route is the escape CALL, the route's onset is 25.85 s (flee).

### N-2 (Q3) — Carry the measured acquisition curve; the escape was driven at −0.75, not −1.0
`record_cluster_fear`: valence −= 0.5 × intensity, clamp [−1, 0]. Measured in B: 13 bubbles at
5.35 s → intensity 0.5 → **−0.25** (below θ 0.5); 12 bubbles at 5.92 s → 1.0 → **−0.75** (clears
θ); `flee` dispatched **6.26 s** (+0.34 s, the next tick); 11 bubbles at 6.65 s → **−1.0** (cap),
booked while the tie-break was in flight; `escape_water` 7.03 s; air 8.38 s. The predecessor's
schedule (−0.25, −0.5, −1.0) assumed 0.5 at both 13 and 12 bubbles; the body gives 1.0 at 12
(`min(1, (20−12−6)×0.5)`), so θ clears one publish EARLIER than predicted. Two consequences worth
one sentence each in v3: arm B ACTS on a need of 0.75 — the same magnitude as the Exp 61
receiver's discounted fear (arm D) — so B, C and D act on structurally identical needs and differ
only in WHEN the need exists; and the "fear −1.0 after" is the post-event state, not the acting
state. `n_pain` = 3 is the deepening count (SF-3), not the acquisition length (2).

### N-3 (Q2) — The first escape has no second failure mode; quote it as ascent + hold
26.61 → 27.84 s = 1.23 s to the sample that read air (3.7 blocks from the floor; the bridge's own
"surfaced" fires ≤ 0.4 s earlier at the sample cadence). The flee tax is one tick, measured
0.76 s in A and 0.77 s in B. The bridge-side ascent (1.81 s from the floor, `apparatus`) includes
the HTTP dispatch; the loop-side is 1.2–1.35 s. Nothing to fix; state the two numbers as one
quantity with its two clocks.

### N-4 (Q6, Q7) — The same-tool cap drops one proposal in six and RESETS; it never breaks a chain
`_MAX_CONSECUTIVE_SAME_TOOL` = 5; on the 6th identical call the loop drops that proposal, resets
the counter to 0, and `continue`s — the 7th executes. v3 §Pilot item 4 ("broke the chain … loses
proposals to that cap") should read "loses one proposal in six". Inside a one-event window the cap
is unreachable (the route's escape is call #2 after `flee`; five 8 s caps exceed the 45 s refusal).

### N-5 (Q7) — v3's snowball count is off by the rows
§Pilot item 4 says "3 more successes each; 7 positive links". The rows show **3 successful
`escape_water` calls TOTAL per row** (the one that surfaced + 2 in the 2 s linger) and a 5th
proposal past the window (31.6 s / 12.1 s on the tick clock). "7 positive links after one event" is
right as a link count (SF-5 on what it counts).

### N-6 (Q4) — The regen-on margin is fuelled by the INJECTED saturation; say so beside "the coarse lever"
`WaterTrial.heal` is `/effect … saturation`; the bridge clamps the read at 10. In A, health held
18–19 hp from damage onset (16.15 s) while saturation drained 10 → 5 (to 21.5 s), sagged 18.8 → 13
as it drained 5 → 0 (25.1 s), then fell at the raw 2 hp/s. Regen-on's +7.7 s of life and +5.8 s of
delayed crossing (net +1.9 s margin: 5.2 vs 3.3) are therefore a function of the start state D4
declares (game-native respawn is saturation 5). Constant across arms, so not a confound; but the
gauntlet file should record the saturation read at the teleport and D5 should say the lever's size
depends on it.

### N-7 (Q4) — What the depth sweep can and cannot move, in one sentence
Every arm pays the same 0.43 s/block; B's slack to damage (7.7 s) needs 18 blocks; so within
5 → 12 depth changes NO contrast and NO arm's binary — only arm A's health-lost (2 hp/s × extra
ascent) and, regen-off, arm A's survival. v3 already says A likely survives every regen-on cell and
freezes at 5; say plainly that R3-cal with regen on measures the floor arm's ascent cost, nothing
about the drive-bearing arms.

---

## Answers to the seven questions, in one place

1. **Threshold.** A step to need 1.0 at health < 14 (`min(1,|dev|)`, dev < −6); confirmed by A:
   13.0 hp read at 25.11 s → `flee` 25.85 s. "< 14" is the faithful statement; "< ~11" is refuted by
   the row. v3's crossing "≈ 25.5 s" should be ≈ 25.0 s (N-1, SF-1).
2. **Ascent.** 1.23 s from the escape call to the air sample = the ascent; no second failure mode.
   The re-submersion at 28.98 s is the actuator's breath contract (600 ms hold, release, sink), real
   not flicker; it is why the event ends at the head clearing, and why `survived` must mean "alive
   at first breath" (SF-4, N-3).
3. **Acquisition.** −0.25 (13 bubbles, 5.35 s) → −0.75 clears θ (12 bubbles, 5.92 s) → first
   proposal 6.26 s → −1.0 cap (6.65 s) while the tie-break is in flight. The predecessor's −0.5
   middle step assumed intensity 0.5 at 12 bubbles; the body gives 1.0. B acts at 0.75, the same
   need as D (N-2).
4. **Regeneration.** Keeping regen ON and reporting survival as a ceiling is the faithful choice —
   the primary and pain-second DVs are unsaturated at depth 5 with nobody dying, and regen never
   enters B/C/D's event at all. A regen-off sweep is a game-native kinetics setting, not the R2
   trap (R2 injected the DRIVE STATE; this changes the world's damage clock, not interoception), so
   the declared fallback is honest — but it calibrates the CONTROL's death rate, not the treatment's
   effect, and v3's margin number for it is wrong and inconsistent (SF-1, N-6, N-7).
5. **DVs.** `t_surface` and per-drive pain-seconds measure "what a carried drive buys" faithfully
   (seconds underwater; air-hunger not paid); `escaped_before_damage` and `health lost` are
   determined by the arm, not measured. Pain-seconds are computable from the rows' sample series
   (22.15 / 2.94 / 0 oxygen; 2.15 / 0 / 0 health), never from the latched publishes; the harness
   must keep the series to `t_surface` at its measured cadence (SF-2, SF-3).
6. **Snowball.** Reinforcing the escape that ended the pain is the job; the "already at surface"
   no-op success and the state-blind innate need after surfacing are not. Ending the window at the
   head clearing excludes all of it from the primary DVs BY CONSTRUCTION (the first link is booked
   ≥ 600 ms after the head clears) — a structural guarantee v3 should state (SF-5).
7. **Else.** The floor arm is a lesion of the fear subscriber AND of an air-hunger reflex the body
   deliberately lacks — A sat at oxygen 0 for 12.8 s with pain at 1.0 and 14 deepening publishes
   and no route; v3 names this correctly as "innate-only = health-innate only". The cap wording,
   the snowball count, and the clock naming are N-4/N-5/N-1.

---

## What I verified

- `docs/experiments/rationale/r3-survival-benchmark/bio-faithful.md` (predecessor; not re-derived),
  `docs/experiments/DESIGN_REVIEW.md`, the v3 prereg in full, `docs/agents/bio-memory.md` §Wire 4
  (θ above the 0.5 floor; max-not-sum with the innate need; no tick decay; encode before pain tick).
- The five rows of `docs/experiments/data/r3_pilot_2026-09-17.jsonl`, sample by sample: `lethal_A`
  (78 samples, mean Δt 0.381 s; first < 14 hp at 25.11 s = 13.0; `flee` 25.846 s, `escape_water`
  26.605 / 28.437 / 29.355 s all `success` True; air 27.839 s; re-submerged 28.98 s; 17 publishes;
  `fear_after` {}); `lethal_B` (28 samples; publishes 5.345/0.5, 5.918/1.0, 6.645/1.0; `flee`
  6.259 s; escape 7.027 s; air 8.377 s; re-submerged 9.50 s; `fear_after` −1.0 on one cluster);
  `drown_on` (first < 14 at 25.24 s = 13.17; death 32.99 s; saturation-fuelled plateau 18–19 hp
  16–21.5 s); `drown_off` (first < 14 at 19.27 s = 12.0; death 25.32 s); `apparatus` (ascent 1.81 /
  0.95 s → 0.428 s/block; flee preflight 0.015 s; rest 60 s zero drain; `naturalRegeneration true`;
  `doDrowningDamage` not a gamerule).
- Pain-seconds computed from the rows with `drive_pain_for_value`'s homeostatic formula
  (set_point 20, band 6, scale 0.5) by left-Riemann on the actual stamps to `t_surface`.
- `src/maxim/embodiment/sem.py::corrective_need_intensity` (homeostatic `min(1,|dev|)` past the
  band — a step) and `drive_pain_for_value` (graded); `src/maxim/embodiment/body.py`
  `_BREACH_DEEPEN_FRACTION` 0.05 / `_BREACH_MIN_EPS` / `_BREACH_HYSTERESIS` 0.2 and the
  `evaluate_failures` latch (re-publish only on severity > latched + eps).
- `src/maxim/runtime/agent_loop.py::_DRIVE_CORRECTIVE_NEEDS` (`health → threat`; no oxygen entry),
  `_read_drive_states` (max breach per need), the `threat = max(innate, fear)` combine in
  `propose_via_substrate`, the consecutive same-tool cap (`_MAX_CONSECUTIVE_SAME_TOOL` 5; drop,
  reset, continue).
- `src/maxim/decisions/nac.py::record_cluster_fear` (α 0.5, clamp [−1, 0], allowlist),
  `cluster_fear`, `anticipatory_threat_need` (θ = `DEFAULT_CLUSTER_FEAR_THRESHOLD`, deepest across
  active clusters), `record_outcome_full` (links keyed by event/outcome/context-hash),
  `get_positive_outcomes`.
- `src/maxim/_data/components/bodies/minecraft_player.yaml` (`health`/`oxygen` homeostatic 20/6/0.5;
  `food` entropic 16/6; the deliberate absence of an innate oxygen reflex).
- `scripts/minecraft_bridge/index.js` `escape_water` (eye-block `headWater`, 600 ms post-clear hold,
  8 s cap → "still submerged (capped)" returned as a non-throwing success, `"already at surface"`
  returned as a non-throwing success) and `is_in_water` (eye height 1.62; water top y = 40 from the
  samples); `flee` fails fast when submerged.
- `src/maxim/runtime/tool_dispatch.py` credit routing (`learn_success` = POSITIVE valence; drive
  relief vs tool-success; interoception cluster only).
- `scripts/survival_world/r3_pilot.py::live_window` / `sample_full` / `row_lethal` (calls on the
  teleport clock; `first_proposals` on the tick clock, offset measured 0.74–0.76 s; `min_health`
  over the whole window incl. the 2 s linger; `n_pain` = publishes in the window; synchronous
  `deaths()` per sample) and `water_trial.py` (`heal` via `/effect saturation`; `positive_escape_links`
  is a link count; `_detach_fear_subscriber`).
- Not verified live: the regen-off route timeline (predicted here from `drown_off` + `lethal_A`, as
  v3 also says it is unmeasured); the column geometry beyond what the `y` series shows (the bot
  re-submerges; no exit is reached).

## Verdict

**FIX-THEN-BUILD (v3 is faithful to the measured mechanism; fix the stated numbers and the DV
map).** The pilot confirms both routes, the acquisition curve, the tie-break and the ascent to the
tick; v3's fold is honest, including regen-on-with-survival-as-ceiling. Before freezing: give ONE
regen-off margin derived from the rows (≈ 3.3 s, lethal only at depth ≈ 12–13), promote per-drive
pain-seconds beside `t_surface` and demote the binary to a route-identity check, define `survived`
as "alive at the first breath", and have the harness integrate the sample series (not the
publishes) and record the bridge `detail` per call.
