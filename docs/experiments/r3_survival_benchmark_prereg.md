# R3 (FROZEN 2026-09-18 at v3.1 — four-lens review FOLDED twice, D1–D5 TAKEN, PILOT MEASURED, R3-cal DONE) — the lethal-window benchmark: what a carried survival drive buys at the one moment it matters, measured on a depth-calibrated, frozen gauntlet

> **STATUS: FROZEN 2026-09-18 (this PR, a MERGE COMMIT: it carries the calibration rows and the gauntlet
> file whose hash the bench looks up on main). R3-cal (build step 4) is DONE: campaign `r3-cal-1`, 12 clean
> floor-arm events at depth 5 at ONE hash (`6b16bbe9`, the harness merge, clean tree), the gauntlet
> `docs/experiments/data/r3_gauntlet.json` validated and every clean row drift-free against it (§Calibration
> result below). No bench row is taken before this PR is on main; the bench runs at one hash from a clean
> checkout at or after it. History: v3.1 folded the delta re-run (all four FIX-THEN-BUILD, no
> DO-NOT-BUILD; reports `rationale/r3-survival-benchmark/*-v3.md`); v3.1 corrects four numbers, pins the
> saturation reservoir the pilot exposed, rewrites the calibration statement, promotes pain-seconds beside
> time-to-air, and lists the harness must-nots the pilot's own code would otherwise pass on. Build step 1, the PILOT, is DONE (2026-09-17, big-mac-mini, hash `5d1e6a62`, clean tree;
> rows committed as a diagnostic at `docs/experiments/data/r3_pilot_2026-09-17.jsonl`): every number the review
> predicted is now measured (§Pilot below), two design consequences are folded (regeneration decides whether the
> floor arm can be titrated by depth; a gamerule name was wrong), and the four-lens re-run the review asked for
> runs on THIS version. The four-lens design review ran on v1 (2026-09-17; confounding, bio-faithful,
> wiring and environment, reports preserved verbatim under
> `docs/experiments/rationale/r3-survival-benchmark/`). All four returned DO-NOT-BUILD as drafted,
> converging independently on the same two premises being false; v2 folds every DO-NOT-BUILD and
> SHOULD-FIX below and records what was dismissed and why. Nothing is built; no live measurement
> has been taken. This is the 1.3 Phase-3 rung (`docs/plans/roadmap_1_3.md` §Phase 3), an
> INSTRUMENT plus a FROZEN BASELINE, never a graduated claim (`minecraft_benchmark.md` D2). The
> owner decisions D1–D5 were TAKEN 2026-09-17 (recorded at the end, with the one design idea
> considered and held out); the build order starts with a MEASURED pilot,
> because two of v1's load-bearing "facts" were extrapolations from windows built to prevent the
> very thing they were cited for.**

## What v1 got wrong, in one paragraph (the reason this document looks different)

v1 assumed a naive agent dies at its first unrescued submersion, so a naive floor of ≈ 0 needed a
calibrating axis (hunger) to give the learned arms a middle. Both halves are false on this world
and this agent. **The naive agent does not drown.** Two mechanisms it carries by default answer
the water: the Wire-4 fear subscriber is auto-wired on every NAc-bearing pain bus (Exp 60 had to
DETACH it for the ablated arm), so a fresh agent books −0.75 on its own water cluster by the
second oxygen-pain publish ≈ 6 s in and escapes ≈ 9–10 s in, before damage at 16 s; and with the
subscriber detached, the innate `health → threat` corrective need (built #683, on by default, the
agent not the world) reaches 1.0 the moment drowning damage takes health below 14 and scores the
escape with no learning at all, ≈ 21 s in, 4–10 s before death. Exp 60's and 61's "0 naive calls
in 90/90 windows" were measured in windows capped at 4.34 s — the pain edge minus a margin —
precisely so that neither path could fire. **And hunger is not a pressure here:** exhaustion at
rest is zero, the shore is a sealed five-block strip, starvation cannot kill at Normal difficulty,
and a hungry reading rotates the world vector the fear is keyed on. Every arm in v1 would have sat
at the ceiling, and the reported "what the drive buys" would have been the schedule's arithmetic.

What survives: the classroom, the apparatus, the agent kinds, the protocols, and the purpose. What
changes: the unit of measurement is ONE lethal-window event per fresh agent (the only unit that
carries arm information — after the first escape the state-blind positive `escape_water` link owns
selection in every arm, measured at 14 dry-land escapes per 10 s in Exp 60), the DV is the COST of
that event, the arms are declared by their learning channel, and the difficulty axis is placement
DEPTH, calibrated on the innate-only floor.

## Purpose — and what this is NOT

**Purpose.** Exp 60 showed a learned drive acts; Exp 61 showed it travels. R3 measures what a
carried drive is WORTH at the one moment the world makes it matter: an agent is in the water, no
one will pull it out, and the routes to air fire in a known order — a carried fear at ≈ 1–3 s, an
in-situ Wire-4 acquisition at ≈ 6–10 s, the innate health reflex at ≈ 21 s — against fixed
deadlines: damage at ≈ 16 s, death at ≈ 25–33 s. Depth stretches the ascent and titrates those
routes out in that order. The number R3 reports is how much air, health and depth each channel
buys, with intervals, on a gauntlet frozen so the number can move.

**Not a claim.** R3 graduates nothing. Contrasts are named after ARMS with the predicted carrying
mechanism stated beside each, never after causes; "the drive buys X" is a reported measurement
with its interval and its mechanism attribution, not a sentence that licenses "agents learn to
survive". A saturated cell is a failed calibration and re-calibrates; it is never a NULL on the
drives.

**Not:** the survival reflex of Phase 1b as a NEW mechanism (the innate `health → threat` prior IS
in every arm and is named as such); R4; intrinsic motivation (unbuilt; the only exploration lever,
`substrate_explore_bonus_weight`, is pinned at 0.0 by the apparatus fingerprint, cited); a survival
WORLD the agent enters by its own movement (not reachable on today's substrate: no locomotor prior
— recorded as the gap, `survival_world_1_3.md`).

## Front-gate scope pressure

Nothing new in `src/`. The gauntlet is the Exp 60 water classroom and its apparatus
(`water_trial.WaterTrial`), the learned arms are Exp 60's training and Exp 61's export → ingest
protocols verbatim, and the analysis is Exp 61's shape. R3 adds one harness method — a
`lethal_event` beside `placement`, which submerges the live agent and does NOT rescue it — plus
depth as a parameter of the placement, a death detector, and the gauntlet file. Every one of these
is a harness change; none is a mechanism.


## Pilot (MEASURED 2026-09-17; the diagnostic the review put first)

`scripts/survival_world/r3_pilot.py`, five rows, one fresh agent each, at the Exp 60 placement (depth 5,
4 blocks of water above the eye), bridge cadence 0.101 s, liveness 6 ticks, live clusters distinct:

| Row | Predicted (v2) | Measured |
|---|---|---|
| **lethal_B** (fresh, subscriber ATTACHED, no rescue) | books fear ≈ 6 s, escapes ≈ 9–10 s, no damage | oxygen pain 5.35 s (0.5 → fear −0.25), 5.92 s (1.0 → **−0.75, clears θ**), `flee` next tick at 6.26 s (fails), third publish 6.65 s (→ −1.0 cap, booked while the tie-break was in flight), `escape_water` at 7.03 s; **head in air at 8.38 s**, min oxygen 9, **health 20**. B ACTS on a need of 0.75 — the same magnitude as arm D's discounted fear |
| **lethal_A** (fresh, subscriber DETACHED, no rescue) | innate `health → threat` at health < 14, ≈ 21 s, survives with damage | damage from 16.15 s; with regeneration ON the first read below 14 is 13.0 hp at **25.11 s**; `flee` one tick later at 25.85 s (fails), `escape_water` at 26.61 s; **head in air at 27.84 s** (2.7 s from the crossing); **min health 9.0**, no fear booked, survived. Attribution by ELIMINATION (the pilot's drive read was empty — a key-path bug; the harness reads the firing drive, below) |
| **drown_on** (regen true, pool capped) | death ≈ 25–33 s | first damage 16.25 s, **death 32.99 s** (health 19 at 20 s, 16.7 at 24 s: regen cancels ≈ 1 hp/s early) |
| **drown_off** (regen false) | death ≈ 25 s | first damage 16.15 s, **death 25.32 s** (12 at 20 s, 4 at 24 s: the game's 2 hp/s) |
| death/respawn seam | never exercised | respawn read **0.4 s** after death: shore (y 40.1), health 20, food 20, oxygen 20, saturation 5, `is_in_water` 0; `deaths` +1 exactly; the loop kept ticking through it (8 ticks in the 6 s after) |
| ascent | ≈ 0.4 s per block | bridge-side escape 1.81 s from the floor, 0.95 s from floor+2: **0.43 ± 0.3 s per block** (two points quantised at the sampler period); executor-side call-to-air 1.23–1.65 s over 3.4 blocks gives 0.31–0.41 s/block |
| `flee` on the shore | a multi-second no-path from a sealed room | "No path to the goal!" in **0.015 s** — fails FAST because the pathfinder exhausts a sealed 9×3×3 chamber in milliseconds (the anchor was world spawn: the bridge ran without `--flee_x/z`). With the anchor AT the shore, `flee` from the shore is a free success by construction (goal already satisfied) — recorded as free, like a dry `escape_water`. In-water `flee` costs 0.70–0.77 s in EVERY arm (the tie-break tax, ≈ 23 % of a carried-fear latency) |
| exhaustion at rest, loop live, 60 s | zero | food 20 → 20, saturation 10 → 10, zero calls (107 ticks) |
| gamerules | unverified | `naturalRegeneration true`, `doInsomnia true` (no code sets it false — fixed in v3.1), the five frozen rules as frozen, difficulty Normal; **the pilot MISSPELLED the drowning rule** (`doDrowningDamage` → "Incorrect argument"): the 1.20.4 rule is **`drowningDamage`** (Java 1.15+, default true) — read under its real name and frozen (drowning measured lethal: two deaths) |

**What the pilot changes in the design (folded below):**

1. **The innate route is real and it is slow, and its margin is funded by an INJECTED reservoir the
   instrument cannot see.** Detached, the agent surfaces at 27.8 s with 9 hp — 2.7 s after the health
   drive's first read below 14 (13.0 hp at 25.11 s). Regeneration delays that crossing from ≈ 19.3 s
   (regen off: 12 hp at 19.27 s on the 2 hp/s clock) to ≈ 25.0 s — and what funds the delay is
   SATURATION: the apparatus heal (`effect give … saturation`, D4) sets the TRUE value to ≈ 20 while the
   bridge clamps the sensed value at 10, so every sample reads 10 while the game spends 20 → 10
   underneath; health holds 18–20 from damage onset until sensed saturation reaches 0 at 25.1 s, then
   collapses. The regen-on margin (33.0 − 27.8 = **5.2 s**) and arm A's health lost (11, NET of
   regeneration) are therefore quantities of the injected reservoir. With regeneration OFF the
   reservoir is inert (a clean 2 hp/s line): crossing 19.3 s + the measured 2.7 s → air ≈ 22.0 s vs
   death 25.3 s → margin **≈ 3.3 s, DERIVED, not measured** (v3 wrote ≈ 2 s in three places; wrong).
2. **Depth cannot titrate the floor arm with regeneration ON, and only marginally with it OFF.** Each
   block adds 0.31–0.43 s of ascent to EVERY arm; the builder's maximum (12) adds 1–5 s at the interval
   bounds — inside the 5.2 s regen-on margin by ≈ 6 SD (the lag's spread: one tick phase ≈ 0.19 s +
   flee 0.03 + ascent 0.15 + crossing 0.1, in quadrature ≈ 0.35 s). With regeneration OFF a 3.3 s margin
   is consumed at ≈ 7.7 blocks → a lethal cell at depth ≈ 12–13, the builder's limit or beyond: a coin
   flip, not a middle. The calibration section is rewritten for this: R3 is calibrated on the COST of
   the event (time to air; per-drive pain-seconds), survival is a declared ceiling with regeneration on,
   and `escaped_before_damage` is a route-order constant (A/E never, B/C/D always at every buildable
   depth), not an "unsaturated DV" as v3 said.
3. **The death/respawn seam holds with the loop live** — the R3 harness may rely on the per-sample
   `deaths` read and the respawn discontinuity as designed.
4. **"Surfaced" is a ≈ 1.1 s breath, and the 2 s linger booked links the event did not.** The escape
   holds jump 600 ms after the head clears, releases, and the bot sinks back (A: clear 27.84 → in
   water 28.98; B: 8.38 → 9.50) — the pathfinder is dead in water, there is no route to the shore, so
   an agent that keeps proposing `escape_water` bobs indefinitely. In the linger both rows logged two
   more real re-escapes (7 links vs Exp 61's 3 per single escape). So the event ENDS at the first clear
   sample, `survived` means "alive at the first breath by its own act", and the post-window teleport is
   the exit. Corollary worth stating: the first positive escape link is booked at call RETURN, ≥ 600 ms
   after the head clears, so the event window holds zero positive links by the actuator's contract.
   **The same-tool cap did NOT fire under the stone cap** (v3 mis-timed it): under the cap the limiter is
   the 8 s BLOCKING escape call (calls 8.1 s apart, the loop frozen inside each), and the escape in
   flight at death RETURNED SUCCESS when respawn put the head in air — a positive link booked on a
   drowning. The cap (`_MAX_CONSECUTIVE_SAME_TOOL` = 5: five execute, the sixth is dropped, reset) fired
   on the post-respawn shore snowball. It cannot bite inside a ≤ 12-block event.

## The gauntlet

One world, the Exp 60 water classroom, one server and bridge (Paper 1.20.4, bridge at
`--state_interval_ms=100`, flee anchor frozen at the SHORE — v1's default was world spawn 69 blocks
away, a multi-second no-path inside a blocking executor call the moment a threat need stands on
the shore), one encoder, fresh persistence per agent, one agent per process.

**The event.** After its protocol (below) and the apparatus checks, the agent is teleported once
into the water column at the frozen DEPTH with the loop live, and left there. The event ends at
the first of: the head clears the water by the agent's own act (bridge truth: the FIRST sample with
`is_in_water` 0 — an EYE-block sensor; the pilot's clears came at feet y 38.7–38.9 with the pool's air
layer at y 40, so no "feet above the water line" clause: it would never fire); death (below); or a
hard cap of 45 s, a named refusal (an agent alive underwater at 45 s is an instrument fault — death is
25.3–33.0 s). At SURFACE the harness teleports to the shore IMMEDIATELY, before stopping the loop
(the pilot joined first and left a re-sinking agent exposed for the join — reversed); at DEATH respawn
already did; at CAP teleport + refusal. No linger. There is no second event. The agent's persistence is
closed and staged per row as the row's evidence.

**Death detection (measured in the pilot; two seams hardened in the harness).** Preflight: the
`deaths` objective EXISTS (`scoreboard objectives list`), is set to 0 and READ BACK 0. Per sample: the
snapshot is synced THEN the objective is read over local RCON (that order is load-bearing — reversed,
a respawn snapshot reads as a surface); a parse failure is an `InstrumentError`, never 0 (the pilot
still carried the silent zero); death = `deaths_delta > 0` corroborated by the respawn discontinuity
(health 20 after < 20, `is_in_water` 0, y = the shore) within ONE sample — disagreement is an
`InstrumentError`. Death and respawn fell inside one sampler period in both pilot rows (health 0 is
never visible). The per-sample RCON read costs ≈ 0.14 s: the pilot sampled at **2.6 Hz**, not 4; the
harness holds 4 Hz (`sleep(max(0, 0.25 − elapsed))`) and stamps `state_age_s` per sample, refusing
> 0.15 s. A post-window `deaths` rise is an `InstrumentError`. The scripted bridge gains a scripted
DROWNING so the death branch and the parse failure are red-gated OFFLINE (today it deals no damage).

**Hunger and regeneration are CONSTANT, declared, recorded and PINNED — never an axis.** Start state =
the Exp 60/61 rescue settle (`WaterTrial.heal`: instant health + saturation) — a harness injection
of interoceptive state, declared as such, kept for parity with every fear read Exp 60/61 ever took.
**The reservoir it injects is the quantity that funds the floor arm's regen-on margin, and the body
cannot see it** (true saturation ≈ 20, sensed 10 at the bridge clamp; game-native respawn is
saturation 5 and would shorten the hold by ≈ 6–7 s). So the harness reads the TRUE
`foodSaturationLevel` / `foodLevel` / `foodExhaustionLevel` over RCON (`data get entity`, the
`bot_pos` pattern) at every event teleport, records them per row, and the accepted values are frozen
in the gauntlet file beside `naturalRegeneration`; a row outside the frozen band is refused. The
heal's 10 s effect is still active for the first ≈ 8 s of the event (harmless: damage at 16 s; stated). `naturalRegeneration` is frozen at its
default `true` and VERIFIED in the gamerule roster (measured: it moves the death edge by 7.7 s);
**`drowningDamage true`** (the real 1.20.4 name) verified — "false" refuses, an unknown-name reply is an
`InstrumentError`, never a recorded absence;
drowning damage MEASURED lethal (two pilot deaths at 25.3 / 33.0 s);
`difficulty normal` pinned by an RCON read in the fingerprint (starvation cannot kill here — stated,
not relied on); `doInsomnia false` added to `setup_world._GAMERULES` and `water_trial.GAMERULES` (v3 said "at setup" but no code set it; the pilot read it true); every gamerule SET is followed by a READ-BACK (the pilot asserted the set's echo); `weather clear` and `time
set day` at setup. Food, saturation, health and oxygen are stamped on every sample.

**The settle guard** (`is_raining == 0`, `nearest_player_dist == 64`, `hostile_count == 0`) runs at
the event teleport and on EVERY sample, not only inside `rescue()` (which R3 never calls): a breach
before the DV refuses the row, after the DV marks it dirty. **The flee anchor** is written into the
anchor record / gauntlet file (`water_anchor_record` carries none today; the bridge takes it at start
only), the bridge is started with it, and a preflight `flee` that answers "No path" or "no flee anchor"
REFUSES (v3's own rule; the pilot recorded a NoPath and did not refuse). **The pool's surface cell is
verified AIR before every campaign start** (`execute if block <surface> minecraft:air`): the pilot's
stone-cap restore discarded the fill reply and no row observed it.

## Arms — declared by learning channel, with the Wire-4 subscriber's state in the gauntlet named

| Arm | Agent, before the event | Subscriber during the event | Predicted route to air (mechanism, onset) |
|---|---|---|---|
| **A innate-only** | fresh persistence, no training | DETACHED (`_detach_fear_subscriber`, as Exp 60's ablated) | innate `health → threat` at health < 14, ≈ 19–21 s; the FLOOR |
| **B in-situ learner** | fresh persistence, no training | ATTACHED (production default) | Wire-4 acquires −0.75 on its own cluster ≈ 6 s, escape ≈ 9–10 s; the acquisition curve is this arm's own number |
| **C self-learned** | Exp 60 FEAR training (10 usable episodes, propose-only, rescued), NO post probe (zero escape links at the boundary, asserted) | ATTACHED | carried fear −1.0, escape ≈ 1.3–3.3 s |
| **D shared** | fresh receiver that ingested a C-protocol donor's export (Exp 61 lifecycle: real CLI export/ingest, reboot, loop-OFF gate reads −0.75; **Exp 61 EARNED 2026-09-17**, `exp61_verdict.json` at `4e25b475`, 12/12) | ATTACHED | carried fear −0.75; the argmax makes D ≡ C structurally (both clear θ, same affinity, same tie-break) — reported side by side, NEVER contrasted as "the discount's price" |
| **E exposed-ablated** | Exp 60 ABLATED training (same exposure, subscriber detached, propose-only) | DETACHED throughout | A plus a pre-formed water cluster and Wire-2 valence: the exposure-without-drive control; predicted ≡ A |

One fresh agent per row; C-protocol donors for D are one per receiver, no reuse. Every learned
arm passes its own protocol's sanity before the event (Exp 60's G2 for C/E; Exp 61's staged
sanity, ingest gate and loop-OFF representation gate for D); every arm asserts
`positive_escape_links() == 0` and `reward_bias == {}` at the gauntlet boundary; a failure is a
named REFUSAL, never a row. Curiosity is unbuilt; `substrate_explore_bonus_weight` 0.0 and
`drive_gate_enabled` false are in the fingerprint and the fingerprint is verified per row.

## Dependent measures

**Primary (per event, one per agent) — the COST of the event in the drives' own currency:**
- `t_surface` — latency from the teleport to the FIRST sample with the head clear (censored at death
  or the cap, reported with the censoring class), with its named components: the first proposal's
  tick, the `flee` tie-break's cost (0.70–0.77 s), the escape call, the ascent;
- **per-drive pain-seconds** — the integral of `drive_pain_for_value` for oxygen and for health over
  the SAMPLE series to `t_surface` (left-Riemann on the stamps; NEVER from the pain publishes, which
  count deepenings only). Pilot: oxygen 22.2 / 2.9 / ≈ 0 s and health 2.2 / 0 / 0 s for innate-only /
  in-situ / carried — the same function the relief credit runs on, so this is the negative-reinforcement
  signal NOT paid, Wire-4's job. What "what the drive buys" means on this world: ≈ 19–25 s of latency,
  ≈ 11 hp and ≈ 22 s of oxygen pain — not life.

**Reported, never gated:** `survived` = alive at the first breath by its own act (binary; a CEILING in
every arm with regeneration on, declared); `escaped_before_damage` (a ROUTE-ORDER flag: A/E ≡ 0, B/C/D
≡ 1 at every buildable depth — the anti-vacuity check, never a tested DV); health lost, cut at
`t_surface` (net of regeneration; reservoir-dependent for A); the executed-tool sequence with call
START and RETURN times and the bridge `detail` per call (`surfaced` / `already at surface` / `capped`
/ `surfaced_by_respawn`); links accounted by call time (≤ `t_surface` = the event's; later = post-event,
recorded); **decision provenance per proposal** (`RecommendCapture` + `decision_decisive`, Exp 61
step 5): the executed escape must be drive-decisive — what that READS is the aggregate `drive` score
component (> 0) with causal 0 and learned 0 on the executed escape's `NAc_RECOMMEND` event; the event
carries no per-need breakdown, so there is no named `threat` term to read (a `src/` change, not this
line's), and the route is named by the arm's declared channel plus that read — a REFUSAL condition for
every executed escape, and the reason the pilot's rows are labelled "attributed by elimination"; for arm B, the acquisition curve (publishes to the first proposal, `cluster_fear_dump`
after); the loop's tick-period distribution per event (measured 1.4–1.8 Hz against a nominal 4 — the
largest variance term in every arm, an INSTRUMENT constant: median/IQR frozen in the gauntlet file, a
row outside the band refused); `t_first_damage` per event (own onset earlier than the anchor's
minimum with `oxygen > 0` = the hostile-leaked / fall refusal).

**Contrasts, named by arm, mechanism beside each, intervals always, on `t_surface` and pain-seconds:**
C − A (carried fear vs the innate floor: ≈ 24 s), B − A (in-situ acquisition vs the floor), C − B (what
carrying it saves over learning it there: ≈ 5 s of latency of which ≈ 5.3 s is the pain-free descent
to the oxygen-12 publish, and ≈ 2.9 s of oxygen pain — B cannot start before that publish at 5.8–6.1 s,
C fires at the first tick, so the gap is structural and complete separation needs n ≈ 4), C − E
(drive vs exposure), D beside C. The DV → arm map, stated: the binary and health-lost separate A/E
from B/C/D only; `t_surface` and pain-seconds carry C − B and B − A. Mann–Whitney with a bootstrap of
the median where < 50 % censored, else the censoring fraction is the number; Fisher only where a
binary can vary. n = 12 buys nothing for C − B; its value is arm A's fault tail and arm B's
acquisition spread.

**Anti-vacuity rows, required (the campaign is INCOMPLETE without them):** one arm-A and one arm-B
first event recorded tick by tick (the pilot's two seeds, re-run inside the campaign at the frozen
cell); one arm-C row whose fear reads at the cap before the event AND whose executed escape is
drive-decisive (the object read paired with the behaviour, never the object alone).

## The Goldilocks calibration — depth, on the floor arm

**What the pilot measured about the axis.** The routes' onsets are fixed by the mechanism (in-situ
fear ≈ 7 s; innate health reflex ≈ 26.6 s with regeneration on, ≈ 21 s off) and the deadlines by the
game (damage 16.2 s; death 33.0 s regen on, 25.3 s off); depth moves the ascent between them at
**0.43 s per block**, so the builder's whole range (5 → 12) is worth ≈ 3 s. **At the Exp 60 depth the
PRIMARY DVs are already unsaturated across the arms** — to-air 27.8 s (innate-only) / 8.4 s (in-situ)
/ ≈ 2.5 s (carried fear, Exp 60/61), health lost 11 / 0 / 0, `escaped_before_damage` 0 / 1 / 1 — and
the benchmark's contrast does not depend on anyone dying. Survival (secondary) has a middle only if the
floor arm's ≈ 5.2 s margin (regeneration on) can be consumed, which depth alone cannot; with
regeneration off the margin is ≈ 2 s and depth can.

**The calibration statement (v3.1).** R3 measures the COST of one lethal-window event, calibrated on
`t_surface` and per-drive pain-seconds. The instrument's range is the window from the first loop tick
(≈ 1 s) to death (25.3 s regen-off / 33.0 s regen-on at the D4 reservoir); each arm's route fires at a
mechanism-fixed onset inside it — C/D **3.15 ± 0.20 s** (Exp 61, n = 12; v3's "≈ 2.5 s" was 3 SD low),
B **8.4 s** (n = 1; predicted SD ≈ 0.35), A **27.8 s** regen-on (n = 1) / ≈ 22 s regen-off DERIVED — and
depth adds a common 0.3–0.43 s per block to EVERY arm: no contrast is a function of depth. `survived`
is a ceiling in every arm at every buildable depth with regeneration on (floor margin ≥ 2.1 s at depth
12 vs SD ≈ 0.35) and is reported as such; `escaped_before_damage` is a route-order flag. **The gauntlet
is the Exp 60 depth (5).** A lethal floor, if ever wanted, is a DECLARED regen-off gauntlet,
reservoir-free, whose margin (≈ 3.3 s derived) is MEASURED before its cells are declared (≥ 3 detached
regen-off events at depth 5; executor-side ascent from depth 8 and 12; the reservoir flat).

**Axis, retained for the record.** Placement depth `d` (game-native: a longer ascent; `escape_water`
holds jump for at most 8 s and reports "still submerged (capped)" as a SUCCESS — recorded per event as
`detail`; a capped escape that dies is a death) — it cannot bite at ≤ 12 blocks (the first escape
surfaces in ≤ 5.5 s).

**R3-cal = ONE verification cell.** Arm A only, depth 5 (the Exp 60 apparatus, every edge measured),
n = 12. Its product is not an accept/reject on survival (the arithmetic gives all-survive to ≈ 6 SD
and a 36-agent sweep would confirm it) but the floor arm's DISTRIBUTIONS: `t_surface` and its margin
to the death edge, the crossing time, health lost, pain-seconds, the loop's tick-period distribution,
the reservoir read at the teleport — the numbers the gauntlet file freezes and the bench refuses
drift against. It ALSO discharges the SF-7 read: the executed escape must be drive-decisive with the
`threat` component read from the capture. If arm A's survival in that cell is NOT ≈ 1.0 — the fault
tail is real — the gauntlet freezes with that fraction and its interval as the floor's number.
(The v3 sweep rule — accept the first cell where arm A's survival lies in **[0.20, 0.80]**
with its Wilson 95 % inside [0.05, 0.95]. At that depth the floor arm is mid-range on the
lethal outcome, arm B is predicted to survive with damage or not at all depending on its
≈ 9–10 s escape plus the ascent, and arms C/D are predicted to escape before damage — a cell
where the ORDER of routes is visible in the outcomes — is retired: with regeneration on it cannot land
by construction, and running it anyway would be the vacuous-gate shape.) A LETHAL floor, if the owner
wants one, is the declared regen-off gauntlet above — the game's other setting, not a synthetic one;
reservoir-free; its margin measured first; recorded as such, never a mid-campaign change.

**Before any live cell (corollary 3):** an offline replay of the cosine between the Exp 60
water node (encoded at the rescue settle) and the submerged reading at each candidate depth —
`y_altitude` is a world-modality sensor and a deeper reading may complete into a NEW node. Then
every learned arm trains at the FROZEN depth (exact-key cache), and its representation gate is
read at that depth before the event.

## Calibration result (R3-cal, 2026-09-18, the frozen numbers)

Campaign `r3-cal-1` on big-mac-mini, 00:53–01:37 UTC (44 min), harness at `6b16bbe9`, clean tree.
Apparatus row PASSED (flee to the shore anchor in 13 ms; bridge escape to air 1.48 s; cadence, liveness,
clusters distinct, gamerules under the real names, surface cell air, `deaths` objective reset and read
back, true reservoir read). Twelve floor-arm events, one fresh agent each, subscriber DETACHED:

| floor arm (A innate-only), n = 12 | value |
|---|---|
| `survived` | **12 / 12** — the declared CEILING, as predicted |
| `t_surface` | median **27.96 s** (27.66–28.29; bootstrap 95 % of the median 27.83–28.16) |
| health lost | median **10.7 hp** (10.7–12.8) |
| oxygen-pain seconds | median **22.5 s** (22.0–22.7) |
| health-pain seconds | 1.9–2.5 s |
| drive-decisive executed escape | 12 / 12 (the aggregate drive component, causal 0, learned 0) |
| true saturation at the teleport | 20 on every row → reservoir band **[19, 21]** (food likewise) |
| loop tick period | band **[0.39, 0.77] s** (median ± 2 IQR) |
| stale-sample tail (clean rows) | ≤ 0.128 s against the 0.15 s limit |

One row (seed 403) was REFUSED on a single stale sample (0.242 s) and re-run under `--resume`; the
re-run superseded it. Three apparatus rows refused before any agent was spent — `doInsomnia` not yet
false on the server, the bridge without its flee anchor, the bridge's bot not in the world after a
restart — each a world precondition now in the runbook. The floor's spread (SD ≈ 0.2 s on `t_surface`)
is narrower than the delta review's ≈ 0.35 s estimate: the innate route is a clock.

## The frozen gauntlet file (D5)

`docs/experiments/data/r3_gauntlet.json`, written by R3-cal, read by R3-bench, refused on drift
like Exp 61's `FROZEN`: the accepted depth and its calibration rows' sha256, `n_cal`, the Wilson
interval; the anchor record verbatim (incl. `t_damage_onset_min_s`, `deaths_objective`); the
fingerprint (incl. `substrate_explore_bonus_weight`, `drive_gate_enabled`, difficulty); the verified
gamerule roster; the flee anchor; the start state; the apparatus record's timestamp and code
hash; the calibration code hash (R3-bench refuses unless `git merge-base --is-ancestor` puts it
on main); the per-arm subscriber declaration; the offline replay's result. Bench rows at a hash
other than the gauntlet's read INCOMPLETE.

**Deferred from the gauntlet file as built (named, not forgotten):** the per-arm subscriber
declaration lives in `r3_run.FROZEN`/`DETACHED` rather than the file; the offline cosine replay vs
depth (build step 2) is moot at a single frozen depth (5, the Exp 60 apparatus) and is owed only if a
depth ever changes; the fingerprint is verified per row (`check_fingerprint`), not copied into the file.

## Refusals and stop rules

Every Exp 60/61 apparatus refusal (fingerprint, cadence ≤ 0.15 s at every sample, liveness,
actuation through the bridge with zero executor calls, clusters distinct, gamerules incl. the two
added, settle guard, one code hash, clean tree, `assert_repo_interpreter`), plus R3's own: a
`flee` preflight on the shore must return "fled to anchor" ≤ 0.5 s (bridge-side, never the
executor); no executor call in flight at the teleport; a death with `oxygen > 0` at the last
sample (a hostile leaked, fall damage) is refused with the cause; a capped-escape death is a
death; `deaths` parse failure is an instrument error; a learned-arm agent that fails its sanity or
carries a positive escape link at the boundary is refused before the event; an event alive
underwater at 45 s is refused.

## What was dismissed, and why (the review's record)

- **Survival time on a repeated-teleport schedule (v1's DV):** after event 1 the state-blind
  positive `escape_water` link owns selection in every arm (Exp 60: 14 dry-land escape successes
  per 10 s roam, 47–51 links); events 2..k are identical across arms; "buys X seconds" = H − 26 s,
  the experimenter's choice. Rejected by all four lenses.
- **Hunger as the calibrating axis (v1 D3 option c):** zero exhaustion at rest on a sealed shore;
  starvation non-lethal at Normal; the `eat` prior fires only below food 11; a hungry reading
  rotates the cluster key; the food supply would be a step function. Not tuning-to-pass — not
  calibratable. Options (a) (a naive agent makes NO actions, so there is no random walk to
  titrate) and (b) (saturates at the ceiling in every arm) rejected for stronger reasons than v1
  gave.
- **`min(health, oxygen, food)/set_point` as drive integrity:** three dynamics collapsed into the
  slowest; `food` has no set-point. Replaced by per-drive pain-seconds.
- **"Naive = nothing learned, nothing carried":** the innate health reflex and the auto-wired
  subscriber are the agent. Replaced by arms A and B, declared.
- **Arm C − D as "the discount's price":** structurally zero on a deterministic argmax; a timing
  failure would have been read as a mechanism result. Reported side by side only.
- **Long loop-live episodes (H up to 600 s):** never proven live beyond 15 s; memory and link
  growth measured per build; the one-event unit removes the need.

## Owner decisions (TAKEN 2026-09-17)

- **D1 — the five arms as tabled, n = 12 each. TAKEN.** (60 agents + 24 trainings for C/E + 12
  donors for D ≈ 36 trainings × ≈ 2.5–3 min + 60 events × ≤ 45 s + apparatus ≈ 2.5–3.5 h; plus
  R3-cal at 12 per cell × the cells walked, ≈ 10 min per cell.) E stays: it is the only control
  that separates "exposure" from "drive" against the floor, and it is cheap.
- **D2 — primary DVs `escaped_before_damage` + `t_surface`; survival secondary. TAKEN as written.**
- **D3 — depth as the axis, calibrated on arm A's survival. TAKEN.** The pilot measures whether
  depth actually moves arm A before any sweep is declared final. **Considered and held out of R3
  (owner idea, 2026-09-17): an internal PRESSURE signal** — a body (SEM) component that transforms
  depth into an interoceptive homeostatic/entropic signal. The categorization is settled in its
  favour: altitude is game-exposed, and a transform of an exposed quantity into an internal one is
  the same class as the eye-height `is_in_water` flag, NOT the invented world fact D1 forbids. It
  is held out of R3 on the CONFOUND, not on D1: as a sensor it re-keys every cluster the trained
  fear sits on (a new channel in the summed vector) and gives R3 nothing depth-through-oxygen does
  not already give (oxygen is the game's pressure analog — the deeper, the longer the air-hunger
  before the head clears — and it is the drive every arm already senses); as a DRIVE it would be a
  route to air that fires on placement in every arm, collapsing the benchmark to the ceiling for
  the same structural reason v1 did. It is a legitimate design candidate for its own front-gate
  and four-lens review after R3-bench (perception line / Phase 1b-adjacent), never inside R3.
- **D4 — start state = the Exp 60/61 heal settle. TAKEN** (declared injection, parity with every
  prior fear read; the game-native respawn state — food 20, saturation 5 — would change the
  cluster key relative to training). v3.1: the heal's TRUE saturation (≈ 20, sensed 10 at the clamp)
  is the reservoir that funds the floor arm's regen-on margin and health-lost; it is READ over RCON at
  every teleport, recorded per row, frozen in the gauntlet file, and a row outside the band refuses.
- **D5 — `naturalRegeneration true` (the game default) frozen and verified for the campaign.
  TAKEN.** Pilot consequence (v3.1): regeneration is the COARSE lever on the floor arm (death 33.0 s
  vs 25.3 s; the innate escape at 27.8 s survives by 5.2 s regen-on vs ≈ 3.3 s DERIVED regen-off), and
  the on-setting's margin is the injected reservoir's (D4). With it on, depth cannot bring the floor
  arm's survival to a middle, so survival is a declared ceiling and the cost DVs carry the contrast;
  regen-off is the reservoir-FREE floor and the cleaner one if a lethal floor is ever wanted — a
  declared gauntlet with its margin measured first, never this campaign's default. Running both settings as two gauntlets was considered (it would show how much of the
  floor arm's survival is regeneration) and rejected as doubling the campaign and splitting the
  frozen file; instead the PILOT records the death edge under BOTH settings (one deliberate
  drowning each way), so the off-regeneration edge is a measured number in the gauntlet file, and
  an off-regeneration baseline is one gamerule flip and a re-run under the ledger's existing
  gamerule trigger.

## Build order

1. **PILOT — DONE 2026-09-17** (`scripts/survival_world/r3_pilot.py`, #752 + fixes #754/#755 — the
   offline test now runs the pilot end to end on the scripted bridge before any rig run; rows at
   `docs/experiments/data/r3_pilot_2026-09-17.jsonl`; §Pilot above). Every "predicted" number is
   measured; two consequences folded (regeneration as the coarse lever; the gamerule roster).
   **Next: the four-lens re-run on THIS version**, then the harness.
2. Offline cosine replay of the submerged reading vs depth (corollary 3).
3. **DONE (the harness PR; two-lens folded).** Harness PR: `WaterTrial.lethal_event(depth)` beside `placement` (no fork) — **what it must NOT
   inherit from the pilot's `live_window`** (wiring delta lens): join-before-teleport (teleport to the
   shore FIRST on a surface, then stop the loop; no linger); the silent `deaths()` zero (raise; preflight
   exists + set-0 + read-back; snapshot-then-scoreboard order); two clocks (one wall `t0` stamped at the
   teleport; the tick reader's dead `t0_monotonic` parameter and empty `threat` column fixed — decision
   provenance from `RecommendCapture`); the end-of-window `deaths_delta` beside the sample detector (DV
   from samples only; a post-window rise is an `InstrumentError`); the set-echo as a read-back; the
   settle guard only in `rescue()`; `close_and_stage(…, None)` (a stage dir per row); bridge `detail`
   and RETURN time per call (the in-flight-at-death success flagged `surfaced_by_respawn`); the 2.6 Hz
   sampler (hold 4 Hz, `state_age_s` per sample). Plus: the reservoir read, the gamerule roster under
   the real names with read-backs, the flee anchor in the record and its refusing preflight, the
   surface-cell check, a scripted DROWNING in `ScriptedWaterBridge` so death and the parse failure are
   red-gated offline (the offline end-to-end test is the template: run every row on the scripted bridge
   before the rig). `r3_run.py cal` / `bench` / `report` with pure halves unit-tested; two-lens code
   review; **cal and bench PRs merge with MERGE COMMITS** (the gauntlet's `merge-base --is-ancestor`
   refusal is only satisfiable that way; #755 was squashed, so the pilot's hash is reachable only from
   its branch — fine for a diagnostic, not for data); `allow_dirty` false; the gauntlet file carries the
   pilot's provenance fields, `rss_mb` (≈ 1 MB per row; no per-episode subprocess needed at one event
   per agent), and `exp60_run.FROZEN` by VALUE (sha256).
4. **DONE (2026-09-18):** R3-cal live (the one verification cell) → gauntlet file → this freeze PR (v3.1 → FROZEN, a merge commit).
5. R3-bench campaign at one hash → merge-commit data PR → §Outcome: the frozen baseline and the
   reported contrasts, each with its mechanism beside it, and what they do not say.

## Apparatus re-check (2026-09-17, after the pilot)

The water classroom was rebuilt in place after the pilot (the surface cell had already tested AIR;
the rebuild was unnecessary but harmless) and the rebuild rewrote the anchor record, so the Exp 60
water check ran again and re-stamped it (hash `6b9720d6`, clean tree, all checks PASS). This is the
"pain edge re-measured on campaign day" the Exp 61 freeze recorded as an honest limit, now done for
R3. Recorded beside the original, never over it — the 2026-09-15 record is the one Exp 60 and 61 ran
against and cite: `docs/experiments/data/exp60_water_apparatus_2026-09-17.json`. **R3's harness reads
THIS record** (pain edge, surfacing) and the re-stamped anchor (`measured.t_damage_onset_min_s`).

| edge | 2026-09-15 (Exp 60/61) | 2026-09-17 (R3) |
|---|---|---|
| pain edge (3 cycles) | 5.09–5.44 s | 5.15–5.28 s |
| damage onset | 16.07–16.65 s | 16.06–16.20 s |
| bridge escape to air | 1.45–1.83 s | 1.40–1.54 s |
| sink-back after the breath | — | ≥ 2.12 s |
| distance from spawn | 69.17 | 69.17 |

Consequence for the frozen numbers: the US-free probe cap derived from this record is 4.40 s
(5.153 − 0.75) instead of Exp 60/61's 4.335 s; the train cap 15.06 s is unchanged to two decimals.

## Delta review record (v3 → v3.1, 2026-09-17)

The four-lens re-run on v3 returned FIX-THEN-BUILD from every lens, no DO-NOT-BUILD, and agreed on
each correction; all are folded above. The one finding the pilot could reveal and v2 could not:
the floor arm's regen-on margin is funded by the heal's saturation reservoir, which the body senses
at the bridge clamp — pinned by a game-native read (D4). Numbers corrected: carried-fear latency
3.15 ± 0.20 s (Exp 61 n = 12), regen-on crossing ≈ 25.0 s, regen-off margin ≈ 3.3 s DERIVED (v3's
≈ 2 s was arithmetic), ascent 0.43 ± 0.3 s/block, depth 12 adds 1–5 s, the same-tool cap's timing and
meaning, the snowball's count. Structural: pain-seconds promoted beside `t_surface`; the binary
demoted to a route-order flag; `survived` = one breath; R3-cal collapsed to one verification cell;
the lethal floor a declared regen-off gauntlet with a measured margin; the loop cadence frozen as
an instrument constant; the harness must-nots listed in the build order. World: `drowningDamage`
under its real name; `doInsomnia` in code; every set read back; the surface cell verified air; the
flee anchor recorded and refused. Pilot rows re-labelled "attributed by elimination" (the drive
read was empty by a key-path bug). Nothing reopens the design; nothing needs `src/`.

## Operator runbook — R3-bench (the frozen protocol, from a clean main checkout at or after the freeze)

1. big-mac-mini: `git checkout main && git pull` (the checkout must contain this FROZEN prereg AND the
   gauntlet file). World preconditions, each a refusal at the apparatus row if missed: the bridge
   RESTARTED with `--state_interval_ms=100 --flee_x=-393 --flee_z=-312` (the `--key=value` form; stop the
   old bridge in its tmux session first, or the port is taken; after a restart confirm the server sees the
   bot: `list` shows `maxim` and `data get entity maxim foodSaturationLevel` answers a number — a bridge
   whose bot never logged in serves a fallback snapshot); `gamerule doInsomnia false` on the server; no
   second player; no rain; the pool's surface cell air.
2. `export PYTHONPATH="$PWD/src"`, then ONE invocation, all five arms interleaved, ≈ 3–3.5 h (60 events,
   12 fear trainings for C, 12 for E, 12 Exp 61 donors + receivers for D):
   `python scripts/survival_world/r3_run.py bench --campaign-id r3-bench-1 --workdir ~/r3_bench --rcon-password '<pw>' --username maxim --write-experiment-results`
   The output is `docs/experiments/data/r3_bench.jsonl`; the tree must be CLEAN; the bench refuses a
   gauntlet that does not validate or whose calibration hash is not on `origin/main`, and refuses every
   row that drifts from it. If the run stops, resume the SAME campaign id at the SAME hash with
   `--resume`; clean rows are skipped, refused rows re-run and supersede. No `git pull` between rows.
3. `python scripts/survival_world/r3_run.py report --data docs/experiments/data/r3_bench.jsonl --gauntlet docs/experiments/data/r3_gauntlet.json --json docs/experiments/data/r3_report.json --campaign-id r3-bench-1`
4. One MERGE-COMMIT data PR with `r3_bench.jsonl` + `r3_report.json`; then §Outcome from the report:
   the frozen baseline and the reported contrasts, each with its mechanism beside it, and what they do
   not say — nothing graduated.
