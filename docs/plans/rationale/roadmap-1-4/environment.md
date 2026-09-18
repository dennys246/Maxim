# Roadmap 1.4 "Anticipation" — ENVIRONMENT lens (design review of the roadmap, 2026-09-18)

**Charter** (`docs/experiments/DESIGN_REVIEW.md`): does the world game-natively afford it (D1 — no
synthetic sensor/reward), are the needed states/acts reachable + measurable, does the bridge/world
behave? Reviewed: `docs/plans/roadmap_1_4.md` (DRAFT v1) against `survival_world_1_3.md`,
`docs/wiring/world-light-sensing.md`, `sensor-range-clamps.md`, `harness-loop-must-be-proven-live.md`,
`scripts/minecraft_bridge/index.js` (+ the INSTALLED `node_modules`: mineflayer 4.38.0,
mineflayer-pathfinder 2.4.5, prismarine-physics 1.11.1), `scripts/survival_world/setup_world.py`,
`water_trial.py`, `exp60_water_check.py`, the R3 prereg §Pilot / §Apparatus re-check / §Outcome, the
apparatus record `docs/experiments/data/exp60_water_apparatus_2026-09-17.json`, and the Exp 62 v2
prereg §"What this means for the pressure idea". The roadmap is not edited.

**Evidence tags.** `[verified: file:line]` = read in the bridge/harness code or a pinned record.
`[measured]` = a number from a dated apparatus row or the R3 pilot. `[game fact — pilot-measure]` =
my knowledge of Minecraft 1.20.4 / Mineflayer, NOT verified in this repo; it must be measured on
the rig before the rung's prereg freezes.

**Headline.** The treasure dive is buildable by RCON and its pressure/relief/cost are game-native
(D1 holds). But four things the roadmap states as given are not what the bridge and the world
actually afford: (1) the primitive verbs as written (`move(direction, duration)`, `turn(degrees)`)
cannot be selected by the substrate argmax, which emits empty params; (2) "sneak = sink" does not
exist in the bot's physics engine; (3) a *lit* column is a different context to the carried fear
(cos 0.588, measured), and the underground light read is the one sensor this world has documented
as unreliable; (4) the pain-free budget is ≈ 5.2 s, not ≈ 6, and it buys 2–3 primitives, not ten —
there is no pain-free round trip at depth ≥ 4 even with the food directly below. None of these
kills the ladder; each changes the text the rung's prereg must carry.

---

## DO-NOT-BUILD (as written — substitute text given)

### DNB-1 — `move(direction, duration)` and `turn(degrees)` are unreachable by the mechanism under test

**Evidence.** The substrate-primary proposal path takes its params from `NAc.recommend_action`,
which always returns `"params": {}` [verified: `src/maxim/decisions/nac.py:2444`]; the loop copies
that verbatim [verified: `src/maxim/runtime/agent_loop.py:1578`]. The shipped body already records
the consequence: `flee` and `escape_water` are "param-free BY DESIGN: substrate-primary
recommend_action emits empty params, so flight must be a fixed action pattern" [verified:
`src/maxim/_data/components/bodies/minecraft_player.yaml:226-235`]. The existing `turn` handler
computes `params.degrees * Math.PI / 180` [verified: `index.js:175`] — with `{}` that is `NaN` and
`bot.look(NaN, …)` is a silent no-op or an error; it has never been selected by the substrate in
any EARNED run (every R3/Exp 60/61 executed call was `flee` or `escape_water`).

**Consequence.** Phase 0 item 4 ("`move(direction, duration)` … plus the existing `turn`") and
Phase 5's "primitives as the motor layer" describe verbs the argmax can name but not parameterize.
A harness that passes params for them would be the D43 shape (a reconstruction that passes while
the loop cannot).

**Substitute.** "Primitive motor verbs on the real bridge are PARAM-FREE, fixed-duration control-state
holds — `swim_forward`, `swim_back`, `swim_left`, `swim_right`, `swim_up`, `sink`, `turn_left`,
`turn_right` — with the duration and the turn angle frozen as bridge constants and printed in the
apparatus row. Each holds its control states inside a `try/finally` that releases them (the
`escape_water` pattern, `index.js:251-269`) and reports the DISPLACEMENT it produced (start/end
position in `detail`), never the elapsed duration." If parameterized primitives are wanted, that is a
`src/` change to the recommend path and enters Phase 0 as such (with the red gate).

### DNB-2 — "sneak = sink" is not in the bot's physics; descent is passive, and the physics is not 1.20.4's

**Evidence.** Bot position is computed CLIENT-side by prismarine-physics and sent to the server; the
installed engine is 1.11.1. In its water branch `sneak` only scales horizontal input by
`sneakSpeed: 0.3` [verified: `node_modules/prismarine-physics/index.js:68,746-748`]; the only sneak
vertical handling is on ladders (`:587`) and on ground (`:175`). The sole vertical control in water
is `jump` → `vel.y += 0.04` per tick (`:723-724`); otherwise `vel.y *= 0.8; vel.y -= waterGravity
(0.02)` (`:78,104,494-495`). So: **sneak does not sink; releasing `jump` sinks**, at a rate set by a
`waterGravity` of 0.02 — the pre-1.13 constant (vanilla 1.20.4 applies gravity/16 = 0.005 per tick
in water, and has a swimming pose the engine lacks) [game fact — pilot-measure]. The engine is the
one that produced the measured ascent 0.31–0.43 s/block and sink-back ≥ 2.1 s after a breath
[measured: R3 pilot; apparatus record `w5_sinkback_s` 2.12–2.32 s]; those numbers are the engine's,
not the game's, and no displacement may be DERIVED from vanilla values.

**Substitute.** "`sink` = release every control state for a fixed duration (passive descent);
`swim_up` = hold `jump`. The pilot measures, on the built column and as an apparatus row: descent
rate (blocks/s, jump released), ascent rate (jump held), horizontal rate (forward held; expected
≈ 2 blocks/s from `liquidAcceleration 0.02` / `waterInertia 0.8` [game fact — pilot-measure]), and
the same three in AIR on the shore. Heading: `bot.look(yaw, pitch, true)` sets the yaw that
`applyHeading` uses; pitch is inert in this engine's water branch (no swimming pose), so the turn
primitives suffice for heading." Record in the variant body's file that the bot's water physics is
prismarine's, not 1.20.4's — a `mineflayer`/`prismarine-physics` version bump is an apparatus
change that re-fires every displacement row.

### DNB-3 — "a light gradient to find it by" in a "lit column": three environment facts against it

**(a) Light is THE context wall for the carried fear.** The shipped body places by light and time:
a lit surface pond reads cos **0.588** to the cave pool, a night pool 0.799, threshold 0.85 — "the
levers with mass are not place" [measured: Exp 62 v2 prereg, the committed replay
`docs/experiments/data/exp62_cross_pool_replay.py`]. Every C-protocol fear in the ledger (Exp 60,
61, R3 arm C) was booked in a sealed shell at **light 0** (`WATER_SHORE_Y = 40` sits at the Exp 58
depth band precisely so that "a surface pool would change light_level and re-open the 'estimated on
a different geometry' trap" [verified: `setup_world.py:423-425`]). A C-protocol agent placed in a lit
column is a cache miss by the roadmap's own measurement — "carried fear (a C-protocol agent)" in E1
would fire nothing, and the arm would read as a fresh agent.

**(b) The underground light read is the one sensor this world has documented as unreliable.** Exp 58
Addendum 3: "an underground cell read 14 at day / 0 at night … a cell with no light source read 13
day and night, and the same coordinates gave different values across repeated probes. Paper/mineflayer
light here is spatially patchy, run-to-run inconsistent, and skylight-leaking underground"
[verified: `docs/experiments/exp58_survival_wants_prereg.md:288-294`]; the classroom moved to depth
because "no geometry fixes a sensor that answers the same question differently each time". Every
stable light value the survival line has since measured is **0** (the sealed pools). A gradient is
exactly the non-zero, spatially varying case Exp 58 found broken. Plausible cause, for the pilot:
mineflayer holds the light arrays it received in chunk packets; blocks placed by `/fill` need the
server's light engine to push an update the client applies — a stale light array reads "13 with no
source" [game fact — pilot-measure: profile the column cell by cell, then restart the bridge and
profile again].

**(c) Nothing in the agent can "find by" a gradient.** `perceivedLight` reads block + sky light at
the bot's own FEET position [verified: `index.js:66-71,127`]; the agent senses the light where it
stands, never in the next cell. A gradient can KEY situations (each depth its own cluster, at gain
weight up to 1.0 over a 0–15 range) — it cannot steer a primitive. If the gradient is built, then
the fear booked at one depth's light is keyed to that light, and the "depth contrast" is a light
contrast whose cluster walls the replay has to place.

**Physics of the gradient, if it survives (b):** sealed shell → sky light 0 everywhere and
`time_of_day` inert (`doDaylightCycle false`, `time set day`, both verified per run by
`WaterTrial.check_gamerules`); a single block-light source (sea lantern / glowstone, emits 15) decays
1 per block of Manhattan distance through water as through air (water's light opacity is 1 in
1.13+; pre-1.13 it was 3) [game fact — pilot-measure] → a monotone 15 → 0 ramp over ≤ 15 blocks,
which IS full-swing on the light sensor. An OPEN-topped column instead gives sky light 15 − depth
[game fact — pilot-measure]: brighter toward the surface (the wrong direction for a food cue), and
darkness-ramped by time if the day cycle ever runs.

**Substitute (E1 §Question and §Arms, and the classroom paragraph of §The thesis).** "The E1 column
is a SEALED shell (sky 0, time inert). Whether it carries a light source is decided by a measured
apparatus row, not assumed: the bot is teleported to every cell of the intended path and
`light_level` is read at 4 Hz for 2 s per cell, the profile is repeated after a bridge restart and
after a rebuild, and the three profiles must agree cell by cell (± 0). If they do not, E1 runs at
light 0 and depth is carried by the OUTCOME DVs (oxygen minimum, pain-seconds, health lost), not by
a depth-keyed cluster. In either case the fear arm's C-protocol trains IN the E1 column at the depth
and light of the test, never imported from the R3 pool; the §Pressure replay step replays
light-at-depth alongside `y_altitude` and `pressure` on vectors captured in the built column."

### DNB-4 — the budget arithmetic ("≈ 6 s … one primitive in ten") is wrong; no pain-free round trip exists at depth ≥ 4

**The budget.** The first oxygen pain (oxygen 13 < the comfort edge 14, intensity 0.5, fear −0.25) lands
at **5.15–5.28 s** after the head submerges [measured: apparatus record 2026-09-17 `t_pain_edge`
5.153/5.235/5.280; R3 pilot 5.35 s]; the oxygen-12 publish the roadmap quotes (5.8–6.1 s) is the
SECOND publish (the −0.75 that clears θ). Sensed oxygen is `Math.round(air / 15)` [verified:
`node_modules/mineflayer/lib/plugins/breath.js:14`] — 0.75 s per bubble — so 20 → 13 is 7 bubbles
= 5.25 s by construction. The harness discipline's US-free cap is the pain edge − 0.75 s = **4.40 s**
[verified: R3 §Apparatus re-check; `exp60_run.py` `probe_cap_margin_s`]. "Pain-free" therefore
means a head-submerged interval ≤ 5.2 s; the window the harness may call US-free is ≤ 4.4 s.

**The cost of one primitive.** The loop is FROZEN inside a blocking bridge call [measured: R3 pilot
"the loop frozen inside each"; `call_action` blocks up to 15 s, `simulation/minecraft.py:76,237`];
each executed primitive costs one idle tick **0.58 s** + its held duration, and while the carried
fear is active every tick first proposes `flee`, which throws in 1 ms in water [verified:
`index.js:215-217`] and costs the next tick — the **0.77 s tie-break tax** [measured: R3 Outcome,
24 % of C's latency] — on EVERY tick the fear proposes, not once. A 1.0 s primitive is ≈ 1.6 s
without the tax and ≈ 2.4 s with it: **3 primitives (2 with the tax) inside 5.2 s, not 10.**

**The round trip, food directly below the entry point, zero horizontal primitives** (ascent
0.31–0.43 s/block [measured]; `eat` = the game's 32-tick consume 1.6 s [game fact — pilot-measure]
+ the bridge's food-change poll ≤ 1.5 s [verified: `index.js:286-296`]; descent [pilot-measure],
taken here at the ascent rate as a floor):

| depth (builder 3–12) | descent ≈ | eat + poll | ascent ≈ | ticks (3 × 0.58) | head-submerged total | vs 5.2 s |
|---|---|---|---|---|---|---|
| 3 (`WATER_MIN_DEPTH`) | 0.6–0.9 s | 1.6–3.1 s | 0.6–0.9 s | 1.7 s | **4.5–6.6 s** | marginal |
| 4 | 0.9–1.3 s | 1.6–3.1 s | 0.9–1.3 s | 1.7 s | 5.1–7.4 s | exceeded |
| 5 (the R3 placement) | 1.2–1.7 s | 1.6–3.1 s | 1.2–1.7 s | 1.7 s | 5.7–8.2 s | exceeded |

Add one horizontal primitive each way and depth 3 is over the budget too. A teleport-in start (R3's
`submerge`, `water_trial.py:488-496`) removes the descent column only.

**Consequence.** E1's "two or three depths chosen so the pain-free budget is exceeded at the deepest"
is satisfiable, but the SHALLOWEST must sit inside the budget, and only depth 3 with the food in the
entry cell can. E3's "dive lengths converge on the pain-free budget from below … while the food is
still reached" over "a longer path" is arithmetically impossible without an air pocket every 2–3
primitives — which makes E2's pocket a PRECONDITION of E3's design, not only of its mechanism.

**Substitute (§Phase 0 "loop cadence as a budget", §Phase 2, §Phase 4, §Risks).** "The pain-free
budget is the measured first-pain edge (5.15–5.28 s; the harness's US-free cap 4.40 s), re-measured
on campaign day. A primitive costs one loop tick plus its held duration, plus the `flee` tie-break tax
on every tick the fear proposes; the pilot measures the round trip at depths 3/4/5 with the food in
the entry cell, and E1's depths are chosen from that table, with the shallowest inside the budget.
E3's path length is set so that no leg between breaths exceeds 2–3 primitives; the pockets are part
of E3's apparatus."

---

## SHOULD-FIX

### SF-1 — Food at depth: what is game-native, and the apparatus rules it needs

- **No chest/take verb exists** and none is needed: the game picks up an item entity automatically
  when the player's hitbox (expanded 1 × 0.5 × 1) overlaps it — Mineflayer emits `playerCollect`;
  no action, D1-clean [game fact — pilot-measure with the bot]. `eat` then finds it: the handler
  matches `bread` or any `foodPoints` item [verified: `index.js:273`].
- **Item entities FLOAT UP in water (1.13+).** A bread summoned on the pool floor rises to the
  surface in seconds [game fact — pilot-measure]. The treasure must sit in a stone-roofed alcove
  (solid block directly above the item cell); the pickup reach means the bot needs only the adjacent
  cell at the same height.
- **Summon syntax (1.20.4 NBT, pre-components):** `summon minecraft:item X Y Z
  {Item:{id:"minecraft:bread",Count:1b},Age:-32768s,PickupDelay:0s}` — `Age:-32768` defeats the
  5-minute despawn; the harness kills stale items (`kill @e[type=item,distance=..N]`) before each
  summon and verifies `execute if entity @e[type=item,x=,y=,z=,distance=..1]` [game fact —
  pilot-measure the reply strings, as `setup_world` does for `fill`].
- **The inventory already holds the treasure.** `prepare` gives 64 bread [verified:
  `setup_world.py:210`] and `keepInventory true` keeps picked-up bread across deaths [verified:
  `water_trial.py:47`]. The dive classroom must `clear <bot> minecraft:bread` before every episode
  and PREFLIGHT that `eat` on the shore throws "no food in inventory" (the analog of
  `check_no_positive_escape_link`), or hunger relief is available without the dive.
- **Eating underwater** is allowed by the game (no submerged restriction) [game fact —
  pilot-measure via `bot.consume()` with the head submerged]; it costs 1.7–3.1 s of the oxygen budget
  (DNB-4) and the bot moves at sneak speed while consuming.
- **Hunger onset is apparatus-controlled or it does not happen.** Natural drain at rest is zero over
  60 s with the loop live [measured: R3 pilot "food 20 → 20, saturation 10 → 10"]; swimming
  exhaustion is 0.01 per metre [game fact]. The proven route is the game's own effect — `effect give
  <bot> minecraft:hunger 1000 20` then `effect clear` [verified: `break3_smoke.py:62-69`,
  `r2_learned_bias.py:112-117`] — the same class as the D4 heal (`effect give … saturation`). The
  prereg must NAME it as the controlled onset and read `foodLevel`/`foodSaturationLevel` over RCON per
  episode [verified: `WaterTrial.read_food_state`] — the reservoir lesson applies twice here: bread
  gives +5 food / +6 saturation, so food must be ≤ 15 for the relief to register at all (the
  food-caps-at-20 lesson), and the sensed `saturation` is clamped at 10.
- **The arms are coupled through regeneration.** With `naturalRegeneration true` every hp healed
  after drowning damage costs 6 exhaustion (1.11+) [game fact — pilot-measure]: R3's arm A lost
  ≈ 11 hp and healed it, i.e. spent ≈ 66 exhaustion = the whole saturation reservoir and more. In E1
  an agent that takes drowning damage gets HUNGRIER — the fear arm's cost feeds the want. Record
  food per episode; consider `naturalRegeneration false` for E1 (R3 rejected it for the death edge,
  D5; E1 has a different DV).

### SF-2 — `y_altitude` is not "full-swing over the column"

Declared range `[0, 128]`, rest 64 [verified: `minecraft_player.yaml:55-60`]; the pools sit at shore
y 40 with floors at 40 − depth − 1 = 27–36 [verified: `setup_world.py:423,460`]. A depth difference
of 3–9 blocks is 0.02–0.07 of the range; the Exp 62 replay measured `y_altitude` at gain weight
**0.09** against three full-swing cues at 1.0/1.0/0.77 [measured: Exp 62 v2]. "Cosine sees direction,
not magnitude; a small one-sided move never separates" (`docs/wiring/cosine-separation-is-directional.md`).
In a dark shell all E1 depths are ONE cluster. That is acceptable for E1's outcome DVs (oxygen
minimum, pain-seconds, health lost, hunger relief are game-native whatever the clusters do) but not
for a per-depth "drive-decisive" provenance read or any depth-keyed learned bias — say which E1
needs. **Substitute** in §Pressure: "if the depth contrast, replayed on vectors captured in the built
column, separates on the existing roster (light-at-depth is the only candidate with mass; `y_altitude`
carries ≈ 0.09), pressure is not built; …". Drop "full-swing".

### SF-3 — Air pockets: buildable and correctly sensed, but "in the pocket" is an ACTIVE state

- **Build.** Water never flows upward [game fact]; a pocket = a 1-block-tall air cell with stone on
  four sides and the ceiling, open only downward into the source column. Extend
  `water_classroom_verifications` with `execute if block … minecraft:air` for the pocket and
  `water[level=0]` for its four water neighbours after the 2 s fluid settle [verified pattern:
  `setup_world.py:509-532,706-713`]. Do it with `setblock`/`fill … air` AFTER the water fill, inside
  the stone the column was carved from.
- **Sensing.** `is_in_water` is the EYE block (`position + eyeHeight 1.62`) [verified:
  `index.js:120-121`] — with feet in the top water cell and the pocket above, the eye block is air →
  0, and the game refills air on the same eye rule → the pocket reads exactly like the open surface.
  Refill is +4 air/tick, 0 → 300 in 3.75 s [game fact]; measured on the shore: sensed oxygen ≥ 19
  within **3.6–3.9 s** of the rescue teleport [measured: apparatus record `t_recover`]. Pilot-measure
  the refill INSIDE a pocket once.
- **Holding station.** The bot rises only while `jump` is held and sinks when released (DNB-2):
  after `escape_water` releases, the head is back in water within ≥ 2.1 s [measured]. A 1-tall pocket
  pins the head under the ceiling while `jump` is held (hitbox 1.8 tall in a 2-block space from the
  feet cell) [game fact — pilot-measure]. So "at the pocket" = a `swim_up` re-issued every tick; the
  same-tool cap (SF-4) throttles it to 5 in 6, and the breath is ~0.6 s per tick of hold. Build pockets
  as a roofed 3 × 3 (or 3 × 1) air layer, not a 1 × 1 chimney — entering a 1-wide cell from below by
  primitives needs ± 0.2 block horizontal alignment (hitbox 0.6 wide) [game fact — pilot-measure].
- **Bubble columns are NOT pockets to this sensor.** Soul-sand bubble columns refill air game-natively
  while the eye block is `bubble_column`, which the bridge counts as water [verified:
  `index.js:121,241`] → `is_in_water` 1 with oxygen rising. A relief in a state the body reads as
  "submerged" would be booked on the submerged cluster. Do not use them unless the sensor semantics
  are decided first.
- **E2's "which pocket" DV** needs pocket identity in the harness (position in the bridge `detail`,
  DNB-1) — the body senses "surfaced" plus `offset_x/z` (whisper-weight), not "pocket A".

### SF-4 — The consecutive-same-tool cap (5) bites every primitive path longer than five

`_MAX_CONSECUTIVE_SAME_TOOL = 5` [verified: `agent_loop.py:2085,3575`]: the sixth identical proposal
is dropped and the counter resets — "a sustained identical fear response executes at a 5/6 duty
cycle" is a filed follow-up [verified: harness lesson §Follow-ups]. E3's long path (a run of
`swim_forward`) and the E2 pocket hold (a run of `swim_up`) both hit it; the dropped sixth costs a
tick from a 5.2 s budget. **Substitute:** the prereg names one of — a substrate-primary exemption in
`src/` (enters Phase 0 with its test), or paths bounded to ≤ 5 identical primitives by construction
(alternating verbs is not a fix; it changes what credit keys to).

### SF-5 — Control-state hygiene and the respawn trap, restated for primitives

- No bridge verb releases control states; `stop` clears only the pathfinder goal [verified:
  `index.js:191-198`]. A primitive whose Python-side call times out (15 s) or whose bridge socket
  drops mid-hold leaves `forward`/`jump` HELD until the process dies — the bot self-moves through every
  later placement (the stale-goal finding, one layer down). Every primitive releases in `finally`;
  `stop` also calls `bot.clearControlStates()`; `WaterTrial.stop_motion` runs at every boundary.
- A call in flight at death RETURNS SUCCESS after respawn put the head in air — "a positive link
  booked on a drowning" [measured: R3 pilot §4]. With primitives the same happens to whichever
  primitive was held at death. The R3 rule (the event ends at the first clear sample or the death;
  the actuator's contract is what keeps the window free of positive links) must be restated:
  a primitive's `ok` means "displaced ≥ the frozen minimum", and the harness marks any call spanning
  a `deaths` increment as respawn-terminated, never a success.
- `doImmediateRespawn` + `spawnpoint` at the shore [verified: `setup_world.py:503-505`] hold; the
  0.4 s respawn read [measured] and the `deaths` objective per classroom carry over unchanged.

### SF-6 — The descent from a shore start has no innate route; the prereg must pick the start

Hunger's affinities are `eat / pick_up / food / consume / feed`; threat's are `flee / hide / retreat /
escape` [verified: `nac.py:580-586`]; `_DRIVE_CORRECTIVE_NEEDS` maps food → hunger, health → threat
only [verified: `agent_loop.py:891-896`]. No need proposes `sink` or `swim_forward`; a fresh agent
descends only by the explore bonus, and a carried-fear agent descends never (its water response is
up). E1's "a fresh agent that never descends (a floor by design)" is therefore true of EVERY arm from
a shore start unless the harness dives them. **Substitute:** "E1 starts each dive by the apparatus
teleport into the entry cell (R3's `submerge`, the proven seam: `is_in_water` 1 within 3 s, refusal
otherwise); the DV `dives attempted` is replaced by `descents executed` (primitives that moved the
agent DOWN or TOWARD the food after entry) and `reaches`." A shore-start variant is a different
experiment (exploration) and says so.

### SF-7 — Bridge-per-classroom constraints the campaign core must carry

- One client per bridge [verified: `index.js:312-315`]; the flee anchor is fixed at bridge start
  (`--flee_x/--flee_z`) and `bot.spawnPoint` is the login-packet WORLD spawn, never updated by
  `/spawnpoint` [verified: `index.js:40-41,203-208`]. A second pool (Exp 62), a dive column and a
  pocket classroom each need their own anchor → a bridge restart per classroom (the
  bridge-restart-after-sensor-change lesson generalizes: restart, then gate on the raw
  `client.latest_state()` roster).
- Bridge state cadence 100 ms and the 4 Hz sampler stand [verified: `WaterTrial.measure_bridge_cadence`;
  `exp60_water_check.py:105`]; oxygen is quantized at 0.75 s per bubble, so "oxygen at surfacing" and
  "oxygen at pocket entry" have a 1-bubble floor on their resolution — E3's "prediction fires at
  different oxygen levels" is measurable only to the bubble.
- RCON `give/clear/effect/summon` referencing the player need the bot ONLINE and the chunks
  forceloaded (`forceload add` is the first build command [verified: `setup_world.py:494`]; item
  entities and pockets live in those chunks).
- Placement guards carry over: ≤ 90 blocks (3D) from world spawn, ≥ 72 from the Exp 58 clustermob
  [verified: `setup_world.py:431,439`]; the column's depth moves `distance_from_spawn` by ≤ depth
  blocks — a whisper, inside the guard.

### SF-8 — `flee` in the variant body is a per-tick tax against a 5.2 s budget

On the shipped body `flee` is the threat need's consumer and sorts before `escape_water`; in water it
throws in 1 ms and the loop pays a tick [verified: `index.js:215-217`; measured 0.70–0.77 s in every
arm]. In R3 that was 24 % of a 3.2 s latency; in a dive where the fear is active on every tick it is a
tax on every tick. Whether the VARIANT body carries `flee` at all is a wiring decision (it changes the
tie-break the EARNED rows were measured under), but the environment cost must be in the prereg's
budget table either way.

---

## NIT

- **N-1** `perceivedLight` reads at the FEET (`me.position`), `is_in_water` at the EYE; in a pocket
  the two cells differ (water vs air, and 1 light level). State which cell the light profile
  measures. [verified: `index.js:66-71,120`]
- **N-2** Gamerule names on 1.20.4: `drowningDamage` (the pilot's `doDrowningDamage` was rejected);
  `naturalRegeneration` matters only after damage (E1 with regen off is a cleaner cost line — see
  SF-1); `doInsomnia` is a phantom belt only, irrelevant in a sealed shell; `doMobSpawning false`
  also prevents drowned spawns that a block-light source would otherwise suppress at light ≥ 1.
  [verified: `water_trial.py:41-56`]
- **N-3** The item summon NBT above is 1.20.4 syntax (`Count:1b`); 1.20.5+ moved items to components
  (`count`). Pin it beside `MC_VERSION` in `setup_world.py`.
- **N-4** New verbs are a bridge PROTOCOL change: update the header of `simulation/minecraft.py` (the
  authority), `index.js`'s header, and the `FakeBridgeServer` lockstep test in the same PR; the
  scripted grid world (Phase 0 item 3) must take its displacement constants from the measured
  apparatus row, never from prismarine's constants.
- **N-5** `time_of_day` is inert only while the shell is sealed AND `doDaylightCycle false` is
  verified per run (`WaterTrial.check_gamerules`); an open-topped column re-admits the darkness ramp
  (`index.js:59-65`).
- **N-6** The sink-back after a breath (≥ 2.1 s) means a "surfaced" state at the open top persists
  ≈ 1.1 s without a hold [measured]; E2's "oxygen at pocket entry" needs the same `SURFACE_HOLD_MS`
  logic in `swim_up` or the entry is a single 100 ms sample.

---

## Pilot measurements REQUIRED before the E1 prereg freezes (each an apparatus row, on the built column)

1. Primitive displacements: `sink`, `swim_up`, `swim_forward` (and `turn_left/right` yaw delta) at
   1.0 s holds, in water and in air, n ≥ 3 each; the bridge `detail` carries start/end positions.
2. Round trip at depths 3/4/5 with a bread in a roofed alcove at the entry cell: teleport in →
   pickup → `eat` → `swim_up` → first `is_in_water` 0 sample; oxygen minimum and whether the pain edge
   fired (the DNB-4 table, measured).
3. Light profile of the column with one lantern at the food: every path cell, 4 Hz × 2 s, repeated
   after a bridge restart and after a rebuild; PASS = identical cell by cell.
4. Item behaviour: floats in the open column (expected yes); stays in the roofed alcove; auto-pickup
   from the adjacent cell; `clear` + shore `eat` throws.
5. Eating submerged: `bot.consume()` with head in water — works, and its wall time.
6. Pocket: build, verify after the fluid settle, teleport the bot's feet into the top water cell under
   it, hold `jump` — `is_in_water` 0, oxygen refill time to ≥ 19, and the sink-back on release.
7. Hunger onset: `effect give hunger` → `foodLevel` ≤ 5 over RCON → `effect clear`; food stays put
   for ≥ 60 s at rest with the loop live (the drain must be stopped, not merely started).
8. The same-tool cap: six consecutive `swim_forward` proposals in the live loop — confirm the sixth
   is dropped (or the exemption shipped).
9. Regeneration coupling: food/saturation before and after one A-arm-style drowning (regen on).

**Verdict for the roadmap as a DESIGN:** the ladder (instrument → Exp 62 → E1 → E2 → E3) and its
D1 posture hold; §Phase 0 item 4, §The thesis' classroom sentence, §Phase 2's "lit column", the
budget paragraph, and §Pressure's "full-swing" clause need the substitute text above, and E1's prereg
cannot be drafted until pilot rows 1–3 exist.
