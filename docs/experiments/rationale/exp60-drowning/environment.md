# Exp 60 (drowning-avoidance) — ENVIRONMENT lens (D1: game-native affordance + reachable/measurable states)

**Reviewer:** environment lens, four-lens DESIGN review, 2026-09-15.
**Verdict:** **DO-NOT-BUILD as drafted.** The DANGER half (submersion + oxygen depletion + drowning
damage) is fully game-native and stageable, and the cue separates far more strongly than dark=danger.
But the ESCAPE half is broken at the physics layer: the only corrective affordance the prereg names
(`flee` → mineflayer-pathfinder) **cannot compute a path out of water**, so the corrective act is
unexecutable and both arms would censor — Exp 58's null-with-cause relocated from the encoder to the
actuator. Buildable only after a new non-pathfinder `surface` actuator ships and passes a live
swim-out check.

Findings below are ranked and each carries evidence, consequence, minimal fix.

---

## DO-NOT-BUILD

### E1. `flee`/pathfinder cannot leave water — the escape is unexecutable from a submerged start

**Evidence (read in `scripts/minecraft_bridge/node_modules/mineflayer-pathfinder/lib/movements.js`,
pathfinder v2.4.5):** every move generator hard-returns on a liquid node —
- `getMoveForward`: `if (blockC.liquid) return // dont go underwater` (line 496); a second
  `// dont go underwater` at line 516 in `getMoveDown`;
- `getMoveUp`: `if (block1.liquid) return` (line 525) — a bot standing IN water generates no "up" move;
- `getMoveParkourForward`: `if (this.getBlock(node,0,0,0).liquid) return // cant jump from water` (line 564);
- landing blocks adjacent to liquid are rejected (lines 259–263, "false if next to liquid").

The constructor (lines 20–80) has **no `allowSwimming`/`canSwim` option** and never sets a jump/swim
control state. `flee` (`index.js` case `"flee"`) builds a plain `new Movements(bot)` with
`canDig=false` and calls `bot.pathfinder.goto(GoalNearXZ(anchor))`. A bot teleported underwater is
sitting on a liquid node, so the move generators produce **zero neighbours** → `goto` throws NoPath /
times out. This is architectural, not tunable: **any** submerged start (even a walkable up-ramp,
because the START node is liquid) kills the pathfinder. A `canDig=false` flee cannot break out either.

**Consequence:** the ONE corrective affordance can't run in the apparatus. Both FEAR and ABLATED never
surface under their own power → every placement censors → no latency contrast → **null-with-cause**.
This is exactly the Exp 58 pattern (mechanism fires, apparatus can't express it), moved from "clusters
won't separate" to "the act can't execute." Building on `flee` guarantees a non-result.

**Minimal fix (mandatory before freeze):** add a NEW param-free `surface` affordance that bypasses the
pathfinder — `bot.setControlState('jump', true)` (jump = swim-up in water) held until the bot reaches
air / `bot.oxygenLevel` recovers, then release. Direct control is reliable where the pathfinder is not.
This is NOT the prereg's parked "reuse flee-up vs a new surface action" choice — **flee is dead in
water, so a new actuator is not optional.** It needs (a) its own live swim-out actuation check (the
Exp 58 flee-actuation-check analog: one real executor `surface` must raise `oxygenLevel` from a
submerged start, or the seed refuses), and (b) wiring to the threat need's consumer / `recommend_action`
so BOTH arms can execute it (wiring lens owns the read path; environment lens owns that this is the
only physically-viable escape). Do not build the harness until that actuator exists and its actuation
check passes live.

---

## SHOULD-FIX

### E2. Reactive (ABLATED) surfacing may not fit before drowning death — DV must isolate the pre-damage window

**Evidence:** 1.20.4 mechanics — ~15 s of air (oxygen 20→0), THEN drowning damage 2 HP/s. From full
20 HP the bot dies ~10 s after damage onset → **death at ~25 s** from submersion. The state loop is
4 Hz (bridge 500 ms interval, pump 250 ms) → ~0.5 s latency resolution, adequate. But the innate
(ABLATED) response cannot begin until damage starts (15 s), leaving only ~10 s for reaction + swim-out
before death → heavy censoring and a compressed/artefactual contrast (ABLATED always dead/censored,
FEAR always surfaces — a difference inflated by the death cap, not a clean latency delta).

**Consequence:** the primary DV is confounded by the death envelope; the measured effect could be a
censoring artefact rather than an anticipation delta.

**Minimal fix:** define the DV as **"surfaces during the 0–15 s pre-damage air-depletion window."**
FEAR acts anticipatorily on the learned underwater cluster *before any damage*; ABLATED has no driver
until damage at 15 s. This both isolates learned-from-innate (the confounding lens's concern, with an
environment face) AND fits inside the death envelope. Pin a fixed **rescue-teleport cap (~18–20 s,
below the ~25 s death)** so no placement ends in death (avoids respawn-location surprises), and record
a rescue as **CENSORED, never as an agent surface** — the rescue teleport restores oxygen exactly as a
real surface would, so the harness must distinguish agent-surface (oxygen rose because the bot reached
air under its own power) from rescue-surface (harness teleported it out).

### E3. Measure separability on THIS water apparatus before building the full harness (verify-the-instrument)

**Evidence:** geometrically the cue is over-determined, unlike dark=danger. `oxygen` swings a full
half-range 0.5→0.0; the A4 gain weight `(|v−0.5|·2)³` goes **0 → 1.0** — the loudest possible
single-sensor move (contrast: Exp 58's `nearest_hostile_dist` moved 0.09, cos 0.977, unfixable). And
submersion co-moves several other sensors (`on_ground` 1→0, `y_altitude` drops, `speed`, `look_pitch`),
all ADDING separation, where the cave held everything else constant. So the preflight is very likely to
PASS. But Exp 58's Phase-0 read 1.0/1.0 in a full-range box and still collapsed in the live classroom
(false confidence — the twice-learned lesson).

**Consequence:** skipping the live measure risks a third "separated offline, diluted live" surprise.

**Minimal fix:** run the existing `scripts/survival_world/l11_geometry_probe.py` (`capture` on
safe-surface vs submerged states on the BUILT water apparatus, then `analyze`) as the cheap
`authorizes_build` gate BEFORE building the trial harness — same staged discipline the L11 plan used.

### E4. Build hygiene — prevent drowned-mob, water-drain, chunk-relocation, and rescue-into-wall confounds

**Evidence + fixes (each an Exp 58 failure mode or a Minecraft water gotcha):**
- **Drowned spawns:** at block-light 0, 1.20.4 water spawns DROWNED zombies → a second pain source and a
  hostile-cluster axis that contaminates the drowning-pain measurement and can shift cluster identity
  mid-arm. Set `doMobSpawning false` (the cave builder already does this) and add no spawner.
- **Source vs flowing water:** `/fill x1 y1 z1 x2 y2 z2 minecraft:water` places a SOURCE block in every
  cell (stable, won't drain) **only if the pool is walled** — an open SIDE lets edge water flow out and
  the column drains. Encase in a stone shell (like the cave's `fill … minecraft:stone`) then fill the
  interior with water. An open TOP is fine (source water exposed above does not spread) and is the
  natural surface escape. No soul-sand/magma → no bubble columns pulling the bot up/down.
- **Chunk relocation:** `forceload add` the water-column chunks BEFORE teleporting the bot in (the
  world-spawn-relocation lesson from the cave build) so the column is resident.
- **Rescue target:** the rescue/`safe` teleport must land on a DRY air platform — teleporting into water
  or a wall re-drowns or suffocates the bot. Keep `doImmediateRespawn` + `keepInventory` (harness
  gamerules) so a death (if any slips past the cap) doesn't park the AUT on a respawn screen.
- Record the full water geometry to an anchor file (surface y, floor y, submerged teleport target, dry
  rescue platform) like `exp58_classroom.json`, and drive the harness off the recorded truth, not the
  bot's live position.

---

## NIT

### E5. Oxygen scale confirmed in source — the prereg's numbers are right

`mineflayer/lib/plugins/breath.js` sets `bot.oxygenLevel = Math.round(metadata.value / 15)` ∈ **[0,20]**
(api.md: "Number in the range [0, 20]"). Surfaced = 300/15 = 20 → normalized 0.5 on the body's
`oxygen range: [0,40]` (rest = midpoint, A4-neutral); fully depleted = 0 → 0.0. The prereg's
"range [0,40], rest 20 = 0.5, descends to 0" is exactly correct. One live instrument-check state line
(submerge → watch oxygen tick 20→0 over ~15 s; surface → watch it restore) confirms before freeze; risk
low, source already matches. Note the metadata fires only on air CHANGE, so on land oxygen holds at 20
(bridge `?? 20`) — correct resting read.

### E6. Underwater `light_level` low and `on_ground` = 0 are expected, not faults

During the instrument check the submerged state will read low `light_level` and `on_ground` 0; these
ADD to separation and are correct — flag so they aren't mis-read as apparatus faults.

---

## Positive findings (the danger half is sound)

- **Submersion is stageable and the danger is real (D1-clean).** A submerged idle mineflayer bot sinks
  (net downward) and does **not** auto-float to the surface — no input, no `setControlState('jump')`,
  so it stays down, oxygen depletes, drowning damage lands. There is no auto-surface confound on the
  danger side.
- **"Surfacing" is genuinely an ACT, not a physics default** — the same idle-sink fact means the bot
  won't passively rise, so a measured surface reflects a commanded upward swim. Good for the claim; and
  the direct reason the escape needs an active jump-actuator (E1).
- **Escape restores oxygen game-natively** — reaching air/surface refills air_supply, so a real surface
  is measurable via `oxygenLevel` recovery (distinct from position).
- **The cue separates far more strongly than dark=danger** (E3) — the apparatus half Exp 58 lacked.
