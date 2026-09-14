# Exp 58 — ENVIRONMENT lens (design review, 2026-09-14)

**Charter:** does the world game-natively afford it (D1 — no synthetic sensor/reward), are the
needed states/acts reachable AND measurable, does the bridge/world behave the way the design
assumes? Reviewed against: the prereg draft (`docs/experiments/exp58_survival_wants_prereg.md`),
`docs/plans/survival_world_1_3.md`, `docs/wiring/world-light-sensing.md` +
`docs/wiring/sensor-range-clamps.md`, `scripts/survival_world/setup_world.py`,
`scripts/minecraft_bridge/index.js`, `bodies/minecraft_player.yaml`,
`docs/experiments/data/survival_phase0.json`, and real Paper 1.20.4 mechanics.

Reviewer confidence notes: findings below cite specific 1.20.4 mechanics. Most are
high-confidence vanilla rules; the two lower-confidence details (burning-mob melee ignition,
zombie through-wall targeting) are flagged inline and are NIT-grade only.

---

## DO-NOT-BUILD

### DNB-1 — "Spawn-control via light + region" does not exist game-natively as specified; on a normal world it delivers the wrong mobs, in the wrong places, at uncontrolled density

**What the prereg assumes (Apparatus, Claim B):** "hostile spawning enabled INSIDE the dark
zone only (spawn-control via light + region), mob type capped (zombies; survivable contact,
no creepers)."

**What 1.20.4 actually affords on the `level-type=normal` world `setup_world.py` builds:**

1. **The dark zone is not the only spawn region.** Natural caves riddle the underground
   everywhere (1.18+ caves are large), and every block-light-0 cave block is spawnable.
   This is already MEASURED: the first 1.20.4 snapshots read **26–30 hostiles at noon**
   (`docs/wiring/world-light-sensing.md`, "hostile_count counts through walls"). Consequences:
   - **Mob-cap starvation of the classroom.** The hostile cap (~70, shared across all
     spawnable chunks in range) is soaked by cave mobs. The classroom's dark room competes
     for the residue — the K ≥ 10 dark-damage episodes can silently under-deliver ("fewer
     than K usable episodes" stop rule fires as an apparatus failure, wasting the arm).
   - **Sensor contamination of the LIT cluster.** `hostile_count` (range −32..32, rest at
     midpoint) and `nearest_hostile_dist` read cave mobs through walls, so the "lit safe"
     situation carries a loud, *fluctuating* hostile channel — a direct threat to the
     Phase-0 dark/lit separability the stop rule re-checks, and to a stable lit-cluster
     identity for the specificity gate.
2. **No game-native mob-type cap exists via light or region.** A block-light-0 room in the
   overworld spawns the whole hostile roster: zombies, **skeletons** (15-block ranged arrows
   — they shoot OUT of the dark into the lit area: damage lands while the LIT cluster is
   active → fear written onto the lit cluster → the specificity gate `lit ≈ 0` fails for
   environmental reasons, indistinguishable from a mechanism failure), **creepers**
   (explosions destroy the classroom geometry — a hole is a silent sky-light leak that
   changes both the light sensor and the spawn surface), **spiders** (climb walls, exit the
   enclosure), and **witches** (poison; multi-tick damage decoupled from position). Geometry
   tricks (low ceilings) cannot select zombies-only.
3. **Uncontrolled density.** Hostiles within 32 blocks of a player never naturally despawn;
   with the agent stationed by the classroom, pack spawns accumulate toward
   `MaxNearbyEntities`-free natural limits. On normal difficulty a zombie deals 3 hp/hit
   about once per second in contact: 2 zombies ≈ 6 hp/s → full health to dead in ~3.5 s.
   "Survivable contact" is a density claim, and natural spawning gives no density knob.

**Failure scenario:** the FEAR arm runs; episodes are a mix of skeleton snipes into the lit
area, a creeper hole in the roof, and 3-zombie packs; the agent dies repeatedly and/or the
lit cluster accrues negative valence; gates fail (or pass) for reasons that are entirely
apparatus, and the headline learning claim is uninterpretable.

**Fix shape (concrete, D1-compatible, all vanilla):**
- `gamerule doMobSpawning false` (kills ALL natural spawning: cave contamination, cap
  competition, and surface risk in one line) **plus a vanilla zombie monster-spawner block**
  (`/setblock` with NBT) inside the dark room. Monster spawners are unaffected by
  `doMobSpawning`, and they provide exactly the missing controls:
  - **type** — spawns only its configured mob; `SpawnData` can pin adult zombies
    (`IsBaby:0` — relevant because ~5% of natural zombie spawns are babies, which are
    faster AND do not burn in sunlight, defeating DNB-2's containment);
  - **density** — `MaxNearbyEntities` (e.g. 1–2) is a hard local cap; `SpawnCount`,
    `Min/MaxSpawnDelay`, `SpawnRange` tune rate;
  - **locality/proximity** — `RequiredPlayerRange` (default 16) means zombies appear only
    while the agent is near the dark zone: episode onset is naturally gated on approach;
  - **the dark contingency stays game-native** — a spawner spawning hostile mobs still
    requires the mob's light conditions (block light 0 since 1.18), so lighting the room
    stops it: "dark → zombies" remains a real game mechanic, not a harness injection.
- Belt: an RCON kill-sweep (`kill @e[type=!player,type=!item,...]`) at arm boundaries so
  arms start from a mob-free world.
- The prereg's Apparatus paragraph should be rewritten to name this mechanism (or an
  equivalent) instead of the non-existent "light + region" control, and the Phase-0
  re-check on classroom geometry should include a spawn check (zombies appear in the dark
  room within N minutes with the agent adjacent; zero spawns in the lit area).

### DNB-2 — Mob pursuit puts damage in the LIT cluster: contact damage happens where the AGENT is, and zombies chase

**What the prereg assumes:** damage episodes are "mob contact in the dark zone", and the
specificity gate expects lit-cluster valence ≈ 0.

**What the game does:** a zombie that acquires the agent in the dark pursues it out of the
zone (follow range ~35 blocks). Every hit lands at the agent's current position — i.e. in
whatever cluster is active AT PAIN TIME. An agent that gets hit in the dark, flees, and is
caught at the cave mouth or inside the lit area writes fear onto the boundary/lit cluster.
This is not noise: flee-and-get-caught is the *expected* trajectory for exactly the
avoidance behaviour the experiment wants, so the contamination is systematic, and it
corrupts both the specificity gate and the mechanism DV (dark-cluster valence trajectory).

**Failure scenario:** FEAR arm trains; half the cumulative damage lands during pursuit in
lit/boundary states; the lit cluster ends at valence −0.3; the specificity gate fails →
the run reads as "the mechanism is not situation-specific" when the environment never
delivered situation-specific pain.

**Fix shape (game-native containment, layered):**
- **The lit safe area must be SKY-EXPOSED at the frozen day** (not torch-lit indoors):
  adult zombies exposed to daylight sky ignite and die in seconds — pursuit into the safe
  area is self-limiting by a real game mechanic. (Requires DNB-1's adult-only pin: baby
  zombies don't burn. Lower-confidence detail, NIT-grade: a burning zombie's melee may
  briefly ignite the player — a hit or two can still land during burn-down; the geometry
  point below absorbs this.)
- **A pursuit barrier the agent can cross but zombies cannot**: e.g. a closed wooden door
  (zombies cannot open doors, and only break them on HARD — difficulty is normal) or a
  ledge/trapdoor step, between the dark room and the lit area. Verify the bot's pathfinder
  can actually traverse the chosen barrier (mineflayer's default `Movements` handles doors
  poorly — test in the wiring smoke), else use geometry (a 1-block step up that zombies
  path around slowly) plus distance.
- **Episode accounting by pain-time cluster:** the harness should record which cluster was
  active at each damage event and count an episode "usable" only if the breaching damage
  landed with the dark cluster active — turning residual contamination into a measured,
  excludable quantity instead of a silent gate-killer.

---

## SHOULD-FIX

### SF-1 — Weather is not frozen: a thunderstorm makes the daytime surface spawnable and rain breaks sunlight containment

`setup_world.py::_GAMERULES` freezes time (`doDaylightCycle false`) but NOT weather. On
1.20.4 with the weather cycle running: during a **thunderstorm the sky darkens enough that
hostile mobs can spawn on the surface in daytime** — the "lit area is naturally
spawn-proof" premise silently fails for the storm's duration; during **rain, burning
zombies are extinguished** — DNB-2's daylight containment fails; and the `is_raining`
sensor flips (range −1..1, rain "shouts" per the body YAML), shifting the world-cluster
identity mid-training for reasons unrelated to any contingency. The bridge's
`perceivedLight` also does not model weather darkening, so the sensed light and the
spawn-relevant light diverge in storms.
**Fix:** add `gamerule doWeatherCycle false` + an initial `weather clear` to `prepare`,
and check it in `verify`. One line each; this closes three holes at once.
(If DNB-1's `doMobSpawning false` is adopted, the storm-spawn half is moot, but the
rain-extinguish and cluster-identity halves still stand.)

### SF-2 — One zombie hit does NOT breach the comfort band; a usable episode needs ≥3 hits, and the breach-to-death window is ~4 more

Health drive: set_point 20, `comfort_band 6.0` → pain fires below 14 hp (and the probe's
iteration history shows damage landing exactly ON the band reads as NO pain). A normal-
difficulty zombie deals **3 hp per hit**: 20→17 (no pain), →14 (band edge — the exact
false-negative trap), →11 (breach). So a "damage episode" is structurally a **multi-hit
engagement (≥3 hits, ~3 s of contact or 2 zombies)** — and death arrives at ~7 hits from
full health. The prereg's stop rule checks episode damage against the band post-hoc, which
is correct but reactive; the exposure protocol should be designed for this ex ante.
**Fix:** define "usable damage episode" in the prereg as *cumulative* health ≤ 13 reached
while the dark cluster is active (not "a contact event"); design the engagement window to
allow ≥3 hits before disengagement is possible; note the survivable corridor (breach at
hit 3, death at hit 7 — the spawner density cap from DNB-1 is what keeps the corridor
open). Do NOT fix this by raising difficulty to hard (4.5/hit breaches in 2 hits, but hard
enables zombie door-breaking — which would remove DNB-2's barrier — and raises spawn
pressure).

### SF-3 — No health-recovery path is specified for the cave classroom; without one, "probe windows with health ≥ 18" starve out

Natural regeneration requires **food ≥ 18** and consumes hunger as it heals (~1 hp/4 s).
The Claim B classroom seeds no food (only the Claim A dining hall does), and K ≥ 10
multi-hit episodes cost ≥ 70 hp cumulative. Without a recovery lane the FEAR arm's food
drops below 18, regen stops, health ratchets down, and the post-training probe windows
(gated on health ≥ 18) become unreachable — the primary DV loses its denominator.
**Fix (pick one, disclose it):** (a) seed bread in the Claim B arm too — the agent has
`eat` and the break-1 deficit prior, so recovery is game-native and even exercises the
composed loop; or (b) a disclosed RCON lane between episodes
(`effect give ... minecraft:regeneration` / `minecraft:saturation`), labelled exactly like
Claim A's hunger induction. Either way, add "agent can return to health ≥ 18 between
episodes" to the wiring smoke.

### SF-4 — Death/respawn handling is unspecified: respawn is at WORLD SPAWN and resets health/food to 20

`keepInventory` + `doImmediateRespawn` are correctly set (verified clean below), but with
no bed/`spawnpoint`, a death **teleports the agent to world spawn** — potentially far from
the classroom, outside the lit/dark geometry, with health and food reset to 20 — a
mid-schedule discontinuity the prereg never accounts for. SF-2 shows deaths are one
mis-timed engagement away.
**Fix:** in classroom setup, RCON `spawnpoint maxim <lit-safe-area coords>` so a death
lands the agent in the lit safe area (a game-native "you wake up somewhere safe" —
also keeps offsets in-range); define in the prereg whether a death (i) counts as a damage
episode (pain fires on the way down), (ii) truncates the current probe/training window,
and (iii) caps the arm (e.g. > M deaths → apparatus stop, density control failed).

### SF-5 — Anticipatory-read timing vs the zone boundary: the fear can only steer BEFORE entry if the boundary state activates the feared cluster

The read path fires when an **ACTIVE** cluster carries valence ≤ −θ, and the dark cluster
activates from *sensed* features — i.e. potentially only once the agent is already in the
dark. Perceived light at a cave mouth falls off gradually (sky light attenuates over
~10–15 blocks inward), so the approach path traverses intermediate-light states. If those
boundary states encode to a *different* cluster (valence ≈ 0), the threat need fires only
after entry — avoidance degenerates to enter-then-retreat, `P(enter dark)` does not drop,
and the experiment returns a **false null against a working mechanism**.
**Fix:** (a) define the measured "dark zone" boundary at a perceived-light threshold that
sits *inside* the natural gradient (so the cluster flips before the zone line is crossed);
(b) extend the mechanism instrument check (the probe re-run the prereg already gates on)
with a boundary-state check: does the cave-mouth state (light ~3–7) map to the same
cluster as the deep-dark damage state, or to a neighbour? If a neighbour, either the
geometry or the DV definition must move. This is cheap to check offline with the scripted
bridge and expensive to discover post-run.

### SF-6 — Claim B's exposure schedule vs baseline: "free-roam" must actually produce both dark entries AND a measurable pre-training P(enter)

The gate is relative ("≤ 0.5 × its own pre-training baseline"), so the baseline must be a
real, nonzero, stable rate — and the training schedule needs the agent in the dark often
enough for K ≥ 10 multi-hit episodes. Nothing in the environment *pulls* a naive agent
into a cave except the explore-bonus novelty nudge; if free-roam entry is rare, training
stalls AND the baseline is too noisy to halve meaningfully.
**Fix:** the prereg should state which of the two protocols training uses — (a) genuinely
free-roam (then pre-pilot the naive entry rate and size the baseline windows for power),
or (b) shepherded exposure (harness-scripted `move_to` into the dark for training
episodes, disclosed like the hunger lane) with free-roam reserved for the pre/post PROBE
windows measured identically in both arms. (b) is the robust shape; it just has to be
named so the "avoids the dark" claim is scoped to the free-roam probes.

### SF-7 — Claim A's satiated-side induction has no named lane: vanilla has no "set food" command

Deficit induction is disclosed (`effect give ... minecraft:hunger` — exists, calibration
knobs present in `prepare --induce-hunger`). But alternating back to "satiated (food ≥ 16)"
needs food to RISE, and vanilla RCON cannot set the food stat directly; the game-exposed
lane is `effect give ... minecraft:saturation` (restores food/saturation per tick) — or
letting the agent eat, which contaminates the selection DV's independence. Also: RCON
effects reach the sensed `food` via server packets with lag (the eat-lag lesson
generalizes), and the sensed value is the truth (sensor-range-clamps rule).
**Fix:** name the saturation-effect lane in the prereg with the same disclosure framing as
hunger; have the harness settle-poll the SENSED food against the 6/16 thresholds (via the
bridge state, 500 ms cadence) before starting each decision cycle; remember the
**saturation buffer** on the deficit side too — the hunger effect drains saturation before
the food bar moves, so induction duration/amplifier must be calibrated against the sensed
food, not wall clock.

### SF-8 — `move_to` traps: GoalNearXZ is Y-agnostic and default Movements DIG

Two mineflayer facts that bite this specific design: (1) the bridge's `move_to` uses
`GoalNearXZ(x, z, 1)` — an XZ-plane goal with **no Y component**, so a target "in the lit
area" directly above a cave passage can be satisfied underground (the bot reads
`light_level 0` at a position the harness calls "lit" — zone bookkeeping and cluster both
disagree with intent); (2) the default `new Movements(bot)` allows **digging and
scaffolding**, so a pathfinder route may tunnel through the classroom wall — a permanent
light/spawn leak in the frozen apparatus, created silently by the agent itself
(`mine_block` in the repertoire can do the same on its own).
**Fix:** for the Claim B classroom, configure Movements with `canDig = false` (bridge-side
flag or a harness-declared variant); place avoidance/approach waypoints so their XZ
columns are unambiguous (no cave under the lit waypoint); and state in the prereg what the
move candidate set IS (e.g. a fixed lit-waypoint / dark-waypoint pair) — avoidance is only
"expressible" (charter question 4) if a move-away candidate with concrete params is
actually on the menu each cycle. Keep waypoints within ±128 of spawn so `offset_x/z`
stay un-clamped.

### SF-9 — Local (regional) difficulty accrues in the classroom chunks: an order confound between FEAR and ABLATED arms

1.20.4's local difficulty rises with chunk *inhabited time*. Running the FEAR arm first
raises the classroom chunks' local difficulty for the ABLATED arm: higher chances of
armored/equipped zombies and reinforcement spawns → systematically different damage per
episode in the second arm. Fresh agent per arm does not reset the WORLD's per-chunk clock.
**Fix (either):** build two identical classroom instances in previously-unvisited chunks,
one per arm; or counterbalance arm order across seeds and record per-episode damage so
the drift is measurable. Cheapest robust shape: fresh world copy per arm from the seeded
setup (the `world_version.json` stamp guard already makes re-setup cheap and reproducible).

---

## NIT

- **N-1** `hostile_count`/`nearest_hostile_dist` still count spawner zombies through the
  classroom wall (within 32/64-block sensor range) even after DNB-1 — the lit cluster will
  carry hostile_count 1–2 whenever the spawner is warm. Bounded by `MaxNearbyEntities`, so
  it is stable rather than fluctuating; check it survives the Phase-0 separability re-check.
- **N-2** Add `gamerule mobGriefing false` as a belt for any stray creeper/enderman terrain
  edits (moot for spawn-side after DNB-1, but endermen can still be summoned by nothing —
  harmless line, protects the frozen geometry).
- **N-3** `attack_nearest` kills grant XP → `xp_level` sensor rises over training (range
  −50..50, so a few levels ≈ 0.02 normalized — negligible, but note it in the frozen-
  apparatus fingerprint rather than being surprised by slow cluster drift).
- **N-4** The bridge `eat` filter is `i.name.includes("bread") || i.foodPoints` —
  `foodPoints` is very likely undefined on mineflayer `Item` objects (it lives in
  minecraft-data's foods table), so the filter is effectively bread-only. Fine while
  `prepare` seeds bread; revisit before any diet change (e.g. SF-3 option (a) with a
  different food).
- **N-5** Phase-0's own dark box sat at y0 = 149 — ABOVE the body's y_altitude clamp
  (`[0, 128]`), so the sensed dark state included a pinned-at-cap altitude. The Claim B
  cave should keep ALL discriminating states inside declared ranges (cave at surface-ish
  y ≈ 40–70, classrooms within ±128 offsets of spawn) per the sensor-range-clamps rule —
  and note the classroom-geometry Phase-0 re-check will re-establish separability on the
  new geometry anyway (the prereg's stop rule already requires this).
- **N-6** `bodies/minecraft_player.yaml`'s header comment says "16 world sensors" while the
  file declares 17 (and Phase-0 records 17) — comment drift only; fix opportunistically.

---

## Verified clean (environment assumptions the prereg gets RIGHT)

1. **The 1.18+ spawn rule is leaned on, not synthesized.** With time frozen at day
   (`doDaylightCycle false` + `time set day` in `prepare`), hostile spawning requires
   block light 0 AND no effective sky light — so a fully-enclosed dark room/cave is
   game-natively the only spawn surface and the sky-lit area is naturally spawn-proof
   (modulo SF-1's weather hole). Darkness-as-mob-source is a real mechanic, no synthetic
   gate. (What's broken is the *type/density/region* control claimed on top of it — DNB-1.)
2. **Claim A's DV is read at the recommendation stage** precisely because Minecraft
   refuses eating at food 20 — the eat-at-cap trap is explicitly designed around rather
   than laundered into a fake satiated gate. Correct and well-argued in the prereg text.
3. **Light sensing is the fixed perceived-brightness read**, live-verified on 1.20.4
   (Phase-0: day min 15.0 / dark max 0.0, 0 settle-timeouts), and the design's thresholds
   are being written against 1.20.4 mechanics rather than ported from 1.16.5 — exactly what
   `world-light-sensing.md` prescribes.
4. **The band-edge trap is acknowledged in the stop rules**, and the mechanism instrument
   (dark_danger_probe) already encodes the lesson (its own iteration history found
   damage == band reads as no pain; it uses 12 damage). SF-2 only asks the live episode
   definition to inherit this ex ante.
5. **`keepInventory` + `doImmediateRespawn`** are already in `_GAMERULES` with the right
   rationale (death must not strip seeded food or park the AUT on a respawn screen). SF-4
   is about *where* respawn lands, not whether it happens.
6. **The hunger-induction lane exists and is honestly labelled** (`effect give
   minecraft:hunger`, `prepare --induce-hunger`, disclosed as harness-injected per the
   2026-09-12 decision; the claim is scoped to the mechanism, not "the environment taught
   it"). SF-7 extends the same honesty to the satiated direction.
7. **Zone occupancy is measurable game-natively and in-range:** harness-side via RCON
   position query, agent-side via `light_level` (range [0,15], the discriminating states 0
   vs 13–15 measured well inside it) and the feature-based world channel; the bridge
   streams state at 500 ms — adequate resolution for P(enter) and time-in-dark. The
   *timing* of cluster activation (SF-5), not the measurability, is the open risk.
8. **The feature-based world channel makes the learned fear portable by construction**
   (keys on `light_level`/hostiles, not coordinates) — consistent with the classroom-
   topology decision in `survival_world_1_3.md`, and the prereg correctly refuses the
   cross-context generalization claim (R1 stands).
9. **Bread + the seeded-inventory flow is live-validated** (break-3 smoke; `verify`'s
   dry-run `clear` check is non-destructive and matches the failure string, not a vacuous
   substring), and Claim A's 6/16 thresholds match the body's declared
   deprivation/satisfaction thresholds and sit inside the sensed food range.
10. **The avoidance behaviour is expressible in the repertoire in principle** — `move_to`
    (pathfinder) + `turn` + staying in lit are sufficient motor vocabulary for the claimed
    behaviour; SF-8 is about pinning the candidate parameterization and pathfinder config,
    not a missing affordance.

---

## Summary for the fold

The two DO-NOT-BUILDs are the same root: **the prereg asserts spawn controls (dark-zone-
only, zombies-only, survivable density) that a normal 1.20.4 world does not game-natively
provide, and mob pursuit moves the damage out of the zone the fear is supposed to key on.**
Both have a clean, vanilla, D1-compatible fix (`doMobSpawning false` + a configured zombie
spawner + a sky-lit safe area + a pursuit barrier + pain-time cluster accounting) that
arguably makes the contingency SHARPER game-natively (light the room → spawner stops).
The SHOULD-FIXes are one-line world rules (weather), episode/recovery/death/baseline
protocol definitions the prereg currently leaves implicit, one measurement-timing check to
add to the already-gated probe re-run (SF-5 — the cheapest false-null insurance in the
design), one induction lane to name (saturation), pathfinder config, and an arm-order
confound (local difficulty). Nothing here argues against the experiment's shape; it argues
the classroom must be built against the mechanics 1.20.4 actually has.
