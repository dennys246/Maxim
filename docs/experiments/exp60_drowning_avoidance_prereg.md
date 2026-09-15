# Exp 60 (DRAFT, not frozen) — learned drowning-avoidance: Wire-4 situation-fear on a separable cue

> **STATUS: DRAFT for four-lens experiment-DESIGN review (2026-09-15). NOT frozen, NO harness.**
> The pivot from Exp 58 (dark=danger, blocked): the SAME Wire-4 situation-fear mechanism — which
> we proved *fires* (dry-run: fear accumulates, flee executes) — applied to a survival cue that
> ACTUALLY SEPARATES on today's substrate. Exp 58 died at the instrument (dark/safe wouldn't
> cluster-separate; verified unfixable by encoding remedy — `docs/plans/l11_slice2_channel_split.md`).
> Drowning is the fix for the apparatus half: `oxygen` (modality world, range [0,40], rest 20 =
> normalized 0.5) descends to 0 underwater — a full half-range swing to an extreme (gain weight
> 0→1.0), so "underwater/low-air" is a genuinely distinct world cluster. This prereg is the
> brainstorm the four-lens review reads; it is NOT the frozen prereg.

## Four-lens review outcome (2026-09-15): VIABLE, but needs 3 additions + a re-cut DV

All four lenses returned DO-NOT-BUILD **as drafted** — but unlike the Slice-2 channel-split
(unfixable), Exp 60 is FIXABLE, and they converge on one buildable design. The cue is a genuine
upgrade over dark=danger (measured below). Rationale: `docs/experiments/rationale/exp60-drowning/`.

**Measured (offline, real encoder bases — the verify-the-instrument gate, done early):** the raw
oxygen swing separates only at the BOTTOM of the dive —

| oxygen (of 40) | cos(shore, state) | pre-damage? |
|---|---|---|
| 20 → 4 (the whole ~15s air window) | 1.000 → 0.936 | same cluster as shore (no fear) |
| 2 | 0.872 | same |
| 0 (damage onset) | 0.779 | distinct |

So the anticipation window is EMPTY on oxygen alone — the feared cluster == the damage state
(confounding-DNB-1, wiring-SF-1). **Fix, also measured: a binary `isInWater` sensor makes the
underwater cluster distinct from dive-second-0** (cos 0.792 at full air → 0.685 depleted), giving a
stable pre-damage cue.

**Required additions before freeze (cross-confirmed across lenses):**
1. **`isInWater` binary world sensor** (mineflayer exposes it; D1-legal) — the stable pre-damage
   underwater cluster. *Verified it separates from second 0.* [confounding, wiring, environment]
2. **An air-hunger / oxygen drive** — oxygen has NO `drive:` block today (only health/food), so
   drowning reaches Wire-4 only as generic `drive:health` at damage onset. A breath drive makes
   asphyxia publish pain EARLY and drowning-specific; add `drive:oxygen` to
   `NACConfig.cluster_fear_failure_modes`. [bio-faithful, wiring]
3. **A `surface`/`escape_water` actuator that BYPASSES the pathfinder** — `flee` is DEAD in water
   (mineflayer-pathfinder hard-returns on liquid nodes, verified in source; `GoalNearXZ` ignores Y).
   Use `bot.setControlState('jump', true)` held until surfaced; add the name to
   `_DRIVE_TOOL_AFFINITIES["threat"]` or the read path emits nothing (Exp 58 dead-path bug); ship a
   live swim-out actuation preflight. [environment-E1, wiring-DNB-2]
4. **DV re-cut to the 0–15s pre-damage air window** — primary = P(surface before first damage tick)
   + latency censored at first damage; rescue-teleport cap ~18–20s (below ~25s death), rescue
   recorded as CENSORED (rescue restores oxygen like a real surface — must distinguish). [confounding, environment]
5. **Live separability gate** — run `scripts/survival_world/l11_geometry_probe.py` on the built water
   pool (shore vs submerged) as the `authorizes_build` gate before the full harness. [all lenses]
6. **Build hygiene** — `doMobSpawning false` (water spawns DROWNED zombies at block-light 0 = a
   second pain source), fully-walled water column (source blocks; open top = the escape), forceload
   before teleport, rescue onto a DRY platform, geometry anchor file. [environment-E4]
7. **Confounding guards** — no-damage probe (full-heal, mob-free, full-air submersion), yoke exposure
   across arms, specificity gate (`|shore_fear| < 0.2·|water_fear|`). [confounding]

**Positives the review confirmed:** idle bot sinks (drowning real, no auto-float); reaching air
restores oxygen game-natively; substrate has headroom (no oxygen drive, pathfinder doesn't
auto-surface); Wire-4 read path is LIVE (`threat→flee` wired, unlike Exp 58's first pass). Front-gate
scope: the drive + sensor + actuator are justified game-native/bio-faithful additions (asphyxia is a
real interoceptive alarm; in-water a real perceived state; surfacing a real act), not gold-plating —
but they DO make this a real build, not a config tweak.

## Apparatus (chunk i, built 2026-09-15 — pending the live apparatus check)

`setup_world.py water_classroom` (pure geometry `water_classroom_geometry`, unit-tested in
`tests/unit/test_exp60_water_classroom.py`) builds ONE stone-encased room at the Exp 58 depth
band (`SHORE_Y=40`) — deliberately the band the offline cos≈0.79 estimate was computed on
(light 0, frozen day, y≈40), so the live vectors match the estimate's base vector; a surface
pool would change `light_level` and re-open the "estimated on a different geometry" trap.

| Element | Blocks (anchor `ax, az`, depth `D=5`) | Purpose |
|---|---|---|
| Stone shell | x [ax−6, ax+9], y [37−D, 45], z [az−4, az+4] | overwrites natural caves/water; every pool face is stone |
| Air chamber | x [ax−3, ax+5], y [40, 42], z [az−1, az+1] | 3-high headroom over shore + pool |
| Shore | dry stone top y=39 under x [ax−3, ax+1]; anchor (ax−1, 40, az) | rest/rescue target, spawnpoint |
| Lip | (ax+2, 39) stone | keeps the shore floor and the top water layer apart (no flow) |
| Pool | x [ax+3, ax+5], y [40−D, 39], z [az−1, az+1], `minecraft:water` | 45 SOURCE blocks, 3 wide, D deep, open top |
| Submerged target | (ax+4, 40−D, az) | feet on the floor, head at 41−D in water: dive-second-0 reads `is_in_water` 1 with full air |
| Surface cell | (ax+4, 40, az) | the reachable air, D−1 = 4 blocks above the submerged head |

Hygiene (environment E4, folded): `forceload add` first; `doMobSpawning false` set by the builder
and treated as APPARATUS-OWNED (the Phase-0 instrument check restores it to `true` on exit, so
the water check VERIFIES it and refuses rather than toggling); `spawnpoint` on the shore;
`exp60_deaths` deathCount objective; no spawner/clustermob/soul sand/magma. Every `fill` reply is
checked, then after a 2 s fluid-tick pause ten `execute if block` assertions must pass (submerged
head cell is water; surface/shore cells air; shore floor, pool floor, lip stone; bottom/top/corner
pool cells `water[level=0]` = sources, not drained). **Placement guard:** the bridge caps
`nearest_hostile_dist` at 64 and 64 is that sensor's neutral midpoint, so the pool centre must be
≥ 72 blocks (horizontal) from both the recorded Exp 58 anchor and its `dark` point (the persistent
clustermob) or both water situations carry constant hostile mass; the builder refuses otherwise.
**Second placement guard (architecture-lens fold, replayed on the real encoder bases —
`docs/experiments/data/exp60_spawn_distance_check.py`):** `distance_from_spawn` is the same sensor
class the other way — a 3D distance to WORLD spawn (the login-packet spawn; `/spawnpoint` never
moves it) capped at 128 on range [−128, 128], so far from spawn it is a full-weight CONSTANT the
offline estimate never modelled (its base vector sat 36 blocks from spawn). Replayed cos(shore,
submerged at full air): 0.786 @36, 0.794 @90, 0.802 @100, 0.834 @120, **0.8525 @128 = same
cluster**. Bound: the submerged target ≤ 90 blocks (3D) from world spawn — checked by the builder
when `--spawn-x/y/z` is given (spawn is not readable over RCON) and ALWAYS gated live by the
check's W1 on the sensed value. (`offset_x/z` are bridge-only, not body sensors.) The Exp 58 cave
sits ~36 from spawn, so both guards are jointly satisfiable on the far side of spawn from it.
Recorded truth: `~/.maxim/exp60_water_classroom.json` (`shore`, `submerged`, `surface_y`,
`depth`, `pool`, `forceload`, `deaths_objective`, `probe_situations{shore,submerged}`) — the
check, the probe (chunk ii, `--anchor-file`) and the harness drive off the record, never live
position.

**Live apparatus check** `scripts/survival_world/exp60_water_check.py` (gated evidence →
`docs/experiments/data/exp60_water_apparatus.json`; 3 cycles, every gate on every cycle; a bridge
that stops delivering fresh state is an INSTRUMENT error, never a measured FAIL): W1 shore
baseline settles dry/grounded/sensed oxygen ≥ 19/health 20 and GATES `nearest_hostile_dist` ≥ 64
and `distance_from_spawn` ≤ 90 (`hostile_count`, `light_level`, `time_of_day` recorded, not gated —
`hostile_count` counts every hostile the SERVER tracks, a tracking-range fact the placement guard
does not control; one far mob is separability-inert); W2 dive from the floor (`is_in_water` 1
within 3 s; idle bot holds at the floor for 6 s — no auto-float; oxygen monotone through the pain
edge ≤ 13 bubbles, oxygen-zero time recorded; ONE drowning-damage tick lands — rarely two, 4 Hz
sampling vs the 1 s tick — so the damage-onset edge of the DV window is MEASURED (~16 s: damage
starts when air reaches −20, one second after 0), gate [12, 20] s; health ≥ 16 at rescue; the
check's own rescue cap is 22 s — the HARNESS's rescue cap is chunk (iii)'s and stays at E2's
18–20 s); W3 rescue restores sensed oxygen ≥ 19 within 10 s; W4 the real registered
`*_escape_water` tool through `aut.executor.execute` (production consumer) puts the head in air
within 6 s by bridge truth (`is_in_water` 0; the actuator now holds a 600 ms breath after the
head clears so the surfaced state outlives one 500 ms snapshot — before this fold it released
jump the same tick and the surface was unobservable; the backend now forwards the bridge's own
"surfaced"/"capped" string as `metadata["detail"]`, recorded beside the truth, never gated on);
W5 sink-back time (informational — the harness's rescue budget; `is_in_water` reads the head
BLOCK, ~0.7 blocks below the eye height the game breathes at, so it reads sink-back early =
conservative). **Definition carried into chunk (iii): "surfaced" = the first bridge read of
`is_in_water` 0 after the dive (momentary); the harness rescues to the shore on that read.** On
PASS the check stamps `measured{t_damage_onset_min/max_s, t_surface_max_s, t_sinkback_min_s,
distance_from_spawn}` into the anchor record so chunks (ii)/(iii) budget their dives from measured
truth (chunk ii's probe must sample within `t_damage_onset_min_s` minus margin and rescue). The
run-authorizing gate remains chunk (ii): `l11_geometry_probe` shore vs submerged on this pool,
cos < 0.85 + distinct frozen-EC ids, measured live.

## The claim

A survival agent LEARNS to escape the drowning situation — it surfaces / leaves water sooner after
experiencing oxygen-deficit pain than a fear-ablated control, driven by Wire-4 cluster-fear booked
onto the underwater world cluster (anticipatory threat need → corrective surfacing), NOT merely by
the innate reactive response to ongoing damage.

- **Arms:** FEAR (Wire-4 active) vs ABLATED (Wire-4 zeroed), fresh agent per arm, frozen seeds.
- **Primary DV (the learned, anticipatory component):** latency to leave water / time-in-water on a
  fresh submersion AFTER conditioning — FEAR should surface *earlier* (before deep deficit) than
  ABLATED. The DV must isolate ANTICIPATION (acting on the learned cluster-fear) from innate
  damage-response (both arms feel drowning damage; only FEAR carries the learned situation-fear).
- **Statistic:** permutation test across seeds, matched to the baseline (per `match-the-statistic`).

## Why this rung is viable where Exp 58 was not

- **The situation separates (the whole point).** `oxygen` at 0 underwater vs 0.5 surfaced is a
  full-range directional swing — the exact property dark/safe lacked (both underground, light=0;
  `nearest_hostile_dist` moved only 0.09 on one side of neutral → cos 0.977, unfixable). An
  instrument-check (safe-surface cluster ≠ underwater cluster) is a preflight, expected to PASS —
  but it is MEASURED first, not assumed (the verify-the-instrument lesson, twice-learned).
- **The mechanism already exists and fired.** Wire-4 (`_cluster_fear`, `record_cluster_fear`,
  `note_active_clusters`, `anticipatory_threat_need`), the pain→cluster subscriber, and the `flee`
  affordance are built and merged (Exp 58 line). Front-gate scope: NO new mechanism — this reuses it
  on a working cue. The failure mode `drive:oxygen` (or the existing air/drowning drive) must be in
  the `cluster_fear_failure_modes` allowlist (currently `drive:health` only) — a config addition to
  verify, not a new Wire.
- **The corrective act is game-native (D1).** Surfacing = swim up / leave water — a real affordance.
  Confirm the bridge exposes an executable "surface / move to air" (the `flee`-up path or a new
  param-free affordance), reachable and measurable, before freezing.

## Open design questions (for the four-lens review)

1. **Confounding — learned vs innate.** Minecraft applies drowning damage innately; the AGENT's
   learned want is anticipatory surfacing driven by Wire-4 cluster-fear. Does the FEAR-vs-ABLATED
   contrast + the anticipation-latency DV cleanly isolate the LEARNED component, or does innate
   damage-avoidance confound it? Is there a "surfaced but recently-drowned" probe that reads the
   learned fear without ongoing damage (the Exp 58 probe analog)?
2. **Bio-faithful — is oxygen-deficit a declared failure mode that publishes pain?** Exp 58 found
   pain needs a DECLARED failure mode (`pain-needs-declared-failure-modes.md`); confirm the
   oxygen/air drive publishes pain (entry/deepening latch) so the subscriber can book fear. Is the
   drowning-danger cluster co-active and noted when the pain fires (encode hoisted above pain tick)?
3. **Wiring — the corrective action + credit path.** Does `flee`/surface execute live from the water
   (pathfinder can swim? bot won't sink-drown mid-flee?), and does the read path
   (`anticipatory_threat_need` → `recommend_action`) actually emit the surface action? Verify with
   the real consumer, not a hand-composed probe (D43).
4. **Environment — the apparatus.** A NEW classroom variant: a deep water column / pool the agent is
   submerged in, with a reachable air/shore escape, built game-natively (real water, not a bespoke
   "oxygen−1 here"). Is submersion + escape reliably stageable (the drowning must actually deplete
   oxygen; the escape must actually restore it)? Does the bot drown-die too fast to measure latency
   (death-cap + rescue like Exp 58's teleport)?
5. **Transferability (the 1.3 headline, optional stretch).** If drowning-fear forms a clean cluster,
   does it bundle/travel between agents (the Exp 56/57 shared-want path)? Parked unless the base
   claim lands — but noted because a separable cue is the prerequisite the dark=danger want lacked.

## Not decided / parked

The exact water apparatus geometry, the surface affordance (reuse `flee`-up vs a new `surface`
action), the failure-mode allowlist entry, the anticipation-latency DV's precise definition, and the
death-cap/rescue. All are the four-lens review's job. This doc fixes the IDEA and its reuse of the
merged Wire-4 mechanism on a cue that separates.
