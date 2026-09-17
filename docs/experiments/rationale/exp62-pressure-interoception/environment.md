# Exp 62 pressure interoception — ENVIRONMENT lens (four-lens design review, 2026-09-17, on DRAFT v1)

**Charter (docs/experiments/DESIGN_REVIEW.md):** does the world game-natively afford it (D1 — the
owner has conceded that depth-below-surface from `blockAt` is the eye-height `is_in_water` class,
not a synthetic world fact; the held-out objection is a DRIVE's pain, deferred to Exp 62b), are the
needed states/acts reachable and measurable, does the bridge/world behave. Read:
`exp62_pressure_interoception_prereg.md` (draft v1, uncommitted), `survival_world_1_3.md` (§World
topology), `exp60_drowning_avoidance_prereg.md` (§Apparatus, §Outcome, amendments),
`scripts/survival_world/setup_world.py`, `water_trial.py`, `exp60_run.py`, `exp61_run.py`,
`exp60_water_check.py`, `l11_geometry_probe.py`, `common.py`, `scripts/minecraft_bridge/index.js`,
`bodies/minecraft_player.yaml`, `docs/wiring/cosine-separation-is-directional.md`, the Exp 60
apparatus/geometry/replay records under `docs/experiments/data/`, and the R3 environment lens
(`rationale/r3-survival-benchmark/environment.md` — its world facts are not re-derived here).

**Verdict line: DO-NOT-BUILD as drafted.** The world cannot supply the floor the claim is built on:
inside the builder's and the spawn-distance bounds there is NO pool-2 placement at which the shipped
roster's submerged reading leaves pool 1's water cluster (offline replay on the real encoder bases,
the same instrument that set the 90-block bound and matched Exp 60's live 0.7874 to 0.001: cos ≥
0.875 at every buildable altitude/distance, ≥ 0.99 for every "ordinary" one). Arms 1, 2 and 4 are
predicted at the ceiling, ABOVE-WALL is unearnable by construction, and arm 4 is degenerate. The
same replay says the shipped body ALREADY carries the fear across pools in this world — a smaller,
different experiment the world does afford. Everything below the first finding is the FIX list for
whichever re-scope the main session picks.

One correction to the brief I was given: the bridge caps `distance_from_spawn` at **128**, not 64
(`index.js::snapshot` — `Math.min(128, me.position.distanceTo(spawn))`; the YAML's "bridge cap 64"
comments belong to `nearest_hostile_dist` and `nearest_player_dist`). So a different distance below
90 IS sensed; the constraint is the ≤ 90 (3D) constant-mass bound and the ≥ 72 Exp 58 clearance,
not a cap.

---

## DO-NOT-BUILD

### DNB-1 — No buildable pool-2 placement makes the shipped roster MISS; the floor is not a floor, and the arms collapse onto the ceiling

**Failure scenario.** Pool 2 is built at the offsets D1 picks; arm 1 (shipped, cross) surfaces
12/12 at pool 2 exactly like arm 2; ABOVE-WALL (arm 2 − arm 1 ≥ 0.20) cannot pass; the campaign
burns ≈ 4 h to measure a contrast the encoder geometry forbids; or D1 is pushed to an extreme
placement whose own shore/submerged separability is marginal and pool 2 fails gate (ii).

**Evidence — the reachable offsets.** The builder's bounds: `shore_y` is a keyword of
`setup_world.py::water_classroom_geometry` (default `WATER_SHORE_Y` 40), depth ∈ [3, 12]
(`WATER_MIN_DEPTH`/`WATER_MAX_DEPTH`), pool centre ≥ 72 blocks HORIZONTAL from the Exp 58 anchor
and `dark` point (`exp58_clearance`), submerged target ≤ 90 blocks 3D from WORLD spawn
(`spawn_clearance`, `WATER_MAX_DIST_FROM_SPAWN`; the bound exists because at 128 the sensor is a
w = 1.0 constant and cos(shore, submerged) replays to 0.8525). Pool 1 sits at 69.17 from spawn
(`exp60_water_apparatus.json::w1_shore.distance_from_spawn`), shore y 40, floor y 35. The Exp 58 cave
sits ≈ 36 (3D) from spawn, so the far side of spawn gives a horizontal component ≥ ~45 for pool 2:
`distance_from_spawn` for any legal pool 2 lies in ≈ [45, 90]. `y_altitude` range [0, 128] clamps at
0, so a submerged target from y ≈ 0 (shore 5, shell bottom −3; world floor is −64 on 1.20.4) to
y ≈ 123 (shore 128) is the whole altitude lever.

**Evidence — what those offsets do to the cosine.** The encoder's gain is
`w = (2·|v − 0.5|)^3` (`docs/experiments/data/exp60_spawn_distance_check.py`, the replay that set
the 90 bound; its pool-1 prediction 0.7873 matches the live gate 0.7874). In every sealed-classroom
reading three sensors dominate: `light_level` 0 → v 0 → **w 1.0** (no rest, constant),
`time_of_day` 0.0417 (frozen day) → **w 0.77** (constant), `is_in_water` 1 → **w 1.0** when
submerged. The two "place" sensors carry, at pool 1, `y_altitude` 35 → w **0.093** and
`distance_from_spawn` 69 → w **0.158**. Replayed on the L11 base vector (same machinery, this
review's `exp62_env_replay.py` in the scratchpad; pool 2 assumed light 0 / day, i.e. the
confound-free case), cos(pool-1 submerged, pool-2 submerged):

| pool-2 submerged y | d = 50 | d = 69 | d = 90 | pool-2 OWN cos(shore, sub) |
|---|---|---|---|---|
| 0 (extreme) | 0.877 | 0.880 | 0.880 | 0.848–0.850 (marginal) |
| 5 | 0.925 | 0.928 | 0.927 | 0.826–0.830 |
| 10 | 0.959 | 0.961 | 0.958 | 0.810–0.815 |
| 20 | 0.991 | 0.992 | 0.988 | 0.792–0.800 |
| 59 / 80 / 95 | 0.998 | 0.999 | 0.993 | 0.786–0.797 |
| 110 | 0.984 | 0.985 | 0.979 | 0.796–0.805 |
| 120 | 0.942 | 0.943 | 0.938 | 0.814–0.822 |

Every cell is ≥ 0.85 = SAME cluster. This is corollary 7 of
`cosine-separation-is-directional.md` in the flesh: pool 1's y (v 0.27) to any other y is a
same-side excursion of a low-mass sensor (20° of arc at w 0.09 → 1.0 only at the y = 0 clamp), and
`distance_from_spawn` cannot leave [45, 90] (v 0.68–0.85, w 0.05–0.34) — neither can open the
~32° the 0.85 threshold needs against ~2.8 units of shared constant mass. The extreme y ≈ 0
placement buys 0.875 — 0.025 above threshold, inside the replay's demonstrated error when a
sensor is mis-modelled (the saturation miss was 0.06) — AND makes pool 2's own separability
marginal (0.848–0.850), the spawn-distance-bound family in reverse.

**Consequences for the arms as drafted.** Arm 1 (shipped, cross) is predicted to HIT the trained
node → fires at pool 2 → ≈ 1.0, not the "cache-wall floor"; arm 2 ≈ arm 1; arm 4 (y and d at gain
0) is degenerate — with those two silenced, the pool-1 and pool-2 submerged readings are IDENTICAL
by construction (cos 1.000), so "does removing place do the same job" has no content; arm 3 and
arm 5 survive as within-pool ceiling/exposure controls. The draft's own step 2 ("prediction to be
checked, not assumed") is written for the OTHER direction (pressure may not lift a miss); it must
also be allowed to say "there is no miss to lift", and the replay must run over the FULL buildable
grid, not at "the planned pool-2 offsets".

**Fold options (the main session's call; the environment facts for each):**

- **(a) Re-scope to the claim the world DOES afford: the shipped body carries the drowning-fear
  across pools (Exp 61 §Not claimed "fears water anywhere", within the sealed / frozen-day class).**
  The replay predicts a HIT for every legal placement; the loop-OFF representation gate (Exp 61
  step 4) is the mechanism read; a stacked pool 2 (SF-4) is the cleanest apparatus; the honest
  caveat is that `light_level` and `time_of_day` are constant across the two pools, so the claim is
  "location-invariant", not "world-invariant". No `src/` change; `pressure` is unmotivated by the
  cache wall and stays a 62b question.
- **(b) Keep the `pressure` contrast by manufacturing a floor with a NON-place difference.** The
  only world levers with mass are light and time: a day-lit surface pond (light 15) → shipped
  0.588, ablated 0.588, `pressure` at FULL weight 0.692 — every arm misses (and Exp 58 recorded
  `light_level` as unreliable in this world); a NIGHT pool (`time set night`, tod 0.99) → shipped
  0.799 (miss), `pressure` at full weight 0.852 (hit by 0.002 — no margin), ablated 0.797. That is
  a different claim ("time-of-day-invariant"), not the altitude/distance claim, and it is decided
  at the replay's noise floor.
- **(c) Move `pressure` to where it can carry mass: a graded-depth claim or the 62b drive**, with
  the range pinned to the reachable depth (SF-1).

---

## SHOULD-FIX

### SF-1 — `pressure` as declared is almost silent: range [−12, 12] puts its gain at w = 0.037 at the ONLY depth the protocol can use, and "12 = max column depth" ties the gain to an unreachable state

- At the Exp 60 placement the feet stand on the floor at y = 35, eye at 36.62 → eye block 36; the
  source column is y 35..39 (`water_classroom_geometry`: pool y [shore_y − depth, shore_y − 1]); the
  upward scan reads 36, 37, 38, 39 water, 40 air → **`pressure` = 4 = depth − 1**. At the builder's
  max depth 12 it would read **11**, not 12 (the draft's "12 = max column depth").
- Depth is PINNED by the US-free window, not by the builder: the probe cap is pain edge − 0.75 s =
  4.335 s (`exp60_run.py::FROZEN["probe_cap_margin_s"]`, `w2_dive.t_pain_edge` 5.085–5.443); the
  measured fear latency is 1.3–3.3 s and the 4-block ascent 1.45–1.83 s (`w4_escape.t_surface`),
  which already sums to 2.8–5.1 s. Each extra block of depth costs ≈ 0.4 s (2.2–2.8 blocks/s
  measured). Depth 5 is the ceiling; the "12" state is never visited.
- Under the range principle (`minecraft_player.yaml` header; corollary 6's converse) a declared
  extreme the world never visits makes the sensor silent: v = (4 + 12)/24 = 0.667 → w 0.037. Even
  pinned to [−4, 4] (w 1.0 at the floor) `pressure` is a SHARED component of both pools' readings
  and lifts cross-pool cos by only +0.02–0.03 (grid: 0.875 → 0.898 at y = 0) — proportional, not
  decisive.
- **Fix:** if the sensor survives the DNB-1 fold, declare the range on the reachable depth (e.g.
  [−4, 4] or [−5, 5]), freeze depth = 5 for BOTH pools (a different depth makes `pressure` itself a
  cross-pool contrast — 4 vs 10 — which pulls the cosine DOWN, the opposite of the claim), and
  state that the graded values 3, 2, 1 are traversed in ≈ 0.4 s each during the ascent (3–4
  snapshots per block at 100 ms).

### SF-2 — `light_level` (and `time_of_day`) must be GATED equal at pool 2, and pool 2's OWN separability must be replayed and live-gated; today W1 only RECORDS light

- `light_level` at 0 is a w = 1.0 constant in both pools' readings; if pool 2 reads anything else
  the light component flips basis and EVERY arm misses (lit pond: 0.588) — a fake floor that would
  read as "pressure did not transfer". `exp60_water_check.py` W1 gates `nearest_hostile_dist` ≥ 64
  and `distance_from_spawn` ≤ 90 and records `light_level`/`time_of_day` without gating. The Exp 58
  builder docstring (`setup_world.py::_classroom`) documents skylight contamination of buried
  cells on this server/mineflayer pair ("a buried cell read 14 at day / 0 at night", neighbours
  reading block light with no source), and `index.js::perceivedLight` reads
  `getBlockLight`/`getSkyLight` at the bot's cell — a freshly `fill`ed box at a different y band is
  exactly where a stale-light read would show. A pool 2 above ground (shore_y ≥ ~64) is a stone box
  in open sky: light 15 outside, 0 inside only if the client's light data is right.
- Pool 2's own cos(shore, submerged) must clear 0.85 for the representation gate to mean anything
  (a merged pool-2 shore/water cluster contaminates the fear read); deep placements are marginal
  (0.848–0.850 at y ≈ 0–5).
- **Fix:** W1 at pool 2 gates `light_level == pool-1 recorded` and `time_of_day == pool-1 recorded`
  (shore AND submerged), the replay reports cos(shore2, sub2) beside the cross-pool cosines for
  every candidate, and gate (ii) (`l11_geometry_probe --anchor-file <pool 2>`) runs on pool 2
  before any row.

### SF-3 — The builder, the check and the harness are single-pool by construction: no `--shore-y`, one anchor file, one `measured` stamp, one flee anchor

- `_water_classroom` calls `water_classroom_geometry(int(ax), int(az), depth=args.depth)` — the
  `shore_y` kwarg is never passed and there is no `--shore-y` argument (`setup_world.py::main`).
  `tests/unit/test_exp60_water_classroom.py` exercises `depth` only.
- `WATER_ANCHOR_FILE` is one module constant; `_water_classroom` reads it for the idempotent
  default anchor and OVERWRITES it on build; `exp60_water_check._stamp_measured` writes `measured`
  into the same constant path (`ANCHOR_FILE`); `exp60_run.ANCHOR_FILE` and `exp61_run` (imports
  `ANCHOR_FILE` from `exp60_run`, derives `probe_cap_s`/`train_cap_s` from that record's `measured`)
  all bind to it. Building pool 2 with `--anchor-x/z` destroys pool 1's record and its stamped
  edges; running the check on pool 2 re-stamps whichever record is there.
- What IS pool-agnostic: `water_classroom_geometry`, `water_classroom_commands`,
  `water_classroom_verifications` (pure in `geom`), `exp58_clearance`, `spawn_clearance`,
  `WaterTrial` (takes `geom` — two instances over one client/agent are possible; `rescue`,
  `submerge`, `check_clusters_distinct`, `placement` are per instance), `l11_geometry_probe
  --anchor-file`. `setup_world.py::_verify` is the GAMERULE/bread verify, not a geometry verifier —
  it is pool-agnostic because it never looks at a pool.
- The bridge takes ONE `--flee_x/--flee_z` per process (`index.js` `FLEE_X`/`FLEE_Z`), else
  `bot.spawnPoint` = WORLD spawn; neither the Exp 60 nor the Exp 61 runbook records what the
  bridge ran with (R3 lens SF-3). In the US-free probe `flee` only fires submerged (fails fast); it
  matters only if a fear need ever stands on a shore — the stacked shores replay as one cluster
  (0.925) with fear 0, so it is a freeze item, not a hazard.
- **Fix:** `--shore-y` and `--anchor-file` (or `--pool <name>`) on `water_classroom` and
  `exp60_water_check`; a pool-vs-pool SHELL clearance guard (shell = x [ax−6, ax+9], y
  [shore_y − depth − 3, shore_y + 5], z [az−4, az+4]; for a stacked pool 2 under pool 1's [32, 45]
  that means shore_y ≤ 26, above it shore_y ≥ 54); the harness loads two records and instantiates
  two `WaterTrial`s; the flee anchor recorded per pool in the freeze.

### SF-4 — Placement: prefer a STACKED pool 2 (same x/z, different y); measure and stamp WORLD spawn, which nothing currently exposes

- The 3D spawn bound couples altitude and distance: with pool 1's horizontal offset from spawn
  ≈ 65 (69.17 at dy ≈ 24 if spawn y ≈ 64), a pool 2 directly below/above at the same x/z reads
  d ≈ 88 at floor y 5, ≈ 85 at y 120, ≈ 72 at y 95 — inside 90 for shore_y ∈ [~10, ~120] — and
  INHERITS pool 1's Exp 58 horizontal clearance exactly. It also removes the client chunk-load
  question: `view-distance=8` (`setup_world.py::SERVER_PROPERTIES`) = 128 blocks; two pools > 128
  apart have a post-`tp` window in which `blockAt` is null → `is_in_water` 0 and `pressure` 0 while
  `y_altitude` already reads pool 2 (position comes from the teleport packet, blocks from
  `map_chunk`). `settle_until` (`common.py`) tolerates it for the harness's teleports, but a loop
  window must never straddle a pool switch; stacking makes the switch a same-chunk teleport.
- World spawn is not readable over RCON (`spawn_clearance` docstring); the bridge has
  `bot.spawnPoint` but the snapshot emits only the distance; pool 1's record carries
  `spawn_clearance_blocks: null`. The stacked candidate's d2 depends on spawn y.
- **Fix:** have the bridge print `bot.spawnPoint` once at start (or emit it in the hello), stamp it
  into both records, and let the builder's `--spawn-x/y/z` guard run for pool 2. Name the
  candidate in the prereg as "pool 2 = pool 1's (x, z), shore_y = N", with the replayed numbers.

### SF-5 — Both pools need their own live apparatus check and `measured` block; the freeze list

- The pain edge is time-based (oxygen ticks), so the probe cap carries over; but `t_surface`
  (ascent, per pool), `distance_from_spawn`, `light_level` and the fill/source verifications are
  per pool. `exp61_run` refuses when its frozen copy of Exp 60 constants drifts — add pool 2's
  `measured` to what it checks.
- Freeze: both anchor records (shore, submerged, depth 5 both, forceload, spawn coords, the two
  `measured` blocks), the `spawnpoint` (ONE per player — the last-built shore owns respawn; US-free
  so unused, but record which), the shared `exp60_deaths` objective (per player;
  `scoreboard objectives add` on an existing objective replies "already exists", which the
  builder's `bad` filter does not flag — fine), the bridge cap **128** for `distance_from_spawn`,
  the flee anchor per pool, the `pressure` scan predicate (`water` OR `bubble_column`, the SAME
  predicate and the SAME eye block as `is_in_water`, in the same snapshot; the fingerprint asserts
  `pressure > 0 ⟺ is_in_water == 1`), the scan cap, and `pressure`'s range as pinned by SF-1.

---

## NIT

- **N1 — the `pressure` scan is well-defined in the classroom.** 3×3 `water[level=0]` column
  (verified: bottom/top/corner sources after a 2 s fluid pause), stone on every side, chamber air
  above: from any eye block in the column the first non-water block upward is the chamber air at
  shore_y. Edge cases: the surface bob (Exp 60's y ≈ 38.75 → eye 40.37 = air → 0; y ≈ 38.3 → eye
  39.9 = water → 1) is the same 0/1 flicker `is_in_water` already has; flowing water is still named
  `water` (level > 0) and counts — none exists here; waterlogged blocks and kelp are NOT counted by
  either sensor (name ≠ `water`) — none can appear in a `fill`ed column (kelp only generates at
  worldgen and grows from existing kelp; `doMobSpawning false` keeps drowned out); bubble columns
  need soul sand/magma — none by hygiene, but keep them in the predicate for parity with
  `is_in_water`; a null eye block (chunk not resident) → 0 for both sensors, consistent. Cap the
  upward loop (e.g. 32) so a natural ocean never walks to build height 320.
- **N2 — forceload.** `forceload add` is limited to 256 chunks per command (Java); a pool shell is
  16 × 9 blocks → ≤ 4 chunks; two pools ≤ 8. Forceload keeps the SERVER ticking those chunks; it
  says nothing about the client's resident columns (SF-4).
- **N3 — `nearest_player_dist`** is 64 at both pools (one bot; no spectator) and `is_raining` is a
  WORLD flag, so the Exp 61 settle guard transfers unchanged.
- **N4 — `on_ground`** at the floor of a source column is 1 in mineflayer (feet on stone), the same
  at both floors; a different floor block does not matter (all stone).
- **N5 — the deep band is buildable** (shore_y 10 → shell y [2, 15]; deepslate/aquifers/lava are
  overwritten inside the shell; 2+ blocks of stone stop lava and its block light) but its own
  separability is marginal (SF-2) and a lava-lit cell outside the shell cannot leak light through
  opaque stone — verify the read anyway.
- **N6 — the replay's base.** These numbers are on the L11 base vector with pool 1's measured
  contrast; the prereg's step-2 replay must use the LIVE captured pool-1 vectors
  (`exp60_geometry_2026-09-15b.json::per_sensor`), as the draft already says — the offline
  instrument was 0.001 off live when every sensor was modelled and 0.06 off when one was not.

---

## What I verified (offline, `file::symbol`)

- Builder: `setup_world.py::water_classroom_geometry` (shore_y kwarg, pool y [shore_y − depth,
  shore_y − 1], submerged feet at shore_y − depth, `shell`, `chamber`, `forceload`),
  `water_classroom_commands` (forceload first, `doMobSpawning false`, `spawnpoint`,
  `exp60_deaths`), `water_classroom_verifications` (ten `execute if block` checks incl. three
  sources), `exp58_clearance` (horizontal, ≥ 72), `spawn_clearance` (3D, ≤ 90, only with
  `--spawn-x/y/z`), `_water_classroom` (no `shore_y` passed; single `WATER_ANCHOR_FILE`; overwrite
  on build), `main` (no `--shore-y`), `_verify` (gamerules + bread only), `SERVER_PROPERTIES`
  (`view-distance=8`).
- Bridge: `index.js::snapshot` (`distance_from_spawn` = `Math.min(128, distanceTo(bot.spawnPoint))`,
  3D; `y_altitude` = `me.position.y`; `is_in_water` from `blockAt(position + eyeHeight)` with
  `water`/`bubble_column`; `perceivedLight` = max(block, sky − darkness) at the bot's cell;
  `STATE_INTERVAL_MS` default 500, harness requires 100); `FLEE_X/FLEE_Z` once per process;
  `escape_water` holds jump ≤ 8 s with a 600 ms post-clear hold.
- Body: `minecraft_player.yaml` ranges — `y_altitude` [0, 128] rest 64; `distance_from_spawn`
  [−128, 128] rest 0; `light_level` [0, 15] no rest; `time_of_day` [0, 1] no rest; `is_in_water`
  [−1, 1] rest 0; `hostile_count` [−32, 32] (the "unobservable negative half" shape the draft
  copies).
- Encoder gain and replay: `exp60_spawn_distance_check.py` (`w = (2|v − 0.5|)^3`, `_stable_basis`
  low/high, threshold 0.85, pool-1 prediction 0.7873 vs live 0.7874 in
  `exp60_geometry_2026-09-15b.json`).
- Harness: `water_trial.py::WaterTrial.__init__` (takes `geom`; per-instance shore/sub),
  `check_clusters_distinct`, `rescue` settle guard; `exp60_run.py::FROZEN` (probe cap margin 0.75 s,
  train margin 1.0 s, liveness 3 s), `ANCHOR_FILE`; `exp61_run.py` (imports Exp 60's
  `ANCHOR_FILE`, derives caps from `geom["measured"]`, step 4 representation gate =
  `submerge` → encode → `rescue`, loop OFF); `exp60_water_check.py` W1 (gates hostile horizon +
  spawn ≤ 90; light/tod recorded), `_stamp_measured` (writes to the constant path);
  `l11_geometry_probe.py --anchor-file` (Exp 60 `probe_situations` shape).
- Measured Exp 60 edges: `t_pain_edge` 5.085–5.443 s, `t_oxygen_zero` 15.08–15.27 s,
  `t_damage_onset` 16.07–16.65 s, `t_surface` 1.45–1.83 s (4 blocks), post-training latency
  1.3–3.3 s, w1 light 0 / tod 0.0417 / spawn 69.17 / nearest_hostile 64.

## What must be measured live (in this order)

1. **WORLD spawn coordinates** (`bot.spawnPoint`) — every d2 number above assumes spawn y ≈ 64.
2. **`light_level` and `time_of_day` at pool 2's shore AND floor after the build and after a
   client reconnect** (the stale-light failure Exp 58 documented) — must equal pool 1's 0 / 0.0417.
3. **Pool 2's own gate (ii)** — cos(shore2, sub2) on live-captured vectors < 0.85 with distinct
   fresh-EC ids.
4. **The cross-pool representation read on ONE shipped-roster agent trained at pool 1** (loop OFF,
   US-free): does pool 2's submerged reading resolve to the pool-1 water node? The replay says yes
   at every legal placement; this single row decides whether DNB-1's floor exists before anything
   else is built.
5. `pressure`'s value at pool 1's floor (expect 4) and its 0/1 agreement with `is_in_water` across
   an ascent at 100 ms.
6. The post-teleport null-`blockAt` window length between pools (only if they are not stacked).

## Verdict

**DO-NOT-BUILD** as drafted: the world offers no pool-2 placement inside the builder's and the
spawn-distance bounds at which the shipped roster misses (cos ≥ 0.875 everywhere, ≥ 0.99 at any
non-extreme placement), so the claim's floor and its ABOVE-WALL gate are unearnable and arm 4 is
degenerate; the only levers with mass are non-place (day-lit pond — all arms miss; night — a
0.002 margin). **FIX-THEN-BUILD** for the re-scoped claim the world does afford — the shipped body
already carries the drowning-fear across pools (stacked pool 2, light/time gated equal, both
anchors and `measured` blocks frozen, builder/check/harness made two-pool) — with `pressure` sent to
a design where it can carry mass (range pinned to the reachable depth; 62b or a graded-depth claim).
