# Exp 62 (DRAFT v1) — WIRING lens

Reviewer: wiring lens of the four-lens design review (`docs/experiments/DESIGN_REVIEW.md`), 2026-09-17,
on branch `exp62/design-draft`, reading `docs/experiments/exp62_pressure_interoception_prereg.md` DRAFT v1.
Charter: real consumers + real credit path (D43), the right encoding and seams, no hand-composed shortcut
that passes while the loop fails. Read `docs/wiring/*.md` first; corollary 3 of
`cosine-separation-is-directional.md` ("replay offline on real captured vectors BEFORE building") is the
lens's main instrument here, and I ran it.

## Headline

The prereg's own build-order step 2 (the offline replay) is decidable NOW from committed data, and it
decides against the design as written. On the shipped `_sensor_embed` with the committed pool-1 vectors
(`docs/experiments/data/exp60_geometry_2026-09-15b.json::per_sensor`, 17 sensors, the Exp 60 site), the
**shipped roster already completes every pool-2 reading inside the builder's bounds into pool 1's water
node**: cos(sub1, sub2) = 0.916–1.000 for every floor altitude 35→123 and spawn distance 20→90 (threshold
0.85). Arm 1's "cache-wall floor" does not exist on this encoder; arm 2 − arm 1 is predicted ≈ 0; `pressure`
at any declared range moves the number by < 0.02. The place of the water situation on this body is not
carried by `y_altitude` (gain w 0.09) or `distance_from_spawn` (w 0.16) — it is carried by the two `rest:
null` constant-mass sensors `light_level` (w 1.0) and `time_of_day` (w 0.77), exactly the roster
`scripts/lint_body_rest_neutral.py` prints every run. A daylight surface pool MISSES (cos 0.58); `pressure`
does not rescue it (0.59; 0.69 even at a [-4,4] range). So the mechanism has a real consumer but no
measurable job, and the experiment has no contrast to measure.

## Findings

### DO-NOT-BUILD

**DNB-1 — The floor arm hits: the shipped body is already location-invariant across altitude and spawn
distance, so the claimed contrast is zero before anything runs.**
Failure scenario: 60 trainings + 60 probes later, arms 1 and 2 both read ≥ 0.9 CROSS, ABOVE-WALL fails, and
the campaign is a null that says nothing about `pressure` — the prereg's own fallback sentence ("that is a
design result, recorded, not a live campaign") fires, but only after the build if the replay is deferred.
Evidence (replayed through `src/maxim/similarity/encoder.py::_sensor_embed`, gain p3.0, unit ranges on the
committed NORMALIZED values; base check reproduces the record's 0.7874 exactly):

```
cross-pool cos(sub1, sub2), SHIPPED roster (pool 2 floor y2 × spawn distance d2 ≤ 90):
  y2= 35  d2=20: 0.9967  d2=40: 0.9976  d2=69: 1.0000  d2=90: 0.9946
  y2= 59  d2=20: 0.9960  d2=40: 0.9968  d2=69: 0.9991  d2=90: 0.9934
  y2= 80  d2=20: 0.9962  d2=40: 0.9970  d2=69: 0.9993  d2=90: 0.9936
  y2=100  d2=20: 0.9946  d2=40: 0.9955  d2=69: 0.9976  d2=90: 0.9919
  y2=123  d2=20: 0.9172  d2=40: 0.9179  d2=69: 0.9196  d2=90: 0.9156   (all ≫ 0.85 = SAME cluster)
per-sensor gain weights in the committed submerged vector:
  is_in_water 1.0 · light_level 1.0 · time_of_day 0.77 · distance_from_spawn 0.164 · y_altitude 0.093 · oxygen 0.043
```
Why: cosine sees direction (corollary 1/7). The shared mass across pools (`is_in_water` 1.0 + `light_level`
1.0 + `time_of_day` 0.77) dwarfs the differing mass (y + distance ≤ 1.0 + 0.34 even at the caps); the
`y_altitude` cap at 128 (`sensor-range-clamps.md`) and the ≤ 90 spawn bound (`setup_world.py::
WATER_MAX_DIST_FROM_SPAWN`) bound the differing mass BY DESIGN — the Exp 60 apparatus rules that keep the
within-pool contrast clean are the same rules that make two pools indistinguishable. Within-pool separation
at pool 2 stays fine (cos(shore2, sub2) 0.785–0.831 < 0.85), so pool 2 would have a distinct water cluster —
it is just the SAME distinct cluster as pool 1's.

Where cross-context actually lives on this body, from the same replay: `cos(sub1, sub2 @ daylight surface,
light 15) = 0.5812` (MISSES); `cos(sub1, sub2 @ same shell at night, time_of_day 0.6) = 0.8934` (borderline
hit). If the owner wants a cross-context water-fear experiment, the context axis that the shipped body can
SEE is light/time, not altitude/spawn — and `pressure` is orthogonal to it (it adds shared mass; light
rotates the vector). That reframing belongs to the confounding + bio-faithful lenses; the wiring fact is
that the encoder cannot express the contrast v1 pre-registers.

Consequence for the mechanism: `pressure` has a real consumer (the world channel `agent_loop::
_read_world_states` → `encode_sensors` → cluster id → `NAc.note_active_clusters` → Wire-4 key) but no
measurable job in this design. Per the front-gate rule, do not add the sensor until an experiment names the
contrast it must move.

**DNB-2 — Arm 4 ("ablated absolutes") is an identity, not an ablation, and has no mechanism.**
Failure scenario: arm 4 reads 12/12 CROSS by construction, the outcome is written as "relative features",
and `pressure`'s contribution (arm 2 − arm 4) is negative or zero for a reason that has nothing to do with
relative features.
Evidence: dropping `y_altitude` + `distance_from_spawn` leaves pool 1 and pool 2 submerged readings
IDENTICAL (every other sensor is apparatus-constant across pools) → replayed cos(sub1, sub2) = **1.0000**
exactly. And "gain 0 on two sensors" does not exist as a knob: the A4 gain is per MODALITY
(`encoder.py::SensorEncoderConfig.gain_modalities`, `gain_exponent`), never per sensor; `component_registry
.deep_merge` replaces non-dict values and cannot DELETE a key, and `spec.py::_parse_entity` calls
`sensor_spec.get(...)` so `y_altitude: null` in an `extends:` child raises — arm 4 needs a full-copy body,
which is a third geometry tag. D4 says arm 4 is required; as designed it cannot discriminate anything.

### SHOULD-FIX

**SF-1 — `pressure` as declared is silent through the whole apparatus (corollary 7).**
Failure scenario: the sensor "ships", arm 3 passes (nothing changed), and any arm-2 effect is attributed to
a sensor whose gain weight never exceeded 0.037.
Evidence: eye block = floor + 1 (feet y=35, eyeHeight 1.62 → block 36); water 35..39 → 4 water blocks from
the eye block up at the default depth 5 (`setup_world.py::water_classroom_geometry`, `WATER_DEPTH_DEFAULT`).
Range [-12, 12] rest 0 → v = 0.667, w = (0.333)³ = **0.037**; the entire dive spans v 0.5→0.667, a same-side
few-degree excursion. Range [-4,4] gives w 1.0 but changes nothing cross-pool (0.9366 at the worst corner)
because the pools already merge. A 12-deep pool 2 (pressure 11 vs 4) reads 0.9076 — a graded depth feature
is a DEPTH feature; it distinguishes pools of different depth, which is the opposite of the job. "12 = the
builder's maximum column depth" declares the range for a pool the experiment never builds.

**SF-2 — "Nothing in `src/` beyond the YAML" is false; adding to `minecraft_player` is a src change + three
pinned tests + a docstring.**
Failure scenario: the mechanism PR touches only the YAML and the bridge, CI goes red on the lockstep pin,
and the fix is to grow the fake — which is fine, but it is not "nothing in src/".
Evidence: `src/maxim/simulation/minecraft_harness.py::FakeBridgeServer._snapshot` is lockstep-pinned to the
body's declared `modality: world` set (`tests/unit/test_minecraft_harness.py::TestBodyFakeLockstep::
test_fake_snapshot_covers_every_declared_world_sensor`, exact set equality); `tests/unit/test_minecraft_seam.py`
pins the literal 17-name `world_owned_sensors` set twice; `src/maxim/simulation/minecraft.py` documents the
roster; the golden's `_WORLD_RANGES` comment says "17-sensor roster" (comment only, see SF-4).

**SF-3 — Put `pressure` on a variant body (`bodies/minecraft_player_pressure`, `extends: bodies/
minecraft_player`), not on `minecraft_player`; it is the honest choice, not a dodge.**
Failure scenario (if added to `minecraft_player`): arm 1 "shipped roster" runs a body that no longer exists
in the repo (a hand-kept copy of the pre-change YAML — the SHIPPED body is then the one with `pressure`, and
the control is the non-shipped one); every `minecraft_player` world node's geometry tag moves from
`g52a03986` (the tag on all 121 committed Exp 61 rows in `docs/experiments/data/exp61_pairs.jsonl`, and the
golden's `world_minecraft_player_gained`) to a new tag; the Exp 60 ledger trigger "`minecraft_player`
sensor-range change" fires by its letter while the harness does NOT refuse — `water_trial.py::
live_fingerprint` pins `sensor_ranges` for `is_in_water`/`oxygen`/`saturation` ONLY, so the fingerprint
(and Exp 61's literal copy in `exp61_run.FROZEN["exp60"]["fingerprint"]`) shows zero drift. A trigger that
fires with no refusal is discharged by convention only.
Evidence for the variant path: precedent `bodies/minecraft_bench_satiated.yaml` (`extends: bodies/
minecraft_bench`); `agent_loop::_read_declared_modality_ranges` walks whatever the resolved body declares, so
the variant's tag carries `pressure` and the shipped body's tag is untouched; `exp61_run.BODY_REF` /
`exp60_run` `entity_ref="bodies/minecraft_player"` would become a per-arm parameter. Trigger accounting:
with the variant, no Exp 60/61 trigger fires (the shipped body, encoder, EC, bridge protocol are
unchanged; the bridge emitting an extra `pressure` key is ignored by the shipped body — pinned by
`test_minecraft_seam.py::test_confirmed_action_succeeds_and_syncs_measured_state`, "unknown_key not in
vital_metrics"). Promotion of `pressure` into `minecraft_player` — if Exp 62 ever earns it — is its own
PR that fires the triggers THEN, with a discharge the replay already quantifies: pool-1 shore vectors are
byte-identical (pressure 0 → zero vector, corollary 4: verified `emb(shore1) == emb(shore1 + pressure=0)`),
pool-1 submerged moves by cos 0.99985, within-pool 0.7874 → 0.7879. Owed at promotion: re-run
`exp60_water_check` + the `l11_geometry_probe` run-gate at pool 1 (not the campaign), a dated ledger
amendment on both rows, and `exp61_run` donor/receiver tags re-derived (the Exp 61 data's `g52a03986` rows
stay historical). Also correct the prereg sentence "persisted world nodes … are re-staled on load by the
geometry-tag change (the H2 invalidate path)": there is no automatic re-stale — `ec.py::
_migrate_legacy_geometries` stamps only UNSTAMPED nodes; stamped nodes under a different tag simply stop
matching (the geometry mask in `scan`) and a deduped WARNING points at `maxim substrate invalidate
--drop-geometry`. They become unreachable-but-present, which for a fresh agent is moot and for the Exp 61
staged bundles is a re-analysis note.

**SF-4 — Delete "golden-pin regen from the commit that produces it" from build-order step 3.**
Failure scenario: someone runs `--regen` on the change commit, pasting new output into the pin — the exact
anti-pattern the fixture's rules forbid — for a change that does not touch the pin at all.
Evidence: `tests/unit/test_encoder_golden_v1.py` holds `_WORLD_RANGES` and `TAG_FIELDS` as FROZEN
transcriptions ("The readings are FROZEN here, not read from body YAML: the pin holds the FUNCTION. A later
range re-declaration in a body must not silently change the golden's inputs"), and its regen rule is
"regenerated ONLY from the pre-change commit, with the diff justified in the PR — never by pasting the new
output". Adding a sensor to a body changes no encoder function: `_sensor_embed`, `_normalize_value`,
`_stable_basis`, `sensor_geometry_fields`, `encoding_geometry_tag` are all untouched. The pin stays green
with no regen, and that is correct. (The `# 17-sensor roster as of 2026-09-16` comment may be updated; that
is not a regen.)

**SF-5 — Everything in the apparatus assumes ONE pool; the harness plan must name the second-pool
plumbing, and "a different distance from spawn" is not a lever.**
Failure scenario: `setup_world water_classroom --anchor-x/--anchor-z` for pool 2 OVERWRITES
`~/.maxim/exp60_water_classroom.json` (including the `measured` block `exp60_water_check._stamp_measured`
wrote, which `exp60_run`/`exp61_run` refuse without), silently re-points every Exp 60/61 re-run at pool 2,
and pool 2 cannot be built at a different altitude band from the CLI at all.
Evidence: `setup_world.py::WATER_ANCHOR_FILE`, `exp60_run.py::ANCHOR_FILE`, `exp60_water_check.py::
ANCHOR_FILE` are three copies of one fixed path; `_water_classroom` does `WATER_ANCHOR_FILE.write_text(...)`
unconditionally; `water_classroom_geometry(..., shore_y=WATER_SHORE_Y)` takes the band as a kwarg but the
subparser exposes only `--anchor-x/--anchor-z/--depth/--spawn-x/y/z/--sweep` (no `--shore-y`, no
`--anchor-file`); `water_classroom_commands` sets `spawnpoint <user>` to the shore of whichever pool built
LAST (a death mid-training at pool 1 respawns at pool 2 — harmless because `rescue` teleports, but the
death cap is 2); `deaths_objective` is the shared `exp60_deaths` (fine: cumulative counter, `deaths0`
baseline); `forceload` is per pool (fine); Exp 58 clearance is checked per build (pool 2 must also clear
72 blocks). `WaterTrial.__init__` takes ONE `geom` (`self.shore`/`self.sub`) and `attach_instruments`
wraps `executor.execute` per instance — two instances on one AUT double-wrap the spy, so the harness needs a
`set_pool(geom)` on one instance, plus a per-pool apparatus check (W1: `light_level`, sensed
`distance_from_spawn` ≤ 90, `nearest_hostile_dist` 64) and a per-pool `check_clusters_distinct`. The ≤ 90
spawn bound applies to pool 2 for exactly the reason it applies to pool 1 (constant mass at the cap) — it
is the point, not a trap — and the replay shows d2 anywhere in [20, 90] cannot produce a miss, so D1's
"spawn distance chosen so the shipped roster's replay MISSES" has no solution inside the bound.

**SF-6 — The replay must be seeded from the Exp 60 record, and its synthetic pool-2 reading is honest only
for the sensors geometry fixes; the live gate stays required.**
Failure scenario: the replay reuses `l11_geometry_2026-09-15.json` (the Exp 58 Slice-1 base that
`exp60_oxygen_separation_check.py` / `exp60_spawn_distance_check.py` use — captured at a different site,
corollary 6) and predicts a geometry the pool never has.
Evidence: `_sensor_embed(sensors, ranges, dim, gain_exponent)` is a pure function (SHA bases via
`_stable_basis`, linear `_normalize_value`, per-sensor sum) — verified by reading and by reproducing the
record's 0.7874 from its `per_sensor` values. Inputs a synthetic pool-2 row needs from committed data:
`exp60_geometry_2026-09-15b.json::per_sensor[].v_safe / v_dark` (17 normalized values; the pool-1 site),
`y_altitude` shifted to the exact pool-2 floor (from `water_classroom_geometry`), `distance_from_spawn`
SWEPT across the bound (world spawn is not RCON-readable — `setup_world.spawn_clearance` docstring — and
pool 1's sensed 69.17 fixes only a sphere), `light_level` = 0 ASSUMED for a sealed stone shell (must be
confirmed by pool 2's W1), `time_of_day` = the frozen 0.0417, `pressure` from the geometry. What synthetic
cannot supply: the live bot's `speed`/`look_pitch`/`on_ground` at pool 2 (near-neutral in the record, w≈0),
the live EC's NAc threshold overrides (`make_fresh_encoder` wires `nac=`), and the actual light reading. So:
synthetic decides go/no-go (it already did — DNB-1); the live `check_clusters_distinct` at pool 2 and the
loop-OFF representation gate remain the required confirmation, as the Exp 60 record's own
`run_gate.necessary_not_sufficient` says.

### NIT

**N-1 — Bridge scan semantics and cost.** `bot.blockAt` is `bot.world.getBlock` (a synchronous column
lookup) plus a painting-map read (`mineflayer/lib/plugins/blocks.js::blockAt`) — microseconds; ≤ 12 calls
per 100 ms snapshot is negligible next to the entity scan `snapshot()` already does. Correctness: an upward
scan that stops at the first block whose `name` is not `water`/`bubble_column` under-counts through
`kelp`/`kelp_plant`/`seagrass`/`tall_seagrass` and waterlogged blocks (prismarine-block exposes
`block.isWaterlogged`, `node_modules/prismarine-block/index.js:197`); the 3×3 stone-walled SOURCE column
holds only `water` with air above (`water_classroom_verifications` asserts sources at bottom/top/corner and
air at `surface`), so it is correct in this classroom. If built anyway: cap the scan at `WATER_MAX_DEPTH`
(12) so a natural ocean never walks to build height, treat `null` (unloaded chunk) as non-water, and count
`{water, bubble_column, kelp*, *seagrass} ∪ isWaterlogged` so the field is not classroom-only.

**N-2 — The field reaches the body under the existing contract; gate its presence on the RAW state.**
`MinecraftWorldBackend.world_owned_sensors` derive from the resolved YAML (`test_minecraft_seam.py::
test_declared_world_sensors_derive_from_the_yaml`); `sync_world_sensors` writes any declared key present in
`client.latest_state()`; `audio_localization.py::world_set_axis` clamps to the declared range. But the body
carries `pressure` at its `initial` (0 = rest = silent) whether or not the bridge emits it, so a body-side
presence check is vacuous (`harness-loop-must-be-proven-live.md` preflight 1): the pressure arms must add
`pressure` to `exp60_water_check.REQUIRED_BRIDGE_SENSORS` / `missing_bridge_sensors` and gate on
`latest_state()` keys, and the bridge must be RESTARTED after the `index.js` change
(`feedback_bridge_restart_after_sensor_change`).

**N-3 — D43 seams.** The pool-2 representation gate is Exp 61 step 4 verbatim (`WaterTrial.
encode_world_cluster` → `NAc.anticipatory_threat_need` / `cluster_fear`; the consumer is `propose_via_
substrate` → `recommend_action` reading the same need — `exp61_run.receiver` lines "4. representation +
readability gate"). First contact is `WaterTrial.placement` (loop ON, AUTONOMOUS controller through
`run_minecraft_aut`, US-free cap = pain edge − 0.75 s) — Exp 60/61's DV unit. Nothing new lacks a real
consumer except arm 4's non-existent per-sensor gain (DNB-2). Note `pressure` is the first sensor whose
value would be non-zero ONLY underwater: its Wire-4 role is entirely through the cluster id, and with w ≤
0.037 the cluster id is unchanged, so the loop-ON read cannot see it either way.

**N-4 — One agent, two pools: the seams Exp 60/61 never exercised.** Train→probe in ONE live agent is the
Exp 60 seam already (`exp60_run._run`: pre-probe → `train` → `live_g2` → post-probe in one session). New:
(a) a cross-pool RCON teleport of 100+ blocks (both shells forceloaded; `settle_until(is_in_water)` handles
arrival; the `SensorEncoder` delta gate re-encodes because the y/distance shift ≫ `min_delta` 0.05);
(b) respawn at the last-built shore (SF-5); (c) the hub session — inherit Exp 61's `_build` (hub opened,
D41/D42) + `WaterTrial.reopen_hub_session` after every loop window + `close_and_stage` for a persisted
record; Exp 60's `_run` never opened the hub, so do not copy its teardown. Use ONE `make_fresh_encoder(aut)`
for both pools (a second instance shares the EC but resets the delta stash — harmless, but the
representation gate must read through the loop's encoder).

**N-5 — The prereg's mechanism paragraph says `pressure` "enters the world vector the fear is keyed on"
as if that were a design choice.** It is the only place a `modality: world` sensor CAN go
(`_SUBSTRATE_CHANNELS` is declaration-driven); D3's "interoception (which Wire-4 never keys on)" is right
for the reason given, and the interoception channel is UNGAINED — a `pressure` there would be range-blind
mass on every cradle body's tag. World is the only coherent tag; D3 is not really open.

## What I verified

- Read: `DESIGN_REVIEW.md`; all nine `docs/wiring/*.md`; the DRAFT v1 prereg; `scripts/minecraft_bridge/
  index.js` (snapshot, `is_in_water` at eye height, `escape_water`, `STATE_INTERVAL_MS`); mineflayer
  `blocks.js::blockAt`, prismarine-world `getBlock`, prismarine-block `isWaterlogged`;
  `bodies/minecraft_player.yaml` (all 17 world sensors, ranges, rests); `backends/minecraft.py::
  sync_world_sensors`; `audio_localization.py::world_set_axis`; `agent_loop.py::_read_declared_modality_
  ranges / _read_world_states / _encode_current_clusters / _SUBSTRATE_CHANNELS`; `encoder.py::
  _stable_basis / _normalize_value / _sensor_embed / sensor_geometry_fields / encoding_geometry_tag /
  SensorEncoderConfig / encode_sensors`; `ec.py::_migrate_legacy_geometries` + the geometry-mismatch warning;
  `hivemind/cli.py` invalidate path; `scripts/lint_body_rest_neutral.py`; `tests/unit/test_encoder_golden_v1.py`
  + fixture keys/tags; `tests/unit/test_minecraft_harness.py::TestBodyFakeLockstep`; `tests/unit/
  test_minecraft_seam.py` roster pins; `minecraft_harness.py::FakeBridgeServer._snapshot / MinecraftSyncPump`;
  `water_trial.py` (init, rescue, submerge, fingerprint, clusters-distinct, placement, probe, train, live_g2,
  deaths, reopen_hub_session); `setup_world.py` (water geometry, commands, verifications, clearances, anchor
  record, `_water_classroom`, subparser args); `exp60_run.py` (FROZEN, ANCHOR_FILE, `_run`); `exp61_run.py`
  (FROZEN literal copy, `_build`, `close_and_stage`, `_Campaign`, `receiver` steps 1–6); `exp60_water_check.py`
  (ANCHOR_FILE, REQUIRED_BRIDGE_SENSORS, `_stamp_measured`); `survival_world/common.py`; the three
  `exp60_*_check.py` replays; `spec.py::_parse_entity` sensor loop; `component_registry.deep_merge`;
  `bodies/minecraft_bench_satiated.yaml` (extends precedent); the ledger's Exp 60 and Exp 61 `Re-run on:`
  lines; `exp61_pairs.jsonl` geometry tags (`g52a03986` on every tagged row).
- Ran (PYTHONPATH=src, shipped `_sensor_embed`, committed `exp60_geometry_2026-09-15b.json::per_sensor`):
  base reproduction 0.7874 = record; the cross-pool SHIPPED grid above; the `pressure` grid at ranges
  [-12,12]/[-8,8]/[-6,6]/[-5,5]/[-4,4]; the 12-deep-pool case (0.9076); the ablated identity (1.0000); the
  within-pool-2 grid (0.785–0.831); the daylight (0.5812) / night (0.8934) cases with and without `pressure`;
  the Exp 60 pool-1 delta from adding `pressure` (shore byte-identical; submerged 0.99985; within-pool
  0.7879). Numbers are from that run; nothing was measured live.
- Did NOT verify: pool-2 `light_level` inside an above-ground sealed shell (assumed 0; W1 must measure);
  the live EC's NAc threshold overrides at pool 2; bridge scan cost under a real server (reasoned from the
  code path, not profiled).

## Verdict

**DO-NOT-BUILD** as written. The replay the prereg schedules as step 2 is already answerable from committed
data and says the floor arm hits (0.916–1.000 ≫ 0.85 everywhere inside the builder's bounds), so there is
no contrast for `pressure` to move; arm 4 is an identity; `pressure` at its declared range is silent
(w ≤ 0.037). Record this as the design result. Two honest successors, for the main session to weigh with the
other lenses: (i) a cheap two-arm SHIPPED-body cross-pool probe (train pool 1 / probe pool 2 vs same-pool),
no mechanism PR, claiming "fears water at any altitude/spawn distance inside the sealed-shell apparatus
class" — reachable on the replay and needing only the SF-5 plumbing; or (ii) a cross-CONTEXT design whose
context axis is what this body actually places by (`light_level` / `time_of_day`), where the replay says
the shipped roster genuinely misses — and where `pressure` is not the remedy. If a `pressure` sensor is ever
built, it goes on an `extends:` variant body (SF-3), with a range declared from the apparatus it runs in
(SF-1), the bridge scan per N-1, and no golden regen (SF-4).
