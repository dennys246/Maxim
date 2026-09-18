# World sensors clamp to their declared ranges — compare against sensed values

**Established:** 2026-09-13, first live run of the survival Phase-0 instrument check
(`scripts/survival_world/instrument_check.py`): 40/60 samples "never settled" because the
harness waited for `y_altitude ≈ 151` from a sensor whose body-declared range is `[0, 128]`.

## The fact

`sync_world_sensors` writes bridge truth into the body's declared sensors via
`world_set_axis`, and each sensor is **clamped to the range its body YAML declares**
(`bodies/minecraft_player.yaml` even annotates `y_altitude: range: [0, 128]` with
"y>128 clamps — the lever, applied"). The vital metric is therefore the *sensed* value, not
the world's: a bot standing at y=150 reads `y_altitude = 128`.

## Why it bites

Any predicate, assertion, or settle loop that compares a vital metric against **raw world
truth** (a teleport target, an RCON-queried position, a computed altitude) can be
unsatisfiable when the truth lies outside the declared range — and the failure looks like
"the world/bridge never reflected the command" (a lag/stall diagnosis) rather than what it
is. The same applies to every ranged world sensor: `distance_from_spawn` and the offsets cap
at ±128, `hostile_count` at 32, `nearest_*_dist` at 64.

**Rule:** when a harness needs "did the world state I commanded arrive in the body?", clamp
the expected value through the sensor's declared range first — read the ranges from the
production ranger (`agent_loop._read_world_ranges(executor)`), never re-typed from the YAML —
and guard that the clamped expectations for your conditions still differ by more than the
tolerance (two positions above the cap are indistinguishable to the sensor).

## The design flip side

Clamping is not a bug — it is the L11 lever: a bounded range keeps a sensor's normalized
swing meaningful. But it means the AGENT cannot distinguish states beyond the cap (all
altitudes above 128 are one altitude to this body), so classroom/apparatus designs must
place their discriminating states *inside* the declared ranges of the sensors expected to
carry them.

## The reservoir behind a clamp (R3, 2026-09-17)

A clamp can hide a QUANTITY the world keeps spending. The apparatus heal (`WaterTrial.heal`:
`effect give … saturation`) sets the player's TRUE saturation to ≈ 20; the bridge clamps the
sensed `saturation` at 10. In the R3 pilot's unrescued drowning the health series held 18–20 for
nine seconds after damage onset while every sample read saturation 10 — the game was spending 20 →
10 underneath — and health collapsed the moment the sensed value reached 0. The floor arm's
regeneration-on margin to death (5.2 s) and its health lost are quantities of that INVISIBLE
reservoir; a game-native start (respawn saturation 5) would move them by ≈ 6–7 s, and no
fingerprint over sensed values would notice. Rule: **when a frozen number depends on a clamped
sensor's true value, read the true value by a game-native path and freeze THAT** — R3 reads
`foodSaturationLevel` / `foodLevel` / `foodExhaustionLevel` over RCON (`data get entity`) at every
event teleport (`WaterTrial.read_food_state`), records them per row, and the gauntlet file carries the
accepted band (`r3_run.write_gauntlet` → `gauntlet_drift` refuses a row outside it). The confounding
lens's delta report has the series: `docs/experiments/rationale/r3-survival-benchmark/confounding-v3.md`.

## See also

[world-light-sensing.md](world-light-sensing.md) (the sibling sensor-semantics lesson);
`docs/limits/l11_sensor_dilution.md` (why ranges are bounded at all).
