# Rung B SUPPORT — environment lens (design review, 2026-09-25)

Evidence: the bridge (`index.js::perceivedLight`/`skyDarkness`), mineflayer's `time.js`, `setup_world.py::water_classroom_geometry`, and an offline sweep through the shipped `_sensor_embed`. On a pool-floor vector the sweep reproduces Exp 61's replay (0.892 / 0.888 / 0.797 against 0.892 / 0.884 / 0.799).

## DO-NOT-BUILD

**D1: the place decides the verdict, and this protocol measures the wrong place.** (Location: §Protocol "idles where it spawned", §What each outcome means.) At an open spawn, `light_level` = 15 − the bridge's linear sky darkness. It crosses the 7.5 midpoint at dusk, so the light component flips sign. In the sealed shell, light is 0 all day: day reads 0 and the formula can only go lower. So only `time_of_day` moves there. Offline, for one day starting at tick 1000:

| place | exact | middle | beyond |
|---|---|---|---|
| open spawn (y 64, light moving) | 0.26 | **0.27 → SUPPORT** | 0.47 |
| pool floor (the fear node) | 0.95 | **0.05 → no SUPPORT** | 0.00 |
| pool shore (dry) | 0.26 | 0.68 | 0.06 |

The same day gives opposite verdicts at the fear-learning place and at spawn. The spawn result is SUPPORT for a state nobody learned a fear in: the first snapshot is the fear-learning *time*, not the fear-learning *state*. The "mostly exact" outcome reading also runs backwards: spawn moves *more* than the shell.

**The rig adds no information as designed.** Mobs, weather and movement are off, and time is game-deterministic. Light at a known cell follows a closed form in `time_of_day`. So the trace is predictable from the body YAML plus the bridge formula. Fix: make the offline sweep over the fear node the primary measurement (it is free and exact). Use the rig only for what is not deterministic (next item), or as a known-answer check that the bridge follows its formula.

## SHOULD-FIX

1. **"Night" is really the wrap.** At the pool floor, midnight (tod 0.75) reads **0.902, which completes**. The whole middle band is tod 0.94–0.99, the minutes before the 1.0→0 wrap: the farthest point in *scalar* distance from 0.0417, not night. The roadmap's "fear misses at night" and this prereg's "the 0.799 night row" both inherit this. The YAML's sin/cos wrap note is the lever. Restate the claim before any run.
2. **Where the bot stands is not controlled.** Mineflayer logs in where the player last logged out. For username `maxim`, that is likely an Exp 60–62 shore (its personal `spawnpoint` was set there). Fix: `tp` the bot to a declared cell, then gate the first snapshot on y, `light_level` (15 by day in the open, or 0 in the shell), `distance_from_spawn` and `is_in_water`, and record the cell.
3. **The start tolerance of 0.01 is 240 ticks, about 12 s.** That must cover RCON, a tmux hop, importing `maxim`, provenance and the connect. Fix: `time set 1000` with the cycle off, start the capture, then run `gamerule doDaylightCycle true`, and have the analyzer drop the frozen leading run.
4. **24 minutes is 1.2 days.** The extra 4 minutes re-sample tod 0.04–0.24, all of which is exact, so they dilute `middle` and bias against SUPPORT. Fix: analyze exactly one cycle (unwrap the ticks and stop on the return to the reference), or weight by bin.
5. **Holding weather and mobs off removes the only non-deterministic support.** `is_raining` = 1 alone reads 0.8525, and real nights spawn hostiles. If the rig is kept, a second arm with weather on (`weather rain`, timed) is the only part a rig can actually tell you.

## NIT

- Make the analyzer REFUSE on `varying_world_sensors` beyond light and time, as the prereg says; today it only reports them. Stop the other bridge and any spectating operator, since `nearest_player_dist` would move.
- Afterwards restore `doDaylightCycle false` and `time set 1000`. Check the rig's `player-idle-timeout` is 0.
- Sleep and chunk unload are not a risk: world time ticks globally.
