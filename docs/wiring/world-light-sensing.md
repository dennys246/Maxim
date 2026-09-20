# Light sensing in a world: block light is not brightness

**Established:** 2026-09-13, first live 1.20.4 survival-world session (bridge fix verified
live: `light_level` 0 → 15 at the same surface position, noon, after the read changed).
Resolves the "light_level read DEAD (0 everywhere)" observation from Exp 56.

## The fact

Minecraft (and engines with the same lighting model) stores **two** light fields per block,
and neither alone is "how bright it is here":

| field | fed by | value on a sunlit surface | value in a torch-lit room | value in a dark cave |
|---|---|---|---|---|
| **block light** | torches, lava, glowstone… | **0** | 14 | 0 |
| **sky light** | sky exposure (NOT time of day) | 15 | 0 (roofed) | 0 |

Perceived brightness — the debug-screen quantity — is
`max(block_light, sky_light − darkness(time))`, where darkness ramps 0 → 11 over dusk
(ticks 12000–13800), holds 11 through the night, and ramps back over dawn (22200–24000).
A moonlit surface reads ~4; noon reads 15; a cave reads 0 day or night.

## Why it bites

A bridge that reports raw `getBlockLight` reads **0 in broad daylight** — indistinguishable
from a lethal cave. That is exactly what Exp 56 saw ("light_level dead, 0 everywhere"): the
bench world ran in daytime with no torches, so block light was *correctly* 0 everywhere. The
sensor was never broken; it measured the wrong quantity, and every design that ruled out
light-keyed contingencies ("cave-distinctness may have to rest on block-census / altitude")
inherited the misdiagnosis.

`scripts/minecraft_bridge/index.js::perceivedLight` now computes the effective value (both
halves game-exposed, D1-clean). Sky light is *time-independent* — the `darkness(time)` term
is what makes night dark; forgetting it makes midnight read like noon.

## How to verify (the 60-second instrument check)

Stand the bot on an open surface at `time set day` and read one state line: `light_level`
must be ~15. Then check the dark side (a cave or a roofed box): ~0. If daylight reads 0, the
read is block-light-only; if midnight reads 15, the darkness term is missing. The bridge
prints a `spawn state:` line to stdout for exactly this check.

## Traps

- **1.18 changed the spawn rule:** hostile mobs spawn only at **block light 0** (was ≤7 in
  1.16.x). So *block* light is the spawn-relevant field, while *perceived* light is the
  agent-relevant sensor — a torch (block light > 0) suppresses spawns even where perceived
  light is low. The dark=danger contingency holds under both, but thresholds calibrated on
  1.16.x do not port.
- **`hostile_count` counts through walls.** The first 1.20.4 snapshots read 26–30 hostiles
  at noon — cave mobs under spawn (1.18+ caves are large). Hostile pressure is not
  surface-visible threat; pair it with light/altitude before treating it as "danger here".
- A light-keyed classroom still needs the **encoder separability check** (the instrument
  lesson): a live sensor is necessary, not sufficient.

## See also

`docs/plans/archive/survival_world_1_3.md` §"World vs. substrate" (the dead-sensor caveats this
entry retires); Exp 56 phase-0 instrument notes; [substrate-learning-channels.md](substrate-learning-channels.md).
