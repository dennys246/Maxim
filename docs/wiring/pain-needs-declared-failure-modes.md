# Pain fires only from DECLARED failure modes — drives alone publish nothing

**Established:** 2026-09-13, dark=danger offline wiring probe
(`scripts/survival_world/dark_danger_probe.py`, 1.3 Step 2): 8 damage episodes (health
20→14, dark world-cluster active, real executor + `record_outcome` + a per-episode
`evaluate_failures()` tick mirroring the live loop) produced `total_published: 0` on the
PainBus and zero negative learning on every channel — while `subscriber_count: 8`
(4 direct pain subscribers, including the pain→NAc attribution) sat wired and idle.

## The fact

The pain pipeline is `sensor change → Body.evaluate_failures() → FailureEvent →
PainBus.publish → subscribers (pain→NAc negative links, percept valence, hippocampus)`.
`evaluate_failures` evaluates **declared failure modes** — and a body that declares none
publishes **no pain, ever**, no matter how violently its sensors move. Drives
(`HomeostaticDriveSpec`/`EntropicDriveSpec`) are a *separate* system: they produce drive
pressure/corrective need (R2 break 1) and measured-relief credit (break 2), but NOT
PainSignals. `bodies/minecraft_player.yaml` declares 17 world sensors and two drives and
**zero failure modes** — so Minecraft damage is painless to the substrate. Five other
bodies (base_humanoid, reachy_mini, host_machine, cybernetic_arm, megarm_v3) declare
failure modes; the survival body never got them.

## Why it bites

- Any "learned aversion / dark=danger / damage avoidance" design premised on "the
  negative-credit path is already wired" is **false on this body**: the path is wired
  *downstream* of the PainBus but starved at the source. An experiment built on it
  nulls with a cause that predates the design.
- The absence is silent and looks healthy: subscribers attached, `evaluate_failures`
  returns cleanly (an empty list), no warnings. **Check `pain_bus.get_stats()`
  `total_published`, not `subscriber_count`** — attached listeners prove nothing about
  signal flow (the same absent-reads-as-present family as the green-PR lesson).
- `record_outcome`'s drive fields cannot substitute: measured relief/worsening is scoped
  to the acting tool's modeled effect (eat→food), so an off-effect health drop rides
  through as a *successful* action and trains a POSITIVE causal link — 8 damage episodes
  made the agent like eating-while-bitten (`causal_pos 0.78`).

## The fix shape (named Phase-1 build item)

Declare the survival body's failure modes (health-damage nociception) in
`minecraft_player.yaml` — a body declaring what hurts is D1-clean (the trigger fires on
the game's own health drop; every embodied body already does this), not a synthetic
sensor. Then RE-RUN the probe: with pain publishing, the A/B readouts (state-blind
negative causal link vs cluster-keyed negative bias, world vs interoception key) become
the meaningful measurement of where fear lands.

## See also

[substrate-learning-channels.md](substrate-learning-channels.md) (the two channels pain
would feed); `docs/experiments/r2_drive_premise_check.md` (drives ≠ pain: R2's breaks were
the drive system's; this is the nociception system's); `src/maxim/proprioception/pain_bus.py`
(`create_pain_nac_subscriber` — the wired-and-waiting consumer).
