# pymaxim 1.1.0 — "Sensorimotor"

The substrate leaves the simulator, and learns to want.

`pip install pymaxim` · [pymaxim.bio](https://pymaxim.bio) · [CHANGELOG](https://github.com/dennys246/Maxim/blob/main/CHANGELOG.md#110---2026-08-26--sensorimotor)

## Upgrade first — one promised removal and two behaviour changes for code that already compiles

- **`maxim.register_persona()` now raises** (`RuntimeError` with a pointer). The persona
  system was removed in 1.0; 0.9 deprecated the call with the promise "raises in 1.1".
- **`maxim.load.agent()` restores fully or fails loudly.** It now defaults to
  `on_corrupt="raise"`: a corrupt Hippocampus / NAc / SCN / EC file raises one
  `MemoryCorruptionError` naming every bad file instead of silently handing you fresh
  state wearing a loaded agent's name. `on_corrupt="fresh"` is the explicit opt-in.
- **`maxim.register_tool()` registrations persist** across every later `run()` /
  `imagine()` / `campaign()` (they were one-shot). New `unregister_tool(name)` and
  `clear_registered_tools()`.

## What this release shows (pre-registered, gates frozen before data)

**The want is learned — Exp 52 "Nurture".** An infant with a hunger drive and *no*
orient drive learns to turn toward its mother's voice because being fed relieves its
hunger; the operant credit is the sign of the relief the feed actually produced, nothing
hand-coded. Embodied sim, 12 seeds per arm: taught **0.878** vs a never-hungry infant fed
identically **0.441** vs no mother **0.413**. Same feed, no need → no learning; same
schedule, no contingency → no learning. → [docs/experiments/52_nurture.md](https://github.com/dennys246/Maxim/blob/main/docs/experiments/52_nurture.md)

**The want reads out on a physical body — Exp 53 / 53b.** Those infants' persisted
substrate — two JSON files each, loaded unchanged, never credited on the robot, SHA-verified
before and after — drove a Reachy Mini toward a speech source: taught **1.00 / 1.00 /
1.00** (36/36), the never-hungry controls **0.00** (no action), the never-fed controls
**0.50** (turn one way regardless of side). The first attempt (Exp 53) chose the right
direction 36/36 but overshot the smallest target with the step size we declared — recorded
as the pre-registered APPARATUS verdict; 53b changed only the step (the robot's own) and
passed. → [docs/experiments/53_cross_context_readout.md](https://github.com/dennys246/Maxim/blob/main/docs/experiments/53_cross_context_readout.md)
— the agent files ship in-repo (`docs/experiments/data/53_agents/`), so the readout is
reproducible on any Reachy Mini with the harness. [Video](https://youtu.be/lLoPM2EkbPU)
(demo, not evidence): the taught infant glances toward a voice; the never-hungry one,
loaded identically, sits still.

## Also in 1.1.0 (everything in 1.1.0rc1)

- **Stable-API contract repairs:** the three upgrade items above; NAc and EC are
  invalidated together (a restored NAc can no longer point at a fresh EC).
- **Architecture-audit regression gate:** the 33 layer-boundary findings are a reviewed
  accepted-debt baseline shipped in the wheel; CI fails on any addition;
  `maxim --audit-architecture` reports against it.
- **Healthy-hardware delivered-shift block** (H1 Part C, n = 8/side through the
  production affordance path): 0.943 / 0.942 of command — the provisional flag on the
  `_big` magnitudes is cleared; head-pose drift (D30) and a DoA fold in the credit path
  (D31) are filed, not hidden.
- **Evidence closure:** raw records committed for every Earned graduation row (S4);
  Exp 44b's non-stationarity result recorded; Exp 37's fires shown non-reproducible across
  time with code held fixed (limit L8); the DoA sweep's central gain shown to be the wrong
  statistic (L9).
- **Loudness bench (item 18):** the XVF3800 already exposes a pre-AGC speech-energy
  level over the daemon's REST API — measured, recorded, and the salience design
  deferred to 1.1.1 on purpose (it is not part of the claim above).

## Known limits, stated

One hardware session per result, n = 3 seeds per arm on the robot; the nursery's learned
representation is three azimuth bins (a source just right of centre turns the wrong way —
predicted before the run, recorded in the write-up); readout, not learning, on the robot;
nothing about loudness or magnitude selection; the LLM-driven action path is not what
these experiments measure. Full ledgers: [limits](https://github.com/dennys246/Maxim/blob/main/docs/limits/README.md) ·
[defects](https://github.com/dennys246/Maxim/blob/main/docs/bugs/README.md) · [graduation candidates](https://github.com/dennys246/Maxim/blob/main/docs/plans/behavioral_graduation_candidates.md).

## Next

1.1.1: the Reachy-native nursery body and the loudness/onset salience design. 1.2: Oasis +
Hivemind — peer substrate sharing, with this release's taught want as the case study
([plan](https://github.com/dennys246/Maxim/blob/main/docs/plans/oasis_case_study_taught_orient.md)).

Artifacts attached to this release are the exact wheel and sdist uploaded to PyPI; the
tag points at the commit that built them.
