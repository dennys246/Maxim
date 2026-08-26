# Embodiment — hardware platform guides

This directory documents **how Maxim embodies specific hardware platforms** — what each robot's sensors and actuators can and cannot do, and how those capabilities map onto Maxim's perception → SEM → substrate → action loop.

It complements, rather than replaces, the existing embodiment docs:

- [`../embodiment_guide.md`](../embodiment_guide.md) — how to *author* a SEM entity (sensors, modulators, affordances, drives) in YAML.
- [`../embodiment_yaml_reference.md`](../embodiment_yaml_reference.md) — the YAML field reference.
- [`../proprioception.md`](../proprioception.md) / [`../harm.md`](../harm.md) — the interoceptive drive + pain machinery.

The pages here are **per-platform**: a concrete robot, its real hardware limits, and the body-YAML + placement choices that follow from them.

## The capability-driven principle

Maxim does not hardcode any particular robot. A platform exposes a set of **capabilities** — which sensors it has, what they can resolve, which actuators it can drive — and the agent adapts to whatever the platform **declares** in its body YAML (sensors, drives, affordances). The same cognition/substrate code runs unchanged across platforms; the differences live entirely in the declarations.

A platform page therefore answers three questions:

1. **What can the hardware actually sense and do?** (and, just as important, what it *can't* — the limits are load-bearing).
2. **How does the body YAML declare those capabilities** so Maxim engages exactly the drives/affordances the hardware supports?
3. **Where does each perception stage run** (the [perception pipeline placement](../plans/perception_pipeline_placement.md) model) — for a self-contained robot, all-local; for a sensor-only peer, distributed.

When a capability is absent (e.g. a microphone array that can't resolve elevation), the body simply doesn't declare it — no code change, no dead config. That "declare what the hardware supports, adapt the rest" pattern is the whole point.

## Cross-platform guides

- [**Porting the orient-to-center learning loop**](porting_orient_loop.md) — the contract
  a new robot (Atlas-class included) must satisfy to run the substrate orient-policy
  learning: what's already robot-agnostic, the four things each robot supplies, the
  the per-robot calibration protocol (**starting with: verify your sensing frame
  actually rotates**),
  **why the design constants must be DERIVED from your robot's own measured gain** (the
  state bins and the action set are duals — bins must be the Voronoi cells of the action
  shifts), **how to submit gauntlet-passed substrate for fleet distribution**, and a
  **failure-mode appendix** (six hypotheses died before the real bug — a wrong actuation
  assumption is indistinguishable from a broken sensor). Validated end-to-end on Reachy
  Mini: [Exp 45](../experiments/45_reachy_orient_live.md) (direction 0.00 → 1.00 in ~10
  trials; cross-session 1.00 at trial 0) + [45c](../experiments/45c_flip_bins.md)
  (**magnitude 1.00** — which way *and how far*).

## Platforms

- [**Reachy Mini**](reachy_mini/README.md) — Pollen Robotics / Hugging Face desktop robot. 6-DOF Stewart head + body yaw, camera, 4-mic array (behind an XVF3800 DSP chip), speaker. A full sub-guide: [getting started](reachy_mini/getting_started.md) (network + SDK **version matching** + first connect), [troubleshooting](reachy_mini/troubleshooting.md) (symptom-indexed, hardware-validated), [engineering reference](reachy_mini/engineering.md) (WS transport, REST endpoints incl. network DoA, motion semantics), and the [audio sound-localization deep-dive](reachy_mini/audio_localization.md) (DoA vs ITD/TDOA, why elevation is impossible on its linear array, **plus the measured DoA response**: a near-perfect linear sensor — gain 0.57 az/rad, R²=0.998 over ±80°, 0.23 s convergence — and the retraction of an earlier "tracking estimator" claim that turned out to be a bug in *our* motion code). **The `head=None` trap lives here too**: `goto_target(body_yaw=X)` counter-rotates the head, so head-mounted sensors do not turn — see the [CLAUDE.md](../../CLAUDE.md) invariant. First substrate-learned hardware policy: sound orienting ([Exp 45](../experiments/45_reachy_orient_live.md) direction + [45c](../experiments/45c_flip_bins.md) magnitude).
- **First nursery-taught want on hardware:** [Exp 53/53b](../experiments/53_cross_context_readout.md) — the Exp 52 infants' persisted NAc+EC loaded unchanged onto the Reachy Mini turn toward a speech source (taught 1.00, never-hungry control 0.00, never-fed 0.50); agent files in `docs/experiments/data/53_agents/`, readout harness `scripts/orient_backbone/exp53_cross_context_readout.py`.

## Adding a platform page

A new platform page should cover: hardware overview; a capabilities/limitations table; the body-YAML declarations (sensors/drives/affordances) that follow from them; the perception-placement story (all-local vs distributed); and a deep-dive on any sensing modality with non-obvious hardware constraints (the kind that would otherwise surface as a nasty surprise at the bench). Cite primary hardware/SDK sources.
