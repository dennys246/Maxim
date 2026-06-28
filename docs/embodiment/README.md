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

## Platforms

- [**Reachy Mini**](reachy_mini.md) — Pollen Robotics / Hugging Face desktop robot. 6-DOF Stewart head + body yaw, camera, 4-mic array (behind an XVF3800 DSP chip), speaker. Includes a deep-dive on **audio sound-localization** (DoA vs ITD/TDOA, why elevation is impossible on its linear array, and how Maxim consumes the chip's onboard direction-of-arrival).

## Adding a platform page

A new platform page should cover: hardware overview; a capabilities/limitations table; the body-YAML declarations (sensors/drives/affordances) that follow from them; the perception-placement story (all-local vs distributed); and a deep-dive on any sensing modality with non-obvious hardware constraints (the kind that would otherwise surface as a nasty surprise at the bench). Cite primary hardware/SDK sources.
