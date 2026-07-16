# Reachy Mini — audio & sound localization (deep-dive)

Why the Reachy Mini can tell left from right but not up from down, why you
cannot build your own TDOA front-end on it, and how Maxim consumes the
chip's onboard direction-of-arrival instead. Platform overview, setup, and
API reference live in the [folder README](README.md) /
[getting_started](getting_started.md) / [engineering](engineering.md).

> **Status:** scoping reference for the perception-pipeline-placement work ([plan](../../plans/perception_pipeline_placement.md)). The audio findings are HIGH-confidence (official Pollen / Seeed / XMOS docs); inter-mic spacing and DoA angular resolution are undocumented and need bench measurement.

## Capabilities & limitations

| Capability | Reachy Mini | Notes |
|---|---|---|
| Head yaw / pitch | ✅ | 6-DOF head + body yaw; task-space poses |
| Vision | ✅ | single camera |
| Sound **azimuth** (left↔right) | ✅ (via onboard DoA) | 180° range |
| Sound **elevation** (up↕down) | ❌ | **linear/coplanar mic array — no vertical baseline** |
| Sound **front/back** disambiguation | ❌ | cone-of-confusion on a linear array |
| Custom multi-mic **TDOA/ITD** | ❌ | XVF3800 exposes only 2 channels (see below) |
| Onboard **DoA** (direction of arrival) | ✅ | `mini.media.get_DoA()` |
| Audio sample rate | 16 kHz | host-exposed max |

---

## Audio & sound localization (deep-dive)

### The theory: how time-of-arrival gives direction

A sound reaching two microphones a few centimetres apart arrives at one slightly before the other. That **inter-microphone time difference** (ITD), generalized across a mic array as **time-difference-of-arrival (TDOA)**, encodes the **azimuth** (horizontal angle) of the source — this is how the brain's superior olivary complex localizes sound left-to-right.

Two consequences matter for any robot:

- **Elevation needs a vertical baseline.** A pair (or line) of microphones all at the same height produces the *same* time difference for a source at a given azimuth regardless of how high or low it is — the **cone of confusion**. To recover elevation from timing you need microphones separated *vertically* (a non-coplanar array). Mammals instead recover elevation from **spectral cues** (the pinna filters sound differently by elevation) — a much harder, separate method.
- **A linear array has front/back ambiguity.** A straight line of mics can't tell a source in front from its mirror image behind.

### The Reachy Mini array: linear, coplanar, behind a DSP chip

The 4 MEMS mics are arranged in a **linear (1-D, coplanar) array** across the head — mic 0 near the right antenna through mic 3 near the left. Two hard limits follow:

**1. No elevation, by geometry.** The array is coplanar — there is no vertical baseline — so only **azimuth** is recoverable, over a **180°** range, with front/back ambiguity. Elevation is impossible on this hardware without physically adding a vertically-offset microphone.

**2. No custom TDOA, by interface.** The 4 mics feed a **Seeed reSpeaker XVF3800 (XMOS)** voice-processor, which sits between the raw mics and the host. The chip exposes only a **2-channel, 16 kHz, already-beamformed/AEC-processed** stream. Per Pollen's own docs: *"the microphone array outputs a stereo channel, so it is not possible to get the raw output of all 4 mics at once"* — you can route at most **2 raw mics at a time**, never all four sample-aligned. This is a **chip-level USB-interface limit**, so dropping to ALSA/PortAudio does **not** bypass it. A classic 4-mic cross-correlation TDOA front-end therefore **cannot be built** on the stock device.

### What Maxim uses instead: the chip's onboard DoA

The XVF3800 computes **Direction-of-Arrival on-chip** and the SDK exposes it directly:

```python
# ONBOARD (the client-side call reads the mic array over local USB —
# SDK >= 1.5 makes this onboard-only):
doa_radians, is_speech_detected = mini.media.get_DoA()

# OFF-ROBOT (laptop / peer): the daemon reads the chip and serves it:
#   GET http://<robot>:8000/api/state/doa
#   -> {"angle": <radians>, "speech_detected": <bool>}
# convention (both paths): 0 = left, π/2 = front/back (ambiguous), π = right
```

So the azimuth signal the orient loop needs is **already computed** — free, running alongside the chip's echo-cancellation, with a built-in `is_speech_detected` flag. Maxim consumes it rather than computing its own:

```
get_DoA()  ──►  normalize  ──►  "audio" substrate modality  ──►  centeredness drive  ──►  orient affordance
 (on-chip)      (−π/2,+π/2)     (frozen-centroid EC node)        (set-point = front=0)     (head yaw via set_target)
                → azimuth∈[-1,1]                                      → drive-pain            → NAc learns the policy
```

- **Normalization:** map `doa → azimuth = (doa − π/2)/(π/2)` so left = −1, front/back = 0, right = +1. Emitting it already in `[-1, 1]` means the shared `_normalize_value` applies unchanged (see plan Q5).
- **Gating:** only update the drive when `is_speech_detected` is true — the hardware's own "is there a sound to localize" signal, which neatly handles the "a single transient clap is gone before the head finishes turning" problem (the source must persist across the LLM-gated cognition cycle).
- **Substrate:** the azimuth reading is encoded under the `"audio"` modality, which is **frozen-centroid** (a densely-streamed continuous signal would otherwise walk a running-mean centroid into collapse — see the [EC centroid-drift lesson](../../../CLAUDE.md)).
- **Re-measurement closes the loop:** after the head turns, `get_DoA()` re-reads the new relative angle — no world-model needed in code; the physics does it.

**The thesis consequence:** because localization happens on-chip, Maxim is **not learning to localize** — it's learning the **sensorimotor orient policy** (turn which way, how much, to drive azimuth-error → 0), credited by drive-pain reduction through NAc. That's the real, defensible embodied-learning claim on this platform.

### Why this validates the capability-driven design

The Reachy Mini is exactly the "**coplanar → declare azimuth + yaw, stop**" case the [placement plan](../../plans/perception_pipeline_placement.md)'s multi-axis design was built for. Its body YAML declares an `azimuth` sensor + yaw orient affordances and nothing more. A *different* robot with a **non-coplanar raw multi-channel array** would declare an `elevation` sensor + pitch affordances too, and run a real TDOA front-end — **with no change to Maxim's cognition/substrate code**. The hardware difference lives entirely in the body declaration. That's the payoff of declaring capabilities instead of forking on them.

### Limitations summary

- **Azimuth only, 180°** — no elevation (linear array), no front/back disambiguation.
- **Localization is on-chip and opaque** — we read the angle but not its (undocumented) resolution, and the algorithm isn't ours to tune or make learnable.
- **16 kHz** host stream; **no sample-aligned 4-mic raw** access (XVF3800 limit).
- **Custom TDOA / elevation require different hardware** — an external raw multi-channel array with known 2-D/3-D geometry. That's a hardware change, not an SDK change.

---

## Measured DoA response (2026-07-16 static sweep — hardware-validated)

[`scripts/orient_backbone/doa_sweep.py`](../../../scripts/orient_backbone/doa_sweep.py)
mapped measured azimuth vs commanded base pose (±1.4 rad, 0.1 rad increments,
ascending + descending, 5 speech-gated reads/pose, fixed front speech source).
These numbers supersede any assumed-linear model — the orient learning loop
(Exp 45) initially mis-modeled the sensor and paid for it:

1. **The DoA is a TRACKING estimator, not a memoryless measurement.** Under small
   incremental motion it is nearly perfectly linear: **tracked gain ≈ 0.58 az/rad**
   (geometric prediction 0.64), quantization ≈ 1°, very tight per-pose reads. After a
   single large jump (1.4 rad in one command) it **loses lock and pins near the stale
   estimate** — one half-sweep stayed wrong end-to-end; worst-pose hysteresis 0.63.
   **Consequence:** any apparatus or behavior that rotates the robot substantially
   must move in small tracked increments (the orient scripts walk ≤ 0.3 rad per step).
   Apparent "gain" measured from ad-hoc large steps is meaningless (we variously
   measured 2-3× and 0.15× geometric before the sweep separated tracked from jump
   behavior).
2. **Endfire bimodal zone.** With the source ~90° off the array axis, samples flip
   bimodally between two stable values (~0.28 and ~0.72 normalized in our setup) —
   the linear array's endfire degeneracy. Reliable tracked readings extend to
   ~|azimuth| 0.69; the orient loop caps placement targets at |az| ≤ 0.65.
3. **Speech-gate rate varies wildly by pose** (5-100%); always gate + median-of-k.

Re-run the sweep after ANY acoustic change (shell mods, mounts, rooms) — it is the
before/after instrument for the eared-shell experiment
([substrate_native_orienting.md](../../plans/substrate_native_orienting.md) follow-ups).
Behavioral results built on this sensor: [Exp 45](../../experiments/45_reachy_orient_live.md).

---

## Sources

- Reachy Mini media stack — https://huggingface.co/blog/pollen-robotics/reachy-mini-media-stack
- Advanced media controls (the verbatim 2-channel / DoA documentation) — https://huggingface.co/docs/reachy_mini/en/platforms/reachy_mini/media_advanced_controls
- Wireless hardware datasheet — https://huggingface.co/docs/reachy_mini/en/platforms/reachy_mini/hardware
- Lite hardware datasheet — https://wiki.seeedstudio.com/reachymini_platforms_reachy_mini_lite_hardware/
- Python SDK — https://huggingface.co/docs/reachy_mini/en/SDK/python-sdk
- reSpeaker XVF3800 host control — https://github.com/respeaker/reSpeaker_XVF3800_USB_4MIC_ARRAY
- XMOS XVF3800 datasheet — https://www.xmos.com/documentation/XM-014888-PC/html/modules/fwk_xvf/doc/datasheet/02_overview.html
