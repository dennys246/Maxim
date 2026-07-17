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

## Measured DoA response (2026-07-16) — and the retraction that produced it

**RETRACTED (2026-07-16, same day): an earlier version of this section claimed
"the XVF3800 DoA is a TRACKING estimator, not a memoryless measurement — tracked
gain 0.58 az/rad, loses lock on large jumps." That finding was an ARTIFACT of a
bug in our own motion code, not a property of the chip.** It is preserved here as
a correction because the failure mode is instructive and the numbers are still
cited elsewhere.

### The bug that faked a sensor pathology

Every measurement behind that claim commanded `goto_target(body_yaw=X)` with
`head=None`. Per Pollen's own
[AGENTS.md](https://github.com/pollen-robotics/reachy_mini/blob/main/AGENTS.md),
**the head 4x4 pose is in the WORLD frame and sits above `body_yaw` in the
kinematic chain**: `head=None` makes the daemon re-solve IK against the *retained*
world-frame head target, so the Stewart platform **counter-rotates to hold the
head's absolute orientation** while the body pivots underneath. **The mic array is
in the head.** So the array barely turned: measured **0.32 rad of mic rotation for
a 0.9 rad body command** (`d(head)/d(body)` = +0.214 in world frame).

Reading a nearly-stationary array while believing it had rotated produces exactly
the signature of a lagging sensor — proportional shortfall, step-size-independent,
wildly variable run to run. It survived six competing hypotheses (settle lag,
backlash, motor under-travel, speech density, increment size, slow adaptation),
each falsified in turn, before the vendor's docs settled it.

### What the chip actually does — TRUE characterization (post-headfix sweep, 2026-07-16)

The first honest sweep of an array that actually rotates. **The XVF3800 DoA is an
excellent sensor**; every pathology previously attributed to it was our bug.

| property | value |
|---|---|
| gain | **0.57 az/rad** full-range fit (central 0.546-0.548; settle test 0.562; mag2 sign-check 0.58 — four independent measurements agreeing within 0.03) |
| linearity | **R² = 0.9982** over the full ±1.4 rad (±80°), *both* sweep directions |
| intercept | +0.001 / +0.014 (source centred) |
| hysteresis | **0.015** mean asc-vs-desc (was 0.109-0.176 with the bug) |
| monotonicity | **complete across ±1.4 rad** — zero non-monotonic zones |
| convergence | **0.23 s** after a 0.9 rad turn, then stable |
| per-pose noise | 0.022 spread (2× the ~1° quantization) |
| speech gate | 23-100% (median 50%) — median-of-k + gating still required |

Gain 0.57 vs the geometric 0.637 is a modest ~10% under-read — plausibly beamformer
angular compression and/or the array sitting off the rotation axis. Stable and
reproducible; not a pathology.

**Retractions now closed by measurement:**
- ~~"tracking estimator, loses lock on large jumps"~~ — **refuted**: R²=0.998, 0.23 s convergence.
- ~~ascending/descending asymmetry~~ — **was head-drag hysteresis**: 0.109-0.176 → 0.015 once the head rides along.
- ~~endfire bimodal zone~~ — **not observed within the swept ±80°** (monotonic throughout). True endfire is ~90°, beyond the sweep — unreproduced where we looked, not disproven in general. The |az| ≤ 0.65 cap was set for the artifact and can widen to ~0.85 **but is load-bearing for a different reason until the controller fix ships** (it keeps placements off the yaw clamp — see Exp 45).
- ~~"gain drifts between sessions"~~ — **refuted**: four measurements, ±0.03.

### The flip point — a derived design constant

The magnitude question ("how far should I turn?") has a **derived** boundary. A
correct-direction step of shift `S = |delta| * gain` takes `|az|` to `|az - S|`, so its
relief is `az - |az - S|`. Step A therefore beats step B exactly when `|az - S_A| <
|az - S_B|` — when az is **nearer A's shift**. The boundary is their midpoint:

    az_boundary = gain * (|delta_big| + |delta_normal|) / 2    # Reachy: 0.546 * 1.2/2 = 0.328

The optimal magnitude policy is simply **nearest-neighbour quantization of |az| onto
the available shifts**, and with N magnitudes there are N-1 such boundaries.

*(Correction: an earlier version of this section gave `|delta_big| * gain / 2` = 0.246.
That is where big's relief crosses zero — a different quantity that decides nothing on
its own. The error was caught by deriving the comparison instead of reasoning about it.)*

Any state bin that **straddles** `az_boundary` contains two opposite correct answers and
cannot be learned cleanly — it receives consistently contradictory evidence. This is
measurable per robot from its own gain, and it is why
[Exp 45b](../../experiments/45b_orient_magnitude.md) scored magnitude 0.75: the `near`
bin spans 0.1-0.5, straddling 0.328. [Exp 45c](../../experiments/45c_flip_bins.md) tests
the fix.

### Why direction learning survived and magnitude did not

**Direction is sign-based; magnitude is threshold-based.** A proportional gain
error leaves every sign intact, so Exp 45's direction/cross-session/merge results
are unaffected by the bug. Magnitude learning depends on *overshoot* — a
threshold on `|delta| * gain` vs `2 * |az|` — which a proportional error destroys.
That asymmetry is why Exp 45 sailed through and Exp 45b was incoherent until the
head was fixed.

---

## Sources

- Reachy Mini media stack — https://huggingface.co/blog/pollen-robotics/reachy-mini-media-stack
- Advanced media controls (the verbatim 2-channel / DoA documentation) — https://huggingface.co/docs/reachy_mini/en/platforms/reachy_mini/media_advanced_controls
- Wireless hardware datasheet — https://huggingface.co/docs/reachy_mini/en/platforms/reachy_mini/hardware
- Lite hardware datasheet — https://wiki.seeedstudio.com/reachymini_platforms_reachy_mini_lite_hardware/
- Python SDK — https://huggingface.co/docs/reachy_mini/en/SDK/python-sdk
- reSpeaker XVF3800 host control — https://github.com/respeaker/reSpeaker_XVF3800_USB_4MIC_ARRAY
- XMOS XVF3800 datasheet — https://www.xmos.com/documentation/XM-014888-PC/html/modules/fwk_xvf/doc/datasheet/02_overview.html
