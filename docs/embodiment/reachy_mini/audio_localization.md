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

## 2026-08-05 session — the version-skew incident (settled) and a CONTESTED second curve (unreconciled)

Full-envelope static sweep (`scripts/orient_backbone/doa_sweep.py`): body yaw
swept ±1.4 rad in 0.1 rad steps (head riding along — post-headfix path,
`automatic_body_yaw` off), fixed sustained speech source ~1–2 m in front of the
neutral heading, 5 speech-gated reads per pose, ascending + descending passes.
Raw data: [`data/2026-08-05_doa_sweep_skewed_and_matched.jsonl`](data/2026-08-05_doa_sweep_skewed_and_matched.jsonl)
(labels `post-1.9-daemon` = the invalid skewed run, `matched-versions` = the
real one). Daemon 1.8.3 throughout.

### The incident first: SDK/daemon version skew silently corrupts BOTH surfaces

The first sweep ran with **SDK 1.9.0 against daemon 1.8.3** (the SDK printed a
version-mismatch RuntimeWarning and carried on). Result: central gain
**+0.015/rad** — a ~40× collapse, readings pinned near az 0 at every pose, with
non-monotonic garbage excursions. The same session's live run had
`goto_target` **motion commands rejected by the controller**. One cause, two
independent symptom surfaces (sensing AND control). Pinning the client back
(`pip install reachy-mini==1.8.3`) restored both: the very next sweep, same
room, same source, measured central gain +0.195/rad with clean sign structure.

**Rule (this is the existing version-match invariant, now with a measured
blast radius):** a skewed client does not fail loudly — it produces
plausible-looking flat data and intermittent motion rejections. Version-match
FIRST (`curl http://<robot>:8000/api/daemon/status` vs
`importlib.metadata.version("reachy_mini")`) before believing any measurement,
and never label a data file with an unverified topology guess (the skewed
run's label says `post-1.9-daemon`; the daemon was 1.8.3 the whole time — it
was the SDK that moved).

The same session also re-verified actuation with the matched stack
(`yaw_verify.py`, daemon-side pose readback): commanded→actual body travel
ratio **0.955** (0.961 small steps / 0.928 large — mild duration limiting, no
scale error) and **d(head)/d(body) = +1.007** — the head, and therefore the
mic array, rides the body essentially 1:1. The 2026-07-16 head-frame fix
holds.

### A CONTESTED second curve — RESOLVED 2026-08-08: degraded-platform artifact

> **RESOLVED by the H1 healthy-hardware re-sweep
> ([results](../../experiments/protocols/h1_healthy_hardware_doa_preregistration.md#results--appended-2026-08-08-session-1-parts-a--b-complete-envelope--part-c-pending-recalibration),
> data [h1_doa_sweep.jsonl](data/h1_doa_sweep.jsonl)).** After motors 2+3 —
> broken for the entire 1.0+ era — were replaced, a version-verified two-geometry
> sweep on `executed_git_hash 38aaddea` measured **gain 0.578 (front, replicated
> at 0.575) / 0.645 (63° displaced, ≈ the geometric 0.637), R² 0.984–0.996,
> monotone, NO staircase at either geometry** (the displaced curve folds at
> ±π/2 relative bearing — linear-array physics, not quantization). Per this
> section's own reconciliation rule, the staircase below was an **instrument
> artifact of the degrading motors**, and the 0.57-vs-0.19 discrepancy resolves
> as **progressive mechanical degradation** — both historical sweeps were real
> measurements of a platform in two states of decline. The data below is
> RETAINED as the degraded-platform record; nothing downstream may build on it.
> Residual caveat: the healthy platform still carried a constant motor-zero
> offset at measurement time (H1 finding F2 — slope-invariant, so the gains
> stand; the offset explains the fitted ψ₀ ≈ −0.13 rad).
>
> The original unreconciled banner and comparison table follow, kept verbatim
> for the historical record. The 08-05 sweep **contradicted the 2026-07-16
> characterization by ~3× in gain and ~13× in quantization**:
>
> | | 2026-07-16 (above) | 2026-08-05 (below) |
> |---|---|---|
> | gain | **0.57 az/rad**, four independent measurements within ±0.03 | 0.19/rad central |
> | linearity | **R² = 0.9982**, both directions | plateaus + steps |
> | monotonicity | **complete across ±1.4 rad** | non-monotonic zones |
> | quantization | ~1° | ~13° sectors claimed |
> | hysteresis | 0.015 | 0.051 |
> | evidence | 4 cross-checked sweeps | **1 run** |
>
> The 08-05 run was taken **immediately after discovering and "fixing" an
> SDK/daemon version skew** (below) — the same run's first pass measured
> 0.015/rad of pure garbage. That a version change moved the gain 0.015 → 0.19
> shows the skew was *a* problem; it does not show it was the *only* problem,
> and 0.19 is still 3× below the established value. **The most probable reading
> is that the 08-05 sweep is still instrument-compromised**, not that the chip
> changed. This repo has already written a phantom DoA pathology into three
> docs before vendor documentation refuted it (see the retraction above) — the
> invariant that produced is exactly why this is flagged rather than published.
>
> **Nothing downstream may rely on the staircase until it is reconciled.**
> Reconciliation = re-sweep on a version-verified stack, at ≥2 source
> geometries, and either reproduce 0.57/R²≈0.998 (→ delete this section as an
> instrument artifact) or reproduce the staircase (→ then investigate what
> changed on the robot between 07-16 and 08-05: firmware, shell, mounting).
> The data below is retained so the re-sweep has a baseline to compare against.

Pooled per-pose medians (both passes, 10 samples/pose), matched versions:

| true offset (body yaw ψ) | reading (az, deg-equivalent) |
|---|---|
| 0–30° | roughly proportional, **gain ≈ 0.19/rad** (geometry predicts 0.637) |
| 30–60° | **plateau at ±12–14°** (az ≈ ±0.13–0.14) regardless of true offset |
| beyond ~65° (left side only) | **second step to −26°** (az −0.289) |

Fitted gains: central (|ψ|≤0.5 rad) +0.195/rad, left tail +0.214, right tail
+0.171, full-range +0.180. Hysteresis is 0.051 — repeatable *within this run*,
which is consistent with a stable sensor shape AND with a stable instrument
fault; it does not discriminate between them.

Three observations, **provisional pending reconciliation**:

1. **Discrete preferred values.** The readings snap to the same exact values
   over and over (−0.289, −0.144, +0.133 az ≈ −26°, −13°, +12°) — consistent
   with the XVF3800 steering a limited set of internal beam sectors (~13°
   pitch) rather than reporting continuous bearing. Angular resolution is
   therefore ~13° at best outside the central zone.
2. **Left/right asymmetry.** The left side reaches −0.289; the right saturates
   at +0.13–0.17 — about 2× weaker. Sensor vs room/source geometry is not yet
   separated (see next steps).
3. **Zero offset.** True center reads az ≈ −0.03; the zero crossing sits near
   ψ +0.1–0.2 rad (~6–11°). Mounting, shell, or source placement.

### Implications for the orient stack — IF the curve survives reconciliation

Stated conditionally on purpose. **If the 08-05 sweep turns out to be an
instrument artifact, none of this applies and the 2026-07-16 implications stand
unchanged.**

- **Direction would remain safe** either way. Sign is correct outside the
  small-offset noise band under *both* characterizations — which is all the
  servo loop, H2-style direction choice, and the Exp 49 direction results need.
  No direction claim is at risk under either reading.
- **Magnitude selection would be impaired.** The big-step boundary (|az| ≈ 0.33,
  derived from the 0.55–0.57 gain) sits above the 08-05 right-side saturation
  ceiling, so `_big` turns would rarely trigger from readings alone. **Under the
  07-16 curve this problem does not exist** — which is precisely why the
  reconciliation matters before any policy work.
- **Do NOT retune the YAML magnitudes.** Load-bearing for the Exp 45b/45c
  boundary, and sourced here from a single contested run in one room.
- **Do NOT build a measured-staircase sim variant yet.** Encoding a contested
  curve into `SimulatedDoAScenario` would propagate a possible instrument fault
  into every downstream sim result — the phantom-pathology failure mode, one
  layer deeper.

### Next steps (reconciliation first)

1. **Re-sweep on a version-verified stack** (`curl /api/daemon/status` vs
   `importlib.metadata.version("reachy_mini")` recorded in the run), at **≥2
   source geometries**, against the 07-16 protocol. This is the discriminator:
   reproduce 0.57 / R²≈0.998 → the 08-05 curve was an instrument artifact and
   this section gets deleted; reproduce the staircase at both geometries → it is
   real and the next question is what changed on the robot since 07-16.
2. Only if the staircase reproduces: `ear_map.py` for a finer directional map,
   then decide about the sim variant and any YAML retune (which carries a
   pre-registered re-run of the 45b/45c boundary).

---

## Sources

- Reachy Mini media stack — https://huggingface.co/blog/pollen-robotics/reachy-mini-media-stack
- Advanced media controls (the verbatim 2-channel / DoA documentation) — https://huggingface.co/docs/reachy_mini/en/platforms/reachy_mini/media_advanced_controls
- Wireless hardware datasheet — https://huggingface.co/docs/reachy_mini/en/platforms/reachy_mini/hardware
- Lite hardware datasheet — https://wiki.seeedstudio.com/reachymini_platforms_reachy_mini_lite_hardware/
- Python SDK — https://huggingface.co/docs/reachy_mini/en/SDK/python-sdk
- reSpeaker XVF3800 host control — https://github.com/respeaker/reSpeaker_XVF3800_USB_4MIC_ARRAY
- XMOS XVF3800 datasheet — https://www.xmos.com/documentation/XM-014888-PC/html/modules/fwk_xvf/doc/datasheet/02_overview.html
