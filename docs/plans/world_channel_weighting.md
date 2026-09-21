# World-channel weighting — CLOSED BY MEASUREMENT

> **Status: CLOSED 2026-09-20 — four-lens design review, all four DO-NOT-BUILD.** Confounding,
> bio-faithful, wiring and environment lenses each re-measured the numbers independently rather than
> arguing them; every §1–§5 figure below reproduced exactly. The *residual* idea this note was
> written to propose — a frozen per-sensor discriminative weighting — is dead, and dead by
> measurement, not by opinion.
>
> **Provenance gap, stated:** unlike `setpoint-neutral`, the four lens reports are **not** preserved
> verbatim under `docs/experiments/rationale/`. Every load-bearing finding and number below was
> re-verified in the main session against the committed records before being folded, but the reports'
> own reasoning chains live only in the session transcript. If this note is ever cited as evidence
> rather than as a decision record, close that gap first.
>
> The note is kept because the **numbers** are worth having and because two of the ideas it kills
> (a dynamic threshold; a homeodynamic/entropy set-point) are the two things anyone looking at the
> 0.85 threshold proposes first. This is now their tombstone.
>
> **It changed nothing.** `encoder_pattern_threshold = 0.85` stays. Exp 62 rung A was mid-campaign
> throughout.

## What was asked

The threshold is a fixed scalar applied to a cosine whose scale depends on how many sensors carry
mass. Does that hold up as the sensor roster grows?

## The measurements (all independently reproduced by ≥2 lenses)

Replays of `docs/experiments/data/exp62_cross_pool_replay.py` against the committed Exp 60 gate-(ii)
record, through the shipped `similarity/encoder.py::_stable_basis` and the shipped gain law
`w = (2|v − 0.5|)³`. The replay reproduces that record's live cosine (0.7874) before anything else
runs, which is the guard that it reads the real thing.

### 1. One sensor does the discriminating

`minecraft_player` declares **17** sensors. At the submerged read, **7** carry nonzero mass and **10**
are exactly zero. Three carry 90.2%: `is_in_water` 32.6%, `light_level` 32.6%, `time_of_day` 25.1%.
`distance_from_spawn` 5.3%, `y_altitude` 3.0%, `oxygen` 1.4%, `speed` 1.3e-7.

But the record's own fields are sharper than the mass table: `movers` is `["is_in_water", "oxygen"]`,
`live_contributors` is `["is_in_water"]`, and `oxygen` is tagged `moved_but_silenced`. **One sensor
separates shore from water.** The other two loud sensors are constants.

The ten silent sensors are the gain law working as designed. It is a *static, memoryless*
nonlinearity of the current reading — state-dependent, **not adaptive**: it has no memory of any
distribution. It silences the *uninformative-right-now*; it does nothing about the
*informative-about-something-else*. (Earlier drafts of this note called it "already dynamic in the
sense that matters". That was wrong and the bio-faithful lens struck it.)

### 2. The margin is one coin-flip wide

Shore-vs-submerged must stay **below** 0.85 for Exp 60's gate (ii) to separate. Today: **0.7874**,
headroom **0.0626**.

Adding one extra full-weight sensor that reads identically in both: over 200 arbitrary sensor names,
**mean 0.8499, sd 0.0082, range [0.8277, 0.8665] — 105/200 (52%) exceed 0.85.** The honest claim is
"a coin flip on the margin", not a point estimate. The dilution curve likewise is a distribution:
N=3 → 0.9063 [0.8912, 0.9158]; N=10 → 0.9587 [0.9527, 0.9656]; N=40 → 0.9880 [0.9843, 0.9902].

A sensor at its **midpoint** costs nothing. Only an off-midpoint, *shared* sensor costs. (An earlier
draft illustrated this with "a microphone in a silent room" — which is a sensor *at rest*, gain 0,
cost zero. The example refuted the point it was making. A camera facing a blank wall survives.)

### 3. The two no-rest constants own 91% of the potential headroom

`lint_body_rest_neutral.py` prints four declared no-rest sensors; two are on this body —
`light_level` and `time_of_day`, `rest: null` by design (light is place/time dependent; time_of_day
is cyclic and wraps, so neither has a meaningful midpoint).

| shore vs submerged | cosine | headroom |
|---|---|---|
| today | 0.7874 | 0.0626 |
| drop `light_level` | 0.5832 | |
| drop `time_of_day` | 0.7410 | |
| **drop both** | **0.1361** | **0.7139** |

L11 named this shape already, on another sensor: *"A declared `rest` the world never rests at is a
constant, not a neutral"* — after the `saturation` range re-centre moved A4 separation 0.0566 → 0.0881
**with no mechanism change**.

## Why the residual idea is dead

### D1. Its effect FLIPS SIGN between the two committed apparatus

The same ablation, the same encoder, a different situation set:

| pair | full | silence light+time | |
|---|---|---|---|
| Exp 60 shore/submerged | 0.7874 | **0.1361** | better |
| Exp 62 pool-2 shore/sub | 0.7875 | 0.1310 | better |
| **Exp 58 safe/dark** | 0.9766 | **0.9911** | **worse — wrong way** |

Removing a shared full-weight constant is a **contrast amplifier**, not a discriminator: it
renormalizes onto whatever the residual is. Where the residual is a neutral→extreme swing
(`is_in_water`) cosine collapses; where it is a same-side excursion (corollary 7) cosine **rises**.
A weighting fitted on the constants has **no predictable sign** on a new set. The "~11×" is a
property of the Exp 60/62 pair, not of the ablation.

This was already committed and this note failed to cite it:
`docs/experiments/data/l11_slice2_cosine_check.py::with_rangefix` runs exactly this ablation, and
`docs/wiring/cosine-separation-is-directional.md` corollary 2 records *"Re-massing (dropping the
constant sensors) also failed (0.991)"*. The set-point review measured the same lever under the name
**weight-only** and rejected it.

### D2. There is no hold-out. Every committed world-vector set holds both sensors constant

Exp 58, Exp 60, both Exp 62 pool-2 records — and the only open-world trace,
`l11_world_trace_2026-09-04.jsonl`: **`light_level` = 0.0 in 1193/1193 snapshots; `time_of_day` =
0.541667 in 1193/1193.** "Evaluate on a set it was not fitted to" is unsatisfiable: every available
hold-out carries the identical confound and will confirm the weighting. Not a risk to be managed —
structurally a fit.

### D3. The forward premise is architecturally unreachable

§2's curve models extra sensors **in the world channel**. The shipped seam forbids it:
`agent_loop::_SUBSTRATE_CHANNELS` encodes one call per channel, *"NEVER merged"* — and its comment
records that the pre-seam merge is what put the embodied orient sim at chance. That bug *is* this
curve; it was fixed by **splitting channels**, not by weighting.
`embodiment/sensory_streams.py::DECLARABLE_MODALITY_TAGS` is `{audio, world}`; the perception fabric
assigns vision → `vision`, audio → `audio`, speech → `text` — **none is `world`** — and vision/audio
content bypass `ModalityChannel` entirely (they emit embeddings, not named scalars), so they never
reach `_sensor_embed`. The fabric is DEFERRED on a physical trigger, and 1.4 adds *affordances*, not
sensors; the one proposed sensor (`pressure`) was refused in E1.

The within-`world` roster growth is real (17 today, against L11's ~12 trigger) — but it grows one
Minecraft sensor at a time.

### D4. Two declarations reach the identical number, with no mechanism

| change | shore↔submerged | mechanism |
|---|---|---|
| today | 0.7874 | — |
| `time set noon` | 0.7408 | none — one RCON word |
| `light_level` range `[-15, 15]`, `rest: 0` | 0.5832 | one YAML line |
| light re-ranged **+** `time_of_day` out of the gained channel | **0.1361** | two YAML lines |
| this note's proposed weight vector | **0.1361** | fitted, frozen, fingerprinted |

The bottom two are the same number. For a **constant** sensor a range declaration *is* the weight —
`w = (2|v−0.5|)³` with `v = (c−lo)/(hi−lo)` reaches any `w ∈ [0,1]`, including exactly 0. Precedent
in this very body: `hostile_count: range [-32, 32]  # rest (0 hostiles) = midpoint`.

### D5. Divergent fitted weights would break transfer SILENTLY, in the shape of Exp 61's own null

`hivemind/merge.py::ec_merge` skips on geometry-tag mismatch; the donor node is inserted under its
own id and the receiver's live encodes are masked away from it. The bundle ingests cleanly, with no
error and no warning — and the receiver does nothing. That is byte-for-byte Exp 61's **dangling**
negative control. A per-apparatus fit makes the experiment's null and its failure mode the same
observation.

### D6. The re-stale bill, against a problem that is not live

Three EARNED rows on live rig time — **Exp 56** (1.2 headline), **Exp 60**, **Exp 61** (1.3 headline
pair, inheriting every Exp 60 trigger) — plus an L11 retirement-grade re-measure (`_sensor_embed` is
"the encoding equation"), plus **Exp 62 rung A mid-campaign**, whose NODE gate *is* live cluster
identity.

## Corrections to this note's own earlier draft

- **"the Exp 60 dark=danger line" does not exist.** dark=danger was **Exp 58**, which **died at the
  instrument** — and died *because* light was constant (*"the exact property dark/safe lacked (both
  underground, light=0)"*). Exp 60 is drowning-avoidance on `is_in_water`/`oxygen`, and its prereg
  lists `light_level` as "recorded, not gated". **No EARNED result rests on `light_level` carrying
  mass.** Caught by three lenses independently. This cuts both ways: it removes the note's stated
  strongest objection, and replaces it with D1, which is harder.
- **`light_level` is a documented-unreliable sensor in this world.** `setup_world::_classroom`:
  *"spatially patchy, inconsistent run-to-run, and skylight-contaminated underground… cannot carry
  the contingency."* Exp 58's classroom was redesigned around it. Fitting a coefficient to it is
  fitting to noise.
- **"at no cost to the rung" was measured on one of three cosines.** Cross-pool *submerged* 0.9992 →
  0.9980 ✓; cross-pool **shore 0.9985 → 0.8750**, headroom 0.1485 → **0.0250**, a 6× collapse. The
  NODE gate reads shore-vs-water distinctness and shore fear for specificity; both go fragile.
- **§5 (the stale-light incident) belonged in the confound section, not the supporting list.** Under
  the proposed weighting the stale and reconnect records read 0.997973 and 0.997986 —
  indistinguishable. **The 2026-09-20 bug that would have reversed the experiment's conclusion would
  have been invisible.** The fix that shipped is provenance-side and depends on the sensor being loud.

## What came out of it worth keeping

Three real defects, none of them this note's proposal:

1. **The encoder leg of every frozen fingerprint is weaker than the experiments assume.**
   `WaterTrial.live_fingerprint` reads `SensorEncoderConfig().pattern_threshold` — a *freshly
   default-constructed* config, not the live encoder's — so it is a tautology against a source
   default that git already guards. It omits `gain_exponent` and `gain_modalities` entirely, and its
   `sensor_ranges` leg covers 3 of 17 sensors, **excluding `light_level` and `time_of_day`**. A range
   or roster change to either passes `check_fingerprint` silently, today, mid-campaign.

   **Resolved 2026-09-20 (issue #783).** `check_fingerprint` now also holds an *encoding identity*
   — every `SensorEncoderConfig` field, enumerated, plus the body's full declared world roster — to
   `water_trial.APPARATUS_ENCODING`, for BOTH encoders a trial depends on: the harness encoder it
   books and reads through, and the probe loop's (which `run_agent_loop` default-constructs). One
   correction to the finding as worded above: the fresh-default read was not quite a tautology,
   since it was compared to a frozen *literal* and so did catch a changed source default. What it
   missed was a non-default config on the encoder actually in use, and everything past
   `pattern_threshold`. The identity is kept out of the experiments' FROZEN blocks because R3's
   gauntlet pins `sha256(FROZEN60)`, and widening a closed block would unfreeze a closed record.
2. **`place_code` is INVERTED by the A4 gain** — and place coding is the one bio-faithful answer for
   a rest-less cyclic variable like `time_of_day` (head-direction/time cells). Measured through the
   shipped `similarity/place_code.py` and the shipped gain law: at value 0.0 the two *nearly-off*
   flank cells (activation 0.0439) draw weight **0.759** while the peak draws 1.0; at value 0.15 or
   0.45 the two *most informative* cells (activation 0.4578) draw **0.0006** each and the whole
   population collapses to total mass **0.0012** — a near-zero vector, which `encode_sensors` returns
   as `None`. A population code's activation already *is* the salience; layering distance-from-
   midpoint gain on top double-counts it with the sign reversed. L11 recorded the symptom once (*"a
   gained RAW azimuth zero-vectors the CENTERED reading"*); this is the general law. Azimuth is safe
   today only because audio is ungained.
3. **"Habituation already lives downstream" names a layer that does not exist for sensor channels.**
   `tools/novelty.py` is keyed on vision track ids; `SalienceMap` is a camera-frame grid; neither is
   reached from `_SUBSTRATE_CHANNELS` or the world path. Any future "put it downstream" proposal is a
   **new build in a place that only looks occupied**, and must be priced as one.

## Closed — do not re-propose

| candidate | why it is dead |
|---|---|
| sensor-count-scaled threshold `1 − k/N` | measured 0.70–0.84, degrading with N (L11 bake-off; an earlier 100% row was superseded) |
| more embedding dimensions | dilution is an averaging problem, not a capacity one |
| sparse / hashed / randomised bases | identical to the plain sum |
| distributional moments | permutation-invariant — cannot say *which* sensor moved |
| static set-point-aware neutral | numerically identical to the range declaration already used |
| adaptive set-point (habituation in the encoder) | 2% jitter separates 20/20; rest embeds to the zero vector; strands frozen prototypes |
| **frozen per-sensor discriminative weighting** | **D1–D6 above** |

The open question was never the threshold. It is **which constants belong in the world channel** —
a roster decision, already named verbatim in `exp62_pressure_interoception_prereg.md` §Rung B, and
answerable by declaration when a rung needs it.

## Related

- `docs/limits/l11_sensor_dilution.md` · `docs/plans/deferred/setpoint_aware_neutral.md`
- `docs/wiring/cosine-separation-is-directional.md` (corollaries 2 and 7)
- `docs/experiments/exp62_pressure_interoception_prereg.md` §Rung B
- `src/maxim/similarity/encoder.py::_sensor_embed` (plan decision D1)
