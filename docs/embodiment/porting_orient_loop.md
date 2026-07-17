# Porting the orient-to-center learning loop to a new robot

**Status:** contract document. Rewritten 2026-07-16 after the loop reached **magnitude
1.00 on hardware** ([Exp 45](../experiments/45_reachy_orient_live.md) /
[45b](../experiments/45b_orient_magnitude.md) / [45c](../experiments/45c_flip_bins.md)).
Written at N=1 robot (Reachy Mini), so per the second-consumer test the **code
abstraction stays deferred until robot #2** — abstracting from one example bakes that
example's assumptions into the interface. What this document pins instead: which layers
are already robot-agnostic, exactly what a new robot supplies, **which design constants
must be DERIVED from that robot's own measurements**, and the failure modes that cost us
a day so they cost you an hour.

## What is already robot-agnostic (do not duplicate these)

| Layer | Where | Why it ports |
|---|---|---|
| Learning core | `NAc.recommend_action` / `update_cluster_reward` / `dump`+`load` ([decisions/nac.py](../../src/maxim/decisions/nac.py)) | Keyed on (agent_id, state-bin string, tool signature). Knows nothing about bodies. |
| Credit rule | `potential_diff = \|az_before\| − \|az_after\|` (Phase 0b verdict; hardware-validated through 45c) | Defined on the normalized azimuth, not on any sensor. **No effort term** — an effort-cost variant was proposed, sim-tested, and refuted. |
| Azimuth normalization contract | azimuth ∈ [-1, 1], 0 = front/centered, ±1 = ±90° ([embodiment/audio_localization.py](../../src/maxim/embodiment/audio_localization.py)) | The single seam every bearing front-end must produce. |
| Percept source | `AzimuthDoASource(doa_reader=...)` — the reader is an **injected callable** | Hardware enters through one function; swap the reader, keep everything else. |
| Body declaration pattern | `azimuth` sensor + centeredness `HomeostaticDriveSpec` (set_point 0, **drift_rate 0** — world-set sensors must not auto-return) + discrete orient affordances ([bodies/reachy_mini.yaml](../../src/maxim/_data/components/bodies/reachy_mini.yaml)) | Copy the pattern; cognition/substrate code unchanged (the capability-driven principle, [README](README.md)). |
| Evaluation methodology | Frozen-policy probes (epsilon-free learning curve), cross-session probe-0, the `--perturb` apparatus + startup sign self-check ([live_3_learn.py](../../scripts/orient_backbone/live_3_learn.py)) | Defined on bins and actions, not hardware. |
| Connection factory | `maxim.hardware` (`RobotController` ABC, `RobotRegistry`, `~/.maxim/robots.yaml`) | New robots register a controller; host lives in robots.yaml, not code. |

**Partially agnostic — read carefully:** `az_bin` is a robot-agnostic *function*, but its
**near/far boundary is NOT a constant** — it is derived from your robot's measured gain
(next section but one). The legacy 0.5 default is arbitrary and was the thing capping
magnitude learning at 0.75.

## What a new robot must supply (the contract — four items)

1. **A bearing source**: a zero-argument callable returning `(bearing, gated)` — the
   current horizontal bearing of the attended source and a validity gate — or `None`.
   Normalize to the azimuth contract before it touches anything else. The gate is
   load-bearing: **never fabricate a direction** (Reachy uses the chip's
   `is_speech_detected`; a custom front-end would use a VAD or energy threshold). Where
   localization runs is your business — see
   [perception pipeline placement](../plans/perception_pipeline_placement.md).
2. **A yaw primitive that rotates the SENSING frame**: "turn the bearing sensor by
   ±delta about the vertical axis." Which joint implements it is your choice (Reachy:
   the *base*, because the head clamps at ±15-18°; a biped: torso or step-turn). Two
   requirements, and the first one is where we lost a day: the motion must **actually
   rotate the sensor** (verify it — §Calibration step 1), and it must settle before the
   next read.
3. **A body YAML** declaring the azimuth sensor + centeredness drive + `orient`
   affordances. The affordance `self_effect` values are the **action magnitudes in your
   robot's own yaw units** (Reachy: ±0.3 / ±0.9 rad) — they are a *declaration*, not a
   calibration output. What IS calibrated is the gain that converts them to azimuth.
4. **Calibration answers** — measured on-device, recorded on your platform page, never
   hardcoded into shared code.

## The design constants are DERIVED — this is the part people get wrong

**The action set and the state bins are duals. You cannot design them independently.**

Your action magnitudes `{d₁ < d₂ < … < dₙ}` produce azimuth shifts `Sᵢ = dᵢ × gain`. A
correct-direction step of shift `S` takes `|az| → |az − S|`, so its relief is
`az − |az − S|`. Therefore **step A beats step B exactly when `|az − S_A| < |az − S_B|`**
— when your current error is *nearer A's shift*. So:

> **The optimal magnitude policy is nearest-neighbour quantization of |error| onto your
> available shifts.** Your action set defines a Voronoi partition of the error axis. Your
> state bins must BE that partition.

The boundaries are the midpoints, derived from *your* measured gain:

```
boundaryᵢ = gain × (dᵢ + dᵢ₊₁) / 2          # N magnitudes → N−1 boundaries → N bins/side
Reachy:     0.546 × (0.3 + 0.9) / 2 = 0.328
```

**A bin that straddles a boundary contains two opposite correct answers.** It receives
consistently contradictory evidence and settles on whichever its experience happened to
favour — not noise, a *systematic* conflict. This is not theoretical:

| | near/far boundary | magnitude learned |
|---|---|---|
| Exp 45b | 0.5 (arbitrary, straddles 0.328) | **0.75** — `near_right` drew placements above 0.328 and correctly learned big; `near_left` drew below and correctly learned normal. Same bin, opposite lessons, **both right**. |
| Exp 45c | 0.328 (derived) | **1.00** — every bin decisive; `near_right` learned the big step is *harmful* there (−0.570) |
| sim A/B (6 seeds) | 0.5 vs derived | **0.375 vs 0.92** |

**"Use finer bins" was the wrong instinct** (it was mine). The problem is not resolution
— it is **alignment**. A perfectly fine-grained grid that still straddles the boundary
fails the same way; a two-bin split at the right place succeeds.

**Consequences for your port:**
- Measure gain **first**, derive boundaries, *then* set bins. Never choose bins by feel.
- Changing action magnitudes **moves the boundaries** — it re-opens the experiment.
- N magnitudes want N bins per side. Bins and actions are designed together or not at all.
- `decision_boundary()` / `placement_ranges()` in
  [live_common.py](../../scripts/orient_backbone/live_common.py) compute this from the
  action set + measured gain; `--flip-bins` uses them.

## Per-robot calibration protocol

Robot-agnostic procedure; every new body re-answers it. Order matters — each step's
answer is garbage if the previous one failed.

1. **VERIFY THE SENSING FRAME ACTUALLY ROTATES.** Command your yaw axis and read the
   *sensor's own frame back from the robot* — not your commanded belief.
   `d(sensor_frame)/d(commanded)` must be **≈ +1.0**. ([`yaw_verify.py`](../../scripts/orient_backbone/yaw_verify.py)
   is the Reachy instance; the portable part is *read the frame back, never assume it*.)
   Why first: we skipped it, and the vendor's daemon was counter-rotating the
   head-mounted mic array to hold its world orientation — the mics rotated **36%** of
   what we commanded. **Reading a nearly-stationary sensor while believing it moved
   perfectly mimics a lagging, compressed, drifting sensor.** We "measured" a tracking
   estimator that does not exist, wrote it into three docs, and burned six hypotheses.
   **And read the vendor's frame-semantics docs before reverse-engineering their
   kinematics** — Pollen had it in `AGENTS.md` the entire time.
2. **Smoke the primitives separately** (connect+enable, bearing tracks a real L/R source,
   yaw visibly moves). Never stack the loop on an unverified primitive (cradle-cascade).
3. **Sign**: does a turn toward the sensed side *reduce* |azimuth|? Step-2 reactive loop,
   or the Step-3 `--perturb` startup self-check (two-sided; aborts before learning on a
   wrong sign). Record the flag.
4. **Response curve → gain.** Sweep the axis, both directions, several gated reads per
   pose ([`doa_sweep.py`](../../scripts/orient_backbone/doa_sweep.py)). You want gain,
   linearity, and the usable range. Reachy post-fix: **0.57 az/rad, R²=0.998 over ±80°,
   0.23 s convergence**. Expect a linear array to have an **endfire degeneracy** near
   ~90° off-axis — but *measure* it rather than assuming (ours turned out not to exist;
   the "bimodal zone" was the head bug too).
5. **Derive the boundaries** from gain + your action magnitudes (previous section). Set
   bins and placement ranges to match.
6. **Settle timing**: the post-turn read must reflect the *completed* turn or
   `potential_diff` credits a stale re-measure (Reachy: 0.6 s + 0.5 s; the DoA itself
   settles in 0.23 s).
7. **Step size vs noise**: the per-action |azimuth| change must clear reading noise
   (≥3× is a reasonable rule). Note the tension with §Design constants: narrow bins have
   *smaller* reliefs and therefore worse SNR — that is the real cost of fine binning.
8. **Axis limits + ambiguities**: yaw clamp; array geometry. A linear/coplanar array has
   front/back ambiguity (keep sources in front); a non-coplanar array can additionally
   declare an `elevation` sensor + pitch affordances with **no substrate-code change**.

## Worked example: Reachy Mini's answers

| Contract item | Reachy answer |
|---|---|
| Bearing source | `GET /api/state/doa` → `doa_to_azimuth`; gate = `speech_detected` |
| Yaw primitive | `goto_target(body_yaw=<abs rad>, head=create_head_pose(yaw=degrees(body_yaw)))` — **the explicit head matrix is mandatory**: `head=None` counter-rotates the head and the mics do not turn ([CLAUDE.md](../../CLAUDE.md) invariant) |
| Body YAML | [bodies/reachy_mini.yaml](../../src/maxim/_data/components/bodies/reachy_mini.yaml): orient magnitudes ±0.3 (normal) / ±0.9 (big) rad |
| Sign | default (turn_left = +yaw → azimuth +); Step 2, 16/16 valid |
| **Gain** | **0.57 az/rad** (0.546/0.548 central, 0.571/0.578 full-fit, 0.562 settle, 0.58 sign-check — four independent measurements within 0.03), R²=0.998, 0.23 s |
| **Derived boundary** | **0.328** → near [0.1, 0.33], far [0.33, ~0.87]; placements near (0.16, 0.27), far (0.39, 0.80) |
| Settle / limits | 0.6 s + 0.5 s; ±1.4 rad soft clamp; front/back-ambiguous linear array; **no endfire degeneracy observed** post-fix (the old \|az\| ≤ 0.65 cap was chasing an artifact) |
| Result | direction 1.00, **magnitude 1.00**, greedy 0.286 → 1.000 |

## Notes for an Atlas-class (biped) port

- **No on-board DoA**: supply a localization front-end (ROS audio stack, external array).
  The azimuth contract is the seam; everything downstream is untouched.
- **Turning is expensive**: step-turns cost seconds and watts. The discrete-step design
  still holds, but trial economics change — larger magnitudes, longer settle, and the
  `--perturb` apparatus becomes *more* valuable (balanced bins per unit of actuation).
  Note your larger magnitudes **push the derived boundaries outward** — recompute, don't
  copy Reachy's 0.328.
- **The sensing frame must turn**: if the mics are torso-mounted and the robot turns its
  head, the loop never closes. Pick the primitive that rotates the *array*, and verify it
  (step 1) — a humanoid with a compensating neck is exactly the Reachy trap at scale.

## Failure modes: how this went wrong, generalized

Six hypotheses died before the real bug surfaced. The pattern is worth more than the
answer:

1. **A wrong actuation assumption is indistinguishable from a broken sensor** — and it
   generates an endless supply of plausible sensor theories (we produced: tracking
   estimator, settle lag, backlash, motor under-travel, speech-density dependence,
   increment-size limits; each falsifiable, each falsified). **Verify the actuation
   before theorizing about the sensor.**
2. **Test against ground truth, not self-consistency.** Two of our tests asked "is the
   reading *stable*?" when the question was "is it *right*?" A frozen-but-wrong estimate
   scores 1.00 on self-consistency. One of them was also underpowered — it looked for
   drift using 0.1 rad steps whose expected drift sat *below the sensor's quantization*.
3. **Silent frame mismatches between components.** The learner binned at 0.5 while the
   apparatus and metric used 0.33 — no error, just a quietly wrong experiment. Found by
   inspecting **what each bin actually saw**, not what the config said. (Same shape as
   the head bug: two components disagreeing about a frame.)
4. **Your sim must model what the hardware does, not what is convenient.** Ours
   teleported the sound source in a mode where the source is physically fixed — which
   *masked* the boundary effect (0.70 vs 0.80, no separation; with honest physics, 0.375
   vs 0.92). Noise can hide a systematic error as easily as it can create one.
5. **Derive; do not reason.** A plausible-looking formula (`|Δbig|×gain/2`) was wrong;
   the derivation (`gain×(|Δbig|+|Δnormal|)/2`) was two lines and correct. The wrong one
   also *survived a review pass* because it looked right.
6. **Read the vendor's docs first.** The answer to the day's central mystery was one
   sentence in Pollen's `AGENTS.md`, published, the whole time.

## When robot #2 arrives (the deferred extraction)

The second consumer triggers the mechanical refactor — not before:

1. Extract an `OrientRig` protocol (`read_azimuth() -> (float, bool) | None`,
   `turn(delta)`, `recenter()`, `sensor_frame() -> float` — **the frame read-back is part
   of the protocol now**, it is not optional) plus the calibration params as a frozen
   config, from `live_3_learn.py`'s `LiveRig`/`DryRig`/`Apparatus`.
2. Move the learning loop + probe + apparatus + `decision_boundary`/`placement_ranges`
   into `src/maxim/embodiment/orient_loop.py` (production, tested); per-robot scripts
   become thin adapters.
3. Wire affordance dispatch through the executor/tool_bridge so orient actions flow the
   same path as every other affordance.

## Submitting learned substrate for distribution (the fleet loop)

Once your policy passes the probe, the learned substrate is a distributable artifact —
tiny (a handful of numbers), and privacy-clean by construction.

1. **Gauntlet + merge** — [`orient_merge_arm.py`](../../scripts/orient_backbone/orient_merge_arm.py)
   probes your NAc, merges with another contributor's via `hivemind.merge.nac_merge`
   (mean-merged `cluster_reward_bias`, provenance-tagged), probes the merged result, and
   **refuses to bundle unless merged ≥ threshold AND ≥ both inputs**. This probe is
   promotion gauntlet #1 of the Queen-tier trust topology
   ([maxim_hivemind.md](../plans/maxim_hivemind.md)) — a poisoned or flipped-calibration
   contribution fails in milliseconds, no hardware needed.
2. **Bundle** — exports via `compose_bundle` as a domain-tagged zip (`robotics-orient`),
   identity filter on. **Bundles never contain episodes** — only distilled substrate.
3. **Distribute** — a published release (the "queen-mind" zip) is the channel at current
   scale; consumers `maxim substrate inspect` / `import`, `nac_merge`, and validate with
   the same probe. Cross-session bootstrap is hardware-validated (Exp 45 arm 2: probe
   1.00 at trial 0); cross-*unit* is the designed follow-on, not yet run.
4. **Cross-robot consumption** — the policy is keyed on normalized-azimuth bins and YAML
   action names, so a different robot satisfying this contract can consume the same
   bundle: its sign/gain/axis apply at *dispatch*, not in the learned state. **Caveat
   (from 45c): the bins must MEAN the same thing.** A robot whose derived boundary
   differs from the producer's has a *different state space* under identical bin names —
   its `near_left` is not yours, and the mismatch is **silent**: every lookup between the
   two boundaries returns the wrong bin with no error.
   **This bit us locally before it could bite a stranger.** `orient_demo.py` assumed the
   legacy 0.5 boundary while replaying a policy trained at 0.33, so the robot took small
   steps where big ones had been learned — the same failure class as the head-frame bug,
   third instance in one day. The fix generalizes and is **not a flag** ("remember how
   this policy was trained" is precisely the assumption that keeps breaking): **the state
   space travels with the policy.** `live_3_learn.py` writes a `<nac>.meta.json` sidecar
   (`bin_boundary`, `band`, `gain`, `action_deltas`, `placements`, `agent_id`) on every
   save; `orient_demo.py` reads it instead of assuming; policies without one load with a
   warning that names the risk. **Substrate bundles need the same fields in their
   manifest before they can safely cross robots** — the sidecar is the prototype. Until
   then, treat 45b-era and 45c-era bundles as **different substrate generations**.
5. **agent_id convention** — `cluster_reward_bias` keys are `(agent_id, bin, tool)`; the
   orient scripts use `agent_id="reachy"`. A consumer under a different agent_id sees
   ZERO bias and the policy silently vanishes. Until a remap-at-import exists, consume
   orient bundles under the same agent_id.

## Pointers

- Umbrella plan + rigor bar: [substrate_native_orienting.md](../plans/substrate_native_orienting.md)
- Live runbook (this protocol's concrete instance): [reachy_orient_live.md](../plans/reachy_orient_live.md)
- Sensor deep-dive + measured response: [reachy_mini/audio_localization.md](reachy_mini/audio_localization.md)
- Magnitude/bins/boundary line: [orient_magnitude_learning.md](../plans/orient_magnitude_learning.md)
- Results: [Exp 45](../experiments/45_reachy_orient_live.md) · [45b](../experiments/45b_orient_magnitude.md) · [45c](../experiments/45c_flip_bins.md)
- Fleet sharing: [maxim_hivemind.md](../plans/maxim_hivemind.md) ("Trust topology")
