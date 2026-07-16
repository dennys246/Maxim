# Porting the orient-to-center learning loop to a new robot

**Status:** contract document (2026-07-16). Written at N=1 robot (Reachy Mini, Phase 1
live — [substrate_native_orienting.md](../plans/substrate_native_orienting.md) /
[reachy_orient_live.md](../plans/reachy_orient_live.md)). Per the second-consumer test
(the front-gate discipline in CLAUDE.md), the **code abstraction is deliberately
deferred until robot #2 exists** — abstracting from one example bakes that example's
assumptions into the interface. What this document does instead is pin the **contract**:
which layers are already robot-agnostic, exactly what a new robot must supply, and the
calibration protocol every new body runs. When robot #2 arrives, the refactor is
mechanical (see "When robot #2 arrives" below).

## What is already robot-agnostic (do not duplicate these)

| Layer | Where | Why it ports |
|---|---|---|
| Learning core | `NAc.recommend_action` / `update_cluster_reward` / `dump`+`load` ([decisions/nac.py](../../src/maxim/decisions/nac.py)) | Keyed on (agent_id, state-bin string, tool signature). Knows nothing about bodies. |
| Credit rule | `potential_diff = \|az_before\| − \|az_after\|` (Phase 0b verdict) | Defined on the normalized azimuth, not on any sensor. |
| State discretization | `az_bin` (5 bins: center, near/far × left/right) | Defined on normalized azimuth. |
| Azimuth normalization contract | azimuth ∈ [-1, 1], 0 = front/centered, ±1 = ±90° ([embodiment/audio_localization.py](../../src/maxim/embodiment/audio_localization.py)) | The single seam every bearing front-end must produce. |
| Percept source | `AzimuthDoASource(doa_reader=...)` — the reader is an **injected callable** | Hardware enters through one function; swap the reader, keep everything else. |
| Body declaration pattern | `azimuth` sensor + centeredness `HomeostaticDriveSpec` (set_point 0, **drift_rate 0** — world-set sensors must not auto-return) + discrete orient affordances ([bodies/reachy_mini.yaml](../../src/maxim/_data/components/bodies/reachy_mini.yaml)) | Copy the pattern into the new body YAML; cognition/substrate code unchanged (the capability-driven principle, [README](README.md)). |
| Evaluation methodology | Frozen-policy probes (epsilon-free learning curve), cross-session probe-0, the `--perturb` apparatus protocol + startup sign self-check ([scripts/orient_backbone/live_3_learn.py](../../scripts/orient_backbone/live_3_learn.py)) | Defined on bins and actions, not hardware. The apparatus needs only the same two primitives the learner needs. |
| Connection factory | `maxim.hardware` (`RobotController` ABC, `RobotRegistry`, `~/.maxim/robots.yaml`) | New robots register a controller; host/config lives in robots.yaml, not code. |

## What a new robot must supply (the contract — four items)

1. **A bearing source**: a zero-argument callable returning `(bearing, gated)` — the
   current horizontal bearing of the attended sound source and a validity gate — or
   `None` when no reading exists. Normalize to the azimuth contract above before it
   touches anything else. The gate is load-bearing: **never fabricate a direction**
   (Reachy uses the chip's `is_speech_detected`; a custom TDOA front-end would use a
   VAD or energy threshold). Where localization runs (on-chip, ROS node, external
   array) is the robot's business — see
   [perception pipeline placement](../plans/perception_pipeline_placement.md).
2. **A discrete yaw primitive**: "turn the sensing frame by ±step about the vertical
   axis." Which joint implements it is the robot's choice (Reachy: base `body_yaw`
   because the head clamps at ±15-18°; a biped: torso or step-turn). Two requirements:
   the motion must **rotate the microphone/bearing frame** (else the loop never closes),
   and it must settle before the next read (see calibration).
3. **A body YAML** declaring the azimuth sensor + centeredness drive + `orient`
   affordances (`turn_left`/`turn_right`), copied from the reachy_mini pattern. The
   affordance `self_effect` signs define the *convention*; hardware magnitude and axis
   are per-robot calibration outputs, not YAML edits.
4. **Calibration answers** (next section) — discovered empirically on-device, recorded
   in the robot's platform page, never hardcoded into shared code.

## Per-robot calibration protocol (run this on every new body)

These are the empirical unknowns the Reachy bring-up hit; every robot re-answers them.
The procedure is robot-agnostic — it is Steps 1-3 of the
[live runbook](../plans/reachy_orient_live.md) with the robot's own primitives:

1. **Smoke the three primitives separately** (connect+enable, bearing read tracks a
   real L/R source, yaw primitive visibly moves) — never stack the loop on an
   unverified primitive (the cradle-cascade lesson).
2. **Sign calibration**: does a turn toward the sensed side *reduce* |azimuth|? Run the
   Step-2 reactive loop or the Step-3 `--perturb` startup self-check (two-sided
   commanded offsets; aborts before learning on a wrong sign). Record the flag.
3. **VERIFY THE SENSOR FRAME ACTUALLY ROTATES — do this FIRST.** The most
   expensive lesson of the Reachy bring-up (a full session, six wrong hypotheses):
   we commanded body rotation and *assumed* the head-mounted mic array turned with
   it. It did not — the vendor's daemon was counter-rotating the head to hold its
   world orientation, and the array rotated 36% of what we commanded. Reading a
   nearly-stationary sensor while believing it moved **perfectly mimics a lagging,
   compressed, drifting sensor** — we "measured" a tracking estimator that does not
   exist and wrote it into three docs.
   **Procedure:** command your yaw axis and read the *sensor's own frame* back from
   the robot (not your commanded belief). `d(sensor_frame)/d(commanded)` must be
   **≈ +1.0**. Anything less means your sensor is not experiencing the motion you
   think it is, and every downstream number is contaminated.
   ([`yaw_verify.py`](../../scripts/orient_backbone/yaw_verify.py) is the Reachy
   instance; the generalizable part is *read the frame back, never assume it*.)
   **And read the vendor's docs on frame semantics before reverse-engineering their
   kinematics** — Pollen documented this verbatim in `AGENTS.md` the whole time.
4. **Response curve.** Only once step 3 passes, sweep the axis and measure gain +
   linearity + usable range ([`doa_sweep.py`](../../scripts/orient_backbone/doa_sweep.py)
   pattern: small increments, both directions, several gated reads per pose).
   Reachy post-fix: **0.562-0.58 az/rad** (geometric 0.637), converging in 0.23 s —
   fast and stable. Expect a linear array to have an **endfire degeneracy** near
   ~90° off-axis; cap placements inside the measured reliable range.
4. **Settle timing**: the post-turn read must reflect the *completed* turn, or
   potential_diff credits a stale re-measurement. Pick motion duration + settle so the
   bearing source has re-estimated (Reachy: 0.6 s + 0.5 s).
5. **Step size**: the per-action |azimuth| change must clear sensor noise (rule of
   thumb: ≥3× the measured reading noise), and cost-per-step matters — a robot whose
   turns are expensive (a biped stepping) wants fewer, larger steps and a longer settle.
6. **Axis limits + ambiguities**: yaw clamp (Reachy: ±1.4 rad soft limit in the
   scripts), and the array's geometry limits — a linear/coplanar array has front/back
   ambiguity (keep sources in front); a non-coplanar array could additionally declare
   an `elevation` sensor + pitch affordances with **no substrate-code change** (the
   multi-axis case the placement plan designed for).

## Worked example: Reachy Mini's contract answers

| Contract item | Reachy answer |
|---|---|
| Bearing source | `GET /api/state/doa` (network) → `doa_to_azimuth`; gate = `speech_detected` |
| Yaw primitive | `goto_target(body_yaw=<abs rad>, head=create_head_pose(yaw=degrees(body_yaw)))` — **the explicit head matrix is mandatory**: `head=None` counter-rotates the head and the mics do not turn |
| Body YAML | [bodies/reachy_mini.yaml](../../src/maxim/_data/components/bodies/reachy_mini.yaml) (PR #387) |
| Sign | default convention (Step 2, 16/16 valid: turn_left = +yaw → azimuth +) |
| Gain | **0.562-0.58 az/rad**, stable, 0.23 s convergence (measured *after* the head-frame fix; every earlier number was contaminated by it) |
| Settle / step / limits | 0.6 s + 0.5 s; 0.25 rad; ±1.4 rad clamp; front/back-ambiguous linear array; endfire bimodal zone → placements capped |az| ≤ 0.65 |

## Notes for an Atlas-class (biped) port

- **No on-board DoA**: supply a localization front-end (ROS audio stack or an external
  mic array). The azimuth contract is the seam — everything downstream is untouched.
- **Turning is expensive**: step-turns cost seconds and watts. The loop's discrete-step
  design still holds, but trial economics change: larger steps, longer settle, and the
  `--perturb` apparatus becomes *more* valuable (balanced bins per unit of actuation).
- **The sensing frame must turn**: if the mics are body-mounted and the robot turns its
  head only, the loop does not close. Pick the yaw primitive that rotates the array.

## When robot #2 arrives (the deferred extraction)

The second consumer triggers the mechanical refactor — not before:

1. Extract an `OrientRig` protocol (`read_azimuth() -> (float, bool) | None`,
   `turn(delta_rad)`, `recenter()`, plus the calibration params as a small frozen
   config) from `live_3_learn.py`'s `LiveRig`/`DryRig`/`Apparatus`.
2. Move the learning loop + probe + apparatus into `src/maxim/embodiment/orient_loop.py`
   (production code with tests); the per-robot scripts become thin adapters that build a
   rig and call it.
3. Wire affordance dispatch through the executor/tool_bridge (the runbook's noted
   follow-up) so the orient actions flow the same path as every other affordance.

Until then, the scripts stay deliberately robot-specific bring-up code — that is what a
device-in-loop runbook is for.

## Submitting learned substrate for distribution (the fleet loop)

Once your robot's policy passes the probe (all bins argmax-correct), the learned
substrate is a distributable artifact. The pipeline is built and dry-validated
end-to-end (bundle round-trips through the substrate CLI; a weak input was measurably
repaired by the merge); the live result is [Exp 45](../experiments/45_reachy_orient_live.md)
arm 3:

1. **Gauntlet + merge** — [`orient_merge_arm.py`](../../scripts/orient_backbone/orient_merge_arm.py)
   probes your NAc, merges it with another contributor's via `hivemind.merge.nac_merge`
   (mean-merged `cluster_reward_bias`, provenance-tagged per contributor), probes the
   merged result, and **refuses to bundle unless merged correctness ≥ threshold AND ≥
   both inputs**. This probe is promotion gauntlet #1 of the Queen-tier trust topology
   ([maxim_hivemind.md](../plans/maxim_hivemind.md) "Trust topology") — a poisoned or
   flipped-calibration contribution fails in milliseconds, no hardware needed.
2. **Bundle** — the gauntlet-passed merge exports via `compose_bundle` as a substrate
   bundle zip (domain-tagged, e.g. `robotics-orient`; identity filter on; manifest with
   contributor provenance). **Bundles never contain episodes** — only distilled
   substrate (NAc biases; EC prototypes when present) — which is what makes them
   publishable without a privacy review.
3. **Distribute** — at current scale, a published release (the "queen-mind" zip) IS the
   distribution channel; consumers verify with `maxim substrate inspect`, extract with
   `maxim substrate import`, merge with `nac_merge`, and validate with the same probe.
   Cross-session bootstrap is hardware-validated on one unit (Exp 45 arm 2: loaded
   NAc probes 1.00 at trial 0); a merged NAc probes 1.00 in-memory, and bootstrapping a
   SECOND physical unit from it is the designed cross-unit follow-on, not yet run.
4. **Contract for new robots**: because the policy is keyed on normalized-azimuth bins
   and YAML action names — not on any hardware detail — a *different* robot that
   satisfies this document's contract can consume the same bundle. Its calibration
   (sign, gain, axis) is applied at dispatch, not baked into the learned state.
5. **agent_id convention (load-bearing for consumption)**: `cluster_reward_bias` keys
   are `(agent_id, bin, tool)` triples, and the orient scripts read/write under
   `agent_id="reachy"`. A consumer probing or acting under a different agent_id sees
   ZERO bias — the policy silently vanishes. Until a remap-at-import step exists,
   orient bundles are consumed under the same `"reachy"` agent_id.

## Pointers

- Umbrella plan + rigor bar: [substrate_native_orienting.md](../plans/substrate_native_orienting.md)
- Live runbook (the calibration protocol's concrete instance): [reachy_orient_live.md](../plans/reachy_orient_live.md)
- Placement/infra: [perception_pipeline_placement.md](../plans/perception_pipeline_placement.md)
- Capability-driven principle + platform pages: [README.md](README.md)
- Fleet sharing of the learned policy: [maxim_hivemind.md](../plans/maxim_hivemind.md) ("Trust topology")
