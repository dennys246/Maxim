# Phase 1 — Reachy orient-to-center, live (hardware-in-loop runbook)

**Status:** active device bring-up, started 2026-07-09. Requires the Phase 0 body wiring
([`bodies/reachy_mini.yaml`](../../src/maxim/_data/components/bodies/reachy_mini.yaml) azimuth sensor
+ orient affordances — PR #387).
**Scope:** take the Phase 0 orient-to-center backbone live on a physical Reachy Mini, driven by the
onboard DoA. Sim-validated in Phase 0a; this is the hardware-in-loop step.

## Relation to the other Reachy plans (different layer — not a duplicate)

- [`ShredderSegmenter/docs/plans/archive/maxim_mvp_plan.md`](../../../ShredderSegmenter/docs/plans/archive/maxim_mvp_plan.md)
  + `maxim_integration.md` are the **camera-streaming / product-registration** layer: Reachy camera →
  RTSP/MediaMTX → ShredderSegmenter site-agent records video (via the existing
  [`scripts/rtsp_bridge.py`](../../scripts/rtsp_bridge.py)). Zero DoA / orienting / NAc content.
- **This runbook is the orient / motor-control + learning layer**: DoA → head turn → NAc learns to
  point at the target. They **combine later** (Maxim orients the head → better footage → Shredder
  segments it) but bring up **independently**.
- **This loop needs ONLY the Reachy SDK** — no MediaMTX, no Shredder, no RTSP. That makes it the
  simpler first hardware bring-up.

## Iteration model

Device-in-the-loop: the operator runs each step's script on the Reachy; the agent reads the JSONL
(`MAXIM_LOG_FILE=/tmp/orient.jsonl`) and iterates. **Verify each hardware primitive before stacking
the next** — do NOT stack 4 unverifiable layers (the "cradle cascade" lesson).

## SDK surface (verified importable; calls confirmed against the installed `reachy_mini`)

- `mini = ReachyMini()` → connect (blocks/TimeoutError if the robot/daemon is down).
- `mini.wake_up()` / `mini.enable_motors()` — enable + home.
- `mini.start_recording()` — start the media stream so DoA produces values.
- `mini.media.get_DoA()` → `(doa_radians, is_speech_detected)` or `None`.
  - `audio_localization.doa_to_azimuth(doa_radians)` → azimuth ∈ [-1,1] (0=front, ±1=side).
  - `AzimuthDoASource` already wraps this as a non-blocking `PerceptSource`.
- `create_head_pose(yaw=<deg>, degrees=True)` → 4×4 matrix; `mini.goto_target(head=pose, duration=…)`
  (min-jerk) for a discrete step; `mini.get_current_head_pose()` to read current yaw.

## Steps (each gates the next)

### Step 1 — hardware smoke test  ([`scripts/orient_backbone/live_1_smoke.py`](../../scripts/orient_backbone/live_1_smoke.py))
Verify the three primitives **separately**: (a) connect + wake; (b) read `get_DoA()` for ~10 s while
the operator makes sounds left/right — print `(doa_radians, is_speech, azimuth)`; (c) `goto_target`
yaw +20°, −20°, recenter. **STOP-if:** DoA never returns / azimuth doesn't track L↔R / head doesn't
move. **Success:** azimuth tracks the sound side; head visibly turns.

### Step 2 — reactive orient, NO learning  (`live_2_reactive.py`)
Loop: read DoA (gated on `is_speech_detected`) → azimuth → if `|az| > comfort_band`, `goto_target`
one discrete step toward center. **Primary purpose: calibrate the coordinate-frame sign** — confirm a
turn *reduces* `|azimuth|`; **flip the step sign if not** (the "shared coordinate frame" open question).
**Success:** head turns toward and holds on a sustained sound.

### Step 3 — learning orient loop  (`live_3_learn.py`)
Load the real `bodies/reachy_mini`; each tick: overwrite the `azimuth` sensor from DoA (world
re-measurement is free on hardware); `state = az_bin`; substrate-primary `recommend_action` over
`turn_left`/`turn_right`; dispatch via `goto_target`; **`potential_diff` credit** = `|az_before| −
|az_after|` (next DoA read) → `update_cluster_reward`; persist NAc (`dump`/`load`) for cross-session.
This is the Phase 0a loop with sim re-measurement swapped for live DoA. **The one genuinely-new
production piece: the `potential_diff` credit as a post-affordance hook** (in-loop for now; the
executor-dispatch integration is a follow-up). **Success:** orient directedness rises across the
session; a second session (loaded NAc) starts already directed.

## Calibration unknowns (resolve empirically on-device)

- **Coordinate-frame sign** — `doa_to_azimuth` (0=left/π/2=front/π=right) vs `head_yaw` turn direction.
  Step 2 verifies a turn reduces `|az|`; flip the step sign if it grows it.
- **DoA rate + motor settle** — pick `goto_target` `duration` so the *next* DoA read reflects the
  completed turn (else `potential_diff` credits a stale re-measure).
- **Transient sounds** — gate on `is_speech_detected` (never fabricate a direction). Learning needs a
  **sustained or repeating** source.
- **Front/back ambiguity** (linear array) — az≈0 for a source directly ahead *and* behind; keep the
  source in front for now (vision resolves this in Phase 3).

## After Step 3
→ Phase 2 (visual `PerceptSource` on the same backbone, after the P1 vision-encoder check) →
Phase 3 (audio+visual fusion). See [`audiovisual_orienting.md`](audiovisual_orienting.md).
