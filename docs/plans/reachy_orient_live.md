# Phase 1 — Reachy orient-to-center, live (hardware-in-loop runbook)

**Status:** active device bring-up, started 2026-07-09. Requires the Phase 0 body wiring
([`bodies/reachy_mini.yaml`](../../src/maxim/_data/components/bodies/reachy_mini.yaml) azimuth sensor
+ orient affordances — PR #387).
**Scope:** take the Phase 0 orient-to-center backbone live on a physical Reachy Mini, driven by the
onboard DoA. Sim-validated in Phase 0a; this is the hardware-in-loop step.

## Relation to the other Reachy plans (different layer — not a duplicate)

- [`ShredderSegmenter/docs/plans/archive/maxim_mvp_plan.md`](../../../ShredderSegmenter/docs/plans/archive/maxim_mvp_plan.md)
  + `maxim_integration.md` are the **camera-streaming / product-registration** layer: Reachy camera →
  RTSP/MediaMTX → ShredderSegmenter site-agent records video. **Note (2026-07):** Maxim's own bridge
  (`tools/rtsp_bridge.py` / `scripts/rtsp_bridge.py`) was **stripped in the v1.0 cleanup and is broken**
  (imports a deleted module) — the working standalone producer now lives as `agent/reachy_streamer.py`
  in ShredderSegmenter (SDK `get_frame` → ffmpeg → MediaMTX, zero `maxim.*`). Zero DoA / orienting / NAc content.
- **This runbook is the orient / motor-control + learning layer**: DoA → head turn → NAc learns to
  point at the target. They **combine later** (Maxim orients the head → better footage → Shredder
  segments it) but bring up **independently**.
- **This loop needs ONLY the Reachy SDK** — no MediaMTX, no Shredder, no RTSP. That makes it the
  simpler first hardware bring-up.
- **Offline-safe (verified):** the Step-1/2/3 scripts import only the Reachy SDK + `maxim.decisions.nac`
  / `maxim.embodiment` (all local) — **no LLM, no HF download, no cloud at runtime** — so they run on
  the robot's internet-less AP with nothing pre-cached. (Phase 2 *visual* is the exception: the vision
  encoder downloads from HF on first use → **pre-cache it before joining the robot's Wi-Fi**.)

## Iteration model

Device-in-the-loop: the operator runs each step's script on the Reachy; the agent reads the JSONL
(`MAXIM_LOG_FILE=/tmp/orient.jsonl`) and iterates. **Verify each hardware primitive before stacking
the next** — do NOT stack 4 unverifiable layers (the "cradle cascade" lesson).

## SDK surface (verified importable; calls confirmed against the installed `reachy_mini`)

- `mini = ReachyMini(host=<ip>, port=8000, connection_mode="network", media_backend="no_media")`
  → WS connect (SDK >= 1.5). Off-robot, always pass the IP.
- `mini.enable_motors()` THEN `mini.wake_up()` — torque first (wake_up alone no longer enables it).
- DoA off-robot: poll `GET /api/state/doa` → `{"angle", "speech_detected"}` (no media stack needed;
  `mini.start_recording()`/`mini.media.get_DoA()` are the ONBOARD path — local USB in >= 1.5).
  - `audio_localization.doa_to_azimuth(angle)` → azimuth ∈ [-1,1] (0=front, ±1=side) — unchanged.
  - `AzimuthDoASource` wraps the onboard read as a non-blocking `PerceptSource`; the off-robot
    variant should poll the REST endpoint instead (same normalization).
- `create_head_pose(yaw=<deg>, degrees=True)` → 4×4 matrix; `mini.goto_target(head=pose, duration=…)`
  (min-jerk) for a discrete step; `mini.get_current_head_pose()` to read current yaw.

## Connecting to a wireless Reachy Mini

**(2026-07-15 rewrite — the section that previously lived here described the
zenoh era, SDK <= 1.4.x, and was obsoleted by the robot reflash to daemon
1.8.3. Full, hardware-validated guidance now lives in
[docs/embodiment/reachy_mini/](../embodiment/reachy_mini/README.md):
[getting started](../embodiment/reachy_mini/getting_started.md) ·
[troubleshooting](../embodiment/reachy_mini/troubleshooting.md) ·
[engineering reference](../embodiment/reachy_mini/engineering.md).)**

The load-bearing facts for this plan:
- **Transport (SDK >= 1.5)**: WebSocket `ws://<host>:8000/ws/sdk`; pass
  `ReachyMini(host=<ip>, port=8000, connection_mode="network")`. No zenoh,
  no :7447, no multicast discovery, no tunnels for wireless daemons.
- **Version-match first**: after ANY reflash, compare
  `curl http://<robot>:8000/api/daemon/status` (daemon version) against the
  laptop's `reachy_mini.__version__`. Cross-era mismatch = unfixable
  connection failures. Maxim pins `>=1.8.3,<1.9`.
- **DoA over the network is REST**: `GET /api/state/doa` (convention
  unchanged: 0=left, pi/2=front, pi=right). Client-side
  `mini.media.get_DoA()` is local-USB — onboard only.
- **`enable_motors()` before anything moves** (daemon boots torque-off with
  `--no-wake-up-on-start`; `wake_up()` no longer enables torque).
- **`media_backend="no_media"`** for the orient loop — motion + DoA need no
  media stack, and the default WebRTC path hits a GStreamer dylib collision
  on macOS.
- **goto_target(body_yaw=...) defaults to 0.0** (actively zeroes the body);
  pass `body_yaw=None` to leave the body alone. Head-yaw workspace clamps
  ~±15-18°; body yaw is the coarse orient axis.

## Steps (each gates the next)

### Step 1 — hardware smoke test  ([`scripts/orient_backbone/live_1_smoke.py`](../../scripts/orient_backbone/live_1_smoke.py)) — **PASSED 2026-07-15**

Result (daemon 1.8.3, station mode, `--host <robot-ip>`): connect+wake OK
(after `enable_motors()`), DoA 20/20 via REST with azimuth tracking + speech
gate flipping, head yaw tracking commanded ±20° at ±14-18° measured
(workspace clamp + small calibration offset — fine for the closed-loop
orient policy). Historical context: the original version of this step
burned a session on the zenoh-era transport; see the troubleshooting doc's
"advice you should now IGNORE" section.
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
Phase 3 (audio+visual fusion). See [`audiovisual_orienting.md`](substrate_native_orienting.md).

**Phase 2 camera notes (from the streaming session's SDK findings):**
- Frames: `mini.media.get_frame()` → BGR `uint8` ~640×480 or `None`. The camera inits on construction
  (`start_recording()` then `get_frame()`; **no `wake_up()` needed** for the feed). `media_backend="no_media"`
  disables it. Wireless uses GStreamer (local) / WebRTC (remote); Lite uses OpenCV.
- **Pre-cache the vision encoder before joining the robot's internet-less AP** (first `_get_encoder()`
  pulls `all-mpnet-base-v2` from HF) — else Phase 2 hangs offline. Phase 1 (audio) has no such fetch.
- **Camera contention:** if Maxim's live loop *and* ShredderSegmenter's standalone streamer
  (`agent/reachy_streamer.py`, which calls `get_frame()` directly) run at once, they fight for frames.
  For simultaneous stream + agent, use the **coexist pattern** — the streamer reads `maxim._last_frame`
  (+ `_last_frame_ts` dedup) instead of `get_frame()` — not the standalone path.
