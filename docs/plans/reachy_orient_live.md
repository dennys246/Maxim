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

- `mini = ReachyMini()` → connect (blocks/TimeoutError if the robot/daemon is down).
- `mini.wake_up()` / `mini.enable_motors()` — enable + home.
- `mini.start_recording()` — start the media stream so DoA produces values.
- `mini.media.get_DoA()` → `(doa_radians, is_speech_detected)` or `None`.
  - `audio_localization.doa_to_azimuth(doa_radians)` → azimuth ∈ [-1,1] (0=front, ±1=side).
  - `AzimuthDoASource` already wraps this as a non-blocking `PerceptSource`.
- `create_head_pose(yaw=<deg>, degrees=True)` → 4×4 matrix; `mini.goto_target(head=pose, duration=…)`
  (min-jerk) for a discrete step; `mini.get_current_head_pose()` to read current yaw.

## Connecting to a wireless Reachy Mini (setup gotchas)

The daemon runs **on the robot** when powered on; the SDK client (`ReachyMini`) finds it. Confirmed
against the installed SDK (1.2.6) + [issue #677](https://github.com/pollen-robotics/reachy_mini/issues/677)
+ the [quickstart](https://huggingface.co/docs/reachy_mini/SDK/quickstart):
- The robot's own hotspot puts the **robot at `10.42.0.1`** (NetworkManager shared-mode gateway); the
  laptop gets `10.42.0.x`. Same L2 network — good.
- From the laptop use **`ReachyMini(connection_mode="network")`** (a.k.a. `localhost_only=False`) — do
  NOT let `auto` burn 5s on a nonexistent localhost daemon. #677: "remote SDK connections from a laptop
  work fine using `ReachyMini(localhost_only=False)`."
- Discovery is **zenoh multicast/gossip** (no explicit-IP option in the constructor). On **macOS,
  "Local Network" permission is MANDATORY** (System Settings → Privacy & Security → Local Network) for
  the terminal app. Without it, discovery **silently times out and looks exactly like "robot down"** —
  an ungranted process sees the whole LAN as dead (even `ping`/`nc` return nothing). This is the single
  most time-wasting gotcha; grant it first.
- **Reliable signal is zenoh `:7447`, not `:8000`.** The daemon's HTTP API (`:8000`) may or may not be
  network-exposed depending on config (one session saw it up, another down) — don't diagnose off it.
  `live_1_smoke.py` pre-flights `:7447` (SDK control) + `:8000` (informational) and branches the hint.
- **If zenoh isn't network-reachable** (robot binds it to localhost — the daemon default), tunnel it:
  `ssh -N -L 7447:127.0.0.1:7447 pollen@10.42.0.1` (pw `root`, keep open) + run with `--via-tunnel`.
- **Best long-term fix:** put the robot on your **home Wi-Fi** (station mode, via its dashboard/`nmcli`)
  instead of its own AP — same LAN *with internet* (the AP has **no uplink**, so pip/HF/cloud/Docker all
  fail while joined), no network-switching, and multicast usually just works.
- **Fallback (sidesteps all network issues):** run onboard — `ssh pollen@reachy-mini` (pw `root`),
  `source /venvs/apps_venv/bin/activate`, run with plain `ReachyMini()`. The Step-1 smoke test is
  dependency-free so it runs there; the full loop needs maxim installed on the Pi.

## Robot address — declared, not guessed (design)

The robot's IP changes location-to-location, so **the operator declares it once; Maxim persists it** —
no magic default (matches the repo's no-magic-config discipline). Resolution order (highest first):
`--host` → `$MAXIM_REACHY_HOST` → **maxim config `embodiment.host`** (planned). If none set, the tools
fail with first-embodiment guidance (and *hint* the gateway — on the robot's own AP the robot IS the
gateway — but never silently use it).

**Planned config shape** (route through the standard config layer per the dev standard —
`resolve_setting` + a `*ConfigSection` + `config_writer`, NOT a new env var):
```
config.json :: "embodiment": { "robot_type": "reachy_mini", "host": "<ip>", "connection_mode": "network" }
```
First embody assigns it (`maxim embody --robot reachy_mini --host <ip>` writes it via `config_writer`);
the runtime reads `resolve_setting("embodiment.host", cli_value=...)`. **Sequencing:** implement this
*after* the first successful SDK connection, so we persist a *validated* config rather than design
around an unproven path. The Step-1 smoke test stays standalone (`--host`/env only) so it runs onboard.

**Robot-agnostic seam (for Atlas / other robots):** `robot_type` dispatches to a robot plugin
(`maxim.robots` entry-point group). Each plugin owns its connection (Reachy → `ReachyMini`/zenoh;
Atlas → its own SDK) and exposes the robot-neutral `PerceptSource`/`ActionSink` + a SEM body YAML.
The orient backbone (drive + affordances + NAc + `potential_diff`) is already modality/robot-agnostic
and does **not** change per robot. Build only the Reachy backend now; a real second robot proves the
abstraction by parallel use (don't speculatively generalize the transport for a hypothetical Atlas).

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
