# Reachy Mini — engineering reference (SDK ≥ 1.5 / daemon 1.8.x)

The API surface Maxim builds on, verified against installed SDK 1.8.3
source + a live 1.8.3 daemon (2026-07-15). For the transport history and
the era table, see the [README](README.md). For hardware limits, see
[audio_localization.md](audio_localization.md).

## Transport

- **Control**: WebSocket `ws://<host>:8000/ws/sdk` on the daemon's FastAPI
  server. JSON pydantic messages; the daemon streams state at ~50 Hz per
  connected client. `ReachyMini(host=, port=, connection_mode=)` — no
  discovery in the connect path; `"auto"` tries localhost then falls back
  to `host:port`; `"network"` goes direct.
- **REST**: same port, `/api` prefix (FastAPI — browse `/docs` /
  `/openapi.json` on the robot for the authoritative live surface).
- **Media**: WebRTC via GStreamer, signaling on `:8443`. Media is fully
  optional (`media_backend="no_media"`), and control never touches 8443.
- **mDNS**: the daemon self-advertises `_reachy-mini._tcp.local.` (TXT
  includes `version`, `ws_path=/ws/sdk`); `reachy_mini.utils.discovery.
  find_robots()` browses it (with a macOS `dns-sd` fallback). Constructor
  never calls discovery.
- **Auth**: none on `/ws/sdk` or the API in 1.8.x. Treat the robot's LAN
  as the trust boundary.
- **Version handshake**: soft — the daemon broadcasts its version; the
  client warns (RuntimeWarning) on mismatch, no hard gate. Cross-era
  mismatches fail as connection errors instead (see troubleshooting).

## REST endpoints Maxim uses

| Endpoint | Returns | Notes |
|---|---|---|
| `GET /api/daemon/status` | daemon `version`, `state`, backend info, `camera_specs_name`, `no_media` | the canonical liveness/readiness probe (replaces the zenoh-era :7447 TCP probe) |
| `GET /api/state/doa` | `{"angle": rad, "speech_detected": bool}` | **the network DoA path** — daemon reads the XVF3800 over its local USB and serves the value. Convention: 0=left, π/2=front, π=right (unchanged from 1.2.x, so `audio_localization.doa_to_azimuth` applies as-is) |
| `GET /api/state/full?with_doa=true&...` | batched state (control_mode, head pose, body_yaw, antennas, optional DoA) | one round-trip for the whole orient-loop input; also `with_head_joints`, `use_pose_matrix`, etc. |
| `GET /api/motors/...` | motor status incl. control mode | browse `/docs` for exact routes per daemon version |

Polling `GET /api/state/doa` at 2–10 Hz is entirely adequate for the orient
loop; there is also a state-streaming WS (`/state/ws/full`-family) if
polling ever becomes the bottleneck.

**Client-side `mini.media.get_DoA()` is a trap on laptops**: in ≥1.5 it
reads the ReSpeaker over *local USB* (`AudioDoA → init_respeaker_usb()`),
so it works onboard only and logs `No Reachy Mini Audio USB device found!`
elsewhere. Use the REST endpoint over the network.

## Motion {#motion}

```python
from reachy_mini.utils import create_head_pose
mini.goto_target(head=create_head_pose(yaw=15, degrees=True), duration=0.6)
```

Facts that bit us, in order of pain:

1. **Torque is a separate, explicit gate.** `enable_motors()` /
   `disable_motors()` send `SetTorqueCmd` over the WS. The daemon's
   `--no-wake-up-on-start` boots torque-off; `wake_up()` does NOT enable
   torque in ≥1.5 (it moves + plays `wake_up.wav`). Commands to limp
   motors are **silently ignored while reads keep working** — always
   enable before commanding.
2. **`goto_target(body_yaw=0.0)` is the DEFAULT** — every head command
   also actively drives the body to zero yaw. Pass `body_yaw=None` to
   leave the body alone, or command it deliberately.
3. **Head-yaw workspace is ~±15-18° in practice** (Stewart platform clamp
   at neutral pose, slightly asymmetric on our unit). The body-yaw axis is
   the coarse stage for large orient moves; the head is the fine stage.
   Design orient policies closed-loop (servo DoA-error → 0) so clamping
   and calibration offsets wash out.
4. `wake_up()` internally calls `mini.media.play_sound(...)` — with
   `no_media` this logs "Audio system is not initialized." Harmless.
5. Interpolation: `goto_target(..., method=)` supports `minjerk` (default),
   `linear`, `ease_in_out`, `cartoon`. `set_target()` is the immediate
   (no-interpolation) variant for tracking loops.

## Media backends

`media_backend=` resolution in 1.8.3:

- `"no_media"` → nothing initializes (verified: the `MediaManager`
  NO_MEDIA branch is a no-op). The client also tells the daemon to
  `release_media()` — daemon-side DoA (its own USB handle) is unaffected.
- `"default"`/`"auto"` → LOCAL (GStreamer IPC) only when
  `connection_mode="localhost_only"` and a local camera exists; otherwise
  **WEBRTC** — which initializes GStreamer on the client. On macOS with a
  homebrew GStreamer installed this collides with the pip bundle (see
  [troubleshooting](troubleshooting.md)); resolve the double-install
  before doing camera work on a Mac.

## Maxim integration points

| Surface | File | 1.8.x status |
|---|---|---|
| SDK pin | `pyproject.toml` `reachy` extra | `reachy-mini[gstreamer]>=1.8.3,<2.0` |
| Connection controller | `src/maxim/hardware/reachy/controller.py` (+ `~/.maxim/robots.yaml`) | probes :8000 + `/api/daemon/status`; resolved host passed to the constructor; `tunnel:` retargeted to forward :8000 (useful only for loopback-bound Lite daemons — zenoh-era :7447 tunnels are obsolete) |
| Legacy connection config | `src/maxim/embodied_runtime/connection.py` | `connection_mode="network"`, `spawn_daemon=False` |
| DoA → azimuth | `src/maxim/embodiment/audio_localization.py` | convention unchanged across the pivot; feed it from `GET /api/state/doa` |
| Smoke test | `scripts/orient_backbone/live_1_smoke.py` | era-aware; REST DoA; `enable_motors()` before wake |
| Diagnostics | `scripts/check_reachy_connection.py`, `src/maxim/utils/reachy_diagnostics.py` (`maxim-diagnostics`) | WS-era probes (:8000 + status), zenoh-era checks retired |
| Runbook | `docs/plans/reachy_orient_live.md` | Step 1 PASSED 2026-07-15 on the WS transport |

## Robot-side facts (Wireless, June-2026 image)

- Daemon: `python -u -m reachy_mini.daemon.app.main --wireless-version
  --no-wake-up-on-start` from `/venvs/mini_daemon`; systemd unit
  `reachy-mini-daemon`. `--wireless-version` = bind 0.0.0.0:8000 + mount
  wireless routers (update/wifi_config/cache/logs) + mDNS "Reachy Mini
  Wireless".
- Listeners: TCP 8000 (FastAPI + WS), TCP 8443 (WebRTC signaling),
  UDP 5353 ×N (daemon's own zeroconf responder — this is why
  `reachy-mini.local` resolves so reliably for *curl*; python's
  getaddrinfo is the flaky one).
- After changing Wi-Fi, reboot: the daemon binds at startup and a network
  flip can leave stale state.
