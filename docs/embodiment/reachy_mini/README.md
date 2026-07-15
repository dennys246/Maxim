# Reachy Mini — Maxim embodiment guide

The [Reachy Mini](https://huggingface.co/docs/reachy_mini) is a small desktop
robot from Pollen Robotics / Hugging Face. This folder documents how Maxim
embodies it — battle-tested against real hardware (the 2025-12 first-contact
sessions and the 2026-07-15 station-mode bring-up).

| Page | What it covers |
|---|---|
| [getting_started.md](getting_started.md) | Network setup (AP vs home Wi-Fi), SDK install + **version matching**, first connect, motors, the Step-1 smoke test |
| [troubleshooting.md](troubleshooting.md) | Symptom-indexed decision tree — every failure mode we have actually hit, with the exact diagnostic and fix |
| [engineering.md](engineering.md) | The 1.8.x transport + API surface Maxim builds on: WebSocket control, REST endpoints (incl. network DoA), motion semantics, media backends, Maxim integration points |
| [audio_localization.md](audio_localization.md) | The audio deep-dive: XVF3800 chip limits, why TDOA/elevation are impossible, on-chip DoA, and the orient-policy thesis |

## Hardware overview

| Subsystem | What it is |
|---|---|
| **Head** | 6-DOF Stewart platform (3 rotations + 3 translations) — commanded as task-space 4×4 poses, not joint angles. Head-yaw travel is **~±15-18° in practice** (workspace clamp, slight asymmetry); use body yaw as the coarse axis |
| **Body** | 1 yaw rotation (turn the whole robot) |
| **Antennas** | 2 expressive antennas |
| **Vision** | Camera (single) |
| **Audio in** | 4× PDM MEMS mic array behind a **Seeed reSpeaker XVF3800 (XMOS)** voice processor — see [audio_localization.md](audio_localization.md) |
| **Audio out** | Speaker |
| **Compute (Wireless)** | Raspberry Pi (aarch64, Debian) running the `reachy-mini-daemon`; venv at `/venvs/mini_daemon` |

Both *Lite* and *Wireless* variants carry the same 4-mic array.

## The one fact that explains most breakage: the transport pivot

Pollen **removed zenoh in SDK v1.5.0** (2026-03-05, PR #858). Everything
about connecting changed at that boundary:

| | **zenoh era (≤ 1.4.x)** | **WebSocket era (≥ 1.5.0)** |
|---|---|---|
| Control channel | zenoh, TCP **:7447**, multicast/gossip discovery | WebSocket **`ws://<host>:8000/ws/sdk`** on the daemon's FastAPI port |
| Addressing | discovery only — **no host kwarg** | **`ReachyMini(host=, port=)`** — direct, no discovery in the connect path |
| Discovery | zenoh multicast (flaky on Wi-Fi/WSL — the reason it was removed) | mDNS advertisement `_reachy-mini._tcp.local.` (daemon is its own responder on :5353); used by `find_robots()`, **not** by the constructor |
| :8443 | — | GStreamer **WebRTC media signaling** only (camera/audio streaming), never motion control |
| Localhost-bind workarounds | `--no-localhost-only` daemon flag, `ssh -L 7447` tunnels | obsolete — `--wireless-version` binds 0.0.0.0:8000 |
| DoA from a laptop | via zenoh topics | **`GET /api/state/doa`** (client-side `media.get_DoA()` reads USB *locally* — onboard only) |
| Auth | none | none (`/ws/sdk` has no pairing/token in 1.8.x) |

**Client and daemon must be on the same side of the pivot.** A 1.2.x client
scouts zenoh at a 1.8.x daemon forever; the reverse 404s the WS upgrade. A
robot reflash silently moves the daemon — after ANY reflash, run the version
check in [getting_started.md](getting_started.md#version-matching) first.

Maxim pins `reachy-mini[gstreamer]>=1.8.3,<2.0` (pyproject `reachy` extra).

## How Maxim embodies it

Reachy Mini is a **self-contained** embodiment: one node runs the whole
perception → cognition → action loop. In
[perception-placement](../../plans/perception_pipeline_placement.md) terms,
every stage is local — nothing crosses a wire.

- **Sensors / drives / affordances** are declared in the body YAML (see
  [`embodiment_guide.md`](../../embodiment_guide.md)). For sound-orienting:
  an `azimuth` sensor, a centeredness homeostatic drive on it, and discrete
  orient affordances.
- **Motor** is the head-pose API: `create_head_pose(yaw=..., degrees=True)`
  + `goto_target(...)` / `set_target(...)`. See
  [engineering.md](engineering.md#motion) for the 1.8.x semantics that bit
  us (`body_yaw=0.0` default, torque gating, workspace clamp).
- **The thesis claim on this platform**: localization is on-chip, so Maxim
  is not learning to localize — it learns the **sensorimotor orient policy**
  (which way, how much, until azimuth-error → 0), credited by drive-pain
  reduction through NAc. Full argument in
  [audio_localization.md](audio_localization.md).
