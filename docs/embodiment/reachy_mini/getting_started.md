# Reachy Mini — getting started with Maxim

The path from unboxing (or reflash) to a passing Step-1 smoke test. Every
step here was validated on real hardware (2026-07-15 bring-up session).

## 0. Credentials & fixed facts

- SSH: `pollen@reachy-mini.local` (or the IP), password `root`.
- Daemon venv on the robot: `/venvs/mini_daemon` (older docs referenced
  `/venvs/apps_venv` — **gone** on current images).
- Dashboard + API + SDK control all share **one port: 8000**.
- On the robot's own hotspot the robot is `10.42.0.1`; on your home Wi-Fi
  it's a DHCP lease (set a reservation on your router — `.local` names are
  flaky from Python, see below).

## 1. Version matching (do this FIRST after any reflash) {#version-matching}

The SDK transport changed incompatibly at v1.5.0 (see
[README](README.md#the-one-fact-that-explains-most-breakage-the-transport-pivot)).
Check both sides:

```bash
# Robot daemon version (over ssh):
/venvs/mini_daemon/bin/python -c "import reachy_mini; print(reachy_mini.__version__)"
# or from the laptop, no ssh needed:
curl -s http://<robot>:8000/api/daemon/status   # includes "version"

# Laptop SDK version:
python -c "import reachy_mini; print(reachy_mini.__version__)"
```

Match the minor line (`pip install "reachy_mini==<daemon version>"`).
Maxim's `reachy` extra pins `>=1.8.3,<2.0`. **Watch for stale venvs**: a
second environment (e.g. a repo-local `.venv`) with an old SDK will
silently resurrect the dead zenoh transport if you run scripts with the
wrong interpreter.

## 2. Network setup

**Recommended: station mode (robot on your home Wi-Fi).** The robot's own
AP has **no internet uplink** — pip/HF/cloud all fail while you're joined
to it — and forces network-switching. Station mode gives one LAN, with
uplink, for laptop + robot.

From the hotspot (`ssh pollen@10.42.0.1`) or the dashboard
(`http://10.42.0.1:8000` → network settings):

```bash
nmcli device wifi rescan && sleep 3
nmcli device wifi list | grep -i <your-ssid>
nmcli device wifi connect "<SSID>" password "<pw>"
```

Gotchas we hit:

- `Error: 802-11-wireless-security.key-mgmt: property is missing` → the
  SSID wasn't in the scan cache (common for 5 GHz right after boot).
  Rescan; if it persists, declare security explicitly:
  ```bash
  sudo nmcli connection add type wifi ifname wlan0 con-name home \
    ssid "<SSID>" wifi-sec.key-mgmt wpa-psk wifi-sec.psk "<pw>"
  sudo nmcli connection up home
  ```
  (WPA3-only networks: `wifi-sec.key-mgmt sae`.)
- **Your SSH session dies the instant it connects — that is success** (one
  radio; the hotspot tears down). Rejoin your home Wi-Fi and
  `ssh pollen@reachy-mini.local`.
- Verify station mode: `nmcli -t -f NAME,DEVICE connection show --active`,
  `ip -4 addr show wlan0` (home-subnet IP), `ping -c2 8.8.8.8` (uplink).
- After the switch, **reboot the robot once** so the daemon re-binds on the
  new network.

## 3. macOS one-time setup

Grant your terminal **Local Network permission** (System Settings →
Privacy & Security → Local Network). An ungranted process sees the whole
LAN as dead — TCP connects fail instantly with "No route to host", and it
looks exactly like "robot down".

## 4. First connect (SDK ≥ 1.5 / WebSocket era)

```python
from reachy_mini import ReachyMini

mini = ReachyMini(
    host="<robot-ip>",           # pass the IP: .local resolution from venv
    port=8000,                   #   python is unreliable (IPv6-first quirk)
    connection_mode="network",   # skip the localhost attempt
    timeout=10.0,
    media_backend="no_media",    # motion+DoA need NO media stack (see below)
)
mini.enable_motors()             # REQUIRED: wake_up() no longer enables torque
mini.wake_up()                   # raises head from sleep pose + flourish
```

Three things that are NOT obvious:

1. **`enable_motors()` is mandatory for movement.** The daemon boots with
   `--no-wake-up-on-start` → torque OFF → head parked in the down/sleep
   pose. In ≥1.5, `wake_up()` only moves and plays a sound; it does NOT
   enable torque. Without `enable_motors()`, every `goto_target` is
   silently ignored (reads still work — the deceptive part).
2. **`media_backend="no_media"` for control work.** The default on network
   connections auto-selects WebRTC → GStreamer on your laptop, which on
   macOS can collide with a homebrew GStreamer and hang or die
   (`No such element: appsink`). Motion rides the WS and DoA rides REST —
   no media stack needed. Only fight the GStreamer battle when you need
   camera frames.
3. **DoA over the network is a REST call**, not `mini.media.get_DoA()`
   (which reads the mic array over *local USB* — onboard only):
   ```bash
   curl -s http://<robot>:8000/api/state/doa
   # {"angle": 1.57, "speech_detected": false}   0=left, π/2=front, π=right
   ```

## 5. Run the Step-1 smoke test

```bash
python scripts/orient_backbone/live_1_smoke.py --host <robot-ip>
```

It preflights `:8000` + `/api/daemon/status`, connects (no_media), enables
motors, wakes, reads DoA ~10 s (make sounds left/right — watch azimuth move
and `speech` flip), then sweeps head yaw ±20°. **Pass criteria**: 20/20 DoA
readings, azimuth tracks your sounds, measured yaw follows commanded
(expect ~±14-18° achieved for ±20° commanded — workspace clamp, normal; a
constant few-degree offset means the dashboard calibration is due).

If anything fails, go straight to
[troubleshooting.md](troubleshooting.md) — it is symptom-indexed.
