# Reachy Mini — troubleshooting

Symptom-indexed. Every entry below is a failure mode we actually hit on
real hardware; the diagnostic commands are copy-paste-ready. When in doubt,
run the fast discriminating sequence:

```bash
# 1. Is the daemon there at all? (also gives you its VERSION and state)
curl -s http://<robot>:8000/api/daemon/status | python3 -m json.tool
# 2. Does the SDK control channel accept?
python3 -c "import websockets.sync.client as w; c=w.connect('ws://<robot>:8000/ws/sdk'); print('handshake OK:', c.recv(timeout=3)[:120])"
# 3. Does DoA flow?
curl -s http://<robot>:8000/api/state/doa
# 4. What's actually listening on the robot? (over ssh)
ss -tlnp | grep python ; ss -ulnp | grep 5353
```

## "Network connection attempt failed. Make sure a Reachy Mini daemon is running and accessible."

One error string, several causes. **Time the failure** — it discriminates:

| Timing | Cause | Fix |
|---|---|---|
| Instant, default host | `reachy-mini.local` didn't resolve from *python* (getaddrinfo/IPv6-first quirk — curl can succeed while python fails!) | Pass `host="<ip>"`. Set a DHCP reservation so the IP is stable |
| Instant, explicit IP | WS handshake refused: daemon HTTP up but its robot backend/ws_server not started (1013 "Daemon not ready"), or TCP blocked by macOS TCC | `curl /api/daemon/status`; `journalctl -u reachy-mini-daemon -n 50` on the robot; check Local Network permission |
| ~5-10 s | Handshake OK but no `joint_positions`+`head_pose` stream — motor bring-up problem on the daemon | Restart daemon / power-cycle; check `journalctl` |

## SDK hangs forever "Waiting for connection with the server..."

That message is the **zenoh-era (≤1.4.x) client**. Your client and daemon
are on opposite sides of the v1.5.0 transport pivot — usually after a robot
reflash, or because you ran with a stale interpreter (check for a second
venv carrying an old `reachy_mini`). Fix: match versions
([getting_started §1](getting_started.md#version-matching)).

## Everything unreachable — ping, nc, curl all dead

macOS **Local Network permission** (System Settings → Privacy & Security).
An ungranted terminal sees the entire LAN as dead — instant
"No route to host" on unicast TCP too. This is the single most
time-wasting gotcha; check it before diagnosing the robot. Note the
asymmetry that confuses people: another already-granted binary (e.g. a
browser) can reach the robot fine while your terminal can't.

## `ping 10.42.0.1` fails after joining home Wi-Fi

Correct behavior — `10.42.0.1` only exists on the robot's own hotspot
(NetworkManager shared-mode gateway). In station mode use the robot's
DHCP address / `.local` name. Stop using 10.42.0.1 as a health check.

## GStreamer wall of errors, `No such element: appsink`, objc duplicate-class warnings, hang during connect

The pip GStreamer bundle (pulled in by `reachy_mini`) and a homebrew
GStreamer loaded into the same process — the objc runtime warns about
`GstCocoaApplicationDelegate` twice, GLib's type registry corrupts
(`g_param_spec_boxed` CRITICALs), then element creation fails. Triggered
whenever the media stack initializes (default backend on a network
connection = WebRTC = GStreamer).

- **For control/DoA work: sidestep entirely** with
  `media_backend="no_media"` — motion is WS, DoA is REST.
- **When you need camera frames**: remove one GStreamer stack (either
  `pip uninstall` the `gstreamer-*` bundle wheels or make the SDK use only
  homebrew's), and expect the `libgstpython.dylib → libpython3.12.dylib`
  rpath issue with venv pythons.

## `ERROR: No Reachy Mini Audio USB device found!`

The SDK's DoA helper probing **your laptop's USB bus** for the mic array —
`mini.media.get_DoA()` is local-USB in ≥1.5 and only works onboard.
Harmless noise on a laptop. Use `GET /api/state/doa` over the network
(convention unchanged: 0=left, π/2=front, π=right).

## Robot connected, reads work, but it does not MOVE (head stays in the down/sleep pose)

Torque is off. The daemon boots `--no-wake-up-on-start` → motors limp; in
≥1.5 `wake_up()` does NOT enable torque (it only moves + plays a sound).
`goto_target` is silently accepted and ignored; position reads keep
working, which makes it look like a software bug. Fix:

```python
mini.enable_motors()   # SetTorqueCmd — THEN wake_up()/goto_target work
```

Check motor state: `curl -s http://<robot>:8000/api/motors/...` (browse
`http://<robot>:8000/docs` for the exact motor routes on your daemon).

## Head moves but undershoots / offset (commanded ±20°, measured ±14-18°)

Normal, two effects: (1) the Stewart platform's yaw workspace clamps around
±15-18° at neutral pose — use **body yaw** as the coarse axis; (2) a
constant few-degree bias means dashboard calibration is due. Closed-loop
code (servo DoA-error → 0) is immune to both.

## `/state/doa` returns 404

All daemon routes are mounted under `/api` — it's **`/api/state/doa`**.
(Wireless-only routers like wifi_config are mounted at root; API routes
are under `/api`. Browse `http://<robot>:8000/docs` for the live surface.)

## `nmcli ... key-mgmt: property is missing` while joining Wi-Fi

SSID not in the scan cache — see
[getting_started §2](getting_started.md#2-network-setup) for the rescan +
explicit `wifi-sec.key-mgmt wpa-psk` recipe.

## `ssh` dies mid `nmcli connection up`

Success, not failure — one radio; joining your home Wi-Fi tears down the
hotspot you were SSH'd over. Rejoin home Wi-Fi and ssh the new address.

## `source /venvs/apps_venv/bin/activate: No such file`

Current images don't have it. The daemon venv is `/venvs/mini_daemon`;
find the daemon's real interpreter with `tr '\0' ' ' </proc/$(pgrep -f
daemon.app.main)/cmdline`.

## Zenoh-era advice you should now IGNORE (≥1.5 daemons)

Historical fixes that no longer apply — if you find them in old notes/docs,
they date from ≤1.4.x: probing `:7447`, `ssh -N -L 7447:...` tunnels,
`--via-tunnel`, `--no-localhost-only`, `localhost_only=False`,
`robot_name=` namespace matching for discovery, zenoh multicast/Local
Network discovery debugging. On a ≥1.5 daemon there is **no zenoh
listener** — nothing on 7447 is the *expected* state, not a symptom.
