#!/usr/bin/env python
"""Phase 1, Step 1 — Reachy hardware smoke test (RUN ON/ AGAINST THE DEVICE).

docs/plans/reachy_orient_live.md Step 1. Verifies the THREE primitives the live
orient loop needs, SEPARATELY, before we stack NAc on top:
  (a) connect + wake,  (b) read onboard DoA,  (c) move head_yaw.

Standalone (only the Reachy SDK) so it can also run ONBOARD via ssh.

CONNECTION (wireless Reachy Mini) — TRANSPORT DEPENDS ON SDK/DAEMON VERSION:
  * SDK >= 1.5 (daemon 1.8.x, June-2026 reflash image): plain WebSocket to the
    FastAPI daemon at ws://<host>:8000/ws/sdk. The constructor takes host=/port=
    directly — NO zenoh, NO discovery in the connect path, no 7447, no tunnel.
    Preflight = TCP :8000 + GET /api/daemon/status (backend state + version).
    :8443 is WebRTC media signaling only (camera/audio), never motion control.
    Failure modes behind "Network connection attempt failed":
      - WS handshake refused (1013 "Daemon not ready"): daemon HTTP is up but its
        robot backend/ws_server didn't start -> check /api/daemon/status +
        `journalctl -u reachy-mini-daemon` on the robot.
      - connect OK but no joint_positions/head_pose within ~5 s: motor bring-up
        problem on the daemon side.
    CLIENT<->DAEMON VERSION MUST MATCH ERAS: a 1.2.x client scouts zenoh at a
    1.8.x daemon and times out; check the robot with
    `/venvs/mini_daemon/bin/python -c "import reachy_mini; print(reachy_mini.__version__)"`.
  * SDK 1.2.x (legacy zenoh era): control channel was zenoh :7447 with multicast
    discovery; --via-tunnel (ssh -N -L 7447:127.0.0.1:7447 pollen@<host>) was the
    localhost-bypass. Kept here only for un-reflashed robots.

macOS: "Local Network" permission (System Settings -> Privacy & Security) gates
LAN traffic for Terminal/python — ungranted looks exactly like "robot down".
BEST long-term setup: robot on home Wi-Fi (station mode) — same LAN + internet.
"""

from __future__ import annotations

import argparse
import math
import os
import socket
import subprocess
import sys
import time


def doa_to_azimuth(doa_radians: float) -> float:
    """Onboard DoA (0=left, pi/2=front/back, pi=right) -> azimuth [-1,1].
    Mirrors maxim.embodiment.audio_localization.doa_to_azimuth (inlined so this
    script is dependency-free and can run onboard)."""
    az = (doa_radians - math.pi / 2.0) / (math.pi / 2.0)
    return max(-1.0, min(1.0, az))


def _yaw_deg(pose) -> float:
    return math.degrees(math.atan2(pose[1][0], pose[0][0]))


def _tcp_open(host: str, port: int, timeout: float = 3.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except Exception:  # noqa: BLE001
        return False


def daemon_status(host: str, port: int = 8000, timeout: float = 3.0) -> dict | None:
    """GET /api/daemon/status (SDK >= 1.5 daemon). None on any failure."""
    import json
    import urllib.request

    try:
        with urllib.request.urlopen(f"http://{host}:{port}/api/daemon/status", timeout=timeout) as r:
            return json.loads(r.read().decode())
    except Exception:  # noqa: BLE001
        return None


def get_doa_rest(host: str, port: int = 8000, timeout: float = 2.0) -> tuple[float, bool] | None:
    """SDK >= 1.5: DoA via the daemon's REST endpoint GET /api/state/doa.

    The client-side ``mini.media.get_DoA()`` reads the ReSpeaker over LOCAL
    USB in 1.8.x — it only works onboard. Over the network the daemon reads
    the chip and serves the value here. Convention unchanged from 1.2.6:
    0=left, pi/2=front, pi=right (audio_localization.doa_to_azimuth applies).

    NOTE: this is INTENTIONALLY duplicated with
    maxim.embodiment.audio_localization.make_reachy_rest_doa_reader (the
    canonical library reader). This script is standalone-by-design so it can run
    onboard via ssh with only the Reachy SDK and no ``maxim`` install; it must
    not import ``maxim.*``. live_common.py (which imports maxim) uses the library
    reader, so the duplication is confined to this one bring-up script.
    """
    import json
    import urllib.request

    try:
        with urllib.request.urlopen(f"http://{host}:{port}/api/state/doa", timeout=timeout) as r:
            data = json.loads(r.read().decode())
    except Exception:  # noqa: BLE001
        return None
    if not data:
        return None
    return float(data["angle"]), bool(data.get("speech_detected", False))


def default_gateway() -> str | None:
    """The default-route gateway IP. On the Reachy's own Wi-Fi AP (NetworkManager
    shared mode) the robot IS the gateway, so this auto-locates it wherever you are."""
    try:
        if sys.platform.startswith(("darwin", "freebsd", "openbsd")):
            out = subprocess.run(["netstat", "-rn", "-f", "inet"], capture_output=True, text=True, timeout=3).stdout
            for line in out.splitlines():
                p = line.split()
                if p and p[0] == "default" and len(p) > 1:
                    return p[1]
        else:  # linux
            out = subprocess.run(["ip", "route"], capture_output=True, text=True, timeout=3).stdout
            for line in out.splitlines():
                if line.startswith("default") and "via" in line:
                    return line.split()[2]
    except Exception:  # noqa: BLE001
        return None
    return None


def local_ip_toward(host: str) -> str | None:
    """The laptop's own IP on the interface that routes to `host` (no packets sent)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect((host, 1))
        return s.getsockname()[0]
    except Exception:  # noqa: BLE001
        return None
    finally:
        s.close()


def _slash24(ip: str | None) -> str | None:
    return ".".join(ip.split(".")[:3]) if ip and ip.count(".") == 3 else None


def resolve_host(explicit: str | None) -> tuple[str | None, str]:
    """Robot IP — the operator must DECLARE it (no magic default): --host >
    $MAXIM_REACHY_HOST > (future: maxim config `embodiment.host`). Returns
    (None, "") if unset so the caller can guide first-embodiment setup."""
    if explicit:
        return explicit, "--host"
    env = os.getenv("MAXIM_REACHY_HOST")
    if env:
        return env, "$MAXIM_REACHY_HOST"
    return None, ""


def preflight(host: str) -> None:
    # --- network sanity: is the robot even on your network? (references YOUR ip) ---
    gw = default_gateway()
    mine = local_ip_toward(host)
    print(f"[net] your IP = {mine or '?'}   gateway = {gw or '?'}   target robot = {host}")
    if gw and host == gw:
        print("[net] target == your gateway -> consistent with the robot's own AP. good.")
    elif mine and _slash24(mine) != _slash24(host) and host != gw:
        print(f"[net] ** SUBNET MISMATCH ** you're on {_slash24(mine)}.x but targeting {host}.")
        print("[net]   You're almost certainly not on the robot's network. Either join the")
        print(f"[net]   robot's Wi-Fi (then it's the gateway{f' = {gw}' if gw else ''}), or pass the")
        print("[net]   robot's real IP via --host / $MAXIM_REACHY_HOST.")

    # SDK >= 1.5: the control channel IS the daemon HTTP port (WebSocket on :8000).
    http_ok = _tcp_open(host, 8000)
    print(f"[preflight] daemon :8000 -> {'OK' if http_ok else 'UNREACHABLE'}  (WS control channel, SDK >= 1.5)")
    if http_ok:
        status = daemon_status(host)
        if status is not None:
            ver = status.get("version", "?")
            state = status.get("state", status.get("backend_state", "?"))
            print(f"[preflight] /api/daemon/status -> version={ver} state={state}")
        else:
            print("[preflight] /api/daemon/status -> no JSON (older daemon, or endpoint moved)")
        legacy_zenoh = _tcp_open(host, 7447, timeout=1.0)
        if legacy_zenoh:
            print("[preflight] :7447 open -> legacy zenoh daemon (SDK 1.2.x era); use a matching 1.2.x client")
    else:
        print("            :8000 unreachable. If you're SURE you're on the robot's network, this is")
        print("            almost certainly macOS LOCAL NETWORK permission for THIS terminal — an")
        print("            ungranted process sees the LAN as dead (even ping/nc silently time out).")
        print("            System Settings -> Privacy & Security -> Local Network, then retry.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--host",
        default=None,
        help="Reachy daemon IP. Default: $MAXIM_REACHY_HOST, else the default "
        "gateway (the robot on its own AP), else 10.42.0.1.",
    )
    ap.add_argument(
        "--via-tunnel",
        action="store_true",
        help="connect via localhost:7447 (through an SSH -L tunnel); bypasses multicast",
    )
    args = ap.parse_args()

    host, source = resolve_host(args.host)
    if host is None:
        gw = default_gateway()
        print("[FAIL] No robot address configured. Declare it (first-embodiment setup):")
        print("       --host <ip>   or   export MAXIM_REACHY_HOST=<ip>")
        if gw:
            print(f"       Hint: on the robot's own Wi-Fi AP the robot is usually your gateway = {gw}")
        print("       (Once embodiment config lands, this persists to your maxim config — see runbook.)")
        return 2
    print(f"[host] using {host}  (source: {source})")
    preflight(host)

    try:
        from reachy_mini import ReachyMini
        from reachy_mini.utils import create_head_pose
    except Exception as e:  # noqa: BLE001
        print(f"[FAIL] reachy_mini not importable: {e}")
        return 2
    # Era detection MUST use package metadata: pre-1.5 SDKs (the whole zenoh
    # line) have no __version__ attribute at all.
    try:
        from importlib.metadata import version as _pkg_version

        sdk_ver = _pkg_version("reachy-mini")
        _vt = tuple(int(x) for x in sdk_ver.split(".")[:2])
    except Exception:  # noqa: BLE001 - unknown/dev install: assume current era
        sdk_ver = "?"
        _vt = (99, 0)
    ws_era = _vt >= (1, 5)  # zenoh removed in v1.5.0 (PR #858)
    print(f"[sdk] reachy_mini {sdk_ver} ({'WebSocket transport' if ws_era else 'legacy zenoh transport'})")
    mode = "localhost_only" if args.via_tunnel else "network"
    try:
        if ws_era:
            if args.via_tunnel:
                print("[note] --via-tunnel is a zenoh-era (SDK <= 1.4.x) workaround; ignored for SDK >= 1.5.")
                mode = "network"
            # host= goes straight into ws://<host>:8000/ws/sdk — no discovery involved.
            # no_media: skip WebRTC/GStreamer entirely (the pip GStreamer bundle's
            # libgstpython can't dlopen libpython on macOS and hangs media setup).
            # Motion rides the WS; DoA rides REST /state/doa — media not needed here.
            mini = ReachyMini(host=host, port=8000, connection_mode="network", timeout=10.0, media_backend="no_media")
        else:
            mini = ReachyMini(connection_mode=mode)
    except Exception as e:  # noqa: BLE001
        print(f"[FAIL] connect ({mode}) failed: {e}")
        if ws_era:
            print("       Fast fail = WS handshake refused (daemon backend not ready) ->")
            print(f"       curl http://{host}:8000/api/daemon/status ; journalctl -u reachy-mini-daemon on the robot.")
            print("       ~10s fail = no joint/pose state stream -> motor bring-up problem on the daemon.")
        else:
            print("       Legacy zenoh path: see --via-tunnel / Local Network permission notes in the docstring.")
        return 2
    print(f"[ok] connected ({mode}).")
    if ws_era:
        # 1.8.x: wake_up() no longer enables torque (it only moves + plays a
        # sound). The daemon's --no-wake-up-on-start leaves torque OFF, so
        # goto_target is silently ignored until SetTorqueCmd. Enable first.
        try:
            mini.enable_motors()
            print("[motors] enable_motors() sent (torque on)")
        except Exception as e:  # noqa: BLE001
            print(f"[motors] enable_motors() failed: {e} — head may not move")
        time.sleep(0.5)
    print("[wake] waking up...")
    mini.wake_up()
    if not ws_era:
        mini.start_recording()  # zenoh-era audio path; ws-era smoke runs no_media
    time.sleep(1.0)

    # --- (b) DoA read (~10s) ---
    print("\n[DoA] reading ~10s — MAKE SOUNDS to the LEFT then RIGHT of the robot.")
    print("      azimuth: -1=left  0=front/back  +1=right   (speech gate must be True)")
    valid = 0
    for _ in range(20):
        try:
            if ws_era:
                reading = get_doa_rest(host)
            else:
                reading = mini.media.get_DoA()
        except Exception as e:  # noqa: BLE001
            print(f"      get_DoA raised: {e}")
            reading = None
        if reading is None:
            print("      (no reading yet)")
        else:
            doa_rad, is_speech = reading
            valid += 1
            print(
                f"      doa={float(doa_rad):+.3f}rad  speech={bool(is_speech)!s:<5}  "
                f"azimuth={doa_to_azimuth(float(doa_rad)):+.2f}"
            )
        time.sleep(0.5)
    print(f"[DoA] {valid}/20 readings returned. " + ("ok" if valid else "STOP: no DoA — check the mic stream."))

    # --- (c) head motion ---
    print("\n[motion] moving head yaw +20deg, -20deg, recenter (watch it turn)...")
    print(f"      start yaw = {_yaw_deg(mini.get_current_head_pose()):+.1f}deg")
    for target in (20.0, -20.0, 0.0):
        mini.goto_target(head=create_head_pose(yaw=target, degrees=True), duration=0.6)
        time.sleep(0.9)
        print(f"      commanded {target:+.0f}deg -> measured {_yaw_deg(mini.get_current_head_pose()):+.1f}deg")

    try:
        if not ws_era:
            mini.stop_recording()
    except Exception:  # noqa: BLE001
        pass
    print("\n[done] Report: DoA valid count, azimuth tracked L/R + speech flipped, head moved. -> Step 2.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
