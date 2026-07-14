#!/usr/bin/env python
"""Phase 1, Step 1 — Reachy hardware smoke test (RUN ON/ AGAINST THE DEVICE).

docs/plans/reachy_orient_live.md Step 1. Verifies the THREE primitives the live
orient loop needs, SEPARATELY, before we stack NAc on top:
  (a) connect + wake,  (b) read onboard DoA,  (c) move head_yaw.

Standalone (only the Reachy SDK) so it can also run ONBOARD via ssh.

CONNECTION (wireless Reachy Mini) — the daemon HTTP (:8000) and zenoh control
(:7447) are exposed INDEPENDENTLY. The preflight probes both:
  * :8000 reachable + :7447 reachable  -> zenoh is on the network; if the SDK still
    times out it's macOS multicast DISCOVERY. Grant Terminal/Python "Local Network"
    permission, or run with --via-tunnel (explicit localhost path, no multicast).
  * :8000 reachable + :7447 NOT         -> zenoh is bound to the robot's localhost
    only (daemon default `--localhost-only`). Use an SSH tunnel + --via-tunnel:
        ssh -N -L 7447:127.0.0.1:7447 pollen@10.42.0.1   # pw: root   (keep open)
        python scripts/orient_backbone/live_1_smoke.py --via-tunnel
  * neither reachable                    -> wrong network / robot not booted.

BEST long-term fix: put the robot on your home Wi-Fi (station mode, via its
dashboard/nmcli) instead of its own AP — same LAN + internet, multicast usually works.
"""

from __future__ import annotations

import argparse
import math
import socket
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


def preflight(host: str) -> None:
    # zenoh :7447 is the channel the SDK actually needs; :8000 is the daemon's HTTP
    # API which may or may not be exposed depending on config (don't rely on it).
    zenoh_ok = _tcp_open(host, 7447)
    http_ok = _tcp_open(host, 8000)
    print(f"[preflight] zenoh ctrl  {host}:7447 -> {'OK' if zenoh_ok else 'UNREACHABLE'}  (SDK needs THIS)")
    print(f"[preflight] daemon HTTP {host}:8000 -> {'OK' if http_ok else 'UNREACHABLE'}  (optional, config-dependent)")
    if zenoh_ok:
        print("            zenoh reachable. If the SDK still times out below, it's macOS multicast")
        print("            DISCOVERY (not the robot) -> use --via-tunnel, or grant Local Network permission.")
    elif http_ok:
        print("            zenoh not network-exposed (robot localhost-only default) -> SSH tunnel + --via-tunnel:")
        print(f"            ssh -N -L 7447:127.0.0.1:7447 pollen@{host}   (keep open)")
    else:
        print("            BOTH unreachable. If you're SURE you're on the robot's Wi-Fi, this is almost")
        print("            certainly macOS LOCAL NETWORK permission for THIS terminal — an ungranted")
        print("            process sees the LAN as dead (even ping/nc silently time out). Grant it:")
        print("            System Settings -> Privacy & Security -> Local Network, then retry.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="10.42.0.1", help="Reachy daemon IP (hotspot gateway)")
    ap.add_argument(
        "--via-tunnel",
        action="store_true",
        help="connect via localhost:7447 (through an SSH -L tunnel); bypasses multicast",
    )
    args = ap.parse_args()

    preflight(args.host)

    try:
        from reachy_mini import ReachyMini
        from reachy_mini.utils import create_head_pose
    except Exception as e:  # noqa: BLE001
        print(f"[FAIL] reachy_mini not importable: {e}")
        return 2
    mode = "localhost_only" if args.via_tunnel else "network"
    try:
        mini = ReachyMini(connection_mode=mode)
    except Exception as e:  # noqa: BLE001
        print(f"[FAIL] zenoh connect ({mode}) failed: {e}")
        print("       If HTTP was OK but zenoh timed out: see the preflight hint above")
        print("       (SSH tunnel + --via-tunnel, or macOS Local Network permission).")
        return 2
    print(f"[ok] connected ({mode}). waking up...")
    mini.wake_up()
    mini.start_recording()
    time.sleep(1.0)

    # --- (b) DoA read (~10s) ---
    print("\n[DoA] reading ~10s — MAKE SOUNDS to the LEFT then RIGHT of the robot.")
    print("      azimuth: -1=left  0=front/back  +1=right   (speech gate must be True)")
    valid = 0
    for _ in range(20):
        try:
            reading = mini.media.get_DoA()
        except Exception as e:  # noqa: BLE001
            print(f"      get_DoA() raised: {e}")
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
        mini.stop_recording()
    except Exception:  # noqa: BLE001
        pass
    print("\n[done] Report: DoA valid count, azimuth tracked L/R + speech flipped, head moved. -> Step 2.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
