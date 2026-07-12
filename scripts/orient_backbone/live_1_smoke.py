#!/usr/bin/env python
"""Phase 1, Step 1 — Reachy hardware smoke test (RUN ON/ AGAINST THE DEVICE).

docs/plans/reachy_orient_live.md Step 1. Verifies the THREE primitives the live
orient loop needs, SEPARATELY, before we stack NAc on top:
  (a) connect + wake,
  (b) read onboard DoA (make sounds left/right; azimuth should track the side),
  (c) move head_yaw +20deg / -20deg / recenter.

Standalone: needs ONLY the Reachy SDK (no maxim / MediaMTX / Shredder), so it can
also run ONBOARD via `ssh pollen@reachy-mini` as a fallback if network discovery
fails. Usage (from the laptop, on the Reachy hotspot):

    python scripts/orient_backbone/live_1_smoke.py            # host defaults to 10.42.0.1
    python scripts/orient_backbone/live_1_smoke.py --host 10.42.0.1

CONNECTION NOTES (wireless Reachy Mini):
  * The daemon runs on the robot when powered on and is at the hotspot gateway
    (10.42.0.1). From a laptop use connection_mode="network" (multicast discovery).
  * macOS: grant Terminal/Python "Local Network" permission (System Settings ->
    Privacy & Security -> Local Network) or zenoh multicast discovery silently times out.
  * Pre-flight below hits http://<host>:8000/docs — if THAT loads, the daemon is up
    and network-reachable, and any remaining failure is zenoh discovery (perm/multicast).
"""

from __future__ import annotations

import argparse
import math
import sys
import time
import urllib.request


def doa_to_azimuth(doa_radians: float) -> float:
    """Onboard DoA (0=left, pi/2=front/back, pi=right) -> azimuth [-1,1].
    Mirrors maxim.embodiment.audio_localization.doa_to_azimuth (inlined to keep
    this script dependency-free so it can run onboard)."""
    az = (doa_radians - math.pi / 2.0) / (math.pi / 2.0)
    return max(-1.0, min(1.0, az))


def _yaw_deg(pose) -> float:
    return math.degrees(math.atan2(pose[1][0], pose[0][0]))


def preflight(host: str) -> None:
    url = f"http://{host}:8000/docs"
    try:
        with urllib.request.urlopen(url, timeout=3) as r:
            print(f"[preflight] daemon HTTP reachable at {url} (status {r.status}) — "
                  "daemon is up + network-exposed.")
    except Exception as e:  # noqa: BLE001
        print(f"[preflight] could NOT reach {url}: {e}")
        print("            -> the daemon isn't network-reachable at that IP. Check the robot")
        print("               is powered/booted and you're on its hotspot (gateway 10.42.0.1).")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="10.42.0.1", help="Reachy daemon IP (hotspot gateway)")
    args = ap.parse_args()

    preflight(args.host)

    # --- (a) connect + wake ---
    try:
        from reachy_mini import ReachyMini
        from reachy_mini.utils import create_head_pose
    except Exception as e:  # noqa: BLE001
        print(f"[FAIL] reachy_mini not importable: {e}")
        return 2
    try:
        # Wireless robot from a laptop: force network (multicast) discovery — do NOT
        # let auto waste 5s on localhost first.
        mini = ReachyMini(connection_mode="network")
    except Exception as e:  # noqa: BLE001
        print(f"[FAIL] could not connect to the daemon ({e}).")
        print("       If preflight above SUCCEEDED, this is zenoh discovery: (1) macOS Local")
        print("       Network permission for your terminal, (2) same hotspot, (3) fallback:")
        print("       ssh pollen@reachy-mini (pw root); source /venvs/apps_venv/bin/activate;")
        print("       run this script there with default ReachyMini().")
        return 2
    print("[ok] connected. waking up...")
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
            az = doa_to_azimuth(float(doa_rad))
            valid += 1
            print(f"      doa={float(doa_rad):+.3f}rad  speech={bool(is_speech)!s:<5}  azimuth={az:+.2f}")
        time.sleep(0.5)
    print(f"[DoA] {valid}/20 readings returned. "
          + ("ok" if valid else "STOP: no DoA — check the mic stream."))

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
    print("\n[done] Report: DoA valid count, whether azimuth tracked L/R + speech flipped,")
    print("       and whether the head visibly moved. Then -> Step 2.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
