#!/usr/bin/env python
"""Phase 1, Step 1 — Reachy hardware smoke test (RUN ON THE DEVICE).

docs/plans/reachy_orient_live.md Step 1. Verifies the THREE primitives the live
orient loop needs, SEPARATELY, before we stack NAc on top:
  (a) connect + wake,
  (b) read onboard DoA (make sounds left/right; azimuth should track the side),
  (c) move head_yaw +20deg / -20deg / recenter.

Needs ONLY the Reachy SDK (no MediaMTX / Shredder / RTSP). Run with logging so the
result can be read back:

    MAXIM_LOG_FILE=/tmp/orient.jsonl python scripts/orient_backbone/live_1_smoke.py

STOP-if: DoA never returns, azimuth doesn't track L<->R, or the head doesn't move.
"""

from __future__ import annotations

import math
import sys
import time

from maxim.embodiment.audio_localization import doa_to_azimuth


def _yaw_deg(pose) -> float:
    """Extract yaw (deg) from a 4x4 head pose (rotation about Z)."""
    return math.degrees(math.atan2(pose[1][0], pose[0][0]))


def main() -> int:
    # --- (a) connect + wake ---
    try:
        from reachy_mini import ReachyMini
        from reachy_mini.utils import create_head_pose
    except Exception as e:  # noqa: BLE001
        print(f"[FAIL] reachy_mini not importable: {e}")
        return 2
    try:
        mini = ReachyMini()
    except Exception as e:  # noqa: BLE001
        print(f"[FAIL] could not connect to the robot ({e}).")
        print("       Is the Reachy powered on and its daemon running?")
        return 2
    print("[ok] connected. waking up...")
    mini.wake_up()
    mini.start_recording()  # start media stream so get_DoA() produces values
    time.sleep(1.0)

    # --- (b) DoA read (~10s) ---
    print("\n[DoA] reading for ~10s — MAKE SOUNDS to the LEFT then RIGHT of the robot.")
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
    start_yaw = _yaw_deg(mini.get_current_head_pose())
    print(f"      start yaw = {start_yaw:+.1f}deg")
    for target in (20.0, -20.0, 0.0):
        mini.goto_target(head=create_head_pose(yaw=target, degrees=True), duration=0.6)
        time.sleep(0.9)
        now = _yaw_deg(mini.get_current_head_pose())
        print(f"      commanded {target:+.0f}deg -> measured {now:+.1f}deg")

    try:
        mini.stop_recording()
    except Exception:  # noqa: BLE001
        pass
    print("\n[done] Smoke test complete. Report: DoA valid count, whether azimuth")
    print("       tracked L/R, and whether the head visibly moved. Then -> Step 2.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
