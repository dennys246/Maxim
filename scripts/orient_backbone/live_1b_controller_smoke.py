#!/usr/bin/env python
"""Hardware smoke for the PRODUCTION controller path (validates PR #397).

`live_1_smoke.py` drives `mini.goto_target()` through the RAW SDK. PR #397's fix
is one layer above it — in `ReachyMiniController.goto_target()` — which
`live_1_smoke.py` never calls. So the standard smoke passes whether or not the
fix is correct; it cannot catch a regression in code it doesn't exercise.

This script drives the controller path and checks the one thing #397 changed:
**when you command `body_yaw`, does the head RIDE ALONG (world yaw tracks the
body) rather than counter-rotate?** The head holds the camera + mic array, so a
counter-rotating head means the sensors don't turn with the body — the bug that
faked a DoA "tracking estimator" and ate a session.

We verify the daemon's ACTUAL head world-yaw (from get_current_head_pose) against
the commanded body_yaw — i.e. we read the frame back rather than trusting our
belief (Principle 4's new checklist item, applied to its own fix).

    ~/Envs/maxim-env/bin/python scripts/orient_backbone/live_1b_controller_smoke.py --host <ip>

No sound source needed — pure motion + proprioception.

PASS: for each commanded body_yaw, measured HEAD world-yaw ≈ body_yaw (ratio ~+1).
FAIL (the bug is back): head world-yaw stays near 0 while the body turns (ratio ~0).
"""

from __future__ import annotations

import argparse
import math
import sys
import time

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import live_1_smoke as smoke  # noqa: E402  (host resolution + preflight)

from maxim.hardware.controller import MotionTarget  # noqa: E402
from maxim.hardware.reachy.controller import ReachyMiniController  # noqa: E402


def _head_world_yaw(ctl: ReachyMiniController) -> float | None:
    """Head yaw in the WORLD frame, straight from the daemon's head pose matrix."""
    m = ctl._mini.get_current_head_pose()  # noqa: SLF001 - smoke reads the raw pose
    if m is None:
        return None
    return math.atan2(m[1][0], m[0][0])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default=None, help="Reachy daemon IP (or $MAXIM_REACHY_HOST)")
    ap.add_argument("--settle", type=float, default=1.2, help="wait after each move before reading back (s)")
    args = ap.parse_args()

    host, source = smoke.resolve_host(args.host)
    if host is None:
        print("[FAIL] no robot address: --host <ip> or export MAXIM_REACHY_HOST=<ip>")
        return 2
    print(f"[host] using {host} (source: {source})")
    smoke.preflight(host)

    # Build + connect through the PRODUCTION controller (not the raw SDK).
    ctl = ReachyMiniController(host=host, connection_mode="network", media_backend="no_media")
    if not ctl.connect(timeout=15.0):
        print("[FAIL] ReachyMiniController.connect() returned False — daemon backend not ready?")
        print(f"       curl http://{host}:8000/api/daemon/status  (want backend_status.ready: true)")
        return 2
    print("[ok] connected via ReachyMiniController")
    try:
        ctl.wake_up()
    except Exception as e:  # noqa: BLE001 - wake is best-effort; motion is the real test
        print(f"[warn] wake_up() raised: {e}")
    try:
        ctl._mini.set_automatic_body_yaw(False)  # noqa: SLF001
    except Exception:  # noqa: BLE001
        pass
    time.sleep(0.5)

    # --- Test 1: body_yaw commanded ALONE — the head must ride along. ---
    print("\n[test 1] body_yaw commanded via the controller; head must RIDE ALONG (world yaw tracks body).")
    print("         (pre-#397 the head counter-rotated to hold world orientation and the mics barely turned)")
    print(f"  {'cmd_body':>9}  {'head_world':>11}  {'ratio':>7}  verdict")
    ratios: list[float] = []
    ctl.goto_target(MotionTarget(head_yaw=0.0, body_yaw=0.0, duration=1.0))
    time.sleep(args.settle)
    for target in (0.3, 0.6, 0.9, 0.0, -0.6, 0.0):
        # A move that ERRORED did not happen — reading head pose after it would
        # misreport "no motion" as "counter-rotates". The 2026-07-17 run did
        # exactly that (the method-enum crash). Fail loud on a rejected move
        # instead of grading a robot that never moved.
        if not ctl.goto_target(MotionTarget(body_yaw=target, duration=0.8)):
            print(f"  {target:>9.2f}   [ABORT] goto_target returned False — the move was REJECTED, not")
            print("             executed. This is not a head-frame result; fix the motion error above.")
            ctl.disconnect()
            return 2
        time.sleep(args.settle)
        hw = _head_world_yaw(ctl)
        if hw is None:
            print(f"  {target:>9.2f}       (no head pose read)")
            continue
        ratio = hw / target if abs(target) > 1e-6 else float("nan")
        if abs(target) > 1e-6:
            ratios.append(ratio)
            verdict = "rides ✓" if ratio > 0.7 else ("COUNTER-ROTATES ✗" if ratio < 0.3 else "partial ?")
        else:
            verdict = f"(recenter: head {hw:+.3f})"
        print(f"  {target:>9.2f}  {hw:>+11.3f}  {ratio:>7.3f}  {verdict}")

    # --- Test 2: head_yaw is BODY-RELATIVE — head_yaw=0.1, body=0.5 => world 0.6. ---
    print("\n[test 2] head_yaw is BODY-RELATIVE: head_yaw=+0.10 with body_yaw=+0.50 => head world ≈ +0.60")
    if not (
        ctl.goto_target(MotionTarget(head_yaw=0.0, body_yaw=0.0, duration=1.0))
        and (time.sleep(args.settle) or True)
        and ctl.goto_target(MotionTarget(head_yaw=0.1, body_yaw=0.5, duration=0.8))
    ):
        print("  [ABORT] a test-2 move was rejected — see the motion error above.")
        ctl.disconnect()
        return 2
    time.sleep(args.settle)
    hw2 = _head_world_yaw(ctl)
    rel_ok = hw2 is not None and abs(hw2 - 0.6) < 0.15
    print(f"  measured head world-yaw = {hw2:+.3f}  (expect ≈ +0.60)  -> {'OK ✓' if rel_ok else 'MISMATCH ✗'}")

    ctl.goto_target(MotionTarget(head_yaw=0.0, body_yaw=0.0, duration=1.0))
    time.sleep(args.settle)
    ctl.disconnect()

    # --- verdict ---
    print("\n[verdict]")
    if not ratios:
        print("  INCONCLUSIVE — no valid head-pose reads. Check the daemon.")
        return 1
    mean_ratio = sum(ratios) / len(ratios)
    print(f"  test 1 mean head-rides-along ratio = {mean_ratio:+.3f}  (want ~+1.0)")
    ride_ok = mean_ratio > 0.7
    if ride_ok and rel_ok:
        print("  PASS — PR #397 verified on hardware: the head rides along with the body, so the")
        print("         camera/mics rotate as commanded, and head_yaw is correctly body-relative.")
        return 0
    if mean_ratio < 0.3:
        print("  FAIL — the head is COUNTER-ROTATING (ratio ~0). PR #397's fix is not taking effect;")
        print("         the mics are not turning. Do NOT rely on any DoA reading taken this way.")
    else:
        print("  FAIL / partial — head neither clearly rides along nor counter-rotates; inspect per-row.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
