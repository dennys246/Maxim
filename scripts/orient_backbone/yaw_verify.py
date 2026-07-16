#!/usr/bin/env python
"""Does the base actually REACH the body_yaw we command? (2-minute diagnostic)

The one unverified link in the orient chain. Every script tracks `body_yaw` as
a BELIEF (what we commanded); nothing has ever read the robot's actual pose
back. If the base under-rotates, every downstream gain measurement is wrong by
that factor — and the Exp 45b numbers say something is off by ~35%:

  sweep (0.1 rad poses, one direction) -> 0.605 az/rad
  learner steps (0.3 and 0.9 rad)      -> 0.39 az/rad, IDENTICAL for both sizes

Identical shortfall at both step sizes rules out settle-lag (bigger steps would
lag more) and backlash (a fixed angular loss would hit the small step harder).
A proportional shortfall is exactly what "the base only travels X% of what we
ask" looks like.

Reads the daemon's real body_yaw from GET /api/state/full and compares against
the command, for a range of step sizes and both directions (a direction
asymmetry would show reversal loss; a size-dependent one would show a speed
limit; a uniform ratio would explain everything).

    ~/Envs/maxim-env/bin/python scripts/orient_backbone/yaw_verify.py --host <ip>

No sound source needed — this is pure motion + proprioception.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import urllib.request

import live_1_smoke as smoke
from live_common import LiveRig, preflight, resolve_host

_head_yaw_deg = smoke._yaw_deg  # 4x4 head pose -> yaw degrees


def read_state(host: str, timeout: float = 3.0) -> dict | None:
    """GET /api/state/full — the daemon's own view (control_mode, head pose, body_yaw)."""
    try:
        url = f"http://{host}:8000/api/state/full"
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.loads(r.read().decode())
    except Exception as e:  # noqa: BLE001
        print(f"      (state read failed: {e})")
        return None


def find_body_yaw(state: dict) -> float | None:
    """Locate body_yaw in the state payload (shape varies by daemon version)."""
    if not isinstance(state, dict):
        return None
    if isinstance(state.get("body_yaw"), (int, float)):
        return float(state["body_yaw"])
    for v in state.values():  # one level of nesting
        if isinstance(v, dict) and isinstance(v.get("body_yaw"), (int, float)):
            return float(v["body_yaw"])
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default=None, help="Reachy daemon IP (or $MAXIM_REACHY_HOST)")
    ap.add_argument("--settle", type=float, default=1.0, help="wait before reading back (s)")
    args = ap.parse_args()

    host, source = resolve_host(args.host)
    if host is None:
        print("[FAIL] no robot address: --host <ip> or export MAXIM_REACHY_HOST=<ip>")
        return 2
    print(f"[host] using {host} (source: {source})")
    preflight(host)

    probe = read_state(host)
    if probe is None:
        print("[FAIL] /api/state/full unreachable — cannot verify pose.")
        return 2
    if find_body_yaw(probe) is None:
        print("[FAIL] body_yaw not found in /api/state/full. Payload keys:")
        print(f"       {sorted(probe)[:20]}")
        print("       Browse http://%s:8000/docs for the right route on this daemon." % host)
        return 2

    rig = LiveRig(host)
    rig.recenter()
    time.sleep(1.0)
    zero = find_body_yaw(read_state(host) or {})
    head0 = _head_yaw_deg(rig.mini.get_current_head_pose())
    print(f"\n[zero] actual body_yaw after recenter: {zero:+.3f} rad   head_yaw: {head0:+.1f} deg")
    print("\n  THE QUESTION: do the MICROPHONES turn? They live in the HEAD, and every")
    print("  goto_target(body_yaw=X) we issue passes head=None. If the daemon holds the")
    print("  head in a WORLD frame it counter-rotates and the mic array never moves —")
    print("  which would explain a -0.9 rad command producing ~0.2 rad of DoA change.")
    print("  head_yaw drifting OPPOSITE to body_yaw = counter-rotation = mystery solved.\n")
    print(
        f"  {'cmd_body':>9}  {'act_body':>9}  {'head_deg':>9}  {'d_cmd':>7}  {'d_act':>7}  {'ratio':>6}  {'mic_frame':>9}"
    )

    # Alternating directions + a range of sizes: a direction asymmetry means
    # reversal loss; a size dependence means a speed/duration limit; a flat
    # ratio < 1 means uniform under-travel (the 45b signature).
    plan = [0.3, 0.6, 0.9, 0.3, 0.0, -0.3, -0.6, -0.9, -0.3, 0.0]
    prev_cmd, prev_act = 0.0, zero or 0.0
    ratios: list[tuple[float, float]] = []
    heads: list[tuple[float, float]] = []
    for target in plan:
        rig.goto_body_yaw(target, duration=0.6)
        time.sleep(args.settle)
        act = find_body_yaw(read_state(host) or {})
        head = _head_yaw_deg(rig.mini.get_current_head_pose())
        if act is None:
            print(f"  {target:>9.2f}     (no state read)")
            continue
        d_cmd = target - prev_cmd
        d_act = act - prev_act
        ratio = (d_act / d_cmd) if abs(d_cmd) > 1e-6 else float("nan")
        # The mic frame = body rotation + head rotation (head_yaw is head-relative-to-body
        # IF the head rides along; if the daemon world-locks the head, head_yaw will show
        # the counter-rotation and the mic frame stays put).
        mic_frame = act + math.radians(head)
        if abs(d_cmd) > 1e-6:
            ratios.append((d_cmd, ratio))
        heads.append((act, head))
        print(
            f"  {target:>9.2f}  {act:>9.3f}  {head:>9.1f}  {d_cmd:>7.3f}  {d_act:>7.3f}  "
            f"{ratio:>6.3f}  {mic_frame:>9.3f}"
        )
        prev_cmd, prev_act = target, act

    print("\n[read]")
    if ratios:
        mean_ratio = sum(r for _, r in ratios) / len(ratios)
        print(f"  mean commanded->actual travel ratio = {mean_ratio:.3f}")
        if mean_ratio > 0.95:
            print("  => the base REACHES its targets. The 0.39-vs-0.605 gain gap is NOT under-travel;")
            print("     next suspect is the source geometry differing between the two measurements.")
        else:
            print(f"  => the base UNDER-TRAVELS by {(1 - mean_ratio) * 100:.0f}%. That alone would explain the")
            print("     apparent gain shortfall (0.605 * ratio ~= measured). Fix the motion, not the metric:")
            print("     longer goto duration, or command the *actual* pose (read-back) instead of the belief.")
        big = [r for d, r in ratios if abs(d) > 0.5]
        small = [r for d, r in ratios if abs(d) <= 0.35]
        if big and small:
            print(f"  by size: small steps {sum(small) / len(small):.3f} vs large steps {sum(big) / len(big):.3f}")
            print("     (size-dependent => duration/speed limit; flat => uniform scale error)")
    # THE head verdict — does the mic frame follow the body?
    if len(heads) >= 4:
        spread = max(h for _, h in heads) - min(h for _, h in heads)
        # correlation sign between body yaw and head yaw
        bs = [b for b, _ in heads]
        hs = [math.radians(h) for _, h in heads]
        mb, mh = sum(bs) / len(bs), sum(hs) / len(hs)
        cov = sum((b - mb) * (h - mh) for b, h in zip(bs, hs))
        var = sum((b - mb) ** 2 for b in bs)
        slope = cov / var if var > 1e-9 else 0.0
        print(f"\n  head_yaw spread over the run: {spread:.1f} deg   d(head)/d(body) = {slope:+.3f}")
        # get_current_head_pose() is WORLD-referenced (their fk() folds body_yaw in),
        # so: slope +1 = head rides along with the body (mics turn = CORRECT);
        # slope ~0 = head holds world orientation (mics DON'T turn = the head=None bug).
        if slope > 0.9:
            print("  => CORRECT: the head rides along with the body, so the MICROPHONES rotate as")
            print("     commanded. (This is what shipping an explicit head= matrix buys you.)")
        elif slope < 0.4:
            print("  => THE head=None BUG: the head is holding its WORLD orientation, so the")
            print("     MICROPHONES barely turn. Every DoA reading taken this way is of a nearly")
            print("     stationary array — it will look exactly like a slow/lagging sensor and it")
            print("     is not. FIX: goto_target(head=create_head_pose(yaw=degrees(body_yaw)), ...)")
            print("     — see CLAUDE.md's Reachy head-frame invariant.")
        else:
            print("  => partial drag (head neither held nor rigid) — inspect per-row head_deg;")
            print("     hysteresis here is the signature of a head being dragged, not commanded.")
    rig.recenter()
    return 0


if __name__ == "__main__":
    sys.exit(main())
