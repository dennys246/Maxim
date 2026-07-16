#!/usr/bin/env python
"""Phase 1, Step 2 — reactive orient, NO learning (coordinate-frame calibration).

docs/plans/reachy_orient_live.md Step 2. PRIMARY PURPOSE: calibrate the
coordinate-frame sign — confirm that a discrete turn toward the sensed side
REDUCES |azimuth|, and determine the sign to pass to Step 3 (--flip-sign).

Design facts this run pins down empirically:
  - ORIENT AXIS IS BODY YAW. The mics ride in the head, but head yaw clamps
    ~±15-18° (≈ ±0.2 normalized azimuth) while azimuth spans ±1 (±90°):
    a head-only step saturates after one move and the closed loop dies at
    the clamp. Body yaw (±π turntable) rotates the whole head+mic assembly.
    Body motion is a NEW primitive — verified first, separately, below.
  - SIGN HYPOTHESIS (default): +body_yaw = turn left = azimuth grows toward
    +1 (the body YAML's sim convention). Centering step is therefore
    delta_yaw = -sign(az) * step. If trials consistently GROW |az|, the
    script auto-flips after 4 consecutive worsenings (and tells you to pass
    --flip-sign to live_3_learn.py).
  - WATCH-ITEM: the robot's own motor noise during a turn could perturb the
    DoA/speech gate — az_after is sampled only after motion completes +
    settle, and each sample is a speech-gated median of 3.

OPERATOR PROTOCOL (device-in-loop):
  1. Robot on the same LAN; run from ~/Envs/maxim-env (SDK 1.8.3):
       ~/Envs/maxim-env/bin/python scripts/orient_backbone/live_2_reactive.py --host <robot-ip>
  2. Sustained speech source (talk continuously, or a phone playing a
     podcast) at ~1-2 m, roughly 30-60° to the robot's LEFT, IN FRONT
     (linear array = front/back ambiguous). Let ~8 trials run.
  3. Move the source to the RIGHT side for the remaining trials (both
     sides matter: a one-sided test can't catch an always-turn-one-way bug).
  4. Report /tmp/orient_step2.jsonl back for analysis.

Offline logic check (no robot): --dry-run [--dry-flip] exercises both world
sign conventions.

SUCCESS: >= 75% of valid trials reduce |azimuth|; head visibly turns toward
and holds on the source. The final line prints the calibrated sign flag for
Step 3. STOP-IF: trials are ~50/50 improve/worsen (noise dominates — raise
--step or fix the source placement) or DoA never passes the speech gate.
"""

from __future__ import annotations

import argparse
import sys
import time

from live_common import DryRig, JsonlLog, LiveRig, gated_azimuth, preflight, resolve_host


def sign(x: float) -> float:
    return 1.0 if x > 0 else -1.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default=None, help="Reachy daemon IP (or $MAXIM_REACHY_HOST)")
    ap.add_argument("--trials", type=int, default=16, help="valid calibration trials to collect")
    ap.add_argument("--step", type=float, default=0.25, help="body-yaw step per trial (rad)")
    ap.add_argument("--band", type=float, default=0.1, help="|azimuth| <= band counts as centered")
    ap.add_argument("--max-yaw", type=float, default=1.4, help="clamp cumulative body yaw (rad)")
    ap.add_argument("--duration", type=float, default=0.6, help="goto_target duration (s)")
    ap.add_argument("--settle", type=float, default=0.5, help="extra settle before az_after read (s)")
    ap.add_argument("--flip-sign", action="store_true", help="start from the flipped sign convention")
    ap.add_argument("--log", default="/tmp/orient_step2.jsonl")
    ap.add_argument("--yes", action="store_true", help="skip the body-motion visual confirm prompt")
    ap.add_argument("--dry-run", action="store_true", help="offline logic check (no robot)")
    ap.add_argument("--dry-flip", action="store_true", help="dry-run world uses the OPPOSITE sign convention")
    args = ap.parse_args()

    log = JsonlLog(args.log)
    # sign_mult=+1: default convention (+yaw -> az grows +)  => center via -sign(az)*step
    sign_mult = -1.0 if args.flip_sign else 1.0
    # Dry-run: reads are synthetic and instant — skip real-time pacing.
    poll_s = 0.0 if args.dry_run else 0.15

    def pace(seconds: float) -> None:
        if not args.dry_run:
            time.sleep(seconds)

    if args.dry_run:
        rig = DryRig(theta_src=-0.7, world_flipped=args.dry_flip)
        print(f"[dry] source at world bearing {rig.theta_src:+.2f} rad, world_flipped={rig.world_flipped}")
    else:
        host, source = resolve_host(args.host)
        if host is None:
            print("[FAIL] no robot address: --host <ip> or export MAXIM_REACHY_HOST=<ip>")
            return 2
        print(f"[host] using {host} (source: {source})")
        preflight(host)
        rig = LiveRig(host)

    log.write("start", step=args.step, band=args.band, flip_sign=args.flip_sign, dry_run=args.dry_run)

    # --- NEW-PRIMITIVE CHECK: body-yaw motion (head yaw was verified in Step 1,
    # body yaw was not — verify it alone before the closed loop stacks on it) ---
    if not args.dry_run:
        print("\n[motion] body-yaw primitive check: +0.2 rad (left), -0.2 rad (right), recenter.")
        print("         WATCH THE ROBOT — the whole head assembly should rotate on its base.")
        for target in (0.2, -0.2, 0.0):
            rig.goto_body_yaw(target, duration=args.duration)
            pace(args.duration + 0.4)
            print(f"         commanded body_yaw {target:+.2f} rad")
        if not args.yes:
            ans = input("         Did the body visibly rotate left, right, center? [y/N] ").strip().lower()
            if ans not in ("y", "yes"):
                print("[STOP] body-yaw motion unverified — do not stack the loop on it.")
                log.write("abort", reason="body_yaw_unverified")
                return 1
        log.write("body_yaw_primitive_ok")

    rig.recenter()
    print(f"\n[calib] collecting {args.trials} valid trials — keep a sustained speech source going.")
    print("        (~8 trials source LEFT, then move it RIGHT for the rest)\n")

    valid = improved = worsened = 0
    consecutive_worse = 0
    d_values: list[float] = []
    no_reading_streak = 0
    while valid < args.trials:
        az_before = gated_azimuth(rig.reader, k=3, timeout_s=6.0, poll_s=poll_s)
        if az_before is None:
            no_reading_streak += 1
            if no_reading_streak >= 10:
                print("[STOP] speech gate not passing for ~1 min — check the source, then rerun.")
                break
            print("      (no speech-gated reading — waiting)")
            log.write("no_reading_before")
            continue
        no_reading_streak = 0
        if abs(az_before) <= args.band:
            print(f"      az={az_before:+.2f} centered (|az|<={args.band}) — holding")
            log.write("centered", az=az_before)
            pace(0.5)
            continue

        delta = -sign(az_before) * args.step * sign_mult
        target_yaw = max(-args.max_yaw, min(args.max_yaw, rig.body_yaw + delta))
        if target_yaw == rig.body_yaw:
            print(f"      at yaw limit ({rig.body_yaw:+.2f}) — recentering; move the source nearer front")
            log.write("yaw_limit", body_yaw=rig.body_yaw)
            rig.recenter()
            continue

        rig.goto_body_yaw(target_yaw, duration=args.duration)
        pace(args.duration + args.settle)
        az_after = gated_azimuth(rig.reader, k=3, timeout_s=6.0, poll_s=poll_s)
        if az_after is None:
            print("      (no speech-gated reading after the turn — trial invalid)")
            log.write("no_reading_after", az_before=az_before, delta=delta)
            continue

        d = abs(az_before) - abs(az_after)
        valid += 1
        d_values.append(d)
        if d > 0.02:
            improved += 1
            consecutive_worse = 0
            tag = "IMPROVED"
        elif d < -0.02:
            worsened += 1
            consecutive_worse += 1
            tag = "WORSENED"
        else:
            consecutive_worse = 0
            tag = "neutral"
        print(
            f"      [{valid:2d}/{args.trials}] az {az_before:+.2f} -> {az_after:+.2f}  "
            f"delta_yaw={delta:+.2f}  d|az|={d:+.3f}  {tag}"
        )
        log.write(
            "trial",
            n=valid,
            az_before=round(az_before, 3),
            az_after=round(az_after, 3),
            delta_yaw=round(delta, 3),
            body_yaw=round(rig.body_yaw, 3),
            d_abs=round(d, 4),
            sign_mult=sign_mult,
        )

        if consecutive_worse >= 4:
            sign_mult = -sign_mult
            consecutive_worse = 0
            # Reset tallies: the verdict should reflect the POST-flip convention,
            # not average the two conventions into a fake "noisy".
            valid = improved = worsened = 0
            d_values.clear()
            print(f"\n[flip] 4 consecutive worsenings -> flipping sign convention (sign_mult={sign_mult:+.0f})\n")
            log.write("sign_flip", sign_mult=sign_mult)

    if not args.dry_run:
        rig.recenter()

    mean_d = sum(d_values) / len(d_values) if d_values else 0.0
    print(f"\n[summary] valid={valid} improved={improved} worsened={worsened} mean d|az|={mean_d:+.3f}")
    final_flip = sign_mult < 0
    if valid and improved / valid >= 0.75:
        verdict = "SIGN CALIBRATED"
        step3 = "--flip-sign" if final_flip else "(default sign, no flag)"
        print(f"[verdict] {verdict} — a turn toward the sensed side reduces |azimuth|.")
        print(f"          Step 3 sign: {step3}")
        rc = 0
    elif valid and worsened / valid >= 0.75:
        verdict = "SIGN CONSISTENTLY WRONG"
        print(f"[verdict] {verdict} — rerun with {'--flip-sign' if not args.flip_sign else 'NO --flip-sign'}.")
        rc = 1
    else:
        verdict = "NOISY / INCONCLUSIVE"
        print(f"[verdict] {verdict} — improve source placement (sustained, in FRONT, 30-60° off) or raise --step.")
        rc = 1
    log.write(
        "summary",
        valid=valid,
        improved=improved,
        worsened=worsened,
        mean_d_abs=round(mean_d, 4),
        final_sign_mult=sign_mult,
        verdict=verdict,
    )
    log.close()
    return rc


if __name__ == "__main__":
    sys.exit(main())
