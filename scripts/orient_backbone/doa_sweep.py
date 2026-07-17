#!/usr/bin/env python
"""DoA static response sweep — characterize the bearing sensor, no learning.

Motivated by the s1 perturb-run forensics (2026-07-16): the XVF3800 DoA is
NOT the linear device the orient loop modeled. Observed: large base rotations
(±0.7-0.9 rad) measured ~0.1 azimuth change while 0.25 rad steps track at
gain ~0.3-0.4 (vs geometric 0.64, vs Step 2's apparent 2-3x); intended far-bin
placements all landed near; and a REPRODUCIBLE anti-physical zone at the
+1.40 rad clamp (correct turn made +az worse, identical values, 3/3 times).

This sweep maps measured azimuth vs commanded base pose directly:
step body_yaw across [min,max] in fixed increments, ascending then descending
(hysteresis check), taking several speech-gated reads per pose. Output answers:
true gain curve + linear region, saturation points, the clamp-zone anomaly,
front/back-mirror effects, per-pose gate rate + read noise.

Also the A/B instrument for HARDWARE acoustic changes (e.g. the pinnae/"ears"
shell-mod idea): run a sweep before and after any shell change and diff the
curves — same source placement, same protocol.

OPERATOR PROTOCOL:
    ~/Envs/maxim-env/bin/python scripts/orient_backbone/doa_sweep.py --host <ip>
  Sustained speech source ~1-2 m away, directly in FRONT of the base's NEUTRAL
  heading (az should read ~0 at pose 0). Do not move it during the sweep.
  ~29 poses x 2 passes x ~2 s/pose ~= 2-3 minutes. Robot turns through its
  full base range — clear the desk.

Offline logic check: --dry-run (linear world, stationary source) — expect
slope ~0.64 in the central region, clean saturation at |az|=1, no hysteresis.

Findings feed: the Step-3 placement targets/limits (set from the measured
curve), docs/embodiment/porting_orient_loop.md calibration step, and the
runbook's calibration-unknowns table.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time

from live_common import DryRig, JsonlLog, LiveRig, doa_to_azimuth, preflight, resolve_host


def collect_pose_reads(reader, *, reads: int, poll_s: float, timeout_s: float = 8.0) -> tuple[list[float], int]:
    """Up to `reads` speech-gated azimuth samples at the current pose.

    Returns (gated_samples, total_attempts). Individual raw reads (not the
    median wrapper) — the sweep wants per-pose noise, not smoothing.
    """
    samples: list[float] = []
    attempts = 0
    deadline = time.time() + timeout_s
    while len(samples) < reads and time.time() < deadline:
        attempts += 1
        try:
            reading = reader()
        except Exception:  # noqa: BLE001
            reading = None
        if reading is not None:
            doa_rad, is_speech = reading
            if is_speech:
                samples.append(doa_to_azimuth(float(doa_rad)))
        time.sleep(poll_s)
    return samples, attempts


def fit_central_slope(points: list[tuple[float, float]], psi_max: float = 0.5) -> float | None:
    """Least-squares slope d(az)/d(psi) over the central |psi| <= psi_max region."""
    pts = [(p, a) for p, a in points if abs(p) <= psi_max]
    if len(pts) < 3:
        return None
    n = len(pts)
    sx = sum(p for p, _ in pts)
    sy = sum(a for _, a in pts)
    sxx = sum(p * p for p, _ in pts)
    sxy = sum(p * a for p, a in pts)
    denom = n * sxx - sx * sx
    if abs(denom) < 1e-12:
        return None
    return (n * sxy - sx * sy) / denom


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default=None, help="Reachy daemon IP (or $MAXIM_REACHY_HOST)")
    ap.add_argument("--min-yaw", type=float, default=-1.4)
    ap.add_argument("--max-yaw", type=float, default=1.4)
    ap.add_argument("--step", type=float, default=0.1, help="pose increment (rad)")
    ap.add_argument("--reads", type=int, default=5, help="gated samples per pose")
    ap.add_argument("--duration", type=float, default=0.5, help="goto_target duration (s)")
    ap.add_argument("--settle", type=float, default=0.8, help="extra settle before reads (s)")
    ap.add_argument("--label", default="baseline", help="sweep label (e.g. baseline / eared-shell-v1)")
    ap.add_argument("--log", default="/tmp/doa_sweep.jsonl")
    ap.add_argument("--yes", action="store_true", help="skip the source-placement confirm prompt")
    ap.add_argument("--dry-run", action="store_true", help="offline logic check (no robot)")
    args = ap.parse_args()

    log = JsonlLog(args.log)
    poll_s = 0.0 if args.dry_run else 0.15

    def pace(seconds: float) -> None:
        if not args.dry_run:
            time.sleep(seconds)

    if args.dry_run:
        rig = DryRig(theta_src=-0.7, jump_prob=0.0)
        print(f"[dry] stationary source at world bearing {rig.theta_src:+.2f} rad")
    else:
        host, source = resolve_host(args.host)
        if host is None:
            print("[FAIL] no robot address: --host <ip> or export MAXIM_REACHY_HOST=<ip>")
            log.close()
            return 2
        print(f"[host] using {host} (source: {source})")
        preflight(host)
        rig = LiveRig(host)
        rig.recenter()
        if not args.yes:
            print("\n[setup] Sustained speech source directly in FRONT of the base's neutral heading,")
            print("        ~1-2 m away. It must NOT move during the sweep (~3 min).")
            ans = input("        Source placed and playing? [y/N] ").strip().lower()
            if ans not in ("y", "yes"):
                print("[abort] place the source, then rerun.")
                log.write("abort", reason="source_not_placed")
                log.close()
                return 1

    n_poses = int(round((args.max_yaw - args.min_yaw) / args.step)) + 1
    ascending = [round(args.min_yaw + i * args.step, 3) for i in range(n_poses)]
    passes = [("ascending", ascending), ("descending", list(reversed(ascending)))]
    log.write(
        "sweep_start",
        label=args.label,
        min_yaw=args.min_yaw,
        max_yaw=args.max_yaw,
        step=args.step,
        reads=args.reads,
        settle=args.settle,
        dry_run=args.dry_run,
    )

    results: dict[str, list[tuple[float, float]]] = {}
    for pass_name, poses in passes:
        print(f"\n[{pass_name}] {len(poses)} poses, {args.reads} gated reads each")
        print(f"      {'psi':>6}  {'n':>4}  {'median':>7}  {'min':>6}  {'max':>6}  {'gate%':>5}")
        pass_points: list[tuple[float, float]] = []
        for psi in poses:
            rig.goto_body_yaw(psi, duration=args.duration)
            pace(args.duration + args.settle)
            samples, attempts = collect_pose_reads(rig.reader, reads=args.reads, poll_s=poll_s)
            if samples:
                med = statistics.median(samples)
                lo, hi = min(samples), max(samples)
                gate = len(samples) / attempts if attempts else 0.0
                pass_points.append((psi, med))
                print(f"      {psi:+.2f}  {len(samples):>4}  {med:+.3f}  {lo:+.3f}  {hi:+.3f}  {gate:5.0%}")
            else:
                med = lo = hi = None
                gate = 0.0
                print(f"      {psi:+.2f}  {0:>4}     (no gated readings)")
            log.write(
                "sweep_point",
                label=args.label,
                sweep_pass=pass_name,
                psi=psi,
                n=len(samples),
                attempts=attempts,
                az_median=None if med is None else round(med, 4),
                az_min=None if lo is None else round(lo, 4),
                az_max=None if hi is None else round(hi, 4),
                samples=[round(s, 4) for s in samples],
            )
        results[pass_name] = pass_points

    if not args.dry_run:
        rig.recenter()

    # --- analysis ---
    print("\n[analysis]")
    for pass_name, pts in results.items():
        slope = fit_central_slope(pts)
        print(
            f"  {pass_name}: central gain (|psi|<=0.5) = {slope:+.3f}/rad"
            if slope is not None
            else f"  {pass_name}: too few central points to fit"
        )
        # non-monotonic zones: where az moves OPPOSITE to psi between adjacent poses
        flips = [(p1, a1, p2, a2) for (p1, a1), (p2, a2) in zip(pts, pts[1:]) if (a2 - a1) * (p2 - p1) < -0.03]
        for p1, a1, p2, a2 in flips:
            print(f"      NON-MONOTONIC: psi {p1:+.2f}->{p2:+.2f} but az {a1:+.3f}->{a2:+.3f}")
    asc = dict(results.get("ascending", []))
    desc = dict(results.get("descending", []))
    common = sorted(set(asc) & set(desc))
    if common:
        hys = [abs(asc[p] - desc[p]) for p in common]
        worst = max(common, key=lambda p: abs(asc[p] - desc[p]))
        print(
            f"  hysteresis: mean |asc-desc| = {statistics.mean(hys):.3f}, worst {abs(asc[worst] - desc[worst]):.3f} at psi {worst:+.2f}"
        )
    geo = 2.0 / 3.141592653589793
    print(f"  (geometric prediction: {geo:+.3f}/rad; s1 apparatus EMA landed ~0.3-0.4)")
    log.write("sweep_done", label=args.label)
    log.close()
    print(f"\n[done] full data in {args.log} (label={args.label}) — send it back for curve analysis.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
