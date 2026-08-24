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
Dry-run REJECTS on full-range admission by design: its saturation makes the
full-range fit non-linear (R2 ~0.98). That is the rig, not a failure.

Every record carries a run_id unique to the invocation; group by run_id, NOT by
--label (labels get reused across re-runs). Only a run whose log ends in
sweep_done is complete — sweep_aborted marks a killed run, exclude it.

Findings feed: the Step-3 placement targets/limits (set from the measured
curve), docs/embodiment/porting_orient_loop.md calibration step, and the
runbook's calibration-unknowns table.
"""

from __future__ import annotations

import argparse
import json
import os
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


def labels_in_log(path: str) -> set[str]:
    """Labels already present in an append-only sweep log (may be empty/absent)."""
    seen: set[str] = set()
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                label = rec.get("label")
                if isinstance(label, str):
                    seen.add(label)
    except OSError:
        pass
    return seen


# L9 (2026-08-23): score the FULL-RANGE fit, never the central one. The gate
# band [0.52, 0.62] around 0.578 was derived from full-range fits, but this
# script used to print only central gain (|psi| <= 0.5, ~11 points over a short
# lever arm) with no R² — a units mismatch that made a healthy instrument read
# as un-scoreable. Across four sessions the admitted full-range fits span 0.013;
# the same curves scored centrally span 0.086. See docs/limits/README.md L9.
ADMIT_R2 = 0.99
ADMIT_N = 25
# The H2 gate band, in CODE rather than only in prose. It lives in three docs
# with three phrasings (the L9 entry, the heartbeat runbook, the H1
# pre-registration) and the operator used to read a number off a terminal and
# compare it by eye — which is the operation that failed and cost a session.
# Provenance: NOT a fit statistic. The H1 pre-registration derives it from a
# +/-0.03-az tolerance on the |az| ~ 0.33 big-step boundary around gain
# 0.55-0.57, mapped through boundary ~ 1/g.
H2_BAND = (0.52, 0.62)


def fit_line(points: list[tuple[float, float]], psi_max: float | None = None) -> tuple[float, float, int] | None:
    """Least-squares d(az)/d(psi) -> (slope, r_squared, n_points).

    psi_max restricts to |psi| <= psi_max (the central region); None fits the
    full swept range, which is the statistic the H2 gate band was built from.
    """
    pts = [(p, a) for p, a in points if psi_max is None or abs(p) <= psi_max]
    n = len(pts)
    if n < 3:
        return None
    mean_p = sum(p for p, _ in pts) / n
    mean_a = sum(a for _, a in pts) / n
    sxx = sum((p - mean_p) ** 2 for p, _ in pts)
    if sxx < 1e-12:
        return None
    slope = sum((p - mean_p) * (a - mean_a) for p, a in pts) / sxx
    intercept = mean_a - slope * mean_p
    ss_tot = sum((a - mean_a) ** 2 for _, a in pts)
    ss_res = sum((a - (slope * p + intercept)) ** 2 for p, a in pts)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")
    return slope, r2, n


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

    # Records are keyed by run_id, NOT by --label. The log is append-only and
    # operators reuse labels across re-runs: on 2026-08-23 an aborted attempt
    # (19% yield, 59% outlier samples) and its clean re-run both wrote under
    # 'heartbeat-1.1-2026-08-23-run2', so any consumer filtering on label
    # silently merged garbage into the analysed set. run_id is unique per
    # invocation; only a run that reaches sweep_done is complete.
    run_id = f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-{os.getpid()}"
    log = JsonlLog(args.log)

    written = {"sweep_point": 0}

    def emit(event: str, **fields: object) -> None:
        # dry_run rides on EVERY record, not just sweep_start: a consumer reading
        # sweep_points could not otherwise tell synthetic from measured without a
        # timestamp join, and docs/experiments/data/45_doa_sweep_baseline.jsonl is
        # half dry-run data with nothing in its point records saying so.
        written[event] = written.get(event, 0) + 1
        log.write(event, run_id=run_id, label=args.label, dry_run=args.dry_run, **fields)

    if args.label in labels_in_log(args.log):
        print(
            f"[warn] label {args.label!r} already appears in {args.log}; this run is"
            f" run_id={run_id} — group by run_id, not label, when analysing."
        )
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
                emit("abort", reason="source_not_placed")
                log.close()
                return 1

    n_poses = int(round((args.max_yaw - args.min_yaw) / args.step)) + 1
    ascending = [round(args.min_yaw + i * args.step, 3) for i in range(n_poses)]
    passes = [("ascending", ascending), ("descending", list(reversed(ascending)))]
    emit(
        "sweep_start",
        min_yaw=args.min_yaw,
        max_yaw=args.max_yaw,
        step=args.step,
        reads=args.reads,
        settle=args.settle,
    )

    results: dict[str, list[tuple[float, float]]] = {}
    try:
        run_passes(
            passes,
            rig=rig,
            args=args,
            emit=emit,
            pace=pace,
            poll_s=poll_s,
            results=results,
        )
    except BaseException as exc:  # KeyboardInterrupt included — the common case
        points_written = written["sweep_point"]
        if not args.dry_run:
            try:
                rig.recenter()  # do not leave the body parked at the last yaw
            except Exception as recenter_err:
                print(f"[warn] recenter after abort failed: {recenter_err}")
        emit("sweep_aborted", reason=type(exc).__name__, points_written=points_written)
        log.close()
        print(
            f"\n[aborted] {type(exc).__name__} after {points_written} points."
            f" run_id={run_id} is marked sweep_aborted — exclude it from analysis."
        )
        raise

    if not args.dry_run:
        rig.recenter()

    verdict = _analyse(results)
    emit("sweep_done", **verdict)
    log.close()
    print(f"\n[done] full data in {args.log} (run_id={run_id}, label={args.label}) — send it back for curve analysis.")
    return 0


def run_passes(passes, *, rig, args, emit, pace, poll_s, results) -> None:
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
            emit(
                "sweep_point",
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


def _analyse(results: dict[str, list[tuple[float, float]]]) -> dict[str, object]:
    print("\n[analysis]")
    admitted: list[float] = []
    for pass_name, pts in results.items():
        full = fit_line(pts)
        central = fit_line(pts, psi_max=0.5)
        if full is None:
            print(f"  {pass_name}: too few points to fit")
        else:
            slope, r2, n = full
            ok = r2 >= ADMIT_R2 and n >= ADMIT_N
            if ok:
                admitted.append(slope)
            verdict = "ADMIT" if ok else f"REJECT (needs R2>={ADMIT_R2}, n>={ADMIT_N})"
            print(f"  {pass_name}: full-range gain = {slope:+.3f}/rad  R2={r2:.4f}  n={n}  -> {verdict}")
            if central is not None:
                print(
                    f"      (central |psi|<=0.5 = {central[0]:+.3f}, R2={central[1]:.4f} — NOT the gate statistic, see L9)"
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
    if admitted:
        lo, hi = min(admitted), max(admitted)
        print(
            f"  ADMITTED: {len(admitted)} pass(es), gain {lo:+.3f}..{hi:+.3f}"
            f" (spread {hi - lo:.3f}, mean {sum(admitted) / len(admitted):+.3f})"
        )
        if len(admitted) < 2:
            print("      WARNING: <2 admitted passes — sign-flip contamination rejects ~half;")
            print("               re-run for more passes before scoring a gate (L9).")
    else:
        print("  ADMITTED: none — no pass met the admission criterion; this sweep is not scoreable.")
    geo = 2.0 / 3.141592653589793
    print(f"  (geometric prediction: {geo:+.3f}/rad; s1 apparatus EMA landed ~0.3-0.4)")

    # Score the gate here, and stamp it into the log, so no downstream consumer
    # has to re-fit (re-fitting by --label is what produced a wrong rejection
    # list in the first draft of L9).
    lo, hi = H2_BAND
    if not admitted:
        h2 = "UNSCOREABLE"
    elif len(admitted) < 2:
        h2 = "PROVISIONAL"
    else:
        h2 = "PASS" if all(lo <= g <= hi for g in admitted) else "FIRES"
    print(f"  H2 [{lo}, {hi}]: {h2}  ({len(admitted)} admitted pass(es))")
    if h2 == "PROVISIONAL":
        print("      one admitted pass is not enough to score H2 — re-run for more (L9).")
    if h2 == "FIRES":
        print("      at least one admitted gain is outside the band — H2 fires.")
    return {
        "admitted_gains": [round(g, 4) for g in admitted],
        "admitted_n": len(admitted),
        "admit_r2": ADMIT_R2,
        "admit_min_points": ADMIT_N,
        "h2_band": list(H2_BAND),
        "h2": h2,
    }


if __name__ == "__main__":
    sys.exit(main())
