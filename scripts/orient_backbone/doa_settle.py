#!/usr/bin/env python
"""DoA settle curve — is the azimuth estimate a FILTER, not a function of pose?

The properly-powered version of a test I already got wrong once. Exp 45b's
gain gap (sweep 0.605 az/rad vs learner trials 0.39, an identical ~65%
shortfall at BOTH 0.3 and 0.9 rad steps) is the signature of a running-average
estimator: after a step of D, the reading reaches D*f(t) where f depends on
elapsed time ONLY — so the shortfall is proportional and step-size-independent,
exactly as measured. (Backlash would hit the small step harder; settle-lag of a
size-dependent kind would hit the big step harder. Neither matches.)

My first attempt to test this looked for within-pose drift in the SWEEP's
samples and found none — but the sweep steps only 0.1 rad, so the expected
drift (~0.009 az) sits BELOW the chip's ~0.011 quantization. The test could not
detect what it was testing for. This script fixes that: it uses a LARGE step so
the drift is far above quantization, and traces az against time.

Second thing this measures, per the operator's observation that the source is
NOT uniform noise: the chip integrates over *speech energy*, not wall-clock —
so convergence should depend on how much speech actually arrived since the
move. The trace records the speech gate per sample, so a sparse-speech stretch
is visible as a stall in the curve rather than a mystery.

    ~/Envs/maxim-env/bin/python scripts/orient_backbone/doa_settle.py --host <ip>

Needs a sustained speech source in front (as usual). ~2 minutes, no learning.

READ: if az climbs past the learner's read time (~1.25 s) and plateaus later,
the fix is in the READER — poll until the estimate stops changing — not in the
credit, the metric, or the substrate.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time

from live_common import LiveRig, doa_to_azimuth, preflight, resolve_host

LEARNER_READ_T = 1.25  # when live_3_learn currently takes its median-of-3 (s)
SWEEP_READ_T = 1.70  # when doa_sweep takes its median-of-5 (s)


def settled_read(rig, *, tol: float = 0.015, window: int = 4, max_wait: float = 6.0) -> float | None:
    """Poll until the gated estimate stops changing — the reference 'true' az."""
    samples: list[float] = []
    deadline = time.time() + max_wait
    while time.time() < deadline:
        r = rig.reader()
        if r is not None:
            doa, speech = r
            if speech:
                samples.append(doa_to_azimuth(float(doa)))
                if len(samples) >= window and (max(samples[-window:]) - min(samples[-window:])) <= tol:
                    return statistics.median(samples[-window:])
        time.sleep(0.15)
    return statistics.median(samples[-window:]) if len(samples) >= window else (samples[-1] if samples else None)


def trace_step(rig, target_yaw: float, *, duration: float, max_t: float, poll_s: float):
    """Command an absolute yaw; trace (t, az, gated) from the moment motion ends."""
    rig.goto_body_yaw(target_yaw, duration=duration)
    t0 = time.time()
    trace: list[tuple[float, float | None, bool]] = []
    while (t := time.time() - t0) < max_t:
        r = rig.reader()
        if r is None:
            trace.append((t, None, False))
        else:
            doa, speech = r
            trace.append((t, doa_to_azimuth(float(doa)) if speech else None, bool(speech)))
        time.sleep(poll_s)
    return trace


def at_time(trace, t_target: float) -> float | None:
    """The gated reading nearest t_target (what a reader firing at t_target would see)."""
    gated = [(t, az) for t, az, g in trace if g and az is not None]
    if not gated:
        return None
    return min(gated, key=lambda p: abs(p[0] - t_target))[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default=None, help="Reachy daemon IP (or $MAXIM_REACHY_HOST)")
    ap.add_argument("--step", type=float, default=0.9, help="step size to trace (rad) — big, so drift >> quantization")
    ap.add_argument("--reps", type=int, default=4, help="traces (alternating direction)")
    ap.add_argument("--max-t", type=float, default=5.0, help="trace duration after motion end (s)")
    ap.add_argument("--poll-s", type=float, default=0.2, help="sample interval (s)")
    ap.add_argument("--duration", type=float, default=0.6, help="goto_target duration (s)")
    ap.add_argument("--log", default="/tmp/doa_settle.jsonl")
    args = ap.parse_args()

    host, source = resolve_host(args.host)
    if host is None:
        print("[FAIL] no robot address: --host <ip> or export MAXIM_REACHY_HOST=<ip>")
        return 2
    print(f"[host] using {host} (source: {source})")
    preflight(host)

    from live_common import JsonlLog

    log = JsonlLog(args.log)
    rig = LiveRig(host)
    rig.recenter()

    print("\n[setup] sustained speech source in FRONT, ~1-2 m, not moving.")
    print(f"[trace] {args.reps} x {args.step:+.2f} rad steps, sampling every {args.poll_s}s for {args.max_t}s\n")

    fracs_learner: list[float] = []
    fracs_sweep: list[float] = []
    gains_plateau: list[float] = []

    for rep in range(args.reps):
        rig.recenter()
        time.sleep(0.5)
        az0 = settled_read(rig)
        if az0 is None:
            print(f"  rep {rep}: no settled baseline (speech gate?) — skipping")
            continue
        target = args.step if rep % 2 == 0 else -args.step
        trace = trace_step(rig, target, duration=args.duration, max_t=args.max_t, poll_s=args.poll_s)
        gated = [(t, az) for t, az, g in trace if g and az is not None]
        if len(gated) < 4:
            print(f"  rep {rep}: only {len(gated)} gated samples — speech too sparse, skipping")
            continue
        plateau = statistics.median([az for t, az in gated if t >= args.max_t - 1.0] or [gated[-1][1]])
        d_true = plateau - az0
        gate_rate = sum(1 for _, _, g in trace if g) / len(trace)
        print(f"  rep {rep}: yaw {target:+.2f}  az0={az0:+.3f} -> plateau={plateau:+.3f}  (gate {gate_rate:.0%})")
        print(f"          {'t(s)':>5} {'az':>7} {'frac of plateau shift':>22}")
        for t, az in gated:
            frac = (az - az0) / d_true if abs(d_true) > 1e-6 else float("nan")
            bar = "#" * max(0, min(30, int(frac * 30)))
            print(f"          {t:5.2f} {az:+7.3f} {frac:6.2f}  {bar}")
        az_l, az_s = at_time(trace, LEARNER_READ_T), at_time(trace, SWEEP_READ_T)
        if abs(d_true) > 1e-6 and az_l is not None and az_s is not None:
            fl, fs = (az_l - az0) / d_true, (az_s - az0) / d_true
            fracs_learner.append(fl)
            fracs_sweep.append(fs)
            gains_plateau.append(abs(d_true / target))
            print(
                f"          -> at learner read t={LEARNER_READ_T}s: {fl:.2f} of final | "
                f"at sweep read t={SWEEP_READ_T}s: {fs:.2f} | plateau gain={abs(d_true / target):.3f}/rad"
            )
        log.write(
            "settle_trace",
            rep=rep,
            target=target,
            az0=round(az0, 4),
            plateau=round(plateau, 4),
            gate_rate=round(gate_rate, 3),
            trace=[(round(t, 2), az, g) for t, az, g in trace],
        )
        print()

    print("[analysis]")
    if fracs_learner:
        fl, fs = statistics.mean(fracs_learner), statistics.mean(fracs_sweep)
        gp = statistics.mean(gains_plateau)
        print(f"  converged at learner read ({LEARNER_READ_T}s): {fl:.2f}   at sweep read ({SWEEP_READ_T}s): {fs:.2f}")
        print(f"  plateau gain: {gp:.3f}/rad   (sweep measured 0.605; learner trials measured 0.39)")
        print(f"  predicted learner gain if lag is the cause: {gp * fl:.3f}/rad  <- compare to 0.39")
        # HISTORY (2026-07-16): this script was written to test a "the DoA is a slow
        # tracking estimator" theory. It is not — the estimate settles in ~0.23s. The
        # apparent lag was the head=None counter-rotation bug (mics weren't turning).
        # Kept as the honest gain/settle characterizer. NOTE the metric below compares
        # each reading to its OWN plateau, so a frozen-but-WRONG estimate scores 1.00 —
        # always read plateau_gain against an independent expectation, not just `frac`.
        if fl < 0.9:
            print("  => the estimate is still converging at the learner's read time — raise --settle.")
        else:
            print("  => settled by the learner's read time (expected: ~0.23s post-headfix).")
        if gp < 0.45:
            print(f"  => LOW GAIN ({gp:.3f}/rad vs ~0.56 expected). Prime suspect: the head is NOT")
            print("     riding along with the body, so the mics aren't rotating. Run yaw_verify.py —")
            print("     d(head)/d(body) must be ~+1.0. See CLAUDE.md's Reachy head-frame invariant.")
    else:
        print("  no usable traces — speech source too sparse. Use continuous speech (podcast) and retry.")
    log.close()
    print(f"\n[done] full traces in {args.log}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
