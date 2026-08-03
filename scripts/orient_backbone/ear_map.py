#!/usr/bin/env python3
"""Ear map — DoA characterization sweep for shell A/B comparison.

The experiment (owner-designed, 2026-08-03): map the XVF3800's detection
and accuracy across azimuth × source height with the STANDARD shell, then
repeat with a modified shell (ear-like acoustic ports) and difference the
maps — does the shell shadow the array, and do ears fix it?

DESIGN DISCIPLINE (the week's lesson — kill actuation confounds first):

- REST-only: this script NEVER constructs the SDK, never touches motors,
  never commands motion. The robot is physically incapable of moving
  during a sweep, so per-cell bias/variance attributes to ACOUSTICS.
- Run AFTER the frame-thief fix (daemon automatic_body_yaw) is merged and
  the robot has been recentered + left untouched.
- The convention travels with the data (make-the-definition-travel): every
  output file carries the DoA convention, ground-truth sign convention,
  shell label, and the robot's hardware_id.

PROTOCOL (per sweep, ~30 min):

  1. Recenter the robot's head/body; do not touch it again.
  2. Fixed playback source (phone looping continuous speech at fixed
     volume), held at each grid cell ~1 m from the head:
     azimuth −90..+90 by 15° (robot's frame: − = its LEFT, + = its RIGHT)
     × heights: below (~30° under array plane), level, above (~30° over).
  3. The script prompts per cell; position the speaker, press Enter,
     it samples /api/state/doa for --cell-seconds, then advances.

  Run once per shell:
    python scripts/orient_backbone/ear_map.py --host 10.6.0.63 --shell standard
    python scripts/orient_backbone/ear_map.py --host 10.6.0.63 --shell eared-v1

  Compare:
    python scripts/orient_backbone/ear_map.py --analyze A.jsonl B.jsonl

METRICS per cell: speech-detection rate; mean measured azimuth; bias vs
ground truth (degrees); std (degrees). EXPECTATIONS to hold results
against: azimuth resolution compresses toward endfire (|az| > ~0.85
documented unreliable); front/back is structurally ambiguous (linear
array); ELEVATION IS NOT MEASURABLE by this chip — the height axis tests
whether source height DEGRADES the azimuth estimate, not elevation
detection (a planar array has no elevation to have holes in).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone

# Library reader (CI-enforced single HTTP surface); needs PYTHONPATH=<repo>/src.
from maxim.embodiment.audio_localization import doa_to_azimuth, make_reachy_rest_doa_reader

AZIMUTH_GRID_DEG = list(range(-90, 91, 15))
HEIGHTS = ("below", "level", "above")

CONVENTIONS = {
    "doa_rad": "XVF3800 via /api/state/doa: 0 = robot's LEFT, pi/2 = front, pi = robot's RIGHT (hardware-verified 2026-08-02)",
    "ground_truth_deg": "-90 = robot's full LEFT ... +90 = robot's full RIGHT (robot's own frame)",
    "azimuth_norm": "doa_to_azimuth: -1 = left ... +1 = right",
    "expected_doa_for_gt": "pi/2 + radians(gt_deg)",
    "height": "below/level/above ~ -30/0/+30 deg from the mic-array plane, source ~1 m from head",
}


def _daemon_identity(host: str) -> dict:
    try:
        from maxim.utils import http as maxim_http

        resp = maxim_http.fetch_url(f"http://{host}:8000/api/daemon/status", timeout=4.0)
        data = resp.json() if hasattr(resp, "json") else {}
        return {
            "hardware_id": data.get("hardware_id"),
            "daemon_version": data.get("version"),
            "motor_control_mode": (data.get("backend_status") or {}).get("motor_control_mode"),
        }
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}


def _sample_cell(reader, seconds: float) -> list[dict]:
    samples = []
    deadline = time.time() + seconds
    while time.time() < deadline:
        try:
            reading = reader()
        except Exception:  # noqa: BLE001
            reading = None
        if reading is not None:
            doa, speech = reading
            samples.append({"doa_rad": float(doa), "speech": bool(speech), "t": time.time()})
        time.sleep(0.15)
    return samples


def _cell_stats(samples: list[dict], gt_deg: float) -> dict:
    speech = [s for s in samples if s["speech"]]
    n = len(samples)
    stats = {
        "n_samples": n,
        "n_speech": len(speech),
        "detection_rate": (len(speech) / n) if n else 0.0,
    }
    if speech:
        az = [doa_to_azimuth(s["doa_rad"]) for s in speech]
        mean_az = sum(az) / len(az)
        mean_deg = mean_az * 90.0
        stats.update(
            {
                "mean_azimuth_norm": round(mean_az, 4),
                "mean_azimuth_deg": round(mean_deg, 2),
                "bias_deg": round(mean_deg - gt_deg, 2),
                "std_deg": round((sum((a * 90.0 - mean_deg) ** 2 for a in az) / len(az)) ** 0.5, 2),
            }
        )
    return stats


def run_sweep(args) -> int:
    reader = make_reachy_rest_doa_reader(args.host)
    out_path = (
        args.out
        or f"docs/experiments/data/ear_map_{args.shell}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.jsonl"
    )
    header = {
        "record": "header",
        "experiment": "ear_map",
        "shell": args.shell,
        "host": args.host,
        "cell_seconds": args.cell_seconds,
        "utc": datetime.now(timezone.utc).isoformat(),
        "conventions": CONVENTIONS,
        "robot": _daemon_identity(args.host),
        "note": "REST-only sweep; robot must remain untouched and recentered throughout.",
    }
    if header["robot"].get("motor_control_mode") not in (None, "disabled"):
        print(
            f"[warn] motor_control_mode={header['robot'].get('motor_control_mode')} — "
            "motors are live. Nothing here commands motion, but for a clean sweep "
            "consider a freshly rebooted (torque-off) daemon.",
        )

    cells = [(gt, h) for h in (args.heights or HEIGHTS) for gt in AZIMUTH_GRID_DEG]
    print(f"ear_map: shell={args.shell} — {len(cells)} cells × {args.cell_seconds}s → {out_path}")
    print("Recenter the robot now. Do not touch it again until the sweep ends.\n")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(header) + "\n")
        for i, (gt_deg, height) in enumerate(cells, 1):
            side = "LEFT" if gt_deg < 0 else ("RIGHT" if gt_deg > 0 else "FRONT")
            input(
                f"[{i}/{len(cells)}] Speaker at {abs(gt_deg)}° {side}, height={height} "
                f"(~1 m, speech playing) — Enter to sample..."
            )
            samples = _sample_cell(reader, args.cell_seconds)
            stats = _cell_stats(samples, float(gt_deg))
            record = {
                "record": "cell",
                "gt_deg": gt_deg,
                "height": height,
                **stats,
                "samples": samples,
            }
            f.write(json.dumps(record) + "\n")
            f.flush()
            print(
                f"    detection={stats['detection_rate']:.0%}"
                + (
                    f"  mean={stats.get('mean_azimuth_deg')}°  bias={stats.get('bias_deg')}°  std={stats.get('std_deg')}°"
                    if stats.get("n_speech")
                    else "  (no speech detected)"
                )
            )
    print(f"\nSweep complete → {out_path}")
    return 0


def _load(path: str):
    header, cells = None, {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("record") == "header":
                header = r
            elif r.get("record") == "cell":
                cells[(r["gt_deg"], r["height"])] = r
    return header, cells


def analyze(paths: list[str]) -> int:
    loaded = [(_load(p), p) for p in paths]
    for (header, cells), p in loaded:
        shell = (header or {}).get("shell", p)
        print(f"\n=== {shell} ({p}) ===")
        print(f"{'height':>6} | " + " ".join(f"{gt:>6}" for gt in AZIMUTH_GRID_DEG))
        for height in HEIGHTS:
            row_det = [cells.get((gt, height), {}).get("detection_rate") for gt in AZIMUTH_GRID_DEG]
            row_bias = [cells.get((gt, height), {}).get("bias_deg") for gt in AZIMUTH_GRID_DEG]
            print(
                f"{height:>6} | "
                + " ".join(f"{(d * 100):5.0f}%" if d is not None else "     -" for d in row_det)
                + "   [detection]"
            )
            print(
                f"{'':>6} | " + " ".join(f"{b:6.1f}" if b is not None else "     -" for b in row_bias) + "   [bias °]"
            )
    if len(loaded) == 2:
        (h_a, cells_a), pa = loaded[0]
        (h_b, cells_b), pb = loaded[1]
        print(f"\n=== DIFFERENCE ({(h_b or {}).get('shell', pb)} − {(h_a or {}).get('shell', pa)}) ===")
        for height in HEIGHTS:
            diffs = []
            for gt in AZIMUTH_GRID_DEG:
                a = cells_a.get((gt, height), {}).get("detection_rate")
                b = cells_b.get((gt, height), {}).get("detection_rate")
                diffs.append((b - a) * 100 if a is not None and b is not None else None)
            print(
                f"{height:>6} | "
                + " ".join(f"{d:+5.0f}%" if d is not None else "     -" for d in diffs)
                + "   [Δ detection]"
            )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="10.6.0.63")
    ap.add_argument("--shell", default="standard", help="Shell label (e.g. standard, eared-v1) — travels with the data")
    ap.add_argument("--cell-seconds", type=float, default=20.0)
    ap.add_argument("--heights", nargs="*", choices=HEIGHTS, help="Subset of heights (default: all three)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--analyze", nargs="+", metavar="JSONL", help="Analyze/compare sweep files instead of sweeping")
    args = ap.parse_args()
    if args.analyze:
        return analyze(args.analyze)
    return run_sweep(args)


if __name__ == "__main__":
    sys.exit(main())
