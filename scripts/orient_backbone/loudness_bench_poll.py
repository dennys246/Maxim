"""Loudness bench poller — item 18's two bench tests, over the daemon's REST API.

Polls three read-only endpoints on the Reachy Mini daemon at ~4 Hz and writes one
JSONL record per sample:

* ``GET /api/audio/config/parameter/AEC_SPENERGY_VALUES`` — the XVF3800's per-beam
  speech energy (focused 1, focused 2, free-running, auto-selected), measured in the
  AEC/beamformer stage, BEFORE the post-processing AGC.
* ``GET /api/audio/config/parameter/PP_AGCGAIN`` — the AGC's current multiplicative
  gain readback; it falls when the room gets loud and ramps back up in silence, so it
  is an inverse loudness envelope for free.
* ``GET /api/state/doa`` — the ``{angle, speech_detected}`` pair the DoA feed already
  reads; polled alongside to check the extra USB handle open/closes do not starve it.

No Maxim runtime is spawned and nothing is written to the robot — every request is a
read. Result + interpretation: ``docs/experiments/h2_loudness_bench_2026-08-25.md``.

Usage::

    python scripts/orient_backbone/loudness_bench_poll.py --host 10.6.0.63 \
        --duration 75 --out h2_loudness_bench.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

_PARAM = "/api/audio/config/parameter/"


def _get(base: str, path: str, timeout: float) -> tuple[dict, float]:
    t = time.monotonic()
    with urllib.request.urlopen(base + path, timeout=timeout) as resp:
        data = json.loads(resp.read())
    return data, time.monotonic() - t


def poll(base: str, duration: float, period: float, out: Path, stamp: dict | None = None) -> tuple[int, int]:
    n = err = 0
    t0 = time.time()
    with out.open("w", encoding="utf-8") as fh:
        print("RUNNING", flush=True)
        while time.time() - t0 < duration:
            rec: dict = {**(stamp or {}), "t": round(time.time() - t0, 2)}
            try:
                sp, l1 = _get(base, _PARAM + "AEC_SPENERGY_VALUES", 5.0)
                ag, l2 = _get(base, _PARAM + "PP_AGCGAIN", 5.0)
                doa, l3 = _get(base, "/api/state/doa", 5.0)
                rec.update(
                    spenergy=sp["values"],
                    agc_gain=ag["values"][0],
                    angle=doa["angle"] if doa else None,
                    speech=doa["speech_detected"] if doa else None,
                    lat_ms=[round(x * 1000) for x in (l1, l2, l3)],
                )
            except Exception as exc:  # noqa: BLE001 — recorded, not swallowed
                err += 1
                rec["error"] = repr(exc)
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            n += 1
            time.sleep(max(0.0, period - (time.time() - t0 - rec["t"])))
    return n, err


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--host", default="10.6.0.63")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--duration", type=float, default=75.0, help="seconds")
    ap.add_argument("--period", type=float, default=0.25, help="seconds per sample")
    ap.add_argument("--out", type=Path, default=Path("h2_loudness_bench.jsonl"))
    ap.add_argument(
        "--allow-dirty",
        action="store_true",
        help="write a GATED record (docs/experiments/data/) from a dirty src/scripts tree; stamps allow_dirty: true "
        "into every record (default: refuse, exit 3 — docs/lessons/experiment-prereg-precedes-data.md)",
    )
    args = ap.parse_args(argv)
    # scripts/_provenance.py by path (stdlib-only): a bench record written under
    # docs/experiments/data/ from a dirty src/scripts tree is refused (exit 3) unless
    # --allow-dirty, which stamps allow_dirty: true into every record (item 16.7).
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root / "scripts"))
    import _provenance

    gate = _provenance.preflight_gated_record_or_exit(repo_root, args.out, allow_dirty=args.allow_dirty)
    stamp = {"allow_dirty": True} if gate["allow_dirty"] else {}
    n, err = poll(f"http://{args.host}:{args.port}", args.duration, args.period, args.out, stamp)
    print(f"DONE samples={n} errors={err} -> {args.out}", flush=True)
    return 0 if err == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
