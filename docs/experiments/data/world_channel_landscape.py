"""The world channel's similarity LANDSCAPE — shape only, measured on the shipped encoder.

One question, and only one: **as the body's world sensors vary across their DECLARED ranges, is the
distribution of pairwise cosine continuous, or does it sit in two lumps with nothing in between?**

That decides whether a graded read at the cluster boundary (Exp 62 §Rung B) has anything to grade.
If situation pairs are either decisively the same or decisively different, grading buys nothing and
Rung B should be closed. If a real fraction land near the threshold, there is a middle to read.

WHAT THIS CANNOT SAY — read before citing it
--------------------------------------------
This measures the SHAPE of the function, not the SUPPORT of the world. A continuous landscape is
irrelevant if the world only ever visits two corners of it. The only committed open-world trace
(`l11_world_trace_2026-09-04.jsonl`, 1,193 snapshots) has `light_level` at 0.0 in 1193/1193 and
`time_of_day` pinned in 1193/1193, so it cannot answer support on the axes that matter either. A
graded read is worth building only where continuous shape AND real support overlap; this script is
half of that and must never be quoted as the whole.

It also cannot be answered by the water classroom at any n: that apparatus has ONE discriminating
world sensor (`exp60_geometry_2026-09-15b.json` → `live_contributors: ["is_in_water"]`, a binary
0.5→1.0 flip), so its situation space is two points and its shape is a cliff BY CONSTRUCTION.
Measuring it there and reporting a body property is the confound this file exists to avoid.

WHY REAL SENSORS, NEVER SYNTHETIC ONES
--------------------------------------
Every value comes from `minecraft_player.yaml`'s declared roster and `range:` — the real names, so
the SHA bases are the shipped ones, and the real bounds, so every sampled point is a state the body
can actually report. Invented filler sensors are how this project has been misled before: the
sensor-count-scaled threshold scored 100% on synthetic data (L11, 2026-09-01) and was superseded by
the bake-off at 0.70–0.84, and an arbitrary filler NAME moves a single-sensor answer across the
threshold 52% of the time. Points are never reported without their spread.

Run:  PYTHONPATH=src python docs/experiments/data/world_channel_landscape.py axes
      PYTHONPATH=src python docs/experiments/data/world_channel_landscape.py landscape --n 400
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, "src")

import yaml  # noqa: E402

from maxim.similarity.encoder import SensorEncoderConfig, _sensor_embed  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
BODY = REPO / "src" / "maxim" / "_data" / "components" / "bodies" / "minecraft_player.yaml"
CFG = SensorEncoderConfig()
THRESHOLD = CFG.pattern_threshold
GAIN = CFG.gain_exponent
# The classroom's observed values for the two sensors that declare NO rest — used only as the
# hold-fixed value in the single-axis sweep, and flagged in the output so it is never mistaken
# for a neutral. A declared rest the world never rests at is a constant, not a neutral (L11).
NO_REST_HOLD = {"light_level": 0.0, "time_of_day": 1000.0 / 24000.0}


def roster() -> dict[str, dict]:
    """Every world sensor the body declares, with its range and rest. The anchor for all of this."""
    spec = yaml.safe_load(BODY.read_text())

    def find(o):
        if isinstance(o, dict):
            if "sensors" in o and isinstance(o["sensors"], dict):
                return o["sensors"]
            for v in o.values():
                got = find(v)
                if got:
                    return got
        elif isinstance(o, list):
            for x in o:
                got = find(x)
                if got:
                    return got
        return None

    sensors = find(spec) or {}
    return {n: s for n, s in sensors.items() if s.get("modality") == "world" and s.get("range")}


def hold_value(name: str, spec: dict) -> float:
    """Where this sensor sits when it is not the one being varied."""
    if spec.get("rest") is not None:
        return float(spec["rest"])
    if name in NO_REST_HOLD:
        return NO_REST_HOLD[name]
    lo, hi = spec["range"]
    return (float(lo) + float(hi)) / 2.0


def cos(a: list[float], b: list[float]) -> float:
    d = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    return d / (na * nb) if na and nb else 0.0


def embed(state: dict[str, float], ranges: dict[str, tuple[float, float]]) -> list[float]:
    """The SHIPPED encoder path — not a hand copy, so a change to the gain law shows up here."""
    return _sensor_embed(state, ranges=ranges, dim=CFG.embedding_dim, gain_exponent=GAIN)


def cmd_axes(args: argparse.Namespace) -> int:
    """One sensor at a time: how far can it move the vector on its own, across its OWN range?"""
    rs = roster()
    ranges = {n: (float(s["range"][0]), float(s["range"][1])) for n, s in rs.items()}
    base = {n: hold_value(n, s) for n, s in rs.items()}
    base_vec = embed(base, ranges)
    print(f"world sensors: {len(rs)}   gain exponent {GAIN}   threshold {THRESHOLD}")
    print(f"{'sensor':24s} {'min cos vs hold':>16s} {'crosses thr?':>13s}   rest")
    rows = []
    for name, spec in sorted(rs.items()):
        lo, hi = ranges[name]
        worst, worst_at = 1.0, None
        for i in range(41):
            v = lo + (hi - lo) * i / 40.0
            c = cos(base_vec, embed({**base, name: v}, ranges))
            if c < worst:
                worst, worst_at = c, v
        crosses = worst < THRESHOLD
        if spec.get("rest") is not None:
            rest = f"declared {spec['rest']}"
        elif name in NO_REST_HOLD:
            rest = f"NO rest declared — held at the classroom's constant ({NO_REST_HOLD[name]})"
        else:
            rest = f"no rest key — held at the range midpoint ({(ranges[name][0] + ranges[name][1]) / 2:g})"
        rows.append({"sensor": name, "min_cos": round(worst, 4), "at_value": worst_at, "crosses": crosses})
        print(f"{name:24s} {worst:16.4f} {('YES' if crosses else 'no'):>13s}   {rest}")
    movers = [r for r in rows if r["crosses"]]
    print(f"\n{len(movers)} of {len(rows)} sensors can cross the threshold ALONE: {[r['sensor'] for r in movers]}")
    if args.json:
        Path(args.json).write_text(json.dumps({"threshold": THRESHOLD, "gain": GAIN, "axes": rows}, indent=2) + "\n")
    return 0


def cmd_landscape(args: argparse.Namespace) -> int:
    """The distribution: sample states within the DECLARED ranges, histogram the pairwise cosines."""
    rs = roster()
    ranges = {n: (float(s["range"][0]), float(s["range"][1])) for n, s in rs.items()}
    rng = random.Random(args.seed)
    states = [{n: rng.uniform(*ranges[n]) for n in rs} for _ in range(args.n)]
    vecs = [embed(s, ranges) for s in states]
    sims = [cos(vecs[i], vecs[j]) for i in range(len(vecs)) for j in range(i + 1, len(vecs))]
    sims.sort()

    band = args.band
    near = [c for c in sims if abs(c - THRESHOLD) <= band]
    print(f"{args.n} sampled states (seed {args.seed}) → {len(sims)} pairs")
    print(f"gain {GAIN}, threshold {THRESHOLD}, band ±{band}\n")
    print("cosine histogram:")
    edges = [i / 10.0 for i in range(11)]
    for lo, hi in zip(edges, edges[1:]):
        k = sum(1 for c in sims if lo <= c < hi)
        bar = "#" * int(60 * k / max(1, len(sims)))
        mark = "  <- threshold" if lo <= THRESHOLD < hi else ""
        print(f"  [{lo:.1f},{hi:.1f})  {k:6d}  {bar}{mark}")
    print()
    print(f"median {statistics.median(sims):.4f}   mean {statistics.fmean(sims):.4f}")
    print(f"above threshold: {sum(1 for c in sims if c >= THRESHOLD) / len(sims) * 100:.1f}%")
    print(f"IN THE BAND (±{band} of {THRESHOLD}): {len(near)} pairs = {len(near) / len(sims) * 100:.2f}%")
    print()
    # The decision-relevant read, stated so a null is as legible as a positive.
    pct = len(near) / len(sims) * 100
    print(
        "READING: a graded read has something to grade here"
        if pct >= args.grade_floor
        else f"READING: <{args.grade_floor}% of pairs land near the threshold — decisively same or "
        "decisively different, so a graded read has little to grade ON THIS ROSTER"
    )
    print("  (SHAPE only — says nothing about which regions the world actually visits.)")
    if args.json:
        Path(args.json).write_text(
            json.dumps(
                {
                    "n_states": args.n,
                    "seed": args.seed,
                    "threshold": THRESHOLD,
                    "gain": GAIN,
                    "band": band,
                    "pairs": len(sims),
                    "median": statistics.median(sims),
                    "pct_above_threshold": sum(1 for c in sims if c >= THRESHOLD) / len(sims) * 100,
                    "pct_in_band": pct,
                    "what_this_cannot_say": "shape only; not the world's support (see module docstring)",
                },
                indent=2,
            )
            + "\n"
        )
    return 0


def cmd_neighbourhood(args: argparse.Namespace) -> int:
    """The measurement that actually decides it: how cosine falls as a state moves AWAY from rest.

    `landscape` samples uniformly over the declared box, which makes every state maximally unlike
    every other (median ≈ 0.24) — it answers "are arbitrary states different", which is not the
    question and was a flaw in the first draft of this script. A graded read lives or dies on the
    NEIGHBOURHOOD: as one state drifts from another, does cosine pass through the threshold
    smoothly (a slope, something to grade) or jump across it (a cliff, nothing to grade)?

    `k` sensors are displaced from rest by fraction `f` of their half-range. Note the gain law
    makes weight grow as f**3, so the transition is expected to be smooth in f and concentrated
    near the extremes; the question is whether it lingers anywhere near 0.85.
    """
    rs = roster()
    ranges = {n: (float(s["range"][0]), float(s["range"][1])) for n, s in rs.items()}
    base = {n: hold_value(n, s) for n, s in rs.items()}
    base_vec = embed(base, ranges)
    rng = random.Random(args.seed)
    names = sorted(rs)
    fracs = [i / 20.0 for i in range(21)]

    print(f"displacing k sensors from rest by fraction f of their half-range; {args.repeats} draws each")
    print(f"gain {GAIN}, threshold {THRESHOLD}\n")
    print("  k \\ f   " + "".join(f"{f:6.2f}" for f in fracs[::4]))
    in_band = 0
    total = 0
    crossings = []
    for k in args.ks:
        row, curve = [], []
        for f in fracs:
            vals = []
            for _ in range(args.repeats):
                picked = rng.sample(names, min(k, len(names)))
                st = dict(base)
                for n in picked:
                    lo, hi = ranges[n]
                    # Displace from the sensor's OWN hold value, not the range midpoint. Anchoring
                    # the perturbation somewhere other than the base makes f=0 a different state
                    # from the base — and for the no-rest sensors it snapped them to the midpoint,
                    # i.e. gain 0 for every sensor at once: the all-silent ZERO vector, which read
                    # as cos 0.0 and looked like a finding. f=0 must be cosine 1.0 by construction,
                    # and that is the check that caught it.
                    bound = hi if rng.random() < 0.5 else lo
                    st[n] = base[n] + (bound - base[n]) * f
                vals.append(cos(base_vec, embed(st, ranges)))
            m = statistics.median(vals)
            curve.append((f, m))
            total += 1
            if abs(m - THRESHOLD) <= args.band:
                in_band += 1
            row.append(m)
        # where does this k-curve cross the threshold, and how steeply?
        cross = None
        for (f0, c0), (f1, c1) in zip(curve, curve[1:]):
            if c0 >= THRESHOLD > c1:
                cross = (f0, f1, c0 - c1)
                break
        crossings.append((k, cross))
        print(f"  {k:2d}     " + "".join(f"{v:6.3f}" for v in row[::4]))

    print()
    for k, cross in crossings:
        if cross is None:
            print(f"  k={k:2d}: never crosses {THRESHOLD} across the full displacement range")
        else:
            f0, f1, drop = cross
            print(f"  k={k:2d}: crosses {THRESHOLD} between f={f0:.2f} and f={f1:.2f} (drop {drop:.3f} over one step)")
    pct = in_band / total * 100 if total else 0.0
    print(f"\n{in_band}/{total} grid points ({pct:.1f}%) sit within ±{args.band} of the threshold")
    print(
        "READING: cosine passes through the threshold GRADUALLY — there is a middle to read"
        if pct >= args.grade_floor
        else f"READING: <{args.grade_floor}% of the grid lingers near the threshold — it is crossed "
        "abruptly, so a graded read has little to grade ON THIS ROSTER"
    )
    print("  (SHAPE only — says nothing about which regions the world actually visits.)")
    if args.json:
        Path(args.json).write_text(
            json.dumps(
                {
                    "threshold": THRESHOLD,
                    "gain": GAIN,
                    "ks": args.ks,
                    "repeats": args.repeats,
                    "band": args.band,
                    "pct_grid_in_band": pct,
                    "crossings": [
                        {"k": k, "between": c[:2] if c else None, "step_drop": c[2] if c else None}
                        for k, c in crossings
                    ],
                    "what_this_cannot_say": "shape only; not the world's support (see module docstring)",
                },
                indent=2,
            )
            + "\n"
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("axes", help="single-sensor sweeps across declared ranges")
    a.add_argument("--json")
    a.set_defaults(func=cmd_axes)
    ls = sub.add_parser("landscape", help="pairwise cosine distribution over sampled states")
    ls.add_argument("--n", type=int, default=300)
    ls.add_argument("--seed", type=int, default=0)
    ls.add_argument("--band", type=float, default=0.05)
    ls.add_argument("--grade-floor", type=float, default=5.0, help="%% in band above which grading has content")
    ls.add_argument("--json")
    ls.set_defaults(func=cmd_landscape)
    nb = sub.add_parser("neighbourhood", help="how cosine falls as a state moves away from rest (the decisive one)")
    nb.add_argument("--ks", type=int, nargs="+", default=[1, 2, 3, 5, 8, 17])
    nb.add_argument("--repeats", type=int, default=25)
    nb.add_argument("--seed", type=int, default=0)
    nb.add_argument("--band", type=float, default=0.05)
    nb.add_argument("--grade-floor", type=float, default=5.0)
    nb.add_argument("--json")
    nb.set_defaults(func=cmd_neighbourhood)
    args = ap.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
