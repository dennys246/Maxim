"""Exp 62 offline replay (corollary 3 of docs/wiring/cosine-separation-is-directional.md): can a
second water pool inside the classroom builder's bounds ever land OUTSIDE the trained water cluster
on the shipped `minecraft_player` body — and can a `pressure` (water-above-eye) sensor move that?

Inputs: the LIVE gate-(ii) per-sensor normalized values of the Exp 60 pool (shore = `v_safe`,
submerged = `v_dark`, names inherited from the L11 probe) in `exp60_geometry_2026-09-15b.json`;
the shipped `_stable_basis`; the shipped gain law w = (2|v-0.5|)^3 (`encoder._sensor_embed`).
The script REFUSES to print a grid unless it reproduces the record's live cos(shore, submerged)
to 0.002 — the same guard `exp60_spawn_distance_check.py` uses.

Run: PYTHONPATH=src python docs/experiments/data/exp62_cross_pool_replay.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, "src")
from maxim.similarity.encoder import _stable_basis  # noqa: E402

DIM, P, TH = 384, 3.0, 0.85
REC = Path(__file__).with_name("exp60_geometry_2026-09-15b.json")
rec = json.load(open(REC))
SHORE = {s["sensor"]: s["v_safe"] for s in rec["per_sensor"]}
SUB = {s["sensor"]: s["v_dark"] for s in rec["per_sensor"]}
LIVE = float(rec["cosine"]["a4_gained"])  # the live gate-(ii) number the replay must reproduce


def embed(vmap: dict[str, float], zero: tuple[str, ...] = ()) -> list[float]:
    vec = [0.0] * DIM
    for name, v in vmap.items():
        if name in zero:
            continue
        w = (abs(v - 0.5) * 2.0) ** P
        if w == 0.0:
            continue
        lo = _stable_basis(name, DIM, salt="low")
        hi = _stable_basis(name, DIM, salt="high")
        for i in range(DIM):
            vec[i] += w * ((1.0 - v) * lo[i] + v * hi[i])
    return vec


def cos(a: list[float], b: list[float]) -> float:
    d = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return d / (na * nb) if na and nb else 0.0


def pool2(
    base: dict[str, float],
    *,
    y: float,
    d: float,
    light: float | None = None,
    tod: float | None = None,
    pressure: float | None = None,
    prange: float = 12.0,
    zero: tuple[str, ...] = (),
) -> dict[str, float]:
    m = dict(base)
    m["y_altitude"] = y / 128.0
    m["distance_from_spawn"] = (d + 128.0) / 256.0
    if light is not None:
        m["light_level"] = light / 15.0
    if tod is not None:
        m["time_of_day"] = tod
    if pressure is not None:
        m["pressure"] = (pressure + prange) / (2 * prange)
    for k in zero:
        m.pop(k, None)
    return m


def main() -> int:
    base = cos(embed(SHORE), embed(SUB))
    print(f"pool 1 live gate (ii): replayed cos(shore, submerged) = {base:.4f}  (record: {LIVE})")
    if LIVE is not None and abs(base - LIVE) > 0.002:
        print("REFUSED: the replay does not reproduce the live record — the sensor→basis mapping is wrong")
        return 2
    y1 = SUB["y_altitude"] * 128.0
    d1 = SUB["distance_from_spawn"] * 256.0 - 128.0
    print(f"pool 1 submerged: y {y1:.1f}, distance_from_spawn {d1:.1f}; threshold {TH}")
    weights = {k: (abs(v - 0.5) * 2.0) ** P for k, v in SUB.items()}
    print(
        "pool 1 submerged gain weights (top 6): "
        + ", ".join(f"{k} {w:.2f}" for k, w in sorted(weights.items(), key=lambda kv: -kv[1])[:6])
    )
    sub1 = embed(SUB)
    print(
        "\n--- shipped roster: cos(pool-1 submerged, pool-2 submerged) over the builder's band; '*' = MISS (< 0.85) ---"
    )
    print("  y2 \\ d2 |   45     69     90  |  pool-2 own cos(shore2, sub2) at d2=69")
    for y2 in (5, 20, 35, 59, 80, 110, 123):
        row = []
        for d2 in (45, 69, 90):
            c = cos(sub1, embed(pool2(SUB, y=y2, d=d2)))
            row.append(f"{c:.3f}{'*' if c < TH else ' '}")
        own = cos(embed(pool2(SHORE, y=y2 + 5, d=69)), embed(pool2(SUB, y=y2, d=69)))
        print(f"  {y2:4d}    | " + "  ".join(row) + f"  |  {own:.3f}{'*' if own < TH else ' '}")
    print("\n--- pressure (water above eye) added at Exp 60's depth (4 blocks), pool 2 at y 59 / d 69 ---")
    for prange in (12.0, 4.0):
        s1 = embed(pool2(SUB, y=y1, d=d1, pressure=4.0, prange=prange))
        s2 = embed(pool2(SUB, y=59, d=69, pressure=4.0, prange=prange))
        w = (abs((4.0 + prange) / (2 * prange) - 0.5) * 2.0) ** P
        print(
            f"  range [-{prange:.0f},{prange:.0f}]: pressure weight {w:.3f}; cross-pool cos {cos(s1, s2):.4f}; pool-1 within cos(shore, sub) {cos(embed(pool2(SHORE, y=y1 + 5, d=d1, pressure=0.0, prange=prange)), s1):.4f}"
        )
    print("\n--- absolutes ablated (y_altitude + distance_from_spawn removed): cross-pool cos ---")
    print(
        f"  {cos(embed(pool2(SUB, y=y1, d=d1, zero=('y_altitude', 'distance_from_spawn'))), embed(pool2(SUB, y=59, d=69, zero=('y_altitude', 'distance_from_spawn')))):.4f}  (an identity by construction)"
    )
    print("\n--- non-place differences (the levers with mass) ---")
    for label, kw in (
        ("lit surface pond (light 15)", dict(y=59, d=69, light=15.0)),
        ("night pool (time_of_day 0.99)", dict(y=35, d=69, tod=0.99)),
        ("night pool (time_of_day 0.75)", dict(y=35, d=69, tod=0.75)),
    ):
        c = cos(sub1, embed(pool2(SUB, **kw)))
        cp = cos(
            embed(pool2(SUB, y=y1, d=d1, pressure=4.0, prange=4.0)), embed(pool2(SUB, pressure=4.0, prange=4.0, **kw))
        )
        print(
            f"  {label:32s} shipped {c:.3f}{'*' if c < TH else ' '}   with pressure [-4,4] {cp:.3f}{'*' if cp < TH else ' '}"
        )
    print("\n--- stacked pool 2 (same x/z as pool 1, different shore altitude) ---")
    for y2 in (10, 95):
        c = cos(sub1, embed(pool2(SUB, y=y2, d=88 if y2 == 10 else 72)))
        print(f"  pool 2 floor y {y2:3d}: cross-pool cos {c:.3f}{'*' if c < TH else ' '}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
