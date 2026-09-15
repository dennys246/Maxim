"""Exp 60 placement replay: does `distance_from_spawn` merge shore and submerged? (2026-09-15)

The offline cos~0.79 estimate (exp60_oxygen_window_check.py) used the L11 Slice-1 SAFE vector as
its base — captured 36 blocks from WORLD spawn (distance_from_spawn v=0.641, gain weight 0.02).
The bridge caps that sensor at 128 (range [-128,128] → neutral at 0, v=1.0/w=1.0 at the cap), so
a pool far from spawn carries a full-weight CONSTANT in both situations — the exact L11 shape.
This replays the water classroom's real shore/submerged contrast (is_in_water 0.5→1.0,
y_altitude 40→35, hostiles at the neutral horizon per the 72-block guard) across spawn distances
through the real encoder bases. Run from the repo root: `PYTHONPATH=src python <this file>`.

Result (drives setup_world.WATER_MAX_DIST_FROM_SPAWN = 90):
    36 → 0.786   90 → 0.794   100 → 0.802   120 → 0.834   128 (cap) → 0.8525 = SAME cluster
"""

import json
import math
import sys

sys.path.insert(0, "src")
from maxim.similarity.encoder import _stable_basis  # noqa: E402

DIM, P = 384, 3.0
THRESHOLD = 0.85
rec = json.load(open("docs/experiments/data/l11_geometry_2026-09-15.json"))
base = {s["sensor"]: s["v_safe"] for s in rec["per_sensor"]}


def embed(vmap):
    vec = [0.0] * DIM
    for name, v in vmap.items():
        w = (abs(v - 0.5) * 2.0) ** P
        if w == 0.0:
            continue
        lo = _stable_basis(name, DIM, salt="low")
        hi = _stable_basis(name, DIM, salt="high")
        for i in range(DIM):
            vec[i] += w * ((1.0 - v) * lo[i] + v * hi[i])
    return vec


def cos(a, b):
    d = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return d / (na * nb) if na and nb else 0.0


def pair(spawn_blocks):
    v = (spawn_blocks + 128) / 256.0  # sensor range [-128, 128]
    shore = dict(base)
    shore.update(nearest_hostile_dist=0.5, hostile_count=0.5)  # 72-block guard: horizon = neutral
    shore["distance_from_spawn"] = v
    shore["is_in_water"] = 0.5  # dry: neutral
    sub = dict(shore)
    sub["y_altitude"] = 35 / 128  # pool floor vs shore y=40
    sub["is_in_water"] = 1.0  # head submerged, full air (dive-second-0)
    return cos(embed(shore), embed(sub))


print("distance_from_spawn (blocks)  gain w   cos(shore, submerged @ full air)   < 0.85 = separates")
for d in (36, 60, 80, 90, 100, 110, 120, 128):
    v = (d + 128) / 256.0
    c = pair(d)
    print(
        f"   {d:>3}                       {(abs(v - 0.5) * 2) ** P:.3f}    {c:.4f}      {'SEPARATES' if c < THRESHOLD else 'SAME CLUSTER'}"
    )
print("\nbound: 90 blocks (3D) keeps cos <= ~0.794 with margin under the 0.85 threshold")
