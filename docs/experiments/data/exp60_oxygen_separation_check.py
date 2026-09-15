import sys
import json
import math

sys.path.insert(0, "src")
from maxim.similarity.encoder import _stable_basis

DIM, P = 384, 3.0
rec = json.load(open("docs/experiments/data/l11_geometry_2026-09-15.json"))
# "surfaced" baseline = the Slice-1 SAFE vector (oxygen at rest 0.5)
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


surfaced = dict(base)
# drowning variants (differ from surfaced only as noted):
drown_ox_only = dict(base)
drown_ox_only["oxygen"] = 0.0  # oxygen 20->0 only
drown_ox_hp = dict(base)
drown_ox_hp["oxygen"] = 0.0
drown_ox_hp["health"] = 0.30  # + health hit (~12hp)
drown_realistic = dict(base)
drown_realistic.update(
    oxygen=0.0, health=0.30, on_ground=0.0, light_level=0.0
)  # underwater cave: light already 0 in base
print(f"cos(surfaced, drown: oxygen-only)      = {cos(embed(surfaced), embed(drown_ox_only)):.4f}")
print(f"cos(surfaced, drown: oxygen+health)    = {cos(embed(surfaced), embed(drown_ox_hp)):.4f}")
print(f"cos(surfaced, drown: oxygen+hp+onground)= {cos(embed(surfaced), embed(drown_realistic)):.4f}")
print("separates iff < 0.85")
