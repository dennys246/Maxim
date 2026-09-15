import sys
import json
import math

sys.path.insert(0, "src")
from maxim.similarity.encoder import _stable_basis

DIM, P = 384, 3.0
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


surf = embed(base)
print("oxygen(bubbles/40)  norm_v   cos(surfaced,state)   >0.85=same cluster as surfaced")
for bub in [20, 18, 16, 14, 12, 10, 8, 6, 4, 2, 0]:
    v = bub / 40.0
    st = dict(base)
    st["oxygen"] = v
    c = cos(surf, embed(st))
    print(f"   {bub:>2}/40           {v:.3f}    {c:.4f}            {'SAME (no fear read)' if c > 0.85 else 'DISTINCT'}")
print("\nWITH a binary isInWater sensor ON underwater (adds a full-weight constant from dive start):")
# simulate isInWater as a world sensor v=1.0 (extreme, weight 1.0) present underwater, 0.5 surfaced
for bub in [20, 16, 12, 8, 4, 0]:
    v = bub / 40.0
    st = dict(base)
    st["oxygen"] = v
    st["is_in_water"] = 1.0
    surf2 = dict(base)
    surf2["is_in_water"] = 0.5  # surfaced: neutral
    c = cos(embed(surf2), embed(st))
    print(f"   oxygen {bub:>2}/40 + isInWater=1 : cos={c:.4f}  {'SAME' if c > 0.85 else 'DISTINCT'}")
