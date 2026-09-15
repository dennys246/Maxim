import sys
import json
import math

sys.path.insert(0, "src")
from maxim.similarity.encoder import _stable_basis  # real SHA-derived bases

DIM = 384
P = 3.0  # A4 gain exponent (world default)

rec = json.load(open("docs/experiments/data/l11_geometry_2026-09-15.json"))
per = {s["sensor"]: s for s in rec["per_sensor"]}


def embed(names, side, gained=True):
    vec = [0.0] * DIM
    for name in names:
        v = per[name][f"v_{side}"]
        w = 1.0 if not gained else (abs(v - 0.5) * 2.0) ** P
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


ALL = list(per.keys())
THREAT = ["nearest_hostile_dist", "hostile_count", "y_altitude", "distance_from_spawn", "nearest_player_dist"]
THREAT_MIN = ["nearest_hostile_dist"]  # the only real mover
RANGEFIX = ALL  # simulate re-centering light_level + time_of_day to neutral 0.5 (weight 0)


def with_rangefix(names, side):
    # emulate re-centered ranges: light_level & time_of_day now rest at neutral -> v=0.5 -> weight 0
    vec = [0.0] * DIM
    for name in names:
        v = per[name][f"v_{side}"]
        if name in ("light_level", "time_of_day"):
            v = 0.5
        w = (abs(v - 0.5) * 2.0) ** P
        if w == 0.0:
            continue
        lo = _stable_basis(name, DIM, salt="low")
        hi = _stable_basis(name, DIM, salt="high")
        for i in range(DIM):
            vec[i] += w * ((1.0 - v) * lo[i] + v * hi[i])
    return vec


print(f"full 16ch  A4 : cos = {cos(embed(ALL, 'safe'), embed(ALL, 'dark')):.4f}  (probe reported 0.9767 — sanity)")
print(f"full 16ch  A0 : cos = {cos(embed(ALL, 'safe', False), embed(ALL, 'dark', False)):.4f}  (probe reported 0.9959)")
print(f"threat 5ch A4 : cos = {cos(embed(THREAT, 'safe'), embed(THREAT, 'dark')):.4f}  (bio-faithful predicted ~0.992)")
print(f"threat 5ch A0 : cos = {cos(embed(THREAT, 'safe', False), embed(THREAT, 'dark', False)):.4f}")
print(
    f"threat MIN(1) A4: cos = {cos(embed(THREAT_MIN, 'safe'), embed(THREAT_MIN, 'dark')):.4f}  (just nearest_hostile_dist)"
)
print(
    f"rangefix 16 A4 : cos = {cos(with_rangefix(ALL, 'safe'), with_rangefix(ALL, 'dark')):.4f}  (re-center light+time to neutral)"
)
print("threshold to clear = 0.85 (below = SEPARATES)")
