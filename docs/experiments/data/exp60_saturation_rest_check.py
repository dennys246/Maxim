"""Exp 60 gate (ii) first-run diagnosis: does `saturation` at an extreme explain cos 0.8502? (2026-09-15)

The live gate run measured cos(shore, submerged) = 0.8502 against a replayed 0.787. The per-sensor
table showed `saturation` at gain weight 1.0 in BOTH situations (an extreme of its declared [0,10]
range) — the Slice-1 base vector every estimate used carried 0.5 there. This replays the live
geometry (values reconstructed from the record's gain weights; hostile sensors at neutral, spawn
distance 69, is_in_water 0.5→1.0, oxygen at the dive mean) with saturation at its captured extreme
vs at rest. Run from the repo root: `PYTHONPATH=src python <this file>`.

Result: extreme 0.8500 (= the live 0.8502), rest 0.7872 (= the design estimate). Fix: declare the
range so the MEASURED rest (fed = bridge clamp 10) is the midpoint — `[0, 20]`, initial 10.
"""

import math
import sys

sys.path.insert(0, "src")
from maxim.similarity.encoder import _sensor_embed  # noqa: E402  (the SHIPPED embed, not a mirror)

P = 3.0  # SensorEncoderConfig.gain_exponent for the world modality


def embed(vmap):
    # values below are already NORMALIZED (0..1): unit ranges make the encoder's normalization the identity
    return _sensor_embed(vmap, ranges={k: (0.0, 1.0) for k in vmap}, gain_exponent=P)


def cos(a, b):
    d = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return d / (na * nb) if na and nb else 0.0


NEUTRAL = {
    k: 0.5
    for k in (
        "food",
        "health",
        "hostile_count",
        "is_raining",
        "look_pitch",
        "nearest_hostile_dist",
        "nearest_player_dist",
        "on_ground",
        "xp_level",
    )
}


def shore(sat):
    return {
        **NEUTRAL,
        "saturation": sat,
        "light_level": 0.0,
        "time_of_day": 0.0417,
        "distance_from_spawn": 0.77,
        "y_altitude": 40 / 128,
        "speed": 0.54,
        "is_in_water": 0.5,
        "oxygen": 0.5,
    }


def submerged(sat):
    return {**shore(sat), "y_altitude": 35 / 128, "speed": 0.5, "is_in_water": 1.0, "oxygen": 0.334}


print("saturation state                              cos(shore, submerged)   < 0.85 = separates")
for label, sat in (
    ("at an extreme (as captured; range [0,10])", 1.0),
    ("at the declared rest / midpoint (w=0)", 0.5),
    ("fed under range [0,20]: 10/20 = 0.5", 0.5),
    ("drained under range [0,20]: 0/20 = 0.0", 0.0),
):
    c = cos(embed(shore(sat)), embed(submerged(sat)))
    print(f"   {label:<44} {c:.4f}      {'SEPARATES' if c < 0.85 else 'SAME CLUSTER'}")
print(
    "\nlive record: 0.8502. A drained bot under the new range is loud again — the protocol keeps the bot satiated (SF-3)."
)
