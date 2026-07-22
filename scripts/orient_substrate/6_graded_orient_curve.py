#!/usr/bin/env python
"""Graded operant orient — the infant must face the CORRECT DIRECTION.

Probes 4/5 used a 2-alternative task (turn_left vs turn_right); it collapses to
~2 EC clusters and 2 rules, so the infant learns in ~10 ticks and there is no
gradual developmental curve. This HARDENS it: the sound comes from one of six
directions, and the infant has six graded turn actions. To be fed it must pick
the turn that actually CENTERS the sound (the right direction AND the right
magnitude) — a small turn on a far sound doesn't face it. Now there are six
distinct (cluster → correct-action) associations, chance is 1/6 ≈ 0.17, and
learning is gradual (each association takes its own reinforcement).

Same operant mechanism (``NAc.credit_operant_reward`` on the infant's own action,
graded by how much closer to centered the turn got — the correct magnitude earns
the most). Same arms: taught / yoked / none. Same "mother is necessary" logic —
controls stay at chance (~0.17).

Read: ``taught`` centering-rate rises GRADUALLY from ~0.17 toward ~1−ε across
bins; controls stay at chance. This is the developmental curve the 2-alternative
task was too easy to show, and the setup where a crèche's coverage pooling
(experiment 47 federation) becomes dramatic — each partial infant only masters
the directions it happened to hear.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

os.environ.pop("MAXIM_NAC_REWARD_BIAS_DISABLED", None)

from maxim.decisions.nac import NAc, NACConfig  # noqa: E402
from maxim.embodiment.body import Embodiment  # noqa: E402
from maxim.embodiment.spec import _parse_entity  # noqa: E402
from maxim.embodiment.tool_bridge import ModulatorAffordanceTool  # noqa: E402
from maxim.similarity.ec import ECConfig, EntorhinalCortex  # noqa: E402
from maxim.similarity.encoder import SensorEncoder  # noqa: E402

# Six graded turns: sign = direction, magnitude = how far. Facing a sound at
# azimuth ``a`` needs the turn whose delta ≈ -a (delta moves azimuth toward 0).
TURNS = {
    "turn_l1": 0.3,
    "turn_l2": 0.6,
    "turn_l3": 0.9,
    "turn_r1": -0.3,
    "turn_r2": -0.6,
    "turn_r3": -0.9,
}
ACTIONS = list(TURNS)
POSITIONS = (-0.9, -0.6, -0.3, 0.3, 0.6, 0.9)  # discrete sound directions
CENTERED = 0.15  # |azimuth| <= this = facing the sound (fed)
AGENT = "infant"
GAIN = 1.0  # reward for FACING the sound (only the correct magnitude centers)

# PLACE-CELL population code for direction. A single azimuth scalar folds into
# just 2 EC clusters (left/right) at every threshold — the substrate resolves
# direction but not MAGNITUDE, so a graded task is unlearnable on the raw scalar.
# Tiling azimuth into narrow direction-tuned "cells" (Gaussian bumps, like place
# cells) gives one distinct cluster per direction (6/6 at width ≤ 0.15), which is
# the perceptual resolution graded orienting needs. This is how brains encode
# space — a population code, not one number.
_TUNING_WIDTH = 0.12
_CELL_CENTERS = (-0.9, -0.6, -0.3, 0.0, 0.3, 0.6, 0.9)


def _place_code(az: float) -> dict[str, float]:
    return {f"dir{i}": float(np.exp(-((az - c) ** 2) / (2 * _TUNING_WIDTH**2))) for i, c in enumerate(_CELL_CENTERS)}


_PLACE_RANGES = {f"dir{i}": (0.0, 1.0) for i in range(len(_CELL_CENTERS))}


def _body():
    affs = {name: {"params": {}, "description": name, "self_effect": {"azimuth": d}} for name, d in TURNS.items()}
    d = {
        "name": "infant",
        "entity_type": "body",
        "sensors": {
            "azimuth": {"unit": "normalized", "range": [-1, 1], "initial": 0.0},
            "hunger": {"unit": "ratio", "range": [0, 1], "initial": 0.5},
        },
        "modulators": {"orient": {"abstract": True, "affordances": affs}},
    }
    return Embodiment(root=_parse_entity(d))


def _tools(emb):
    body = emb.root
    mod = body.modulators["orient"]
    return body, {a: ModulatorAffordanceTool(body, mod, a, mod.affordances[a], a, embodiment=emb) for a in ACTIONS}


def run(arm, *, seed, ticks, bin_size, epsilon):
    rng = np.random.default_rng(seed)
    nac = NAc(NACConfig())
    enc = SensorEncoder(
        ec=EntorhinalCortex(ECConfig(frozen_centroid_modalities=frozenset({"interoception", "audio"}))), atl=None
    )
    emb = _body()
    body, tools = _tools(emb)

    bins: list[list[int]] = []
    cur: list[int] = []
    raw: list[int] = []
    for _ in range(ticks):
        az = float(POSITIONS[int(rng.integers(len(POSITIONS)))])
        body.vital_metrics["azimuth"] = az
        state = enc.encode_sensors(agent_id=AGENT, sensors=_place_code(az), ranges=_PLACE_RANGES)
        if rng.random() < epsilon:
            action = ACTIONS[int(rng.integers(len(ACTIONS)))]
        else:
            rec = nac.recommend_action(
                agent_id=AGENT,
                available_tools=ACTIONS,
                current_drives=None,
                current_cluster_id=state,
                min_confidence=0.0,
            )
            action = rec["tool_name"] if rec else ACTIONS[int(rng.integers(len(ACTIONS)))]

        nac.set_pending_operant_action(AGENT, state, f"tool:{action}")
        tools[action].execute()
        az_after = float(body.vital_metrics.get("azimuth", az))
        centered = abs(az_after) <= CENTERED
        progress = abs(az) - abs(az_after)  # how much closer to facing the sound
        cur.append(1 if centered else 0)
        raw.append(1 if centered else 0)

        # Mother's operant shaping — she feeds when the infant FACES her (centered),
        # not for partial turns. Only the correct magnitude+direction centers the
        # sound, so only it is reinforced (a graded progress reward saturates the
        # cluster cap and erases the magnitude distinction — the direction is
        # learned but not the amount). ``progress`` is unused for the reward.
        _ = progress
        if centered:
            if arm == "taught":
                nac.credit_operant_reward(AGENT, GAIN)
            elif arm == "yoked":
                nac.set_pending_operant_action(AGENT, state, f"tool:{ACTIONS[int(rng.integers(len(ACTIONS)))]}")
                nac.credit_operant_reward(AGENT, GAIN)
            # none: no credit

        if len(cur) >= bin_size:
            bins.append(cur)
            cur = []
    if cur:
        bins.append(cur)
    return [sum(b) / len(b) for b in bins], raw


def main():
    p = argparse.ArgumentParser(description="Graded operant orient learning curve")
    p.add_argument("--ticks", type=int, default=600)
    p.add_argument("--bin", type=int, default=50)
    p.add_argument("--seeds", type=int, default=8)
    p.add_argument("--epsilon", type=float, default=0.2)
    args = p.parse_args()

    seeds = list(range(args.seeds))
    print(f"graded operant orient — 6 directions × 6 turns (chance ≈ {1 / len(ACTIONS):.2f})")
    print(f"ticks={args.ticks} bin={args.bin} ε={args.epsilon} seeds={args.seeds}\n")
    FIRST_N = 6
    curves: dict[str, np.ndarray] = {}
    first: dict[str, list[float]] = {}
    for arm in ("taught", "yoked", "none"):
        res = [run(arm, seed=s, ticks=args.ticks, bin_size=args.bin, epsilon=args.epsilon) for s in seeds]
        nb = min(len(r[0]) for r in res)
        curves[arm] = np.array([r[0][:nb] for r in res]).mean(axis=0)
        first[arm] = [x for r in res for x in r[1][:FIRST_N]]

    nb = len(curves["taught"])
    header = "bin:      " + " ".join(f"{i + 1:>5}" for i in range(nb))
    print(header)
    print("-" * len(header))
    for arm in ("taught", "yoked", "none"):
        print(f"{arm:9} " + " ".join(f"{v:5.2f}" for v in curves[arm]))

    base = float(np.mean(first["taught"]))
    t = float(np.mean(curves["taught"][-3:]))
    y = float(np.mean(curves["yoked"][-3:]))
    n = float(np.mean(curves["none"][-3:]))
    print("\nverdict:")
    print(f"  first-{FIRST_N}-ticks baseline: taught={base:.2f} (chance ≈ {1 / len(ACTIONS):.2f})")
    print(f"  settled: taught={t:.2f}   yoked={y:.2f}   none={n:.2f}")
    learned = t >= 0.70 and (t - base) >= 0.30
    mother = (t - max(y, n)) >= 0.30
    print(f"  LEARNED (settled ≥ 0.70 and rose ≥ 0.30): {'PASS' if learned else 'FAIL'}")
    print(f"  MOTHER-TAUGHT (taught ≥ controls + 0.30): {'PASS' if mother else 'FAIL'}")
    if learned and mother:
        print("\n**the infant learned to face the sound's direction, taught by the mother's feeding alone.**")
    return 0


if __name__ == "__main__":
    sys.exit(main())
