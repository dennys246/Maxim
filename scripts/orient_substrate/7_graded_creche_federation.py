#!/usr/bin/env python
"""Graded orient CRÈCHE — coverage pooling on the hard task.

Probe 5 showed federation on the 2-alternative task, but that task is so easy a
single infant learns it in ~10 ticks, so the crèche's value was modest. The
GRADED task (probe 6: 6 directions × 6 graded turns, place-cell code) is where
pooling gets dramatic: a sample-limited infant only ever hears a FEW of the six
directions, so it masters only those — its policy has HOLES. Different infants
have different holes. Merging the crèche (real ``hivemind/merge.py::nac_merge``,
shared place-cell encoder so clusters align) should fill every hole → the merged
policy covers all six directions that no single partial infant does.

This is the baseline for the habituation experiment (48): does a crèche-raised
agent, carrying the pooled orient policy, habituate differently than a solo one?

Arms:
  single_partial : one infant, K ticks — partial DIRECTION COVERAGE.
  single_full    : one infant, N×K ticks — full coverage (ceiling).
  creche_taught  : N infants × K ticks, merged — federation (the test).
  creche_none    : N × K, no mother, merged — pooling noise stays at chance.

Read: ``creche_taught`` ≈ ``single_full`` ≫ ``single_partial`` → the crèche's
coverage pooling recovers the full graded policy. Chance ≈ 1/6.
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
from maxim.hivemind.merge import nac_merge  # noqa: E402
from maxim.similarity.ec import ECConfig, EntorhinalCortex  # noqa: E402
from maxim.similarity.encoder import SensorEncoder  # noqa: E402

TURNS = {"turn_l1": 0.3, "turn_l2": 0.6, "turn_l3": 0.9, "turn_r1": -0.3, "turn_r2": -0.6, "turn_r3": -0.9}
ACTIONS = list(TURNS)
POSITIONS = (-0.9, -0.6, -0.3, 0.3, 0.6, 0.9)
CENTERED = 0.15
AGENT = "infant"
GAIN = 1.0
_TUNING_WIDTH = 0.12
_CELL_CENTERS = (-0.9, -0.6, -0.3, 0.0, 0.3, 0.6, 0.9)


def _place_code(az):
    return {f"dir{i}": float(np.exp(-((az - c) ** 2) / (2 * _TUNING_WIDTH**2))) for i, c in enumerate(_CELL_CENTERS)}


_PLACE_RANGES = {f"dir{i}": (0.0, 1.0) for i in range(len(_CELL_CENTERS))}


def _body():
    affs = {name: {"params": {}, "description": name, "self_effect": {"azimuth": d}} for name, d in TURNS.items()}
    d = {
        "name": "infant",
        "entity_type": "body",
        "sensors": {"azimuth": {"unit": "normalized", "range": [-1, 1], "initial": 0.0}},
        "modulators": {"orient": {"abstract": True, "affordances": affs}},
    }
    return Embodiment(root=_parse_entity(d))


def _tools(emb):
    body = emb.root
    mod = body.modulators["orient"]
    return body, {a: ModulatorAffordanceTool(body, mod, a, mod.affordances[a], a, embodiment=emb) for a in ACTIONS}


def train(nac, enc, *, ticks, epsilon, seed, taught=True):
    rng = np.random.default_rng(seed)
    emb = _body()
    body, tools = _tools(emb)
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
        if taught and abs(float(body.vital_metrics.get("azimuth", az))) <= CENTERED:
            nac.credit_operant_reward(AGENT, GAIN)


def evaluate(nac, enc, *, trials, seed):
    rng = np.random.default_rng(seed + 99_991)
    emb = _body()
    body, tools = _tools(emb)
    hits = 0
    for _ in range(trials):
        az = float(POSITIONS[int(rng.integers(len(POSITIONS)))])
        body.vital_metrics["azimuth"] = az
        state = enc.encode_sensors(agent_id=AGENT, sensors=_place_code(az), ranges=_PLACE_RANGES)
        rec = nac.recommend_action(
            agent_id=AGENT, available_tools=ACTIONS, current_drives=None, current_cluster_id=state, min_confidence=0.0
        )
        action = rec["tool_name"] if rec else ACTIONS[int(rng.integers(len(ACTIONS)))]
        tools[action].execute()
        hits += abs(float(body.vital_metrics.get("azimuth", az))) <= CENTERED
    return hits / trials


def merge_creche(dumps):
    merged = dumps[0]
    for i, d in enumerate(dumps[1:], start=1):
        merged = nac_merge(merged, d, left_source="creche" if i > 1 else "infant0", right_source=f"infant{i}")
    return merged


def run_trial(*, N, K, epsilon, eval_trials, seed):
    rng = np.random.default_rng(seed)
    enc = SensorEncoder(
        ec=EntorhinalCortex(ECConfig(frozen_centroid_modalities=frozenset({"interoception", "audio"}))), atl=None
    )  # shared perceptual system → aligned place-cell clusters

    def fresh():
        return NAc(NACConfig())

    sp = fresh()
    train(sp, enc, ticks=K, epsilon=epsilon, seed=int(rng.integers(1 << 30)))
    single_partial = evaluate(sp, enc, trials=eval_trials, seed=seed)

    sf = fresh()
    train(sf, enc, ticks=N * K, epsilon=epsilon, seed=int(rng.integers(1 << 30)))
    single_full = evaluate(sf, enc, trials=eval_trials, seed=seed)

    dumps = []
    for _ in range(N):
        a = fresh()
        train(a, enc, ticks=K, epsilon=epsilon, seed=int(rng.integers(1 << 30)))
        dumps.append(a.dump())
    mc = fresh()
    mc.load_state(merge_creche(dumps))
    creche_taught = evaluate(mc, enc, trials=eval_trials, seed=seed)

    dumps_none = []
    for _ in range(N):
        a = fresh()
        train(a, enc, ticks=K, epsilon=epsilon, seed=int(rng.integers(1 << 30)), taught=False)
        dumps_none.append(a.dump())
    mn = fresh()
    mn.load_state(merge_creche(dumps_none))
    creche_none = evaluate(mn, enc, trials=eval_trials, seed=seed)

    return single_partial, single_full, creche_taught, creche_none


def main():
    p = argparse.ArgumentParser(description="Graded operant crèche federation")
    p.add_argument("--agents", type=int, default=12)
    p.add_argument(
        "--ticks", type=int, default=25, help="K ticks each — few enough that one infant covers only some directions"
    )
    p.add_argument("--epsilon", type=float, default=0.2)
    p.add_argument("--eval-trials", type=int, default=600)
    p.add_argument("--seeds", type=int, default=8)
    args = p.parse_args()

    rows = np.array(
        [
            run_trial(N=args.agents, K=args.ticks, epsilon=args.epsilon, eval_trials=args.eval_trials, seed=s)
            for s in range(args.seeds)
        ]
    )
    sp, sf, ct, cn = rows.mean(axis=0)
    print(
        f"graded crèche — N={args.agents} × K={args.ticks} ticks (ε={args.epsilon}, {args.seeds} seeds, chance ≈ {1 / len(ACTIONS):.2f})\n"
    )
    print(f"  single_partial (1 infant, {args.ticks} ticks)          : {sp:.2f}")
    print(f"  single_full    (1 infant, {args.agents * args.ticks} ticks)         : {sf:.2f}")
    print(f"  creche_taught  ({args.agents} × {args.ticks}, MERGED)  : {ct:.2f}")
    print(f"  creche_none    ({args.agents} × {args.ticks}, no mother): {cn:.2f}")
    print("\nverdict:")
    pooled = ct >= sp + 0.15 and ct >= sf - 0.12
    noise = cn <= 0.30
    print(
        f"  FEDERATION POOLS COVERAGE (creche ≥ single_partial+0.15 and ≈ single_full): {'PASS' if pooled else 'FAIL'}"
    )
    print(f"  MERGE POOLS SIGNAL NOT NOISE (creche_none ≤ 0.30): {'PASS' if noise else 'FAIL'}")
    if pooled and noise:
        print("\n**the crèche pooled its partial direction-coverage into the full graded orient policy.**")
    return 0


if __name__ == "__main__":
    sys.exit(main())
