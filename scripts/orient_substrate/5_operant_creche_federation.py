#!/usr/bin/env python
"""Operant orient CRÈCHE — does pooling partial learners reach the full policy?

Probe 4 showed a single infant learns to orient FAST (~10 ticks) once the mother
feeds it. So to make federation MEANINGFUL we make each infant SAMPLE-LIMITED:
only K ticks of experience — not enough to fully learn (a single K-tick infant
sits partway up the curve). The question: if a CRÈCHE of N such infants each get
their own K ticks with their own mother, and we MERGE what they learned, does the
pooled policy reach what a single infant needs N×K ticks for?

This is the honest scripted proof-of-concept for the Hivemind/Oasis substrate
share (``hivemind/merge.py::nac_merge`` — the REAL infra, not a toy). The one
thing the scripted setting lets us control that the full system can't yet: the
infants SHARE a perceptual encoder, so a left sound maps to the SAME EC cluster
id for every infant. That is biologically honest — every infant has the same
cochlea, so the same sound clusters the same way; only the LEARNED POLICY
(``cluster_reward_bias``, keyed on the shared clusters) differs and gets pooled.
(Fully independent agents encode to DIFFERENT uuid clusters and need ``ec_merge``
alignment first — the 1.1 Oasis pipeline; this PoC proves the pooling concept.)

Arms:
  single_partial : one infant, K ticks.                    (the per-agent floor)
  single_full    : one infant, N×K ticks.          (the "all experience in one" ceiling)
  creche_taught  : N infants × K ticks, merged.              (federation — the test)
  creche_none    : N infants × K ticks with NO mother, merged.  (pooling noise ≠ signal)

Read: ``creche_taught`` ≈ ``single_full`` ≫ ``single_partial`` → pooling partial
learners recovers the full-experience policy. ``creche_none`` stays at chance →
the merge pools LEARNING, not noise.
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

ACTIONS = ["turn_left", "turn_right"]
AGENT = "infant"  # SAME id for every infant so the merged cluster keys align
GAIN = 1.0


def _body():
    d = {
        "name": "infant",
        "entity_type": "body",
        "sensors": {
            "azimuth": {"unit": "normalized", "range": [-1, 1], "initial": 0.0},
            "hunger": {"unit": "ratio", "range": [0, 1], "initial": 0.5},
        },
        "modulators": {
            "orient": {
                "abstract": True,
                "affordances": {
                    "turn_left": {"params": {}, "description": "l", "self_effect": {"azimuth": 0.3}},
                    "turn_right": {"params": {}, "description": "r", "self_effect": {"azimuth": -0.3}},
                },
            }
        },
    }
    return Embodiment(root=_parse_entity(d))


def sound_azimuth(rng):
    side = -1.0 if rng.random() < 0.5 else 1.0
    return side * float(rng.uniform(0.3, 0.9))


def _tools(emb):
    body = emb.root
    mod = body.modulators["orient"]
    return body, {a: ModulatorAffordanceTool(body, mod, a, mod.affordances[a], a, embodiment=emb) for a in ACTIONS}


def train(nac, enc, *, ticks, epsilon, seed, taught=True):
    """Operant-train ``nac`` in place for ``ticks`` ticks using the shared ``enc``."""
    rng = np.random.default_rng(seed)
    emb = _body()
    body, tools = _tools(emb)
    for _ in range(ticks):
        az = sound_azimuth(rng)
        body.vital_metrics["azimuth"] = az
        state = enc.encode_sensors(agent_id=AGENT, sensors={"azimuth": az}, ranges={"azimuth": (-1.0, 1.0)})
        if rng.random() < epsilon:
            action = ACTIONS[int(rng.integers(2))]
        else:
            rec = nac.recommend_action(
                agent_id=AGENT,
                available_tools=ACTIONS,
                current_drives=None,
                current_cluster_id=state,
                min_confidence=0.0,
            )
            action = rec["tool_name"] if rec else ACTIONS[int(rng.integers(2))]
        nac.set_pending_operant_action(AGENT, state, f"tool:{action}")
        tools[action].execute()
        progress = abs(az) - abs(float(body.vital_metrics.get("azimuth", az)))
        if taught and progress > 1e-9:
            nac.credit_operant_reward(AGENT, GAIN)


def evaluate(nac, enc, *, trials, seed):
    """Greedy directedness of ``nac``'s policy on the shared clusters (no learning)."""
    rng = np.random.default_rng(seed + 99_991)
    emb = _body()
    body, tools = _tools(emb)
    hits = 0
    for _ in range(trials):
        az = sound_azimuth(rng)
        body.vital_metrics["azimuth"] = az
        state = enc.encode_sensors(agent_id=AGENT, sensors={"azimuth": az}, ranges={"azimuth": (-1.0, 1.0)})
        rec = nac.recommend_action(
            agent_id=AGENT, available_tools=ACTIONS, current_drives=None, current_cluster_id=state, min_confidence=0.0
        )
        action = rec["tool_name"] if rec else ACTIONS[int(rng.integers(2))]
        want = "turn_left" if az < 0 else "turn_right"
        hits += action == want
    return hits / trials


def merge_creche(dumps):
    """Fold the REAL ``nac_merge`` over the crèche's dumps → one pooled state."""
    merged = dumps[0]
    for i, d in enumerate(dumps[1:], start=1):
        merged = nac_merge(merged, d, left_source="creche" if i > 1 else "infant0", right_source=f"infant{i}")
    return merged


def run_trial(*, N, K, epsilon, eval_trials, seed):
    rng = np.random.default_rng(seed)
    enc = SensorEncoder(
        ec=EntorhinalCortex(ECConfig(frozen_centroid_modalities=frozenset({"interoception", "audio"}))), atl=None
    )  # shared perceptual system

    def fresh():
        return NAc(NACConfig())

    # single partial: one infant, K ticks
    sp = fresh()
    train(sp, enc, ticks=K, epsilon=epsilon, seed=int(rng.integers(1 << 30)))
    single_partial = evaluate(sp, enc, trials=eval_trials, seed=seed)

    # single full: one infant, N*K ticks
    sf = fresh()
    train(sf, enc, ticks=N * K, epsilon=epsilon, seed=int(rng.integers(1 << 30)))
    single_full = evaluate(sf, enc, trials=eval_trials, seed=seed)

    # creche taught: N infants x K ticks, merged
    dumps = []
    for _ in range(N):
        a = fresh()
        train(a, enc, ticks=K, epsilon=epsilon, seed=int(rng.integers(1 << 30)))
        dumps.append(a.dump())
    merged = merge_creche(dumps)
    mc = fresh()
    mc.load_state(merged)
    creche_taught = evaluate(mc, enc, trials=eval_trials, seed=seed)

    # creche none: N infants x K ticks with NO mother, merged (pooling noise)
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
    p = argparse.ArgumentParser(description="Operant crèche federation (scripted)")
    p.add_argument("--agents", type=int, default=10, help="N infants in the crèche")
    p.add_argument("--ticks", type=int, default=4, help="K ticks each (SAMPLE-LIMITED — a single infant stays partial)")
    p.add_argument("--epsilon", type=float, default=0.2)
    p.add_argument("--eval-trials", type=int, default=400)
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
        f"operant crèche — N={args.agents} infants × K={args.ticks} ticks each (ε={args.epsilon}, {args.seeds} seeds)\n"
    )
    print(f"  single_partial (1 infant, {args.ticks} ticks)           : {sp:.2f}")
    print(f"  single_full    (1 infant, {args.agents * args.ticks} ticks)          : {sf:.2f}")
    print(f"  creche_taught  ({args.agents} infants × {args.ticks}, MERGED)   : {ct:.2f}")
    print(f"  creche_none    ({args.agents} × {args.ticks}, no mother, merged): {cn:.2f}")
    print("\nverdict:")
    pooled = ct >= sp + 0.15 and ct >= sf - 0.10
    noise = cn <= 0.60
    print(
        f"  FEDERATION POOLS LEARNING (creche_taught ≥ single_partial+0.15 and ≈ single_full): {'PASS' if pooled else 'FAIL'}"
    )
    print(f"  MERGE POOLS SIGNAL NOT NOISE (creche_none ≤ 0.60): {'PASS' if noise else 'FAIL'}")
    if pooled and noise:
        print("\n**a crèche of sample-limited infants pooled their learning into the full-experience orient policy.**")
    return 0


if __name__ == "__main__":
    sys.exit(main())
