#!/usr/bin/env python
"""Operant orient LEARNING CURVE — the developmental trajectory, scripted.

The embodied cradle_mother sim proved the wrong instrument for MEASURING this: an
LLM narrator (non-deterministic, slow, GPU-contended), a confidence gate, 22-tool
competition, turn caps and response timeouts all wrap the weak operant signal in
machinery that fights it and needs endless tuning (tool competition → explore
weight → confidence-gate stalls — textbook divergence). So we measure the
mechanism where it is clean: scripted, no LLM, deterministic, thousands of ticks.

This EXTENDS ``3_operant_feed_probe.py`` (which validated the mechanism at a
single end-point: contingent 1.000 vs yoked/none ~0.48) to emit DIRECTEDNESS OVER
TICK-BINS — the learning curve. A hungry infant with NO intrinsic orient drive
(azimuth sensor, ``drive: null``) turns toward a sound only if a mother rewards
it for doing so (``NAc.credit_operant_reward`` on the infant's OWN action). The
question this answers that the end-point could not: HOW FAST, and does the curve
actually rise from chance to learned?

Exploration is a clean external ε-greedy (like probe 2), NOT the sim's
explore-bonus machinery — the thing that pinned the embodied run at chance was
``substrate_explore_bonus_weight=1.5`` exploring far more than the probe's ε=0.2
ever did. Here ε is explicit and tunable.

Arms:
  taught : mother credits the infant's own toward-turn (operant shaping).
  yoked  : same reward rate, RANDOM action — superstition control.
  none   : no operant credit.                              [chance baseline]

Read: ``taught`` directedness RISES across bins from ~0.5 (chance) toward the
ε-ceiling (~1 − ε/2); ``yoked`` and ``none`` stay flat at ~0.5. The gap, and its
onset, is the operant learning curve — and ``none``/``yoked`` at chance is the
"mother is necessary" proof (an intrinsic-drive body would rise even here).
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

ACTIONS = ["turn_left", "turn_right"]
AGENT = "infant"
GAIN = 1.0  # operant reward magnitude the mother delivers per toward-turn
MIN_CONF = 0.0  # scripted: no cold-start gate (the sim's gate stalled at low explore)


def _body():
    """Infant: hunger drive (relievable) + azimuth SENSOR WITHOUT a drive."""
    d = {
        "name": "infant",
        "entity_type": "body",
        "sensors": {
            "azimuth": {"unit": "normalized", "range": [-1, 1], "initial": 0.0},  # NO drive
            "hunger": {
                "unit": "ratio",
                "range": [0, 1],
                "initial": 0.5,
                "drive": {
                    "drift_mode": "entropic",
                    "drift_direction": "up",
                    "drift_rate": 0.0,
                    "deprivation_threshold": 0.7,
                    "deprivation_pain": 0.3,
                    "satisfaction_threshold": 0.3,
                },
            },
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


def make_encoder():
    ec = EntorhinalCortex(ECConfig(frozen_centroid_modalities=frozenset({"interoception", "audio"})))
    return SensorEncoder(ec=ec, atl=None)


def sound_azimuth(rng, regime):
    side = -1.0 if rng.random() < 0.5 else 1.0
    mag = 0.7 if regime == "wide" else float(rng.uniform(0.3, 0.9))
    return side * mag


def run(arm, regime, *, seed, ticks, bin_size, epsilon):
    """One infant's operant session. Returns per-bin directedness (learning curve)."""
    rng = np.random.default_rng(seed)
    nac = NAc(NACConfig())
    enc = make_encoder()
    emb = _body()
    body = emb.root
    mod = body.modulators["orient"]
    tools = {a: ModulatorAffordanceTool(body, mod, a, mod.affordances[a], a, embodiment=emb) for a in ACTIONS}

    bins: list[list[int]] = []
    cur: list[int] = []
    raw: list[int] = []
    for t in range(ticks):
        az = sound_azimuth(rng, regime)
        body.vital_metrics["azimuth"] = az
        state = enc.encode_sensors(agent_id=AGENT, sensors={"azimuth": az}, ranges={"azimuth": (-1.0, 1.0)})

        # ε-greedy exploration (clean + explicit — not the sim's explore bonus).
        if rng.random() < epsilon:
            action = ACTIONS[int(rng.integers(len(ACTIONS)))]
        else:
            rec = nac.recommend_action(
                agent_id=AGENT,
                available_tools=ACTIONS,
                current_drives=None,
                current_cluster_id=state,
                min_confidence=MIN_CONF,
            )
            action = rec["tool_name"] if rec else ACTIONS[int(rng.integers(len(ACTIONS)))]

        nac.set_pending_operant_action(AGENT, state, f"tool:{action}")
        tools[action].execute()  # azimuth += ±0.3
        az_after = float(body.vital_metrics.get("azimuth", az))
        progress = abs(az) - abs(az_after)  # > 0 => turned TOWARD the sound
        directed = progress > 1e-9
        cur.append(1 if directed else 0)

        # Mother's operant shaping.
        if directed:
            if arm == "taught":
                nac.credit_operant_reward(AGENT, GAIN)
            elif arm == "yoked":
                nac.set_pending_operant_action(AGENT, state, f"tool:{ACTIONS[int(rng.integers(len(ACTIONS)))]}")
                nac.credit_operant_reward(AGENT, GAIN)
            # none: no credit

        raw.append(1 if directed else 0)
        if len(cur) >= bin_size:
            bins.append(cur)
            cur = []
    if cur:
        bins.append(cur)
    return [sum(b) / len(b) for b in bins], raw


def main():
    p = argparse.ArgumentParser(description="Operant orient learning curve (scripted)")
    p.add_argument("--ticks", type=int, default=400)
    p.add_argument("--bin", type=int, default=50)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--epsilon", type=float, default=0.2)
    p.add_argument("--regime", default="full", choices=["wide", "full"])
    args = p.parse_args()

    seeds = list(range(args.seeds))
    print(
        f"operant orient learning curve — regime={args.regime} ticks={args.ticks} "
        f"bin={args.bin} ε={args.epsilon} seeds={args.seeds}\n"
    )
    FIRST_N = 5  # pre-learning baseline window (chance by construction)
    curves: dict[str, np.ndarray] = {}
    first_ticks: dict[str, list[float]] = {}
    for arm in ("taught", "yoked", "none"):
        results = [
            run(arm, args.regime, seed=s, ticks=args.ticks, bin_size=args.bin, epsilon=args.epsilon) for s in seeds
        ]
        per_seed = [r[0] for r in results]
        n_bins = min(len(c) for c in per_seed)
        mat = np.array([c[:n_bins] for c in per_seed])
        curves[arm] = mat.mean(axis=0)
        # mean directedness over the first FIRST_N ticks, pooled across seeds
        first_ticks[arm] = [x for r in results for x in r[1][:FIRST_N]]

    n_bins = len(curves["taught"])
    header = "bin:      " + " ".join(f"{i + 1:>5}" for i in range(n_bins))
    print(header)
    print("-" * len(header))
    for arm in ("taught", "yoked", "none"):
        print(f"{arm:9} " + " ".join(f"{v:5.2f}" for v in curves[arm]))

    # Baseline: the first few ticks (before any learning) are chance by
    # construction — reported so the rise from chance→learned is visible even
    # though it completes inside the first bin (the mechanism learns in ~10 ticks).
    baseline = {arm: float(np.mean(first_ticks[arm])) for arm in first_ticks}

    t_last = float(np.mean(curves["taught"][-3:]))  # settled value
    y_last = float(np.mean(curves["yoked"][-3:]))
    n_last = float(np.mean(curves["none"][-3:]))
    print("\nverdict:")
    print(f"  first-{FIRST_N}-ticks (pre-learning baseline): taught={baseline['taught']:.2f} (chance ≈ 0.50)")
    print(f"  settled: taught={t_last:.2f}   yoked={y_last:.2f}   none={n_last:.2f}")
    # A FRESH agent starts at chance, so a high settled value IS learning; the
    # rise is baseline→settled (fast). Separation from the controls is the proof
    # the MOTHER taught it (an intrinsic-drive body would rise even in none).
    learned = t_last >= 0.80 and (t_last - baseline["taught"]) >= 0.15
    mother = (t_last - max(y_last, n_last)) >= 0.20
    print(f"  LEARNED (settled ≥ 0.80 and rose from baseline ≥ 0.15): {'PASS' if learned else 'FAIL'}")
    print(f"  MOTHER-TAUGHT (taught settled ≥ controls + 0.20): {'PASS' if mother else 'FAIL'}")
    if learned and mother:
        print("\n**the infant learned to orient toward the sound, taught by the mother's feeding alone.**")
    return 0


if __name__ == "__main__":
    sys.exit(main())
