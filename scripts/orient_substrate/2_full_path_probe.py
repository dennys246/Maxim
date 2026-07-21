#!/usr/bin/env python
"""Orient motor-credit probe — FULL PRODUCTION PATH (the authoritative gate).

``1_motor_credit_probe.py`` exercised only the cluster channel
(update_cluster_reward + recommend_action) and so OVER-STATED GAP 1: the
pre-merge review noted production also routes every action through
``record_outcome``, which books a NEGATIVE causal link whenever
``embodiment_failed`` is True — and a correct toward-turn leaves the body still
off-center, so ``drive:azimuth:discomfort`` fires and it IS embodiment_failed.
That negative causal link subtracts from BOTH turn candidates in
``recommend_action`` and could suppress selection below the confidence gate.

This probe runs the ENTIRE real path with no LLM:
  encode_sensors -> EC cluster  ->  recommend_action (select)
  ModulatorAffordanceTool.execute (real self_effect + failure eval + the
      collateral-harm gate + drive_potential_diff emission)
  -> extract drive_potential_diff + embodiment_failed from side_effects
     (exactly as runtime/agent_loop.py does)
  -> record_outcome (real: books the causal link AND the cluster reward)

Arms isolate the review's concern:
  full        : real record_outcome — cluster reward AND causal-negative both booked
  cluster_only: only update_cluster_reward (the old probe #1 path, no causal link)
  none        : no learning (chance baseline)

Read: if full ~= cluster_only  -> the causal-negative does NOT suppress the
policy; GAP 1 holds on the real path. If full << cluster_only -> the causal
link suppresses selection and GAP 1 needs the transition-based drive-pain fix
(so a relieving turn stops re-firing discomfort) before it works end-to-end.
Eval tiebreak is RANDOM so `none` sits at true chance (not the alphabetical
pedestal probe #1 had).
"""

from __future__ import annotations

import os
import sys

import numpy as np

os.environ.pop("MAXIM_NAC_REWARD_BIAS_DISABLED", None)

from maxim.decisions.nac import NAc, NACConfig  # noqa: E402
from maxim.embodiment.body import Embodiment  # noqa: E402
from maxim.embodiment.spec import _parse_entity  # noqa: E402
from maxim.embodiment.tool_bridge import ModulatorAffordanceTool  # noqa: E402
from maxim.runtime.tool_dispatch import record_outcome  # noqa: E402
from maxim.similarity.ec import ECConfig, EntorhinalCortex  # noqa: E402
from maxim.similarity.encoder import SensorEncoder  # noqa: E402

ACTIONS = ["turn_left", "turn_right"]
EPSILON = 0.2
MIN_CONF = 0.02
AGENT = "infant"


class _NullPool:
    """Minimal context_pool for record_outcome (no-op sink)."""

    def add_outcome(self, **kwargs):  # noqa: D401
        pass


def _body():
    d = {
        "name": "agent",
        "entity_type": "body",
        "sensors": {
            "azimuth": {
                "unit": "normalized",
                "range": [-1, 1],
                "initial": 0.0,
                "drive": {
                    "drift_mode": "homeostatic",
                    "set_point": 0.0,
                    "drift_rate": 0.0,
                    "comfort_band": 0.1,
                    "pain_scale": 0.3,
                },
            }
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
    body = _parse_entity(d)
    return body, Embodiment(root=body)


def make_encoder():
    ec = EntorhinalCortex(ECConfig(frozen_centroid_modalities=frozenset({"interoception", "audio"})))
    return SensorEncoder(ec=ec, atl=None)


def sound_azimuth(rng, regime):
    side = -1.0 if rng.random() < 0.5 else 1.0
    mag = 0.7 if regime == "wide" else float(rng.uniform(0.3, 0.9))
    return side * mag


def run(arm, regime, *, seed, ticks=3000):
    rng = np.random.default_rng(seed)
    nac = NAc(NACConfig())
    enc = make_encoder()
    body, emb = _body()
    mod = body.modulators["orient"]
    tools = {a: ModulatorAffordanceTool(body, mod, a, mod.affordances[a], a, embodiment=emb) for a in ACTIONS}
    pool = _NullPool()

    for _ in range(ticks):
        az = sound_azimuth(rng, regime)
        body.vital_metrics["azimuth"] = az  # place the sound
        state = enc.encode_sensors(agent_id=AGENT, sensors={"azimuth": az})
        if rng.random() < EPSILON:
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

        result = tools[action].execute()  # real self_effect + failure eval + gate
        se = result.side_effects or {}
        dpd = se.get("drive_potential_diff")
        emb_failed = bool(se.get("embodiment_failures"))

        if arm == "full":
            record_outcome(
                agent_id=AGENT,
                tool_name=action,
                success=True,
                result_summary="ok",
                error=None,
                reasoning="",
                recent_outcomes=[],
                max_recent=10,
                llm_worker=None,
                context_pool=pool,
                nac=nac,
                cluster_id=state,
                embodiment_failed=emb_failed,
                drive_potential_diff=dpd,
            )
        elif arm == "cluster_only":
            if dpd is not None:
                nac.update_cluster_reward(AGENT, state, f"tool:{action}", float(dpd))
        # none: no learning

    # ---- eval: directedness with a RANDOM no-signal tiebreak -----------------
    er = np.random.default_rng(seed + 10_000)
    correct = total = lc = ln = rc = rn = 0
    for _ in range(400):
        az = sound_azimuth(er, regime)
        state = enc.encode_sensors(agent_id=AGENT, sensors={"azimuth": az})
        rec = nac.recommend_action(
            agent_id=AGENT,
            available_tools=ACTIONS,
            current_drives=None,
            current_cluster_id=state,
            min_confidence=MIN_CONF,
        )
        action = rec["tool_name"] if rec else ACTIONS[int(er.integers(len(ACTIONS)))]
        want = "turn_left" if az < 0 else "turn_right"
        hit = action == want
        correct += hit
        total += 1
        if az < 0:
            ln += 1
            lc += hit
        else:
            rn += 1
            rc += hit
    return {"directedness": correct / total, "left": lc / max(1, ln), "right": rc / max(1, rn)}


def main():
    seeds = [0, 1, 2, 3, 4]
    print("orient motor-credit — FULL PRODUCTION PATH (execute + record_outcome)\n")
    print(f"{'regime':6} {'arm':13} {'directed':>9} {'left':>7} {'right':>7}")
    print("-" * 46)
    summ = {}
    for regime in ("wide", "full"):
        for arm in ("full", "cluster_only", "none"):
            ds = [run(arm, regime, seed=s) for s in seeds]
            d = float(np.mean([x["directedness"] for x in ds]))
            lft = float(np.mean([x["left"] for x in ds]))
            rgt = float(np.mean([x["right"] for x in ds]))
            summ[(regime, arm)] = d
            print(f"{regime:6} {arm:13} {d:9.3f} {lft:7.3f} {rgt:7.3f}")
        print("-" * 46)

    print("\nverdict (does the causal-negative suppress the policy?):")
    for regime in ("wide", "full"):
        f = summ[(regime, "full")]
        c = summ[(regime, "cluster_only")]
        n = summ[(regime, "none")]
        real = f > n + 0.15
        survives = f > c - 0.10  # full within 0.10 of cluster-only => not suppressed
        print(
            f"  {regime:5}: full {f:.3f} | cluster_only {c:.3f} | none {n:.3f}  -> "
            f"{'REAL' if real else 'NO policy'}; causal-negative "
            f"{'does NOT suppress' if survives else 'SUPPRESSES'}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
