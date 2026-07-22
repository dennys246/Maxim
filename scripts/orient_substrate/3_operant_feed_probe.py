#!/usr/bin/env python
"""Operant-feed orient probe — does a MOTHER's contingent feeding teach orienting
when the infant has NO intrinsic drive to orient? (cradle_mother, 2026-07-21)

Probes 1 & 2 proved the INTRINSIC centeredness drive teaches orienting on its own
(contingent 1.000 vs ~0.50 chance) — which is exactly why it is a CONFOUND for the
"raised by a mother" claim: an infant with that drive orients whether or not a
mother is present. This probe removes the confound and tests the honest claim:

  Body: hunger drive (entropic) + an azimuth SENSOR with NO drive — the infant
  perceives direction (encodes an EC cluster) but turning relieves nothing
  intrinsically. So a turn earns no drive-relief motor-credit; its ONLY possible
  reward is the mother.

  Mother: operant shaping. When the infant's turn moves it TOWARD her (|azimuth|
  decreases), she feeds it (hunger relief) and that relief credits the infant's
  own preceding action via the NEW ``NAc.credit_operant_reward`` — the operant
  mechanism this probe gates. Credit lands on ``_cluster_reward_bias`` (the
  action-selection surface ``recommend_action`` reads), NOT ``distribute_reward``
  (which credits recognition, the wrong surface — see the design notes).

Arms (mirrors the June gaze-probe / probe-1 design):
  contingent : the mother credits the infant's ACTUAL action when it orients.
  yoked      : same reward volume/rate, but on a RANDOM action — superstition
               control. If contingent >> yoked, it is the CONTINGENCY (crediting
               the right action) that teaches, not the reward volume.
  none       : no operant credit.                         [chance baseline]

The decisive contrast vs probes 1/2: here ``none`` sits at CHANCE (~0.50). An
intrinsic-drive body would orient even in ``none``; this body does not. That gap
is the evidence the mother is NECESSARY — orienting is LEARNED from her, not innate.

Two floor regimes, because removing the drive exposes the tool-success fallback:
  no_floor  : the infant's turns get NO intrinsic reward — the mother is the sole
              teacher (the clean mechanism test).
  tool_floor: each turn also books the uniform ``+1`` tool-success fallback that
              ``record_outcome`` gives a driveless action. This SATURATES both
              directions to the cluster-reward cap and may drown the operant
              signal — if ``contingent`` collapses under ``tool_floor`` but works
              under ``no_floor``, the embodied sim needs an "operant-only" mode
              (suppress tool-success for the infant's exploratory turns) BEFORE it
              can teach. That is the whole point of running this first.

Read: contingent >> yoked ~ none (~0.50) under no_floor -> the operant mechanism
teaches orienting from feeding alone. Then compare tool_floor to size the sim fix.
"""

from __future__ import annotations

import os
import sys

import numpy as np

# Keep the substrate honest: no reward-bias disable leaking in from the env.
os.environ.pop("MAXIM_NAC_REWARD_BIAS_DISABLED", None)

from maxim.decisions.nac import NAc, NACConfig  # noqa: E402
from maxim.embodiment.body import Embodiment  # noqa: E402
from maxim.embodiment.spec import _parse_entity  # noqa: E402
from maxim.embodiment.tool_bridge import ModulatorAffordanceTool  # noqa: E402
from maxim.similarity.ec import ECConfig, EntorhinalCortex  # noqa: E402
from maxim.similarity.encoder import SensorEncoder  # noqa: E402

ACTIONS = ["turn_left", "turn_right"]
EPSILON = 0.2
MIN_CONF = 0.02
AGENT = "infant"
ORIENTED = 0.15  # |azimuth| <= this counts as oriented-toward-mother
GAIN = 1.0  # operant reward magnitude the mother delivers per shaping event


def _body():
    """Infant: hunger drive (relievable) + azimuth SENSOR WITHOUT a drive.

    The missing ``drive`` block on azimuth is the whole design: turning changes
    azimuth but relieves nothing, so the infant has no innate reason to orient —
    only the mother's operant feeding can teach it."""
    d = {
        "name": "infant",
        "entity_type": "body",
        "sensors": {
            "azimuth": {
                "unit": "normalized",
                "range": [-1, 1],
                "initial": 0.0,
                # NO drive — perceivable direction, no intrinsic orient reward.
            },
            "hunger": {
                "unit": "ratio",
                "range": [0, 1],
                "initial": 0.5,
                "drive": {
                    "drift_mode": "entropic",
                    "drift_direction": "up",
                    "drift_rate": 0.0,  # probe controls hunger explicitly
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
    body = _parse_entity(d)
    return body, Embodiment(root=body)


def make_encoder():
    ec = EntorhinalCortex(ECConfig(frozen_centroid_modalities=frozenset({"interoception", "audio"})))
    return SensorEncoder(ec=ec, atl=None)


def sound_azimuth(rng, regime):
    side = -1.0 if rng.random() < 0.5 else 1.0
    mag = 0.7 if regime == "wide" else float(rng.uniform(0.3, 0.9))
    return side * mag


def run(arm, regime, *, floor, seed, ticks=3000):
    rng = np.random.default_rng(seed)
    nac = NAc(NACConfig())
    enc = make_encoder()
    body, emb = _body()
    mod = body.modulators["orient"]
    tools = {a: ModulatorAffordanceTool(body, mod, a, mod.affordances[a], a, embodiment=emb) for a in ACTIONS}
    seen_states: list[str] = []

    for _ in range(ticks):
        az = sound_azimuth(rng, regime)
        body.vital_metrics["azimuth"] = az  # place the sound
        state = enc.encode_sensors(agent_id=AGENT, sensors={"azimuth": az}, ranges={"azimuth": (-1.0, 1.0)})
        if state and state not in seen_states:
            seen_states.append(state)

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

        # Remember the infant's action for operant credit (the NEW mechanism).
        nac.set_pending_operant_action(AGENT, state, f"tool:{action}")

        tools[action].execute()  # real self_effect: azimuth += ±0.3
        az_after = float(body.vital_metrics.get("azimuth", az))
        progress = abs(az) - abs(az_after)  # >0 => turned TOWARD the mother

        # tool-success fallback: a driveless turn "succeeds" -> uniform +1 on
        # BOTH directions (this is what record_outcome books in the real sim).
        if floor == "tool_floor":
            nac.update_cluster_reward(AGENT, state, f"tool:{action}", 1.0)

        # Mother's operant shaping: reward movement toward her.
        if progress > 1e-9:
            # feed (hunger relief) — fidelity; the credit magnitude is what learns
            body.vital_metrics["hunger"] = max(0.0, float(body.vital_metrics.get("hunger", 0.5)) - 0.1)
            if arm == "contingent":
                nac.credit_operant_reward(AGENT, GAIN)  # credits the infant's pending action
            elif arm == "yoked":
                # same reward, RANDOM action in the real state — superstition control
                ract = ACTIONS[int(rng.integers(len(ACTIONS)))]
                nac.update_cluster_reward(AGENT, state, f"tool:{ract}", GAIN)
            # none: no operant credit

    # ---- eval: directedness with a RANDOM no-signal tiebreak -----------------
    er = np.random.default_rng(seed + 10_000)
    correct = total = lc = ln = rc = rn = 0
    for _ in range(400):
        az = sound_azimuth(er, regime)
        state = enc.encode_sensors(agent_id=AGENT, sensors={"azimuth": az}, ranges={"azimuth": (-1.0, 1.0)})
        rec = nac.recommend_action(
            agent_id=AGENT,
            available_tools=ACTIONS,
            current_drives=None,
            current_cluster_id=state,
            min_confidence=MIN_CONF,
        )
        action = rec["tool_name"] if rec else ACTIONS[int(er.integers(len(ACTIONS)))]
        want = "turn_left" if az < 0 else "turn_right"  # toward the sound
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
    print("operant-feed orient probe — mother teaches orienting, NO intrinsic drive\n")
    for floor in ("no_floor", "tool_floor"):
        print(f"### floor={floor}")
        print(f"{'regime':6} {'arm':11} {'directed':>9} {'left':>7} {'right':>7}")
        print("-" * 44)
        summ = {}
        for regime in ("wide", "full"):
            for arm in ("contingent", "yoked", "none"):
                ds = [run(arm, regime, floor=floor, seed=s) for s in seeds]
                d = float(np.mean([x["directedness"] for x in ds]))
                lft = float(np.mean([x["left"] for x in ds]))
                rgt = float(np.mean([x["right"] for x in ds]))
                summ[(regime, arm)] = d
                print(f"{regime:6} {arm:11} {d:9.3f} {lft:7.3f} {rgt:7.3f}")
            print("-" * 44)
        print(f"\nverdict (floor={floor}):")
        for regime in ("wide", "full"):
            c = summ[(regime, "contingent")]
            y = summ[(regime, "yoked")]
            n = summ[(regime, "none")]
            real = c > y + 0.15 and c > n + 0.15
            print(
                f"  {regime:5}: contingent {c:.3f} | yoked {y:.3f} | none {n:.3f}  -> "
                f"{'MOTHER TEACHES (contingency)' if real else 'no operant policy'}"
            )
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
