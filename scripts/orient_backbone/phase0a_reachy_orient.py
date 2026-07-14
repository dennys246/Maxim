#!/usr/bin/env python
"""Phase 0a — closed-loop orient validation on the REAL Reachy body.

docs/plans/audiovisual_orienting.md Phase 0a. The affect-study harness
(phase0_affect_study.py) validated the credit LOGIC with toy actions; this proves
it on the production artifact: the real `bodies/reachy_mini` SEM entity, sourcing
the orient actions from its declared `orient` affordances' `self_effect` on
`head_yaw`, with `potential_diff` credit on the real NAc and substrate-primary
selection (no LLM).

World-coupling (sim stand-in for hardware re-measurement): after an orient step
turns the head by the affordance's head_yaw delta, the target's head-relative
azimuth shifts by that amount; on hardware the DoA/vision percept re-measures it.
The azimuth sensor is overwritten each tick (the tick_vital_drift gotcha: the
body's azimuth drive has drift_rate 0 so nothing fabricates "centered").

Metric: directedness by region (does the learned policy turn TOWARD the target),
near vs far — centering-rate saturates (Phase 0b), directedness is the real test.
With only two discrete turns (no `stay`, step 0.3 > band 0.1) the head oscillates
around center by design; what we validate is that it turns the correct WAY.
"""

from __future__ import annotations

import os

import numpy as np

os.environ.pop("MAXIM_NAC_REWARD_BIAS_DISABLED", None)
from maxim.decisions.nac import NAc, NACConfig  # noqa: E402
from maxim.embodiment.component_registry import ComponentRegistry  # noqa: E402

EPSILON = 0.15
ARGMAX = -1.0e9  # always return argmax (no positive gate — see Phase 0b)


def load_orient_actions():
    """Pull the orient affordances' head_yaw deltas from the REAL body YAML."""
    ent = ComponentRegistry().instantiate("bodies/reachy_mini")
    assert "azimuth" in ent.vital_metrics, "reachy body missing azimuth sensor"
    assert "azimuth" in ent.drive_specs, "reachy body missing azimuth drive"
    orient = ent.modulators["orient"]
    actions = {an: sch.self_effect["head_yaw"] for an, sch in orient.affordances.items()}
    band = ent.drive_specs["azimuth"].comfort_band
    return actions, band


def az_bin(az: float, band: float) -> str:
    if abs(az) <= band:
        return "center"
    side = "left" if az < 0 else "right"
    return f"{'far' if abs(az) > 0.5 else 'near'}_{side}"


def train(actions, band, seed, ticks=2000):
    rng = np.random.default_rng(seed)
    nac = NAc(NACConfig())
    names = list(actions)
    az = float(rng.uniform(-1, 1))
    drift = float(rng.choice([-1.0, 1.0]))
    for _ in range(ticks):
        state = az_bin(az, band)
        if rng.random() < EPSILON:
            action = str(rng.choice(names))
        else:
            rec = nac.recommend_action(
                agent_id="reachy",
                available_tools=names,
                current_drives=None,
                current_cluster_id=state,
                min_confidence=ARGMAX,
            )
            action = rec["tool_name"] if rec else str(rng.choice(names))
        # orient step: head turns by the affordance's head_yaw delta; the target's
        # head-relative azimuth shifts by that amount (world re-measurement).
        delta = actions[action]
        az_after = float(np.clip(az + delta, -1.0, 1.0))
        # potential_diff (relief): credit the per-step REDUCTION in |azimuth|
        nac.update_cluster_reward("reachy", state, f"tool:{action}", abs(az) - abs(az_after))
        # target drifts independently, then next tick re-measures
        if rng.random() < 0.1:
            drift = -drift
        az = float(np.clip(az_after + drift * 0.06 * rng.uniform(0.3, 1.0), -1.0, 1.0))
        if abs(az) >= 1.0 - 1e-6:
            drift = -drift
    return nac, names


def eval_dir(nac, names, actions, lo, hi, seed):
    rng = np.random.default_rng(seed)
    correct = 0
    for _ in range(400):
        az = float(rng.choice([-1.0, 1.0])) * float(rng.uniform(lo, hi))
        rec = nac.recommend_action(
            agent_id="reachy",
            available_tools=names,
            current_drives=None,
            current_cluster_id=az_bin(az, 0.1),
            min_confidence=ARGMAX,
        )
        delta = actions[rec["tool_name"]] if rec else 0.0
        if delta != 0.0 and (delta < 0) == (az > 0):  # turned toward center
            correct += 1
    return correct / 400


def main():
    actions, band = load_orient_actions()
    print("\nPhase 0a — orient loop on the REAL reachy_mini body")
    print(f"  orient affordances (head_yaw deltas): {actions}   comfort_band={band}\n")
    seeds = list(range(12))
    near, far, untrained = [], [], []
    for s in seeds:
        nac, names = train(actions, band, s)
        near.append(eval_dir(nac, names, actions, 0.12, 0.45, 500 + s))
        far.append(eval_dir(nac, names, actions, 0.55, 0.95, 700 + s))
        cold = NAc(NACConfig())  # baseline: untrained body, same eval
        untrained.append(eval_dir(cold, names, actions, 0.12, 0.95, 900 + s))
    print(f"  near-dir (trained) = {np.mean(near):.3f}")
    print(f"  far-dir  (trained) = {np.mean(far):.3f}")
    print(f"  untrained baseline = {np.mean(untrained):.3f}")
    print("\n  Read: trained near/far -> ~1.0 means the real body's orient affordances +")
    print("  potential_diff credit learn to turn the head toward the target from every")
    print("  state (far included — dense relief credit, no terminal-reward coverage hole).")
    print("  Production body validated in sim; Phase 1 swaps the sim re-measurement for the")
    print("  live DoA/vision percept and dispatches the affordances through the executor.")


if __name__ == "__main__":
    main()
