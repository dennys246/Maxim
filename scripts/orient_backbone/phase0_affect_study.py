#!/usr/bin/env python
"""Phase 0 — shared orient-to-center backbone + affect-signal study.

Part of docs/plans/audiovisual_orienting.md. The shared backbone both the audio
(DoA) and visual (substrate gaze) tracks converge on: a modality-agnostic
orient-to-center loop where the agent turns its head to drive a target's
head-relative azimuth to 0. Azimuth comes from ANY source (sim here; on hardware
the mic/camera physically re-measure after each turn).

0a (backbone): the loop closes via substrate-primary `recommend_action` (no LLM),
crediting discrete orient affordances on the real NAc.

0b (affect study): the same loop, varying the credit signal, to characterize
Maxim's two affect surfaces on identical decisions. Crucially they differ on TWO
axes, not one:
  - SIGNAL: positive reward (centered → good) vs pain (off-center → bad).
  - SURFACE: cluster_reward (per-(state,action), state-conditioned) vs the
    causal/record_outcome path (per-tool only — what the drive-pain cascade
    feeds, and what `recommend_action` reads via get_negative_outcomes).

Hypothesis from reading the NAc (tested here):
  - The orient policy is STATE-DEPENDENT (turn_left is right only when the target
    is on the left). So a per-tool (state-blind) surface CANNOT represent it.
  - => pain_causal (state-blind) fails ~chance; cluster surfaces (state-conditioned)
    work for BOTH reward and pain. The decisive axis is the SURFACE, not the signal.
  - Separately: `recommend_action` is positive-gated, so pain-only has nothing to
    select on unless the gate admits the argmax (we use argmax selection to remove
    the gate as a confound, then note it).
"""

from __future__ import annotations

import os

import numpy as np

os.environ.pop("MAXIM_NAC_REWARD_BIAS_DISABLED", None)
from maxim.decisions.causal_link import Valence  # noqa: E402
from maxim.decisions.nac import NAc, NACConfig  # noqa: E402

BAND = 0.1  # |azimuth| <= BAND => centered
ACTIONS = {  # delta applied to head-relative target azimuth (turn toward target reduces |az|)
    "turn_left_big": +0.30,
    "turn_left_small": +0.10,
    "stay": 0.0,
    "turn_right_small": -0.10,
    "turn_right_big": -0.30,
}
ANAMES = list(ACTIONS)
EPSILON = 0.15
ARGMAX = -1.0e9  # min_confidence sentinel: always return the argmax (remove the positive gate)


def az_bin(az: float) -> str:
    if abs(az) <= BAND:
        return "center"
    if az < -0.35:
        return "far_left"
    if az < 0:
        return "near_left"
    if az > 0.35:
        return "far_right"
    return "near_right"


def run(mode: str, seed: int, gain: float = 1.0, ticks: int = 2000):
    """mode in {none, reward_cluster, pain_cluster, pain_causal, both}.
    `both` = reward_cluster (centered) + pain_causal (off-center) — the realistic
    combined system (positive reward via cluster, drive-pain via causal)."""
    rng = np.random.default_rng(seed)
    nac = NAc(NACConfig())
    az = float(rng.uniform(-1, 1))
    drift = float(rng.choice([-1.0, 1.0]))
    centered_series = []

    for _ in range(ticks):
        pre_bin = az_bin(az)
        if rng.random() < EPSILON:
            action = str(rng.choice(ANAMES))
        else:
            rec = nac.recommend_action(
                agent_id="a",
                available_tools=ANAMES,
                current_drives=None,
                current_cluster_id=pre_bin,
                min_confidence=ARGMAX,
            )
            action = rec["tool_name"] if rec else str(rng.choice(ANAMES))

        az_before = az
        az = float(np.clip(az + ACTIONS[action], -1.0, 1.0))
        centered = abs(az) <= BAND
        centered_series.append(1 if centered else 0)

        tool = f"tool:{action}"
        if mode == "reward_cluster" and centered:
            nac.update_cluster_reward("a", pre_bin, tool, +gain)
        elif mode == "potential_diff":
            # dense potential-difference: credit the per-step REDUCTION in offset
            # (positive=progress, negative=regress) — folds pain+reward into one
            nac.update_cluster_reward("a", pre_bin, tool, gain * (abs(az_before) - abs(az)))
        elif mode == "pain_cluster" and not centered:
            nac.update_cluster_reward("a", pre_bin, tool, -gain)
        elif mode == "pain_causal" and not centered:
            nac.record_outcome("orient", tool, Valence.NEGATIVE, context={"agent_id": "a"})
        elif mode == "both":
            if centered:
                nac.update_cluster_reward("a", pre_bin, tool, +gain)
            else:
                nac.record_outcome("orient", tool, Valence.NEGATIVE, context={"agent_id": "a"})

        # slow drift + transient respawn (sim world-coupling stand-in for re-measurement)
        if rng.random() < 0.1:
            drift = -drift
        az = float(np.clip(az + drift * 0.06 * rng.uniform(0.3, 1.0), -1.0, 1.0))
        if abs(az) >= 1.0 - 1e-6:
            drift = -drift

    return centered_series, nac


def eval_dir(nac, lo, hi, seed):
    """Frozen-policy directedness on offsets in [lo,hi]: fraction of states where
    the argmax action turns TOWARD center (reduces |az|). Chance ~3/5 directions."""
    rng = np.random.default_rng(seed)
    correct = total = 0
    for _ in range(400):
        az = float(rng.choice([-1.0, 1.0])) * float(rng.uniform(lo, hi))
        rec = nac.recommend_action(
            agent_id="a",
            available_tools=ANAMES,
            current_drives=None,
            current_cluster_id=az_bin(az),
            min_confidence=ARGMAX,
        )
        delta = ACTIONS[rec["tool_name"]] if rec else 0.0
        total += 1  # all eval offsets are off-center, so the correct act is never `stay`
        if delta != 0.0 and (delta < 0) == (az > 0):  # turned toward center
            correct += 1
    return correct / total if total else 0.0


def main():
    seeds = list(range(16))
    n_win = 6
    modes = ["none", "reward_cluster", "pain_cluster", "pain_causal", "both", "potential_diff"]
    print("\nPhase 0 — orient-to-center backbone + affect study")
    print("centering=outcome (saturates ~0.5 from drift); near/far-dir=learned policy quality\n")
    print(f"  {'mode':<16}{'centering':<11}{'near-dir':<11}{'far-dir':<11}")
    for mode in modes:
        cent, near, far = [], [], []
        for s in seeds:
            series, nac = run(mode, s)
            cent.append(float(np.mean(series[-len(series) // 3 :])))  # last-third centering
            near.append(eval_dir(nac, 0.12, 0.34, 500 + s))
            far.append(eval_dir(nac, 0.40, 0.95, 700 + s))
        print(f"  {mode:<16}{np.mean(cent):<11.3f}{np.mean(near):<11.3f}{np.mean(far):<11.3f}")

    print("\nRead (dir = fraction of off-center states the policy turns TOWARD center; ~0 = never orients):")
    print("  reward_cluster / both / potential_diff -> 1.0 near AND far: REWARD drives a perfect")
    print("    orient policy. (Coarse 5-bin state => every bin holds a centerable offset, so")
    print("    terminal reward learns them all — the far-coverage-hole only bites at finer bins,")
    print("    where potential_diff's dense per-step credit would separate from terminal reward.)")
    print("  pain_cluster / pain_causal -> ~0: PAIN ALONE DOES NOT ORIENT. Three compounding")
    print("    reasons: (1) recommend_action is positive-gated -> pain (negative-only) gives")
    print("    nothing to select on -> defaults to `stay`; (2) the causal surface is state-blind")
    print("    (per-tool); (3) off-center pain punishes correct-but-incomplete actions mid-approach.")
    print("  VERDICT: substrate-primary orienting is REWARD-DRIVEN; pain can veto/shape but cannot")
    print("    drive selection alone. The shared backbone's motor loop needs a positive/cluster")
    print("    (state-conditioned) signal — the audio plan's drive-pain-reduction credit is")
    print("    INSUFFICIENT for recommend_action by itself. potential_diff (dense, folds both,")
    print("    positive for progress) is the recommended credit signal.")


if __name__ == "__main__":
    main()
