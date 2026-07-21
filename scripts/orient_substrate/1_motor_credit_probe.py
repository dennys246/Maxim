#!/usr/bin/env python
"""Orient motor-credit probe on Maxim's REAL substrate machinery.

NOTE: this probe exercises the CLUSTER CHANNEL IN ISOLATION
(update_cluster_reward + recommend_action). It does NOT model the negative
causal link production also books via record_outcome when a turn leaves the
body off-center — so on its own it OVER-STATES GAP 1. The authoritative gate is
``2_full_path_probe.py``, which runs the full execute -> record_outcome path and
shows the causal-negative does not suppress the policy (identical numbers). Keep
this file as the isolated-channel measurement; cite #2 for the end-to-end claim.

The gate before building the mother-scaffolded cradle (docs/plans/
cradle_orient_learning.md): does GAP 1 (the drive-relief motor-credit just
wired — drive_pain_for_value + drive_potential_diff + tool_dispatch consumption)
actually produce a *turn-toward-the-sound* policy, with NO LLM in the action
path?

What this exercises (all real, all offline — sensor encoding is pure hashing,
no model):
  - the REAL SensorEncoder.encode_sensors -> EntorhinalCortex cluster for the
    azimuth sensor (so it also tests whether the current _normalize_value
    encodes DIRECTION well enough — the P1 question);
  - the REAL drive relief as the reward: potential_diff =
    drive_pain_for_value(centeredness, before) - drive_pain_for_value(after),
    exactly the signal tool_bridge now emits on side_effects;
  - the REAL NAc.update_cluster_reward (credit) + NAc.recommend_action (select).

The correct action is NEVER hand-coded. turn_left moves azimuth +0.3 (toward
center for a left/negative sound), turn_right -0.3. The drive geometry alone
decides which turn earns relief, so if the loop learns "turn toward the sound"
it did so from the substrate reward, not a label.

Arms (falsifiable, per the June gaze-probe design):
  contingent : credit the (encoded-state, action) with the real potential_diff
  yoked      : credit a RANDOM (state, action) at the same reward   [superstition control]
  none       : never credit                                         [chance baseline]

Two stimulus regimes, to locate the P1 boundary directly:
  wide  : sounds at azimuth +/- 0.7 only         (current encoding SHOULD separate)
  full  : sounds spread over +/- [0.3, 0.9]       (current fold may collapse near-center)

Read: contingent >> yoked ~ none  -> the credit path produces a real policy.
If `wide` works but `full` collapses -> the _normalize_value fold (P1) is the
next fix, and now we have evidence it is actually needed (before perturbing the
[-1,1] thermal clusters Exp 42 depends on).
"""

from __future__ import annotations

import os
import sys

import numpy as np

# Keep the substrate honest: no reward-bias disable leaking in from the env.
os.environ.pop("MAXIM_NAC_REWARD_BIAS_DISABLED", None)

from maxim.decisions.nac import NAc, NACConfig  # noqa: E402
from maxim.embodiment.sem import HomeostaticDriveSpec, drive_comfort_progress  # noqa: E402
from maxim.similarity.ec import ECConfig, EntorhinalCortex  # noqa: E402
from maxim.similarity.encoder import SensorEncoder  # noqa: E402

# ---- world / policy constants (mirror base_humanoid.yaml) --------------------
CENTEREDNESS = HomeostaticDriveSpec(set_point=0.0, drift_rate=0.0, comfort_band=0.1, pain_scale=0.3)
TURN_DELTA = 0.3  # base_humanoid orient turn_left {azimuth:+0.3} / turn_right {-0.3}
ACTIONS = ["turn_left", "turn_right"]  # alphabetical: ties -> turn_left
EPSILON = 0.2  # exploration (motor-babble prior)
MIN_CONF = 0.02  # recommend_action gate: a single reinforced hit clears it
AGENT = "infant"


def apply_turn(az: float, action: str) -> float:
    delta = TURN_DELTA if action == "turn_left" else -TURN_DELTA
    return max(-1.0, min(1.0, az + delta))


def relief(az_before: float, az_after: float) -> float:
    """The motor-credit signal: value-progress toward center (matches production
    drive_comfort_progress). The reward path signs this (+1/-1); this probe uses
    the raw value for the isolated-channel measurement."""
    return drive_comfort_progress(CENTEREDNESS, az_before, az_after)


def make_encoder():
    ec = EntorhinalCortex(ECConfig(frozen_centroid_modalities=frozenset({"interoception", "audio"})))
    return SensorEncoder(ec=ec, atl=None)


def sound_azimuth(rng, regime: str) -> float:
    side = -1.0 if rng.random() < 0.5 else 1.0
    if regime == "wide":
        mag = 0.7
    else:  # full range
        mag = float(rng.uniform(0.3, 0.9))
    return side * mag


def run(arm: str, regime: str, *, seed: int, ticks: int = 3000) -> dict:
    rng = np.random.default_rng(seed)
    nac = NAc(NACConfig())
    enc = make_encoder()

    for _ in range(ticks):
        az = sound_azimuth(rng, regime)
        state = enc.encode_sensors(agent_id=AGENT, sensors={"azimuth": az})
        # epsilon-greedy over the real substrate selector
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

        r = relief(az, apply_turn(az, action))
        if arm == "contingent":
            nac.update_cluster_reward(AGENT, state, f"tool:{action}", r)
        elif arm == "yoked":
            # same reward magnitude, but credited to a RANDOM state/action
            fake_az = sound_azimuth(rng, regime)
            fake_state = enc.encode_sensors(agent_id=AGENT, sensors={"azimuth": fake_az})
            fake_action = ACTIONS[int(rng.integers(len(ACTIONS)))]
            nac.update_cluster_reward(AGENT, fake_state, f"tool:{fake_action}", r)
        # arm == "none": never credit

    # ---- evaluation: directedness on a held-out probe set --------------------
    correct = 0
    total = 0
    left_correct = right_correct = left_n = right_n = 0
    eval_rng = np.random.default_rng(seed + 10_000)
    for _ in range(400):
        az = sound_azimuth(eval_rng, regime)
        state = enc.encode_sensors(agent_id=AGENT, sensors={"azimuth": az})
        rec = nac.recommend_action(
            agent_id=AGENT,
            available_tools=ACTIONS,
            current_drives=None,
            current_cluster_id=state,
            min_confidence=MIN_CONF,
        )
        action = rec["tool_name"] if rec else "turn_left"  # tie/no-signal -> alphabetical
        # correct = the turn that reduces |azimuth|
        want = "turn_left" if az < 0 else "turn_right"
        hit = action == want
        correct += hit
        total += 1
        if az < 0:
            left_n += 1
            left_correct += hit
        else:
            right_n += 1
            right_correct += hit

    return {
        "arm": arm,
        "regime": regime,
        "directedness": correct / total,
        "left": left_correct / max(1, left_n),
        "right": right_correct / max(1, right_n),
    }


def main() -> int:
    seeds = [0, 1, 2, 3, 4]
    print("orient motor-credit probe — real encode_sensors + drive relief + NAc\n")
    print(f"{'regime':6} {'arm':11} {'directed':>9} {'left':>7} {'right':>7}")
    print("-" * 44)
    summary: dict[tuple[str, str], float] = {}
    for regime in ("wide", "full"):
        for arm in ("contingent", "yoked", "none"):
            ds = [run(arm, regime, seed=s) for s in seeds]
            mean_dir = float(np.mean([d["directedness"] for d in ds]))
            mean_l = float(np.mean([d["left"] for d in ds]))
            mean_r = float(np.mean([d["right"] for d in ds]))
            summary[(regime, arm)] = mean_dir
            print(f"{regime:6} {arm:11} {mean_dir:9.3f} {mean_l:7.3f} {mean_r:7.3f}")
        print("-" * 44)

    print("\nverdict:")
    for regime in ("wide", "full"):
        c = summary[(regime, "contingent")]
        y = summary[(regime, "yoked")]
        n = summary[(regime, "none")]
        real = c > y + 0.15 and c > n + 0.15
        tag = "REAL policy" if real else "no separation"
        print(f"  {regime:5}: contingent {c:.3f} vs yoked {y:.3f} vs none {n:.3f}  -> {tag}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
