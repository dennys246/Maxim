#!/usr/bin/env python
"""Operant gaze-redirection probe on Maxim's real NAc cluster-reward machinery.

Question
--------
If gaze starts as random motor babbling and reward fires only when a subject
is foveated, does the NAc reward signal *redirect* behaviour toward the
subject — or is any change just superstition (Skinner) from reward arriving
during random activity?

Design (no LLM, no EC, no orchestrator — isolates the operant mechanism)
-----------------------------------------------------------------------
1D pan world. A subject sits at a bearing and slow-random-walks. The agent's
gaze is one of 5 discrete pan actions. Relative bearing (subject - gaze) is
binned into 5 perceptual states = NAc *clusters*. The action is the NAc
*tool*. Learning rides entirely on:

    nac.update_cluster_reward(agent_id, pre_bin, "tool:<action>", reward)   # credit
    nac.recommend_action(agent_id, available_tools, current_cluster_id=bin) # select

Reward = +1 iff the subject is within the fovea after the move, else 0
("nothing without it"). ε-greedy: explore randomly, else follow substrate.

Three arms (the falsifiable comparison)
---------------------------------------
  contingent : credit the (state-it-was-chosen-in, action) pair        [real]
  yoked      : credit a RANDOM (state, action) pair at the same reward  [superstition control]
  none       : never credit                                            [chance baseline]

If contingent > none AND contingent > yoked  -> real operant redirection.
If contingent ~= yoked                        -> superstition; the contingency
                                                 wasn't learnable as set up.

Cross-session (the thesis): dump() NAc after a session, load_state() into a
fresh agent, and check whether session 2 *starts* more directed than
session 1 did — learning carried across sessions with no fine-tuning.

Notes / scoping (deliberate, documented per CLAUDE.md no-bandaid rule)
---------------------------------------------------------------------
* Subject is always sensed as a bearing (peripheral vision). The Layer-0
  entropic "search when FOV is empty" drive is a v2 extension — this probe
  isolates pursuit-redirection only.
* decay_cluster_reward_biases() is NOT called: we measure raw accumulation.
  Production decays per tick (tau); add it back when calibrating retention.
* The `none` arm is the empirical chance baseline under identical dynamics,
  which is cleaner than an analytic chance level.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

# Keep the substrate honest: no reward-bias disable leaking in from the env.
os.environ.pop("MAXIM_NAC_REWARD_BIAS_DISABLED", None)

from maxim.decisions.nac import NAc, NACConfig  # noqa: E402

# ---- world / policy constants -------------------------------------------------
GAZE_LIMIT = 90.0
FOVEA = 10.0            # |rel| <= FOVEA  => foveated => reward
NEAR_EDGE = 35.0        # boundary between near and far bins
ACTIONS = {             # name -> gaze delta (degrees)
    "pan_left_big": -15.0,
    "pan_left_small": -5.0,
    "stay": 0.0,
    "pan_right_small": +5.0,
    "pan_right_big": +15.0,
}
ACTION_NAMES = list(ACTIONS)
EPSILON = 0.15          # exploration rate (the random motor-babble prior)
MIN_CONF = 0.05         # recommend_action gate: any single reinforced hit (0.15) clears it
DRIFT = 3.0             # subject random-walk step (deg/tick)


def bin_of(rel: float) -> str:
    """Map relative bearing (subject - gaze) to a perceptual cluster id."""
    if abs(rel) <= FOVEA:
        return "center"
    if rel < -NEAR_EDGE:
        return "far_left"
    if rel < 0:
        return "near_left"
    if rel > NEAR_EDGE:
        return "far_right"
    return "near_right"


def run_session(arm: str, seed: int, ticks: int, nac: NAc, agent_id: str) -> list[float]:
    """Run one agent for `ticks`. Returns per-tick foveated(0/1) and logs
    directedness via the returned 'good' list. Mutates `nac` in place."""
    rng = np.random.default_rng(seed)
    gaze = float(rng.uniform(-GAZE_LIMIT, GAZE_LIMIT))
    subject = float(rng.uniform(-GAZE_LIMIT, GAZE_LIMIT))
    drift_dir = float(rng.choice([-1.0, 1.0]))

    foveated_series: list[int] = []
    good_series: list[int] = []

    for _ in range(ticks):
        rel = subject - gaze
        pre_bin = bin_of(rel)
        foveated_before = abs(rel) <= FOVEA

        # --- action selection: ε-greedy over the substrate's opinion ---
        if rng.random() < EPSILON:
            action = str(rng.choice(ACTION_NAMES))
        else:
            rec = nac.recommend_action(
                agent_id=agent_id,
                available_tools=ACTION_NAMES,
                current_drives=None,            # skip name-substring heuristic
                current_cluster_id=pre_bin,
                min_confidence=MIN_CONF,
            )
            action = rec["tool_name"] if rec else str(rng.choice(ACTION_NAMES))

        # --- apply action ---
        gaze = float(np.clip(gaze + ACTIONS[action], -GAZE_LIMIT, GAZE_LIMIT))
        rel_after = subject - gaze
        foveated_after = abs(rel_after) <= FOVEA
        reward = 1.0 if foveated_after else 0.0

        # --- directedness: did this move help? (mechanism check) ---
        if foveated_before:
            good = action == "stay"               # already centred: hold
        else:
            good = abs(rel_after) < abs(rel)       # moved subject toward centre
        good_series.append(1 if good else 0)
        foveated_series.append(1 if foveated_after else 0)

        # --- credit assignment (the only difference between arms) ---
        if arm == "contingent":
            nac.update_cluster_reward(agent_id, pre_bin, f"tool:{action}", reward)
        elif arm == "yoked" and reward > 0:
            # same reward event, broken contingency -> superstition control
            rbin = str(rng.choice(["center", "far_left", "near_left", "near_right", "far_right"]))
            raction = str(rng.choice(ACTION_NAMES))
            nac.update_cluster_reward(agent_id, rbin, f"tool:{raction}", reward)
        # arm == "none": never credit

        # --- subject slow random walk ---
        if rng.random() < 0.1:
            drift_dir = -drift_dir
        subject = float(np.clip(subject + drift_dir * DRIFT * rng.uniform(0.3, 1.0),
                                -GAZE_LIMIT, GAZE_LIMIT))
        if abs(subject) >= GAZE_LIMIT - 1e-6:      # bounce off the edges
            drift_dir = -drift_dir

    return foveated_series, good_series


def windowed(series: list[int], n_windows: int) -> list[float]:
    arr = np.array(series, dtype=float)
    return [float(w.mean()) for w in np.array_split(arr, n_windows)]


def main() -> None:
    agents = int(os.environ.get("PROBE_AGENTS", "24"))
    ticks = int(os.environ.get("PROBE_TICKS", "1500"))
    n_windows = 6
    arms = ["contingent", "yoked", "none"]

    results: dict[str, dict] = {}
    for arm in arms:
        fov_curves, good_curves = [], []
        for a in range(agents):
            nac = NAc(NACConfig())
            agent_id = f"{arm}_agent_{a}"
            fov, good = run_session(arm, seed=1000 * a + hash(arm) % 97, ticks=ticks,
                                    nac=nac, agent_id=agent_id)
            fov_curves.append(windowed(fov, n_windows))
            good_curves.append(windowed(good, n_windows))
        results[arm] = {
            "foveation": np.array(fov_curves),   # (agents, windows)
            "directed": np.array(good_curves),
        }

    # ---- report ----
    print(f"\nOperant gaze probe  |  {agents} agents/arm  x  {ticks} ticks  |  {n_windows} windows\n")
    for metric, label in [("foveation", "FOVEATION RATE (outcome)"),
                          ("directed", "DIRECTEDNESS (mechanism)")]:
        print(f"== {label} ==")
        print(f"  {'window':<10}" + "".join(f"w{i+1:<7}" for i in range(n_windows)))
        for arm in arms:
            m = results[arm][metric].mean(axis=0)
            print(f"  {arm:<10}" + "".join(f"{v:<8.3f}" for v in m))
        print()

    # ---- verdict ----
    c = results["contingent"]["foveation"].mean(axis=0)[-1]
    y = results["yoked"]["foveation"].mean(axis=0)[-1]
    n = results["none"]["foveation"].mean(axis=0)[-1]
    # crude across-agent spread for a sanity CI on the final window
    c_sd = results["contingent"]["foveation"][:, -1].std()
    print("== VERDICT (final-window foveation rate) ==")
    print(f"  contingent = {c:.3f}  (sd {c_sd:.3f})")
    print(f"  yoked      = {y:.3f}")
    print(f"  none       = {n:.3f}")
    gap_n = c - n
    gap_y = c - y
    print(f"\n  contingent - none  = {gap_n:+.3f}   (reward changed behaviour?)")
    print(f"  contingent - yoked = {gap_y:+.3f}   (contingency, not just reward presence?)")
    if gap_n > 3 * (c_sd / np.sqrt(24)) and gap_y > 0.03:
        print("\n  -> Real operant redirection: reward bent gaze toward the subject,")
        print("     and the effect needs the *contingency* (yoked control did not match).")
    elif gap_n > 0.03 and gap_y <= 0.03:
        print("\n  -> SUPERSTITION RISK: contingent ~= yoked. Reward changed behaviour")
        print("     but not via the action->outcome contingency. Slow the subject /")
        print("     widen the fovea so the agent's pan can reliably cause centering.")
    else:
        print("\n  -> No redirection detected. Check exploration / reward sparsity.")

    # ---- cross-session (thesis): does a LOADED agent out-learn a FRESH one
    #      on identical dynamics?  Same eval seed cancels start-position luck;
    #      the gap in early windows is carryover through dump()/load_state(). ----
    print("\n== CROSS-SESSION (loaded vs fresh-cold, identical dynamics) ==")
    trained = NAc(NACConfig())
    run_session("contingent", seed=7, ticks=ticks, nac=trained, agent_id="xs")
    state = trained.dump()

    eval_seed = 8
    loaded = NAc(NACConfig())
    loaded.load_state(state)
    fov_loaded, _ = run_session("contingent", seed=eval_seed, ticks=ticks, nac=loaded, agent_id="xs")
    fresh = NAc(NACConfig())
    fov_fresh, _ = run_session("contingent", seed=eval_seed, ticks=ticks, nac=fresh, agent_id="xs2")

    wl = windowed(fov_loaded, n_windows)
    wf = windowed(fov_fresh, n_windows)
    print(f"  {'window':<10}" + "".join(f"w{i+1:<7}" for i in range(n_windows)))
    print(f"  {'loaded':<10}" + "".join(f"{v:<8.3f}" for v in wl))
    print(f"  {'fresh':<10}" + "".join(f"{v:<8.3f}" for v in wf))
    early_gap = float(np.mean(wl[:3]) - np.mean(wf[:3]))   # first-half advantage
    print(f"\n  early (w1-3) advantage of loaded over fresh = {early_gap:+.3f}")
    print("  (>0 => learning survived dump/load: the loaded agent is competent"
          "\n   sooner than a cold agent on the same world.)")


if __name__ == "__main__":
    sys.exit(main())
