#!/usr/bin/env python
"""Operant gaze probe v2 — harder world: search (Layer 0) + multi-subject coverage.

Builds on the validated v1 (gaze_operant_probe.py) which proved reward
redirects gaze on real NAc cluster-reward, with a yoked control ruling out
superstition. v1 was too easy (cross-session carryover only +0.107). v2 adds:

  Step 2 (search / Layer 0): FINITE FOV. A subject outside the FOV is not
    sensed -> the perceptual state is `empty_<last-seen-side>`. The agent must
    pan to re-acquire. Acquisition (empty -> in-FOV) earns a small reward, so
    *directional search* becomes a learnable contingency.

  Step 1 (harder world for a real cross-session signal):
    - finer bins (7 in-FOV + 2 empty = 9 states), more pan actions (7)
    - probabilistic reward (a centered frame only sometimes pays) -> sparser
    - MULTIPLE subjects + HABITUATION: a subject's reward (freshness) decays
      while you stare and recovers while you don't. Without it the optimal
      policy is to lock one subject forever (the "beautiful starer"); with it
      the salient target rotates, so coverage behaviour must emerge.

Same three arms as v1, kept as the bug detector:
  contingent / yoked (reward -> random (state,action)) / none (no credit).
`none` and `yoked` MUST stay at chance; if they don't, the world leaks
directedness and the result is invalid.

Reward shape (preserves "nothing without a subject in view"):
  empty            -> 0
  in-FOV, off-fovea-> 0.1 * freshness * (1 - |rel|/FOV_HALF)   [acquisition gradient]
  in-FOV, foveated -> 1.0 * freshness                          [the payoff]
  then gated by P_REWARD (probabilistic).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

os.environ.pop("MAXIM_NAC_REWARD_BIAS_DISABLED", None)
from maxim.decisions.nac import NAc, NACConfig  # noqa: E402

GAZE_LIMIT = 90.0
FOVEA = 10.0
ACTIONS = {
    "pan_left_big": -20.0, "pan_left_med": -10.0, "pan_left_small": -4.0,
    "stay": 0.0,
    "pan_right_small": +4.0, "pan_right_med": +10.0, "pan_right_big": +20.0,
}
ACTION_NAMES = list(ACTIONS)
EMPTY_STATES = ["empty_left", "empty_right"]
ALL_STATES = ["far_left", "mid_left", "near_left", "center",
              "near_right", "mid_right", "far_right"] + EMPTY_STATES
EPSILON = 0.15
MIN_CONF = 0.05
DRIFT = 3.0


@dataclass
class Config:
    num_subjects: int = 1
    fov_half: float = 45.0
    p_reward: float = 0.4
    habituation: bool = False
    hab_decay: float = 0.90       # freshness *= this while foveated
    hab_recover: float = 0.03     # freshness += this while not foveated
    transient: bool = True        # subjects respawn on lifetime end; False = persistent (lockable)


@dataclass
class Subject:
    bearing: float
    drift_dir: float
    freshness: float = 1.0
    life: int = 0
    sid: int = 0


def in_fov_bin(rel: float, fov_half: float) -> str:
    a = abs(rel)
    if a <= FOVEA:
        return "center"
    side = "left" if rel < 0 else "right"
    if a <= 25.0:
        return f"near_{side}"
    if a <= 40.0:
        return f"mid_{side}"
    return f"far_{side}"


def run_session(arm: str, cfg: Config, seed: int, nac: NAc, agent_id: str, ticks: int = 2500):
    rng = np.random.default_rng(seed)
    gaze = float(rng.uniform(-GAZE_LIMIT, GAZE_LIMIT))
    next_sid = [0]

    def spawn() -> Subject:
        life = 10 ** 9 if not cfg.transient else int(rng.integers(150, 400))
        s = Subject(bearing=float(rng.uniform(-GAZE_LIMIT, GAZE_LIMIT)),
                    drift_dir=float(rng.choice([-1.0, 1.0])),
                    freshness=1.0, life=life, sid=next_sid[0])
        next_sid[0] += 1
        return s

    subjects = [spawn() for _ in range(cfg.num_subjects)]
    last_side = "right"

    fov_series, foveated_series, directed_series, foveated_sid = [], [], [], []

    def sense(g: float):
        """Return (state, target_subject, rel) given gaze g. Salience=freshness."""
        visible = [(s, s.bearing - g) for s in subjects if abs(s.bearing - g) <= cfg.fov_half]
        if not visible:
            return None, None, None
        s, rel = max(visible, key=lambda sr: sr[0].freshness)
        return in_fov_bin(rel, cfg.fov_half), s, rel

    for _ in range(ticks):
        # --- perceive (pre-action) ---
        pre_state, pre_tgt, pre_rel = sense(gaze)
        if pre_state is None:
            pre_state = f"empty_{last_side}"
        else:
            last_side = "left" if pre_rel < 0 else "right"
        foveated_before = pre_tgt is not None and abs(pre_rel) <= FOVEA

        # --- act (epsilon-greedy over substrate opinion) ---
        if rng.random() < EPSILON:
            action = str(rng.choice(ACTION_NAMES))
        else:
            rec = nac.recommend_action(agent_id=agent_id, available_tools=ACTION_NAMES,
                                       current_drives=None, current_cluster_id=pre_state,
                                       min_confidence=MIN_CONF)
            action = rec["tool_name"] if rec else str(rng.choice(ACTION_NAMES))
        delta = ACTIONS[action]
        gaze = float(np.clip(gaze + delta, -GAZE_LIMIT, GAZE_LIMIT))

        # --- resulting state + reward ---
        post_state, post_tgt, post_rel = sense(gaze)
        if post_tgt is None:
            reward = 0.0
            foveated = False
        else:
            fov = abs(post_rel) <= FOVEA
            if fov:
                reward_raw = 1.0 * post_tgt.freshness
            else:
                reward_raw = 0.1 * post_tgt.freshness * (1.0 - abs(post_rel) / cfg.fov_half)
            reward = reward_raw if rng.random() < cfg.p_reward else 0.0
            foveated = fov

        # --- directedness (mechanism check) ---
        if pre_tgt is None:                       # searching: pan toward last-seen side?
            good = (last_side == "left" and delta < 0) or (last_side == "right" and delta > 0)
        elif foveated_before:
            good = action == "stay"
        else:
            good = post_tgt is not None and abs(post_rel) < abs(pre_rel)
        directed_series.append(1 if good else 0)
        fov_series.append(1 if post_tgt is not None else 0)
        foveated_series.append(1 if foveated else 0)
        foveated_sid.append(post_tgt.sid if foveated else -1)

        # --- credit (only difference between arms) ---
        if arm == "contingent":
            nac.update_cluster_reward(agent_id, pre_state, f"tool:{action}", reward)
        elif arm == "yoked" and reward > 0:
            rb = str(rng.choice(ALL_STATES)); ra = str(rng.choice(ACTION_NAMES))
            nac.update_cluster_reward(agent_id, rb, f"tool:{ra}", reward)

        # --- habituation: drain the stared-at subject, recover the rest ---
        if cfg.habituation:
            for s in subjects:
                if foveated and post_tgt is not None and s.sid == post_tgt.sid:
                    s.freshness *= cfg.hab_decay
                else:
                    s.freshness = min(1.0, s.freshness + cfg.hab_recover)

        # --- subject dynamics: slow walk + transient respawn ---
        for i, s in enumerate(subjects):
            if rng.random() < 0.1:
                s.drift_dir = -s.drift_dir
            s.bearing = float(np.clip(s.bearing + s.drift_dir * DRIFT * rng.uniform(0.3, 1.0),
                                      -GAZE_LIMIT, GAZE_LIMIT))
            if abs(s.bearing) >= GAZE_LIMIT - 1e-6:
                s.drift_dir = -s.drift_dir
            s.life -= 1
            if s.life <= 0:
                subjects[i] = spawn()

    return {
        "in_fov": fov_series, "foveated": foveated_series,
        "directed": directed_series, "foveated_sid": foveated_sid,
    }


def win(series, n):
    return [float(w.mean()) for w in np.array_split(np.array(series, float), n)]


def coverage_per_window(sids, n):
    """Distinct subject-ids foveated in each window (anti-fixation outcome)."""
    out = []
    for w in np.array_split(np.array(sids), n):
        out.append(len(set(int(x) for x in w if x >= 0)))
    return out


def run_arms(cfg: Config, label: str, agents=16, ticks=2500, n_windows=6):
    print(f"\n########## {label} ##########")
    print(f"  subjects={cfg.num_subjects} fov_half={cfg.fov_half} p_reward={cfg.p_reward} "
          f"habituation={cfg.habituation}  | {agents} agents x {ticks} ticks")
    res = {}
    for arm in ["contingent", "yoked", "none"]:
        fov, foveated, directed = [], [], []
        for a in range(agents):
            nac = NAc(NACConfig())
            r = run_session(arm, cfg, seed=1000 * a + (hash(arm + label) % 97),
                            nac=nac, agent_id=f"{arm}_{a}")
            fov.append(win(r["in_fov"], n_windows))
            foveated.append(win(r["foveated"], n_windows))
            directed.append(win(r["directed"], n_windows))
        res[arm] = {"in_fov": np.array(fov), "foveated": np.array(foveated),
                    "directed": np.array(directed)}
    for metric, name in [("foveated", "FOVEATION (pursuit)"),
                         ("in_fov", "IN-FOV / acquisition (search, Layer 0)"),
                         ("directed", "DIRECTEDNESS (mechanism)")]:
        print(f"  -- {name} --")
        print(f"     {'arm':<11}" + "".join(f"w{i+1:<6}" for i in range(n_windows)))
        for arm in ["contingent", "yoked", "none"]:
            m = res[arm][metric].mean(axis=0)
            print(f"     {arm:<11}" + "".join(f"{v:<7.3f}" for v in m))
    return res


def cross_session(cfg: Config, label: str, pairs=16, eval_ticks=600, n_windows=6):
    """Powered version: average over many (trained -> loaded vs fresh) pairs on a
    SHORT eval session, so the ~learning-phase head-start isn't diluted by the
    long converged tail, and per-pair noise averages out."""
    print(f"\n  == CROSS-SESSION ({label}): {pairs} pairs, {eval_ticks}-tick eval ==")
    loaded_curves, fresh_curves, gaps = [], [], []
    for p in range(pairs):
        trained = NAc(NACConfig())
        run_session("contingent", cfg, seed=5000 + p, nac=trained, agent_id="xs", ticks=2500)
        state = trained.dump()
        ev = 9000 + p                                   # eval dynamics shared by both
        loaded = NAc(NACConfig()); loaded.load_state(state)
        rl = run_session("contingent", cfg, seed=ev, nac=loaded, agent_id="xs", ticks=eval_ticks)
        fresh = NAc(NACConfig())
        rf = run_session("contingent", cfg, seed=ev, nac=fresh, agent_id="xs2", ticks=eval_ticks)
        loaded_curves.append(win(rl["foveated"], n_windows))
        fresh_curves.append(win(rf["foveated"], n_windows))
        gaps.append(float(np.mean(rl["foveated"]) - np.mean(rf["foveated"])))
    wl = np.array(loaded_curves).mean(axis=0)
    wf = np.array(fresh_curves).mean(axis=0)
    print(f"     {'':<8}" + "".join(f"w{i+1:<6}" for i in range(n_windows)))
    print(f"     {'loaded':<8}" + "".join(f"{v:<7.3f}" for v in wl))
    print(f"     {'fresh':<8}" + "".join(f"{v:<7.3f}" for v in wf))
    g = np.array(gaps)
    se = g.std() / np.sqrt(len(g))
    print(f"     whole-eval foveation advantage (loaded-fresh) = {g.mean():+.3f} "
          f"+/- {se:.3f} SE  over {pairs} pairs  (v1 was +0.107, noisy single-pair)")


def dwell_concentration(sids):
    """Of all foveated ticks, fraction spent on the single most-foveated subject.
    High => fixation on one subject; ~1/N => even coverage."""
    fov = [int(x) for x in sids if x >= 0]
    if not fov:
        return 0.0
    counts = {}
    for x in fov:
        counts[x] = counts.get(x, 0) + 1
    return max(counts.values()) / len(fov)


def coverage_demo(agents=16, ticks=2500, n_windows=6):
    print("\n########## ANTI-FIXATION: 3 PERSISTENT subjects, habituation ON vs OFF ##########")
    print("  (persistent = lockable, so the 'beautiful starer' failure is now possible)")
    for hab in (False, True):
        cfg = Config(num_subjects=3, habituation=hab, transient=False)
        fovs, covs, conc = [], [], []
        for a in range(agents):
            nac = NAc(NACConfig())
            r = run_session("contingent", cfg, seed=2000 + a, nac=nac,
                            agent_id=f"cov_{hab}_{a}", ticks=ticks)
            fovs.append(np.mean(r["foveated"]))
            covs.append(np.mean(coverage_per_window(r["foveated_sid"], n_windows)))
            conc.append(dwell_concentration(r["foveated_sid"]))
        tag = "habituation ON " if hab else "habituation OFF"
        print(f"  {tag}:  foveation={np.mean(fovs):.3f}   "
              f"distinct/window={np.mean(covs):.2f}/3   "
              f"dwell-on-top-subject={np.mean(conc):.3f}")
    print("  (OFF fixates -> high dwell-on-top + low distinct. "
          "ON -> lower dwell concentration, more distinct subjects.)")


def main():
    cfgA = Config(num_subjects=1, fov_half=45.0, p_reward=0.4, habituation=False)
    run_arms(cfgA, "CONFIG A: search + finer bins + probabilistic reward")
    cross_session(cfgA, "Config A")
    coverage_demo()


if __name__ == "__main__":
    main()
