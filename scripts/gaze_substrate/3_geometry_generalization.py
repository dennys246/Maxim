#!/usr/bin/env python
"""Step 3 — does the SUBSTRATE (EC pattern completion) buy generalization?

v1/v2 proved operant gaze learning + cross-session persistence on real NAc,
but with SYNTHETIC string bins: `near_left` and `far_left` are distinct keys
with zero transfer. That's tabular RL that merely persists through Maxim's
machinery. Step 3 asks the question that decides whether "substrate" means
anything here:

  Train the agent ONLY on near offsets. Freeze. Probe FAR offsets it never
  saw. Does it pan the correct DIRECTION zero-shot?

  - synthetic bins  -> far bins were never created -> recommend_action returns
                       None -> random -> directedness ~= chance. (No transfer.)
  - EC encoding     -> a far-bearing embedding may pattern-complete to a
                       same-side NEAR node (cosine >= threshold) and inherit its
                       learned action. Transfer is possible.

Bearing -> embedding is a place-cell population code (overlapping Gaussian
tuning curves over basis angles). Tuning width sigma controls generalization.
We SWEEP sigma and expect an inverted-U on far-directedness:
  sigma too small -> near/far don't overlap -> no transfer (~ synthetic)
  sigma too large -> left/right also overlap -> wrong-direction errors
A monotonic "bigger is better" would mean we're just erasing discrimination,
so we also report the WRONG-direction rate.

Honest controls: directedness is measured read-only on a held-out far probe
set; EC eval does NOT mutate the substrate; "sensor" modality is frozen
(no running-mean centroid drift — see the EC drift lesson in CLAUDE.md).
"""

from __future__ import annotations

import os

import numpy as np

os.environ.pop("MAXIM_NAC_REWARD_BIAS_DISABLED", None)
from maxim.decisions.nac import NAc, NACConfig  # noqa: E402
from maxim.similarity.ec import ECConfig, EntorhinalCortex  # noqa: E402

GAZE_LIMIT = 90.0
FOVEA = 10.0
NEAR_LO, NEAR_HI = 10.0, 22.0  # trained band (capped < 25 so synthetic's
#                                    `mid` bin is NEVER touched in training)
FAR_LO, FAR_HI = 30.0, 44.0  # held-out probe band (never trained, all `mid`/`far`)
ACTIONS = {
    "pan_left_big": -20.0,
    "pan_left_med": -10.0,
    "pan_left_small": -4.0,
    "stay": 0.0,
    "pan_right_small": +4.0,
    "pan_right_med": +10.0,
    "pan_right_big": +20.0,
}
ANAMES = list(ACTIONS)
EPSILON = 0.15
MIN_CONF = 0.05
THRESHOLD = 0.44  # EC default pattern_complete_threshold
BASIS = np.arange(-55.0, 56.0, 5.0)  # place-cell basis angles


def bearing_emb(rel: float, sigma: float) -> list[float]:
    v = np.exp(-((rel - BASIS) ** 2) / (2.0 * sigma**2))
    n = float(np.linalg.norm(v))
    return (v / n).tolist() if n > 0 else v.tolist()


def synth_bin(rel: float) -> str:
    a = abs(rel)
    if a <= FOVEA:
        return "center"
    side = "left" if rel < 0 else "right"
    if a <= 25.0:
        return f"near_{side}"
    if a <= 40.0:
        return f"mid_{side}"
    return f"far_{side}"


def cos(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else 0.0


def train(encoder: str, sigma: float, seed: int, ticks: int = 4000):
    """Full-dynamics training with subjects confined to the NEAR band.
    Returns (nac, ec_or_None)."""
    rng = np.random.default_rng(seed)
    nac = NAc(NACConfig())
    ec = None
    if encoder == "ec":
        ec = EntorhinalCortex(
            ECConfig(
                pattern_complete_threshold=THRESHOLD,
                frozen_centroid_modalities=frozenset({"interoception", "audio", "sensor"}),
            )
        )

    gaze = float(rng.uniform(-40, 40))
    subj = gaze + float(rng.uniform(-22, 22))
    drift = float(rng.choice([-1.0, 1.0]))

    def state_of(rel: float) -> str:
        if encoder == "synthetic":
            return synth_bin(rel)
        emb = bearing_emb(rel, sigma)
        res = ec.pattern_complete_or_separate(emb, "sensor")
        # EC is stateless on the separation path — the caller registers the
        # new node (mirrors LinguisticEncoder's register-after-ATL contract).
        if res.is_new:
            ec.register_substrate_node(res.node_id, emb, "sensor")
        return res.node_id

    for _ in range(ticks):
        rel = subj - gaze
        pre_state = state_of(rel)
        abs(rel) <= FOVEA

        if rng.random() < EPSILON:
            action = str(rng.choice(ANAMES))
        else:
            rec = nac.recommend_action(
                agent_id="a",
                available_tools=ANAMES,
                current_drives=None,
                current_cluster_id=pre_state,
                min_confidence=MIN_CONF,
            )
            action = rec["tool_name"] if rec else str(rng.choice(ANAMES))
        gaze = float(np.clip(gaze + ACTIONS[action], -GAZE_LIMIT, GAZE_LIMIT))

        rel_after = subj - gaze
        if abs(rel_after) <= FOVEA:
            reward = 1.0
        elif NEAR_LO < abs(rel_after) <= NEAR_HI:
            reward = 0.1
        else:
            reward = 0.0
        reward = reward if rng.random() < 0.6 else 0.0
        nac.update_cluster_reward("a", pre_state, f"tool:{action}", reward)

        # subject slow walk, kept inside the near band (so FAR is never trained)
        if rng.random() < 0.1:
            drift = -drift
        subj = subj + drift * 3.0 * rng.uniform(0.3, 1.0)
        if abs(subj - gaze) > NEAR_HI or abs(subj) >= GAZE_LIMIT:
            subj = gaze + float(rng.uniform(-19, 19))  # respawn strictly inside near band
    return nac, ec


def best_trained_node(ec: EntorhinalCortex, emb: list[float]) -> str | None:
    """Read-only nearest trained node (does NOT mutate the substrate)."""
    e = np.array(emb)
    best_id, best_sim = None, -1.0
    for nid, (semb, smod) in ec._substrate_nodes.items():
        if smod != "sensor":
            continue
        s = cos(e, np.array(semb))
        if s > best_sim:
            best_sim, best_id = s, nid
    return best_id if best_sim >= THRESHOLD else None


def eval_band(encoder, nac, ec, sigma, lo, hi, seed, trials=600):
    """Probe the FROZEN policy on random offsets in [lo, hi]. Returns
    (directedness, wrong_dir_rate, transfer_rate)."""
    rng = np.random.default_rng(seed)
    correct = wrong = matched = matched_correct = 0
    for _ in range(trials):
        float(rng.uniform(-45, 45))
        side = float(rng.choice([-1.0, 1.0]))
        rel = side * float(rng.uniform(lo, hi))
        node = None
        if encoder == "synthetic":
            rec = nac.recommend_action(
                agent_id="a",
                available_tools=ANAMES,
                current_drives=None,
                current_cluster_id=synth_bin(rel),
                min_confidence=MIN_CONF,
            )
        else:
            node = best_trained_node(ec, bearing_emb(rel, sigma))
            if node is not None:
                matched += 1
            rec = (
                nac.recommend_action(
                    agent_id="a",
                    available_tools=ANAMES,
                    current_drives=None,
                    current_cluster_id=node,
                    min_confidence=MIN_CONF,
                )
                if node
                else None
            )
        action = rec["tool_name"] if rec else str(rng.choice(ANAMES))
        delta = ACTIONS[action]
        is_corr = delta != 0 and (delta < 0) == (rel < 0)
        if is_corr:
            correct += 1
        elif delta != 0:
            wrong += 1
        if node is not None and is_corr:
            matched_correct += 1
    cond = matched_correct / matched if matched else 0.0  # dir-correct GIVEN a transfer fired
    return correct / trials, wrong / trials, matched / trials, cond


def main():
    seeds = list(range(8))
    print("\nStep 3 — EC generalization: train NEAR only, probe FAR (held-out)\n")
    print("Chance directedness ~ 3/7 = 0.429 (random over 7 actions).\n")

    # synthetic baseline (sigma-independent)
    syn_near, syn_far = [], []
    for s in seeds:
        nac, _ = train("synthetic", sigma=0.0, seed=s)
        syn_near.append(eval_band("synthetic", nac, None, 0.0, NEAR_LO, NEAR_HI, 500 + s)[0])
        syn_far.append(eval_band("synthetic", nac, None, 0.0, FAR_LO, FAR_HI, 700 + s)[0])
    print(
        f"  {'SYNTHETIC bins':<22}  near(trained) dir={np.mean(syn_near):.3f}   "
        f"FAR(held-out) dir={np.mean(syn_far):.3f}   transfer=0.000 (distinct keys)\n"
    )

    print(f"  {'EC sigma':<10}{'near dir':<10}{'FAR dir':<10}{'FAR wrong':<11}{'transfer%':<11}{'dir|matched':<12}")
    for sigma in (4.0, 8.0, 15.0, 25.0, 40.0):
        near_d, far_d, far_w, far_t, far_c = [], [], [], [], []
        for s in seeds:
            nac, ec = train("ec", sigma=sigma, seed=s)
            near_d.append(eval_band("ec", nac, ec, sigma, NEAR_LO, NEAR_HI, 500 + s)[0])
            d, w, t, c = eval_band("ec", nac, ec, sigma, FAR_LO, FAR_HI, 700 + s)
            far_d.append(d)
            far_w.append(w)
            far_t.append(t)
            far_c.append(c)
        print(
            f"  {sigma:<10.0f}{np.mean(near_d):<10.3f}{np.mean(far_d):<10.3f}"
            f"{np.mean(far_w):<11.3f}{np.mean(far_t):<11.3f}{np.mean(far_c):<12.3f}"
        )

    print("\n  'dir|matched' = direction-correct GIVEN a transfer fired. High value =>")
    print("  when EC pattern-completes an unseen far offset onto a trained node, it")
    print("  inherits the CORRECT direction. 'transfer%' = how often that happens.")
    print("  Synthetic's transfer is 0 by construction (distinct keys, never matches).")


if __name__ == "__main__":
    main()
