#!/usr/bin/env python
"""Step 4 — VISUAL-category transfer: the test where the substrate is provably
load-bearing.

Steps 1-3 encoded only gaze GEOMETRY (a bearing), where a dict is nearly as
good as EC. The user's point: visual percepts are themselves substrate products,
and we never encoded visual CONTENT. The substrate's real power is recognizing
*what a thing is* and generalizing a learned value to NOVEL instances of that
category — something a dict cannot do, because the appearance space is unbounded.

Task ("orient or ignore"):
  Each tick one candidate entity appears with a feature vector (its appearance)
  and a hidden type: PERSON or DISTRACTOR. The agent picks orient | ignore.
    orient + person     -> +1     ignore + person     -> -1   (missed a person)
    orient + distractor -> -1     ignore + distractor -> +1   (correctly skipped)
  So the rewarding behaviour ("look at people, skip objects") is LEARNED from
  reward, not assigned. State = the encoded appearance; action value is learned
  per state via the real NAc cluster-reward; selection via recommend_action.

Two encoders:
  dict : state = quantized exact feature vector. Memorizes SEEN individuals;
         a novel individual is a brand-new key -> no learned value -> chance.
  EC   : state = EntorhinalCortex.pattern_complete_or_separate(vec,"vision").
         People cluster in feature space -> one "person" node -> a novel person
         pattern-completes to it and inherits the learned "orient".

Held-out test = NOVEL individuals (fresh feature draws from the same category
prototypes, never seen in training). Metric = discrimination:
    P(orient | person) - P(orient | distractor)
  dict -> ~0 on novel (can't recognize).  EC -> high (transfers the category).

Honest bound (swept): EC transfer only works if the encoder actually CLUSTERS
the category. We sweep within-category spread `eps`; as people become more
visually diverse than the encoder can cluster, EC transfer collapses toward the
dict. The substrate's generalization is only as good as the embedding geometry
it is handed.
"""

from __future__ import annotations

import os

import numpy as np

os.environ.pop("MAXIM_NAC_REWARD_BIAS_DISABLED", None)
from maxim.decisions.nac import NAc, NACConfig          # noqa: E402
from maxim.similarity.ec import ECConfig, EntorhinalCortex  # noqa: E402

D = 16
ACTIONS = ["orient", "ignore"]
EPSILON = 0.15
MIN_CONF = 0.05
THRESHOLD = 0.44

# Fixed category geometry (shared across arms/seeds): one person prototype,
# three distractor prototypes. Random vectors in 16-d are near-orthogonal.
_g = np.random.default_rng(12345)


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


P_PERSON = _unit(_g.standard_normal(D))
P_OBJS = [_unit(_g.standard_normal(D)) for _ in range(3)]


def make_individual(rng, kind: str, eps: float):
    proto = P_PERSON if kind == "person" else P_OBJS[int(rng.integers(len(P_OBJS)))]
    return _unit(proto + eps * rng.standard_normal(D))


def reward_of(action: str, kind: str) -> float:
    good = (action == "orient" and kind == "person") or (action == "ignore" and kind == "distractor")
    return 1.0 if good else -1.0


def quant_key(vec) -> str:
    return ",".join(str(int(round(x * 50))) for x in vec)   # stable for identical vecs


def cos(a, b) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else 0.0


def best_node(ec, vec):
    e = np.array(vec)
    best_id, best_sim = None, -1.0
    for nid, (semb, smod) in ec._substrate_nodes.items():
        if smod != "vision":
            continue
        s = cos(e, np.array(semb))
        if s > best_sim:
            best_sim, best_id = s, nid
    return best_id if best_sim >= THRESHOLD else None


def run_arm(encoder: str, eps: float, seed: int, n_train=8, train_ticks=3000, test_n=400):
    rng = np.random.default_rng(seed)
    nac = NAc(NACConfig())
    ec = None
    if encoder == "ec":
        ec = EntorhinalCortex(ECConfig(
            pattern_complete_threshold=THRESHOLD,
            frozen_centroid_modalities=frozenset({"interoception", "audio", "vision"}),
        ))

    # fixed training individuals (so the dict CAN memorize seen ones)
    train_pool = {"person": [make_individual(rng, "person", eps) for _ in range(n_train)],
                  "distractor": [make_individual(rng, "distractor", eps) for _ in range(n_train)]}

    def state_of(vec) -> str:
        if encoder == "dict":
            return quant_key(vec)
        emb = list(map(float, vec))
        res = ec.pattern_complete_or_separate(emb, "vision")
        if res.is_new:
            ec.register_substrate_node(res.node_id, emb, "vision")
        return res.node_id

    # --- train ---
    for _ in range(train_ticks):
        kind = "person" if rng.random() < 0.5 else "distractor"
        vec = train_pool[kind][int(rng.integers(n_train))]
        state = state_of(vec)
        if rng.random() < EPSILON:
            action = ACTIONS[int(rng.integers(2))]
        else:
            rec = nac.recommend_action(agent_id="a", available_tools=ACTIONS,
                                       current_drives=None, current_cluster_id=state,
                                       min_confidence=MIN_CONF)
            action = rec["tool_name"] if rec else ACTIONS[int(rng.integers(2))]
        nac.update_cluster_reward("a", state, f"tool:{action}", reward_of(action, kind))

    # --- frozen eval ---
    def p_orient(kind: str, novel: bool) -> float:
        orient = 0
        for _ in range(test_n):
            vec = make_individual(rng, kind, eps) if novel else train_pool[kind][int(rng.integers(n_train))]
            if encoder == "dict":
                state = quant_key(vec)
                rec = nac.recommend_action(agent_id="a", available_tools=ACTIONS,
                                           current_drives=None, current_cluster_id=state,
                                           min_confidence=MIN_CONF)
            else:
                node = best_node(ec, vec)
                rec = (nac.recommend_action(agent_id="a", available_tools=ACTIONS,
                                            current_drives=None, current_cluster_id=node,
                                            min_confidence=MIN_CONF) if node else None)
            action = rec["tool_name"] if rec else ACTIONS[int(rng.integers(2))]
            orient += action == "orient"
        return orient / test_n

    seen = p_orient("person", False) - p_orient("distractor", False)
    novel = p_orient("person", True) - p_orient("distractor", True)
    nodes = len(ec._substrate_nodes) if ec is not None else 0   # EC.__len__ counts signatures(=0); use is-not-None
    return seen, novel, nodes


def main():
    seeds = list(range(8))
    print("\nStep 4 — visual-category transfer to NOVEL instances")
    print("discrimination = P(orient|person) - P(orient|distractor);  1.0=perfect, 0=chance\n")

    # dict baseline (eps=0.15)
    ds, dn = [], []
    for s in seeds:
        seen, novel, _ = run_arm("dict", 0.15, s)
        ds.append(seen); dn.append(novel)
    print(f"  {'DICT (exact vec)':<20}  seen={np.mean(ds):.3f}   NOVEL={np.mean(dn):.3f}   "
          "(novel = brand-new key -> chance)\n")

    print(f"  {'EC eps':<10}{'seen':<9}{'NOVEL':<9}{'person-nodes(approx)':<22}")
    for eps in (0.10, 0.15, 0.30, 0.60, 1.00):
        es, en, nn = [], [], []
        for s in seeds:
            seen, novel, nodes = run_arm("ec", eps, s)
            es.append(seen); en.append(novel); nn.append(nodes)
        print(f"  {eps:<10.2f}{np.mean(es):<9.3f}{np.mean(en):<9.3f}{np.mean(nn):<22.1f}")

    print("\n  Read: EC NOVEL >> dict NOVEL => the substrate recognized never-seen people")
    print("  as people and transferred the learned 'orient'. As eps grows (people more")
    print("  visually diverse than the encoder can cluster) EC NOVEL collapses toward dict")
    print("  and node count explodes -> substrate transfer is bounded by embedding geometry.")


if __name__ == "__main__":
    main()
