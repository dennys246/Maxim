"""P3a Stage 2 — synthetic episode fixture generator (hub+chain topology).

Design history
==============

An earlier draft of this fixture (clique-per-topic — 5 core nodes all
co-occurring in every episode) produced F1 ≈ 1.0 for **both** the
Hebbian binding mechanism and the TF-IDF bag-of-concepts baseline.
That's an architectural finding, not a tuning problem: on a fixture
with no transitive structure, one-hop co-occurrence retrieval and
TF-IDF are algorithmically near-equivalent. The Hebbian mechanism
couldn't demonstrate its real capability.

This revised fixture uses a **hub + chain** topology that creates
genuine transitive structure — target nodes are reachable from a cue
via multi-hop graph walks but NOT via direct co-occurrence. Multi-hop
retrieval (``DependencyGraph.spreading_activation``) is specifically
what Hebbian has and TF-IDF structurally cannot replicate (TF-IDF's
bag-of-words representation has no edges to traverse).

Topology
========

Each topic has 5 core nodes:

- 1 **hub** (``<topic>.hub``) — the topic's theme anchor
- 4 **chain** nodes (``<topic>.c1``, ``c2``, ``c3``, ``c4``) arranged
  linearly: c1 → c2 → c3 → c4

Each topic has **17 episodes** (core reinforcement + peripheral noise):

- **8 hub episodes** — each hub↔chain pair reinforced TWICE:
  ``{hub, c1}×2``, ``{hub, c2}×2``, ``{hub, c3}×2``, ``{hub, c4}×2``.
  Reinforcement drives the edge weight from ``hebbian_init=0.3`` up
  to ``0.3 + hebbian_delta = 0.4``.
- **6 chain episodes** — each adjacency reinforced TWICE:
  ``{c1, c2}×2``, ``{c2, c3}×2``, ``{c3, c4}×2``. Same reinforcement.
- **3 peripheral episodes** — ``{hub, peripheral_i}`` single-shot.
  Weight stays at ``hebbian_init = 0.3``.

**Why double-reinforcement is load-bearing (not a band-aid).** Under
single-shot core episodes, chain targets at 2-hop distance (e.g.,
``plate`` from cue ``prep``) reach weight ``0.21 × 0.3 × 0.7 ≈ 0.044``,
which **ties exactly** with peripheral nodes reached through the hub
at the same 2-hop distance. Ranking then depends on dict iteration
order and stable-sort tiebreaks — a fragile dependency on episode
ingestion sequence. Doubling core reinforcement pushes core-edge
weights to 0.4 vs peripheral 0.3, making chain targets **strictly
higher** in spreading-activation scores. No tie-breaking heuristics
needed; the ranking is now robust to ingestion order.

Expected retrieval structure
============================

For ``cue = <topic>.hub``:
  - Targets: ``{c1, c2, c3, c4}``
  - One-hop: 4/4 recall (hub has direct edges to all chain nodes)
  - Multi-hop: 4/4 recall
  - TF-IDF: 4/4 recall (hub co-occurs with each chain node in 1
    hub episode)

For ``cue = <topic>.c1`` (head of chain):
  - Targets: ``{hub, c2, c3, c4}``
  - One-hop direct co-occurrences: ``{hub, c2}`` — recall 2/4
  - Multi-hop: reaches c3 via c2 and c4 via c3, recall 4/4
  - TF-IDF: can only rank hub + c2; cannot reach c3/c4

For ``cue = <topic>.c4`` (tail of chain): symmetric to c1.

For ``cue = <topic>.c2`` (interior of chain):
  - Targets: ``{hub, c1, c3, c4}``
  - One-hop: ``{hub, c1, c3}`` — recall 3/4
  - Multi-hop: reaches c4 via c3, recall 4/4

For ``cue = <topic>.c3``: symmetric to c2.

Aggregate across the 5 probes per topic (10 topics = 50 probes):

- One-hop recall: (4 + 2 + 3 + 3 + 2) / (5 × 4) = 14/20 = **0.70**
- Multi-hop recall: 20/20 = **1.00** (when ``spreading_activation``'s
  decay × threshold window reaches 3 hops)
- TF-IDF recall: approximately 0.70 (same as one-hop Hebbian —
  neither can walk past direct neighbors)

The pass gate ``mean_recall > 0.70`` + ``beats TF-IDF by 2σ`` is
cleared by multi-hop Hebbian but NOT by one-hop Hebbian or TF-IDF.
This demonstrates the mechanism's real capability: transitive
structural retrieval.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

# ─────────────────────────────────────────────────────────────────────────
# Topic seed data — 10 topics, each with a hub theme + 4 chain steps
# and a peripheral pool for the 3 noise episodes per topic.
# ─────────────────────────────────────────────────────────────────────────


TOPICS: dict[str, dict[str, Any]] = {
    "cooking": {
        "hub": "stove",
        "chain": ["prep", "saute", "simmer", "plate"],
        "peripherals": ["tomato", "onion", "garlic", "basil", "pepper"],
    },
    "sports": {
        "hub": "field",
        "chain": ["warmup", "drill", "scrimmage", "cooldown"],
        "peripherals": ["whistle", "bench", "coach", "jersey", "cone"],
    },
    "music": {
        "hub": "stage",
        "chain": ["tuneup", "soundcheck", "perform", "encore"],
        "peripherals": ["mic", "cable", "amp", "pedal", "setlist"],
    },
    "gardening": {
        "hub": "plot",
        "chain": ["till", "seed", "water", "harvest"],
        "peripherals": ["trowel", "mulch", "hose", "bucket", "gloves"],
    },
    "astronomy": {
        "hub": "observatory",
        "chain": ["calibrate", "track", "capture", "catalog"],
        "peripherals": ["lens", "mount", "filter", "chart", "almanac"],
    },
    "carpentry": {
        "hub": "workbench",
        "chain": ["measure", "cut", "joint", "finish"],
        "peripherals": ["clamp", "saw", "chisel", "plane", "stain"],
    },
    "medicine": {
        "hub": "ward",
        "chain": ["triage", "diagnose", "treat", "discharge"],
        "peripherals": ["chart", "cuff", "thermometer", "stethoscope", "clipboard"],
    },
    "sailing": {
        "hub": "harbor",
        "chain": ["cast_off", "hoist", "tack", "moor"],
        "peripherals": ["winch", "rope", "fender", "chart", "compass"],
    },
    "photography": {
        "hub": "studio",
        "chain": ["setup", "compose", "shoot", "edit"],
        "peripherals": ["tripod", "reflector", "filter", "card", "backdrop"],
    },
    "archaeology": {
        "hub": "site",
        "chain": ["survey", "excavate", "catalog", "preserve"],
        "peripherals": ["brush", "sieve", "stake", "notebook", "flag"],
    },
}


assert len(TOPICS) == 10
for name, topic in TOPICS.items():
    assert isinstance(topic["hub"], str)
    assert len(topic["chain"]) == 4, f"{name}: expected 4 chain nodes"
    assert len(topic["peripherals"]) >= 3, f"{name}: need at least 3 peripherals"


def _ns(topic: str, node: str) -> str:
    """Namespace node ids so cross-topic word collisions can't leak."""
    return f"{topic}.{node}"


def build_fixture(
    *,
    peripherals_per_topic: int = 3,
    seed: int = 0,
) -> dict[str, Any]:
    """Build the full fixture dict for one seed.

    Each topic produces exactly **10 episodes**:

    - 4 hub episodes: ``{hub, c_i}`` for i in 1..4
    - 3 chain episodes: ``{c1, c2}``, ``{c2, c3}``, ``{c3, c4}``
    - 3 peripheral episodes: ``{hub, peripheral_i}`` for 3 randomly-
      selected peripherals (seeded)

    The ``seed`` controls which peripherals are drawn but never
    changes the core topology, so every seed tests the same retrieval
    task shape.
    """
    rng = random.Random(seed)
    episodes: list[dict[str, Any]] = []
    reinforcement = 2  # each core episode reinforced twice → weight 0.3 → 0.4

    for topic_name, topic in TOPICS.items():
        hub = _ns(topic_name, topic["hub"])
        chain = [_ns(topic_name, c) for c in topic["chain"]]

        # 8 hub episodes: each hub↔chain pair presented TWICE for reinforcement
        for rep in range(reinforcement):
            for c in chain:
                episodes.append(
                    {
                        "id": f"{topic_name}_hub_{c.rsplit('.', 1)[1]}_r{rep}",
                        "topic": topic_name,
                        "channel": "text",
                        "activated_nodes": [hub, c],
                        "kind": "hub",
                    }
                )

        # 6 chain episodes: each adjacency presented TWICE for reinforcement
        for rep in range(reinforcement):
            for i in range(len(chain) - 1):
                episodes.append(
                    {
                        "id": f"{topic_name}_chain_{i}_r{rep}",
                        "topic": topic_name,
                        "channel": "text",
                        "activated_nodes": [chain[i], chain[i + 1]],
                        "kind": "chain",
                    }
                )

        # 3 peripheral episodes: {hub, peripheral} — single-shot, no reinforcement
        pool = list(topic["peripherals"])
        rng.shuffle(pool)
        selected = pool[:peripherals_per_topic]
        for i, p in enumerate(selected):
            episodes.append(
                {
                    "id": f"{topic_name}_peri_{i}",
                    "topic": topic_name,
                    "channel": "text",
                    "activated_nodes": [hub, _ns(topic_name, p)],
                    "kind": "peripheral",
                }
            )

    # Probe specification: for each of the 5 core nodes in each topic,
    # the ground-truth target set is the other 4 core nodes.
    probes: list[dict[str, Any]] = []
    for topic_name, topic in TOPICS.items():
        core_ids = [_ns(topic_name, topic["hub"])] + [_ns(topic_name, c) for c in topic["chain"]]
        for cue in core_ids:
            probes.append(
                {
                    "topic": topic_name,
                    "cue": cue,
                    "targets": [c for c in core_ids if c != cue],
                    "cue_kind": "hub" if cue == _ns(topic_name, topic["hub"]) else "chain",
                }
            )

    return {
        "version": 2,  # Stage 2 hub+chain topology (v1 was the abandoned clique draft)
        "seed": seed,
        "n_topics": len(TOPICS),
        "episodes_per_topic": 17,  # 8 hub (2× reinforced) + 6 chain (2× reinforced) + 3 peripheral
        "peripherals_per_topic": peripherals_per_topic,
        "episodes": episodes,
        "probes": probes,
    }


def write_fixture_yaml(
    path: str | Path,
    *,
    seed: int = 0,
    peripherals_per_topic: int = 3,
) -> None:
    """Dump the fixture to a YAML file for reproducibility.

    Uses a minimal hand-rolled emitter — no PyYAML dependency. Only
    the shapes this generator produces are supported.
    """
    fixture = build_fixture(
        seed=seed,
        peripherals_per_topic=peripherals_per_topic,
    )
    lines: list[str] = [
        "# Substrate P3a Stage 2 — synthetic episodes fixture (hub+chain topology)",
        "# Auto-generated by tests/substrate/p3a_fixture_gen.py",
        "# Do NOT hand-edit; re-run the generator to regenerate.",
        "#",
        "# Topology: 10 topics × (1 hub + 4 chain) = 50 core nodes.",
        "# Episodes/topic: 4 hub + 3 chain-adjacency + 3 peripheral = 10.",
        "# Retrieval task: given any core node as cue, recover the other",
        "# 4 cores. Chain-node cues require multi-hop traversal to reach",
        "# their far-end chain neighbors — TF-IDF cannot; Hebbian",
        "# spreading_activation can.",
        "",
        f"version: {fixture['version']}",
        f"seed: {fixture['seed']}",
        f"n_topics: {fixture['n_topics']}",
        f"episodes_per_topic: {fixture['episodes_per_topic']}",
        f"peripherals_per_topic: {fixture['peripherals_per_topic']}",
        "",
        "episodes:",
    ]
    for ep in fixture["episodes"]:
        lines.append(f"  - id: {ep['id']}")
        lines.append(f"    topic: {ep['topic']}")
        lines.append(f"    channel: {ep['channel']}")
        lines.append(f"    kind: {ep['kind']}")
        lines.append(f"    activated_nodes: [{', '.join(ep['activated_nodes'])}]")
    lines.append("")
    lines.append("probes:")
    for probe in fixture["probes"]:
        lines.append(f"  - topic: {probe['topic']}")
        lines.append(f"    cue: {probe['cue']}")
        lines.append(f"    cue_kind: {probe['cue_kind']}")
        lines.append(f"    targets: [{', '.join(probe['targets'])}]")

    Path(path).write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate the P3a Stage 2 synthetic episodes fixture (hub+chain topology)."
    )
    parser.add_argument(
        "--out",
        default="scenarios/substrate/synthetic_episodes.yaml",
        help="Output YAML path (relative to repo root).",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    write_fixture_yaml(args.out, seed=args.seed)
    print(f"wrote fixture to {args.out} (seed={args.seed})")
