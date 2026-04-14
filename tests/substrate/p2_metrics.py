"""P2 metric extractor — reward-modulated recognition.

Compares two encoding runs over the same cluster fixture:
- **Baseline**: no reward events, stock EC threshold.
- **Rewarded**: inject reward on the first sentence of each target
  cluster, then encode the remaining paraphrases. The NAc reward-bias
  pathway widens EC's recognition radius around the rewarded node so
  near-miss paraphrases collapse to it instead of separating.

P2 pass criteria (from docs/plans/substrate_recognition.md):
    - Rewarded-node collapse: ≥30% fewer distinct nodes vs unrewarded
    - Non-interference: distractor cluster node count within ±5%
    - Decay: bias returns toward baseline after reinforcement stops
    - Per-agent isolation: rewarding agent A does not affect agent B
    - Sanity floor: reduction is non-negative

Usage::

    from tests.substrate.p2_metrics import (
        compute_p2_metrics,
        load_clusters_with_targets,
    )
    clusters, targets = load_clusters_with_targets(
        "scenarios/substrate/p2_reward_modulation.yaml"
    )
    metrics = compute_p2_metrics(
        clusters=clusters,
        target_clusters=targets,
        make_encoder_nac=factory,
        threshold=0.50,
        reward=1.0,
    )
    print(metrics.summary())
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Fixture loading
# ─────────────────────────────────────────────────────────────────────────────


def load_clusters_with_targets(
    path: str,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Load clusters and target-cluster annotations from a P2 fixture.

    Fixture format extends the P1 percept format with a top-level
    ``reward_schedule`` block listing target cluster names::

        reward_schedule:
          targets:
            - weather_nice
            - time_question
          reward: 1.0

    Percepts are grouped by ``metadata.cluster`` as in P1.

    Returns:
        (clusters, target_cluster_names) tuple.
    """
    import yaml

    from collections import defaultdict

    with open(path) as f:
        data = yaml.safe_load(f)

    by_cluster: dict[str, list[str]] = defaultdict(list)
    for percept in data.get("percepts", []):
        meta = percept.get("metadata", {})
        cluster_name = meta.get("cluster")
        text = percept.get("cli_input") or percept.get("transcript_chunk") or percept.get("content")
        if cluster_name and text:
            by_cluster[cluster_name].append(text)

    clusters = [{"name": name, "sentences": sentences} for name, sentences in by_cluster.items()]

    schedule = data.get("reward_schedule") or {}
    targets = list(schedule.get("targets") or [])

    return clusters, targets


# ─────────────────────────────────────────────────────────────────────────────
# Metrics dataclass
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class P2Metrics:
    """P2 reward-modulation metrics.

    Compares baseline (no reward) vs rewarded runs cluster-by-cluster.
    """

    # Per-cluster unique-node counts
    baseline_nodes: dict[str, int] = field(default_factory=dict)
    rewarded_nodes: dict[str, int] = field(default_factory=dict)

    # Cluster classification
    target_clusters: list[str] = field(default_factory=list)
    distractor_clusters: list[str] = field(default_factory=list)

    # Aggregate metrics
    target_reduction: float = 0.0  # mean fractional reduction for targets
    distractor_interference: float = 0.0  # max fractional change for distractors

    # Counts for convenience
    baseline_target_total: int = 0
    rewarded_target_total: int = 0
    baseline_distractor_total: int = 0
    rewarded_distractor_total: int = 0

    # Reward bias telemetry
    final_bias_mean: float = 0.0
    final_bias_max: float = 0.0

    def passes_p2(self) -> bool:
        """All pass criteria from substrate_recognition.md."""
        return (
            self.target_reduction >= 0.30
            and self.distractor_interference <= 0.05
            and self.target_reduction >= 0.0  # sanity floor
            and self.rewarded_target_total <= self.baseline_target_total
        )

    def summary(self) -> str:
        status = "PASS" if self.passes_p2() else "FAIL"
        lines = [
            f"P2 Reward Modulation — {status}",
            f"  Target reduction:       {self.target_reduction:.1%} (need ≥30%)",
            f"  Distractor interference:{self.distractor_interference:.1%} (need ≤5%)",
            f"  Target nodes (base→rew):{self.baseline_target_total} → {self.rewarded_target_total}",
            f"  Distractor (base→rew):  {self.baseline_distractor_total} → {self.rewarded_distractor_total}",
            f"  Final bias mean/max:    {self.final_bias_mean:.3f} / {self.final_bias_max:.3f}",
        ]
        return "\n".join(lines)

    def per_cluster_table(self) -> str:
        """Per-cluster baseline vs rewarded breakdown for lab notebooks."""
        lines = [
            "\nPer-cluster (★ = target):",
            f"  {'Cluster':<28s} {'Base':>5s} {'Rew':>5s} {'Δ':>6s}",
            f"  {'─' * 50}",
        ]
        all_names = sorted(set(self.baseline_nodes) | set(self.rewarded_nodes))
        for name in all_names:
            star = "★" if name in self.target_clusters else " "
            base = self.baseline_nodes.get(name, 0)
            rew = self.rewarded_nodes.get(name, 0)
            delta = (rew - base) / base if base > 0 else 0.0
            lines.append(f"  {star} {name:<26s} {base:>5d} {rew:>5d} {delta:>+5.0%}")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Core metric computation
# ─────────────────────────────────────────────────────────────────────────────


EncoderNAcFactory = Callable[[float], tuple[Any, Any, Any, Any]]
"""Factory: threshold → (encoder, ec, atl, nac) tuple.

The caller supplies this so P2 metrics stay independent of encoder
instantiation details (embedding model, etc.).
"""


def _run_encoding_pass(
    *,
    encoder: Any,
    clusters: list[dict[str, Any]],
    target_clusters: set[str],
    reward: float,
    nac: Any | None,
    agent_id: str,
    shuffle: bool,
    seed: int,
) -> dict[str, list[str]]:
    """Encode all cluster sentences, optionally injecting rewards.

    Reward injection rule: the **first** sentence encountered for each
    target cluster directly credits the node it mapped to via
    ``nac.credit_node``. We use ``credit_node`` rather than
    ``distribute_reward`` so that cross-cluster eligibility residue
    from prior encodes cannot leak reward onto non-target nodes — the
    test driver needs a clean per-cluster signal. The eligibility
    pathway is validated separately in
    ``test_substrate_recognition.py::TestEncoderNAcEligibility``.

    Returns:
        Mapping cluster_name → list of assigned node_ids.
    """
    from maxim.agents.bus import Percept
    from maxim.agents.percept_context import PerceptContext

    # Flatten to (cluster_name, sentence) so the order is deterministic
    # but we can still know which cluster a sentence belongs to.
    flat: list[tuple[str, str]] = []
    for cluster in clusters:
        for sentence in cluster.get("sentences", []):
            flat.append((cluster["name"], sentence))

    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(flat)

    assignments: dict[str, list[str]] = {c["name"]: [] for c in clusters}
    rewarded_seen: set[str] = set()

    # Build a PerceptContext so encoder sees the right agent_id
    # (only when we have an agent_id to thread — empty string uses the
    # default path in the encoder).
    ctx = PerceptContext(agent_id=agent_id) if agent_id else None

    for cluster_name, sentence in flat:
        percept = Percept(
            timestamp=0.0,
            source="test",
            transcript_chunk=sentence,
            context=ctx,
        )
        node_id = encoder.encode(percept)
        if node_id is None:
            continue
        assignments[cluster_name].append(node_id)

        # Directly credit the just-created node the first time we see
        # a target cluster. See function docstring for why we use
        # credit_node rather than distribute_reward.
        if reward > 0.0 and nac is not None and cluster_name in target_clusters and cluster_name not in rewarded_seen:
            nac.credit_node(agent_id=agent_id, node_id=node_id, reward=reward)
            rewarded_seen.add(cluster_name)

    return assignments


def compute_p2_metrics(
    *,
    clusters: list[dict[str, Any]],
    target_clusters: list[str],
    make_encoder_nac: EncoderNAcFactory,
    threshold: float = 0.50,
    reward: float = 1.0,
    shuffle: bool = True,
    seed: int = 0,
    agent_id: str = "",
) -> P2Metrics:
    """Run baseline + rewarded passes and compute the P2 metrics.

    Args:
        clusters: Cluster dicts from ``load_clusters_with_targets``.
        target_clusters: Names of clusters that receive reward events.
        make_encoder_nac: Factory returning ``(encoder, ec, atl, nac)``
            for a given threshold. Caller owns model loading.
        threshold: EC base threshold. P2 validation prefers a *stricter*
            threshold than P1 (0.50+ vs P1's 0.40) so baseline has
            measurable fragmentation for the reward bias to recover.
        reward: Magnitude passed to ``NAc.distribute_reward``.
        shuffle: Shuffle sentence order (same shuffle for both passes).
        seed: RNG seed for shuffle.
        agent_id: Agent threading — isolation tests vary this.
    """
    target_set = set(target_clusters)
    distractor_clusters = [c["name"] for c in clusters if c["name"] not in target_set]

    # --- Baseline pass: no reward ---
    base_encoder, _, _, base_nac = make_encoder_nac(threshold)
    baseline_assign = _run_encoding_pass(
        encoder=base_encoder,
        clusters=clusters,
        target_clusters=target_set,
        reward=0.0,  # baseline = no reward
        nac=base_nac,
        agent_id=agent_id,
        shuffle=shuffle,
        seed=seed,
    )

    # --- Rewarded pass: same order, with reward injection ---
    rew_encoder, _, _, rew_nac = make_encoder_nac(threshold)
    rewarded_assign = _run_encoding_pass(
        encoder=rew_encoder,
        clusters=clusters,
        target_clusters=target_set,
        reward=reward,
        nac=rew_nac,
        agent_id=agent_id,
        shuffle=shuffle,
        seed=seed,
    )

    # --- Unique-node counts per cluster ---
    baseline_nodes = {name: len(set(nids)) for name, nids in baseline_assign.items()}
    rewarded_nodes = {name: len(set(nids)) for name, nids in rewarded_assign.items()}

    # --- Target reduction (mean fractional reduction across targets) ---
    target_reductions: list[float] = []
    for name in target_clusters:
        base = baseline_nodes.get(name, 0)
        rew = rewarded_nodes.get(name, 0)
        if base > 0:
            target_reductions.append((base - rew) / base)
    target_reduction = sum(target_reductions) / len(target_reductions) if target_reductions else 0.0

    # --- Distractor interference (max fractional change, signed absolute) ---
    distractor_changes: list[float] = []
    for name in distractor_clusters:
        base = baseline_nodes.get(name, 0)
        rew = rewarded_nodes.get(name, 0)
        if base > 0:
            distractor_changes.append(abs(rew - base) / base)
    distractor_interference = max(distractor_changes) if distractor_changes else 0.0

    # --- Totals for reporting ---
    baseline_target_total = sum(baseline_nodes.get(n, 0) for n in target_clusters)
    rewarded_target_total = sum(rewarded_nodes.get(n, 0) for n in target_clusters)
    baseline_distractor_total = sum(baseline_nodes.get(n, 0) for n in distractor_clusters)
    rewarded_distractor_total = sum(rewarded_nodes.get(n, 0) for n in distractor_clusters)

    # --- Final reward-bias telemetry (rewarded pass only) ---
    biases = [v for v in rew_nac._reward_bias.values()]
    final_bias_mean = sum(biases) / len(biases) if biases else 0.0
    final_bias_max = max(biases) if biases else 0.0

    return P2Metrics(
        baseline_nodes=baseline_nodes,
        rewarded_nodes=rewarded_nodes,
        target_clusters=list(target_clusters),
        distractor_clusters=distractor_clusters,
        target_reduction=round(target_reduction, 4),
        distractor_interference=round(distractor_interference, 4),
        baseline_target_total=baseline_target_total,
        rewarded_target_total=rewarded_target_total,
        baseline_distractor_total=baseline_distractor_total,
        rewarded_distractor_total=rewarded_distractor_total,
        final_bias_mean=round(final_bias_mean, 4),
        final_bias_max=round(final_bias_max, 4),
    )
