"""P2 metric extractor — reward-modulated recognition.

Compares two encoding runs over the same cluster fixture:
- **Baseline**: no reward events, stock EC threshold.
- **Rewarded**: inject reward on the first sentence of each target
  cluster, then encode the remaining paraphrases. The NAc reward-bias
  pathway widens EC's recognition radius around the rewarded node so
  near-miss paraphrases collapse to it instead of separating.

**Primary metric — per-cluster self-collapse rate with plurality ownership.**
For each cluster, compute the fraction of within-cluster paraphrase pairs
that map to the same ATL node AND that node is plurality-owned by the
cluster itself (the cluster contributed the largest share of the node's
members). Ties in plurality ownership are treated as unowned.

The plurality-ownership refinement is load-bearing: without it, the
metric spuriously rewards cross-cluster contamination. At high EC
threshold + large reward magnitude, reward bias on target nodes can
widen their recognition radius far enough to STEAL distractor
sentences — all 5 sentences of a distractor cluster land on the same
target node. Raw pair-collapse sees "5 pairs collapsed" and calls it
a win, but the distractor cluster's identity is destroyed. With
plurality ownership, the stolen distractor node is owned by the target
cluster, so the distractor's self-collapse rate goes to 0, which is
the correct signal — reward bias is over-tuned, not working better.

Properties:
- bounded in [0, 1]
- independent of the total number of distinct nodes across the run
  (phantom nodes from centroid drift don't count)
- symmetric with P1's collapse rate when there's no cross-cluster
  contamination; diverges (correctly) when there is
- directional: drift downward indicates contamination, drift upward
  on unrewarded clusters indicates centroid-drift coupling

Pre-Stage-3 this module used node-count-based metrics (rewarded_target_total
vs baseline_target_total). Node-count metrics are coupled across clusters
via the stateful centroid-update path: a target sentence completing to a
reward-biased node (in the rewarded pass) but separating (in baseline)
changes the global node set, which changes how later distractor sentences
get matched. Real-embedding validation on paraphrase-mpnet-base-v2
(Stage 3, 2026-04-14) showed distractor interference spiking to 100%
with ±33 pp std on the v1 node-count metric — all measurement artifact,
no mechanism defect. The Stage 3 draft then tried raw pair-collapse and
found it spuriously rewarded cross-cluster contamination (see above).
Plurality-ownership self-collapse is the metric that survived the third
pass.

P2 pass criteria (collapse-rate semantics, Stage 3 restatement):
  - **Target gain** ≥ +30 pp (mean over target clusters of
    rewarded_rate - baseline_rate)
  - **Distractor drift** ≤ 5 pp (mean over distractor clusters of
    |rewarded_rate - baseline_rate|)
  - Sanity: per-cluster rewarded_rate >= baseline_rate for at least
    half the target clusters (directional check — reward bias only
    widens recognition radius, never tightens)
  - Reported alongside mean ± std over N seeds (default 10)

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
        threshold=0.55,
        reward=2.0,
    )
    print(metrics.summary())
"""

from __future__ import annotations

import logging
import random
from collections import defaultdict
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
    """P2 reward-modulation metrics, collapse-rate semantics.

    All ``*_rate`` fields are in [0, 1] and represent the fraction of
    within-cluster paraphrase pairs that share an ATL node.
    """

    # Per-cluster collapse rates
    baseline_rates: dict[str, float] = field(default_factory=dict)
    rewarded_rates: dict[str, float] = field(default_factory=dict)

    # Cluster classification
    target_clusters: list[str] = field(default_factory=list)
    distractor_clusters: list[str] = field(default_factory=list)

    # Aggregate metrics (all in pp units — absolute percentage points)
    target_gain: float = 0.0  # mean rewarded - baseline over targets
    distractor_drift: float = 0.0  # mean |rewarded - baseline| over distractors

    # Mean rates for readability
    baseline_target_mean: float = 0.0
    rewarded_target_mean: float = 0.0
    baseline_distractor_mean: float = 0.0
    rewarded_distractor_mean: float = 0.0

    # Directional sanity: fraction of target clusters where reward did
    # not DECREASE the collapse rate (reward bias only widens recognition
    # radius, so monotone non-decreasing is the physical expectation).
    target_monotone_fraction: float = 0.0

    # Node count telemetry — retained for continuity with the
    # pre-Stage-3 node-count metric so lab notebooks can still
    # cross-reference earlier numbers. Not part of the pass criteria.
    baseline_nodes: dict[str, int] = field(default_factory=dict)
    rewarded_nodes: dict[str, int] = field(default_factory=dict)

    # Reward bias telemetry (rewarded pass only)
    final_bias_mean: float = 0.0
    final_bias_max: float = 0.0

    #: Target-gain pass threshold (pp)
    TARGET_GAIN_PP: float = 0.30
    #: Distractor-drift pass ceiling (pp)
    DISTRACTOR_DRIFT_PP: float = 0.05
    #: Fraction of target clusters whose rewarded rate must be >= baseline
    MONOTONE_FRACTION: float = 0.5

    def passes_p2(self) -> bool:
        return (
            self.target_gain >= self.TARGET_GAIN_PP
            and self.distractor_drift <= self.DISTRACTOR_DRIFT_PP
            and self.target_monotone_fraction >= self.MONOTONE_FRACTION
        )

    def summary(self) -> str:
        status = "PASS" if self.passes_p2() else "FAIL"
        lines = [
            f"P2 Reward Modulation — {status}",
            f"  Target gain:            {self.target_gain:+.1%} pp (need >=+{self.TARGET_GAIN_PP:.0%})",
            f"  Distractor drift:       {self.distractor_drift:.1%} pp (need <={self.DISTRACTOR_DRIFT_PP:.0%})",
            f"  Target rate base->rew:  {self.baseline_target_mean:.1%} -> {self.rewarded_target_mean:.1%}",
            f"  Distract rate base->rew:{self.baseline_distractor_mean:.1%} -> {self.rewarded_distractor_mean:.1%}",
            f"  Target monotone:        {self.target_monotone_fraction:.0%} of clusters",
            f"  Final bias mean/max:    {self.final_bias_mean:.3f} / {self.final_bias_max:.3f}",
        ]
        return "\n".join(lines)

    def per_cluster_table(self) -> str:
        lines = [
            "\nPer-cluster collapse rates (* = target):",
            f"  {'Cluster':<28s} {'Base':>7s} {'Rew':>7s} {'dpp':>7s}",
            f"  {'-' * 56}",
        ]
        all_names = sorted(set(self.baseline_rates) | set(self.rewarded_rates))
        for name in all_names:
            star = "*" if name in self.target_clusters else " "
            base = self.baseline_rates.get(name, 0.0)
            rew = self.rewarded_rates.get(name, 0.0)
            delta = rew - base
            lines.append(f"  {star} {name:<26s} {base:>6.1%} {rew:>6.1%} {delta:>+6.1%}")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Core metric computation
# ─────────────────────────────────────────────────────────────────────────────


EncoderNAcFactory = Callable[[float], tuple[Any, Any, Any, Any]]
"""Factory: threshold → (encoder, ec, atl, nac) tuple."""


def _cluster_self_collapse_rates(
    assignments: dict[str, list[str]],
) -> dict[str, float]:
    """Per-cluster self-collapse rate — plurality-ownership variant.

    For each node present in ``assignments``, compute the plurality-
    owning cluster (the cluster that contributed the largest share of
    the node's members). A within-cluster pair "self-collapses" only
    if both sentences map to the same node AND that node is plurality-
    owned by the cluster itself. If a distractor cluster's sentences
    get stolen onto a widened target node (reward-bias cross-cluster
    contamination), they share a node but the node is owned by the
    target cluster — the distractor's self-collapse rate goes DOWN,
    not up, which is the correct signal.

    Ties in plurality ownership are treated as "unowned" (no cluster
    claims the node), so ambiguous mergers don't spuriously credit
    either side. This makes the metric directional and honest about
    contamination — exactly what the pre-Stage-3 raw-pair-collapse
    metric failed to distinguish.
    """
    from collections import defaultdict

    # node_id -> cluster_name -> member count
    node_members: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for cluster_name, nodes in assignments.items():
        for n in nodes:
            node_members[n][cluster_name] += 1

    node_owner: dict[str, str | None] = {}
    for node_id, counts in node_members.items():
        max_count = max(counts.values())
        owners = [c for c, v in counts.items() if v == max_count]
        node_owner[node_id] = owners[0] if len(owners) == 1 else None

    rates: dict[str, float] = {}
    for cluster_name, nodes in assignments.items():
        n = len(nodes)
        if n < 2:
            rates[cluster_name] = 0.0
            continue
        pairs = n * (n - 1) // 2
        collapsed = 0
        for i in range(n):
            for j in range(i + 1, n):
                if nodes[i] == nodes[j] and node_owner.get(nodes[i]) == cluster_name:
                    collapsed += 1
        rates[cluster_name] = collapsed / pairs
    return rates


def _collapse_rate(node_ids: list[str]) -> float:
    """DEPRECATED: raw within-sequence pair collapse rate.

    Kept for reference and debugging. Use
    :func:`_cluster_self_collapse_rates` for the metric that honestly
    handles cross-cluster contamination. Raw pair-collapse counts any
    shared-node pair regardless of whether the node is "the cluster's
    own" — so if reward bias is strong enough to steal distractor
    sentences onto widened target nodes, raw pair-collapse on the
    distractor side spuriously shows "collapse increased" even though
    the distractor cluster's identity was destroyed.
    """
    n = len(node_ids)
    if n < 2:
        return 0.0
    pairs = n * (n - 1) // 2
    collapsed = 0
    for i in range(n):
        for j in range(i + 1, n):
            if node_ids[i] == node_ids[j]:
                collapsed += 1
    return collapsed / pairs


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
        Mapping cluster_name → list of assigned node_ids (one per
        sentence, in encode order).
    """
    from maxim.agents.bus import Percept
    from maxim.agents.percept_context import PerceptContext

    flat: list[tuple[str, str]] = []
    for cluster in clusters:
        for sentence in cluster.get("sentences", []):
            flat.append((cluster["name"], sentence))

    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(flat)

    assignments: dict[str, list[str]] = {c["name"]: [] for c in clusters}
    rewarded_seen: set[str] = set()
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

        if reward > 0.0 and nac is not None and cluster_name in target_clusters and cluster_name not in rewarded_seen:
            nac.credit_node(agent_id=agent_id, node_id=node_id, reward=reward)
            rewarded_seen.add(cluster_name)

    return assignments


def compute_p2_metrics(
    *,
    clusters: list[dict[str, Any]],
    target_clusters: list[str],
    make_encoder_nac: EncoderNAcFactory,
    threshold: float = 0.55,
    reward: float = 2.0,
    shuffle: bool = True,
    seed: int = 0,
    agent_id: str = "",
) -> P2Metrics:
    """Run baseline + rewarded passes and compute collapse-rate metrics.

    Args:
        clusters: Cluster dicts from ``load_clusters_with_targets``.
        target_clusters: Names of clusters that receive reward events.
        make_encoder_nac: Factory returning ``(encoder, ec, atl, nac)``
            for a given threshold.
        threshold: EC base threshold.
        reward: Magnitude passed to ``NAc.credit_node``.
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
        reward=0.0,
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

    # --- Per-cluster self-collapse rates (plurality-ownership) ---
    baseline_rates = _cluster_self_collapse_rates(baseline_assign)
    rewarded_rates = _cluster_self_collapse_rates(rewarded_assign)

    # --- Node-count telemetry (continuity with pre-Stage-3 metrics) ---
    baseline_nodes = {name: len(set(nids)) for name, nids in baseline_assign.items()}
    rewarded_nodes = {name: len(set(nids)) for name, nids in rewarded_assign.items()}

    # --- Aggregate: target gain ---
    target_gains: list[float] = []
    for name in target_clusters:
        if name in baseline_rates:
            target_gains.append(rewarded_rates[name] - baseline_rates[name])
    target_gain = sum(target_gains) / len(target_gains) if target_gains else 0.0

    target_monotone_fraction = sum(1 for g in target_gains if g >= 0) / len(target_gains) if target_gains else 0.0

    # --- Aggregate: distractor drift (absolute) ---
    distractor_drifts: list[float] = []
    for name in distractor_clusters:
        if name in baseline_rates:
            distractor_drifts.append(abs(rewarded_rates[name] - baseline_rates[name]))
    distractor_drift = sum(distractor_drifts) / len(distractor_drifts) if distractor_drifts else 0.0

    # --- Mean rates ---
    def _mean(rates: dict[str, float], names: list[str]) -> float:
        vals = [rates[n] for n in names if n in rates]
        return sum(vals) / len(vals) if vals else 0.0

    baseline_target_mean = _mean(baseline_rates, list(target_clusters))
    rewarded_target_mean = _mean(rewarded_rates, list(target_clusters))
    baseline_distractor_mean = _mean(baseline_rates, distractor_clusters)
    rewarded_distractor_mean = _mean(rewarded_rates, distractor_clusters)

    # --- Reward bias telemetry ---
    biases = list(rew_nac._reward_bias.values())
    final_bias_mean = sum(biases) / len(biases) if biases else 0.0
    final_bias_max = max(biases) if biases else 0.0

    return P2Metrics(
        baseline_rates=baseline_rates,
        rewarded_rates=rewarded_rates,
        target_clusters=list(target_clusters),
        distractor_clusters=distractor_clusters,
        target_gain=round(target_gain, 4),
        distractor_drift=round(distractor_drift, 4),
        baseline_target_mean=round(baseline_target_mean, 4),
        rewarded_target_mean=round(rewarded_target_mean, 4),
        baseline_distractor_mean=round(baseline_distractor_mean, 4),
        rewarded_distractor_mean=round(rewarded_distractor_mean, 4),
        target_monotone_fraction=round(target_monotone_fraction, 4),
        baseline_nodes=baseline_nodes,
        rewarded_nodes=rewarded_nodes,
        final_bias_mean=round(final_bias_mean, 4),
        final_bias_max=round(final_bias_max, 4),
    )
