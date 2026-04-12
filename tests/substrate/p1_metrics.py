"""P1 metric extractor — paraphrase collapse, cluster distinctness, node stability.

Reads the paraphrase_clusters.yaml fixture and the EC/ATL substrate
state after a fixture run to compute the P1 pass criteria metrics.

Usage:
    from tests.substrate.p1_metrics import P1Metrics, compute_p1_metrics, load_clusters_from_fixture

    clusters = load_clusters_from_fixture("scenarios/substrate/paraphrase_clusters.yaml")
    metrics = compute_p1_metrics(ec=ec, atl=atl, clusters=clusters, encoder=encoder)
    print(metrics.summary())
    print(metrics.diagnostics())
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class SentenceDiag:
    """Per-sentence diagnostic record."""

    cluster: str
    sentence: str
    node_id: str
    is_new: bool  # True = EC separated (new node), False = pattern completed
    best_sim: float  # Cosine sim to the closest existing node
    best_match_node: str  # Which node was the closest match
    best_match_cluster: str  # Which cluster owns that best-match node (if known)


@dataclass
class P1Metrics:
    """P1 pass criteria metrics."""

    # Paraphrase collapse: fraction of within-cluster sentence pairs
    # that mapped to the same ATL node.
    collapse_rate: float = 0.0

    # Cluster distinctness: fraction of cross-cluster sentence pairs
    # that incorrectly collapsed to the same node.
    cross_cluster_rate: float = 0.0

    # Node count stability: growth rate of unique nodes over the final
    # 20% of sentences processed.
    node_growth_final_20pct: float = 0.0

    # Total unique ATL nodes created
    total_nodes: int = 0

    # Per-cluster details
    cluster_details: list[dict[str, Any]] = field(default_factory=list)

    # Modality isolation: whether any text cluster collapsed into
    # a non-text node (should be 0)
    modality_violations: int = 0

    # Per-sentence diagnostics (populated when diagnostics=True)
    sentence_diags: list[SentenceDiag] = field(default_factory=list)

    # Similarity score distribution for within-cluster misses
    missed_sims: list[float] = field(default_factory=list)

    def passes_p1(self) -> bool:
        """Check if all P1 pass criteria are met."""
        return (
            self.collapse_rate >= 0.90
            and self.cross_cluster_rate <= 0.05
            and self.node_growth_final_20pct < 0.10
            and self.modality_violations == 0
        )

    def summary(self) -> str:
        """Human-readable summary for lab notebook."""
        status = "PASS" if self.passes_p1() else "FAIL"
        lines = [
            f"P1 Recognition — {status}",
            f"  Collapse rate:      {self.collapse_rate:.1%} (need ≥90%)",
            f"  Cross-cluster:      {self.cross_cluster_rate:.1%} (need ≤5%)",
            f"  Node growth final:  {self.node_growth_final_20pct:.1%} (need <10%)",
            f"  Total nodes:        {self.total_nodes}",
            f"  Modality violations:{self.modality_violations} (need 0)",
        ]
        return "\n".join(lines)

    def diagnostics(self) -> str:
        """Detailed diagnostic report for threshold tuning.

        Shows:
        - Similarity score histogram for within-cluster misses
        - Per-cluster collapse breakdown
        - Worst-performing clusters
        - Threshold recommendation
        """
        if not self.sentence_diags:
            return "(no diagnostics collected — re-run with diagnostics=True)"

        lines = ["\n" + "=" * 70, "P1 DIAGNOSTICS", "=" * 70]

        # --- Similarity distribution for missed within-cluster pairs ---
        missed = self.missed_sims
        if missed:
            missed_sorted = sorted(missed)
            lines.append(f"\nWithin-cluster MISSES (separated when should have collapsed): {len(missed)}")
            lines.append(f"  Min sim:    {missed_sorted[0]:.4f}")
            lines.append(f"  Max sim:    {missed_sorted[-1]:.4f}")
            lines.append(f"  Median sim: {missed_sorted[len(missed_sorted) // 2]:.4f}")
            lines.append(f"  Mean sim:   {sum(missed) / len(missed):.4f}")

            # Histogram buckets
            buckets = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]
            lines.append("\n  Similarity histogram (missed pairs):")
            for i in range(len(buckets) - 1):
                lo, hi = buckets[i], buckets[i + 1]
                count = sum(1 for s in missed if lo <= s < hi)
                bar = "#" * count
                lines.append(f"    [{lo:.1f}-{hi:.1f}): {count:3d} {bar}")

            # Threshold recommendation
            if missed_sorted:
                p90 = missed_sorted[int(len(missed_sorted) * 0.10)]  # 10th percentile
                lines.append(f"\n  Recommended threshold (captures 90% of misses): {p90:.3f}")
        else:
            lines.append("\nNo within-cluster misses — all paraphrases collapsed!")

        # --- Per-cluster breakdown ---
        lines.append(f"\n{'─' * 70}")
        lines.append("Per-cluster collapse:")
        lines.append(f"  {'Cluster':<30s} {'Sents':>5s} {'Nodes':>5s} {'Rate':>6s}")
        lines.append(f"  {'─' * 50}")

        failed_clusters = []
        for cd in sorted(self.cluster_details, key=lambda x: x["collapse_rate"]):
            rate = cd["collapse_rate"]
            marker = " ✗" if rate < 1.0 else ""
            lines.append(f"  {cd['name']:<30s} {cd['sentences']:>5d} {cd['unique_nodes']:>5d} {rate:>5.0%}{marker}")
            if rate < 1.0:
                failed_clusters.append(cd["name"])

        if failed_clusters:
            lines.append(f"\n  Failed clusters ({len(failed_clusters)}/{len(self.cluster_details)}):")
            for name in failed_clusters[:10]:
                # Show the actual sentences and their nodes for this cluster
                cluster_diags = [d for d in self.sentence_diags if d.cluster == name]
                lines.append(f"    {name}:")
                for d in cluster_diags:
                    status = "COMPLETE" if not d.is_new else f"NEW (best_sim={d.best_sim:.3f})"
                    lines.append(f'      "{d.sentence[:60]}" → {d.node_id[:8]} [{status}]')

        lines.append("=" * 70)
        return "\n".join(lines)


def load_clusters_from_fixture(path: str) -> list[dict[str, Any]]:
    """Load paraphrase clusters from the percept-based YAML fixture.

    The fixture format uses percepts with metadata.cluster as ground truth:
        percepts:
          - at: 0
            source: cli
            cli_input: "Hello, how are you?"
            metadata: { cluster: greeting, split: train, difficulty: easy }

    Returns list of cluster dicts: [{"name": "greeting", "sentences": [...]}]
    """
    import yaml

    with open(path) as f:
        data = yaml.safe_load(f)

    # Group percepts by cluster name
    by_cluster: dict[str, list[str]] = defaultdict(list)
    for percept in data.get("percepts", []):
        meta = percept.get("metadata", {})
        cluster_name = meta.get("cluster")
        text = percept.get("cli_input") or percept.get("transcript_chunk") or percept.get("content")
        if cluster_name and text:
            by_cluster[cluster_name].append(text)

    return [{"name": name, "sentences": sentences} for name, sentences in by_cluster.items()]


def compute_p1_metrics(
    *,
    ec: Any,
    atl: Any,
    clusters: list[dict[str, Any]],
    encoder: Any,
    diagnostics: bool = True,
) -> P1Metrics:
    """Compute P1 metrics by encoding all cluster sentences and checking collapse.

    Args:
        ec: EntorhinalCortex instance
        atl: ATL instance
        clusters: List of cluster dicts with "name" and "sentences" keys
        encoder: LinguisticEncoder instance
        diagnostics: If True, collect per-sentence similarity diagnostics
    """
    from maxim.agents.bus import Percept
    from maxim.similarity.ec import _cosine_similarity

    # Track which node each sentence was assigned to
    sentence_nodes: list[tuple[str, str, str]] = []  # (cluster_name, sentence, node_id)
    node_to_cluster: dict[str, str] = {}  # node_id → cluster that created it
    node_history: list[int] = []  # unique node count after each sentence
    diags: list[SentenceDiag] = []

    all_sentences = []
    for cluster in clusters:
        for sentence in cluster.get("sentences", []):
            all_sentences.append((cluster["name"], sentence))

    for cluster_name, sentence in all_sentences:
        percept = Percept(
            timestamp=0.0,
            source="test",
            content=sentence,
            transcript_chunk=sentence,
        )

        # Get embedding before encode (so we can compute diagnostics)
        embedding = encoder.embed(sentence)

        # Find best match manually for diagnostics
        best_sim = 0.0
        best_match_node = ""
        best_match_cluster = ""
        if diagnostics:
            for nid, (stored_emb, stored_mod) in ec._substrate_nodes.items():
                if stored_mod != "text":
                    continue
                sim = _cosine_similarity(embedding, stored_emb)
                if sim > best_sim:
                    best_sim = sim
                    best_match_node = nid
                    best_match_cluster = node_to_cluster.get(nid, "?")

        # Now run the actual encode
        node_id = encoder.encode(percept)
        if node_id is None:
            logger.warning("Encoder returned None for sentence: %s", sentence[:50])
            continue

        is_new = node_id not in node_to_cluster
        if is_new:
            node_to_cluster[node_id] = cluster_name

        sentence_nodes.append((cluster_name, sentence, node_id))
        unique_nodes = len(set(n for _, _, n in sentence_nodes))
        node_history.append(unique_nodes)

        if diagnostics:
            diags.append(
                SentenceDiag(
                    cluster=cluster_name,
                    sentence=sentence,
                    node_id=node_id,
                    is_new=is_new,
                    best_sim=round(best_sim, 4),
                    best_match_node=best_match_node[:8] if best_match_node else "",
                    best_match_cluster=best_match_cluster,
                )
            )

    if not sentence_nodes:
        return P1Metrics()

    # --- Collapse rate: within-cluster pairs that share a node ---
    within_total = 0
    within_collapsed = 0
    cluster_details = []

    for cluster in clusters:
        cluster_name = cluster["name"]
        cluster_nodes = [n for cn, _, n in sentence_nodes if cn == cluster_name]
        if len(cluster_nodes) < 2:
            continue

        pairs = 0
        collapsed = 0
        for i in range(len(cluster_nodes)):
            for j in range(i + 1, len(cluster_nodes)):
                pairs += 1
                if cluster_nodes[i] == cluster_nodes[j]:
                    collapsed += 1

        within_total += pairs
        within_collapsed += collapsed

        unique_in_cluster = len(set(cluster_nodes))
        cluster_details.append(
            {
                "name": cluster_name,
                "sentences": len(cluster_nodes),
                "unique_nodes": unique_in_cluster,
                "collapse_rate": collapsed / pairs if pairs > 0 else 0.0,
            }
        )

    collapse_rate = within_collapsed / within_total if within_total > 0 else 0.0

    # --- Collect missed similarity scores ---
    # For each sentence that was NEW but whose cluster already had a node,
    # record the similarity to the cluster's existing node.
    missed_sims: list[float] = []
    if diagnostics:
        for d in diags:
            if d.is_new and d.best_match_cluster == d.cluster and d.best_sim > 0:
                missed_sims.append(d.best_sim)

    # --- Cross-cluster rate ---
    cross_total = 0
    cross_collapsed = 0
    cluster_names = [c["name"] for c in clusters]

    for i, name_i in enumerate(cluster_names):
        nodes_i = [n for cn, _, n in sentence_nodes if cn == name_i]
        for j in range(i + 1, len(cluster_names)):
            name_j = cluster_names[j]
            nodes_j = [n for cn, _, n in sentence_nodes if cn == name_j]
            for ni in nodes_i:
                for nj in nodes_j:
                    cross_total += 1
                    if ni == nj:
                        cross_collapsed += 1

    cross_cluster_rate = cross_collapsed / cross_total if cross_total > 0 else 0.0

    # --- Node stability: growth in final 20% ---
    total_nodes = node_history[-1] if node_history else 0
    n = len(node_history)
    if n >= 5:
        cutoff = int(n * 0.80)
        nodes_at_80 = node_history[cutoff] if cutoff < n else node_history[-1]
        growth = (total_nodes - nodes_at_80) / max(nodes_at_80, 1)
    else:
        growth = 0.0

    # --- Modality isolation ---
    modality_violations = 0
    for _, _, node_id in sentence_nodes:
        node_data = ec._substrate_nodes.get(node_id)
        if node_data is not None and node_data[1] != "text":
            modality_violations += 1

    return P1Metrics(
        collapse_rate=round(collapse_rate, 4),
        cross_cluster_rate=round(cross_cluster_rate, 4),
        node_growth_final_20pct=round(growth, 4),
        total_nodes=total_nodes,
        cluster_details=cluster_details,
        modality_violations=modality_violations,
        sentence_diags=diags,
        missed_sims=missed_sims,
    )
