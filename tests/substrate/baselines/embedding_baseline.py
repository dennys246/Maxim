"""FAISS + cosine embedding baseline for paraphrase cluster evaluation.

P0.2 of substrate_p0_pilot. Embeds each sentence with sentence-transformers,
clusters by cosine similarity with a fixed threshold, and scores against
ground-truth cluster labels.

Requires: sentence-transformers, numpy. Optional: faiss-cpu (falls back to
brute-force cosine if unavailable).

Usage:
    from tests.substrate.baselines.embedding_baseline import run_embedding_baseline
    result = run_embedding_baseline(
        fixture_path="scenarios/substrate/paraphrase_clusters.yaml",
        model_name="all-MiniLM-L6-v2",
        threshold=0.75,
        seed=42,
    )
    print(result)  # {"collapse_rate": 0.82, "cross_cluster_rate": 0.03, ...}
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


@dataclass
class BaselineResult:
    """Result of a baseline evaluation run."""

    model_name: str = ""
    threshold: float = 0.0
    seed: int = 0

    # Core metrics
    collapse_rate: float = 0.0  # within-cluster → same predicted cluster
    cross_cluster_rate: float = 0.0  # distinct clusters that merged
    num_predicted_clusters: int = 0
    num_ground_truth_clusters: int = 0
    num_sentences: int = 0

    # Per-difficulty breakdown
    easy_collapse: float = 0.0
    medium_collapse: float = 0.0
    hard_collapse: float = 0.0

    # Split breakdown
    train_collapse: float = 0.0
    holdout_collapse: float = 0.0

    # Near-miss pair accuracy (did the baseline separate near-miss pairs?)
    near_miss_separation: float = 0.0

    details: dict[str, Any] = field(default_factory=dict)


def load_fixture_sentences(fixture_path: str | Path) -> list[dict[str, Any]]:
    """Load sentences and metadata from a paraphrase cluster fixture YAML."""
    path = Path(fixture_path)
    with open(path) as f:
        data = yaml.safe_load(f)

    sentences = []
    for percept in data.get("percepts", []):
        text = percept.get("cli_input") or percept.get("content") or ""
        meta = percept.get("metadata", {})
        if text and "cluster" in meta:
            sentences.append(
                {
                    "text": text,
                    "cluster": meta["cluster"],
                    "split": meta.get("split", "train"),
                    "difficulty": meta.get("difficulty", "medium"),
                }
            )
    return sentences


def _compute_collapse_rate(
    sentences: list[dict[str, Any]],
    labels: list[int],
) -> float:
    """Fraction of within-cluster sentence pairs that got the same predicted label."""
    from collections import defaultdict

    cluster_to_indices: dict[str, list[int]] = defaultdict(list)
    for i, s in enumerate(sentences):
        cluster_to_indices[s["cluster"]].append(i)

    correct = 0
    total = 0
    for indices in cluster_to_indices.values():
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                total += 1
                if labels[indices[i]] == labels[indices[j]]:
                    correct += 1

    return correct / total if total > 0 else 0.0


def _compute_cross_cluster_rate(
    sentences: list[dict[str, Any]],
    labels: list[int],
) -> float:
    """Fraction of distinct ground-truth clusters that share a predicted label."""
    from collections import defaultdict

    cluster_to_predicted: dict[str, set[int]] = defaultdict(set)
    for i, s in enumerate(sentences):
        cluster_to_predicted[s["cluster"]].add(labels[i])

    # Two distinct clusters "merged" if they share any predicted label
    clusters = list(cluster_to_predicted.keys())
    merged = 0
    pairs = 0
    for i in range(len(clusters)):
        for j in range(i + 1, len(clusters)):
            pairs += 1
            if cluster_to_predicted[clusters[i]] & cluster_to_predicted[clusters[j]]:
                merged += 1

    return merged / pairs if pairs > 0 else 0.0


# Near-miss pairs: clusters that share vocabulary but should stay separate
NEAR_MISS_PAIRS = [
    ("weather_nice", "weather_bad"),
    ("request_delete_files", "request_list_files"),
    ("robot_pick_up", "robot_put_down"),
    ("explain_gravity", "explain_magnetism"),
    ("emotional_happy", "emotional_sad"),
    ("location_kitchen", "location_bedroom"),
    ("code_bug_report", "code_feature_request"),
    ("schedule_meeting", "cancel_meeting"),
    ("navigation_left", "navigation_right"),
    ("request_help_implicit", "expressing_capability"),
    ("sarcastic_compliment", "genuine_compliment"),
    ("urgency_real", "urgency_fake"),
    ("describe_pain_physical", "describe_pain_emotional"),
    ("causal_reasoning", "hypothetical_reasoning"),
    ("temporal_past", "temporal_future"),
    ("negation_understanding", "affirmation_delete"),
    ("tool_use_implicit", "tool_use_explicit"),
    ("context_switch_cooking", "context_switch_code"),
    ("quantity_large", "quantity_small"),
]


def _compute_near_miss_separation(
    sentences: list[dict[str, Any]],
    labels: list[int],
) -> float:
    """Fraction of near-miss cluster pairs that the baseline kept separate."""
    from collections import defaultdict

    cluster_to_predicted: dict[str, set[int]] = defaultdict(set)
    for i, s in enumerate(sentences):
        cluster_to_predicted[s["cluster"]].add(labels[i])

    separated = 0
    total = 0
    for c1, c2 in NEAR_MISS_PAIRS:
        if c1 in cluster_to_predicted and c2 in cluster_to_predicted:
            total += 1
            if not (cluster_to_predicted[c1] & cluster_to_predicted[c2]):
                separated += 1

    return separated / total if total > 0 else 0.0


def run_embedding_baseline(
    fixture_path: str | Path,
    *,
    model_name: str = "all-MiniLM-L6-v2",
    threshold: float = 0.75,
    seed: int = 42,
) -> BaselineResult:
    """Run the FAISS+cosine embedding baseline on a fixture.

    Args:
        fixture_path: Path to paraphrase_clusters.yaml
        model_name: sentence-transformers model name
        threshold: cosine similarity threshold for same-cluster assignment
        seed: random seed for reproducibility

    Returns:
        BaselineResult with metrics
    """
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)

    sentences = load_fixture_sentences(fixture_path)
    if not sentences:
        return BaselineResult(model_name=model_name, threshold=threshold, seed=seed)

    texts = [s["text"] for s in sentences]

    # Embed
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(model_name)
        embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    except ImportError:
        logger.warning("sentence-transformers not installed, cannot run embedding baseline")
        return BaselineResult(
            model_name=model_name,
            threshold=threshold,
            seed=seed,
            details={"error": "sentence-transformers not installed"},
        )

    # Greedy clustering by cosine similarity
    # First sentence starts cluster 0. Each subsequent sentence joins the
    # nearest existing cluster if similarity > threshold, else starts new.
    n = len(embeddings)
    labels = [-1] * n
    centroids: list[Any] = []
    next_label = 0

    for i in range(n):
        if not centroids:
            labels[i] = next_label
            centroids.append(embeddings[i])
            next_label += 1
            continue

        # Cosine similarity to each centroid (embeddings are already normalized)
        sims = np.dot(np.array(centroids), embeddings[i])
        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])

        if best_sim >= threshold:
            labels[i] = best_idx
            # Update centroid (running mean)
            count = sum(1 for lb in labels[:i] if lb == best_idx)
            centroids[best_idx] = (centroids[best_idx] * count + embeddings[i]) / (count + 1)
            centroids[best_idx] = centroids[best_idx] / np.linalg.norm(centroids[best_idx])
        else:
            labels[i] = next_label
            centroids.append(embeddings[i])
            next_label += 1

    # Compute metrics
    collapse = _compute_collapse_rate(sentences, labels)
    cross = _compute_cross_cluster_rate(sentences, labels)
    near_miss = _compute_near_miss_separation(sentences, labels)

    # Per-difficulty breakdown
    for diff in ("easy", "medium", "hard"):
        subset = [(s, lb) for s, lb in zip(sentences, labels) if s["difficulty"] == diff]
        if subset:
            sub_sentences, sub_labels = zip(*subset)
            rate = _compute_collapse_rate(list(sub_sentences), list(sub_labels))
        else:
            rate = 0.0
        if diff == "easy":
            easy_collapse = rate
        elif diff == "medium":
            medium_collapse = rate
        else:
            hard_collapse = rate

    # Per-split breakdown
    for split in ("train", "holdout"):
        subset = [(s, lb) for s, lb in zip(sentences, labels) if s["split"] == split]
        if subset:
            sub_sentences, sub_labels = zip(*subset)
            rate = _compute_collapse_rate(list(sub_sentences), list(sub_labels))
        else:
            rate = 0.0
        if split == "train":
            train_collapse = rate
        else:
            holdout_collapse = rate

    gt_clusters = len(set(s["cluster"] for s in sentences))

    return BaselineResult(
        model_name=model_name,
        threshold=threshold,
        seed=seed,
        collapse_rate=round(collapse, 4),
        cross_cluster_rate=round(cross, 4),
        num_predicted_clusters=next_label,
        num_ground_truth_clusters=gt_clusters,
        num_sentences=n,
        easy_collapse=round(easy_collapse, 4),
        medium_collapse=round(medium_collapse, 4),
        hard_collapse=round(hard_collapse, 4),
        train_collapse=round(train_collapse, 4),
        holdout_collapse=round(holdout_collapse, 4),
        near_miss_separation=round(near_miss, 4),
    )
