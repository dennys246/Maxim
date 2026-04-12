"""OpenCLIP text-encoder baseline for paraphrase cluster evaluation.

P0.3 of substrate_p0_pilot. Uses OpenCLIP's text encoder to embed
sentences and clusters by cosine similarity. The number pinned here
becomes P4's gate baseline — the architecture must beat it.

Requires: open_clip_torch, torch. Falls back gracefully if unavailable.

Usage:
    from tests.substrate.baselines.openclip_baseline import run_openclip_baseline
    result = run_openclip_baseline(
        fixture_path="scenarios/substrate/paraphrase_clusters.yaml",
        threshold=0.80,
        seed=42,
    )
"""

from __future__ import annotations

import logging
from pathlib import Path

from tests.substrate.baselines.embedding_baseline import (
    BaselineResult,
    _compute_collapse_rate,
    _compute_cross_cluster_rate,
    _compute_near_miss_separation,
    load_fixture_sentences,
)

logger = logging.getLogger(__name__)


def run_openclip_baseline(
    fixture_path: str | Path,
    *,
    model_name: str = "ViT-B-32",
    pretrained: str = "openai",
    threshold: float = 0.80,
    seed: int = 42,
) -> BaselineResult:
    """Run the OpenCLIP text-encoder baseline on a fixture.

    This uses OpenCLIP's text encoder only (no vision). The score
    is pinned before P4 starts and becomes the gate baseline that
    the hippocampus-mediated cross-modal binding must beat.

    Args:
        fixture_path: Path to paraphrase_clusters.yaml
        model_name: OpenCLIP model name (default: ViT-B-32)
        pretrained: Pretrained weights (default: openai)
        threshold: cosine similarity threshold for clustering
        seed: random seed

    Returns:
        BaselineResult with metrics
    """
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)

    sentences = load_fixture_sentences(fixture_path)
    if not sentences:
        return BaselineResult(model_name=f"openclip:{model_name}", threshold=threshold, seed=seed)

    texts = [s["text"] for s in sentences]

    # Embed via OpenCLIP text encoder
    try:
        import open_clip
        import torch

        model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        tokenizer = open_clip.get_tokenizer(model_name)

        with torch.no_grad():
            tokens = tokenizer(texts)
            text_features = model.encode_text(tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            embeddings = text_features.cpu().numpy()
    except ImportError:
        logger.warning("open_clip_torch not installed, cannot run OpenCLIP baseline")
        return BaselineResult(
            model_name=f"openclip:{model_name}",
            threshold=threshold,
            seed=seed,
            details={"error": "open_clip_torch not installed"},
        )

    # Same greedy clustering as embedding baseline
    n = len(embeddings)
    labels = [-1] * n
    centroids: list[np.ndarray] = []
    next_label = 0

    for i in range(n):
        if not centroids:
            labels[i] = next_label
            centroids.append(embeddings[i])
            next_label += 1
            continue

        sims = np.dot(np.array(centroids), embeddings[i])
        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])

        if best_sim >= threshold:
            labels[i] = best_idx
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

    # Per-difficulty
    easy_c = medium_c = hard_c = 0.0
    for diff, attr in [("easy", "easy_collapse"), ("medium", "medium_collapse"), ("hard", "hard_collapse")]:
        subset = [(s, lb) for s, lb in zip(sentences, labels) if s["difficulty"] == diff]
        if subset:
            sub_s, sub_l = zip(*subset)
            val = _compute_collapse_rate(list(sub_s), list(sub_l))
        else:
            val = 0.0
        if diff == "easy":
            easy_c = val
        elif diff == "medium":
            medium_c = val
        else:
            hard_c = val

    # Per-split
    train_c = holdout_c = 0.0
    for split in ("train", "holdout"):
        subset = [(s, lb) for s, lb in zip(sentences, labels) if s["split"] == split]
        if subset:
            sub_s, sub_l = zip(*subset)
            val = _compute_collapse_rate(list(sub_s), list(sub_l))
        else:
            val = 0.0
        if split == "train":
            train_c = val
        else:
            holdout_c = val

    gt_clusters = len(set(s["cluster"] for s in sentences))

    return BaselineResult(
        model_name=f"openclip:{model_name}:{pretrained}",
        threshold=threshold,
        seed=seed,
        collapse_rate=round(collapse, 4),
        cross_cluster_rate=round(cross, 4),
        num_predicted_clusters=next_label,
        num_ground_truth_clusters=gt_clusters,
        num_sentences=n,
        easy_collapse=round(easy_c, 4),
        medium_collapse=round(medium_c, 4),
        hard_collapse=round(hard_c, 4),
        train_collapse=round(train_c, 4),
        holdout_collapse=round(holdout_c, 4),
        near_miss_separation=round(near_miss, 4),
    )
