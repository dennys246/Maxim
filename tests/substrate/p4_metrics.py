"""P4 Stage 3 — cross-modal retrieval metric extractor.

Provides the ``ArmMetrics`` dataclass and two extraction functions:

- ``compute_hippocampus_metrics`` — for Arms A and B (hippocampus-based
  cross-modal retrieval via ``retrieve_cross_modal``).
- ``compute_cosine_metrics`` — for Arm C (raw cosine similarity baseline,
  no hippocampus).

**Critical invariant:** all rates are **pooled per-probe** with n=50
(10 classes × 5 vision nodes). This is NOT the mean of 10 per-class
rates — that would mask single-class collapse. The pooled discipline
is pinned in the reproduction protocol before any seeds run.

Metric definitions:

- ``forward_rate``: for each of 10 text cues, retrieve top-5 vision
  nodes. Count correct hits across all 50 slots. Rate = total / 50.
- ``reverse_rate``: for each of 50 vision nodes, retrieve top-5 text
  nodes and check if the correct text class is present. Rate = hits / 50.
- ``false_binding_rate``: wrong-class vision nodes in forward top-5,
  pooled. Rate = total_wrong / 50.
- ``f1``: harmonic mean of ``forward_rate`` and ``reverse_rate``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ArmMetrics:
    """Per-seed per-arm retrieval metrics, all pooled per-probe."""

    forward_rate: float
    reverse_rate: float
    false_binding_rate: float
    f1: float


def _compute_f1(forward_rate: float, reverse_rate: float) -> float:
    """Harmonic mean of forward_rate and reverse_rate.

    Pre-merge review caught that the original formula (harmonic mean of
    forward_rate and 1-false_binding_rate) is algebraically redundant:
    forward_rate + false_binding_rate = 1.0 by construction (every
    top-5 slot is either correct or wrong), so 1-fbr = forward_rate,
    and harmonic_mean(x, x) = x. Using reverse_rate as the second leg
    makes F1 a genuine composite of two independent measurements
    (text→vision and vision→text retrieval quality).
    """
    if forward_rate + reverse_rate == 0:
        return 0.0
    return 2.0 * forward_rate * reverse_rate / (forward_rate + reverse_rate)


# ─────────────────────────────────────────────────────────────────────────
# Hippocampus-based metrics (Arms A and B)
# ─────────────────────────────────────────────────────────────────────────


def compute_hippocampus_metrics(
    hippocampus: Any,
    class_results: list[Any],
    vision_node_to_class: dict[str, str],
) -> ArmMetrics:
    """Compute pooled retrieval metrics using hippocampus cross-modal retrieval.

    Args:
        hippocampus: A wired ``Hippocampus`` instance from ``build_and_bind``.
        class_results: ``list[ClassResult]`` from ``BuildResult``.
        vision_node_to_class: Mapping from vision node id to class name.

    Returns:
        ``ArmMetrics`` with all rates pooled across 50 probes.
    """
    n_probes = sum(len(cr.vision_node_ids) for cr in class_results)

    # ── Forward: 10 text probes × top-5 ──
    total_forward_correct = 0
    total_forward_wrong = 0

    for cr in class_results:
        results = hippocampus.retrieve_cross_modal(cr.text_node_id, target_modality="vision", limit=20)
        top5 = [nid for nid, _ in results[:5]]
        for nid in top5:
            actual_class = vision_node_to_class.get(nid)
            if actual_class == cr.class_name:
                total_forward_correct += 1
            else:
                total_forward_wrong += 1

    forward_rate = total_forward_correct / n_probes
    false_binding_rate = total_forward_wrong / n_probes

    # ── Reverse: 50 vision probes × top-5 text ──
    total_reverse_correct = 0

    for cr in class_results:
        for vid in cr.vision_node_ids:
            results = hippocampus.retrieve_cross_modal(vid, target_modality="text", limit=5)
            retrieved_text_ids = {nid for nid, _ in results}
            if cr.text_node_id in retrieved_text_ids:
                total_reverse_correct += 1

    reverse_rate = total_reverse_correct / n_probes

    f1 = _compute_f1(forward_rate, reverse_rate)
    return ArmMetrics(
        forward_rate=forward_rate,
        reverse_rate=reverse_rate,
        false_binding_rate=false_binding_rate,
        f1=f1,
    )


# ─────────────────────────────────────────────────────────────────────────
# Cosine baseline metrics (Arm C)
# ─────────────────────────────────────────────────────────────────────────


def compute_cosine_metrics(
    text_embeddings: dict[str, np.ndarray],
    vision_entries: list[tuple[str, np.ndarray]],
) -> ArmMetrics:
    """Compute pooled retrieval metrics using raw cosine similarity.

    No hippocampus, no EC, no episodes — pure embedding math.

    Args:
        text_embeddings: ``{class_name: normalized_embedding}`` for 10 classes.
        vision_entries: ``[(class_name, normalized_embedding), ...]`` for 50 images.

    Returns:
        ``ArmMetrics`` with all rates pooled across 50 probes.
    """
    class_names = list(text_embeddings.keys())
    text_matrix = np.array([text_embeddings[cn] for cn in class_names])  # (10, D)

    vision_classes = [cls for cls, _ in vision_entries]
    vision_matrix = np.array([emb for _, emb in vision_entries])  # (50, D)

    n_probes = len(vision_entries)

    # ── Forward: 10 text probes × top-5 vision ──
    # scores shape: (10, 50) — cosine similarity (vectors already normalized)
    scores_fwd = text_matrix @ vision_matrix.T
    total_forward_correct = 0
    total_forward_wrong = 0

    for i, cn in enumerate(class_names):
        top5_indices = np.argsort(scores_fwd[i])[::-1][:5]
        for idx in top5_indices:
            if vision_classes[idx] == cn:
                total_forward_correct += 1
            else:
                total_forward_wrong += 1

    forward_rate = total_forward_correct / n_probes
    false_binding_rate = total_forward_wrong / n_probes

    # ── Reverse: 50 vision probes × top-5 text ──
    scores_rev = vision_matrix @ text_matrix.T  # (50, 10)
    total_reverse_correct = 0

    for j, actual_class in enumerate(vision_classes):
        top5_text_indices = np.argsort(scores_rev[j])[::-1][:5]
        top5_text_classes = {class_names[idx] for idx in top5_text_indices}
        if actual_class in top5_text_classes:
            total_reverse_correct += 1

    reverse_rate = total_reverse_correct / n_probes

    f1 = _compute_f1(forward_rate, reverse_rate)
    return ArmMetrics(
        forward_rate=forward_rate,
        reverse_rate=reverse_rate,
        false_binding_rate=false_binding_rate,
        f1=f1,
    )
