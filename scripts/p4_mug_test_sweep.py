#!/usr/bin/env python
"""P4 Stage 2 v1 — Mug test sweep on real CLIP embeddings (single-shot).

**IMPORTANT — v1 script retained for backward compat, NOT the v2 sweep.**

This is the Stage 2 v1 one-shot mug test runner. Phase 2D v1 used this
script to produce the initial ``docs/experiments/p4_mug_test_sweep.md``
report with the tautological 1.000 ± 0.000 recall that Round 2 review
flagged as unfalsifiable (see ``scripts/p4_mug_test_sweep_v2.py`` for
the non-tautological replacement).

This script is kept for:

- Reproducing the v1 report exactly as-shipped (the v1 doc is still
  authoritative for "this is what Stage 2 v1 reported and why the
  review caught it").
- Smoke-testing the shared orchestrator at the simplest fixture shape
  (no noise, no bridges, per-modality thresholds 0.60/1.01).

The **shared orchestrator** ``build_and_bind`` now lives in
``tests/substrate/p4_build_and_bind.py`` and is parameterized over
encoder + threshold + optional noise + optional bridges. Both this
v1 script and the new v2 sweep import from there.

Outputs (v1-compat paths):

- ``docs/experiments/p4_mug_test_sweep.md``
- ``docs/experiments/results/p4_mug_test_sweep.json``

Usage::

    PYTHONPATH=src python scripts/p4_mug_test_sweep.py

For the non-tautological Phase 2D v2 sweep, see
``scripts/p4_mug_test_sweep_v2.py``.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger("p4_mug_test")


# Per-modality EC thresholds empirically calibrated for this fixture
# in Stage 2 v1. Documented in docs/experiments/p4_mug_test_sweep.md's
# "Calibration footnote" section. Stage 2 v2 re-parameterizes these.
TEXT_EC_THRESHOLD = 0.60
VISION_EC_THRESHOLD = 1.01


def _shortest_path_hops(binding_graph: Any, source: str, target: str, max_depth: int = 5) -> int | None:
    """Minimum edges from source to target in the binding graph
    (undirected BFS), or None if unreachable within max_depth."""
    if source == target:
        return 0

    visited = {source}
    queue: deque[tuple[str, int]] = deque([(source, 0)])
    while queue:
        node, depth = queue.popleft()
        if depth >= max_depth:
            continue
        neighbors = [t for t, _ in binding_graph.get_associated(node)]
        for n in neighbors:
            if n == target:
                return depth + 1
            if n not in visited:
                visited.add(n)
                queue.append((n, depth + 1))
    return None


def _run_forward_retrieval(
    hippocampus: Any,
    text_node_id: str,
    expected_vision_nodes: tuple[str, ...],
    limit: int = 20,
) -> tuple[float, list[int], list[int]]:
    results = hippocampus.retrieve_cross_modal(text_node_id, target_modality="vision", limit=limit)
    result_ids = [nid for nid, _ in results[:5]]
    per_sample_hits = [1 if vid in result_ids else 0 for vid in expected_vision_nodes]
    top5_recall = sum(per_sample_hits) / len(expected_vision_nodes)

    path_hops: list[int] = []
    for vid in expected_vision_nodes:
        if vid in result_ids:
            hops = _shortest_path_hops(hippocampus._binding_graph, text_node_id, vid)
            path_hops.append(hops if hops is not None else -1)
    return top5_recall, per_sample_hits, path_hops


def _run_reverse_retrieval(
    hippocampus: Any,
    vision_node_ids: tuple[str, ...],
    expected_text_id: str,
    limit: int = 5,
) -> float:
    matches = 0
    for vid in vision_node_ids:
        results = hippocampus.retrieve_cross_modal(vid, target_modality="text", limit=limit)
        ids = [nid for nid, _ in results]
        if expected_text_id in ids:
            matches += 1
    return matches / len(vision_node_ids)


def _compute_retrievals(hippocampus: Any, class_results: list) -> None:
    for cls in class_results:
        top5, hits, path_hops = _run_forward_retrieval(hippocampus, cls.text_node_id, cls.vision_node_ids, limit=20)
        cls.forward_top5_recall = top5
        cls.forward_hits_in_topk = hits
        cls.forward_path_hops = path_hops
        cls.reverse_text_match_rate = _run_reverse_retrieval(
            hippocampus, cls.vision_node_ids, cls.text_node_id, limit=5
        )


def _summarize(results: list) -> dict[str, Any]:
    forward_rates = [r.forward_top5_recall for r in results]
    reverse_rates = [r.reverse_text_match_rate for r in results]
    all_hops: list[int] = []
    for r in results:
        all_hops.extend(r.forward_path_hops)

    hop_histogram: dict[int, int] = {}
    for h in all_hops:
        hop_histogram[h] = hop_histogram.get(h, 0) + 1

    return {
        "n_classes": len(results),
        "forward_top5_recall_mean": float(np.mean(forward_rates)) if forward_rates else 0.0,
        "forward_top5_recall_std": float(np.std(forward_rates)) if forward_rates else 0.0,
        "forward_top5_recall_min": float(min(forward_rates)) if forward_rates else 0.0,
        "forward_top5_recall_max": float(max(forward_rates)) if forward_rates else 0.0,
        "reverse_top5_recall_mean": float(np.mean(reverse_rates)) if reverse_rates else 0.0,
        "reverse_top5_recall_std": float(np.std(reverse_rates)) if reverse_rates else 0.0,
        "path_hop_histogram": hop_histogram,
        "total_forward_hits": len(all_hops),
    }


def _write_report(
    out_md: Path,
    out_json: Path,
    results: list,
    summary: dict[str, Any],
    wall_clock_s: float,
    encoder_arm: str,
) -> None:
    md_lines = [
        "# Substrate P4 Stage 2 — Mug test sweep (real CLIP + paraphrase-mpnet)",
        "",
        f"**Wall clock:** {wall_clock_s:.1f}s",
        f"**Encoder arm:** {encoder_arm}",
        "**Fixture:** scenarios/substrate/p4_mug_test.yaml (10 classes × 5 samples)",
        "",
        "## Summary",
        "",
        f"- Forward top-5 recall (text → vision): **{summary['forward_top5_recall_mean']:.3f} ± {summary['forward_top5_recall_std']:.3f}** "
        f"(min {summary['forward_top5_recall_min']:.2f}, max {summary['forward_top5_recall_max']:.2f})",
        f"- Reverse top-5 recall (vision → text): **{summary['reverse_top5_recall_mean']:.3f} ± {summary['reverse_top5_recall_std']:.3f}**",
        f"- Total successful forward hits: {summary['total_forward_hits']} / {summary['n_classes'] * 5}",
        "",
    ]
    md_lines.extend(
        [
            "## Per-class results",
            "",
            "| class | forward top-5 | reverse top-5 | forward hits |",
            "|---|---|---|---|",
        ]
    )
    for r in results:
        md_lines.append(
            f"| {r.class_name} | {r.forward_top5_recall:.2f} | "
            f"{r.reverse_text_match_rate:.2f} | {sum(r.forward_hits_in_topk)}/5 |"
        )

    md_lines.extend(
        [
            "",
            "## How to reproduce",
            "",
            "```bash",
            "PYTHONPATH=src python scripts/p4_mug_test_sweep.py",
            "```",
        ]
    )
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(md_lines) + "\n")
    logger.info("wrote report %s", out_md)

    payload = {
        "encoder_arm": encoder_arm,
        "wall_clock_s": round(wall_clock_s, 2),
        "summary": summary,
        "per_class": [
            {
                "class_name": r.class_name,
                "text_node_id": r.text_node_id,
                "vision_node_ids": list(r.vision_node_ids),
                "forward_top5_recall": round(r.forward_top5_recall, 4),
                "forward_hits_in_topk": r.forward_hits_in_topk,
                "reverse_text_match_rate": round(r.reverse_text_match_rate, 4),
                "forward_path_hops": r.forward_path_hops,
            }
            for r in results
        ],
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")
    logger.info("wrote json %s", out_json)


def main() -> int:
    # Configure logging inside main() so importing this module has no
    # side effects — Exec #4 / Arch #9 fold.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    repo_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(repo_root / "tests"))

    from substrate.p4_build_and_bind import (
        BuildConfig,
        FixtureBridgeConfig,
        FixtureNoiseConfig,
        build_and_bind,
        clip_vision_encoder,
        paraphrase_mpnet_text_encoder,
    )
    from substrate.p4_fixture_loader import load_fixture_descriptor, load_fixture_images

    logger.info("P4 Stage 2 v1 mug test sweep (smoke test of shared orchestrator)")
    descriptor = load_fixture_descriptor()
    images = load_fixture_images()
    logger.info(
        "loaded %d classes × %d samples = %d pairs",
        len(descriptor.classes),
        descriptor.samples_per_class,
        len(images),
    )

    config = BuildConfig(
        text_encoder=paraphrase_mpnet_text_encoder(),
        vision_encoder=clip_vision_encoder(),
        text_ec_threshold=TEXT_EC_THRESHOLD,
        vision_ec_threshold=VISION_EC_THRESHOLD,
        noise=FixtureNoiseConfig(noise_reps=0),
        bridges=FixtureBridgeConfig(enabled=False),
        seed=0,
    )

    start = time.monotonic()
    logger.info("build_and_bind with Arm A (paraphrase-mpnet + clip-ViT-B-32)")
    build = build_and_bind(descriptor, images, config)
    logger.info("running retrievals")
    _compute_retrievals(build.hippocampus, build.class_results)
    wall_clock = time.monotonic() - start

    summary = _summarize(build.class_results)
    logger.info(
        "forward top-5 recall mean=%.3f std=%.3f",
        summary["forward_top5_recall_mean"],
        summary["forward_top5_recall_std"],
    )

    # Write to a smoke-test output path that does NOT clobber the
    # frozen v1 historical report at docs/experiments/p4_mug_test_sweep.md.
    # The v1 report stays as-shipped (with the tautological 1.000 recall
    # and the WRONG Option 2 decision) so docs/experiments/p4_mug_test_sweep_v2.md
    # can explicitly supersede it. This script now only proves the
    # migrated build_and_bind orchestrator still wires correctly; the
    # report it emits is a smoke-test artifact, not a published finding.
    out_md = repo_root / "docs" / "experiments" / "p4_mug_test_sweep_v1_smoke.md"
    out_json = repo_root / "docs" / "experiments" / "results" / "p4_mug_test_sweep_v1_smoke.json"
    _write_report(
        out_md,
        out_json,
        build.class_results,
        summary,
        wall_clock,
        encoder_arm=f"Arm A ({config.text_encoder.name} text + {config.vision_encoder.name} vision + hippocampus)",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
