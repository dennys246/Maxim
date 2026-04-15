#!/usr/bin/env python
"""P4 Stage 2 v2 — Mug test calibration sweep (non-tautological).

Stage 2 v1's mug test was tautological under ``VISION_EC_THRESHOLD=1.01``:
every class had 5 distinct vision nodes direct-bound to 1 text node and
no cross-class reachability paths, so forward top-5 retrieval was
mechanically forced to 1.000 regardless of whether the substrate's
ranker was actually doing anything. The v1 "100% 1-hop → defer Option 2"
decision rested on unfalsifiable evidence.

v2 sweeps two dimensions of the fixture construction:

1. **Noise level** (`noise_reps` ∈ {0, 1, 2, 3, 4, 5}): each class gains
   1 cross-class contamination pair reinforced N times. Creates weak
   Hebbian edges at weight `0.3 + (N-1)*0.1` that the substrate's
   ranker has to distinguish from the strong (N=5 → weight 0.7)
   class-correct edges.
2. **Bridge topology** (`bridges` ∈ {none, shared_superclass}): when
   enabled, adds a `text_flower` superclass text node bound to each
   class's text_X and vision_X_0, creating 2-hop paths
   `text_X → text_flower → vision_Y_0` that are blocked under Stage
   1's single-hop `retrieve_cross_modal` filter but reachable under
   Option 2's hypothetical split filter.

For each of the 12 combinations, the sweep measures:

- Same-class top-5 recall (Approach A signal-vs-noise ranking validation)
- Cross-class single-hop reachability rate (current Stage 1 behavior)
- Cross-class simulated-multi-hop reachability rate (Option 2
  prediction, computed via raw BFS over the binding graph ignoring
  the modality filter)
- Option 2 lift = (multi-hop − single-hop) as a fraction of all
  cross-class (text_X, vision_Y, X≠Y) pairs
- Binding graph topology histogram

Outputs:

- ``docs/experiments/p4_mug_test_sweep_v2.md``
- ``docs/experiments/results/p4_mug_test_sweep_v2.json``

Operating point selection rule (from the P4 plan's Stage 2 v2 fold
status section): pick the LARGEST `noise_reps` at which same-class
top-5 recall ≥ 0.90. Bridge topology is always ``shared_superclass``
for shipping; the ``none`` rows are included as baselines that
demonstrate the metric degradation curve.

Usage::

    PYTHONPATH=src python scripts/p4_mug_test_sweep_v2.py
"""

from __future__ import annotations

import json
import logging
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger("p4_mug_test_v2")


# Stage 2 v2 per-modality EC thresholds. Same as v1 because the
# calibration reasoning hasn't changed (bare class names + high
# vision threshold), but the v2 sweep now uses these via BuildConfig
# so Stage 3's Arm B can swap them for CLIP-text calibration.
TEXT_EC_THRESHOLD = 0.60
VISION_EC_THRESHOLD = 1.01

NOISE_REPS_SWEEP = [0, 1, 2, 3, 4, 5]
BRIDGE_CONDITIONS = [False, True]  # False=none, True=shared_superclass


@dataclass
class SweepPoint:
    """One row in the 12-combination sweep table."""

    noise_reps: int
    bridge_enabled: bool

    # Same-class top-5 recall (signal-vs-noise ranker test)
    mean_forward_top5_recall: float
    std_forward_top5_recall: float
    min_forward_top5_recall: float

    # Cross-class reachability — the Option 2 decision data
    n_cross_class_pairs: int  # 10×9 = 90
    n_single_hop_reachable: int  # current Stage 1 behavior
    n_multi_hop_reachable: int  # Option 2 simulation (raw BFS)
    option_2_lift_pct: float  # (multi − single) / total

    # Binding graph topology
    n_class_episodes: int
    n_noise_episodes: int
    n_bridge_episodes: int
    n_total_episodes: int
    n_vision_nodes: int
    n_text_nodes: int

    wall_clock_s: float


def _shortest_path_hops_raw(
    binding_graph: Any,
    source: str,
    target: str,
    max_depth: int = 5,
) -> int | None:
    """Minimum edges from source to target in the binding graph —
    pure BFS, NO modality filter. This is what Option 2's split
    filter would see under ``traversal_filter=None, result_filter=
    modality_membership``. Compare against the current retrieve_cross_modal
    behavior (which rejects same-modality intermediates) to compute
    the Option 2 lift.
    """
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


def _compute_forward_top5_recall(hippocampus: Any, class_results: list) -> list[float]:
    """Same-class forward top-5 recall per class. For each class, ask
    retrieve_cross_modal(text_X, "vision") for top-5 and count how
    many of the 5 class-correct vision nodes are in the top-5 result.

    With noise present, the retrieval pool includes the noise vision
    node(s). A correctly-ranking substrate should place the 5 strong
    (weight 0.7) edges above the weak (weight 0.3+) noise edges, so
    top-5 recall stays at 1.0 until noise weight approaches signal
    weight. At noise_reps=5 both sides tie at 0.7 and top-5 recall
    degrades because noise nodes displace signal nodes in the top
    band.
    """
    recalls: list[float] = []
    for cls in class_results:
        results = hippocampus.retrieve_cross_modal(cls.text_node_id, target_modality="vision", limit=20)
        result_ids = {nid for nid, _ in results[:5]}
        hits = sum(1 for vid in cls.vision_node_ids if vid in result_ids)
        recalls.append(hits / len(cls.vision_node_ids))
    return recalls


def _count_cross_class_reachability(
    hippocampus: Any,
    class_results: list,
) -> tuple[int, int, int]:
    """For every (text_X, vision_Y, X≠Y) pair, count:

    - n_total: 10 × 9 × 5 = 450 pairs (each class's text vs every
      other class's 5 vision nodes)
    - n_single_hop: reachable under current retrieve_cross_modal
      (runs the actual retrieval and counts hits)
    - n_multi_hop: reachable under raw BFS (Option 2 simulation)

    The Option 2 lift is (n_multi - n_single) / n_total as a
    percentage.
    """
    n_total = 0
    n_single_hop = 0
    n_multi_hop = 0

    # Pre-compute each class's "single-hop reachable from text_X"
    # set once via retrieve_cross_modal with a generous limit
    single_hop_reachable_by_class: dict[str, set[str]] = {}
    for cls in class_results:
        results = hippocampus.retrieve_cross_modal(cls.text_node_id, target_modality="vision", limit=200)
        single_hop_reachable_by_class[cls.class_name] = {nid for nid, _ in results}

    for src_cls in class_results:
        for dst_cls in class_results:
            if src_cls.class_name == dst_cls.class_name:
                continue
            for vid in dst_cls.vision_node_ids:
                n_total += 1
                if vid in single_hop_reachable_by_class[src_cls.class_name]:
                    n_single_hop += 1
                # Multi-hop check via raw BFS
                hops = _shortest_path_hops_raw(
                    hippocampus._binding_graph,
                    src_cls.text_node_id,
                    vid,
                    max_depth=5,
                )
                if hops is not None:
                    n_multi_hop += 1

    return n_total, n_single_hop, n_multi_hop


def _run_one_combination(
    descriptor: Any,
    images: dict,
    text_encoder: Any,
    vision_encoder: Any,
    noise_reps: int,
    bridge_enabled: bool,
) -> SweepPoint:
    """Run build_and_bind once with the given parameters, compute the
    metrics, return a SweepPoint."""
    from tests.substrate.p4_build_and_bind import (
        BuildConfig,
        FixtureBridgeConfig,
        FixtureNoiseConfig,
        build_and_bind,
    )

    config = BuildConfig(
        text_encoder=text_encoder,
        vision_encoder=vision_encoder,
        text_ec_threshold=TEXT_EC_THRESHOLD,
        vision_ec_threshold=VISION_EC_THRESHOLD,
        noise=FixtureNoiseConfig(noise_reps=noise_reps),
        bridges=FixtureBridgeConfig(enabled=bridge_enabled),
        seed=0,
    )

    start = time.monotonic()
    build = build_and_bind(descriptor, images, config)

    forward_recalls = _compute_forward_top5_recall(build.hippocampus, build.class_results)
    n_total, n_single, n_multi = _count_cross_class_reachability(build.hippocampus, build.class_results)

    wall = time.monotonic() - start

    lift_pct = 100.0 * (n_multi - n_single) / n_total if n_total > 0 else 0.0

    return SweepPoint(
        noise_reps=noise_reps,
        bridge_enabled=bridge_enabled,
        mean_forward_top5_recall=float(np.mean(forward_recalls)),
        std_forward_top5_recall=float(np.std(forward_recalls)),
        min_forward_top5_recall=float(min(forward_recalls)),
        n_cross_class_pairs=n_total,
        n_single_hop_reachable=n_single,
        n_multi_hop_reachable=n_multi,
        option_2_lift_pct=lift_pct,
        n_class_episodes=build.stats.n_class_episodes,
        n_noise_episodes=build.stats.n_noise_episodes,
        n_bridge_episodes=build.stats.n_bridge_episodes,
        n_total_episodes=build.stats.n_total_episodes,
        n_vision_nodes=build.stats.n_vision_nodes,
        n_text_nodes=build.stats.n_text_nodes,
        wall_clock_s=wall,
    )


def _pick_operating_point(points: list[SweepPoint]) -> SweepPoint | None:
    """Pick the LARGEST noise_reps at which same-class mean forward
    top-5 recall ≥ 0.90, among bridge_enabled=True rows (the canonical
    ship shape). Returns None if no row meets the bar.
    """
    candidates = [p for p in points if p.bridge_enabled and p.mean_forward_top5_recall >= 0.90]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.noise_reps)


def _write_report(
    out_md: Path,
    out_json: Path,
    points: list[SweepPoint],
    operating_point: SweepPoint | None,
    total_wall_clock_s: float,
    encoder_arm: str,
) -> None:
    md_lines = [
        "# Substrate P4 Stage 2 v2 — Mug test calibration sweep (non-tautological)",
        "",
        f"**Total wall clock:** {total_wall_clock_s:.1f}s",
        f"**Encoder arm:** {encoder_arm}",
        "**Sweep dimensions:** noise_reps ∈ {0,1,2,3,4,5} × bridges ∈ {none, shared_superclass}",
        "**Total combinations:** 12",
        "",
        "## Why v2 exists",
        "",
        "Stage 2 v1's mug test (`docs/experiments/p4_mug_test_sweep.md`)",
        "reported 1.000 ± 0.000 forward top-5 recall and 100% 1-hop retrievals,",
        "and the v1 report concluded 'Option 2 can be deferred.' Round 2",
        "Architecture-lens review caught this as tautological: with",
        "`VISION_EC_THRESHOLD=1.01`, each class had 5 distinct vision nodes",
        "direct-bound to 1 text node and NO cross-class reachability paths.",
        "`retrieve_cross_modal(text_X, 'vision')` could only ever return",
        "the 5 class-correct vision nodes because nothing else was reachable.",
        "Top-5 recall was mechanically forced to 1.0. The metric did not",
        "distinguish a working substrate from a broken one.",
        "",
        "v2 rebuilds Phase 2D with two new fixture layers that create real",
        "signal-vs-noise ranking pressure AND real multi-hop cross-modal",
        "reachability paths, then sweeps the parameter space to find an",
        "operating point where both metrics are non-trivially measurable.",
        "",
        "## Sweep results table",
        "",
        "| noise_reps | bridges | mean recall | std | min | cross pairs | 1-hop | multi-hop | Option 2 lift | episodes | wall |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for p in points:
        bridge = "shared" if p.bridge_enabled else "none"
        md_lines.append(
            f"| {p.noise_reps} | {bridge} | "
            f"{p.mean_forward_top5_recall:.3f} | "
            f"{p.std_forward_top5_recall:.3f} | "
            f"{p.min_forward_top5_recall:.2f} | "
            f"{p.n_cross_class_pairs} | "
            f"{p.n_single_hop_reachable} | "
            f"{p.n_multi_hop_reachable} | "
            f"{p.option_2_lift_pct:+.1f}% | "
            f"{p.n_total_episodes} | "
            f"{p.wall_clock_s:.1f}s |"
        )

    md_lines.extend(
        [
            "",
            "## Same-class top-5 recall degradation curve",
            "",
            "For the bridge-enabled rows (the canonical ship shape), how does",
            "same-class forward top-5 recall degrade as noise_reps increases?",
            "This is Approach A — the signal-vs-noise ranker test.",
            "",
            "| noise_reps | mean recall | pass (≥0.90)? |",
            "|---|---|---|",
        ]
    )
    for p in [p for p in points if p.bridge_enabled]:
        passes = "✅" if p.mean_forward_top5_recall >= 0.90 else "❌"
        md_lines.append(f"| {p.noise_reps} | {p.mean_forward_top5_recall:.3f} | {passes} |")

    md_lines.extend(
        [
            "",
            "## Option 2 lift across noise levels (bridge topology ON)",
            "",
            "How many cross-class (text_X, vision_Y, X≠Y) pairs become",
            "reachable when we SIMULATE Option 2's split filter (raw BFS",
            "over the binding graph, ignoring modality membership at",
            "traversal time)? If Option 2 would unlock non-zero retrieval",
            "paths at the chosen operating point, Option 2 becomes a Stage 3",
            "blocker.",
            "",
            "| noise_reps | 1-hop reachable | multi-hop reachable | Option 2 lift |",
            "|---|---|---|---|",
        ]
    )
    for p in [p for p in points if p.bridge_enabled]:
        md_lines.append(
            f"| {p.noise_reps} | {p.n_single_hop_reachable} | {p.n_multi_hop_reachable} | {p.option_2_lift_pct:+.1f}% |"
        )

    md_lines.extend(["", "## Operating point selection", ""])
    if operating_point is None:
        md_lines.extend(
            [
                "**NO operating point found.** No bridge-enabled row has",
                "mean forward top-5 recall ≥ 0.90. This suggests either:",
                "",
                "- The noise floor is too aggressive (unlikely at noise_reps=0)",
                "- The encoder or EC calibration is wrong for this fixture",
                "- The substrate's ranker is structurally broken",
                "",
                "**Stage 2 v2 ships without a new pinned fixture.** The",
                "v1 fixture stays in place; manual investigation is required",
                "before Stage 3.",
            ]
        )
    else:
        md_lines.extend(
            [
                f"**Operating point: `noise_reps={operating_point.noise_reps}, bridges=shared_superclass`.**",
                "",
                f"- Mean forward top-5 recall: **{operating_point.mean_forward_top5_recall:.3f}** (≥0.90 ✅)",
                f"- Std: {operating_point.std_forward_top5_recall:.3f}",
                f"- Min: {operating_point.min_forward_top5_recall:.2f}",
                f"- Cross-class pairs: {operating_point.n_cross_class_pairs}",
                f"- Single-hop reachable: {operating_point.n_single_hop_reachable}",
                f"- Multi-hop reachable (Option 2 simulated): {operating_point.n_multi_hop_reachable}",
                f"- **Option 2 lift: {operating_point.option_2_lift_pct:+.1f}%**",
                "",
                "Selection rule: largest noise_reps at which mean recall is",
                "still ≥ 0.90 (user-chosen tighter threshold from the fold",
                "planning session). Bridge topology is always",
                "`shared_superclass` for the ship shape.",
                "",
                "## Option 2 decision — re-opened from Stage 2 v1",
                "",
            ]
        )
        lift = operating_point.option_2_lift_pct
        if abs(lift) < 0.5:
            md_lines.extend(
                [
                    f"**Option 2 lift is ~0% ({lift:+.1f}%) at the operating point.**",
                    "The shared_superclass bridge creates text-text paths that",
                    "retrieve_cross_modal's current single-hop filter BLOCKS",
                    "(text_flower is a text node, not in the vision bucket, so",
                    "BFS truncates there). Raw BFS (Option 2 simulation) reaches",
                    "the same count of cross-class vision nodes — suggesting",
                    "the bridge is being walked under the current filter too.",
                    "This is an unexpected result that needs investigation —",
                    "either the retrieve_cross_modal implementation is NOT",
                    "blocking bridges, or the cross-class reachability counter",
                    "has a bug.",
                    "",
                    "**Option 2 decision: HOLD, investigate before committing.**",
                ]
            )
        elif lift > 0:
            md_lines.extend(
                [
                    f"**Option 2 lift is {lift:+.1f}% at the operating point.** The",
                    "shared_superclass bridge creates cross-class retrieval paths",
                    "that Stage 1's single-hop filter blocks; Option 2's split",
                    "filter would unlock them. Given non-zero lift under realistic",
                    "fixture conditions (bridge-enabled, noise ≥1), Option 2 is",
                    "empirically needed for the P4 mechanism to reach its full",
                    "advertised capability.",
                    "",
                    "**Option 2 decision: SHIP.** Per the fold-planning commit,",
                    "Option 2 goes in a SEPARATE follow-up PR after this fold",
                    "merges. That PR renames node_filter→traversal_filter, adds",
                    "result_filter, provides a P3b compat shim, re-validates",
                    "P3a's 10-seed sweep, and flips TestStageThreeLimitation.",
                ]
            )
        else:
            md_lines.extend(
                [
                    f"**Option 2 lift is negative ({lift:+.1f}%) at the operating point.**",
                    "This means multi-hop BFS reaches FEWER cross-class pairs than",
                    "single-hop retrieve_cross_modal, which is impossible under",
                    "correct semantics — indicates a bug in either the metric or",
                    "the retrieval.",
                    "",
                    "**Option 2 decision: HOLD, debug the metric first.**",
                ]
            )

    md_lines.extend(
        [
            "",
            "## How to reproduce",
            "",
            "```bash",
            "PYTHONPATH=src python scripts/p4_mug_test_sweep_v2.py",
            "```",
            "",
            "The sweep is deterministic given the fixture descriptor + config",
            "seed=0. Re-running on the same torchvision cache + sentence-",
            "transformers model downloads produces byte-identical metrics.",
            "",
            "Supersedes Stage 2 v1's tautological report at",
            "`p4_mug_test_sweep.md`. The v1 report is kept for historical",
            "record of the Round 2 review catch; DO NOT act on its",
            "'defer Option 2' conclusion.",
        ]
    )

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(md_lines) + "\n")
    logger.info("wrote report %s", out_md)

    out_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "encoder_arm": encoder_arm,
        "total_wall_clock_s": round(total_wall_clock_s, 2),
        "text_ec_threshold": TEXT_EC_THRESHOLD,
        "vision_ec_threshold": VISION_EC_THRESHOLD,
        "sweep_points": [
            {
                "noise_reps": p.noise_reps,
                "bridge_enabled": p.bridge_enabled,
                "mean_forward_top5_recall": round(p.mean_forward_top5_recall, 4),
                "std_forward_top5_recall": round(p.std_forward_top5_recall, 4),
                "min_forward_top5_recall": round(p.min_forward_top5_recall, 4),
                "n_cross_class_pairs": p.n_cross_class_pairs,
                "n_single_hop_reachable": p.n_single_hop_reachable,
                "n_multi_hop_reachable": p.n_multi_hop_reachable,
                "option_2_lift_pct": round(p.option_2_lift_pct, 4),
                "n_class_episodes": p.n_class_episodes,
                "n_noise_episodes": p.n_noise_episodes,
                "n_bridge_episodes": p.n_bridge_episodes,
                "n_total_episodes": p.n_total_episodes,
                "n_vision_nodes": p.n_vision_nodes,
                "n_text_nodes": p.n_text_nodes,
                "wall_clock_s": round(p.wall_clock_s, 2),
            }
            for p in points
        ],
        "operating_point": (
            {
                "noise_reps": operating_point.noise_reps,
                "bridge_enabled": operating_point.bridge_enabled,
                "mean_forward_top5_recall": round(operating_point.mean_forward_top5_recall, 4),
                "option_2_lift_pct": round(operating_point.option_2_lift_pct, 4),
            }
            if operating_point is not None
            else None
        ),
    }
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")
    logger.info("wrote json %s", out_json)


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    repo_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(repo_root))

    from tests.substrate.p4_build_and_bind import (
        clip_vision_encoder,
        paraphrase_mpnet_text_encoder,
    )
    from tests.substrate.p4_fixture_loader import (
        load_fixture_descriptor,
        load_fixture_images,
    )

    logger.info("P4 Stage 2 v2 — mug test calibration sweep")
    descriptor = load_fixture_descriptor()
    images = load_fixture_images()
    logger.info(
        "loaded %d classes × %d samples = %d pairs",
        len(descriptor.classes),
        descriptor.samples_per_class,
        len(images),
    )

    # Load encoders ONCE across the sweep so we don't pay the
    # model-load cost 12 times. Each build_and_bind call gets the
    # same encoder instance via the TextEncoder/VisionEncoder
    # dataclass closure.
    logger.info("loading encoders once for the full sweep...")
    text_encoder = paraphrase_mpnet_text_encoder()
    vision_encoder = clip_vision_encoder()
    # Pre-encode all fixture images to numpy arrays once to save time
    # across the 12 combinations — but for now we encode per-combination
    # because build_and_bind wants the encoder callable, not pre-encoded
    # vectors. If the sweep takes too long we can optimize later.

    start = time.monotonic()
    points: list[SweepPoint] = []
    combo_idx = 0
    n_combos = len(NOISE_REPS_SWEEP) * len(BRIDGE_CONDITIONS)
    for bridge_enabled in BRIDGE_CONDITIONS:
        for noise_reps in NOISE_REPS_SWEEP:
            combo_idx += 1
            logger.info(
                "[%d/%d] noise_reps=%d bridges=%s",
                combo_idx,
                n_combos,
                noise_reps,
                "shared_superclass" if bridge_enabled else "none",
            )
            point = _run_one_combination(
                descriptor=descriptor,
                images=images,
                text_encoder=text_encoder,
                vision_encoder=vision_encoder,
                noise_reps=noise_reps,
                bridge_enabled=bridge_enabled,
            )
            logger.info(
                "  -> recall=%.3f lift=%+.1f%% (1hop=%d, multi=%d / %d pairs)",
                point.mean_forward_top5_recall,
                point.option_2_lift_pct,
                point.n_single_hop_reachable,
                point.n_multi_hop_reachable,
                point.n_cross_class_pairs,
            )
            points.append(point)

    total_wall = time.monotonic() - start
    logger.info("sweep done in %.1fs", total_wall)

    operating_point = _pick_operating_point(points)
    if operating_point is not None:
        logger.info(
            "operating point: noise_reps=%d, lift=%+.1f%%",
            operating_point.noise_reps,
            operating_point.option_2_lift_pct,
        )
    else:
        logger.warning("no operating point found (no bridge-enabled row ≥ 0.90 recall)")

    out_md = repo_root / "docs" / "experiments" / "p4_mug_test_sweep_v2.md"
    out_json = repo_root / "docs" / "experiments" / "results" / "p4_mug_test_sweep_v2.json"
    _write_report(
        out_md,
        out_json,
        points,
        operating_point,
        total_wall,
        encoder_arm=f"Arm A ({text_encoder.name} text + {vision_encoder.name} vision + hippocampus)",
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
