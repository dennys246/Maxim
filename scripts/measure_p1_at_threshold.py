#!/usr/bin/env python3
"""Re-measure P1 paraphrase-collapse sweep at a configurable EC threshold.

Mirrors tests/substrate/test_p1_recognition.py::TestP1RecognitionSweep::test_sweep_10_seeds
but parameterizes the threshold so Phase 2 of docs/plans/archive/ec_centroid_drift_fix.md
can measure the regression delta at the Phase 1 winning threshold (0.50) vs
the pinned baseline at 0.40.

Usage:

    PYTHONPATH=src python scripts/measure_p1_at_threshold.py --threshold 0.50 \\
        --output /tmp/p1_at_0_50.json

The script runs the same 10-seed shuffled sweep with paraphrase-mpnet-base-v2
that the P1 gate test does. Output JSON matches the schema of
docs/experiments/results/p1_recognition_sweep.json so direct diff is trivial.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--model", default="paraphrase-mpnet-base-v2")
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--frozen-text",
        action="store_true",
        help="Add 'text' to ECConfig.frozen_centroid_modalities (Phase 1 candidate "
        "d0_f1_*). Default is current behavior — only 'interoception' frozen.",
    )
    args = parser.parse_args()

    # Import after sys.path manipulation. The test file lives outside the
    # package but imports tests.substrate.p1_metrics — add the repo root too.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    from maxim.memory.atl import ATL
    from maxim.similarity.ec import ECConfig, EntorhinalCortex
    from maxim.similarity.encoder import EncoderConfig, LinguisticEncoder

    from tests.substrate.p1_metrics import compute_p1_metrics, load_clusters_from_fixture

    fixture_path = Path(__file__).resolve().parent.parent / "scenarios" / "substrate" / "paraphrase_clusters.yaml"
    if not fixture_path.exists():
        print(f"ERROR: P1 fixture not found at {fixture_path}", file=sys.stderr)
        return 2

    clusters = load_clusters_from_fixture(str(fixture_path))

    fcm = frozenset({"interoception", "text"}) if args.frozen_text else frozenset({"interoception"})
    results: list[dict] = []
    for seed in range(args.seeds):
        ec = EntorhinalCortex(ECConfig(pattern_complete_threshold=args.threshold, frozen_centroid_modalities=fcm))
        atl = ATL()
        encoder = LinguisticEncoder(ec=ec, atl=atl, config=EncoderConfig(model_name=args.model))
        metrics = compute_p1_metrics(
            ec=ec,
            atl=atl,
            clusters=clusters,
            encoder=encoder,
            shuffle=True,
            seed=seed,
        )
        results.append(
            {
                "seed": seed,
                "collapse_rate": metrics.collapse_rate,
                "cross_cluster_rate": metrics.cross_cluster_rate,
                "node_growth_final_20pct": metrics.node_growth_final_20pct,
                "total_nodes": metrics.total_nodes,
                "modality_violations": metrics.modality_violations,
                "passes": metrics.passes_p1(),
            }
        )
        print(
            f"[seed={seed}] collapse={metrics.collapse_rate:.1%} cross={metrics.cross_cluster_rate:.1%} "
            f"growth={metrics.node_growth_final_20pct:.1%} nodes={metrics.total_nodes}",
            file=sys.stderr,
        )

    collapse = [r["collapse_rate"] for r in results]
    cross = [r["cross_cluster_rate"] for r in results]
    growth = [r["node_growth_final_20pct"] for r in results]

    mean_c = statistics.mean(collapse)
    std_c = statistics.stdev(collapse) if len(collapse) > 1 else 0.0
    mean_x = statistics.mean(cross)
    std_x = statistics.stdev(cross) if len(cross) > 1 else 0.0
    mean_g = statistics.mean(growth)
    std_g = statistics.stdev(growth) if len(growth) > 1 else 0.0

    # Standard P1 gate (production test): collapse >= 0.90, cross <= 0.05, growth < 0.10
    means_pass = mean_c >= 0.90 and mean_x <= 0.05 and mean_g < 0.10
    # Phase 2 regression tolerance (looser): collapse >= 0.85
    phase2_pass = mean_c >= 0.85

    summary = {
        "model": args.model,
        "threshold": args.threshold,
        "shuffle": True,
        "centroid_update": not args.frozen_text,
        "frozen_centroid_modalities": sorted(fcm),
        "seeds": len(results),
        "seeds_passing_standard_gate": sum(1 for r in results if r["passes"]),
        "mean_collapse_rate": round(mean_c, 4),
        "std_collapse_rate": round(std_c, 4),
        "mean_cross_cluster_rate": round(mean_x, 4),
        "std_cross_cluster_rate": round(std_x, 4),
        "mean_node_growth": round(mean_g, 4),
        "std_node_growth": round(std_g, 4),
        "means_pass_standard_gate": means_pass,
        "means_pass_phase2_regression_tolerance": phase2_pass,
        "per_seed": results,
    }

    print()
    label = f"threshold {args.threshold}" + (" + frozen-text" if args.frozen_text else "")
    print(f"P1 sweep @ {label}")
    print(
        f"  collapse: {mean_c:.1%} ± {std_c:.1%}  (P1 gate ≥90%: {'PASS' if mean_c >= 0.90 else 'fail'};  Phase 2 ≥85%: {'PASS' if phase2_pass else 'FAIL'})"
    )
    print(f"  cross-cluster: {mean_x:.1%} ± {std_x:.1%}  (gate ≤5%: {'PASS' if mean_x <= 0.05 else 'fail'})")
    print(f"  node growth: {mean_g:.1%} ± {std_g:.1%}  (gate <10%: {'PASS' if mean_g < 0.10 else 'fail'})")
    print(f"  seeds passing individual P1 gate: {summary['seeds_passing_standard_gate']}/{len(results)}")

    if args.output:
        out_path = Path(args.output).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2))
        print(f"Wrote {out_path}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
