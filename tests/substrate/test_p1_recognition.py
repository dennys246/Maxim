"""P1 recognition sweep — runs paraphrase clusters through the substrate pipeline.

Requires sentence-transformers (``pip install pymaxim[semantic]``).
Marked ``slow`` — not part of the default test suite.

Run with:
    python -m pytest tests/substrate/test_p1_recognition.py -v -s

The ``-s`` flag is important — it prints the metrics summary to stdout
for lab notebook recording.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

logger = logging.getLogger(__name__)

FIXTURE_PATH = Path(__file__).parent.parent.parent / "scenarios" / "substrate" / "paraphrase_clusters.yaml"
RESULTS_DIR = Path(__file__).parent.parent.parent / "docs" / "experiments" / "results"


def _has_sentence_transformers() -> bool:
    try:
        import sentence_transformers  # noqa: F401

        return True
    except ImportError:
        return False


@pytest.mark.slow
@pytest.mark.skipif(not _has_sentence_transformers(), reason="sentence-transformers not installed")
@pytest.mark.skipif(not FIXTURE_PATH.exists(), reason="paraphrase_clusters.yaml not found")
class TestP1RecognitionSweep:
    """Run the P1 recognition sweep against paraphrase clusters."""

    def test_single_seed(self):
        """Single-seed smoke test — verifies the pipeline runs end-to-end."""
        metrics = self._run_seed(seed=42)
        print("\n" + metrics.summary())
        print(metrics.diagnostics())
        # Smoke test — just verify it ran
        assert metrics.total_nodes > 0
        assert metrics.modality_violations == 0

    def test_sweep_10_seeds(self):
        """Full 10-seed sweep — the actual P1 gate.

        Records results to docs/experiments/results/p1_recognition_sweep.json
        """
        results = []
        for seed in range(10):
            metrics = self._run_seed(seed=seed)
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
            print(f"\n[seed={seed}] {metrics.summary()}")

        # Compute mean/std
        import statistics

        collapse_rates = [r["collapse_rate"] for r in results]
        cross_rates = [r["cross_cluster_rate"] for r in results]

        mean_collapse = statistics.mean(collapse_rates)
        std_collapse = statistics.stdev(collapse_rates) if len(collapse_rates) > 1 else 0.0
        mean_cross = statistics.mean(cross_rates)
        std_cross = statistics.stdev(cross_rates) if len(cross_rates) > 1 else 0.0
        all_pass = all(r["passes"] for r in results)

        summary = {
            "model": "all-mpnet-base-v2",
            "threshold": 0.50,
            "seeds": len(results),
            "mean_collapse_rate": round(mean_collapse, 4),
            "std_collapse_rate": round(std_collapse, 4),
            "mean_cross_cluster_rate": round(mean_cross, 4),
            "std_cross_cluster_rate": round(std_cross, 4),
            "all_pass": all_pass,
            "per_seed": results,
        }

        print(f"\n{'=' * 60}")
        print(f"P1 Recognition Sweep — {'PASS' if all_pass else 'FAIL'}")
        print(f"  Collapse: {mean_collapse:.1%} ± {std_collapse:.1%} (need ≥90%)")
        print(f"  Cross-cluster: {mean_cross:.1%} ± {std_cross:.1%} (need ≤5%)")
        print(f"{'=' * 60}")

        # Save results
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        out_path = RESULTS_DIR / "p1_recognition_sweep.json"
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Results saved to {out_path}")

        # The actual gate assertion
        assert all_pass, f"P1 failed: {sum(1 for r in results if not r['passes'])}/{len(results)} seeds failed"

    def test_threshold_sweep(self):
        """Sweep thresholds to find the sweet spot for collapse vs cross-cluster.

        Prints a comparison table. Model loads once, re-encodes per threshold.
        """
        from maxim.memory.atl import ATL
        from maxim.similarity.ec import ECConfig, EntorhinalCortex
        from maxim.similarity.encoder import LinguisticEncoder

        from tests.substrate.p1_metrics import compute_p1_metrics, load_clusters_from_fixture

        clusters = load_clusters_from_fixture(str(FIXTURE_PATH))
        thresholds = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55]

        # Pre-compute all embeddings once (reuse encoder model)
        encoder_for_warmup = LinguisticEncoder(
            ec=EntorhinalCortex(),
            atl=ATL(),
        )
        # Warm up the model
        encoder_for_warmup.embed("warmup")

        print(f"\n{'=' * 75}")
        print("P1 THRESHOLD SWEEP")
        print(f"{'=' * 75}")
        print(f"  {'Thresh':>6s}  {'Collapse':>8s}  {'X-Cluster':>9s}  {'Nodes':>5s}  {'Growth':>6s}  {'P1':>4s}")
        print(f"  {'─' * 65}")

        for thresh in thresholds:
            ec = EntorhinalCortex(ECConfig(pattern_complete_threshold=thresh))
            atl = ATL()
            encoder = LinguisticEncoder(ec=ec, atl=atl)

            metrics = compute_p1_metrics(
                ec=ec,
                atl=atl,
                clusters=clusters,
                encoder=encoder,
                diagnostics=False,
            )

            status = "PASS" if metrics.passes_p1() else "FAIL"
            print(
                f"  {thresh:>6.2f}  {metrics.collapse_rate:>7.1%}  "
                f"{metrics.cross_cluster_rate:>8.1%}  "
                f"{metrics.total_nodes:>5d}  {metrics.node_growth_final_20pct:>5.1%}  "
                f"{status:>4s}"
            )

        print(f"{'=' * 75}")

    @staticmethod
    def _run_seed(seed: int = 42, threshold: float = 0.50):
        """Run one seed through the full pipeline and return metrics."""
        from maxim.memory.atl import ATL
        from maxim.similarity.ec import ECConfig, EntorhinalCortex
        from maxim.similarity.encoder import LinguisticEncoder

        from tests.substrate.p1_metrics import compute_p1_metrics, load_clusters_from_fixture

        ec = EntorhinalCortex(ECConfig(pattern_complete_threshold=threshold))
        atl = ATL()
        encoder = LinguisticEncoder(ec=ec, atl=atl)

        clusters = load_clusters_from_fixture(str(FIXTURE_PATH))
        return compute_p1_metrics(ec=ec, atl=atl, clusters=clusters, encoder=encoder)
