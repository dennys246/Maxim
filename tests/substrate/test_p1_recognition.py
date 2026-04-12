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

        Uses shuffled sentence order (node growth is an ordering artifact
        with sequential delivery — confirmed by test_shuffle_check).
        Records results to docs/experiments/results/p1_recognition_sweep.json
        """
        model = "paraphrase-mpnet-base-v2"
        threshold = 0.40

        results = []
        for seed in range(10):
            metrics = self._run_seed(seed=seed, threshold=threshold, shuffle=True)
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
        growth_rates = [r["node_growth_final_20pct"] for r in results]

        mean_collapse = statistics.mean(collapse_rates)
        std_collapse = statistics.stdev(collapse_rates) if len(collapse_rates) > 1 else 0.0
        mean_cross = statistics.mean(cross_rates)
        std_cross = statistics.stdev(cross_rates) if len(cross_rates) > 1 else 0.0
        mean_growth = statistics.mean(growth_rates)
        std_growth = statistics.stdev(growth_rates) if len(growth_rates) > 1 else 0.0
        all_pass = all(r["passes"] for r in results)

        summary = {
            "model": model,
            "threshold": threshold,
            "shuffle": True,
            "centroid_update": True,
            "seeds": len(results),
            "mean_collapse_rate": round(mean_collapse, 4),
            "std_collapse_rate": round(std_collapse, 4),
            "mean_cross_cluster_rate": round(mean_cross, 4),
            "std_cross_cluster_rate": round(std_cross, 4),
            "mean_node_growth": round(mean_growth, 4),
            "std_node_growth": round(std_growth, 4),
            "all_pass": all_pass,
            "per_seed": results,
        }

        print(f"\n{'=' * 60}")
        print(f"P1 Recognition Sweep — {'PASS' if all_pass else 'FAIL'}")
        print(f"  Model: {model} @ {threshold}")
        print(f"  Collapse: {mean_collapse:.1%} ± {std_collapse:.1%} (need ≥90%)")
        print(f"  Cross-cluster: {mean_cross:.1%} ± {std_cross:.1%} (need ≤5%)")
        print(f"  Node growth: {mean_growth:.1%} ± {std_growth:.1%} (need <10%)")
        print(f"{'=' * 60}")

        # Save results
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        out_path = RESULTS_DIR / "p1_recognition_sweep.json"
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Results saved to {out_path}")

        # The actual gate assertion
        assert all_pass, f"P1 failed: {sum(1 for r in results if not r['passes'])}/{len(results)} seeds failed"

    def test_shuffle_check(self):
        """Check if node growth improves with shuffled sentence order.

        Compares sequential vs shuffled at the best threshold (0.40)
        with paraphrase-mpnet-base-v2.
        """
        from maxim.memory.atl import ATL
        from maxim.similarity.ec import ECConfig, EntorhinalCortex
        from maxim.similarity.encoder import EncoderConfig, LinguisticEncoder

        from tests.substrate.p1_metrics import compute_p1_metrics, load_clusters_from_fixture

        clusters = load_clusters_from_fixture(str(FIXTURE_PATH))
        model = "paraphrase-mpnet-base-v2"
        thresh = 0.40

        print(f"\n{'=' * 75}")
        print(f"SHUFFLE CHECK — {model} @ {thresh}")
        print(f"{'=' * 75}")
        print(f"  {'Order':<15s}  {'Collapse':>8s}  {'X-Cluster':>9s}  {'Nodes':>5s}  {'Growth':>6s}  {'P1':>4s}")
        print(f"  {'─' * 60}")

        for label, do_shuffle, seed in [
            ("sequential", False, 0),
            ("shuffle s=0", True, 0),
            ("shuffle s=1", True, 1),
            ("shuffle s=2", True, 2),
            ("shuffle s=42", True, 42),
        ]:
            ec = EntorhinalCortex(ECConfig(pattern_complete_threshold=thresh))
            atl = ATL()
            encoder = LinguisticEncoder(ec=ec, atl=atl, config=EncoderConfig(model_name=model))

            metrics = compute_p1_metrics(
                ec=ec,
                atl=atl,
                clusters=clusters,
                encoder=encoder,
                diagnostics=False,
                shuffle=do_shuffle,
                seed=seed,
            )

            status = "PASS" if metrics.passes_p1() else "FAIL"
            print(
                f"  {label:<15s}  {metrics.collapse_rate:>7.1%}  "
                f"{metrics.cross_cluster_rate:>8.1%}  "
                f"{metrics.total_nodes:>5d}  {metrics.node_growth_final_20pct:>5.1%}  "
                f"{status:>4s}"
            )

        print(f"{'=' * 75}")

    def test_threshold_sweep(self):
        """Sweep thresholds to find the sweet spot for collapse vs cross-cluster.

        Prints a comparison table. Model loads once, re-encodes per threshold.
        """
        self._run_sweep(models=["all-mpnet-base-v2"])

    def test_model_comparison(self):
        """Compare embedding models across thresholds.

        Tests mpnet (general purpose) vs paraphrase-MiniLM (paraphrase-trained)
        to see if a model swap breaks past the 80% collapse ceiling.
        """
        self._run_sweep(
            models=[
                "all-mpnet-base-v2",
                "paraphrase-MiniLM-L6-v2",
                "paraphrase-mpnet-base-v2",
            ],
        )

    def _run_sweep(self, models: list[str] | None = None):
        """Run threshold sweep across one or more models."""
        from maxim.memory.atl import ATL
        from maxim.similarity.ec import ECConfig, EntorhinalCortex
        from maxim.similarity.encoder import EncoderConfig, LinguisticEncoder

        from tests.substrate.p1_metrics import compute_p1_metrics, load_clusters_from_fixture

        if models is None:
            models = ["all-mpnet-base-v2"]

        clusters = load_clusters_from_fixture(str(FIXTURE_PATH))
        thresholds = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]

        for model_name in models:
            # Warm up model
            warmup_encoder = LinguisticEncoder(
                ec=EntorhinalCortex(),
                atl=ATL(),
                config=EncoderConfig(model_name=model_name),
            )
            try:
                warmup_encoder.embed("warmup")
            except Exception as e:
                print(f"\n⚠ Skipping {model_name}: {e}")
                continue

            print(f"\n{'=' * 80}")
            print(f"  MODEL: {model_name}")
            print(f"{'=' * 80}")
            print(f"  {'Thresh':>6s}  {'Collapse':>8s}  {'X-Cluster':>9s}  {'Nodes':>5s}  {'Growth':>6s}  {'P1':>4s}")
            print(f"  {'─' * 65}")

            best_collapse = 0.0
            best_row = ""

            for thresh in thresholds:
                ec = EntorhinalCortex(ECConfig(pattern_complete_threshold=thresh))
                atl = ATL()
                encoder = LinguisticEncoder(
                    ec=ec,
                    atl=atl,
                    config=EncoderConfig(model_name=model_name),
                )

                metrics = compute_p1_metrics(
                    ec=ec,
                    atl=atl,
                    clusters=clusters,
                    encoder=encoder,
                    diagnostics=False,
                )

                status = "PASS" if metrics.passes_p1() else "FAIL"
                marker = " ◀" if metrics.passes_p1() else ""
                row = (
                    f"  {thresh:>6.2f}  {metrics.collapse_rate:>7.1%}  "
                    f"{metrics.cross_cluster_rate:>8.1%}  "
                    f"{metrics.total_nodes:>5d}  {metrics.node_growth_final_20pct:>5.1%}  "
                    f"{status:>4s}{marker}"
                )
                print(row)

                if metrics.collapse_rate > best_collapse:
                    best_collapse = metrics.collapse_rate
                    best_row = f"thresh={thresh} collapse={metrics.collapse_rate:.1%} x-cluster={metrics.cross_cluster_rate:.1%}"

            print(f"  Best: {best_row}")
            print(f"{'=' * 80}")

    @staticmethod
    def _run_seed(seed: int = 42, threshold: float = 0.40, shuffle: bool = False):
        """Run one seed through the full pipeline and return metrics."""
        from maxim.memory.atl import ATL
        from maxim.similarity.ec import ECConfig, EntorhinalCortex
        from maxim.similarity.encoder import LinguisticEncoder

        from tests.substrate.p1_metrics import compute_p1_metrics, load_clusters_from_fixture

        ec = EntorhinalCortex(ECConfig(pattern_complete_threshold=threshold))
        atl = ATL()
        encoder = LinguisticEncoder(ec=ec, atl=atl)

        clusters = load_clusters_from_fixture(str(FIXTURE_PATH))
        return compute_p1_metrics(
            ec=ec,
            atl=atl,
            clusters=clusters,
            encoder=encoder,
            shuffle=shuffle,
            seed=seed,
        )
