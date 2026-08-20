"""P1 recognition sweep — runs paraphrase clusters through the substrate pipeline.

Requires sentence-transformers (``pip install 'pymaxim[semantic]'``).
Marked ``slow`` — not part of the default test suite.

Run with:
    python -m pytest tests/substrate/test_p1_recognition.py -v -s

The ``-s`` flag is important — it prints the metrics summary to stdout
for lab notebook recording.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

logger = logging.getLogger(__name__)

FIXTURE_PATH = Path(__file__).parent.parent.parent / "scenarios" / "substrate" / "paraphrase_clusters.yaml"
# Sweep output no longer lands here directly — see the publish_sweep_results
# fixture in tests/substrate/conftest.py (bugs ledger D24).


def _has_sentence_transformers() -> bool:
    try:
        import sentence_transformers  # noqa: F401

        return True
    except ImportError:
        return False


@pytest.mark.slow
# Needs the real encoder WEIGHTS, not just the package: offline, the encoder
# falls back and the assertions below report a fake substrate REGRESSION
# ('beats random by -0.9%') instead of a missing model. Installed-vs-cached
# is the D20 distinction; skip cleanly rather than publish a false result.
@pytest.mark.requires_model_cache
@pytest.mark.skipif(not _has_sentence_transformers(), reason="sentence-transformers not installed")
@pytest.mark.skipif(not FIXTURE_PATH.exists(), reason="paraphrase_clusters.yaml not found")
class TestP1RecognitionSweep:
    """Run the P1 recognition sweep against paraphrase clusters."""

    @pytest.fixture(autouse=True)
    def _require_real_encoder(self) -> None:
        """Refuse to measure on the hash fallback (bugs ledger D25).

        Without this the sweep still produces numbers, and they read as a
        substrate regression rather than a missing model.
        """
        from maxim.similarity.encoder import require_semantic_encoder

        require_semantic_encoder(context="P1 recognition sweep")

    def test_single_seed(self):
        """Single-seed smoke test — verifies the pipeline runs end-to-end."""
        metrics = self._run_seed(seed=42)
        print("\n" + metrics.summary())
        print(metrics.diagnostics())
        # Smoke test — just verify it ran
        assert metrics.total_nodes > 0
        assert metrics.modality_violations == 0

    def test_sweep_10_seeds(self, publish_sweep_results):
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
        # Gate on means (plan says "Mean + std across ≥10 seeds")
        means_pass = mean_collapse >= 0.90 and mean_cross <= 0.05 and mean_growth < 0.10
        seeds_passing = sum(1 for r in results if r["passes"])

        summary = {
            "model": model,
            "threshold": threshold,
            "shuffle": True,
            "centroid_update": True,
            "seeds": len(results),
            "seeds_passing": seeds_passing,
            "mean_collapse_rate": round(mean_collapse, 4),
            "std_collapse_rate": round(std_collapse, 4),
            "mean_cross_cluster_rate": round(mean_cross, 4),
            "std_cross_cluster_rate": round(std_cross, 4),
            "mean_node_growth": round(mean_growth, 4),
            "std_node_growth": round(std_growth, 4),
            "means_pass": means_pass,
            "per_seed": results,
        }

        print(f"\n{'=' * 60}")
        print(f"P1 Recognition Sweep — {'PASS' if means_pass else 'FAIL'}")
        print(f"  Model: {model} @ {threshold}")
        print(f"  Collapse: {mean_collapse:.1%} ± {std_collapse:.1%} (need ≥90%)")
        print(f"  Cross-cluster: {mean_cross:.1%} ± {std_cross:.1%} (need ≤5%)")
        print(f"  Node growth: {mean_growth:.1%} ± {std_growth:.1%} (need <10%)")
        print(f"  Seeds passing individually: {seeds_passing}/{len(results)}")
        print(f"{'=' * 60}")

        # Save results (D24: tmp by default; --write-experiment-results to
        # update the committed S4 record deliberately).
        publish_sweep_results("p1_recognition_sweep.json", summary)

        # Gate on means, not individual seeds — per plan spec
        assert means_pass, (
            f"P1 means failed: collapse={mean_collapse:.1%} cross={mean_cross:.1%} growth={mean_growth:.1%}"
        )

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

    def test_degenerate_control(self):
        """Verify substrate beats random node assignment by >30pp.

        Random baseline: each sentence gets assigned to a random node
        (uniform over num_clusters). Expected collapse ≈ 1/num_clusters
        for within-cluster pairs ≈ 1.8% for 55 clusters.
        """
        import random

        from tests.substrate.p1_metrics import load_clusters_from_fixture

        clusters = load_clusters_from_fixture(str(FIXTURE_PATH))

        # --- Random baseline ---
        rng = random.Random(42)
        num_clusters = len(clusters)
        random_nodes: dict[str, str] = {}  # sentence → random node_id

        for cluster in clusters:
            for sentence in cluster["sentences"]:
                random_nodes[sentence] = f"rand-{rng.randint(0, num_clusters - 1)}"

        # Compute random collapse rate
        within_total = 0
        within_collapsed = 0
        for cluster in clusters:
            sents = cluster["sentences"]
            for i in range(len(sents)):
                for j in range(i + 1, len(sents)):
                    within_total += 1
                    if random_nodes[sents[i]] == random_nodes[sents[j]]:
                        within_collapsed += 1

        random_collapse = within_collapsed / within_total if within_total else 0.0

        # --- Substrate result ---
        metrics = self._run_seed(seed=42, shuffle=True)

        gap = metrics.collapse_rate - random_collapse

        print(f"\n{'=' * 60}")
        print("DEGENERATE CONTROL")
        print(f"  Random baseline: {random_collapse:.1%}")
        print(f"  Substrate:       {metrics.collapse_rate:.1%}")
        print(f"  Gap:             {gap:.1%} (need >30pp)")
        print(f"{'=' * 60}")

        assert gap > 0.30, f"Substrate only beats random by {gap:.1%}, need >30pp"

    def test_persistence_round_trip(self, tmp_path):
        """Verify ≥95% of node activations are preserved after save/load.

        Encodes all sentences, saves EC+ATL, loads into fresh instances,
        re-encodes the same sentences, checks that ≥95% map to the same node.
        """
        from maxim.agents.bus import Percept
        from maxim.memory.atl import ATL, ATLConfig
        from maxim.similarity.ec import ECConfig, EntorhinalCortex
        from maxim.similarity.encoder import LinguisticEncoder

        from tests.substrate.p1_metrics import load_clusters_from_fixture

        clusters = load_clusters_from_fixture(str(FIXTURE_PATH))
        all_sentences = []
        for cluster in clusters:
            for sentence in cluster["sentences"]:
                all_sentences.append(sentence)

        # --- First pass: encode everything ---
        ec_path = str(tmp_path / "ec.json")
        atl_path = str(tmp_path / "atl.json")

        ec = EntorhinalCortex(ECConfig(pattern_complete_threshold=0.40))
        atl = ATL(ATLConfig(persistence_path=atl_path))
        encoder = LinguisticEncoder(ec=ec, atl=atl)

        first_pass: dict[str, str] = {}  # sentence → node_id
        for sentence in all_sentences:
            percept = Percept(timestamp=0.0, source="test", transcript_chunk=sentence)
            node_id = encoder.encode(percept)
            first_pass[sentence] = node_id

        # Save
        ec.save(ec_path)
        atl.save()

        # --- Second pass: load and re-encode ---
        ec2 = EntorhinalCortex(ECConfig(pattern_complete_threshold=0.40))
        ec2.load(ec_path)
        atl2 = ATL(ATLConfig(persistence_path=atl_path))
        atl2.load()
        encoder2 = LinguisticEncoder(ec=ec2, atl=atl2)

        matches = 0
        total = 0
        mismatches = []
        for sentence in all_sentences:
            percept = Percept(timestamp=0.0, source="test", transcript_chunk=sentence)
            node_id = encoder2.encode(percept)
            total += 1
            if node_id == first_pass[sentence]:
                matches += 1
            else:
                mismatches.append((sentence[:50], first_pass[sentence][:8], node_id[:8]))

        preservation_rate = matches / total if total else 0.0

        print(f"\n{'=' * 60}")
        print("PERSISTENCE ROUND-TRIP")
        print(f"  Sentences:     {total}")
        print(f"  Preserved:     {matches} ({preservation_rate:.1%})")
        print(f"  Mismatches:    {total - matches}")
        if mismatches[:5]:
            print("  Sample mismatches:")
            for sent, old, new in mismatches[:5]:
                print(f'    "{sent}" {old} → {new}')
        print(f"{'=' * 60}")

        assert preservation_rate >= 0.95, f"Only {preservation_rate:.1%} preserved, need ≥95%"

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
