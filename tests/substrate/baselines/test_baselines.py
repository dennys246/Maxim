"""Tests for P0 baselines — fixture loading + baseline mechanics."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.substrate.baselines.embedding_baseline import (
    BaselineResult,
    NEAR_MISS_PAIRS,
    load_fixture_sentences,
    run_embedding_baseline,
)


FIXTURE_PATH = Path("scenarios/substrate/paraphrase_clusters.yaml")


class TestFixtureLoading:
    def test_fixture_exists(self):
        assert FIXTURE_PATH.exists(), f"Fixture not found: {FIXTURE_PATH}"

    def test_loads_sentences(self):
        sentences = load_fixture_sentences(FIXTURE_PATH)
        assert len(sentences) > 100, f"Expected >100 sentences, got {len(sentences)}"

    def test_at_least_50_clusters(self):
        sentences = load_fixture_sentences(FIXTURE_PATH)
        clusters = set(s["cluster"] for s in sentences)
        assert len(clusters) >= 50, f"Expected >=50 clusters, got {len(clusters)}"

    def test_has_difficulty_tiers(self):
        sentences = load_fixture_sentences(FIXTURE_PATH)
        difficulties = set(s["difficulty"] for s in sentences)
        assert difficulties == {"easy", "medium", "hard"}

    def test_has_train_holdout_split(self):
        sentences = load_fixture_sentences(FIXTURE_PATH)
        splits = set(s["split"] for s in sentences)
        assert splits == {"train", "holdout"}

    def test_train_holdout_ratio(self):
        """Approximately 60/40 split."""
        sentences = load_fixture_sentences(FIXTURE_PATH)
        train = sum(1 for s in sentences if s["split"] == "train")
        holdout = sum(1 for s in sentences if s["split"] == "holdout")
        ratio = train / (train + holdout)
        assert 0.50 <= ratio <= 0.70, f"Train ratio {ratio:.2f} not in [0.50, 0.70]"

    def test_every_cluster_has_at_least_2(self):
        sentences = load_fixture_sentences(FIXTURE_PATH)
        from collections import Counter

        counts = Counter(s["cluster"] for s in sentences)
        small = {k: v for k, v in counts.items() if v < 2}
        assert not small, f"Clusters with <2 sentences: {small}"

    def test_near_miss_pairs_exist_in_fixture(self):
        sentences = load_fixture_sentences(FIXTURE_PATH)
        clusters = set(s["cluster"] for s in sentences)
        for c1, c2 in NEAR_MISS_PAIRS:
            assert c1 in clusters, f"Near-miss cluster '{c1}' not found in fixture"
            assert c2 in clusters, f"Near-miss cluster '{c2}' not found in fixture"


class TestEmbeddingBaseline:
    def test_returns_result_without_deps(self):
        """Even without sentence-transformers, returns a BaselineResult (with error detail)."""
        try:
            import sentence_transformers  # noqa: F401

            pytest.skip("sentence-transformers is installed — test targets missing-dep path")
        except ImportError:
            pass

        result = run_embedding_baseline(FIXTURE_PATH, seed=42)
        assert isinstance(result, BaselineResult)
        assert result.details.get("error")

    @pytest.mark.slow
    def test_full_baseline_run(self):
        """Full baseline with model — slow, requires sentence-transformers."""
        pytest.importorskip("sentence_transformers")
        result = run_embedding_baseline(
            FIXTURE_PATH,
            model_name="all-MiniLM-L6-v2",
            threshold=0.75,
            seed=42,
        )
        assert result.num_sentences > 100
        assert result.num_ground_truth_clusters >= 50
        assert 0.0 <= result.collapse_rate <= 1.0
        assert 0.0 <= result.cross_cluster_rate <= 1.0
        assert 0.0 <= result.near_miss_separation <= 1.0
        # Sanity: easy tier should be easier than hard tier
        assert result.easy_collapse >= result.hard_collapse

    @pytest.mark.slow
    def test_deterministic_across_seeds(self):
        """Same seed → same result."""
        pytest.importorskip("sentence_transformers")
        r1 = run_embedding_baseline(FIXTURE_PATH, seed=42)
        r2 = run_embedding_baseline(FIXTURE_PATH, seed=42)
        assert r1.collapse_rate == r2.collapse_rate
        assert r1.cross_cluster_rate == r2.cross_cluster_rate

    @pytest.mark.slow
    def test_different_seeds_same_result(self):
        """Embedding baseline is deterministic regardless of seed (no randomness in pipeline)."""
        pytest.importorskip("sentence_transformers")
        r1 = run_embedding_baseline(FIXTURE_PATH, seed=42)
        r2 = run_embedding_baseline(FIXTURE_PATH, seed=99)
        # The embedding model is deterministic — only the seed for numpy matters
        # and there's no randomness in the greedy clustering
        assert r1.collapse_rate == r2.collapse_rate


class TestOpenCLIPBaseline:
    def test_returns_result_without_deps(self):
        """Even without open_clip, returns a BaselineResult."""
        try:
            import open_clip  # noqa: F401

            pytest.skip("open_clip installed — test targets missing-dep path")
        except ImportError:
            pass

        from tests.substrate.baselines.openclip_baseline import run_openclip_baseline

        result = run_openclip_baseline(FIXTURE_PATH, seed=42)
        assert isinstance(result, BaselineResult)
        assert result.details.get("error")
