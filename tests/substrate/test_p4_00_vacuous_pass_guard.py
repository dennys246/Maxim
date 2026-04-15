"""P4 Stage 1.5 — vacuous-pass guard (RUN-FIRST).

This file is named ``test_p4_00_*`` so pytest's alphabetical collection
order runs it before any other ``test_p4_*`` file. The guard is the
precondition that makes every Stage 1 mechanism test interpretable:
if the synthetic cluster-aware fixture does NOT actually cluster
paired same-modality samples to one EC node id (and does NOT keep
cross-pair samples in distinct ids), then the binding tests below
would still produce GREEN results on a stub mechanism — Round 1
review's "vacuous pass" risk.

The guard tests the fixture against a real ``EntorhinalCortex``
instance with default config (``pattern_complete_threshold = 0.40``).
It does NOT depend on any encoder (sentence-transformers, CLIP,
LinguisticEncoder) — embeddings are pure numpy from
``p4_fixture_gen.make_cluster_aware_pairs``. This keeps Stage 1
encoder-agnostic in line with the P4 three-arm methodology
constraint: Stage 1's API surface must not bake in any text encoder
identity.
"""

from __future__ import annotations

import numpy as np
import pytest

from maxim.similarity.ec import EntorhinalCortex

from .p4_fixture_gen import (
    cosine_similarity,
    make_cluster_aware_pairs,
)


# ─────────────────────────────────────────────────────────────────────────
# Fixture math sanity — the embeddings themselves must satisfy the
# cluster-separability property BEFORE we hand them to EC. Catches
# fixture regressions independently of EC behavior.
# ─────────────────────────────────────────────────────────────────────────


class TestFixtureGeometry:
    def test_within_pair_text_text_above_ec_threshold(self) -> None:
        """Paired text-text cosine similarity must clear EC's 0.40 with
        substantial headroom — the safety margin is what lets the
        EC-side guard test below produce a deterministic pass."""
        pairs = make_cluster_aware_pairs(5, dim=64, k_text=3, k_vision=3, noise_scale=0.4, seed=42)
        within = [
            cosine_similarity(p.text_samples[i], p.text_samples[j])
            for p in pairs
            for i in range(len(p.text_samples))
            for j in range(i + 1, len(p.text_samples))
        ]
        assert min(within) > 0.60, (
            f"within-pair text-text similarity floor too low: min={min(within):.3f}, "
            f"need > 0.60 for safe EC clustering above the 0.40 threshold"
        )

    def test_within_pair_vision_vision_above_ec_threshold(self) -> None:
        """Same as text-text but for vision modality."""
        pairs = make_cluster_aware_pairs(5, dim=64, k_text=3, k_vision=3, noise_scale=0.4, seed=42)
        within = [
            cosine_similarity(p.vision_samples[i], p.vision_samples[j])
            for p in pairs
            for i in range(len(p.vision_samples))
            for j in range(i + 1, len(p.vision_samples))
        ]
        assert min(within) > 0.60, f"within-pair vision-vision similarity floor too low: min={min(within):.3f}"

    def test_cross_pair_below_ec_threshold(self) -> None:
        """Cross-pair same-modality similarity must stay below EC's
        0.40 — orthogonal centroids guarantee this. If two distinct
        pairs accidentally share a centroid direction, two unrelated
        objects would silently merge into one cluster and the binding
        tests below would silently mis-attribute partners."""
        pairs = make_cluster_aware_pairs(5, dim=64, k_text=3, k_vision=3, noise_scale=0.4, seed=42)
        cross = [
            cosine_similarity(pairs[i].text_samples[0], pairs[j].text_samples[0])
            for i in range(len(pairs))
            for j in range(i + 1, len(pairs))
        ]
        assert max(cross) < 0.30, (
            f"cross-pair text-text similarity ceiling too high: max={max(cross):.3f}, "
            f"need < 0.30 to stay safely below EC's 0.40 threshold"
        )

    def test_orthogonal_centroids_are_actually_orthogonal(self) -> None:
        """QR-based centroid generation must produce mutually
        orthogonal unit vectors. Validates the fixture generator
        itself, not the surrounding noise."""
        pairs = make_cluster_aware_pairs(5, dim=64, k_text=1, k_vision=1, noise_scale=0.0, seed=42)
        for i in range(len(pairs)):
            assert abs(float(np.linalg.norm(pairs[i].centroid)) - 1.0) < 1e-9
            for j in range(i + 1, len(pairs)):
                inner = float(np.dot(pairs[i].centroid, pairs[j].centroid))
                assert abs(inner) < 1e-9, f"centroids {i} and {j} not orthogonal: <c_i,c_j>={inner:.6f}"

    @pytest.mark.parametrize("dim", [64, 128, 384, 512, 768])
    def test_within_pair_similarity_is_dim_invariant(self, dim: int) -> None:
        """Round 2 Arch-lens fold: with noise_scale (not bare noise_std),
        within-pair expected cosine similarity depends ONLY on
        noise_scale, not on dim. The default noise_scale=0.4 should
        produce within-pair sim ≈ 1 / (1 + 0.16) ≈ 0.862 at every
        reasonable embedding dim. This test verifies the dim-invariance
        property explicitly so a future Stage 2 caller picking CLIP's
        512-d space does not silently regress the fixture's EC-
        clustering guarantee."""
        pairs = make_cluster_aware_pairs(3, dim=dim, k_text=3, k_vision=3, noise_scale=0.4, seed=42)
        within = [
            cosine_similarity(p.text_samples[i], p.text_samples[j])
            for p in pairs
            for i in range(len(p.text_samples))
            for j in range(i + 1, len(p.text_samples))
        ]
        mean_within = float(np.mean(within))
        # Expected ≈ 0.862; accept [0.75, 0.95] band to absorb finite-
        # sample variance without letting a regression slip through.
        assert 0.75 < mean_within < 0.95, (
            f"dim={dim}: within-pair mean similarity {mean_within:.3f} "
            f"outside the dim-invariant band [0.75, 0.95] — the 1/sqrt(dim) "
            f"noise scaling may have regressed"
        )
        assert min(within) > 0.60, (
            f"dim={dim}: within-pair min {min(within):.3f} ≤ 0.60 would cluster unsafely at EC's 0.40 threshold"
        )


# ─────────────────────────────────────────────────────────────────────────
# EC clustering — the Stage 1.5 vacuous-pass guard proper. Hands the
# fixture's embeddings to a real EntorhinalCortex and asserts the
# pattern_complete_or_separate behavior matches the assumption every
# Stage 1 mechanism test below depends on.
# ─────────────────────────────────────────────────────────────────────────


class TestECClusteringVacuousPassGuard:
    @pytest.fixture
    def ec(self) -> EntorhinalCortex:
        return EntorhinalCortex()

    def test_paired_text_samples_collapse_to_one_node(self, ec: EntorhinalCortex) -> None:
        """All text samples for one pair must share the same EC node id.
        Run the first sample through pattern_complete_or_separate (which
        will create a new node), register it, then run the remaining
        samples and assert each is_new=False with the same node_id."""
        pairs = make_cluster_aware_pairs(5, dim=64, k_text=3, k_vision=3, noise_scale=0.4, seed=42)

        for pair in pairs:
            first = ec.pattern_complete_or_separate(pair.text_samples[0].tolist(), modality="text")
            assert first.is_new is True
            ec.register_substrate_node(first.node_id, pair.text_samples[0].tolist(), "text")

            for sample in pair.text_samples[1:]:
                result = ec.pattern_complete_or_separate(sample.tolist(), modality="text")
                assert result.is_new is False, (
                    f"pair {pair.pair_index}: text sample landed in a NEW node "
                    f"(similarity={result.similarity:.3f}) when it should have completed "
                    f"to the existing pair-text node {first.node_id}"
                )
                assert result.node_id == first.node_id, (
                    f"pair {pair.pair_index}: text sample completed to {result.node_id} "
                    f"instead of pair's first node {first.node_id}"
                )

    def test_paired_vision_samples_collapse_to_one_node(self, ec: EntorhinalCortex) -> None:
        """Symmetric to the text test — all vision samples for one pair
        share one node id."""
        pairs = make_cluster_aware_pairs(5, dim=64, k_text=3, k_vision=3, noise_scale=0.4, seed=42)

        for pair in pairs:
            first = ec.pattern_complete_or_separate(pair.vision_samples[0].tolist(), modality="vision")
            assert first.is_new is True
            ec.register_substrate_node(first.node_id, pair.vision_samples[0].tolist(), "vision")

            for sample in pair.vision_samples[1:]:
                result = ec.pattern_complete_or_separate(sample.tolist(), modality="vision")
                assert result.is_new is False, f"pair {pair.pair_index}: vision sample failed to complete to first node"
                assert result.node_id == first.node_id

    def test_cross_pair_samples_land_in_distinct_nodes(self, ec: EntorhinalCortex) -> None:
        """Different pairs must end up in different EC node ids (no
        accidental cluster collisions). Register the first text sample
        of each pair, then re-query and assert each first sample maps
        to ITS own node, never another pair's."""
        pairs = make_cluster_aware_pairs(5, dim=64, k_text=3, k_vision=3, noise_scale=0.4, seed=42)

        registered_ids: list[str] = []
        for pair in pairs:
            result = ec.pattern_complete_or_separate(pair.text_samples[0].tolist(), modality="text")
            assert result.is_new is True
            ec.register_substrate_node(result.node_id, pair.text_samples[0].tolist(), "text")
            registered_ids.append(result.node_id)

        assert len(set(registered_ids)) == len(pairs), (
            "registered ids collided: cluster centroids were not distinct enough"
        )

        for idx, pair in enumerate(pairs):
            result = ec.pattern_complete_or_separate(pair.text_samples[1].tolist(), modality="text")
            assert result.node_id == registered_ids[idx], (
                f"pair {idx} second text sample mapped to {result.node_id}, expected {registered_ids[idx]}"
            )

    def test_text_and_vision_for_same_pair_get_distinct_nodes(self, ec: EntorhinalCortex) -> None:
        """EC filters by modality — even though text and vision samples
        of the same pair share a centroid (by construction), they MUST
        land in different EC node ids because EC's
        pattern_complete_or_separate restricts the search to the same
        modality bucket. This is the precondition for the cross-modal
        binding mechanism: two distinct nodes that co-activate in one
        episode and get bound via Hebbian close, NOT one merged node."""
        pairs = make_cluster_aware_pairs(3, dim=64, k_text=2, k_vision=2, noise_scale=0.4, seed=42)

        for pair in pairs:
            text_result = ec.pattern_complete_or_separate(pair.text_samples[0].tolist(), modality="text")
            ec.register_substrate_node(text_result.node_id, pair.text_samples[0].tolist(), "text")

            vision_result = ec.pattern_complete_or_separate(pair.vision_samples[0].tolist(), modality="vision")
            assert vision_result.is_new is True, (
                f"pair {pair.pair_index}: vision sample completed to an existing node "
                f"({vision_result.node_id}) instead of separating — EC's modality bucket "
                f"filter is broken"
            )
            assert vision_result.node_id != text_result.node_id, (
                f"pair {pair.pair_index}: text and vision landed in the same node id despite different modalities"
            )
            ec.register_substrate_node(vision_result.node_id, pair.vision_samples[0].tolist(), "vision")
