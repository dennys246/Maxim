"""P4 Stage 1 — synthetic cluster-aware embedding fixture (in-test, no on-disk files, no encoder dependency).

Stage 1 tests the cross-modal binding mechanism — auto-tagging at
episode close, the per-node modality sidecar, and the snapshot-pattern
filter in ``Hippocampus.retrieve_cross_modal``. The mechanism is
agnostic to which encoder produced the embeddings; Stage 2 swaps in
``sentence-transformers`` ``clip-ViT-B-32``. Stage 1 must therefore not
bake any encoder identity into the fixture.

This module produces purely synthetic embeddings via numpy with
deterministic seeding. The shape is "cluster-aware paired centroids":

- N pair centroids, each a unit vector. Centroids are mutually
  near-orthogonal (random gaussians in high dimension are
  approximately orthogonal; exact orthogonality is enforced via QR
  decomposition for fully deterministic test behavior at any
  embedding dimension).
- Each pair has K_text text samples and K_vision vision samples, all
  drawn from ``centroid + small_noise`` where ``small_noise`` is
  scaled so that within-pair cosine similarity stays well above the
  EC ``pattern_complete_threshold`` (0.40) and between-pair cosine
  similarity stays below it.

This guarantees:

1. Paired same-modality samples collapse to one EC node id (Stage 1.5
   vacuous-pass guard precondition).
2. Cross-pair same-modality samples land in distinct EC nodes (no
   accidental cluster collisions).
3. Text and vision samples for the same pair are NOT in the same EC
   bucket (EC filters by modality at lookup), so the pair's text node
   and vision node are distinct ids — exactly the cross-modal binding
   case P4 needs.

Why not random gaussians?
=========================

Random gaussians (no shared centroid) would produce essentially
unique embeddings per sample. Every text sample would land in its own
EC node; the binding episode would activate dozens of distinct text
nodes; cross-modal retrieval would have nothing meaningful to bind
to its vision partner. The Stage 1 mechanism tests would still PASS
(retrieve_cross_modal would return SOMETHING, even if the something
is meaningless), and Stage 2 would discover the mechanism is broken
when real CLIP embeddings hit it. The cluster-aware fixture catches
this class of bug at Stage 1 — Round 1 review's "vacuous-pass risk"
fold.

This is a Stage 1-only fixture. Stage 2 replaces it with real CLIP
encodings of Oxford Flowers-102 images.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PairFixture:
    """A single (text, vision) cluster-aware pair.

    Attributes:
        pair_index: 0-based index of this pair in the fixture.
        centroid: Unit-norm centroid the (text, vision) samples are
            drawn around. Shared between modalities by convention so
            Stage 2's real CLIP embeddings have a clean analogy
            (text "mug" and image of mug land near each other in
            shared CLIP space).
        text_samples: K_text noisy embeddings around the centroid.
        vision_samples: K_vision noisy embeddings around the centroid.
    """

    pair_index: int
    centroid: np.ndarray
    text_samples: list[np.ndarray]
    vision_samples: list[np.ndarray]


def generate_orthogonal_centroids(n_pairs: int, dim: int, seed: int) -> list[np.ndarray]:
    """Return ``n_pairs`` unit-norm centroids that are mutually
    orthogonal (cosine similarity == 0 between distinct centroids).

    Uses QR decomposition of a random gaussian matrix. For
    ``n_pairs <= dim`` (always true in Stage 1 fixtures) the first
    ``n_pairs`` columns of Q form an orthonormal basis subset.
    """
    if n_pairs > dim:
        raise ValueError(f"cannot generate {n_pairs} mutually orthogonal centroids in {dim}-d space")
    rng = np.random.default_rng(seed)
    random_matrix = rng.standard_normal(size=(dim, n_pairs))
    q, _ = np.linalg.qr(random_matrix)
    return [q[:, i].copy() for i in range(n_pairs)]


def make_cluster_aware_pairs(
    n_pairs: int,
    *,
    dim: int = 64,
    k_text: int = 2,
    k_vision: int = 2,
    noise_scale: float = 0.4,
    seed: int = 0,
) -> list[PairFixture]:
    """Build ``n_pairs`` cluster-aware paired fixtures.

    ``noise_scale`` is **dim-invariant** — the per-sample noise
    standard deviation is ``noise_scale / sqrt(dim)`` so the expected
    within-pair cosine similarity
    ``≈ 1 / (1 + noise_scale^2) ≈ 0.862`` at the default
    ``noise_scale=0.4`` is INDEPENDENT of the embedding dimension.
    This is the Round 2 Arch-lens fold: an earlier draft used a bare
    ``noise_std`` parameter that scaled with ``dim`` — a Stage 2
    caller picking CLIP's native 512-d space with the old default
    would produce within-pair similarity near 0.44, right at EC's
    0.40 threshold, silently breaking the cluster-aware guarantee.
    The dim-invariant form lets any reasonable ``dim`` (64, 128, 384,
    512, 768) produce the same within-pair similarity band so future
    callers do not have to re-tune the parameter.

    Between-pair similarity is ~0 at any dim because centroids are
    exactly orthogonal via QR decomposition.
    """
    centroids = generate_orthogonal_centroids(n_pairs, dim, seed)
    rng = np.random.default_rng(seed + 1)
    # Scale per-element noise so the total noise vector's squared norm
    # is E[||noise||^2] = dim * (noise_scale^2 / dim) = noise_scale^2,
    # independent of dim. Within-pair expected cosine similarity is
    # ||centroid||^2 / (||centroid||^2 + noise_scale^2) =
    # 1 / (1 + noise_scale^2).
    per_element_std = noise_scale / float(np.sqrt(dim))
    pairs: list[PairFixture] = []
    for idx, centroid in enumerate(centroids):
        text_samples = [centroid + per_element_std * rng.standard_normal(dim) for _ in range(k_text)]
        vision_samples = [centroid + per_element_std * rng.standard_normal(dim) for _ in range(k_vision)]
        pairs.append(
            PairFixture(
                pair_index=idx,
                centroid=centroid,
                text_samples=text_samples,
                vision_samples=vision_samples,
            )
        )
    return pairs


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Plain cosine similarity. Helper for the Stage 1.5 vacuous-pass
    guard so the test does not depend on any production similarity
    helper that might mask a fixture issue under the hood.

    Does NOT guard zero-norm inputs — numpy's ``0/0`` produces ``nan``
    which surfaces as a loud test failure, which is what we want from
    a fixture-debugging helper. Round 2 Exec-lens fold: the old
    defensive ``return 0.0`` silently masked degenerate embedding
    inputs.
    """
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
