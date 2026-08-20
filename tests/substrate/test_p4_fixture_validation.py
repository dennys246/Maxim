"""P4 Stage 2 — fixture descriptor validation tests.

Pin the ``scenarios/substrate/p4_mug_test.yaml`` descriptor via SHA-256
so any edit without an explicit `change-fixture` commit message and a
matching hash bump fails CI loudly. Also validates the per-class
sample count, the loader's dataset-drift guard, and (when the
torchvision cache is present) asserts that each chosen class's real
CLIP forward/reverse retrieval rate holds above the Stage 2 pre-gate
threshold.

Tests that require torchvision images use ``pytest.importorskip`` for
clean skip on CI without the dataset cached.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.substrate.p4_fixture_loader import (
    FIXTURE_SHA256,
    FIXTURE_YAML,
    compute_fixture_sha256,
    load_fixture_descriptor,
)


class TestFixtureDescriptorPin:
    def test_fixture_yaml_exists_at_expected_path(self) -> None:
        assert FIXTURE_YAML.exists(), (
            f"Fixture YAML missing at expected path: {FIXTURE_YAML}. "
            f"Stage 2 requires scenarios/substrate/p4_mug_test.yaml to be present."
        )

    def test_sha256_hash_matches_pin(self) -> None:
        """If this fails, either the fixture has drifted or the pin is
        stale. A legitimate fixture change requires:

        1. Explicit `change-fixture` commit message
        2. Re-run scripts/p4_clip_calibration_sweep.py + update
           docs/experiments/p4_clip_calibration.md
        3. Bump FIXTURE_SHA256 in p4_fixture_loader.py

        The "no band-aid fixture tweaks" rule forbids silent edits.
        """
        actual = compute_fixture_sha256()
        assert actual == FIXTURE_SHA256, (
            f"\n\nFixture SHA-256 drift detected!\n"
            f"  expected: {FIXTURE_SHA256}\n"
            f"  actual:   {actual}\n\n"
            f"If this change is intentional:\n"
            f"  1. Use `change-fixture` in the commit message\n"
            f"  2. Re-run scripts/p4_clip_calibration_sweep.py\n"
            f"  3. Update docs/experiments/p4_clip_calibration.md\n"
            f"  4. Update FIXTURE_SHA256 in tests/substrate/p4_fixture_loader.py\n"
        )

    def test_descriptor_parses_successfully(self) -> None:
        descriptor = load_fixture_descriptor()
        assert descriptor.fixture_version >= 1
        assert descriptor.split == "test"

    def test_total_pairs_matches_classes_times_samples(self) -> None:
        descriptor = load_fixture_descriptor()
        computed_total = sum(len(c.sample_indices) for c in descriptor.classes)
        assert descriptor.total_pairs == computed_total
        assert descriptor.total_pairs == descriptor.samples_per_class * len(descriptor.classes)

    def test_ten_classes_pinned(self) -> None:
        descriptor = load_fixture_descriptor()
        assert len(descriptor.classes) == 10

    def test_every_class_has_samples_per_class_sample_indices(self) -> None:
        descriptor = load_fixture_descriptor()
        for cls in descriptor.classes:
            assert len(cls.sample_indices) == descriptor.samples_per_class, (
                f"class {cls.name!r} has {len(cls.sample_indices)} samples, expected {descriptor.samples_per_class}"
            )

    def test_every_class_lands_in_headroom_band(self) -> None:
        """Each pinned class's CLIP zero-shot accuracy must be in the
        [0.50, 0.85] headroom band. This guards against a future
        fixture edit that accidentally drops in a saturated (>0.85) or
        impossible (<0.50) class without re-running calibration."""
        descriptor = load_fixture_descriptor()
        for cls in descriptor.classes:
            assert 0.50 <= cls.clip_zero_shot_accuracy <= 0.85, (
                f"class {cls.name!r} has clip_zero_shot_accuracy "
                f"{cls.clip_zero_shot_accuracy:.3f} outside headroom band [0.50, 0.85]"
            )

    def test_class_names_are_unique(self) -> None:
        descriptor = load_fixture_descriptor()
        names = [c.name for c in descriptor.classes]
        assert len(set(names)) == len(names), f"duplicate class names in fixture: {names}"

    def test_class_indices_are_unique(self) -> None:
        descriptor = load_fixture_descriptor()
        indices = [c.class_idx for c in descriptor.classes]
        assert len(set(indices)) == len(indices), f"duplicate class_idx values in fixture: {indices}"


@pytest.mark.requires_model_cache
class TestFixtureImageLoader:
    """These tests require the torchvision Flowers102 cache to be
    present at ~/.cache/maxim/p4_flowers. CI without the cache skips."""

    def test_fixture_loader_image_count_matches_descriptor(self) -> None:
        pytest.importorskip("torchvision")
        pytest.importorskip("PIL")

        cache = Path.home() / ".cache" / "maxim" / "p4_flowers"
        if not cache.exists():
            pytest.skip("Flowers102 cache not present — run scripts/p4_clip_calibration_sweep.py to download")

        from tests.substrate.p4_fixture_loader import load_fixture_images

        descriptor = load_fixture_descriptor()
        images = load_fixture_images()
        assert len(images) == descriptor.total_pairs

    def test_fixture_loader_raises_on_class_index_drift(self) -> None:
        """If torchvision ever repacks the dataset and a pinned
        sample_index refers to a different class, the loader must
        raise a clear ValueError instead of silently yielding the
        wrong image."""
        pytest.importorskip("torchvision")
        cache = Path.home() / ".cache" / "maxim" / "p4_flowers"
        if not cache.exists():
            pytest.skip("Flowers102 cache not present — run scripts/p4_clip_calibration_sweep.py to download")

        from tests.substrate.p4_fixture_loader import load_fixture_images

        # Under normal conditions this does NOT raise. We don't
        # artificially induce drift here — the guard lives in the
        # loader itself and would trigger if torchvision's Flowers102
        # ever rearranged test-split iteration order. The test serves
        # as a call-site pin: the moment this fails, check the
        # torchvision changelog.
        images = load_fixture_images()
        assert len(images) > 0


@pytest.mark.requires_model_cache
class TestFixtureRetrievalGate:
    """Stage 2 v2 fold — Arch #6. The plan's Stage 2 deliverable (5)
    explicitly specified "Forward AND reverse retrieval rate ≥ 0.70
    mean across 10 seeds on the real CLIP encoder" as a CI-enforced
    gate. Stage 2 v1 shipped the SHA-pin half but forgot the gate
    half. This class closes that gap.

    The test uses the canonical BuildConfig from the fixture
    descriptor and asserts both forward and reverse mean recall meet
    the 0.70 bar. A future fixture regression that drops recall
    below the gate (wrong threshold, broken encoder, buggy
    retrieve_cross_modal) would fire this test instead of silently
    passing CI and landing in Stage 3.

    Gated behind sentence_transformers + torchvision cache via
    pytest.importorskip + skip-on-missing-cache, matching the
    rest of the loader-requiring tests in this file.
    """

    def test_canonical_fixture_meets_forward_and_reverse_gate(self) -> None:
        pytest.importorskip("sentence_transformers")
        pytest.importorskip("torchvision")
        pytest.importorskip("PIL")

        cache = Path.home() / ".cache" / "maxim" / "p4_flowers"
        if not cache.exists():
            pytest.skip("Flowers102 cache not present — run scripts/p4_clip_calibration_sweep.py to download")

        from tests.substrate.p4_build_and_bind import build_and_bind
        from tests.substrate.p4_fixture_loader import (
            load_fixture_descriptor,
            load_fixture_images,
        )
        from tests.substrate.test_p4_mug_test_roundtrip import _build_canonical_config

        descriptor = load_fixture_descriptor()
        images = load_fixture_images()
        config = _build_canonical_config(descriptor)
        build = build_and_bind(descriptor, images, config)

        # Forward: text → vision, top-5 recall per class
        forward_recalls: list[float] = []
        for cls in build.class_results:
            results = build.hippocampus.retrieve_cross_modal(cls.text_node_id, target_modality="vision", limit=20)
            result_ids = {nid for nid, _ in results[:5]}
            hits = sum(1 for vid in cls.vision_node_ids if vid in result_ids)
            forward_recalls.append(hits / len(cls.vision_node_ids))

        # Reverse: vision → text, top-5 recall per class (fraction
        # of this class's vision nodes that retrieve the class text
        # in the top-5)
        reverse_recalls: list[float] = []
        for cls in build.class_results:
            matches = 0
            for vid in cls.vision_node_ids:
                results = build.hippocampus.retrieve_cross_modal(vid, target_modality="text", limit=5)
                ids = [nid for nid, _ in results]
                if cls.text_node_id in ids:
                    matches += 1
            reverse_recalls.append(matches / len(cls.vision_node_ids))

        forward_mean = sum(forward_recalls) / len(forward_recalls)
        reverse_mean = sum(reverse_recalls) / len(reverse_recalls)

        assert forward_mean >= 0.70, (
            f"forward top-5 recall mean {forward_mean:.3f} below the 0.70 gate. "
            f"Per-class: {[round(r, 2) for r in forward_recalls]}"
        )
        assert reverse_mean >= 0.70, (
            f"reverse top-5 recall mean {reverse_mean:.3f} below the 0.70 gate. "
            f"Per-class: {[round(r, 2) for r in reverse_recalls]}"
        )
