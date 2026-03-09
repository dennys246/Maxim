"""Unit tests for SimilarityIndex (LSH-based) and percept_to_canonical.

Tests the core functionality of MinHash + LSH indexing, similarity queries,
persistence, and percept canonicalization.
"""

from __future__ import annotations

import pytest


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def index():
    """Fresh SimilarityIndex."""
    from maxim.memory.context_index import SimilarityIndex

    return SimilarityIndex()


# ── SimilarityIndex Registration ──────────────────────────────────────────────


class TestSimilarityIndexRegistration:
    """Test memory registration and indexing."""

    def test_register_adds_to_signatures(self, index):
        index.register("mem_1", "the quick brown fox jumps over the lazy dog")
        assert "mem_1" in index.signatures
        assert len(index) == 1

    def test_register_ignores_short_text(self, index):
        index.register("mem_1", "hi")
        assert len(index) == 0

    def test_register_ignores_empty_text(self, index):
        index.register("mem_1", "")
        assert len(index) == 0

    def test_register_multiple_memories(self, index):
        index.register("mem_1", "the quick brown fox jumps over the lazy dog")
        index.register("mem_2", "a fast brown fox leaps over a sleeping dog")
        assert len(index) == 2

    def test_remove_clears_signature_and_bands(self, index):
        index.register("mem_1", "the quick brown fox jumps over the lazy dog")
        index.remove("mem_1")
        assert len(index) == 0
        assert "mem_1" not in index.signatures

    def test_remove_nonexistent_is_safe(self, index):
        index.remove("nonexistent")  # Should not raise


# ── SimilarityIndex Queries ──────────────────────────────────────────────────


class TestSimilarityIndexQueries:
    """Test similarity queries."""

    def test_identical_text_returns_high_similarity(self, index):
        text = "the robot detected a person standing near the table"
        index.register("mem_1", text)
        results = index.query_similar(text, min_similarity=0.5)
        assert len(results) >= 1
        assert results[0][0] == "mem_1"
        assert results[0][1] >= 0.9

    def test_similar_text_found(self, index):
        # Use nearly identical text to ensure LSH band collision
        index.register("mem_1", "the robot saw a person near the table in the kitchen")
        results = index.query_similar(
            "the robot saw a person near the table in the room", min_similarity=0.2
        )
        result_ids = {r[0] for r in results}
        assert "mem_1" in result_ids

    def test_dissimilar_text_excluded(self, index):
        index.register("mem_1", "the robot saw a person near the table in the kitchen")
        results = index.query_similar(
            "quantum physics and mathematical topology research papers",
            min_similarity=0.5,
        )
        result_ids = {r[0] for r in results}
        assert "mem_1" not in result_ids

    def test_results_sorted_by_similarity_descending(self, index):
        text = "the robot is detecting objects in the room"
        index.register("mem_1", text)
        index.register("mem_2", "the robot detects items in the room area")

        results = index.query_similar(text, min_similarity=0.1)
        if len(results) >= 2:
            assert results[0][1] >= results[1][1]

    def test_query_empty_index_returns_empty(self, index):
        results = index.query_similar("some text here for testing")
        assert results == []


# ── Shingling ─────────────────────────────────────────────────────────────────


class TestShingling:
    """Test the shingling helper."""

    def test_shingle_short_text(self, index):
        shingles = index._shingle("hello")
        assert len(shingles) == 1

    def test_shingle_normal_text(self, index):
        shingles = index._shingle("the quick brown fox")
        assert "the quick" in shingles
        assert "quick brown" in shingles
        assert "brown fox" in shingles

    def test_shingle_empty(self, index):
        shingles = index._shingle("")
        assert shingles == set()


# ── MinHash ───────────────────────────────────────────────────────────────────


class TestMinHash:
    """Test MinHash signature computation."""

    def test_minhash_length_matches_num_hashes(self, index):
        shingles = {"hello world", "world is", "is great"}
        sig = index._minhash(shingles)
        assert len(sig) == index.num_hashes

    def test_minhash_empty_shingles(self, index):
        sig = index._minhash(set())
        assert len(sig) == index.num_hashes
        assert all(h == 0 for h in sig)

    def test_identical_shingles_give_identical_signatures(self, index):
        shingles = {"hello world", "world is", "is great"}
        sig1 = index._minhash(shingles)
        sig2 = index._minhash(shingles)
        assert sig1 == sig2


# ── Persistence ───────────────────────────────────────────────────────────────


class TestSimilarityIndexPersistence:
    """Test save/load roundtrip."""

    def test_save_load_preserves_data(self, index, tmp_path):
        from maxim.memory.context_index import SimilarityIndex

        index.register("mem_1", "the robot saw a person near the table today")
        index.register("mem_2", "another memory about detecting objects nearby")

        path = str(tmp_path / "index.json")
        index.save(path)

        loaded = SimilarityIndex.load(path)
        assert len(loaded) == 2
        assert "mem_1" in loaded.signatures
        assert "mem_2" in loaded.signatures

    def test_save_load_preserves_query_results(self, index, tmp_path):
        from maxim.memory.context_index import SimilarityIndex

        text = "the robot detected a person standing near the table"
        index.register("mem_1", text)

        path = str(tmp_path / "index.json")
        index.save(path)
        loaded = SimilarityIndex.load(path)

        results = loaded.query_similar(text, min_similarity=0.5)
        assert len(results) >= 1
        assert results[0][0] == "mem_1"


# ── percept_to_canonical ─────────────────────────────────────────────────────


class TestPerceptToCanonical:
    """Test percept dict to canonical string conversion."""

    def test_objects_included(self):
        from maxim.memory.context_index import percept_to_canonical

        percept = {
            "objects": [
                {"label": "mug", "confidence": 0.95},
                {"label": "table", "confidence": 0.88},
            ]
        }
        result = percept_to_canonical(percept)
        assert "mug 1.0" in result or "mug 0.9" in result
        assert "table" in result

    def test_people_included(self):
        from maxim.memory.context_index import percept_to_canonical

        percept = {
            "people": [{"label": "person_1", "posture": "standing"}],
        }
        result = percept_to_canonical(percept)
        assert "person_1" in result
        assert "standing" in result

    def test_speech_included(self):
        from maxim.memory.context_index import percept_to_canonical

        percept = {"speech": "hello there"}
        result = percept_to_canonical(percept)
        assert "hello there" in result

    def test_empty_percept_returns_empty(self):
        from maxim.memory.context_index import percept_to_canonical

        assert percept_to_canonical({}) == ""

    def test_spatial_fields_omitted(self):
        from maxim.memory.context_index import percept_to_canonical

        percept = {
            "objects": [
                {"label": "mug", "confidence": 0.9, "distance": 1.5, "bearing": 45},
            ]
        }
        result = percept_to_canonical(percept)
        assert "distance" not in result
        assert "bearing" not in result

    def test_missing_label_skipped(self):
        from maxim.memory.context_index import percept_to_canonical

        percept = {"objects": [{"confidence": 0.9}]}
        assert percept_to_canonical(percept) == ""
