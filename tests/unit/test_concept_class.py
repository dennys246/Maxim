"""Unit tests for the Concept class and ATL concept memory integration.

Tests cover:
- Concept class fields and methods (add_ref, remove_ref, ref_count)
- Memory_refs bounded by MAX_REFS_PER_LAYER with FIFO pruning
- Serialization round-trip (to_dict / from_dict with set↔list)
- ATL load() dispatch on _concept flag
- ATL find_or_create() creates Concept instances
- Backward compatibility with existing SemanticMemory records
- Memory access patterns: get(), recall(), recall_by_ids(), recall_associated()
"""

from __future__ import annotations

import json
import time

import pytest

from maxim.memory.atl import ATL, ATLConfig
from maxim.memory.semantic_types import (
    CompressedSemantic,
    Concept,
    ConceptProvenance,
    SemanticMemory,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def atl():
    """Fresh ATL instance with no persistence."""
    return ATL(ATLConfig(persistence_path=None))


@pytest.fixture
def concept():
    """A simple Concept instance."""
    return Concept(
        id="concept-1",
        timestamp=time.time(),
        name="mug",
        category="object",
        definition="A drinking vessel",
        confidence=0.5,
    )


@pytest.fixture
def atl_with_persistence(tmp_path):
    """ATL instance with persistence to a temp file."""
    path = str(tmp_path / "atl_state.json")
    return ATL(ATLConfig(persistence_path=path))


# ─────────────────────────────────────────────────────────────────────────────
# Concept class basics
# ─────────────────────────────────────────────────────────────────────────────


class TestConceptClass:
    """Test Concept extends SemanticMemory correctly."""

    def test_concept_is_semantic_memory(self, concept):
        """Concept inherits all SemanticMemory fields."""
        assert isinstance(concept, SemanticMemory)
        assert concept.name == "mug"
        assert concept.category == "object"
        assert concept.definition == "A drinking vessel"
        assert concept.confidence == 0.5

    def test_concept_has_empty_memory_refs(self, concept):
        """New concept starts with empty memory_refs."""
        assert concept.ref_count() == 0
        assert concept.ref_count("hippocampus") == 0

    def test_concept_keywords(self, concept):
        """keywords() returns name and category (inherited)."""
        kws = concept.keywords()
        assert "mug" in kws
        assert "object" in kws


# ─────────────────────────────────────────────────────────────────────────────
# Memory refs: add, remove, count
# ─────────────────────────────────────────────────────────────────────────────


class TestMemoryRefs:
    """Test cross-layer reference tracking."""

    def test_add_ref_tracks_memory_id(self, concept):
        """add_ref registers a memory ID under the given layer."""
        concept.add_ref("hippocampus", "mem-1")
        assert concept.ref_count("hippocampus") == 1
        assert concept.ref_count() == 1

    def test_add_ref_multiple_layers(self, concept):
        """Refs tracked independently per layer."""
        concept.add_ref("hippocampus", "mem-1")
        concept.add_ref("hippocampus", "mem-2")
        concept.add_ref("angular_gyrus", "math-1")

        assert concept.ref_count("hippocampus") == 2
        assert concept.ref_count("angular_gyrus") == 1
        assert concept.ref_count() == 3

    def test_add_ref_deduplicates(self, concept):
        """Adding the same ref twice doesn't duplicate."""
        concept.add_ref("hippocampus", "mem-1")
        concept.add_ref("hippocampus", "mem-1")
        assert concept.ref_count("hippocampus") == 1

    def test_remove_ref(self, concept):
        """remove_ref removes a tracked memory ID."""
        concept.add_ref("hippocampus", "mem-1")
        concept.add_ref("hippocampus", "mem-2")

        concept.remove_ref("hippocampus", "mem-1")
        assert concept.ref_count("hippocampus") == 1

    def test_remove_ref_missing_is_noop(self, concept):
        """Removing a ref that doesn't exist is safe."""
        concept.remove_ref("hippocampus", "nonexistent")
        assert concept.ref_count("hippocampus") == 0

    def test_remove_ref_missing_layer_is_noop(self, concept):
        """Removing a ref from a nonexistent layer is safe."""
        concept.remove_ref("nonexistent_layer", "mem-1")

    def test_add_ref_updates_access_tracking(self, concept):
        """add_ref calls touch(), updating access_count and accessed_at."""
        initial_count = concept.access_count
        initial_access = concept.accessed_at

        time.sleep(0.01)
        concept.add_ref("hippocampus", "mem-1")

        assert concept.access_count > initial_count
        assert concept.accessed_at >= initial_access

    def test_ref_count_none_returns_total(self, concept):
        """ref_count(None) returns total across all layers."""
        concept.add_ref("hippocampus", "mem-1")
        concept.add_ref("hippocampus", "mem-2")
        concept.add_ref("angular_gyrus", "math-1")
        assert concept.ref_count(None) == 3
        assert concept.ref_count() == 3


# ─────────────────────────────────────────────────────────────────────────────
# Bounded memory_refs with FIFO pruning
# ─────────────────────────────────────────────────────────────────────────────


class TestMemoryRefsBounded:
    """Test that memory_refs are bounded per layer."""

    def test_refs_capped_at_max(self):
        """Refs per layer cannot exceed MAX_REFS_PER_LAYER."""
        concept = Concept(
            id="c1",
            timestamp=time.time(),
            name="test",
            category="object",
        )
        # Use a small cap for testing
        concept.MAX_REFS_PER_LAYER = 5

        for i in range(10):
            concept.add_ref("hippocampus", f"mem-{i}")

        assert concept.ref_count("hippocampus") == 5

    def test_fifo_pruning_removes_oldest(self):
        """When cap is exceeded, the oldest (first inserted) ref is pruned."""
        concept = Concept(
            id="c1",
            timestamp=time.time(),
            name="test",
            category="object",
        )
        concept.MAX_REFS_PER_LAYER = 3

        concept.add_ref("hippocampus", "mem-A")
        concept.add_ref("hippocampus", "mem-B")
        concept.add_ref("hippocampus", "mem-C")
        # At cap, now adding one more should evict mem-A
        concept.add_ref("hippocampus", "mem-D")

        refs = concept.memory_refs["hippocampus"]
        assert "mem-A" not in refs
        assert "mem-B" in refs
        assert "mem-C" in refs
        assert "mem-D" in refs
        assert len(refs) == 3

    def test_fifo_pruning_preserves_order(self):
        """After pruning, iteration order is still FIFO."""
        concept = Concept(
            id="c1",
            timestamp=time.time(),
            name="test",
            category="object",
        )
        concept.MAX_REFS_PER_LAYER = 3

        concept.add_ref("hippocampus", "mem-1")
        concept.add_ref("hippocampus", "mem-2")
        concept.add_ref("hippocampus", "mem-3")
        concept.add_ref("hippocampus", "mem-4")
        concept.add_ref("hippocampus", "mem-5")

        refs = list(concept.memory_refs["hippocampus"].keys())
        assert refs == ["mem-3", "mem-4", "mem-5"]

    def test_pruning_per_layer_independent(self):
        """Each layer has its own cap — pruning one doesn't affect others."""
        concept = Concept(
            id="c1",
            timestamp=time.time(),
            name="test",
            category="object",
        )
        concept.MAX_REFS_PER_LAYER = 2

        concept.add_ref("hippocampus", "h1")
        concept.add_ref("hippocampus", "h2")
        concept.add_ref("hippocampus", "h3")  # Evicts h1

        concept.add_ref("angular_gyrus", "a1")
        concept.add_ref("angular_gyrus", "a2")

        assert concept.ref_count("hippocampus") == 2
        assert concept.ref_count("angular_gyrus") == 2
        assert "h1" not in concept.memory_refs["hippocampus"]
        assert "a1" in concept.memory_refs["angular_gyrus"]


# ─────────────────────────────────────────────────────────────────────────────
# Serialization round-trip
# ─────────────────────────────────────────────────────────────────────────────


class TestConceptSerialization:
    """Test to_dict / from_dict round-trip."""

    def test_to_dict_has_concept_flag(self, concept):
        """to_dict() includes _concept: True flag."""
        data = concept.to_dict()
        assert data["_concept"] is True

    def test_to_dict_memory_refs_as_sorted_lists(self, concept):
        """to_dict() converts dict keys to sorted lists."""
        concept.add_ref("hippocampus", "mem-c")
        concept.add_ref("hippocampus", "mem-a")
        concept.add_ref("hippocampus", "mem-b")

        data = concept.to_dict()
        assert data["memory_refs"]["hippocampus"] == ["mem-a", "mem-b", "mem-c"]

    def test_to_dict_skips_empty_layers(self, concept):
        """to_dict() doesn't include empty layer refs."""
        concept.add_ref("hippocampus", "mem-1")
        data = concept.to_dict()
        assert "angular_gyrus" not in data.get("memory_refs", {})

    def test_from_dict_restores_concept(self, concept):
        """from_dict() restores a Concept with correct fields."""
        concept.add_ref("hippocampus", "mem-1")
        concept.add_ref("hippocampus", "mem-2")
        concept.add_ref("angular_gyrus", "math-1")

        data = concept.to_dict()
        restored = Concept.from_dict(data)

        assert restored.id == concept.id
        assert restored.name == "mug"
        assert restored.category == "object"
        assert restored.definition == "A drinking vessel"
        assert restored.confidence == concept.confidence
        assert restored.ref_count("hippocampus") == 2
        assert restored.ref_count("angular_gyrus") == 1

    def test_round_trip_preserves_provenance(self):
        """Provenance enum survives serialization."""
        c = Concept(
            id="c1",
            timestamp=time.time(),
            name="test",
            category="fact",
            provenance=ConceptProvenance.AGENT_INFERENCE,
        )
        data = c.to_dict()
        restored = Concept.from_dict(data)
        assert restored.provenance == ConceptProvenance.AGENT_INFERENCE

    def test_round_trip_through_json(self, concept):
        """Full JSON round-trip works."""
        concept.add_ref("hippocampus", "mem-1")

        json_str = json.dumps(concept.to_dict())
        data = json.loads(json_str)
        restored = Concept.from_dict(data)

        assert restored.name == "mug"
        assert restored.ref_count("hippocampus") == 1

    def test_from_dict_handles_missing_memory_refs(self):
        """from_dict() handles data without memory_refs (backward compat)."""
        data = {
            "id": "old-concept",
            "timestamp": time.time(),
            "name": "chair",
            "category": "object",
            "_concept": True,
        }
        concept = Concept.from_dict(data)
        assert concept.ref_count() == 0


# ─────────────────────────────────────────────────────────────────────────────
# ATL integration: load/save dispatch
# ─────────────────────────────────────────────────────────────────────────────


class TestATLConceptDispatch:
    """Test ATL correctly dispatches Concept, SemanticMemory, CompressedSemantic."""

    def test_find_or_create_creates_concept(self, atl):
        """find_or_create() creates Concept instances (not plain SemanticMemory)."""
        cid, created = atl.find_or_create("mug", "object")
        assert created is True

        concept = atl.get(cid)
        assert isinstance(concept, Concept)
        assert concept.name == "mug"
        assert concept.category == "object"

    def test_find_or_create_reinforces_existing(self, atl):
        """Second find_or_create for same name/category reinforces."""
        cid1, created1 = atl.find_or_create("mug", "object", source_episode_id="ep-1")
        cid2, created2 = atl.find_or_create("mug", "object", source_episode_id="ep-2")

        assert cid1 == cid2
        assert created1 is True
        assert created2 is False

        concept = atl.get(cid1)
        assert concept.reinforcement_count == 2

    def test_load_dispatches_concept_flag(self, atl_with_persistence):
        """ATL load() deserializes _concept-flagged records as Concept."""
        atl = atl_with_persistence
        cid, _ = atl.find_or_create("chair", "object")

        # Add a ref so we can verify it survives
        concept = atl.get(cid)
        concept.add_ref("hippocampus", "mem-42")

        atl.save()

        # Load into fresh ATL
        atl2 = ATL(ATLConfig(persistence_path=atl.config.persistence_path))
        atl2.load()

        loaded = atl2.get(cid)
        assert isinstance(loaded, Concept)
        assert loaded.name == "chair"
        assert loaded.ref_count("hippocampus") == 1

    def test_load_dispatches_compressed_flag(self, atl_with_persistence):
        """ATL load() still handles _compressed records correctly."""
        atl = atl_with_persistence

        # Manually store a CompressedSemantic
        compressed = CompressedSemantic(
            id="comp-1",
            timestamp=time.time(),
            name="old_pattern",
            category="causal_pattern",
            confidence=0.3,
        )
        atl.store(compressed)
        atl.save()

        atl2 = ATL(ATLConfig(persistence_path=atl.config.persistence_path))
        atl2.load()

        loaded = atl2.get("comp-1")
        assert isinstance(loaded, CompressedSemantic)
        assert loaded.name == "old_pattern"

    def test_load_dispatches_plain_semantic(self, atl_with_persistence):
        """ATL load() handles plain SemanticMemory (no _concept flag)."""
        atl = atl_with_persistence

        # Manually store a plain SemanticMemory (legacy path)
        sm = SemanticMemory(
            id="legacy-1",
            timestamp=time.time(),
            name="old_concept",
            category="fact",
        )
        atl.store(sm)
        atl.save()

        atl2 = ATL(ATLConfig(persistence_path=atl.config.persistence_path))
        atl2.load()

        loaded = atl2.get("legacy-1")
        assert isinstance(loaded, SemanticMemory)
        assert not isinstance(loaded, Concept)

    def test_mixed_types_survive_save_load(self, atl_with_persistence):
        """All three types coexist after save/load cycle."""
        atl = atl_with_persistence

        # Concept via find_or_create
        cid, _ = atl.find_or_create("mug", "object")
        concept = atl.get(cid)
        concept.add_ref("hippocampus", "ep-1")

        # Plain SemanticMemory (legacy)
        sm = SemanticMemory(
            id="sm-1",
            timestamp=time.time(),
            name="legacy",
            category="fact",
        )
        atl.store(sm)

        # CompressedSemantic
        comp = CompressedSemantic(
            id="comp-1",
            timestamp=time.time(),
            name="compressed",
            category="pattern",
        )
        atl.store(comp)

        atl.save()

        atl2 = ATL(ATLConfig(persistence_path=atl.config.persistence_path))
        atl2.load()

        assert isinstance(atl2.get(cid), Concept)
        assert isinstance(atl2.get("sm-1"), SemanticMemory)
        assert not isinstance(atl2.get("sm-1"), Concept)
        assert isinstance(atl2.get("comp-1"), CompressedSemantic)
        assert len(atl2) == 3


# ─────────────────────────────────────────────────────────────────────────────
# ATL memory access patterns
# ─────────────────────────────────────────────────────────────────────────────


class TestATLMemoryAccess:
    """Test memory access patterns: get, recall, recall_by_ids, recall_associated."""

    def test_get_touches_concept(self, atl):
        """get() updates access tracking on the concept."""
        cid, _ = atl.find_or_create("mug", "object")
        concept = atl.get(cid)
        initial_count = concept.access_count

        # Second get should increment access_count
        concept2 = atl.get(cid)
        assert concept2.access_count > initial_count

    def test_get_returns_none_for_missing(self, atl):
        """get() returns None for nonexistent IDs."""
        assert atl.get("nonexistent") is None

    def test_recall_by_name(self, atl):
        """recall() with name filter finds concepts."""
        atl.find_or_create("mug", "object")
        atl.find_or_create("chair", "object")

        results = atl.recall(name="mug")
        assert len(results) == 1
        assert results[0].name == "mug"

    def test_recall_by_category(self, atl):
        """recall() with category filter finds matching concepts."""
        atl.find_or_create("mug", "object")
        atl.find_or_create("chair", "object")
        atl.find_or_create("Dennis", "person")

        results = atl.recall(category="object")
        assert len(results) == 2
        names = {r.name for r in results}
        assert names == {"mug", "chair"}

    def test_recall_respects_limit(self, atl):
        """recall() respects the limit parameter."""
        for i in range(10):
            atl.find_or_create(f"item_{i}", "object")

        results = atl.recall(category="object", limit=3)
        assert len(results) == 3

    def test_recall_by_ids_returns_concepts(self, atl):
        """recall_by_ids() bulk-loads concepts without touching."""
        cid1, _ = atl.find_or_create("mug", "object")
        cid2, _ = atl.find_or_create("chair", "object")

        initial_count = atl.get(cid1).access_count

        results = atl.recall_by_ids([cid1, cid2])
        assert len(results) == 2

        # recall_by_ids should NOT update access tracking
        concept = atl._concepts[cid1]
        # access_count should be from the get() above + 1, not from recall_by_ids
        assert concept.access_count == initial_count

    def test_recall_by_ids_skips_missing(self, atl):
        """recall_by_ids() skips IDs that don't exist."""
        cid, _ = atl.find_or_create("mug", "object")

        results = atl.recall_by_ids([cid, "nonexistent"])
        assert len(results) == 1

    def test_recall_associated_via_graph(self, atl):
        """recall_associated() uses spreading activation to find related concepts."""
        cid1, _ = atl.find_or_create("mug", "object")
        cid2, _ = atl.find_or_create("kitchen", "location")

        # Create associative edge
        atl.define_relationship(cid1, cid2, "RELATED_TO", weight=0.8)

        results = atl.recall_associated([cid1], limit=5)
        assert len(results) >= 1
        found_ids = {r[0].id for r in results}
        assert cid2 in found_ids

    def test_recall_concepts_are_correct_type(self, atl):
        """Recalled concepts from find_or_create are Concept instances."""
        atl.find_or_create("mug", "object")
        results = atl.recall(name="mug")
        assert len(results) == 1
        assert isinstance(results[0], Concept)

    def test_concept_reinforce_via_find_or_create(self, atl):
        """Reinforcing via find_or_create updates confidence."""
        cid, _ = atl.find_or_create("mug", "object", source_episode_id="ep-1")
        concept = atl.get(cid)
        conf1 = concept.confidence

        atl.find_or_create("mug", "object", source_episode_id="ep-2")
        concept = atl.get(cid)
        assert concept.confidence > conf1

    def test_add_ref_then_recall(self, atl):
        """Concept with refs is accessible and refs are readable."""
        cid, _ = atl.find_or_create("mug", "object")
        concept = atl.get(cid)
        concept.add_ref("hippocampus", "ep-1")
        concept.add_ref("hippocampus", "ep-2")
        concept.add_ref("angular_gyrus", "math-1")

        # Re-get and verify refs
        concept2 = atl.get(cid)
        assert concept2.ref_count("hippocampus") == 2
        assert concept2.ref_count("angular_gyrus") == 1
        assert concept2.ref_count() == 3

    def test_iterator_yields_all_concepts(self, atl):
        """ATL __iter__ yields all stored concepts."""
        atl.find_or_create("mug", "object")
        atl.find_or_create("chair", "object")
        atl.find_or_create("Dennis", "person")

        concepts = list(atl)
        assert len(concepts) == 3

    def test_stats_reflect_stored_concepts(self, atl):
        """stats() reflects the number of stored concepts."""
        atl.find_or_create("mug", "object")
        atl.find_or_create("chair", "object")

        stats = atl.stats()
        assert stats["total_concepts"] == 2
        assert stats["concepts_stored"] == 2

    def test_remove_concept(self, atl):
        """Removing a concept makes it inaccessible."""
        cid, _ = atl.find_or_create("mug", "object")
        assert atl.get(cid) is not None

        atl.remove(cid)
        assert atl.get(cid) is None
        assert len(atl) == 0

    def test_define_and_find_relationship(self, atl):
        """Relationships between concepts are queryable."""
        cid1, _ = atl.find_or_create("mug", "object")
        cid2, _ = atl.find_or_create("kitchen", "location")

        atl.define_relationship(cid1, cid2, "RELATED_TO", weight=0.8, confidence=0.7)

        rels = atl.find_by_relationship(cid1, rel_type="RELATED_TO")
        assert len(rels) >= 1
        related_ids = [r[0] for r in rels]
        assert cid2 in related_ids
