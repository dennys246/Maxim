"""Unit tests for ConceptExtractor (Phase A2).

Tests cover:
- Concept extraction from episodic memories (objects, people, locations, goals, actions)
- Queue-based async processing
- Inline relationship formation between co-occurring concepts
- Reverse index maintenance and rebuild
- Deletion and compression cleanup callbacks
- Goal tokenization via normalize_tokens
- Relationship type inference
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest

from maxim.memory.atl import ATL, ATLConfig
from maxim.memory.concept_extractor import ConceptExtractor, _is_structured_goal
from maxim.memory.cross_layer import CrossLayerGraph
from maxim.memory.semantic_types import Concept
from maxim.memory.text import normalize_tokens
from maxim.memory.types import (
    Action,
    Context,
    Decision,
    EpisodicMemory,
    Outcome,
    Perception,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def atl():
    """Fresh ATL instance."""
    return ATL(ATLConfig(persistence_path=None))


@pytest.fixture
def cross_layer(atl):
    """CrossLayerGraph with ATL registered."""
    return CrossLayerGraph(layers={"atl": atl})


@pytest.fixture
def extractor(atl, cross_layer):
    """ConceptExtractor with ATL and CrossLayerGraph."""
    ext = ConceptExtractor(atl=atl, cross_layer=cross_layer, scn=None)
    yield ext
    ext.shutdown()


def _make_episode(
    memory_id: str = "ep-1",
    objects: list[str] | None = None,
    people: list[str] | None = None,
    observations: dict | None = None,
    goal: str | None = None,
    tool_name: str = "",
) -> EpisodicMemory:
    """Helper to create a minimal EpisodicMemory."""
    return EpisodicMemory(
        id=memory_id,
        timestamp=time.time(),
        perception=Perception(
            detected_objects=objects or [],
            detected_people=people or [],
            observations=observations or {},
        ),
        context=Context(active_goal=goal),
        decision=Decision(intent={"goal": goal} if goal else {}),
        action=Action(tool_name=tool_name),
        outcome=Outcome(),
    )


# ─────────────────────────────────────────────────────────────────────────────
# normalize_tokens tests
# ─────────────────────────────────────────────────────────────────────────────


class TestNormalizeTokens:
    def test_basic_split(self):
        tokens = normalize_tokens("navigate to kitchen")
        assert "navigate" in tokens
        assert "kitchen" in tokens
        assert "to" not in tokens  # stop word

    def test_underscore_split(self):
        tokens = normalize_tokens("navigate_to_kitchen")
        assert "navigate" in tokens
        assert "kitchen" in tokens
        assert "to" not in tokens

    def test_lemmatize_ing(self):
        tokens = normalize_tokens("grasping running")
        assert "grasp" in tokens
        assert "run" in tokens

    def test_lemmatize_ed(self):
        tokens = normalize_tokens("grasped carried")
        assert "grasp" in tokens
        assert "carry" in tokens

    def test_lemmatize_s(self):
        tokens = normalize_tokens("mugs chairs")
        assert "mug" in tokens
        assert "chair" in tokens

    def test_short_words_filtered(self):
        tokens = normalize_tokens("a I x")
        assert len(tokens) == 0

    def test_empty_string(self):
        assert normalize_tokens("") == []

    def test_lemmatize_ies(self):
        tokens = normalize_tokens("batteries")
        assert "battery" in tokens

    def test_punctuation_stripped(self):
        """'plan,' and 'action.' should normalize to 'plan' and 'action'."""
        tokens = normalize_tokens("plan, action.")
        assert "plan" in tokens
        assert "action" in tokens
        # Punctuation-attached variants should NOT appear
        assert "plan," not in tokens
        assert "action." not in tokens

    def test_contraction_split_not_lemmatized_to_garbage(self):
        """'it's' previously became 'it'' via the -s stripper — now it
        splits on the apostrophe and both halves are filtered."""
        tokens = normalize_tokens("it's not possible")
        # 'it' and 's' are both too short / stopwords
        assert "it'" not in tokens
        assert "it's" not in tokens
        assert "it" not in tokens

    def test_taking_does_not_become_tak(self):
        """The -ing stripper must not produce garbage stems from short words."""
        tokens = normalize_tokens("taking")
        assert "tak" not in tokens
        # "taking" is kept as-is because the stem would be too short
        assert "taking" in tokens

    def test_expanded_stopwords(self):
        """Common filler words added to the stopword list in the concept
        bloat fix should be filtered."""
        for filler in ["so", "before", "best", "just", "very", "you", "some"]:
            tokens = normalize_tokens(f"plan {filler} action")
            assert filler not in tokens, f"{filler!r} should be a stopword"

    def test_natural_language_sentence_is_quiet(self):
        """A typical free-form narrative goal should yield a small number
        of meaningful tokens — not one per word. Regression coverage for
        the concept-bloat bug where a 17-word sentence produced 12 per-
        word concepts like 'so', 'it'', 'tak', 'plan,', 'action.'"""
        goal = "The user has not specified a detailed plan, so it's best to seek clarification before taking action."
        tokens = normalize_tokens(goal)
        # Each of the bug-output tokens should be cleaned up
        assert "plan," not in tokens
        assert "action." not in tokens
        assert "it'" not in tokens
        assert "tak" not in tokens
        assert "so" not in tokens
        assert "before" not in tokens
        assert "best" not in tokens
        # The clean core nouns and verbs should still be there
        assert "user" in tokens
        assert "plan" in tokens
        assert "action" in tokens
        assert "clarification" in tokens


# ─────────────────────────────────────────────────────────────────────────────
# _is_structured_goal gate — the architectural fix that prevents ATL bloat
# ─────────────────────────────────────────────────────────────────────────────


class TestIsStructuredGoal:
    def test_compound_identifier_is_structured(self):
        assert _is_structured_goal("navigate_to_kitchen") is True
        assert _is_structured_goal("pick_up_red_mug") is True

    def test_short_phrase_is_structured(self):
        assert _is_structured_goal("find red cup") is True
        assert _is_structured_goal("grab the mug") is True
        assert _is_structured_goal("pick up bottle") is True
        assert _is_structured_goal("go to kitchen now") is True  # 4 tokens, edge

    def test_long_sentence_is_not_structured(self):
        """Free-form narrative goals should NOT trigger concept tokenization."""
        assert _is_structured_goal("gather information about the server room") is False
        assert _is_structured_goal("complete the mission successfully and return home safely") is False
        assert (
            _is_structured_goal("The user has not specified a detailed plan, so it's best to seek clarification")
            is False
        )

    def test_empty_is_not_structured(self):
        assert _is_structured_goal("") is False
        assert _is_structured_goal(None) is False  # type: ignore[arg-type]


# ─────────────────────────────────────────────────────────────────────────────
# ConceptExtractor tests
# ─────────────────────────────────────────────────────────────────────────────


class TestConceptExtraction:
    def test_extracts_objects(self, extractor, atl):
        ep = _make_episode(objects=["mug", "Table"])
        extractor.on_memory_captured("ep-1", ep)
        extractor.flush()

        # Should create two object concepts (lowercased)
        mug = atl.recall(limit=1, name="mug")
        table = atl.recall(limit=1, name="table")
        assert len(mug) == 1
        assert mug[0].category == "object"
        assert len(table) == 1
        assert table[0].category == "object"

    def test_extracts_people(self, extractor, atl):
        ep = _make_episode(people=["Dennis"])
        extractor.on_memory_captured("ep-1", ep)
        extractor.flush()

        results = atl.recall(limit=1, name="Dennis")
        assert len(results) == 1
        assert results[0].category == "person"

    def test_extracts_location_from_observations(self, extractor, atl):
        ep = _make_episode(observations={"location": "Kitchen"})
        extractor.on_memory_captured("ep-1", ep)
        extractor.flush()

        results = atl.recall(limit=1, name="kitchen")
        assert len(results) == 1
        assert results[0].category == "location"

    def test_extracts_room_fallback(self, extractor, atl):
        ep = _make_episode(observations={"room": "Bedroom"})
        extractor.on_memory_captured("ep-1", ep)
        extractor.flush()

        results = atl.recall(limit=1, name="bedroom")
        assert len(results) == 1

    def test_extracts_action(self, extractor, atl):
        ep = _make_episode(tool_name="navigate")
        extractor.on_memory_captured("ep-1", ep)
        extractor.flush()

        results = atl.recall(limit=1, name="navigate")
        assert len(results) == 1
        assert results[0].category == "action"

    def test_extracts_goal_tokens(self, extractor, atl):
        ep = _make_episode(goal="navigate_to_kitchen")
        extractor.on_memory_captured("ep-1", ep)
        extractor.flush()

        # Should tokenize into "navigate" and "kitchen"
        nav = atl.recall(limit=1, name="navigate")
        kitchen = atl.recall(limit=1, name="kitchen")
        assert len(nav) == 1
        assert len(kitchen) == 1

    def test_reinforces_existing_concept(self, extractor, atl):
        ep1 = _make_episode(memory_id="ep-1", objects=["mug"])
        ep2 = _make_episode(memory_id="ep-2", objects=["mug"])

        extractor.on_memory_captured("ep-1", ep1)
        extractor.flush()
        extractor.on_memory_captured("ep-2", ep2)
        extractor.flush()

        results = atl.recall(limit=1, name="mug")
        assert len(results) == 1
        concept = results[0]
        # Should have been reinforced (confidence > initial 0.5)
        assert concept.reinforcement_count >= 2

    def test_adds_memory_ref(self, extractor, atl):
        ep = _make_episode(memory_id="ep-1", objects=["mug"])
        extractor.on_memory_captured("ep-1", ep)
        extractor.flush()

        results = atl.recall(limit=1, name="mug")
        assert len(results) == 1
        concept = results[0]
        assert isinstance(concept, Concept)
        assert concept.ref_count("hippocampus") >= 1

    def test_creates_cross_layer_edge(self, extractor, cross_layer):
        ep = _make_episode(memory_id="ep-1", objects=["mug"])
        extractor.on_memory_captured("ep-1", ep)
        extractor.flush()

        # Should have INSTANCE_OF edge from hippocampus to atl
        edges = cross_layer._outgoing.get(("hippocampus", "ep-1"), [])
        assert len(edges) >= 1
        assert any(e.target_layer == "atl" for e in edges)

    def test_ignores_non_episodic(self, extractor, atl):
        """Non-EpisodicMemory records should be silently ignored."""
        extractor.on_memory_captured("id-1", MagicMock())
        extractor.flush()
        assert len(atl) == 0


class TestInlineRelationships:
    def test_forms_relationships_between_active_and_passive(self, extractor, atl):
        ep = _make_episode(objects=["mug"], tool_name="grasp", goal="grasp_mug")
        extractor.on_memory_captured("ep-1", ep)
        extractor.flush()

        # "grasp" (action) should have relationships with other concepts
        grasp_results = atl.recall(limit=1, name="grasp")
        assert len(grasp_results) >= 1
        grasp = grasp_results[0]

        rels = atl.find_by_relationship(grasp.id, direction="both")
        assert len(rels) > 0

    def test_no_relationship_between_two_passive_objects(self, extractor, atl):
        """Two objects without an active concept should not be related."""
        ep = _make_episode(objects=["mug", "chair"])
        extractor.on_memory_captured("ep-1", ep)
        extractor.flush()

        mug = atl.recall(limit=1, name="mug")
        assert len(mug) == 1
        rels = atl.find_by_relationship(mug[0].id, direction="both")
        # Should have no relationships (both are passive objects)
        mug_chair_rels = [(oid, r) for oid, r in rels if atl.get(oid) and atl.get(oid).name == "chair"]
        assert len(mug_chair_rels) == 0

    def test_relationship_reinforcement(self, extractor, atl):
        """Repeated co-occurrence should reinforce relationships."""
        for i in range(3):
            ep = _make_episode(
                memory_id=f"ep-{i}",
                objects=["mug"],
                observations={"location": "kitchen"},
                tool_name="grasp",
            )
            extractor.on_memory_captured(f"ep-{i}", ep)
            extractor.flush()

        # grasp->mug relationship should exist and be reinforced
        grasp = atl.recall(limit=1, name="grasp")
        assert len(grasp) >= 1
        rels = atl.find_by_relationship(grasp[0].id, direction="both")
        assert len(rels) > 0

    def test_max_relationships_per_episode(self, extractor, atl):
        """Should cap relationships formed per episode."""
        # Create many concepts in one episode
        ep = _make_episode(
            objects=["a", "b", "c", "d", "e", "f", "g", "h"],
            tool_name="scan",  # active concept to trigger relationships
        )
        extractor.on_memory_captured("ep-1", ep)
        extractor.flush()

        scan = atl.recall(limit=1, name="scan")
        if scan:
            # Use direction="outgoing" to avoid double-counting symmetric edges
            rels = atl.find_by_relationship(scan[0].id, direction="outgoing", limit=50)
            assert len(rels) <= ConceptExtractor.MAX_RELATIONSHIPS_PER_EPISODE


class TestRelationshipTypeInference:
    def test_person_location(self):
        assert ConceptExtractor._infer_relationship_type("person", "location") == "RELATED_TO"

    def test_object_location(self):
        assert ConceptExtractor._infer_relationship_type("object", "location") == "RELATED_TO"

    def test_action_object(self):
        assert ConceptExtractor._infer_relationship_type("action", "object") == "RELATED_TO"

    def test_goal_action(self):
        assert ConceptExtractor._infer_relationship_type("goal", "action") == "HAS_PART"

    def test_object_object(self):
        assert ConceptExtractor._infer_relationship_type("object", "object") == "RELATED_TO"

    def test_unknown_defaults_to_associates(self):
        assert ConceptExtractor._infer_relationship_type("goal", "location") == "ASSOCIATES"


class TestCleanupCallbacks:
    def test_deletion_removes_refs(self, extractor, atl):
        ep = _make_episode(memory_id="ep-1", objects=["mug"])
        extractor.on_memory_captured("ep-1", ep)
        extractor.flush()

        concept = atl.recall(limit=1, name="mug")[0]
        assert isinstance(concept, Concept)
        assert concept.ref_count("hippocampus") >= 1

        # Simulate deletion
        extractor.on_memory_deleted("ep-1")

        concept = atl.recall(limit=1, name="mug")[0]
        assert isinstance(concept, Concept)
        assert concept.ref_count("hippocampus") == 0

    def test_compression_removes_refs(self, extractor, atl):
        ep = _make_episode(memory_id="ep-1", objects=["mug"])
        extractor.on_memory_captured("ep-1", ep)
        extractor.flush()

        concept = atl.recall(limit=1, name="mug")[0]
        assert isinstance(concept, Concept)
        assert concept.ref_count("hippocampus") >= 1

        # Simulate compression
        extractor.on_memory_compressed("ep-1")

        concept = atl.recall(limit=1, name="mug")[0]
        assert isinstance(concept, Concept)
        assert concept.ref_count("hippocampus") == 0

    def test_deletion_of_unknown_memory_is_safe(self, extractor):
        """Deleting an unknown memory should not raise."""
        extractor.on_memory_deleted("nonexistent-id")

    def test_compression_of_unknown_memory_is_safe(self, extractor):
        """Compressing an unknown memory should not raise."""
        extractor.on_memory_compressed("nonexistent-id")


class TestReverseIndex:
    def test_rebuild_from_atl(self, atl, cross_layer):
        """Reverse index should be rebuilt from ATL concept memory_refs."""
        # Manually create a concept with refs
        concept = Concept(
            id="c-1",
            timestamp=time.time(),
            name="mug",
            category="object",
        )
        concept.add_ref("hippocampus", "ep-1")
        concept.add_ref("hippocampus", "ep-2")
        atl.store(concept)

        ext = ConceptExtractor(atl=atl, cross_layer=cross_layer, scn=None)
        ext.rebuild_reverse_index()

        assert "ep-1" in ext._reverse_index
        assert "c-1" in ext._reverse_index["ep-1"]
        assert "ep-2" in ext._reverse_index
        assert "c-1" in ext._reverse_index["ep-2"]
        ext.shutdown()

    def test_reverse_index_tracks_multiple_concepts(self, extractor, atl):
        """A single episode can create multiple concept refs."""
        ep = _make_episode(memory_id="ep-1", objects=["mug", "chair"], tool_name="scan")
        extractor.on_memory_captured("ep-1", ep)
        extractor.flush()

        # ep-1 should track refs to mug, chair, and scan concepts
        assert "ep-1" in extractor._reverse_index
        assert len(extractor._reverse_index["ep-1"]) >= 3


class TestQueueBehavior:
    def test_flush_completes(self, extractor, atl):
        """flush() should wait until queue is drained."""
        for i in range(5):
            ep = _make_episode(memory_id=f"ep-{i}", objects=[f"obj-{i}"])
            extractor.on_memory_captured(f"ep-{i}", ep)

        result = extractor.flush()
        assert result is True
        assert len(atl) >= 5

    def test_shutdown_stops_worker(self, atl, cross_layer):
        """shutdown() should stop the worker thread."""
        ext = ConceptExtractor(atl=atl, cross_layer=cross_layer, scn=None)
        assert ext._worker.is_alive()
        ext.shutdown()
        assert not ext._worker.is_alive()


class TestSCNIntegration:
    def test_registers_with_scn_when_available(self, atl, cross_layer):
        """When SCN is provided, concepts should be registered with SCN."""
        mock_scn = MagicMock()
        ext = ConceptExtractor(
            atl=atl,
            cross_layer=cross_layer,
            scn=mock_scn,
        )

        ep = _make_episode(memory_id="ep-1", objects=["mug"])
        ext.on_memory_captured("ep-1", ep)
        ext.flush()

        # SCN.register should have been called
        assert mock_scn.register.called
        ext.shutdown()

    def test_works_without_scn(self, extractor, atl):
        """Should work fine when SCN is None."""
        ep = _make_episode(memory_id="ep-1", objects=["mug"])
        extractor.on_memory_captured("ep-1", ep)
        extractor.flush()
        assert len(atl) >= 1
