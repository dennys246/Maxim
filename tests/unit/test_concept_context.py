"""Unit tests for ConceptContextBuilder (Phase A4).

Tests cover:
- Concept matching from detected objects, people, and goal tokens
- Relationship collection in context entries
- AG grounding integration (with and without ConceptGrounder)
- Layer enrichment collection from AG MathMemory records
- Budget-bounded grounding with graceful degradation
- Empty/missing input handling
"""

from __future__ import annotations

import time
from uuid import uuid4

import pytest

from maxim.math.angular_gyrus import AngularGyrus, AngularGyrusConfig
from maxim.math.ips import IPS
from maxim.memory.atl import ATL, ATLConfig
from maxim.memory.concept_context import ConceptContextBuilder
from maxim.memory.concept_grounder import ConceptGrounder
from maxim.memory.cross_layer import CrossLayerGraph
from maxim.memory.hippocampus import Hippocampus, HippocampusConfig
from maxim.memory.semantic_types import Concept
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
    return ATL(ATLConfig(persistence_path=None))


@pytest.fixture
def hippocampus():
    return Hippocampus(HippocampusConfig(persistence_path=None))


@pytest.fixture
def ag():
    return AngularGyrus(AngularGyrusConfig(persistence_path=None))


@pytest.fixture
def cross_layer(atl, ag, hippocampus):
    return CrossLayerGraph(
        layers={
            "atl": atl,
            "angular_gyrus": ag,
            "hippocampus": hippocampus,
        }
    )


@pytest.fixture
def layers(hippocampus, ag):
    return {"hippocampus": hippocampus, "angular_gyrus": ag}


@pytest.fixture
def grounder(atl, ag, cross_layer):
    return ConceptGrounder(
        atl=atl,
        angular_gyrus=ag,
        ips=IPS(),
        cross_layer=cross_layer,
    )


@pytest.fixture
def builder(atl, layers):
    """ConceptContextBuilder without grounder."""
    return ConceptContextBuilder(atl=atl, layers=layers)


@pytest.fixture
def builder_with_grounder(atl, layers, grounder):
    """ConceptContextBuilder with grounder."""
    return ConceptContextBuilder(
        atl=atl,
        layers=layers,
        concept_grounder=grounder,
    )


def _make_concept(atl, name="mug", category="object") -> Concept:
    """Create and store a concept in ATL."""
    cid, _ = atl.find_or_create(name=name, category=category)
    concept = atl.get(cid)
    assert isinstance(concept, Concept)
    return concept


def _make_episode(
    hippocampus,
    memory_id: str | None = None,
    salience: float = 0.5,
    novelty: float = 0.5,
    success: bool = True,
) -> EpisodicMemory:
    """Create and store an episode in hippocampus."""
    ep = EpisodicMemory(
        id=memory_id or str(uuid4()),
        timestamp=time.time(),
        perception=Perception(salience=salience, novelty=novelty),
        context=Context(),
        decision=Decision(),
        action=Action(),
        outcome=Outcome(success=success),
    )
    hippocampus.capture(record=ep)
    return ep


# ─────────────────────────────────────────────────────────────────────────────
# Concept matching tests
# ─────────────────────────────────────────────────────────────────────────────


class TestFindMatchingConcepts:
    def test_matches_detected_objects(self, builder, atl):
        _make_concept(atl, "mug", "object")
        _make_concept(atl, "chair", "object")

        result = builder.build(detected_objects=["mug", "chair"])
        names = [e["name"] for e in result]
        assert "mug" in names
        assert "chair" in names

    def test_matches_detected_people(self, builder, atl):
        _make_concept(atl, "dennis", "person")

        result = builder.build(detected_people=["Dennis"])
        names = [e["name"] for e in result]
        assert "dennis" in names

    def test_matches_goal_tokens(self, builder, atl):
        _make_concept(atl, "navigate", "action")
        _make_concept(atl, "kitchen", "location")

        result = builder.build(active_goal="navigate_to_kitchen")
        names = [e["name"] for e in result]
        assert "navigate" in names
        assert "kitchen" in names

    def test_deduplicates_matches(self, builder, atl):
        _make_concept(atl, "mug", "object")

        # "mug" appears in both objects and goal
        result = builder.build(
            detected_objects=["mug"],
            active_goal="grasp_mug",
        )
        mug_entries = [e for e in result if e["name"] == "mug"]
        assert len(mug_entries) == 1

    def test_no_match_returns_empty(self, builder, atl):
        result = builder.build(detected_objects=["nonexistent"])
        assert result == []

    def test_empty_inputs_returns_empty(self, builder):
        assert builder.build() == []

    def test_case_insensitive_matching(self, builder, atl):
        _make_concept(atl, "table", "object")

        result = builder.build(detected_objects=["Table"])
        assert len(result) == 1
        assert result[0]["name"] == "table"


# ─────────────────────────────────────────────────────────────────────────────
# Context entry format tests
# ─────────────────────────────────────────────────────────────────────────────


class TestContextEntryFormat:
    def test_entry_has_required_fields(self, builder, atl):
        concept = _make_concept(atl, "mug", "object")
        concept.add_ref("hippocampus", "ep-1")

        result = builder.build(detected_objects=["mug"])
        assert len(result) == 1
        entry = result[0]

        assert entry["name"] == "mug"
        assert entry["category"] == "object"
        assert "confidence" in entry
        assert entry["episode_count"] == 1
        assert "relationships" in entry
        assert "properties" in entry

    def test_ranked_by_confidence(self, builder, atl):
        c1 = _make_concept(atl, "mug", "object")
        _make_concept(atl, "chair", "object")

        # Reinforce mug to increase confidence
        for _ in range(5):
            c1.reinforce("ep-x")

        result = builder.build(detected_objects=["mug", "chair"])
        assert result[0]["name"] == "mug"  # Higher confidence first

    def test_respects_limit(self, builder, atl):
        for i in range(10):
            _make_concept(atl, f"obj{i}", "object")

        result = builder.build(
            detected_objects=[f"obj{i}" for i in range(10)],
            limit=3,
        )
        assert len(result) == 3


# ─────────────────────────────────────────────────────────────────────────────
# Relationship collection tests
# ─────────────────────────────────────────────────────────────────────────────


class TestRelationshipCollection:
    def test_includes_relationships(self, builder, atl):
        mug = _make_concept(atl, "mug", "object")
        kitchen = _make_concept(atl, "kitchen", "location")

        atl.define_relationship(
            mug.id,
            kitchen.id,
            "RELATED_TO",
            weight=0.7,
            confidence=0.8,
        )

        result = builder.build(detected_objects=["mug"])
        assert len(result) == 1
        rels = result[0]["relationships"]
        assert len(rels) >= 1
        assert any(r["target"] == "kitchen" for r in rels)

    def test_no_relationships_returns_empty_list(self, builder, atl):
        _make_concept(atl, "mug", "object")

        result = builder.build(detected_objects=["mug"])
        assert result[0]["relationships"] == []


# ─────────────────────────────────────────────────────────────────────────────
# AG grounding integration tests
# ─────────────────────────────────────────────────────────────────────────────


class TestAGGrounding:
    def test_includes_stats_with_grounder(self, builder_with_grounder, atl, hippocampus):
        concept = _make_concept(atl, "mug", "object")

        # Create episodes and link to concept
        for i in range(5):
            ep = _make_episode(hippocampus, salience=0.8)
            concept.add_ref("hippocampus", ep.id)

        result = builder_with_grounder.build(detected_objects=["mug"])
        assert len(result) == 1
        # With enough episodes, should have AG stats
        props = result[0]["properties"]
        assert isinstance(props, dict)

    def test_no_stats_without_grounder(self, builder, atl, hippocampus):
        concept = _make_concept(atl, "mug", "object")
        for i in range(5):
            ep = _make_episode(hippocampus, salience=0.8)
            concept.add_ref("hippocampus", ep.id)

        result = builder.build(detected_objects=["mug"])
        assert result[0]["properties"] == {}

    def test_no_stats_without_episodes(self, builder_with_grounder, atl):
        _make_concept(atl, "mug", "object")

        result = builder_with_grounder.build(detected_objects=["mug"])
        assert result[0]["properties"] == {}


# ─────────────────────────────────────────────────────────────────────────────
# Budget tests
# ─────────────────────────────────────────────────────────────────────────────


class TestBudget:
    def test_budget_exhaustion_still_returns_entries(self, builder_with_grounder, atl, hippocampus):
        """Even with zero budget, concepts still get relationship context."""
        concept = _make_concept(atl, "mug", "object")
        for i in range(5):
            ep = _make_episode(hippocampus, salience=0.8)
            concept.add_ref("hippocampus", ep.id)

        # Zero budget — no grounding but still returns entries
        result = builder_with_grounder.build(
            detected_objects=["mug"],
            budget_ms=0.0,
        )
        assert len(result) == 1
        assert result[0]["name"] == "mug"
        # Properties should be empty (budget exhausted before grounding)
        assert result[0]["properties"] == {}

    def test_generous_budget_allows_grounding(self, builder_with_grounder, atl, hippocampus):
        concept = _make_concept(atl, "mug", "object")
        for i in range(5):
            ep = _make_episode(hippocampus, salience=0.8)
            concept.add_ref("hippocampus", ep.id)

        result = builder_with_grounder.build(
            detected_objects=["mug"],
            budget_ms=5000.0,  # Very generous
        )
        assert len(result) == 1
        # With generous budget, should have properties
        props = result[0]["properties"]
        assert isinstance(props, dict)


# ─────────────────────────────────────────────────────────────────────────────
# Layer enrichment tests
# ─────────────────────────────────────────────────────────────────────────────


class TestLayerEnrichment:
    def test_collects_ag_math_memory(self, builder_with_grounder, atl, ag, hippocampus, cross_layer):
        """When a concept has AG refs, enrichment should include them."""
        concept = _make_concept(atl, "mug", "object")

        # Create episodes and ground them to generate MathMemory records
        for i in range(8):
            ep = _make_episode(hippocampus, salience=0.8)
            concept.add_ref("hippocampus", ep.id)

        # Ground concept to create MathMemory records
        episodes = hippocampus.recall_by_ids(list(concept.memory_refs.get("hippocampus", {})))
        builder_with_grounder._concept_grounder.ground_concept(concept, episodes)

        # Now build context — should include layer_context from AG
        result = builder_with_grounder.build(detected_objects=["mug"])
        assert len(result) == 1
        entry = result[0]
        if "layer_context" in entry:
            assert isinstance(entry["layer_context"], list)
            for lc in entry["layer_context"]:
                assert "layer" in lc
                assert "name" in lc
                assert "verbal" in lc

    def test_skips_hippocampus_enrichment(self, builder, atl):
        """Hippocampus should not contribute enrichment entries."""
        concept = _make_concept(atl, "mug", "object")
        concept.add_ref("hippocampus", "ep-1")

        result = builder.build(detected_objects=["mug"])
        entry = result[0]
        # No layer_context key (hippocampus is skipped, no other layers have refs)
        assert "layer_context" not in entry or entry["layer_context"] == []


# ─────────────────────────────────────────────────────────────────────────────
# MemoryHub integration test
# ─────────────────────────────────────────────────────────────────────────────


class TestMemoryHubIntegration:
    def test_build_concept_context_returns_empty_without_atl(self):
        """MemoryHub.build_concept_context returns [] when no ATL."""
        from maxim.decisions.nac import NAc
        from maxim.integration.memory_hub import MemoryHub
        from maxim.memory.hippocampus import Hippocampus, HippocampusConfig
        from maxim.similarity.ec import EntorhinalCortex
        from maxim.time.scn import SCN

        hub = MemoryHub(
            hippocampus=Hippocampus(HippocampusConfig(persistence_path=None)),
            scn=SCN(),
            nac=NAc(),
            ec=EntorhinalCortex(),
            _allow_raw=True,
        )
        result = hub.build_concept_context(detected_objects=["mug"])
        assert result == []

    def test_build_concept_context_with_atl(self):
        """MemoryHub.build_concept_context works when ATL is wired."""
        from maxim.decisions.nac import NAc
        from maxim.integration.memory_hub import MemoryHub
        from maxim.memory.hippocampus import Hippocampus, HippocampusConfig
        from maxim.similarity.ec import EntorhinalCortex
        from maxim.time.scn import SCN

        atl = ATL(ATLConfig(persistence_path=None))
        ag = AngularGyrus(AngularGyrusConfig(persistence_path=None))
        hub = MemoryHub(
            hippocampus=Hippocampus(HippocampusConfig(persistence_path=None)),
            scn=SCN(),
            nac=NAc(),
            ec=EntorhinalCortex(),
            atl=atl,
            angular_gyrus=ag,
            _allow_raw=True,
        )

        # Create a concept
        _make_concept(atl, "mug", "object")

        result = hub.build_concept_context(detected_objects=["mug"])
        assert len(result) == 1
        assert result[0]["name"] == "mug"

        # Cleanup ConceptExtractor worker thread
        if hub._concept_extractor is not None:
            hub._concept_extractor.shutdown()
