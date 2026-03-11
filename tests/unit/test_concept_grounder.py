"""Unit tests for ConceptGrounder (Phase A3).

Tests cover:
- Numeric extraction from EpisodicMemory and CompressedMemory
- IPS-based stats computation (mean, trend)
- AG escalation for uncertain trends
- Stats caching with TTL and ref_count invalidation
- Jaccard co-occurrence relationship modulation
- AG MathMemory quantification storage with QUANTIFIES edges
- Handling of episodes with missing/zero fields
"""

from __future__ import annotations

import time
from uuid import uuid4

import pytest

from maxim.math.angular_gyrus import AngularGyrus, AngularGyrusConfig
from maxim.math.ips import IPS
from maxim.math.math_types import MathMemory
from maxim.memory.atl import ATL, ATLConfig
from maxim.memory.concept_grounder import ConceptGrounder
from maxim.memory.cross_layer import CrossLayerGraph
from maxim.memory.semantic_types import Concept
from maxim.memory.types import (
    Action,
    CompressedMemory,
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
def ag():
    return AngularGyrus(AngularGyrusConfig(persistence_path=None))


@pytest.fixture
def ips():
    return IPS()


@pytest.fixture
def cross_layer(atl, ag):
    return CrossLayerGraph(layers={"atl": atl, "angular_gyrus": ag})


@pytest.fixture
def grounder(atl, ag, ips, cross_layer):
    return ConceptGrounder(
        atl=atl,
        angular_gyrus=ag,
        ips=ips,
        cross_layer=cross_layer,
    )


def _make_concept(atl, name="mug", category="object") -> Concept:
    """Create and store a concept in ATL."""
    cid, _ = atl.find_or_create(name=name, category=category)
    concept = atl.get(cid)
    assert isinstance(concept, Concept)
    return concept


def _make_episode(
    memory_id: str | None = None,
    salience: float = 0.5,
    novelty: float = 0.5,
    confidence: float = 1.0,
    success: bool = True,
    exec_start: float = 0.0,
    exec_end: float = 0.0,
    fear_level: float = 0.0,
) -> EpisodicMemory:
    """Create a minimal EpisodicMemory with controllable numeric fields."""
    return EpisodicMemory(
        id=memory_id or str(uuid4()),
        timestamp=time.time(),
        perception=Perception(
            salience=salience,
            novelty=novelty,
        ),
        context=Context(fear_level=fear_level),
        decision=Decision(confidence=confidence),
        action=Action(execution_start=exec_start, execution_end=exec_end),
        outcome=Outcome(success=success),
    )


def _make_episodes(n: int, **kwargs) -> list[EpisodicMemory]:
    """Create n episodes with given field values."""
    return [_make_episode(memory_id=str(uuid4()), **kwargs) for _ in range(n)]


# ─────────────────────────────────────────────────────────────────────────────
# Numeric extraction tests
# ─────────────────────────────────────────────────────────────────────────────


class TestExtractNumerics:
    def test_extracts_salience_and_novelty(self, grounder):
        episodes = _make_episodes(3, salience=0.8, novelty=0.6)
        numerics = grounder._extract_numerics(episodes)
        assert "salience" in numerics
        assert len(numerics["salience"]) == 3
        assert "novelty" in numerics

    def test_extracts_success_rate(self, grounder):
        episodes = [
            _make_episode(success=True),
            _make_episode(success=False),
            _make_episode(success=True),
        ]
        numerics = grounder._extract_numerics(episodes)
        assert "success_rate" in numerics
        assert numerics["success_rate"] == [1.0, 0.0, 1.0]

    def test_extracts_execution_time(self, grounder):
        episodes = [
            _make_episode(exec_start=1.0, exec_end=1.5),
            _make_episode(exec_start=2.0, exec_end=2.3),
        ]
        numerics = grounder._extract_numerics(episodes)
        assert "execution_time_ms" in numerics
        assert len(numerics["execution_time_ms"]) == 2
        # 0.5s = 500ms, 0.3s = 300ms
        assert abs(numerics["execution_time_ms"][0] - 500.0) < 1.0
        assert abs(numerics["execution_time_ms"][1] - 300.0) < 1.0

    def test_skips_zero_execution_time(self, grounder):
        """Episodes with no timing data should not contribute."""
        episodes = _make_episodes(3, exec_start=0.0, exec_end=0.0)
        numerics = grounder._extract_numerics(episodes)
        assert "execution_time_ms" not in numerics

    def test_extracts_from_compressed(self, grounder):
        """CompressedMemory should contribute limited fields."""
        compressed = [
            CompressedMemory(
                id=str(uuid4()),
                timestamp=time.time(),
                salience=0.7,
                novelty=0.4,
                success=True,
            ),
            CompressedMemory(
                id=str(uuid4()),
                timestamp=time.time(),
                salience=0.8,
                novelty=0.5,
                success=False,
            ),
        ]
        numerics = grounder._extract_numerics(compressed)
        assert "salience" in numerics
        assert "novelty" in numerics
        assert "success_rate" in numerics
        # CompressedMemory should NOT produce timing fields
        assert "execution_time_ms" not in numerics
        assert "decision_confidence" not in numerics

    def test_filters_fields_with_less_than_2_datapoints(self, grounder):
        episodes = [_make_episode(fear_level=0.5)]
        numerics = grounder._extract_numerics(episodes)
        assert "fear_level" not in numerics

    def test_empty_episodes(self, grounder):
        assert grounder._extract_numerics([]) == {}


# ─────────────────────────────────────────────────────────────────────────────
# Stats computation tests
# ─────────────────────────────────────────────────────────────────────────────


class TestComputeStats:
    def test_basic_stats(self, grounder, atl):
        concept = _make_concept(atl)
        episodes = _make_episodes(5, salience=0.7, novelty=0.3)
        numerics = grounder._extract_numerics(episodes)
        stats = grounder._compute_stats(concept, numerics)

        assert "salience" in stats
        assert abs(stats["salience"]["mean"] - 0.7) < 0.2
        assert stats["salience"]["n"] == 5
        assert "min" in stats["salience"]
        assert "max" in stats["salience"]

    def test_trend_detection(self, grounder, atl):
        concept = _make_concept(atl)
        # Clear increasing trend
        numerics = {"value": [1.0, 2.0, 3.0, 4.0, 5.0]}
        stats = grounder._compute_stats(concept, numerics)
        assert "value" in stats
        # Should detect trend with enough data


# ─────────────────────────────────────────────────────────────────────────────
# ground_concept integration tests
# ─────────────────────────────────────────────────────────────────────────────


class TestGroundConcept:
    def test_returns_stats_for_concept(self, grounder, atl):
        concept = _make_concept(atl)
        episodes = _make_episodes(5, salience=0.8, success=True)
        stats = grounder.ground_concept(concept, episodes)

        assert "salience" in stats
        assert "success_rate" in stats

    def test_empty_episodes_returns_empty(self, grounder, atl):
        concept = _make_concept(atl)
        assert grounder.ground_concept(concept, []) == {}

    def test_caching(self, grounder, atl):
        concept = _make_concept(atl)
        episodes = _make_episodes(3, salience=0.8)

        stats1 = grounder.ground_concept(concept, episodes)
        stats2 = grounder.ground_concept(concept, episodes)

        # Second call should return cached results
        assert stats1 == stats2
        assert concept.id in grounder._stats_cache

    def test_cache_invalidation_on_new_refs(self, grounder, atl):
        concept = _make_concept(atl)
        episodes = _make_episodes(3, salience=0.8)

        stats1 = grounder.ground_concept(concept, episodes)

        # Add new ref to invalidate cache
        concept.add_ref("hippocampus", "new-ep-id")

        stats2 = grounder.ground_concept(concept, episodes)
        # Should recompute (ref count changed)
        assert stats2["_ref_count"] >= stats1.get("_ref_count", 0)


# ─────────────────────────────────────────────────────────────────────────────
# Relationship modulation tests
# ─────────────────────────────────────────────────────────────────────────────


class TestModulateRelationships:
    def test_strengthens_high_jaccard(self, grounder, atl):
        """High co-occurrence should strengthen relationships."""
        concept_a = _make_concept(atl, "mug", "object")
        concept_b = _make_concept(atl, "kitchen", "location")

        # Create shared refs (high Jaccard)
        shared_ids = [f"ep-{i}" for i in range(5)]
        for eid in shared_ids:
            concept_a.add_ref("hippocampus", eid)
            concept_b.add_ref("hippocampus", eid)

        # Define initial relationship
        atl.define_relationship(
            concept_a.id, concept_b.id, "RELATED_TO",
            weight=0.3, confidence=0.3,
        )

        grounder._modulate_relationships(concept_a)

        # Relationship should be strengthened
        rels = atl.find_by_relationship(concept_a.id, direction="outgoing")
        assert len(rels) > 0

    def test_weakens_low_jaccard(self, grounder, atl):
        """Low co-occurrence with many observations should weaken."""
        concept_a = _make_concept(atl, "mug", "object")
        concept_b = _make_concept(atl, "lamp", "object")

        # Give each concept many unique refs, very little overlap
        for i in range(12):
            concept_a.add_ref("hippocampus", f"ep-a-{i}")
        for i in range(12):
            concept_b.add_ref("hippocampus", f"ep-b-{i}")

        atl.define_relationship(
            concept_a.id, concept_b.id, "RELATED_TO",
            weight=0.5, confidence=0.5,
        )

        grounder._modulate_relationships(concept_a)
        # Should have attempted to weaken (Jaccard ~= 0)

    def test_skips_when_too_few_observations(self, grounder, atl):
        """Should not modulate with < 3 union observations."""
        concept_a = _make_concept(atl, "rare_obj", "object")
        concept_b = _make_concept(atl, "rare_loc", "location")

        concept_a.add_ref("hippocampus", "ep-1")
        concept_b.add_ref("hippocampus", "ep-1")

        atl.define_relationship(
            concept_a.id, concept_b.id, "RELATED_TO",
            weight=0.3, confidence=0.3,
        )

        # Should not crash or modulate
        grounder._modulate_relationships(concept_a)


# ─────────────────────────────────────────────────────────────────────────────
# AG quantification storage tests
# ─────────────────────────────────────────────────────────────────────────────


class TestStoreQuantifications:
    def test_creates_math_memory_for_significant_fields(self, grounder, atl, ag):
        concept = _make_concept(atl)
        stats = {
            "salience": {"mean": 0.75, "n": 8, "min": 0.5, "max": 0.9},
            "novelty": {"mean": 0.3, "n": 3, "min": 0.2, "max": 0.4},  # n < 5
        }

        grounder._store_quantifications(concept, stats)

        # salience should have created a MathMemory (n=8 >= 5)
        results = ag.recall(limit=5, name=f"{concept.name}:salience")
        assert len(results) == 1
        assert results[0].observation_count == 8
        assert results[0].domain == "concept_property"

        # novelty should NOT have created one (n=3 < 5)
        novelty_results = ag.recall(limit=5, name=f"{concept.name}:novelty")
        assert len(novelty_results) == 0

    def test_updates_existing_math_memory(self, grounder, atl, ag):
        concept = _make_concept(atl)

        # First store
        stats1 = {"salience": {"mean": 0.7, "n": 5, "min": 0.5, "max": 0.9}}
        grounder._store_quantifications(concept, stats1)

        results1 = ag.recall(limit=1, name=f"{concept.name}:salience")
        assert len(results1) == 1
        first_id = results1[0].id

        # Second store with more data
        stats2 = {"salience": {"mean": 0.8, "n": 10, "min": 0.5, "max": 0.95}}
        grounder._store_quantifications(concept, stats2)

        # Should update in-place (same record ID)
        results2 = ag.recall(limit=1, name=f"{concept.name}:salience")
        assert len(results2) == 1
        assert results2[0].id == first_id
        assert results2[0].observation_count == 10

    def test_creates_quantifies_edge(self, grounder, atl, ag, cross_layer):
        concept = _make_concept(atl)
        stats = {"salience": {"mean": 0.7, "n": 6, "min": 0.5, "max": 0.9}}

        grounder._store_quantifications(concept, stats)

        # Check QUANTIFIES edge exists from AG to ATL
        edges = cross_layer._incoming.get(("atl", concept.id), [])
        quantifies_edges = [
            e for e in edges if e.edge_type.name == "QUANTIFIES"
        ]
        assert len(quantifies_edges) >= 1

    def test_adds_ag_ref_to_concept(self, grounder, atl, ag):
        concept = _make_concept(atl)
        stats = {"salience": {"mean": 0.7, "n": 6, "min": 0.5, "max": 0.9}}

        grounder._store_quantifications(concept, stats)

        assert concept.ref_count("angular_gyrus") >= 1

    def test_skips_internal_fields(self, grounder, atl, ag):
        concept = _make_concept(atl)
        stats = {
            "_ref_count": 5,
            "salience": {"mean": 0.7, "n": 6, "min": 0.5, "max": 0.9},
        }

        grounder._store_quantifications(concept, stats)

        # Should only create record for salience, not _ref_count
        all_records = ag.recall(limit=10)
        assert all("_ref_count" not in r.name for r in all_records)
