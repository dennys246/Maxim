"""Unit tests for ConceptGrounder async WorkerPool integration.

Tests cover:
- ground_concept() with WorkerPool enqueues to review lane instead of sync
- Review lane computes correct Jaccard proposals
- Record lane applies edge updates and quantifications
- Fallback: sync path works when worker_pool is None
- Evicted concept: review/record jobs handle missing concept gracefully
- Dedup cooldown prevents redundant review enqueues
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from maxim.math.angular_gyrus import AngularGyrus, AngularGyrusConfig
from maxim.math.ips import IPS
from maxim.memory.atl import ATL, ATLConfig
from maxim.memory.concept_grounder import ConceptGrounder
from maxim.memory.cross_layer import CrossLayerGraph
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
def ag():
    return AngularGyrus(AngularGyrusConfig(persistence_path=None))


@pytest.fixture
def ips():
    return IPS()


@pytest.fixture
def cross_layer(atl, ag):
    return CrossLayerGraph(layers={"atl": atl, "angular_gyrus": ag})


@pytest.fixture
def mock_pool():
    """Create a mock WorkerPool that tracks submitted jobs."""
    pool = MagicMock()
    pool.submit = MagicMock(return_value="mock-job-id")
    return pool


@pytest.fixture
def grounder_sync(atl, ag, ips, cross_layer):
    """ConceptGrounder without WorkerPool (sync fallback)."""
    return ConceptGrounder(
        atl=atl,
        angular_gyrus=ag,
        ips=ips,
        cross_layer=cross_layer,
    )


@pytest.fixture
def grounder_async(atl, ag, ips, cross_layer, mock_pool):
    """ConceptGrounder with mock WorkerPool."""
    return ConceptGrounder(
        atl=atl,
        angular_gyrus=ag,
        ips=ips,
        cross_layer=cross_layer,
        worker_pool=mock_pool,
    )


def _make_concept(atl, name="mug", category="object") -> Concept:
    cid, _ = atl.find_or_create(name=name, category=category)
    concept = atl.get(cid)
    assert isinstance(concept, Concept)
    return concept


def _make_episodes(n: int, salience: float = 0.7, success: bool = True) -> list[EpisodicMemory]:
    return [
        EpisodicMemory(
            id=str(uuid4()),
            timestamp=time.time(),
            perception=Perception(salience=salience, novelty=0.5),
            context=Context(),
            decision=Decision(confidence=0.8),
            action=Action(),
            outcome=Outcome(success=success),
        )
        for _ in range(n)
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Sync fallback tests
# ─────────────────────────────────────────────────────────────────────────────


class TestSyncFallback:
    def test_no_pool_runs_inline(self, grounder_sync, atl, ag):
        """Without WorkerPool, ground_concept() runs modulate + store synchronously."""
        concept = _make_concept(atl)
        episodes = _make_episodes(6)

        stats = grounder_sync.ground_concept(concept, episodes)

        assert "salience" in stats
        # Quantification should have been stored inline (n=6 >= 5)
        results = ag.recall(limit=5, name=f"{concept.name}:salience")
        assert len(results) == 1

    def test_pool_is_none_attribute(self, grounder_sync):
        assert grounder_sync._pool is None


# ─────────────────────────────────────────────────────────────────────────────
# Async path tests
# ─────────────────────────────────────────────────────────────────────────────


class TestAsyncPath:
    def test_ground_concept_submits_to_review_lane(self, grounder_async, atl, mock_pool):
        """With WorkerPool, ground_concept() should submit to review lane."""
        concept = _make_concept(atl)
        episodes = _make_episodes(5)

        stats = grounder_async.ground_concept(concept, episodes)

        assert "salience" in stats
        mock_pool.submit.assert_called_once()
        call_kwargs = mock_pool.submit.call_args
        assert call_kwargs[1]["lane"] == "review" or call_kwargs[0][0] == "review"

    def test_does_not_call_modulate_or_store_inline(self, grounder_async, atl, ag, mock_pool):
        """With WorkerPool, should NOT run _modulate_relationships or _store_quantifications inline."""
        concept = _make_concept(atl)
        episodes = _make_episodes(6)

        with patch.object(grounder_async, "_modulate_relationships") as mock_mod, \
             patch.object(grounder_async, "_store_quantifications") as mock_store:
            grounder_async.ground_concept(concept, episodes)
            mock_mod.assert_not_called()
            mock_store.assert_not_called()

    def test_returns_stats_immediately(self, grounder_async, atl, mock_pool):
        """Stats should be returned immediately without waiting for background work."""
        concept = _make_concept(atl)
        episodes = _make_episodes(5)

        stats = grounder_async.ground_concept(concept, episodes)

        assert stats is not None
        assert "salience" in stats
        assert stats["salience"]["n"] == 5


# ─────────────────────────────────────────────────────────────────────────────
# Dedup cooldown tests
# ─────────────────────────────────────────────────────────────────────────────


class TestDedupCooldown:
    def test_second_enqueue_within_cooldown_skipped(self, grounder_async, atl, mock_pool):
        """Same concept grounded twice within cooldown should only enqueue once."""
        concept = _make_concept(atl)
        episodes = _make_episodes(5)

        grounder_async.ground_concept(concept, episodes)
        # Invalidate cache so ground_concept runs stats again
        grounder_async._stats_cache.clear()
        grounder_async.ground_concept(concept, episodes)

        # Should only have submitted once (second was within cooldown)
        assert mock_pool.submit.call_count == 1

    def test_enqueue_after_cooldown_allowed(self, grounder_async, atl, mock_pool):
        """After cooldown expires, should allow re-enqueue."""
        grounder_async.REVIEW_COOLDOWN_S = 0.0  # Disable cooldown for test

        concept = _make_concept(atl)
        episodes = _make_episodes(5)

        grounder_async.ground_concept(concept, episodes)
        grounder_async._stats_cache.clear()
        grounder_async.ground_concept(concept, episodes)

        assert mock_pool.submit.call_count == 2


# ─────────────────────────────────────────────────────────────────────────────
# Review concept tests
# ─────────────────────────────────────────────────────────────────────────────


class TestReviewConcept:
    def test_review_computes_edge_updates(self, grounder_async, atl, mock_pool):
        """_review_concept should compute relationship updates and submit to record lane."""
        concept_a = _make_concept(atl, "mug", "object")
        concept_b = _make_concept(atl, "kitchen", "location")

        # Create shared refs (high Jaccard)
        for i in range(5):
            concept_a.add_ref("hippocampus", f"ep-{i}")
            concept_b.add_ref("hippocampus", f"ep-{i}")

        atl.define_relationship(
            concept_a.id, concept_b.id, "RELATED_TO",
            weight=0.3, confidence=0.3,
        )

        stats = {"salience": {"mean": 0.7, "n": 6, "min": 0.5, "max": 0.9}}

        # Call _review_concept directly (simulating review lane execution)
        grounder_async._review_concept(concept_a.id, stats)

        # Should have submitted to record lane
        assert mock_pool.submit.call_count == 1
        call_kwargs = mock_pool.submit.call_args
        assert call_kwargs[1]["lane"] == "record" or call_kwargs[0][0] == "record"

    def test_review_handles_evicted_concept(self, grounder_async, atl, mock_pool):
        """_review_concept should return early if concept was evicted."""
        grounder_async._review_concept("nonexistent-id", {})
        # Should not submit anything
        mock_pool.submit.assert_not_called()

    def test_review_skips_when_no_changes(self, grounder_async, atl, mock_pool):
        """No relationships + no qualifying stats → no record submission."""
        concept = _make_concept(atl, "isolated", "object")
        stats = {"salience": {"mean": 0.7, "n": 3, "min": 0.5, "max": 0.9}}  # n < 5

        grounder_async._review_concept(concept.id, stats)
        mock_pool.submit.assert_not_called()


# ─────────────────────────────────────────────────────────────────────────────
# Apply updates tests
# ─────────────────────────────────────────────────────────────────────────────


class TestApplyUpdates:
    def test_apply_edge_updates(self, grounder_async, atl):
        """_apply_updates should apply edge confidence changes."""
        concept_a = _make_concept(atl, "mug", "object")
        concept_b = _make_concept(atl, "kitchen", "location")

        atl.define_relationship(
            concept_a.id, concept_b.id, "RELATED_TO",
            weight=0.3, confidence=0.3,
        )

        edge_updates = [
            (concept_a.id, concept_b.id, "RELATED_TO", 0.8, 0.05),
        ]

        grounder_async._apply_updates(concept_a.id, edge_updates, [])
        # Should not crash; edge update applied

    def test_apply_quantification_proposals(self, grounder_async, atl, ag, cross_layer):
        """_apply_updates should create AG MathMemory records."""
        concept = _make_concept(atl)

        quant_proposals = [{
            "field_name": "salience",
            "concept_name": concept.name,
            "concept_id": concept.id,
            "mean": 0.75,
            "n": 8,
        }]

        grounder_async._apply_updates(concept.id, [], quant_proposals)

        results = ag.recall(limit=5, name=f"{concept.name}:salience")
        assert len(results) == 1
        assert results[0].observation_count == 8

    def test_apply_handles_evicted_concept(self, grounder_async):
        """_apply_updates should return early for evicted concepts."""
        # Should not crash
        grounder_async._apply_updates("nonexistent-id", [], [])

    def test_apply_updates_creates_quantifies_edge(self, grounder_async, atl, ag, cross_layer):
        """Quantification proposals should create QUANTIFIES cross-layer edges."""
        concept = _make_concept(atl)

        quant_proposals = [{
            "field_name": "salience",
            "concept_name": concept.name,
            "concept_id": concept.id,
            "mean": 0.7,
            "n": 6,
        }]

        grounder_async._apply_updates(concept.id, [], quant_proposals)

        edges = cross_layer._incoming.get(("atl", concept.id), [])
        quantifies_edges = [e for e in edges if e.edge_type.name == "QUANTIFIES"]
        assert len(quantifies_edges) >= 1

    def test_apply_adds_ag_ref_to_concept(self, grounder_async, atl, ag):
        """Quantification should add angular_gyrus ref to concept."""
        concept = _make_concept(atl)

        quant_proposals = [{
            "field_name": "salience",
            "concept_name": concept.name,
            "concept_id": concept.id,
            "mean": 0.7,
            "n": 6,
        }]

        grounder_async._apply_updates(concept.id, [], quant_proposals)
        assert concept.ref_count("angular_gyrus") >= 1


# ─────────────────────────────────────────────────────────────────────────────
# Compute relationship updates tests
# ─────────────────────────────────────────────────────────────────────────────


class TestComputeRelationshipUpdates:
    def test_high_jaccard_produces_strengthen_update(self, grounder_async, atl):
        concept_a = _make_concept(atl, "mug", "object")
        concept_b = _make_concept(atl, "kitchen", "location")

        for i in range(5):
            concept_a.add_ref("hippocampus", f"ep-{i}")
            concept_b.add_ref("hippocampus", f"ep-{i}")

        atl.define_relationship(
            concept_a.id, concept_b.id, "RELATED_TO",
            weight=0.3, confidence=0.3,
        )

        updates = grounder_async._compute_relationship_updates(concept_a)
        assert len(updates) > 0
        # Should be a strengthen update (positive confidence_delta)
        _, _, _, _, conf_delta = updates[0]
        assert conf_delta > 0

    def test_low_jaccard_produces_weaken_update(self, grounder_async, atl):
        concept_a = _make_concept(atl, "mug", "object")
        concept_b = _make_concept(atl, "lamp", "object")

        for i in range(12):
            concept_a.add_ref("hippocampus", f"ep-a-{i}")
        for i in range(12):
            concept_b.add_ref("hippocampus", f"ep-b-{i}")

        atl.define_relationship(
            concept_a.id, concept_b.id, "RELATED_TO",
            weight=0.5, confidence=0.5,
        )

        updates = grounder_async._compute_relationship_updates(concept_a)
        assert len(updates) > 0
        _, _, _, _, conf_delta = updates[0]
        assert conf_delta < 0

    def test_no_relationships_returns_empty(self, grounder_async, atl):
        concept = _make_concept(atl, "isolated", "object")
        updates = grounder_async._compute_relationship_updates(concept)
        assert updates == []


# ─────────────────────────────────────────────────────────────────────────────
# Pool submit failure handling
# ─────────────────────────────────────────────────────────────────────────────


class TestPoolFailureHandling:
    def test_enqueue_failure_clears_pending(self, grounder_async, atl, mock_pool):
        """If pool.submit raises, pending flag should be cleared for retry."""
        mock_pool.submit.side_effect = RuntimeError("pool full")

        concept = _make_concept(atl)
        episodes = _make_episodes(5)

        # Should not raise
        grounder_async.ground_concept(concept, episodes)

        # Pending should be cleared so next attempt can retry
        assert concept.id not in grounder_async._pending_reviews
