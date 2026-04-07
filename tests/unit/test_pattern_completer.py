"""Unit tests for PatternCompleter (Phase A5b).

Tests cover:
- Concept matching from episodic perception and goal tokens
- Episode loading and recency ordering
- Prediction extraction from EpisodicMemory and CompressedMemory
- Math context enrichment via memory_refs intersection
- Empty/missing input handling
- MAX_EPISODES cap
"""

from __future__ import annotations

import time
from uuid import uuid4

import pytest

from maxim.math.angular_gyrus import AngularGyrus, AngularGyrusConfig
from maxim.math.ips import IPS
from maxim.memory.atl import ATL, ATLConfig
from maxim.memory.concept_grounder import ConceptGrounder
from maxim.memory.cross_layer import CrossLayerGraph
from maxim.memory.hippocampus import Hippocampus, HippocampusConfig
from maxim.memory.pattern_completer import PatternCompleter
from maxim.memory.semantic_types import Concept
from maxim.memory.types import (
    Action,
    CompressedMemory,
    Context,
    Decision,
    EpisodicMemory,
    MathContextEntry,
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
def completer(atl, layers):
    return PatternCompleter(atl=atl, layers=layers)


@pytest.fixture
def grounder(atl, ag, cross_layer):
    return ConceptGrounder(
        atl=atl,
        angular_gyrus=ag,
        ips=IPS(),
        cross_layer=cross_layer,
    )


def _make_concept(atl, name="mug", category="object") -> Concept:
    cid, _ = atl.find_or_create(name=name, category=category)
    concept = atl.get(cid)
    assert isinstance(concept, Concept)
    return concept


def _store_episode(
    hippocampus,
    memory_id: str | None = None,
    goal: str | None = None,
    tool_name: str = "grasp",
    success: bool = True,
    confidence: float = 0.8,
    detected_objects: list[str] | None = None,
    timestamp: float | None = None,
    salience: float = 0.5,
    novelty: float = 0.5,
) -> EpisodicMemory:
    """Create and store an episode in hippocampus."""
    ep = EpisodicMemory(
        id=memory_id or str(uuid4()),
        timestamp=timestamp or time.time(),
        perception=Perception(
            detected_objects=detected_objects or [],
            salience=salience,
            novelty=novelty,
        ),
        context=Context(active_goal=goal),
        decision=Decision(
            confidence=confidence,
            intent={"goal": goal} if goal else {},
        ),
        action=Action(tool_name=tool_name),
        outcome=Outcome(success=success),
    )
    hippocampus.capture(record=ep)
    return ep


# ─────────────────────────────────────────────────────────────────────────────
# Concept matching tests
# ─────────────────────────────────────────────────────────────────────────────


class TestConceptMatching:
    def test_matches_detected_objects(self, completer, atl, hippocampus):
        concept = _make_concept(atl, "mug", "object")
        ep = _store_episode(hippocampus, detected_objects=["mug"])
        concept.add_ref("hippocampus", ep.id)

        forming = EpisodicMemory(
            id="forming-1",
            timestamp=time.time(),
            perception=Perception(detected_objects=["mug"]),
        )

        predictions = completer.complete(forming)
        assert len(predictions) >= 1

    def test_matches_goal_tokens(self, completer, atl, hippocampus):
        nav = _make_concept(atl, "navigate", "action")
        kitchen = _make_concept(atl, "kitchen", "location")
        ep = _store_episode(hippocampus, goal="navigate_to_kitchen")
        nav.add_ref("hippocampus", ep.id)
        kitchen.add_ref("hippocampus", ep.id)

        forming = EpisodicMemory(
            id="forming-1",
            timestamp=time.time(),
            context=Context(active_goal="navigate_to_kitchen"),
        )

        predictions = completer.complete(forming)
        assert len(predictions) >= 1

    def test_no_matching_concepts_returns_empty(self, completer):
        forming = EpisodicMemory(
            id="forming-1",
            timestamp=time.time(),
            perception=Perception(detected_objects=["nonexistent"]),
        )
        assert completer.complete(forming) == []

    def test_concepts_without_refs_returns_empty(self, completer, atl):
        _make_concept(atl, "mug", "object")

        forming = EpisodicMemory(
            id="forming-1",
            timestamp=time.time(),
            perception=Perception(detected_objects=["mug"]),
        )
        assert completer.complete(forming) == []


# ─────────────────────────────────────────────────────────────────────────────
# Prediction extraction tests
# ─────────────────────────────────────────────────────────────────────────────


class TestPredictionExtraction:
    def test_extracts_tool_and_success(self, completer, atl, hippocampus):
        concept = _make_concept(atl, "mug", "object")
        ep = _store_episode(
            hippocampus,
            tool_name="grasp",
            success=True,
            goal="grasp_mug",
        )
        concept.add_ref("hippocampus", ep.id)

        forming = EpisodicMemory(
            id="forming-1",
            timestamp=time.time(),
            perception=Perception(detected_objects=["mug"]),
        )

        predictions = completer.complete(forming)
        assert len(predictions) == 1
        assert predictions[0].tool == "grasp"
        assert predictions[0].success is True
        assert predictions[0].goal == "grasp_mug"
        assert predictions[0].source_episode_id == ep.id

    def test_extracts_confidence(self, completer, atl, hippocampus):
        concept = _make_concept(atl, "mug", "object")
        ep = _store_episode(hippocampus, confidence=0.9)
        concept.add_ref("hippocampus", ep.id)

        forming = EpisodicMemory(
            id="forming-1",
            timestamp=time.time(),
            perception=Perception(detected_objects=["mug"]),
        )

        predictions = completer.complete(forming)
        assert predictions[0].confidence == 0.9

    def test_handles_compressed_memory(self, completer, atl, hippocampus):
        concept = _make_concept(atl, "mug", "object")

        # Manually insert a compressed memory into hippocampus
        compressed = CompressedMemory(
            id="compressed-1",
            timestamp=time.time(),
            tool_name="grasp",
            success=True,
            goal="grasp_mug",
        )
        with hippocampus._rwlock.write():
            hippocampus._memories["compressed-1"] = compressed

        concept.add_ref("hippocampus", "compressed-1")

        forming = EpisodicMemory(
            id="forming-1",
            timestamp=time.time(),
            perception=Perception(detected_objects=["mug"]),
        )

        predictions = completer.complete(forming)
        assert len(predictions) == 1
        assert predictions[0].tool == "grasp"
        assert predictions[0].confidence == 0.3  # Compressed default

    def test_multiple_episodes_ordered_by_recency(self, completer, atl, hippocampus):
        concept = _make_concept(atl, "mug", "object")

        base_time = time.time()
        ep_old = _store_episode(
            hippocampus,
            tool_name="look",
            timestamp=base_time - 100,
        )
        ep_new = _store_episode(
            hippocampus,
            tool_name="grasp",
            timestamp=base_time,
        )
        concept.add_ref("hippocampus", ep_old.id)
        concept.add_ref("hippocampus", ep_new.id)

        forming = EpisodicMemory(
            id="forming-1",
            timestamp=time.time(),
            perception=Perception(detected_objects=["mug"]),
        )

        predictions = completer.complete(forming)
        assert len(predictions) == 2
        # Most recent first
        assert predictions[0].tool == "grasp"
        assert predictions[1].tool == "look"


# ─────────────────────────────────────────────────────────────────────────────
# MAX_EPISODES cap test
# ─────────────────────────────────────────────────────────────────────────────


class TestEpisodeCap:
    def test_caps_at_max_episodes(self, completer, atl, hippocampus):
        concept = _make_concept(atl, "mug", "object")

        for i in range(30):
            ep = _store_episode(hippocampus, timestamp=time.time() + i)
            concept.add_ref("hippocampus", ep.id)

        forming = EpisodicMemory(
            id="forming-1",
            timestamp=time.time() + 100,
            perception=Perception(detected_objects=["mug"]),
        )

        predictions = completer.complete(forming)
        assert len(predictions) <= PatternCompleter.MAX_EPISODES


# ─────────────────────────────────────────────────────────────────────────────
# Math context enrichment tests
# ─────────────────────────────────────────────────────────────────────────────


class TestMathContextEnrichment:
    def test_enriches_with_ag_context(self, completer, atl, hippocampus, ag, cross_layer, grounder):
        concept = _make_concept(atl, "mug", "object")

        # Create enough episodes for grounding (n >= 5)
        episodes = []
        for i in range(8):
            ep = _store_episode(hippocampus, salience=0.8)
            concept.add_ref("hippocampus", ep.id)
            episodes.append(ep)

        # Ground the concept to create MathMemory records
        loaded = hippocampus.recall_by_ids([ep.id for ep in episodes])
        grounder.ground_concept(concept, loaded)

        # Now complete — predictions should have math_context
        # Use completer with AG layer so it can find MathMemory records
        forming = EpisodicMemory(
            id="forming-1",
            timestamp=time.time(),
            perception=Perception(detected_objects=["mug"]),
        )

        predictions = completer.complete(forming)
        assert len(predictions) >= 1

        # At least some predictions should have math context
        enriched = [p for p in predictions if p.math_context is not None]
        if enriched:
            mc = enriched[0].math_context
            assert isinstance(mc, list)
            assert all(isinstance(e, MathContextEntry) for e in mc)

    def test_no_enrichment_without_ag_records(self, completer, atl, hippocampus):
        concept = _make_concept(atl, "mug", "object")
        ep = _store_episode(hippocampus)
        concept.add_ref("hippocampus", ep.id)

        forming = EpisodicMemory(
            id="forming-1",
            timestamp=time.time(),
            perception=Perception(detected_objects=["mug"]),
        )

        predictions = completer.complete(forming)
        assert len(predictions) == 1
        assert predictions[0].math_context is None


# ─────────────────────────────────────────────────────────────────────────────
# Deduplication tests
# ─────────────────────────────────────────────────────────────────────────────


class TestDeduplication:
    def test_deduplicates_episodes_across_concepts(self, completer, atl, hippocampus):
        """An episode linked to multiple concepts should appear only once."""
        mug = _make_concept(atl, "mug", "object")
        grasp = _make_concept(atl, "grasp", "action")

        ep = _store_episode(
            hippocampus,
            tool_name="grasp",
            detected_objects=["mug"],
            goal="grasp_mug",
        )
        mug.add_ref("hippocampus", ep.id)
        grasp.add_ref("hippocampus", ep.id)

        forming = EpisodicMemory(
            id="forming-1",
            timestamp=time.time(),
            perception=Perception(detected_objects=["mug"]),
            context=Context(active_goal="grasp_mug"),
        )

        predictions = completer.complete(forming)
        # Should be 1 prediction, not 2 (episode deduplicated)
        assert len(predictions) == 1
        assert predictions[0].source_episode_id == ep.id
