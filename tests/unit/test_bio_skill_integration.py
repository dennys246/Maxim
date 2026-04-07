"""Unit tests for Bio-Skill Integration (ATL Phase A7).

Tests cover:
- A7.0a: to_context_dict() includes record ID for all memory types
- A7.0b: ConceptContextBuilder output includes concept and relationship IDs
- A7.0e: BioContext facade delegates to correct MemoryHub components
- A7.1: Skill name injection via protocol context into perception observations
- A7.2: ConceptExtractor registers skill concepts from _skill_name
- A7.3: Phrase-level compound alias creation for new relationships
- A7.4a: EXECUTES_WITH registered in RelationshipRegistry, inferred for skill_execution
- A7.4b: PatternCompleter discovers skill concepts via EXECUTES_WITH edges
- A7.5: ConceptContextBuilder.rank_available_skills() uses ATL relationship graph
"""

from __future__ import annotations

import time
from typing import Any
from unittest.mock import Mock
from uuid import uuid4

import pytest

from maxim.math.angular_gyrus import AngularGyrus, AngularGyrusConfig
from maxim.math.ips import IPS
from maxim.memory.atl import ATL, ATLConfig
from maxim.memory.concept_context import ConceptContextBuilder
from maxim.memory.concept_extractor import ConceptExtractor
from maxim.memory.concept_grounder import ConceptGrounder
from maxim.memory.cross_layer import CrossLayerGraph
from maxim.memory.hippocampus import Hippocampus, HippocampusConfig
from maxim.memory.pattern_completer import PatternCompleter
from maxim.memory.semantic_types import (
    Concept,
    RelationshipRegistry,
)
from maxim.memory.types import (
    Action,
    CompressedMemory,
    Context,
    Decision,
    EpisodicMemory,
    Outcome,
    Perception,
)
from maxim.skills.base import Skill, SkillResult, SkillState
from maxim.skills.protocol import Protocol, WorkspaceBounds


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
def builder(atl, layers):
    return ConceptContextBuilder(atl=atl, layers=layers)


@pytest.fixture
def grounder(atl, ag, cross_layer):
    return ConceptGrounder(
        atl=atl,
        angular_gyrus=ag,
        ips=IPS(),
        cross_layer=cross_layer,
    )


def _make_episode(
    hippocampus=None,
    memory_id: str | None = None,
    detected_objects: list[str] | None = None,
    detected_people: list[str] | None = None,
    goal: str | None = None,
    tool_name: str = "",
    success: bool = True,
    observations: dict[str, Any] | None = None,
) -> EpisodicMemory:
    ep = EpisodicMemory(
        id=memory_id or str(uuid4()),
        timestamp=time.time(),
        perception=Perception(
            detected_objects=detected_objects or [],
            detected_people=detected_people or [],
            salience=0.5,
            novelty=0.5,
            observations=observations or {},
        ),
        context=Context(active_goal=goal or ""),
        decision=Decision(confidence=0.8),
        action=Action(tool_name=tool_name),
        outcome=Outcome(success=success),
    )
    if hippocampus is not None:
        hippocampus.capture(record=ep)
    return ep


def _make_concept(atl, name="mug", category="object") -> Concept:
    cid, _ = atl.find_or_create(name=name, category=category)
    concept = atl.get(cid)
    assert isinstance(concept, Concept)
    return concept


# ─────────────────────────────────────────────────────────────────────────────
# A7.0a: to_context_dict() includes ID
# ─────────────────────────────────────────────────────────────────────────────


class TestContextDictId:
    def test_episodic_memory_has_id(self):
        ep = _make_episode(memory_id="ep-123")
        d = ep.to_context_dict()
        assert d["id"] == "ep-123"
        assert d["type"] == "episodic"

    def test_compressed_memory_has_id(self):
        ep = _make_episode(memory_id="ep-456", tool_name="grab")
        cm = CompressedMemory.from_episodic(ep)
        d = cm.to_context_dict()
        assert d["id"] == cm.id
        assert d["type"] == "compressed_episodic"

    def test_semantic_memory_has_id(self, atl):
        concept = _make_concept(atl, "table", "object")
        d = concept.to_context_dict()
        assert d["id"] == concept.id
        assert d["type"] == "semantic"

    def test_compressed_semantic_has_id(self):
        from maxim.memory.semantic_types import CompressedSemantic

        cs = CompressedSemantic(
            id="cs-789",
            timestamp=time.time(),
            name="table",
            category="object",
        )
        d = cs.to_context_dict()
        assert d["id"] == "cs-789"
        assert d["type"] == "compressed_semantic"

    def test_id_is_first_key_in_episodic_dict(self):
        """ID should be the first key for easy visual scanning of context."""
        ep = _make_episode(memory_id="ep-order")
        d = ep.to_context_dict()
        assert list(d.keys())[0] == "id"

    def test_id_is_first_key_in_semantic_dict(self, atl):
        concept = _make_concept(atl, "mug", "object")
        d = concept.to_context_dict()
        assert list(d.keys())[0] == "id"


# ─────────────────────────────────────────────────────────────────────────────
# A7.0b: ConceptContextBuilder output includes IDs
# ─────────────────────────────────────────────────────────────────────────────


class TestConceptContextIds:
    def test_build_includes_concept_id(self, builder, atl):
        concept = _make_concept(atl, "mug", "object")
        result = builder.build(detected_objects=["mug"])
        assert len(result) == 1
        assert result[0]["id"] == concept.id
        assert result[0]["name"] == "mug"

    def test_relationship_includes_target_id(self, builder, atl):
        c1 = _make_concept(atl, "kitchen", "location")
        c2 = _make_concept(atl, "mug", "object")
        atl.define_relationship(c1.id, c2.id, "RELATED_TO", weight=0.5)

        result = builder.build(detected_objects=["kitchen"])
        assert len(result) == 1
        rels = result[0]["relationships"]
        assert len(rels) >= 1
        rel = next(r for r in rels if r["target"] == "mug")
        assert rel["id"] == c2.id

    def test_multiple_concepts_each_have_unique_ids(self, builder, atl):
        c1 = _make_concept(atl, "mug", "object")
        c2 = _make_concept(atl, "chair", "object")

        result = builder.build(detected_objects=["mug", "chair"])
        ids = [e["id"] for e in result]
        assert len(ids) == 2
        assert ids[0] != ids[1]
        assert set(ids) == {c1.id, c2.id}


# ─────────────────────────────────────────────────────────────────────────────
# A7.0e: BioContext facade
# ─────────────────────────────────────────────────────────────────────────────


class TestBioContext:
    def test_resolve_concepts_delegates(self, atl, layers):
        from maxim.skills.bio_context import BioContext

        hub = Mock()
        hub._concept_context_builder = ConceptContextBuilder(atl=atl, layers=layers)
        hub.atl = atl
        _make_concept(atl, "chair", "object")

        bio = BioContext(hub)
        result = bio.resolve_concepts(detected_objects=["chair"])
        assert len(result) == 1
        assert result[0]["name"] == "chair"

    def test_resolve_concepts_returns_empty_when_no_builder(self):
        from maxim.skills.bio_context import BioContext

        hub = Mock()
        hub._concept_context_builder = None
        bio = BioContext(hub)
        assert bio.resolve_concepts(detected_objects=["anything"]) == []

    def test_recall_associated_delegates(self, hippocampus):
        from maxim.skills.bio_context import BioContext

        hub = Mock()
        hub.hippocampus = hippocampus

        _make_episode(hippocampus, memory_id="seed-1")  # side-effect: adds to hippocampus
        bio = BioContext(hub)
        result = bio.recall_associated(seed_ids=["seed-1"])
        # Should not raise; returns list (may be empty if no associations)
        assert isinstance(result, list)

    def test_detect_trend(self):
        from maxim.skills.bio_context import BioContext

        hub = Mock()
        bio = BioContext(hub)
        result = bio.detect_trend([1.0, 2.0, 3.0, 4.0, 5.0])
        assert hasattr(result, "direction")

    def test_ground_concept_returns_none_when_no_grounder(self, atl):
        from maxim.skills.bio_context import BioContext

        hub = Mock()
        hub._concept_grounder = None
        hub.atl = atl
        bio = BioContext(hub)
        assert bio.ground_concept("nonexistent") is None

    def test_ground_concept_returns_none_for_missing_concept(self, atl, hippocampus):
        from maxim.skills.bio_context import BioContext

        hub = Mock()
        hub._concept_grounder = Mock()
        hub.atl = atl
        hub.hippocampus = hippocampus
        bio = BioContext(hub)
        assert bio.ground_concept("no-such-id") is None

    def test_complete_pattern_returns_empty_when_no_completer(self):
        from maxim.skills.bio_context import BioContext

        hub = Mock()
        hub._pattern_completer = None
        bio = BioContext(hub)
        ep = _make_episode()
        assert bio.complete_pattern(ep) == []

    def test_detect_anomaly(self):
        from maxim.skills.bio_context import BioContext

        hub = Mock()
        bio = BioContext(hub)
        result = bio.detect_anomaly(100.0, [1.0, 2.0, 3.0, 2.0, 1.5])
        assert hasattr(result, "is_anomalous")

    def test_estimate_mean(self):
        from maxim.skills.bio_context import BioContext

        hub = Mock()
        bio = BioContext(hub)
        result = bio.estimate_mean([10.0, 20.0, 30.0])
        assert hasattr(result, "value")

    def test_resolve_concepts_handles_exception_gracefully(self):
        from maxim.skills.bio_context import BioContext

        hub = Mock()
        hub._concept_context_builder = Mock()
        hub._concept_context_builder.build.side_effect = RuntimeError("boom")
        bio = BioContext(hub)
        assert bio.resolve_concepts(detected_objects=["x"]) == []

    def test_direct_access_properties(self, hippocampus, atl, ag):
        from maxim.skills.bio_context import BioContext

        hub = Mock()
        hub.hippocampus = hippocampus
        hub.atl = atl
        hub.angular_gyrus = ag

        bio = BioContext(hub)
        assert bio.hippocampus is hippocampus
        assert bio.atl is atl
        assert bio.angular_gyrus is ag


# ─────────────────────────────────────────────────────────────────────────────
# A7.0e: BioContext wired via _resolve_bio_systems
# ─────────────────────────────────────────────────────────────────────────────


class StubSkill(Skill):
    def __init__(self, name: str = "stub", deps: list[str] | None = None):
        self._name = name
        self._state = SkillState.IDLE
        self._last_result: SkillResult | None = None
        self._bio: dict[str, Any] = {}
        self._deps = deps or []

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return f"Stub: {self._name}"

    def tools(self) -> list:
        return []

    def can_activate(self, maxim: Any) -> tuple[bool, str]:
        return (True, "")

    def activate(self, maxim: Any, context: dict[str, Any] | None = None) -> SkillResult:
        self._state = SkillState.ACTIVE
        return SkillResult(state=SkillState.ACTIVE, message="ok")

    def deactivate(self) -> SkillResult:
        self._state = SkillState.IDLE
        return SkillResult(state=SkillState.IDLE)

    def bio_dependencies(self) -> list[str]:
        return self._deps


class SimpleProtocol(Protocol):
    def __init__(self, skills: list[Skill] | None = None):
        super().__init__()
        self._skills = skills or []

    @property
    def name(self) -> str:
        return "test_protocol"

    @property
    def description(self) -> str:
        return "test"

    def skills(self) -> list[Skill]:
        return self._skills

    def workspace_bounds(self) -> WorkspaceBounds | None:
        return None


class TestBioContextWiring:
    def test_memory_hub_dependency_returns_bio_context(self):
        from maxim.skills.bio_context import BioContext

        skill = StubSkill("bio_skill", deps=["memory_hub"])
        proto = SimpleProtocol(skills=[skill])

        mock_maxim = Mock()
        mock_maxim._memory_hub = Mock()
        mock_maxim._memory_hub.hippocampus = Mock()
        mock_maxim._memory_hub.atl = Mock()
        mock_maxim._memory_hub.angular_gyrus = Mock()
        mock_maxim._memory_hub.scn = Mock()
        mock_maxim._nac = Mock()

        proto.on_activate(mock_maxim)

        assert isinstance(skill._bio["memory_hub"], BioContext)

    def test_individual_systems_still_resolved_alongside_bio_context(self):
        """Skills can request both memory_hub and individual systems."""
        from maxim.skills.bio_context import BioContext

        skill = StubSkill("combo_skill", deps=["memory_hub", "hippocampus", "nac"])
        proto = SimpleProtocol(skills=[skill])

        mock_maxim = Mock()
        mock_maxim._memory_hub = Mock()
        mock_maxim._memory_hub.hippocampus = Mock()
        mock_maxim._memory_hub.atl = Mock()
        mock_maxim._memory_hub.angular_gyrus = Mock()
        mock_maxim._memory_hub.scn = Mock()
        mock_maxim._nac = Mock()

        proto.on_activate(mock_maxim)

        assert isinstance(skill._bio["memory_hub"], BioContext)
        assert skill._bio["hippocampus"] is mock_maxim._memory_hub.hippocampus
        assert skill._bio["nac"] is mock_maxim._nac

    def test_no_memory_hub_still_resolves_other_systems(self):
        """Without MemoryHub, skills still get NAc and IPS."""
        skill = StubSkill("limited_skill", deps=["nac", "ips"])
        proto = SimpleProtocol(skills=[skill])

        mock_maxim = Mock(spec=[])  # No attributes by default
        mock_maxim._nac = Mock()

        proto.on_activate(mock_maxim)

        assert skill._bio["nac"] is mock_maxim._nac
        assert skill._bio.get("memory_hub") is None


# ─────────────────────────────────────────────────────────────────────────────
# A7.1: Skill name injection
# ─────────────────────────────────────────────────────────────────────────────


class TestSkillNameInjection:
    def test_active_skill_name_stored_in_context(self):
        skill = StubSkill("patrol_monitor")
        proto = SimpleProtocol(skills=[skill])

        mock_maxim = Mock()
        mock_maxim._memory_hub = None
        mock_maxim._nac = None

        proto.on_activate(mock_maxim)

        assert proto._context.get("_active_skill_name") == "patrol_monitor"

    def test_last_activated_skill_wins(self):
        s1 = StubSkill("first_skill")
        s2 = StubSkill("second_skill")
        proto = SimpleProtocol(skills=[s1, s2])

        mock_maxim = Mock()
        mock_maxim._memory_hub = None
        mock_maxim._nac = None

        proto.on_activate(mock_maxim)

        assert proto._context.get("_active_skill_name") == "second_skill"

    def test_memory_agent_injects_skill_name_from_registry(self):
        """MemoryAgent reads active protocol and injects _skill_name into observations."""
        from maxim.agents.bus import Percept
        from maxim.agents.memory_agent import MemoryAgent

        bus = Mock()
        bus.publish = Mock()
        ma = MemoryAgent(bus, max_short_term=10, max_long_term=50)

        # Set up a mock protocol registry with an active protocol
        proto = SimpleProtocol(skills=[StubSkill("patrol_monitor")])
        mock_maxim = Mock()
        mock_maxim._memory_hub = None
        mock_maxim._nac = None
        proto.on_activate(mock_maxim)

        mock_registry = Mock()
        mock_registry.get_active.return_value = [proto]
        ma._protocol_registry = mock_registry

        percept = Percept(
            timestamp=time.time(),
            source="vision",
            detections=[{"label": "person"}],
            salience=0.5,
            novelty=0.5,
        )

        entry = ma._begin_memory_formation(percept, run_id="test-run-12345678")
        assert entry.record.perception.observations.get("_skill_name") == "patrol_monitor"

    def test_memory_agent_no_injection_without_registry(self):
        """Without _protocol_registry, no _skill_name is injected."""
        from maxim.agents.bus import Percept
        from maxim.agents.memory_agent import MemoryAgent

        bus = Mock()
        bus.publish = Mock()
        ma = MemoryAgent(bus, max_short_term=10, max_long_term=50)

        percept = Percept(
            timestamp=time.time(),
            source="vision",
            detections=[{"label": "person"}],
            salience=0.5,
            novelty=0.5,
        )

        entry = ma._begin_memory_formation(percept, run_id="test-run-12345678")
        assert "_skill_name" not in entry.record.perception.observations

    def test_memory_agent_no_injection_without_active_protocols(self):
        """With registry but no active protocols, no _skill_name is injected."""
        from maxim.agents.bus import Percept
        from maxim.agents.memory_agent import MemoryAgent

        bus = Mock()
        bus.publish = Mock()
        ma = MemoryAgent(bus, max_short_term=10, max_long_term=50)

        mock_registry = Mock()
        mock_registry.get_active.return_value = []
        ma._protocol_registry = mock_registry

        percept = Percept(
            timestamp=time.time(),
            source="vision",
            salience=0.5,
            novelty=0.5,
        )

        entry = ma._begin_memory_formation(percept, run_id="test-run-12345678")
        assert "_skill_name" not in entry.record.perception.observations


# ─────────────────────────────────────────────────────────────────────────────
# A7.2: Skill-memory auto-association
# ─────────────────────────────────────────────────────────────────────────────


class TestSkillAutoAssociation:
    def test_skill_concept_created_from_observation(self, atl, hippocampus, cross_layer):
        from maxim.time.scn import SCN

        scn = SCN()
        extractor = ConceptExtractor(atl=atl, cross_layer=cross_layer, scn=scn)

        ep = _make_episode(
            hippocampus,
            detected_objects=["person"],
            observations={"_skill_name": "patrol_monitor"},
        )

        extractor._process_capture(ep.id, ep)

        # Should have created a "skill:patrol_monitor" concept
        results = atl.recall(limit=10, category="skill_execution")
        skill_concepts = [c for c in results if isinstance(c, Concept)]
        assert any(c.name == "skill:patrol_monitor" for c in skill_concepts)

    def test_no_skill_concept_without_observation(self, atl, hippocampus, cross_layer):
        from maxim.time.scn import SCN

        scn = SCN()
        extractor = ConceptExtractor(atl=atl, cross_layer=cross_layer, scn=scn)

        ep = _make_episode(hippocampus, detected_objects=["person"])

        extractor._process_capture(ep.id, ep)

        results = atl.recall(limit=10, category="skill_execution")
        assert len(results) == 0

    def test_skill_concept_reused_across_episodes(self, atl, hippocampus, cross_layer):
        """Same skill name in two episodes should reuse the concept, not duplicate."""
        from maxim.time.scn import SCN

        scn = SCN()
        extractor = ConceptExtractor(atl=atl, cross_layer=cross_layer, scn=scn)

        ep1 = _make_episode(
            hippocampus,
            detected_objects=["person"],
            observations={"_skill_name": "patrol_monitor"},
        )
        ep2 = _make_episode(
            hippocampus,
            detected_objects=["vehicle"],
            observations={"_skill_name": "patrol_monitor"},
        )

        extractor._process_capture(ep1.id, ep1)
        extractor._process_capture(ep2.id, ep2)

        results = atl.recall(limit=10, category="skill_execution")
        patrol_concepts = [c for c in results if c.name == "skill:patrol_monitor"]
        assert len(patrol_concepts) == 1, "Skill concept should be reused, not duplicated"

    def test_skill_concept_forms_executes_with_relationships(self, atl, hippocampus, cross_layer):
        """Skill concept should form EXECUTES_WITH edges to co-occurring concepts."""
        from maxim.time.scn import SCN

        scn = SCN()
        extractor = ConceptExtractor(atl=atl, cross_layer=cross_layer, scn=scn)

        ep = _make_episode(
            hippocampus,
            detected_objects=["person"],
            goal="watch area",
            observations={"_skill_name": "patrol_monitor"},
        )

        extractor._process_capture(ep.id, ep)

        # Find the skill concept
        results = atl.recall(limit=10, category="skill_execution")
        skill_concepts = [c for c in results if isinstance(c, Concept) and c.name == "skill:patrol_monitor"]
        assert len(skill_concepts) == 1

        # Check it has EXECUTES_WITH relationships
        rels = atl.find_by_relationship(
            skill_concepts[0].id,
            rel_type="EXECUTES_WITH",
            direction="both",
            limit=20,
        )
        assert len(rels) > 0, "Skill concept should have EXECUTES_WITH edges"


# ─────────────────────────────────────────────────────────────────────────────
# A7.3: Phrase-level concept indexing
# ─────────────────────────────────────────────────────────────────────────────


class TestPhraseConceptIndexing:
    def test_compound_alias_created_for_new_relationship(self, atl, hippocampus, cross_layer):
        from maxim.time.scn import SCN

        scn = SCN()
        extractor = ConceptExtractor(atl=atl, cross_layer=cross_layer, scn=scn)

        ep = _make_episode(
            hippocampus,
            detected_objects=["mug"],
            goal="navigate",
        )

        extractor._process_capture(ep.id, ep)

        # Should have created compound aliases for new relationships
        aliases = atl.recall(limit=20, category="compound_alias")
        assert len(aliases) > 0, "Expected compound aliases to be created"
        alias_names = [a.name for a in aliases]
        # At least one compound should exist between the concepts
        assert any(" " in name for name in alias_names)

    def test_alias_has_alias_of_relationships(self, atl, hippocampus, cross_layer):
        """Compound aliases should have ALIAS_OF edges to their component concepts."""
        from maxim.time.scn import SCN

        scn = SCN()
        extractor = ConceptExtractor(atl=atl, cross_layer=cross_layer, scn=scn)

        ep = _make_episode(
            hippocampus,
            detected_objects=["mug"],
            goal="navigate",
        )

        extractor._process_capture(ep.id, ep)

        aliases = atl.recall(limit=20, category="compound_alias")
        assert len(aliases) > 0

        for alias in aliases:
            if isinstance(alias, Concept):
                rels = atl.find_by_relationship(
                    alias.id,
                    rel_type="ALIAS_OF",
                    direction="outgoing",
                    limit=10,
                )
                assert len(rels) == 2, f"Alias '{alias.name}' should link to exactly 2 components"

    def test_no_duplicate_aliases_on_reinforcement(self, atl, hippocampus, cross_layer):
        """Reinforcing an existing relationship should NOT create duplicate aliases."""
        from maxim.time.scn import SCN

        scn = SCN()
        extractor = ConceptExtractor(atl=atl, cross_layer=cross_layer, scn=scn)

        ep1 = _make_episode(
            hippocampus,
            detected_objects=["mug"],
            goal="navigate",
        )
        ep2 = _make_episode(
            hippocampus,
            detected_objects=["mug"],
            goal="navigate",
        )

        extractor._process_capture(ep1.id, ep1)
        alias_count_after_first = len(atl.recall(limit=50, category="compound_alias"))

        extractor._process_capture(ep2.id, ep2)
        alias_count_after_second = len(atl.recall(limit=50, category="compound_alias"))

        # Reinforcement should not create new aliases
        assert alias_count_after_second == alias_count_after_first


# ─────────────────────────────────────────────────────────────────────────────
# A7.4a: EXECUTES_WITH relationship type
# ─────────────────────────────────────────────────────────────────────────────


class TestExecutesWithRelationship:
    def test_executes_with_registered(self):
        registry = RelationshipRegistry()
        assert "EXECUTES_WITH" in registry._types

    def test_alias_of_registered(self):
        registry = RelationshipRegistry()
        assert "ALIAS_OF" in registry._types

    def test_infer_skill_execution_returns_executes_with(self):
        result = ConceptExtractor._infer_relationship_type("skill_execution", "object")
        assert result == "EXECUTES_WITH"

    def test_infer_skill_execution_symmetric(self):
        result = ConceptExtractor._infer_relationship_type("object", "skill_execution")
        assert result == "EXECUTES_WITH"


# ─────────────────────────────────────────────────────────────────────────────
# A7.4b: PatternCompleter skill prediction
# ─────────────────────────────────────────────────────────────────────────────


class TestPatternCompleterSkillPrediction:
    def test_skill_concepts_discovered_via_executes_with(self, atl, hippocampus, ag):
        layers = {"hippocampus": hippocampus, "angular_gyrus": ag}
        completer = PatternCompleter(atl=atl, layers=layers)

        # Create an object concept and a skill concept with EXECUTES_WITH
        obj_concept = _make_concept(atl, "person", "object")
        skill_concept = _make_concept(atl, "skill:patrol", "skill_execution")
        atl.define_relationship(
            skill_concept.id,
            obj_concept.id,
            "EXECUTES_WITH",
            weight=0.8,
        )

        # Create an episode linked to the object concept
        ep = _make_episode(
            hippocampus,
            detected_objects=["person"],
            tool_name="observe",
            success=True,
        )
        # Link episode to concept
        obj_concept.add_ref("hippocampus", ep.id)
        skill_concept.add_ref("hippocampus", ep.id)

        # New episode for pattern completion
        new_ep = _make_episode(detected_objects=["person"])

        predictions = completer.complete(new_ep)
        # Should find predictions via the skill concept's EXECUTES_WITH edge
        assert len(predictions) >= 1

    def test_no_skill_concepts_when_no_executes_with_edges(self, atl, hippocampus, ag):
        """Skill concepts without EXECUTES_WITH edges to matched concepts are ignored."""
        layers = {"hippocampus": hippocampus, "angular_gyrus": ag}
        completer = PatternCompleter(atl=atl, layers=layers)

        # Object concept with an episode
        obj_concept = _make_concept(atl, "person", "object")
        ep = _make_episode(hippocampus, detected_objects=["person"], tool_name="observe")
        obj_concept.add_ref("hippocampus", ep.id)

        # Skill concept exists but has no EXECUTES_WITH to person
        _make_concept(atl, "skill:unrelated", "skill_execution")

        new_ep = _make_episode(detected_objects=["person"])
        predictions = completer.complete(new_ep)

        # Should still get predictions from the object concept's episodes
        # but no additional contributions from the unrelated skill
        assert isinstance(predictions, list)

    def test_skill_discovery_skipped_when_no_skill_concepts_exist(self, atl, hippocampus, ag):
        """When ATL has no skill_execution concepts, complete() still works."""
        layers = {"hippocampus": hippocampus, "angular_gyrus": ag}
        completer = PatternCompleter(atl=atl, layers=layers)

        # Object concept with episode but no skill concepts
        obj = _make_concept(atl, "person", "object")
        ep = _make_episode(hippocampus, detected_objects=["person"], tool_name="look")
        obj.add_ref("hippocampus", ep.id)

        new_ep = _make_episode(detected_objects=["person"])
        predictions = completer.complete(new_ep)
        # Should still produce predictions from the object concept
        assert isinstance(predictions, list)


# ─────────────────────────────────────────────────────────────────────────────
# A7.5: Concept-driven skill discovery
# ─────────────────────────────────────────────────────────────────────────────


class TestSkillDiscovery:
    def test_rank_available_skills_empty_registry(self, builder, atl):
        concept = _make_concept(atl, "person", "object")
        ranked = builder.rank_available_skills([concept])
        assert ranked == []

    def test_rank_available_skills_finds_matching(self, builder, atl):
        # Register a mock skill
        mock_skill = Mock()
        mock_skill.name = "patrol"
        builder.set_skill_registry([mock_skill])

        # Create concepts
        obj_concept = _make_concept(atl, "person", "object")
        skill_concept = _make_concept(atl, "skill:patrol", "skill_execution")

        # Create EXECUTES_WITH edge
        atl.define_relationship(
            skill_concept.id,
            obj_concept.id,
            "EXECUTES_WITH",
            weight=0.8,
        )

        ranked = builder.rank_available_skills([obj_concept])
        assert len(ranked) == 1
        assert ranked[0]["name"] == "patrol"
        assert ranked[0]["relevance"] == 1
        assert ranked[0]["skill_concept_id"] == skill_concept.id
        assert len(ranked[0]["matching_concepts"]) == 1
        assert ranked[0]["matching_concepts"][0]["id"] == obj_concept.id

    def test_rank_skills_sorted_by_relevance(self, builder, atl):
        # Two skills, one with more matching concepts
        s1 = Mock()
        s1.name = "patrol"
        s2 = Mock()
        s2.name = "grab"
        builder.set_skill_registry([s1, s2])

        c_person = _make_concept(atl, "person", "object")
        c_vehicle = _make_concept(atl, "vehicle", "object")

        sk_patrol = _make_concept(atl, "skill:patrol", "skill_execution")
        sk_grab = _make_concept(atl, "skill:grab", "skill_execution")

        # patrol has 2 edges, grab has 1
        atl.define_relationship(sk_patrol.id, c_person.id, "EXECUTES_WITH", weight=0.8)
        atl.define_relationship(sk_patrol.id, c_vehicle.id, "EXECUTES_WITH", weight=0.7)
        atl.define_relationship(sk_grab.id, c_person.id, "EXECUTES_WITH", weight=0.6)

        ranked = builder.rank_available_skills([c_person, c_vehicle])
        assert len(ranked) == 2
        assert ranked[0]["name"] == "patrol"
        assert ranked[0]["relevance"] == 2
        assert ranked[1]["name"] == "grab"
        assert ranked[1]["relevance"] == 1

    def test_skill_without_edges_not_ranked(self, builder, atl):
        mock_skill = Mock()
        mock_skill.name = "idle"
        builder.set_skill_registry([mock_skill])

        _make_concept(atl, "skill:idle", "skill_execution")
        obj = _make_concept(atl, "chair", "object")

        # No EXECUTES_WITH edge
        ranked = builder.rank_available_skills([obj])
        assert ranked == []

    def test_skill_concept_without_registered_skill_not_ranked(self, builder, atl):
        """ATL has skill:foo concept, but no 'foo' skill in registry."""
        mock_skill = Mock()
        mock_skill.name = "bar"
        builder.set_skill_registry([mock_skill])

        obj = _make_concept(atl, "person", "object")
        sk = _make_concept(atl, "skill:foo", "skill_execution")
        atl.define_relationship(sk.id, obj.id, "EXECUTES_WITH", weight=0.8)

        ranked = builder.rank_available_skills([obj])
        assert ranked == [], "skill:foo has no matching skill in registry"

    def test_rank_includes_past_success_rate(self, builder, atl):
        """past_success_rate should reflect the skill concept's confidence."""
        mock_skill = Mock()
        mock_skill.name = "patrol"
        builder.set_skill_registry([mock_skill])

        obj = _make_concept(atl, "person", "object")
        sk = _make_concept(atl, "skill:patrol", "skill_execution")
        atl.define_relationship(sk.id, obj.id, "EXECUTES_WITH", weight=0.8)

        ranked = builder.rank_available_skills([obj])
        assert len(ranked) == 1
        assert ranked[0]["past_success_rate"] == sk.confidence

    def test_rank_with_no_skill_concepts_in_atl(self, builder, atl):
        """When ATL has no skill concepts, ranking returns empty."""
        mock_skill = Mock()
        mock_skill.name = "patrol"
        builder.set_skill_registry([mock_skill])

        obj = _make_concept(atl, "person", "object")
        ranked = builder.rank_available_skills([obj])
        assert ranked == []

    def test_rank_with_empty_matched_concepts(self, builder, atl):
        """Empty matched concepts list means nothing to match against."""
        mock_skill = Mock()
        mock_skill.name = "patrol"
        builder.set_skill_registry([mock_skill])

        _make_concept(atl, "skill:patrol", "skill_execution")  # side-effect: adds to ATL

        ranked = builder.rank_available_skills([])
        assert ranked == []
