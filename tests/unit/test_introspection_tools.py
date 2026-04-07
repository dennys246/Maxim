"""Tests for introspection tools (biological self-awareness)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock


from maxim.tools.introspection import (
    CausalLinksTool,
    ConceptQueryTool,
    EnergyStatusTool,
    MemoryRecallTool,
    PainHistoryTool,
    PredictOutcomeTool,
    SceneSummaryTool,
    SimilaritySearchTool,
    SystemStatsTool,
    TemporalPatternsTool,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fakes
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class FakeOutcome:
    success: bool = True
    result: Any = None
    error: str | None = None


@dataclass
class FakeAction:
    tool_name: str = "grab"


@dataclass
class FakePerception:
    salience: float = 0.7
    novelty: float = 0.5
    detected_objects: list[str] = field(default_factory=lambda: ["cup"])


@dataclass
class FakeContext:
    active_goal: str = "pick up cup"
    active_mode: str = "active"


@dataclass
class FakeMemory:
    id: str = "mem-001"
    timestamp: float = 0.0
    perception: FakePerception = field(default_factory=FakePerception)
    context: FakeContext = field(default_factory=FakeContext)
    action: FakeAction = field(default_factory=FakeAction)
    outcome: FakeOutcome = field(default_factory=FakeOutcome)


@dataclass
class FakeValence:
    value: str = "positive"


@dataclass
class FakePrediction:
    predicted_outcome: str = "success"
    predicted_valence: FakeValence = field(default_factory=FakeValence)
    predicted_value: float = 0.8
    confidence: float = 0.9
    predicted_delay: float = 2.0
    delay_bounds: tuple[float, float] = (1.0, 3.0)
    context_match: float = 0.85


@dataclass
class FakeTemporalDelta:
    observed_deltas: tuple[float, ...] = (1.0, 2.0)

    @property
    def mean(self):
        return sum(self.observed_deltas) / len(self.observed_deltas)


@dataclass
class FakeCausalLink:
    id: str = "link-001"
    event_signature: str = "tool:grab"
    outcome_signature: str = "result:success"
    outcome_valence: FakeValence = field(default_factory=FakeValence)
    predicted_value: float = 0.8
    confidence: float = 0.9
    observation_count: int = 5
    temporal_delta: FakeTemporalDelta = field(default_factory=FakeTemporalDelta)
    last_rpe: float | None = 0.3
    memory_ids: list[str] = field(default_factory=lambda: ["mem-001", "mem-002"])


@dataclass
class FakeConcept:
    id: str = "concept-001"
    name: str = "cup"
    category: str = "object"
    confidence: float = 0.85
    definition: str = "A drinking vessel"
    access_count: int = 10


@dataclass
class FakeRelationship:
    relationship_type: str = "IS_A"
    weight: float = 0.9
    confidence: float = 0.8


# ─────────────────────────────────────────────────────────────────────────────
# MemoryRecallTool
# ─────────────────────────────────────────────────────────────────────────────


class TestMemoryRecallTool:
    def test_graceful_without_hippocampus(self):
        tool = MemoryRecallTool(hippocampus=None)
        result = tool.execute()
        assert result.success
        assert result.output["available"] is False

    def test_basic_recall(self):
        hip = MagicMock()
        hip.recall.return_value = [FakeMemory(), FakeMemory(id="mem-002")]
        tool = MemoryRecallTool(hippocampus=hip)
        result = tool.execute(limit=5)
        assert result.success
        assert result.output["count"] == 2
        assert result.output["memories"][0]["id"] == "mem-001"
        assert result.output["memories"][0]["tool"] == "grab"
        assert result.output["memories"][0]["success"] is True

    def test_filters_passed_through(self):
        hip = MagicMock()
        hip.recall.return_value = []
        tool = MemoryRecallTool(hippocampus=hip)
        tool.execute(tool_name="grab", success=True, object="cup")
        hip.recall.assert_called_once_with(limit=5, tool="grab", success=True, object_detected="cup")

    def test_expand_uses_spreading_activation(self):
        mem = FakeMemory()
        assoc_mem = FakeMemory(id="assoc-001")
        hip = MagicMock()
        hip.recall.return_value = [mem]
        hip.recall_associated.return_value = [(assoc_mem, 0.7)]
        tool = MemoryRecallTool(hippocampus=hip)
        result = tool.execute(expand=True)
        assert result.success
        assert result.output["expanded"] is True
        assert result.output["count"] == 2
        hip.recall_associated.assert_called_once()

    def test_reflection_memory_formatted(self):
        mem = FakeMemory(outcome=FakeOutcome(success=False, result={"reflection": "tool timed out"}))
        hip = MagicMock()
        hip.recall.return_value = [mem]
        tool = MemoryRecallTool(hippocampus=hip)
        result = tool.execute()
        assert result.output["memories"][0]["reflection"] == "tool timed out"

    def test_limit_capped_at_20(self):
        hip = MagicMock()
        hip.recall.return_value = []
        tool = MemoryRecallTool(hippocampus=hip)
        tool.execute(limit=100)
        hip.recall.assert_called_once_with(limit=20)


# ─────────────────────────────────────────────────────────────────────────────
# PredictOutcomeTool
# ─────────────────────────────────────────────────────────────────────────────


class TestPredictOutcomeTool:
    def test_graceful_without_nac(self):
        tool = PredictOutcomeTool(nac=None)
        result = tool.execute(tool_name="grab")
        assert result.success
        assert result.output["available"] is False

    def test_prediction_returned(self):
        nac = MagicMock()
        nac.predict.return_value = FakePrediction()
        nac.predict_all_outcomes.return_value = [FakePrediction()]
        tool = PredictOutcomeTool(nac=nac)
        result = tool.execute(tool_name="grab")
        assert result.success
        pred = result.output["prediction"]
        assert pred["valence"] == "positive"
        assert pred["value"] == 0.8
        assert pred["confidence"] == 0.9

    def test_no_prediction_yet(self):
        nac = MagicMock()
        nac.predict.return_value = None
        nac.predict_all_outcomes.return_value = []
        tool = PredictOutcomeTool(nac=nac)
        result = tool.execute(tool_name="new_tool")
        assert result.success
        assert result.output["prediction"] is None
        assert "No learned predictions" in result.output["message"]


# ─────────────────────────────────────────────────────────────────────────────
# CausalLinksTool
# ─────────────────────────────────────────────────────────────────────────────


class TestCausalLinksTool:
    def test_graceful_without_nac(self):
        tool = CausalLinksTool(nac=None)
        result = tool.execute()
        assert result.success
        assert result.output["available"] is False

    def test_query_by_event(self):
        nac = MagicMock()
        nac.get_links_for_event.return_value = [FakeCausalLink()]
        tool = CausalLinksTool(nac=nac)
        result = tool.execute(event="tool:grab")
        assert result.success
        assert result.output["count"] == 1
        link = result.output["links"][0]
        assert link["event"] == "tool:grab"
        assert link["valence"] == "positive"
        assert link["confidence"] == 0.9

    def test_query_by_valence_filter(self):
        positive = FakeCausalLink(outcome_valence=FakeValence("positive"))
        negative = FakeCausalLink(id="link-002", outcome_valence=FakeValence("negative"))
        nac = MagicMock()
        nac.get_links_for_event.return_value = [positive, negative]
        tool = CausalLinksTool(nac=nac)
        result = tool.execute(event="tool:grab", valence="negative")
        assert result.output["count"] == 1
        assert result.output["links"][0]["valence"] == "negative"

    def test_query_by_memory_id(self):
        nac = MagicMock()
        nac.get_links_for.return_value = [FakeCausalLink()]
        tool = CausalLinksTool(nac=nac)
        tool.execute(memory_id="mem-001")  # verify delegation to nac
        nac.get_links_for.assert_called_once_with("mem-001")


# ─────────────────────────────────────────────────────────────────────────────
# PainHistoryTool
# ─────────────────────────────────────────────────────────────────────────────


class TestPainHistoryTool:
    def test_graceful_without_subsystems(self):
        tool = PainHistoryTool(pain_detector=None, fear_agent=None)
        result = tool.execute()
        assert result.success
        assert result.output["pain_stats"]["available"] is False

    def test_pain_stats_returned(self):
        pain = MagicMock()
        pain.get_stats.return_value = {"total_signals": 5, "tool_failure": 3}
        tool = PainHistoryTool(pain_detector=pain)
        result = tool.execute()
        assert result.output["pain_stats"]["total_signals"] == 5

    def test_fear_gate_check(self):
        fear = MagicMock()
        fear.review_action.return_value = MagicMock(allow=True, risk="low", findings=[])
        tool = PainHistoryTool(fear_agent=fear)
        result = tool.execute(check_action="bash", action_params={"command": "ls"})
        assert result.output["fear_check"]["allowed"] is True


# ─────────────────────────────────────────────────────────────────────────────
# TemporalPatternsTool
# ─────────────────────────────────────────────────────────────────────────────


class TestTemporalPatternsTool:
    def test_graceful_without_scn(self):
        tool = TemporalPatternsTool(scn=None)
        result = tool.execute()
        assert result.success
        assert result.output["available"] is False

    def test_query_by_hour(self):
        scn = MagicMock()
        scn.query_intersection.return_value = {"mem-1", "mem-2", "mem-3"}
        tool = TemporalPatternsTool(scn=scn)
        result = tool.execute(hour=9)
        assert result.output["total_matches"] == 3

    def test_discover_rhythms(self):
        scn = MagicMock()
        scn.find_rhythmic_patterns.return_value = {
            "hourly": [(9, 5), (14, 3)],
        }
        tool = TemporalPatternsTool(scn=scn)
        result = tool.execute(discover_rhythms=True)
        assert "hourly" in result.output["rhythmic_patterns"]


# ─────────────────────────────────────────────────────────────────────────────
# EnergyStatusTool
# ─────────────────────────────────────────────────────────────────────────────


class TestEnergyStatusTool:
    def test_graceful_without_tracker(self):
        tool = EnergyStatusTool(energy_tracker=None)
        result = tool.execute()
        assert result.success
        assert result.output["available"] is False

    def test_returns_stats(self):
        tracker = MagicMock()
        tracker.get_window_stats.return_value = {"count": 10, "total_energy": 500}
        tracker.get_total_stats.return_value = {"total_count": 100, "total_energy": 5000}
        tool = EnergyStatusTool(energy_tracker=tracker)
        result = tool.execute(window_seconds=60.0)
        assert result.output["recent"]["count"] == 10
        assert result.output["total"]["total_count"] == 100


# ─────────────────────────────────────────────────────────────────────────────
# ConceptQueryTool
# ─────────────────────────────────────────────────────────────────────────────


class TestConceptQueryTool:
    def test_graceful_without_atl(self):
        tool = ConceptQueryTool(atl=None)
        result = tool.execute()
        assert result.success
        assert result.output["available"] is False

    def test_search_by_name(self):
        atl = MagicMock()
        atl.recall.return_value = [FakeConcept()]
        tool = ConceptQueryTool(atl=atl)
        result = tool.execute(name="cup")
        assert result.output["count"] == 1
        assert result.output["concepts"][0]["name"] == "cup"

    def test_relationship_query(self):
        atl = MagicMock()
        atl.find_by_relationship.return_value = [("concept-002", FakeRelationship())]
        tool = ConceptQueryTool(atl=atl)
        result = tool.execute(concept_id="concept-001", relationship_type="IS_A")
        assert len(result.output["relationships"]) == 1
        assert result.output["relationships"][0]["type"] == "IS_A"


# ─────────────────────────────────────────────────────────────────────────────
# SceneSummaryTool
# ─────────────────────────────────────────────────────────────────────────────


class TestSceneSummaryTool:
    def test_graceful_in_headless(self):
        tool = SceneSummaryTool(salience_network=None, attention_network=None)
        result = tool.execute()
        assert result.success
        assert result.output["available"] is False

    def test_returns_salient_objects(self):
        salience = MagicMock()
        salience.get_top_salient.return_value = [
            {"label": "cup", "salience": 0.9, "novelty": 0.7},
        ]
        tool = SceneSummaryTool(salience_network=salience)
        result = tool.execute(top_n=3)
        assert len(result.output["salient_objects"]) == 1


# ─────────────────────────────────────────────────────────────────────────────
# SimilaritySearchTool
# ─────────────────────────────────────────────────────────────────────────────


class TestSimilaritySearchTool:
    def test_graceful_without_ec(self):
        tool = SimilaritySearchTool(ec=None)
        result = tool.execute()
        assert result.success
        assert result.output["available"] is False

    def test_search_by_tool(self):
        ec = MagicMock()
        ec.find_similar_situations_for_action.return_value = [("mem-1", "sig1"), ("mem-2", "sig2")]
        tool = SimilaritySearchTool(ec=ec)
        result = tool.execute(tool_name="grab")
        assert result.output["count"] == 2

    def test_search_by_memory_id(self):
        ec = MagicMock()
        ec.find_similar_by_memory.return_value = [("mem-2", 0.85)]
        tool = SimilaritySearchTool(ec=ec)
        result = tool.execute(memory_id="mem-1")
        assert result.output["similar"][0]["similarity"] == 0.85

    def test_no_params_returns_message(self):
        ec = MagicMock()
        tool = SimilaritySearchTool(ec=ec)
        result = tool.execute()
        assert "Provide either" in result.output["message"]


# ─────────────────────────────────────────────────────────────────────────────
# SystemStatsTool
# ─────────────────────────────────────────────────────────────────────────────


class TestSystemStatsTool:
    def test_works_with_no_subsystems(self):
        tool = SystemStatsTool()
        result = tool.execute()
        assert result.success
        for key in ("hippocampus", "nac", "ec", "atl", "energy", "pain"):
            assert result.output[key]["available"] is False

    def test_aggregates_available_stats(self):
        hip = MagicMock()
        hip.stats.return_value = {"memories_captured": 100}
        nac = MagicMock()
        nac.stats.return_value = {"total_links": 50}
        tool = SystemStatsTool(hippocampus=hip, nac=nac)
        result = tool.execute()
        assert result.output["hippocampus"]["memories_captured"] == 100
        assert result.output["nac"]["total_links"] == 50
        assert result.output["ec"]["available"] is False

    def test_handles_subsystem_errors(self):
        hip = MagicMock()
        hip.stats.side_effect = RuntimeError("db corrupted")
        tool = SystemStatsTool(hippocampus=hip)
        result = tool.execute()
        assert "error" in result.output["hippocampus"]
