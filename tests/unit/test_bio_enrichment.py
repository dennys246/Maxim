"""Tests for BioEnrichmentPipeline (L0).

Covers:
- Novelty gating (gate bypassed for think, respected for percepts)
- Hippocampus query integration
- NAc prediction surfacing
- ComponentIndex affordance matching
- Valence computation
- Thought response formatting
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from maxim.integration.bio_enrichment import (
    BioEnrichmentPipeline,
    CausalPrediction,
    ConceptLink,
    EnrichmentContext,
    EnrichmentResult,
    EpisodicSummary,
)
from maxim.runtime.gating import GateScore, GatingContext, TextSalienceScorer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_scorer(novelty: float = 1.0, salience: float = 0.5) -> MagicMock:
    """Create a mock TextSalienceScorer that returns fixed scores."""
    scorer = MagicMock(spec=TextSalienceScorer)
    scorer.score.return_value = GateScore(
        novelty=novelty,
        salience=salience,
        combined=novelty * salience,
    )
    return scorer


def _make_hippocampus(memories: list | None = None) -> MagicMock:
    hippo = MagicMock()
    if memories is None:
        # Default: one memory with a tool action
        mem = MagicMock()
        mem.id = "mem_1"
        mem.tool_name = "rusty_sword_slash"
        mem.success = True
        mem.valence = 0.5
        mem.action = MagicMock()
        mem.action.tool_name = "rusty_sword_slash"
        mem.decision = MagicMock()
        mem.decision.intent = {"goal": "test combat"}
        mem.outcome = MagicMock()
        mem.outcome.result = "Hit the target"
        mem.outcome.success = True
        memories = [mem]
    hippo.search_by_content.return_value = memories
    return hippo


def _make_nac(predictions: list | None = None) -> MagicMock:
    from maxim.decisions.causal_link import Valence

    nac = MagicMock()
    if predictions is None:
        link = MagicMock()
        link.event_signature = "rusty_sword_slash"
        link.outcome_signature = "target_hit"
        link.confidence = 0.8
        link.outcome_valence = Valence.POSITIVE
        nac.get_links_for_event.return_value = [link]
    else:
        nac.get_links_for_event.return_value = predictions
    return nac


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBioEnrichmentPipeline:
    def test_empty_text_returns_none(self):
        pipeline = BioEnrichmentPipeline()
        assert pipeline.enrich("") is None
        assert pipeline.enrich("   ") is None

    def test_bypass_gate_always_enriches(self):
        scorer = _make_scorer(novelty=0.1, salience=0.1)  # below threshold
        pipeline = BioEnrichmentPipeline(scorer=scorer, novelty_threshold=0.5)
        # Without bypass: should be gated
        result = pipeline.enrich("boring text")
        assert result is None
        # With bypass: should enrich
        result = pipeline.enrich("boring text", bypass_gate=True)
        assert result is not None

    def test_novel_text_passes_gate(self):
        scorer = _make_scorer(novelty=0.9, salience=0.8)
        pipeline = BioEnrichmentPipeline(scorer=scorer, novelty_threshold=0.3)
        result = pipeline.enrich("the crystal hums with energy")
        assert result is not None
        assert result.novel is True

    def test_familiar_text_gated(self):
        scorer = _make_scorer(novelty=0.1, salience=0.2)
        pipeline = BioEnrichmentPipeline(scorer=scorer, novelty_threshold=0.5)
        result = pipeline.enrich("hello how are you")
        assert result is None

    def test_no_scorer_always_enriches(self):
        """Without a scorer, pipeline always enriches (no gating)."""
        pipeline = BioEnrichmentPipeline()
        result = pipeline.enrich("any text at all", bypass_gate=True)
        assert result is not None

    def test_hippocampus_query(self):
        hippo = _make_hippocampus()
        pipeline = BioEnrichmentPipeline(hippocampus=hippo)
        result = pipeline.enrich("slash the sword", bypass_gate=True)
        assert result is not None
        assert len(result.memories) > 0
        assert "slash" in result.memories[0].summary.lower() or "sword" in result.memories[0].summary.lower()
        hippo.search_by_content.assert_called_once()

    def test_nac_predictions(self):
        nac = _make_nac()
        pipeline = BioEnrichmentPipeline(nac=nac)
        result = pipeline.enrich("rusty_sword_slash attack", bypass_gate=True)
        assert result is not None
        # NAc should return predictions for keywords
        nac.get_links_for_event.assert_called()

    def test_component_index_affordances(self):
        index = MagicMock()
        match = MagicMock()
        match.name = "rusty_sword"
        match.score = 0.7
        index.find_similar.return_value = [match]

        pipeline = BioEnrichmentPipeline(component_index=index)
        result = pipeline.enrich("attack with sword", bypass_gate=True)
        assert result is not None
        assert "rusty_sword" in result.affordances

    def test_valence_computation_positive(self):
        hippo = _make_hippocampus()  # success memory, valence 0.5
        pipeline = BioEnrichmentPipeline(hippocampus=hippo)
        result = pipeline.enrich("slash target", bypass_gate=True)
        assert result is not None
        assert result.valence > 0  # positive memories → positive valence

    def test_valence_computation_negative(self):
        mem = MagicMock()
        mem.id = "mem_pain"
        mem.tool_name = "force_open"
        mem.success = False
        mem.valence = -0.6
        mem.action = MagicMock()
        mem.action.tool_name = "force_open"
        mem.decision = MagicMock()
        mem.decision.intent = {}
        mem.outcome = MagicMock()
        mem.outcome.result = "Door jammed, hand injured"
        mem.outcome.success = False

        hippo = _make_hippocampus([mem])
        pipeline = BioEnrichmentPipeline(hippocampus=hippo)
        result = pipeline.enrich("force open gate", bypass_gate=True)
        assert result is not None
        assert result.valence < 0  # negative memory → negative valence

    def test_all_systems_none_returns_empty_result(self):
        """Pipeline with no bio-systems returns an empty but valid result."""
        pipeline = BioEnrichmentPipeline()
        result = pipeline.enrich("some text", bypass_gate=True)
        assert result is not None
        assert result.memories == ()
        assert result.predictions == ()
        assert result.concepts == ()
        assert result.affordances == ()
        assert result.valence == 0.0


class TestFormatThoughtResponse:
    def test_format_with_memories(self):
        pipeline = BioEnrichmentPipeline()
        result = EnrichmentResult(
            memories=(EpisodicSummary("m1", "forced open a rusty lock", 0.6, 0.8),),
        )
        text = pipeline.format_thought_response(result)
        assert "experience suggests" in text.lower()
        assert "rusty lock" in text

    def test_format_with_predictions(self):
        pipeline = BioEnrichmentPipeline()
        result = EnrichmentResult(
            predictions=(CausalPrediction("force_open", "success", 0.8, "positive"),),
        )
        text = pipeline.format_thought_response(result)
        assert "force_open" in text
        assert "high confidence" in text.lower()

    def test_format_with_affordances(self):
        pipeline = BioEnrichmentPipeline()
        result = EnrichmentResult(
            affordances=("force_open", "pick_lock", "examine_hinge"),
        )
        text = pipeline.format_thought_response(result)
        assert "Available actions" in text
        assert "force_open" in text

    def test_format_empty_result(self):
        pipeline = BioEnrichmentPipeline()
        result = EnrichmentResult()
        text = pipeline.format_thought_response(result)
        assert text == ""

    def test_format_combined(self):
        pipeline = BioEnrichmentPipeline()
        result = EnrichmentResult(
            memories=(EpisodicSummary("m1", "broke a rusty lock with force", 0.5, 0.9),),
            predictions=(CausalPrediction("force_open", "success", 0.7, "positive"),),
            concepts=(ConceptLink("degradation", "material", 0.6),),
            affordances=("force_open", "pick_lock"),
        )
        text = pipeline.format_thought_response(result)
        assert "rusty lock" in text
        assert "force_open" in text
        assert "degradation" in text
        assert "Available actions" in text


class TestEnrichmentContext:
    def test_to_gating_context(self):
        ctx = EnrichmentContext(
            active_goal="find the key",
            goal_keywords=("find", "key"),
        )
        gctx = ctx.to_gating_context()
        assert gctx.active_goal == "find the key"
        assert gctx.goal_keywords == ("find", "key")
