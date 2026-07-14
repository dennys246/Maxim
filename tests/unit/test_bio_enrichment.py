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

from unittest.mock import MagicMock


from maxim.integration.bio_enrichment import (
    BioEnrichmentPipeline,
    CausalPrediction,
    ConceptLink,
    EnrichmentContext,
    EnrichmentResult,
    EpisodicSummary,
)
from maxim.runtime.gating import GateScore, TextSalienceScorer


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
        link.event_signature = "tool:rusty_sword_slash"
        link.outcome_signature = "target_hit"
        link.confidence = 0.8
        link.outcome_valence = Valence.POSITIVE
        nac.scan_links_for_keywords.return_value = [link]
    else:
        nac.scan_links_for_keywords.return_value = predictions
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
        # NAc should be queried via scan_links_for_keywords (substring
        # match against compound tool signatures), not the legacy exact
        # get_links_for_event lookup that was missing every narrative
        # keyword.
        nac.scan_links_for_keywords.assert_called()
        assert any(p.event == "tool:rusty_sword_slash" for p in result.predictions)

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

    def test_resolved_entities_default_empty(self):
        ctx = EnrichmentContext()
        assert ctx.resolved_entities == ()

    def test_resolved_entities_passthrough(self):
        ctx = EnrichmentContext(resolved_entities=("creatures/dragon", "weapons/rusty_sword"))
        assert ctx.resolved_entities == ("creatures/dragon", "weapons/rusty_sword")


# ---------------------------------------------------------------------------
# Phase 2: Resolved entity affordance surfacing
# ---------------------------------------------------------------------------


def _make_registry(specs: dict[str, dict] | None = None) -> MagicMock:
    """Create a mock ComponentRegistry that returns specs by ref.

    Mimics real ComponentRegistry.get() which raises KeyError on unknown refs.
    """
    registry = MagicMock()
    _specs = specs or {}

    def _get(ref: str) -> dict:
        if ref not in _specs:
            raise KeyError(f"Component not found: {ref}")
        return _specs[ref]

    registry.get = MagicMock(side_effect=_get)
    return registry


_DRAGON_SPEC = {
    "entity": {
        "name": "dragon",
        "entity_type": "creature",
        "sensors": {},
        "modulators": {
            "head": {
                "affordances": {
                    "bite": {"params": {"target": "str"}, "description": "Bite"},
                    "roar": {"params": {}, "description": "Roar"},
                },
            },
            "combat": {
                "affordances": {
                    "fire_breath": {"params": {"target": "str"}, "description": "Fire"},
                    "claw_strike": {"params": {"target": "str"}, "description": "Claw"},
                },
            },
        },
    }
}


class TestResolvedEntityAffordances:
    """Phase 2: BioEnrichmentPipeline surfaces affordances from resolved entities."""

    def test_resolved_entities_surface_all_affordances(self):
        """When resolved_entities are provided, all affordances from those entities appear."""
        registry = _make_registry({"creatures/dragon": _DRAGON_SPEC})
        pipeline = BioEnrichmentPipeline(component_registry=registry)
        ctx = EnrichmentContext(resolved_entities=("creatures/dragon",))
        result = pipeline.enrich("a dragon breathes fire", bypass_gate=True, context=ctx)
        assert result is not None
        # Should have all 4 dragon affordances
        aff_text = " ".join(result.affordances)
        assert "bite" in aff_text
        assert "roar" in aff_text
        assert "fire_breath" in aff_text
        assert "claw_strike" in aff_text

    def test_resolved_entities_prefixed_with_entity_name(self):
        """Each affordance is prefixed with entity name for clarity."""
        registry = _make_registry({"creatures/dragon": _DRAGON_SPEC})
        pipeline = BioEnrichmentPipeline(component_registry=registry)
        ctx = EnrichmentContext(resolved_entities=("creatures/dragon",))
        result = pipeline.enrich("a dragon", bypass_gate=True, context=ctx)
        assert result is not None
        assert any(a.startswith("dragon:") for a in result.affordances)

    def test_resolved_entities_with_nac_valence_annotation(self):
        """Affordances get NAc valence annotations when NAc + ATL are available."""
        registry = _make_registry({"creatures/dragon": _DRAGON_SPEC})

        # Mock ATL that returns a concept for "fire"
        atl = MagicMock()
        concept = MagicMock()
        concept.id = "node_fire"
        concept.name = "fire"
        concept.category = "substrate"
        atl.recall.return_value = [concept]

        # Mock NAc that returns negative reward_bias for fire
        nac = MagicMock()
        nac.reward_bias.return_value = -0.5
        nac.get_links_for_event.return_value = []

        pipeline = BioEnrichmentPipeline(component_registry=registry, nac=nac, atl=atl)
        ctx = EnrichmentContext(resolved_entities=("creatures/dragon",))
        result = pipeline.enrich("a dragon", bypass_gate=True, context=ctx)
        assert result is not None
        # fire_breath should be annotated as DANGEROUS
        aff_text = " ".join(result.affordances)
        assert "DANGEROUS" in aff_text

    def test_no_resolved_entities_falls_back_to_text_similarity(self):
        """Without resolved_entities, falls back to text-similarity search."""
        index = MagicMock()
        match = MagicMock()
        match.name = "rusty_sword"
        match.score = 0.7
        index.find_similar.return_value = [match]

        pipeline = BioEnrichmentPipeline(component_index=index)
        result = pipeline.enrich("attack with sword", bypass_gate=True)
        assert result is not None
        assert "rusty_sword" in result.affordances[0]

    def test_resolved_entities_skips_fallback(self):
        """When resolved_entities produce affordances, text-similarity is not called."""
        registry = _make_registry({"creatures/dragon": _DRAGON_SPEC})
        index = MagicMock()
        index.find_similar.return_value = []

        pipeline = BioEnrichmentPipeline(component_registry=registry, component_index=index)
        ctx = EnrichmentContext(resolved_entities=("creatures/dragon",))
        result = pipeline.enrich("a dragon", bypass_gate=True, context=ctx)
        assert result is not None
        assert len(result.affordances) > 0
        # Fallback should NOT have been called
        index.find_similar.assert_not_called()

    def test_multiple_resolved_entities(self):
        """Multiple entities each contribute their affordances."""
        sword_spec = {
            "entity": {
                "name": "rusty_sword",
                "entity_type": "weapon",
                "sensors": {},
                "modulators": {
                    "blade": {
                        "affordances": {
                            "slash": {"params": {"target": "str"}},
                            "thrust": {"params": {"target": "str"}},
                        },
                    },
                },
            }
        }
        registry = _make_registry(
            {
                "creatures/dragon": _DRAGON_SPEC,
                "weapons/rusty_sword": sword_spec,
            }
        )
        pipeline = BioEnrichmentPipeline(component_registry=registry)
        ctx = EnrichmentContext(resolved_entities=("creatures/dragon", "weapons/rusty_sword"))
        result = pipeline.enrich("dragon and sword", bypass_gate=True, context=ctx)
        assert result is not None
        aff_text = " ".join(result.affordances)
        assert "fire_breath" in aff_text
        assert "slash" in aff_text

    def test_unknown_ref_in_resolved_entities_skipped(self):
        """Unknown refs in resolved_entities are silently skipped."""
        registry = _make_registry({})
        pipeline = BioEnrichmentPipeline(component_registry=registry)
        ctx = EnrichmentContext(resolved_entities=("creatures/unknown",))
        result = pipeline.enrich("some text", bypass_gate=True, context=ctx)
        assert result is not None
        assert result.affordances == ()

    def test_deduplicate_affordances_across_entities(self):
        """Affordances with the same name from different entities appear only once."""
        spec_a = {
            "entity": {
                "name": "wolf",
                "sensors": {},
                "modulators": {
                    "head": {"affordances": {"bite": {"params": {"target": "str"}}}},
                },
            }
        }
        spec_b = {
            "entity": {
                "name": "dog",
                "sensors": {},
                "modulators": {
                    "head": {"affordances": {"bite": {"params": {"target": "str"}}}},
                },
            }
        }
        registry = _make_registry(
            {
                "creatures/wolf": spec_a,
                "creatures/dog": spec_b,
            }
        )
        pipeline = BioEnrichmentPipeline(component_registry=registry)
        ctx = EnrichmentContext(resolved_entities=("creatures/wolf", "creatures/dog"))
        result = pipeline.enrich("wolf and dog", bypass_gate=True, context=ctx)
        assert result is not None
        bite_count = sum(1 for a in result.affordances if "bite" in a)
        assert bite_count == 1  # deduplicated
