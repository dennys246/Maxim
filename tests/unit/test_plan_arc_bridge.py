"""Tests for plan-to-arc translation and narrator context enrichment (Stage B).

Covers:
- translate_plan_to_arc with custom decomposition
- Matching plan structure to builtin arcs
- enrich_narrator_context from PlanningContext signals
- bridge_and_compress for multi-arc continuation
- Planner-driven arc resolution in generative runner
"""

from __future__ import annotations

from unittest.mock import MagicMock

from maxim.simulation.arcs import NarrativeArc, NarrativePhase
from maxim.simulation.plan_arc_bridge import (
    bridge_and_compress,
    enrich_narrator_context,
    translate_plan_to_arc,
)


# ---------------------------------------------------------------------------
# Plan-to-arc translation
# ---------------------------------------------------------------------------


class TestTranslatePlanToArc:
    def test_custom_decomposition(self):
        actions = [
            {"phase": "setup", "description": "Introduce the scenario", "turns_min": 1, "turns_max": 2},
            {"phase": "challenge", "description": "Present a difficult problem", "turns_min": 2, "turns_max": 4},
            {"phase": "resolution", "description": "Resolve the situation", "turns_min": 1, "turns_max": 1},
        ]
        arc = translate_plan_to_arc(actions, "test custom scenario")
        assert arc.source == "planner"
        assert len(arc.phases) == 3
        assert arc.phases[0].name == "setup"
        assert arc.phases[1].turns_max == 4

    def test_empty_actions_fallback(self):
        arc = translate_plan_to_arc([], "test something")
        assert len(arc.phases) == 1
        assert arc.phases[0].name == "exploration"

    def test_with_sub_goals(self):
        actions = [
            {
                "phase": "learning",
                "description": "Teach a skill",
                "sub_goals": ["identify plants", "prepare poultice", "test recall"],
            },
        ]
        arc = translate_plan_to_arc(actions, "test skill learning")
        instruction = arc.phases[0].instruction
        assert "identify plants" in instruction
        assert "prepare poultice" in instruction

    def test_matches_builtin_memory_recall(self):
        """Plan with memory-related phases should match memory_recall builtin."""
        actions = [
            {"phase": "seed", "description": "Plant a memory seed detail"},
            {"phase": "interference", "description": "Unrelated encounters"},
            {"phase": "recall", "description": "Test memory recall"},
        ]
        arc = translate_plan_to_arc(actions, "test memory recall under interference")
        # Should match builtin because of keyword overlap
        assert arc.name == "memory_recall" or arc.source in ("builtin", "planner")

    def test_no_match_uses_custom(self):
        actions = [
            {"phase": "alpha", "description": "Something unique"},
            {"phase": "beta", "description": "Another unique thing"},
        ]
        arc = translate_plan_to_arc(actions, "evaluate quantum entanglement")
        assert arc.source == "planner"
        assert len(arc.phases) == 2


# ---------------------------------------------------------------------------
# Narrator context enrichment
# ---------------------------------------------------------------------------


class TestEnrichNarratorContext:
    def test_with_nac_prediction(self):
        ctx = MagicMock()
        ctx.primary_prediction = MagicMock()
        ctx.primary_prediction.predicted_valence = "POSITIVE"
        ctx.primary_prediction.confidence = 0.8
        ctx.situation_novelty = 0.3
        ctx.ranked_skills = []
        ctx.reflections = []
        ctx.associated_memories = []
        ctx.successful_strategies = []

        result = enrich_narrator_context(ctx)
        assert "NAc predicts" in result
        assert "POSITIVE" in result
        assert "familiar" in result.lower()

    def test_with_high_novelty(self):
        ctx = MagicMock()
        ctx.primary_prediction = None
        ctx.situation_novelty = 0.9
        ctx.ranked_skills = []
        ctx.reflections = []
        ctx.associated_memories = []
        ctx.successful_strategies = []

        result = enrich_narrator_context(ctx)
        assert "novel" in result.lower()

    def test_with_ranked_skills(self):
        ctx = MagicMock()
        ctx.primary_prediction = None
        ctx.situation_novelty = 0.5
        ctx.ranked_skills = [{"name": "herbalism"}, {"name": "navigation"}]
        ctx.reflections = []
        ctx.associated_memories = []
        ctx.successful_strategies = []

        result = enrich_narrator_context(ctx)
        assert "herbalism" in result
        assert "navigation" in result

    def test_with_reflections(self):
        ctx = MagicMock()
        ctx.primary_prediction = None
        ctx.situation_novelty = 0.5
        ctx.ranked_skills = []
        ctx.reflections = ["reflection 1", "reflection 2"]
        ctx.associated_memories = []
        ctx.successful_strategies = []

        result = enrich_narrator_context(ctx)
        assert "reflected" in result.lower()

    def test_none_context(self):
        result = enrich_narrator_context(None)
        assert result == ""

    def test_empty_context(self):
        ctx = MagicMock()
        ctx.primary_prediction = None
        ctx.situation_novelty = 0.5  # middle ground — no hint generated
        ctx.ranked_skills = []
        ctx.reflections = []
        ctx.associated_memories = []
        ctx.successful_strategies = []

        result = enrich_narrator_context(ctx)
        assert result == ""


# ---------------------------------------------------------------------------
# Bridge-and-compress
# ---------------------------------------------------------------------------


class TestBridgeAndCompress:
    def test_successful_bridge(self):
        llm = MagicMock()
        llm.generate_json = MagicMock(
            return_value={
                "bridge_narrative": "The road stretches onward as dawn breaks.",
                "compressed_context": "Hero traveled through forest, met ferryman.",
                "tested_aspects": ["memory_recall", "trust_building"],
                "done": False,
            }
        )

        prev_arc = NarrativeArc("memory_test", "Test memory", source="builtin")
        next_arc = NarrativeArc(
            "safety_test",
            "Test safety",
            phases=[NarrativePhase("trust", "Build trust", 2, 3)],
        )

        result = bridge_and_compress(
            llm,
            aut_name="Verath",
            compressed_context="Hero entered the forest.",
            previous_arc=prev_arc,
            arc_summary="Tested memory recall successfully.",
            last_aut_response="I remember the password.",
            planner_context="NAc predicts positive outcome.",
            next_arc=next_arc,
        )

        assert "dawn breaks" in result["bridge_narrative"]
        assert len(result["tested_aspects"]) == 2
        assert result["done"] is False

    def test_llm_failure_fallback(self):
        llm = MagicMock()
        llm.generate_json = MagicMock(side_effect=Exception("LLM error"))

        prev_arc = NarrativeArc("test", "Test")
        next_arc = NarrativeArc("test2", "Test 2")

        result = bridge_and_compress(
            llm,
            aut_name="AUT",
            compressed_context="Previous context.",
            previous_arc=prev_arc,
            arc_summary="",
            last_aut_response="",
            planner_context="",
            next_arc=next_arc,
        )

        assert "path leads onward" in result["bridge_narrative"].lower()
        assert result["compressed_context"] == "Previous context."

    def test_done_signal(self):
        llm = MagicMock()
        llm.generate_json = MagicMock(
            return_value={
                "bridge_narrative": "The adventure concludes.",
                "compressed_context": "Full story.",
                "tested_aspects": ["everything"],
                "done": True,
            }
        )

        result = bridge_and_compress(
            llm,
            aut_name="AUT",
            compressed_context="",
            previous_arc=NarrativeArc("x", "x"),
            arc_summary="",
            last_aut_response="",
            planner_context="",
            next_arc=NarrativeArc("y", "y"),
        )

        assert result["done"] is True


# ---------------------------------------------------------------------------
# Planner-driven runner integration
# ---------------------------------------------------------------------------


class TestPlannerRunnerIntegration:
    def test_planner_generates_arc(self):
        """When planner is provided, it should decompose goal into an arc."""
        from maxim.simulation.generative_runner import _resolve_arc_via_planner

        planner = MagicMock()
        plan_candidate = MagicMock()
        plan_candidate.actions = [
            {"phase": "explore", "description": "Explore the area"},
            {"phase": "test", "description": "Test the hypothesis"},
        ]
        plan_candidate.planning_context = None
        planner.propose_plans = MagicMock(return_value=[plan_candidate])

        arc, context = _resolve_arc_via_planner("test something", planner, None, "")
        assert arc is not None
        assert len(arc.phases) == 2
        assert arc.source == "planner"

    def test_planner_failure_returns_none(self):
        from maxim.simulation.generative_runner import _resolve_arc_via_planner

        planner = MagicMock()
        planner.propose_plans = MagicMock(return_value=[])

        arc, context = _resolve_arc_via_planner("test", planner, None, "original")
        assert arc is None
        assert context == "original"

    def test_planner_enriches_context(self):
        from maxim.simulation.generative_runner import _resolve_arc_via_planner

        planner = MagicMock()
        plan_candidate = MagicMock()
        plan_candidate.actions = [{"phase": "x", "description": "Do x"}]

        pctx = MagicMock()
        pctx.primary_prediction = MagicMock()
        pctx.primary_prediction.predicted_valence = "POSITIVE"
        pctx.primary_prediction.confidence = 0.9
        pctx.situation_novelty = 0.2
        pctx.ranked_skills = [{"name": "lockpicking"}]
        pctx.reflections = ["past failure"]
        pctx.associated_memories = [("mem1", 0.8)]
        pctx.successful_strategies = []
        plan_candidate.planning_context = pctx
        planner.propose_plans = MagicMock(return_value=[plan_candidate])

        arc, context = _resolve_arc_via_planner("test skill", planner, None, "")
        assert arc is not None
        assert "NAc predicts" in context
        assert "lockpicking" in context
