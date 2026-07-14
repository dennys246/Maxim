"""Tests for runtime/gating.py — shared gating abstractions.

Covers:
- GateScore / GateDecision / GatingContext data types
- AdaptiveThresholdController adaptation + persistence
- TextSalienceScorer novelty + salience + goal relevance scoring
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from maxim.runtime.gating import (
    AdaptiveThresholdConfig,
    AdaptiveThresholdController,
    GateDecision,
    GateScore,
    GatingContext,
    TextSalienceScorer,
)


# ---------------------------------------------------------------------------
# GateScore / GateDecision / GatingContext
# ---------------------------------------------------------------------------


class TestDataTypes:
    def test_gate_score_frozen(self):
        score = GateScore(novelty=0.8, salience=0.6, goal_relevance=0.3, combined=0.48)
        assert score.novelty == 0.8
        assert score.combined == 0.48
        with pytest.raises(Exception):
            score.novelty = 0.5  # type: ignore

    def test_gate_decision(self):
        score = GateScore(novelty=0.8, salience=0.6, combined=0.48)
        decision = GateDecision(passed=True, score=score, reason="novel", threshold_used=0.3)
        assert decision.passed
        assert decision.reason == "novel"

    def test_gating_context_defaults(self):
        ctx = GatingContext()
        assert ctx.active_goal is None
        assert ctx.energy == 1.0
        assert ctx.processing_load == 0


# ---------------------------------------------------------------------------
# AdaptiveThresholdController
# ---------------------------------------------------------------------------


class TestAdaptiveThresholdController:
    def test_initial_thresholds(self):
        ctrl = AdaptiveThresholdController()
        assert ctrl.novelty_threshold == pytest.approx(0.7)
        assert ctrl.salience_threshold == pytest.approx(0.6)

    def test_custom_config(self):
        config = AdaptiveThresholdConfig(
            base_novelty_threshold=0.5,
            base_salience_threshold=0.4,
        )
        ctrl = AdaptiveThresholdController(config)
        assert ctrl.novelty_threshold == pytest.approx(0.5)

    def test_combined_threshold(self):
        ctrl = AdaptiveThresholdController()
        assert ctrl.combined_threshold == pytest.approx(0.7 * 0.6)

    def test_record_escalation(self):
        ctrl = AdaptiveThresholdController()
        ctrl.record_escalation(True)
        ctrl.record_escalation(False)
        assert len(ctrl._escalation_history) == 2

    def test_record_outcome(self):
        ctrl = AdaptiveThresholdController()
        ctrl.record_outcome("action_taken")
        assert len(ctrl._outcome_history) == 1

    def test_adapt_raises_threshold_on_high_rate(self):
        config = AdaptiveThresholdConfig(
            base_novelty_threshold=0.5,
            adaptation_interval=0.0,  # adapt immediately
        )
        ctrl = AdaptiveThresholdController(config)
        # Simulate high escalation rate (90%)
        for i in range(20):
            ctrl.record_escalation(i < 18)  # 18/20 = 90%
        ctrl.maybe_adapt()
        # Threshold should have increased
        assert ctrl.novelty_threshold > 0.5

    def test_adapt_lowers_threshold_on_low_rate(self):
        config = AdaptiveThresholdConfig(
            base_novelty_threshold=0.8,
            adaptation_interval=0.0,
        )
        ctrl = AdaptiveThresholdController(config)
        # Simulate low escalation rate (1%)
        for i in range(100):
            ctrl.record_escalation(i == 0)  # 1/100
        ctrl.maybe_adapt()
        # Threshold should have decreased
        assert ctrl.novelty_threshold < 0.8

    def test_reset(self):
        ctrl = AdaptiveThresholdController()
        ctrl.record_escalation(True)
        ctrl._novelty_threshold = 0.9
        ctrl.reset()
        assert ctrl.novelty_threshold == pytest.approx(0.7)
        assert len(ctrl._escalation_history) == 0

    def test_persistence_save_load(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        try:
            config = AdaptiveThresholdConfig(persist_path=path)
            ctrl = AdaptiveThresholdController(config)
            ctrl._novelty_threshold = 0.42
            ctrl.record_escalation(True)
            ctrl.save()

            # Load into a fresh controller
            ctrl2 = AdaptiveThresholdController(AdaptiveThresholdConfig(persist_path=path))
            assert ctrl2.novelty_threshold == pytest.approx(0.42)
        finally:
            Path(path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# TextSalienceScorer
# ---------------------------------------------------------------------------


class TestTextSalienceScorer:
    def test_empty_input_scores_zero(self):
        scorer = TextSalienceScorer()
        score = scorer.score("", GatingContext())
        assert score.novelty == 0.0
        assert score.combined == 0.0

    def test_first_input_is_novel(self):
        scorer = TextSalienceScorer()
        score = scorer.score("the crystal hums with blue light", GatingContext())
        assert score.novelty == 1.0  # First input is always novel

    def test_repeated_input_low_novelty(self):
        scorer = TextSalienceScorer()
        ctx = GatingContext()
        scorer.score("the crystal hums", ctx)
        scorer.score("the crystal hums", ctx)
        # Third identical input should have low novelty
        score = scorer.score("the crystal hums", ctx)
        assert score.novelty < 0.5

    def test_different_input_stays_novel(self):
        scorer = TextSalienceScorer()
        ctx = GatingContext()
        scorer.score("the crystal hums", ctx)
        score = scorer.score("a dragon appears from the shadows", ctx)
        assert score.novelty > 0.7  # Different words = still novel

    def test_goal_relevance_with_keywords(self):
        scorer = TextSalienceScorer()
        ctx = GatingContext(
            active_goal="find the sword",
            goal_keywords=("find", "sword"),
        )
        score = scorer.score("I see a rusty sword on the ground", ctx)
        assert score.goal_relevance > 0.0

    def test_goal_relevance_without_goal(self):
        scorer = TextSalienceScorer()
        ctx = GatingContext()
        score = scorer.score("I see a rusty sword", ctx)
        assert score.goal_relevance == 0.0

    def test_nac_boosts_salience(self):
        """When NAc has causal links for a word, salience increases."""
        nac = MagicMock()
        link = MagicMock()
        link.confidence = 0.8
        nac.get_links_for_event.return_value = [link]

        scorer = TextSalienceScorer(nac=nac)
        ctx = GatingContext()
        score = scorer.score("slash the target", ctx)
        assert score.salience > 0.5  # Boosted by NAc

    def test_no_nac_neutral_salience(self):
        scorer = TextSalienceScorer()
        ctx = GatingContext()
        score = scorer.score("something happens", ctx)
        assert score.salience == 0.5  # Neutral without NAc

    def test_reset_clears_history(self):
        scorer = TextSalienceScorer()
        ctx = GatingContext()
        scorer.score("hello world", ctx)
        scorer.reset()
        # After reset, same text should be novel again
        score = scorer.score("hello world", ctx)
        assert score.novelty == 1.0

    def test_ec_semantic_novelty(self):
        """When EC is available, use semantic novelty instead of keyword overlap."""
        ec = MagicMock()
        # EC says this is 80% similar to something known
        ec.find_semantic.return_value = [("memory_42", 0.8)]

        scorer = TextSalienceScorer(ec=ec)
        ctx = GatingContext()
        score = scorer.score("familiar concept", ctx)
        assert score.novelty == pytest.approx(0.2)  # 1.0 - 0.8
