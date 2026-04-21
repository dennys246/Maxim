"""Tests for L1+L2 bio-enrichment wiring — ThinkTool + percept enrichment + deliberation.

Covers:
- L1: ThinkTool calls BioEnrichmentPipeline.enrich(bypass_gate=True)
- L1: ThinkTool includes formatted bio_response in output
- L1: ThinkTool degrades gracefully when pipeline is None or raises
- L1: Percept enrichment injects into StructuredContext.bio_enrichment_context
- L2: Consecutive think counter + deliberation_hop field
- L2: Convergence detection via keyword overlap
- L2: Hard cap after max_deliberation_hops
- L2: reset_deliberation() clears state
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from maxim.integration.bio_enrichment import (
    BioEnrichmentPipeline,
    EnrichmentResult,
    EpisodicSummary,
    CausalPrediction,
)
from maxim.tools.narrative import ThinkTool


# ---------------------------------------------------------------------------
# L1 — ThinkTool enrichment tests
# ---------------------------------------------------------------------------


class TestThinkToolEnrichment:
    def test_think_without_pipeline(self):
        """ThinkTool works normally without a pipeline (backward compat)."""
        tool = ThinkTool()
        result = tool.execute(thought="I should explore the cave")
        assert result.success is True
        assert result.output["thought"] == "I should explore the cave"
        assert "bio_response" not in result.output
        assert result.output["deliberation_hop"] == 1

    def test_think_with_pipeline_enriches(self):
        """ThinkTool calls pipeline.enrich with bypass_gate=True."""
        pipeline = MagicMock(spec=BioEnrichmentPipeline)
        pipeline.enrich.return_value = EnrichmentResult(
            memories=(EpisodicSummary("m1", "explored a dark cave before", 0.4, 0.8),),
            predictions=(CausalPrediction("explore_cave", "find_treasure", 0.7, "positive"),),
            valence=0.3,
        )
        pipeline.format_thought_response.return_value = (
            "Your experience suggests:\n  [+] explored a dark cave before\n"
            "Predictions from past actions:\n  - explore_cave → find_treasure (moderate confidence, positive)"
        )

        tool = ThinkTool(pipeline=pipeline)
        result = tool.execute(thought="I should explore the cave")

        assert result.success is True
        assert result.output["thought"] == "I should explore the cave"
        assert "bio_response" in result.output
        assert "explored a dark cave" in result.output["bio_response"]
        assert result.output["valence"] == pytest.approx(0.3)

        # Verify bypass_gate=True passed (context is also passed from L2)
        call_args = pipeline.enrich.call_args
        assert call_args.args[0] == "I should explore the cave"
        assert call_args.kwargs["bypass_gate"] is True

    def test_think_pipeline_returns_none(self):
        """If pipeline returns None (e.g. empty text internally), no bio_response."""
        pipeline = MagicMock(spec=BioEnrichmentPipeline)
        pipeline.enrich.return_value = None

        tool = ThinkTool(pipeline=pipeline)
        result = tool.execute(thought="hmm")

        assert result.success is True
        assert "bio_response" not in result.output

    def test_think_pipeline_empty_format(self):
        """If format_thought_response returns empty string, no bio_response."""
        pipeline = MagicMock(spec=BioEnrichmentPipeline)
        pipeline.enrich.return_value = EnrichmentResult()  # all empty
        pipeline.format_thought_response.return_value = ""

        tool = ThinkTool(pipeline=pipeline)
        result = tool.execute(thought="thinking about nothing")

        assert result.success is True
        assert "bio_response" not in result.output

    def test_think_pipeline_raises(self):
        """If pipeline raises, tool still succeeds (enrichment is best-effort)."""
        pipeline = MagicMock(spec=BioEnrichmentPipeline)
        pipeline.enrich.side_effect = RuntimeError("EC unavailable")

        tool = ThinkTool(pipeline=pipeline)
        result = tool.execute(thought="risky thought")

        assert result.success is True
        assert result.output["thought"] == "risky thought"
        assert "bio_response" not in result.output

    def test_think_empty_thought_still_fails(self):
        """Empty thought should still fail, regardless of pipeline."""
        pipeline = MagicMock(spec=BioEnrichmentPipeline)
        tool = ThinkTool(pipeline=pipeline)
        result = tool.execute(thought="")

        assert result.success is False
        pipeline.enrich.assert_not_called()

    def test_think_accepts_text_alias(self):
        """ThinkTool accepts 'text' param alias with pipeline."""
        pipeline = MagicMock(spec=BioEnrichmentPipeline)
        pipeline.enrich.return_value = None

        tool = ThinkTool(pipeline=pipeline)
        result = tool.execute(text="alternative param")

        assert result.success is True
        call_args = pipeline.enrich.call_args
        assert call_args.args[0] == "alternative param"


# ---------------------------------------------------------------------------
# L2 — Active deliberation tests
# ---------------------------------------------------------------------------


class TestDeliberationTracking:
    def test_first_think_is_hop_1(self):
        tool = ThinkTool()
        result = tool.execute(thought="first thought")
        assert result.output["deliberation_hop"] == 1

    def test_consecutive_thinks_increment(self):
        tool = ThinkTool()
        r1 = tool.execute(thought="first")
        r2 = tool.execute(thought="second")
        r3 = tool.execute(thought="third")
        assert r1.output["deliberation_hop"] == 1
        assert r2.output["deliberation_hop"] == 2
        assert r3.output["deliberation_hop"] == 3

    def test_reset_deliberation_clears_counter(self):
        tool = ThinkTool()
        tool.execute(thought="first")
        tool.execute(thought="second")
        assert tool._consecutive_thinks == 2

        tool.reset_deliberation()
        assert tool._consecutive_thinks == 0
        assert tool._recent_thought_keywords == []

        result = tool.execute(thought="fresh start")
        assert result.output["deliberation_hop"] == 1

    def test_hard_cap_at_3_hops(self):
        """After 3 consecutive thinks, a termination signal is injected."""
        tool = ThinkTool(max_deliberation_hops=3)
        tool.execute(thought="explore the cave entrance")
        tool.execute(thought="check for traps first")
        r3 = tool.execute(thought="look for hidden passages")

        assert "deliberation_signal" in r3.output
        assert "time to act" in r3.output["deliberation_signal"].lower()

    def test_no_signal_before_cap(self):
        """No termination signal before reaching the cap."""
        tool = ThinkTool(max_deliberation_hops=3)
        r1 = tool.execute(thought="explore the cave entrance")
        r2 = tool.execute(thought="completely different topic about swords")

        # First two hops should not have termination signal
        assert "deliberation_signal" not in r1.output
        assert "deliberation_signal" not in r2.output

    def test_custom_max_hops(self):
        """max_deliberation_hops is configurable."""
        diverse_thoughts = [
            "the ancient dragon breathes fire",
            "underwater caves contain rare crystals",
            "trading with merchants requires gold coins",
            "climbing the mountain needs rope equipment",
            "crossing the desert without water kills",
        ]
        tool = ThinkTool(max_deliberation_hops=5)
        for thought in diverse_thoughts[:4]:
            result = tool.execute(thought=thought)
            assert "deliberation_signal" not in result.output

        r5 = tool.execute(thought=diverse_thoughts[4])
        assert "deliberation_signal" in r5.output

    def test_hard_cap_after_reset(self):
        """Hard cap resets after reset_deliberation()."""
        tool = ThinkTool(max_deliberation_hops=2)
        tool.execute(thought="first")
        r2 = tool.execute(thought="second")
        assert "deliberation_signal" in r2.output

        tool.reset_deliberation()
        r3 = tool.execute(thought="fresh first")
        assert "deliberation_signal" not in r3.output


class TestConvergenceDetection:
    def test_no_convergence_on_first_think(self):
        tool = ThinkTool()
        result = tool.execute(thought="explore the dark cave carefully")
        assert "deliberation_signal" not in result.output

    def test_convergence_on_similar_thoughts(self):
        """Highly similar thoughts trigger convergence signal."""
        tool = ThinkTool(max_deliberation_hops=10)  # high cap so we only test convergence
        tool.execute(thought="explore the dark cave carefully")
        result = tool.execute(thought="explore the dark cave carefully")  # exact repeat

        assert "deliberation_signal" in result.output
        assert "converging" in result.output["deliberation_signal"].lower()

    def test_no_convergence_on_different_thoughts(self):
        """Different thoughts don't trigger convergence."""
        tool = ThinkTool(max_deliberation_hops=10)
        tool.execute(thought="explore the dark cave carefully")
        result = tool.execute(thought="sharpen the rusty sword blade")

        assert "deliberation_signal" not in result.output

    def test_convergence_overridden_by_hard_cap(self):
        """Hard cap takes precedence over convergence when both fire."""
        tool = ThinkTool(max_deliberation_hops=2)
        tool.execute(thought="explore the dark cave carefully")
        result = tool.execute(thought="explore the dark cave carefully")

        # Both convergence and hard cap fire — hard cap wins (it's checked last)
        assert "deliberation_signal" in result.output
        assert "time to act" in result.output["deliberation_signal"].lower()

    def test_convergence_keyword_overlap_threshold(self):
        """Convergence requires >= 0.8 Jaccard similarity."""
        tool = ThinkTool(max_deliberation_hops=10)
        # 5 content words: explore, dark, cave, carefully, hidden
        tool.execute(thought="explore the dark cave carefully for hidden treasure")
        # Change just one word: barely under 0.8 overlap
        result = tool.execute(thought="explore the dark cave carefully for valuable gems")

        # "hidden treasure" → "valuable gems" changes 2 of ~6 keywords
        # Overlap should be around 4/8 = 0.5, below 0.8 threshold
        assert "deliberation_signal" not in result.output

    def test_no_convergence_on_short_repeated_thoughts(self):
        """Short thoughts (< 3 keywords) never trigger convergence — prevents false positives."""
        tool = ThinkTool(max_deliberation_hops=10)
        tool.execute(thought="run away")  # 2 keywords: {"run", "away"}
        result = tool.execute(thought="run away")  # exact repeat
        # Should NOT converge — too few keywords for reliable similarity
        assert "deliberation_signal" not in result.output

    def test_no_convergence_on_single_word(self):
        """Single-word thoughts never trigger convergence."""
        tool = ThinkTool(max_deliberation_hops=10)
        tool.execute(thought="attack")
        result = tool.execute(thought="attack")
        assert "deliberation_signal" not in result.output

    def test_recent_thoughts_capped_at_5(self):
        """Only last 5 thought keyword sets are tracked."""
        tool = ThinkTool(max_deliberation_hops=20)
        for i in range(7):
            tool.execute(thought=f"completely unique topic number {i} with extra words")

        assert len(tool._recent_thought_keywords) <= 5


class TestKeywordExtraction:
    def test_extracts_meaningful_words(self):
        keywords = ThinkTool._extract_thought_keywords("I think the rusty sword is dangerous")
        assert "rusty" in keywords
        assert "sword" in keywords
        assert "dangerous" in keywords
        # Stop words removed
        assert "the" not in keywords
        assert "think" not in keywords

    def test_short_words_excluded(self):
        keywords = ThinkTool._extract_thought_keywords("I am at it")
        assert len(keywords) == 0

    def test_underscores_split(self):
        keywords = ThinkTool._extract_thought_keywords("rusty_sword_slash attack")
        assert "rusty" in keywords
        assert "sword" in keywords
        assert "slash" in keywords
        assert "attack" in keywords


# ---------------------------------------------------------------------------
# L1 — StructuredContext enrichment field tests
# ---------------------------------------------------------------------------


class TestStructuredContextEnrichmentField:
    def test_field_defaults_to_empty(self):
        from maxim.agents.bus import StructuredContext

        ctx = StructuredContext(timestamp=0.0)
        assert ctx.bio_enrichment_context == ""

    def test_field_settable(self):
        from maxim.agents.bus import StructuredContext

        ctx = StructuredContext(timestamp=0.0)
        ctx.bio_enrichment_context = "Your experience suggests:\n  [+] something"
        assert "experience suggests" in ctx.bio_enrichment_context
