"""Deliberation system stress test — exercises L1+L2 critical paths.

Run: python -m pytest tests/experiments/test_deliberation_stress.py -v
Or standalone: python tests/experiments/test_deliberation_stress.py

Exercises:
1. Hard cap never exceeded (convergence/cap signals always fire)
2. Convergence detection with realistic thought sequences
3. NAc-gated termination via declining valence
4. Reset isolation (non-think tools properly reset state)
5. Pipeline failure resilience (enrichment errors don't break deliberation)
6. High-volume think sequences (100+ rapid-fire calls)
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest

from maxim.integration.bio_enrichment import (
    BioEnrichmentPipeline,
    EnrichmentResult,
    EpisodicSummary,
)
from maxim.tools.narrative import ThinkTool


# ---------------------------------------------------------------------------
# Stress test 1: Hard cap is NEVER exceeded
# ---------------------------------------------------------------------------


class TestHardCapGuarantee:
    """The hard cap is a safety invariant — it must NEVER be exceeded."""

    @pytest.mark.parametrize("max_hops", [1, 2, 3, 5, 10])
    def test_hard_cap_always_fires(self, max_hops: int):
        """Regardless of thought content, cap fires at exactly max_hops."""
        tool = ThinkTool(max_deliberation_hops=max_hops)
        for i in range(max_hops - 1):
            result = tool.execute(thought=f"unique_{i}_" + "x" * 50)
            # Before cap: may or may not have convergence signal, but not the cap signal
            if "deliberation_signal" in result.output:
                assert "time to act" not in result.output["deliberation_signal"].lower() or i >= max_hops - 1

        # At cap: must have termination signal
        result = tool.execute(thought="this triggers the cap")
        assert "deliberation_signal" in result.output
        assert "time to act" in result.output["deliberation_signal"].lower()

    def test_hard_cap_on_100_thinks(self):
        """100 consecutive thinks — every one at or after hop 3 has the cap signal."""
        tool = ThinkTool(max_deliberation_hops=3)
        for i in range(100):
            result = tool.execute(thought=f"iteration {i} with random content {hash(i)}")
            if i >= 2:  # hop 3+ (0-indexed)
                assert "deliberation_signal" in result.output
                assert "time to act" in result.output["deliberation_signal"].lower()


# ---------------------------------------------------------------------------
# Stress test 2: Convergence detection with realistic sequences
# ---------------------------------------------------------------------------


class TestConvergenceRealisticSequences:
    """Test convergence with sequences that mimic real agent deliberation."""

    def test_sword_deliberation_sequence(self):
        """Agent thinking about using a sword — progressive refinement."""
        tool = ThinkTool(max_deliberation_hops=10)
        thoughts = [
            "the rusty sword is on the ground, should I pick it up?",
            "picking up the sword might be risky, it looks corroded",
            "I could examine the sword first to check its condition",
            "examining objects has worked before, let me try that approach",
        ]
        signals = []
        for t in thoughts:
            result = tool.execute(thought=t)
            signals.append(result.output.get("deliberation_signal", ""))

        # Different enough thoughts — no early convergence expected
        # (these share some words but diverge significantly)
        assert signals[0] == ""  # first thought never converges
        # Later thoughts may or may not converge depending on overlap

    def test_looping_deliberation_detected(self):
        """Agent stuck in a loop is detected within 3 iterations."""
        tool = ThinkTool(max_deliberation_hops=10)
        # Agent keeps rephrasing the same thought
        loop_thoughts = [
            "how do I get past this locked gate",
            "the gate is locked, how can I get past it",
            "I need to get past the locked gate somehow",
        ]
        converged = False
        for t in loop_thoughts:
            result = tool.execute(thought=t)
            if "deliberation_signal" in result.output and "converging" in result.output["deliberation_signal"].lower():
                converged = True
                break

        assert converged, "Looping deliberation should be detected"

    def test_progressive_refinement_not_flagged_early(self):
        """Genuinely different thoughts exploring different facets don't trigger."""
        tool = ThinkTool(max_deliberation_hops=10)
        refinement_thoughts = [
            "the bridge looks unstable, what are my options",
            "I have rope in my inventory that could help",
            "there might be another path around the gorge",
        ]
        for t in refinement_thoughts:
            result = tool.execute(thought=t)
            # None of these should trigger convergence (very different content)
            signal = result.output.get("deliberation_signal", "")
            assert "converging" not in signal.lower()


# ---------------------------------------------------------------------------
# Stress test 3: NAc-gated termination via declining valence
# ---------------------------------------------------------------------------


class TestNAcGatedTermination:
    """Declining enrichment valence biases toward action (no hard block)."""

    def test_declining_valence_sequence(self):
        """Enrichment valence declines as thoughts repeat — visible to LLM."""
        pipeline = MagicMock(spec=BioEnrichmentPipeline)
        # Each successive call returns lower valence (declining novelty)
        valences = [0.6, 0.3, 0.1, 0.0]
        call_count = [0]

        def declining_enrichment(text, **kwargs):
            idx = min(call_count[0], len(valences) - 1)
            call_count[0] += 1
            return EnrichmentResult(
                memories=(EpisodicSummary("m1", "past experience", valences[idx], 0.7),),
                valence=valences[idx],
            )

        pipeline.enrich.side_effect = declining_enrichment
        pipeline.format_thought_response.return_value = "some response"

        tool = ThinkTool(pipeline=pipeline, max_deliberation_hops=10)
        output_valences = []
        for i in range(4):
            result = tool.execute(thought=f"different thought {i} about {chr(65 + i)} topic")
            v = result.output.get("valence", None)
            if v is not None:
                output_valences.append(v)

        # Valence should be declining — LLM sees this and is biased toward action
        assert output_valences == pytest.approx(valences[: len(output_valences)])
        assert output_valences[-1] <= output_valences[0]


# ---------------------------------------------------------------------------
# Stress test 4: Reset isolation
# ---------------------------------------------------------------------------


class TestResetIsolation:
    """Non-think tools properly reset deliberation state."""

    def test_reset_clears_all_state(self):
        """After reset, deliberation counter and history are fully cleared."""
        tool = ThinkTool(max_deliberation_hops=3)
        # Build up state
        tool.execute(thought="first thought about swords")
        tool.execute(thought="second thought about shields")
        assert tool._consecutive_thinks == 2
        assert len(tool._recent_thought_keywords) == 2

        tool.reset_deliberation()
        assert tool._consecutive_thinks == 0
        assert tool._recent_thought_keywords == []

    def test_reset_prevents_stale_convergence(self):
        """After reset, old thoughts don't cause false convergence."""
        tool = ThinkTool(max_deliberation_hops=10)
        tool.execute(thought="the rusty sword is dangerous")
        tool.reset_deliberation()
        # Same thought after reset — should NOT converge
        result = tool.execute(thought="the rusty sword is dangerous")
        assert "deliberation_signal" not in result.output

    def test_interleaved_actions_reset_properly(self):
        """Simulates think → action → think — second sequence starts fresh."""
        tool = ThinkTool(max_deliberation_hops=3)
        # First deliberation sequence
        tool.execute(thought="planning first approach")
        tool.execute(thought="refining first approach")
        # Action fires (simulated by reset)
        tool.reset_deliberation()
        # Second sequence starts from hop 1
        r = tool.execute(thought="planning second approach")
        assert r.output["deliberation_hop"] == 1


# ---------------------------------------------------------------------------
# Stress test 5: Pipeline failure resilience
# ---------------------------------------------------------------------------


class TestPipelineResilience:
    """Enrichment failures never break the deliberation system."""

    def test_pipeline_exception_every_call(self):
        """Pipeline that always raises — deliberation still tracks correctly."""
        pipeline = MagicMock(spec=BioEnrichmentPipeline)
        pipeline.enrich.side_effect = RuntimeError("Catastrophic failure")

        tool = ThinkTool(pipeline=pipeline, max_deliberation_hops=3)
        for i in range(3):
            result = tool.execute(thought=f"thought {i}")
            assert result.success is True
            assert result.output["deliberation_hop"] == i + 1

        # Hard cap still fires
        assert "deliberation_signal" in result.output

    def test_pipeline_intermittent_failures(self):
        """Pipeline that fails intermittently — deliberation unaffected."""
        pipeline = MagicMock(spec=BioEnrichmentPipeline)
        call_count = [0]

        def intermittent(text, **kwargs):
            call_count[0] += 1
            if call_count[0] % 2 == 0:
                raise ConnectionError("Temporary failure")
            return EnrichmentResult(valence=0.5)

        pipeline.enrich.side_effect = intermittent
        pipeline.format_thought_response.return_value = ""

        tool = ThinkTool(pipeline=pipeline, max_deliberation_hops=10)
        for i in range(6):
            result = tool.execute(thought=f"diverse thought {chr(65 + i)} about unique topic")
            assert result.success is True
            assert result.output["deliberation_hop"] == i + 1


# ---------------------------------------------------------------------------
# Stress test 6: High-volume rapid-fire
# ---------------------------------------------------------------------------


class TestHighVolume:
    """Performance under high call volume — no memory leaks, no slowdowns."""

    def test_100_thinks_memory_bounded(self):
        """100 consecutive thinks — keyword history stays bounded."""
        tool = ThinkTool(max_deliberation_hops=200)
        for i in range(100):
            tool.execute(thought=f"iteration {i} with completely unique content {hash(i) * 37}")

        # History is bounded at 5 entries
        assert len(tool._recent_thought_keywords) <= 5
        assert tool._consecutive_thinks == 100

    def test_latency_under_1ms_per_call(self):
        """ThinkTool without pipeline completes in < 1ms per call."""
        tool = ThinkTool()
        start = time.perf_counter()
        for i in range(1000):
            tool.execute(thought=f"thought {i}")
        elapsed = time.perf_counter() - start

        # 1000 calls should take < 1s (< 1ms each)
        assert elapsed < 1.0, f"1000 calls took {elapsed:.3f}s ({elapsed / 1000 * 1000:.3f}ms each)"

    def test_repeated_reset_cycles(self):
        """50 think-reset cycles — no state leakage."""
        tool = ThinkTool(max_deliberation_hops=3)
        for cycle in range(50):
            for i in range(3):
                tool.execute(thought=f"cycle {cycle} step {i}")
            tool.reset_deliberation()

        assert tool._consecutive_thinks == 0
        assert tool._recent_thought_keywords == []


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
