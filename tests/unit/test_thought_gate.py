"""Tests for ThoughtGate (Stage 2).

Covers:
- Refractory guard
- Energy guard
- Scoring + adaptive threshold
- No-scorer fallback (always pass)
- Empty working memory rejection
- Outcome feedback
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from maxim.runtime.gating import (
    GateScore,
    GatingContext,
)
from maxim.runtime.thought_gate import ThoughtGate, ThoughtGateConfig


class _FakeScorer:
    """Scorer that returns a fixed score."""

    def __init__(self, combined: float = 0.8) -> None:
        self._combined = combined

    def score(self, input: Any, context: GatingContext) -> GateScore:
        return GateScore(
            novelty=self._combined,
            salience=self._combined,
            combined=self._combined,
        )


class _FakeWMS:
    """Minimal working memory stub."""

    def __init__(self, entries: int = 3) -> None:
        self._entries = entries
        self._tick = entries

    def recent(self, limit: int = 20) -> list[Any]:
        if self._entries == 0:
            return []
        return [MagicMock(content=f"entry-{i}") for i in range(min(limit, self._entries))]

    @property
    def current_tick(self) -> int:
        return self._tick


class TestThoughtGateRefractory:
    def test_refractory_blocks_immediate_refire(self):
        gate = ThoughtGate(config=ThoughtGateConfig(refractory_ticks=3))
        # First call should pass (no scorer → always pass)
        d1 = gate.should_think(working_memory=_FakeWMS(), current_tick=0)
        assert d1.passed

        # Tick 1: within refractory
        d2 = gate.should_think(working_memory=_FakeWMS(), current_tick=1)
        assert not d2.passed
        assert "refractory" in d2.reason

    def test_refractory_passes_after_window(self):
        gate = ThoughtGate(config=ThoughtGateConfig(refractory_ticks=2))
        gate.should_think(working_memory=_FakeWMS(), current_tick=0)
        # Tick 2: exactly at refractory boundary → should pass
        d = gate.should_think(working_memory=_FakeWMS(), current_tick=2)
        assert d.passed


class TestThoughtGateEnergy:
    def test_energy_blocks_when_exhausted(self):
        tracker = MagicMock()
        tracker.get_token_budget_status.return_value = {
            "budget": 100_000,
            "remaining": 5_000,  # 5% remaining
            "percentage": 95.0,
            "is_over_budget": False,
        }
        gate = ThoughtGate(
            energy_tracker=tracker,
            config=ThoughtGateConfig(min_energy_fraction=0.15, refractory_ticks=0),
        )
        d = gate.should_think(working_memory=_FakeWMS(), current_tick=0)
        assert not d.passed
        assert "energy" in d.reason

    def test_energy_passes_when_available(self):
        tracker = MagicMock()
        tracker.get_token_budget_status.return_value = {
            "budget": 100_000,
            "remaining": 80_000,  # 80% remaining
            "percentage": 20.0,
            "is_over_budget": False,
        }
        gate = ThoughtGate(
            energy_tracker=tracker,
            config=ThoughtGateConfig(refractory_ticks=0),
        )
        d = gate.should_think(working_memory=_FakeWMS(), current_tick=0)
        assert d.passed

    def test_energy_tracker_failure_does_not_block(self):
        tracker = MagicMock()
        tracker.get_token_budget_status.side_effect = RuntimeError("tracker broken")
        gate = ThoughtGate(
            energy_tracker=tracker,
            config=ThoughtGateConfig(refractory_ticks=0),
        )
        d = gate.should_think(working_memory=_FakeWMS(), current_tick=0)
        assert d.passed  # graceful degradation


class TestThoughtGateScoring:
    def test_low_score_blocked(self):
        scorer = _FakeScorer(combined=0.1)
        gate = ThoughtGate(
            scorer=scorer,
            config=ThoughtGateConfig(refractory_ticks=0, min_combined_score=0.4),
        )
        d = gate.should_think(working_memory=_FakeWMS(), current_tick=0)
        assert not d.passed
        assert "below threshold" in d.reason

    def test_high_score_passes(self):
        scorer = _FakeScorer(combined=0.9)
        gate = ThoughtGate(
            scorer=scorer,
            config=ThoughtGateConfig(refractory_ticks=0),
        )
        d = gate.should_think(working_memory=_FakeWMS(), current_tick=0)
        assert d.passed

    def test_no_scorer_always_passes(self):
        gate = ThoughtGate(config=ThoughtGateConfig(refractory_ticks=0))
        d = gate.should_think(working_memory=_FakeWMS(), current_tick=0)
        assert d.passed
        assert d.score.combined == 1.0

    def test_empty_working_memory_rejected(self):
        scorer = _FakeScorer(combined=0.9)
        gate = ThoughtGate(
            scorer=scorer,
            config=ThoughtGateConfig(refractory_ticks=0),
        )
        d = gate.should_think(working_memory=_FakeWMS(entries=0), current_tick=0)
        assert not d.passed
        assert "empty" in d.reason


class TestThoughtGateOutcome:
    def test_record_outcome_useful(self):
        gate = ThoughtGate()
        d = gate.should_think(working_memory=_FakeWMS(), current_tick=0)
        # Should not raise
        gate.record_outcome(d, was_useful=True)

    def test_record_outcome_not_useful(self):
        gate = ThoughtGate()
        d = gate.should_think(working_memory=_FakeWMS(), current_tick=0)
        gate.record_outcome(d, was_useful=False)


class TestThoughtGateComposite:
    def test_full_cascade_refractory_first(self):
        """Refractory fires before energy or scoring."""
        tracker = MagicMock()
        tracker.get_token_budget_status.return_value = {
            "budget": 100_000,
            "remaining": 80_000,  # plenty of energy
            "percentage": 20.0,
            "is_over_budget": False,
        }
        gate = ThoughtGate(
            scorer=_FakeScorer(combined=0.9),
            energy_tracker=tracker,
            config=ThoughtGateConfig(refractory_ticks=5),
        )
        # First pass — enough energy, high score
        d1 = gate.should_think(working_memory=_FakeWMS(), current_tick=0)
        assert d1.passed
        # Immediate refire blocked by refractory (not energy, not scoring)
        d2 = gate.should_think(working_memory=_FakeWMS(), current_tick=1)
        assert not d2.passed
        assert "refractory" in d2.reason


class TestGoalRewardBiasModulation:
    """Tests for goal_reward_bias parameter (Phase 3)."""

    def test_positive_bias_lowers_threshold(self):
        """Positive goal bias makes borderline scores pass."""
        scorer = _FakeScorer(combined=0.45)
        gate = ThoughtGate(
            scorer=scorer,
            config=ThoughtGateConfig(refractory_ticks=0, min_combined_score=0.1),
        )
        # Without bias: score 0.45 may be below the adaptive threshold
        d1 = gate.should_think(working_memory=_FakeWMS(), current_tick=0)
        threshold_no_bias = d1.threshold_used

        # With positive bias: threshold drops (floor is 0.1, room for bias)
        d2 = gate.should_think(working_memory=_FakeWMS(), current_tick=100, goal_reward_bias=0.2)
        threshold_with_bias = d2.threshold_used
        assert threshold_with_bias < threshold_no_bias

    def test_negative_bias_raises_threshold(self):
        """Negative goal bias makes it harder to deliberate."""
        scorer = _FakeScorer(combined=0.8)
        gate = ThoughtGate(
            scorer=scorer,
            config=ThoughtGateConfig(refractory_ticks=0, min_combined_score=0.3),
        )
        # Without bias: should pass
        d1 = gate.should_think(working_memory=_FakeWMS(), current_tick=0)
        assert d1.passed

        # With large negative bias: threshold rises above score
        d2 = gate.should_think(working_memory=_FakeWMS(), current_tick=100, goal_reward_bias=-0.5)
        threshold_with_neg = d2.threshold_used
        assert threshold_with_neg > d1.threshold_used

    def test_zero_bias_is_default(self):
        """Zero bias has no effect (backward compatible)."""
        gate = ThoughtGate(config=ThoughtGateConfig(refractory_ticks=0))
        d1 = gate.should_think(working_memory=_FakeWMS(), current_tick=0)
        d2 = gate.should_think(working_memory=_FakeWMS(), current_tick=100, goal_reward_bias=0.0)
        assert d1.threshold_used == d2.threshold_used

    def test_threshold_floor_respected(self):
        """Even with large positive bias, threshold doesn't go below min."""
        scorer = _FakeScorer(combined=0.1)
        gate = ThoughtGate(
            scorer=scorer,
            config=ThoughtGateConfig(refractory_ticks=0, min_combined_score=0.3),
        )
        d = gate.should_think(working_memory=_FakeWMS(), current_tick=0, goal_reward_bias=10.0)
        # min_combined_score = 0.3, score = 0.1 → still fails
        assert not d.passed
        assert d.threshold_used >= 0.3
