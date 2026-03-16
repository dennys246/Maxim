"""Unit tests for Significance heuristics and learnable weights.

Tests the core functionality of CycleContext evaluation, weight learning,
RPE normalization, and persistence.
"""

from __future__ import annotations

from unittest.mock import Mock

import pytest


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def mock_hippocampus():
    """Mock hippocampus with graph for weight learner."""
    hippo = Mock()
    hippo.graph = Mock()
    hippo.graph.get_associated = Mock(return_value=[])
    return hippo


@pytest.fixture
def weight_learner(tmp_path, mock_hippocampus):
    """Fresh SignificanceWeightLearner with temp persistence."""
    from maxim.decisions.significance import SignificanceWeightLearner

    path = str(tmp_path / "weights.json")
    return SignificanceWeightLearner(path, mock_hippocampus)


@pytest.fixture
def config():
    """Default SignificanceConfig."""
    from maxim.decisions.significance import SignificanceConfig

    return SignificanceConfig()


@pytest.fixture
def cycle_context():
    """Standard CycleContext for testing."""
    from maxim.decisions.significance import CycleContext

    return CycleContext(
        rpe_raw=0.5,
        has_user_input=True,
        novelty=0.7,
        is_plan_boundary=False,
        energy_crossed_threshold=False,
        outcome_valence=0.8,
    )


# ── Sigmoid ───────────────────────────────────────────────────────────────────


class TestSigmoid:
    """Test sigmoid utility function."""

    def test_sigmoid_at_zero(self):
        from maxim.decisions.significance import sigmoid

        assert sigmoid(0.0) == pytest.approx(0.5)

    def test_sigmoid_positive(self):
        from maxim.decisions.significance import sigmoid

        assert sigmoid(10.0) > 0.99

    def test_sigmoid_negative(self):
        from maxim.decisions.significance import sigmoid

        assert sigmoid(-10.0) < 0.01

    def test_sigmoid_clamps_extreme(self):
        from maxim.decisions.significance import sigmoid

        # Should not overflow
        result = sigmoid(1000.0)
        assert 0.0 <= result <= 1.0


# ── CycleContext ──────────────────────────────────────────────────────────────


class TestCycleContext:
    """Test CycleContext dataclass."""

    def test_defaults(self):
        from maxim.decisions.significance import CycleContext

        ctx = CycleContext()
        assert ctx.rpe_raw == 0.0
        assert ctx.has_user_input is False
        assert ctx.novelty == 0.5
        assert ctx.is_plan_boundary is False
        assert ctx.energy_crossed_threshold is False
        assert ctx.outcome_valence == 0.5

    def test_custom_values(self, cycle_context):
        assert cycle_context.rpe_raw == 0.5
        assert cycle_context.has_user_input is True
        assert cycle_context.novelty == 0.7


# ── Heuristic Evaluation Functions ───────────────────────────────────────────


class TestHeuristicEvaluations:
    """Test individual heuristic evaluation functions."""

    def test_eval_rpe_returns_raw(self):
        from maxim.decisions.significance import CycleContext, _eval_rpe

        ctx = CycleContext(rpe_raw=0.75)
        assert _eval_rpe(ctx) == 0.75

    def test_eval_user_interaction_true(self):
        from maxim.decisions.significance import CycleContext, _eval_user_interaction

        ctx = CycleContext(has_user_input=True)
        assert _eval_user_interaction(ctx) == 1.0

    def test_eval_user_interaction_false(self):
        from maxim.decisions.significance import CycleContext, _eval_user_interaction

        ctx = CycleContext(has_user_input=False)
        assert _eval_user_interaction(ctx) == 0.0

    def test_eval_novelty(self):
        from maxim.decisions.significance import CycleContext, _eval_novelty

        ctx = CycleContext(novelty=0.42)
        assert _eval_novelty(ctx) == 0.42

    def test_eval_plan_boundary(self):
        from maxim.decisions.significance import CycleContext, _eval_plan_boundary

        assert _eval_plan_boundary(CycleContext(is_plan_boundary=True)) == 1.0
        assert _eval_plan_boundary(CycleContext(is_plan_boundary=False)) == 0.0

    def test_eval_energy_change(self):
        from maxim.decisions.significance import CycleContext, _eval_energy_change

        assert _eval_energy_change(CycleContext(energy_crossed_threshold=True)) == 1.0
        assert _eval_energy_change(CycleContext(energy_crossed_threshold=False)) == 0.0

    def test_eval_valence_extremity_neutral(self):
        from maxim.decisions.significance import CycleContext, _eval_valence_extremity

        ctx = CycleContext(outcome_valence=0.5)
        assert _eval_valence_extremity(ctx) == pytest.approx(0.0)

    def test_eval_valence_extremity_positive(self):
        from maxim.decisions.significance import CycleContext, _eval_valence_extremity

        ctx = CycleContext(outcome_valence=1.0)
        assert _eval_valence_extremity(ctx) == pytest.approx(1.0)

    def test_eval_valence_extremity_negative(self):
        from maxim.decisions.significance import CycleContext, _eval_valence_extremity

        ctx = CycleContext(outcome_valence=0.0)
        assert _eval_valence_extremity(ctx) == pytest.approx(1.0)


# ── SignificanceWeightLearner ─────────────────────────────────────────────────


class TestSignificanceWeightLearner:
    """Test the weight learner core functionality."""

    def test_initializes_with_all_heuristic_weights(self, weight_learner):
        from maxim.decisions.significance import BASELINE_HEURISTICS

        for h in BASELINE_HEURISTICS:
            assert h.name in weight_learner.weights

    def test_weights_sum_to_one(self, weight_learner):
        total = sum(w.weight for w in weight_learner.weights.values())
        assert total == pytest.approx(1.0, abs=0.01)

    def test_evaluate_returns_score_and_heuristic_scores(
        self, weight_learner, config, cycle_context
    ):
        score, heuristic_scores = weight_learner.evaluate(cycle_context, config)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
        assert "rpe_magnitude" in heuristic_scores
        assert "user_interaction" in heuristic_scores
        assert "novelty" in heuristic_scores

    def test_evaluate_user_input_increases_score(self, weight_learner, config):
        from maxim.decisions.significance import CycleContext

        ctx_no_input = CycleContext(has_user_input=False)
        ctx_with_input = CycleContext(has_user_input=True)

        score_no, _ = weight_learner.evaluate(ctx_no_input, config)
        score_with, _ = weight_learner.evaluate(ctx_with_input, config)

        assert score_with > score_no

    def test_evaluate_high_rpe_increases_score(self, weight_learner, config):
        from maxim.decisions.significance import CycleContext

        # Seed RPE stats so normalization works
        for rpe in [0.1, 0.2, 0.3, 0.4, 0.5]:
            weight_learner.update_rpe_stats(rpe)

        ctx_low = CycleContext(rpe_raw=0.1)
        ctx_high = CycleContext(rpe_raw=0.9)

        score_low, _ = weight_learner.evaluate(ctx_low, config)
        score_high, _ = weight_learner.evaluate(ctx_high, config)

        assert score_high > score_low

    def test_update_rpe_stats(self, weight_learner):
        for rpe in [0.1, 0.2, 0.3, 0.4, 0.5]:
            weight_learner.update_rpe_stats(rpe)

        assert weight_learner.rpe_running_mean == pytest.approx(0.3, abs=0.01)
        assert weight_learner.rpe_running_std > 0

    def test_evaluate_zero_rpe_skips_normalization(self, weight_learner, config):
        from maxim.decisions.significance import CycleContext

        ctx = CycleContext(rpe_raw=0.0)
        score, scores = weight_learner.evaluate(ctx, config)
        assert scores["rpe_magnitude"] == 0.0


class TestSignificanceWeightLearnerPersistence:
    """Test save/load roundtrip."""

    def test_save_load_preserves_weights(self, tmp_path, mock_hippocampus):
        from maxim.decisions.significance import SignificanceWeightLearner

        path = str(tmp_path / "weights.json")
        learner1 = SignificanceWeightLearner(path, mock_hippocampus)

        # Modify a weight
        learner1.weights["novelty"].weight = 0.42
        learner1.update_rpe_stats(0.5)
        learner1.save()

        # Load fresh
        learner2 = SignificanceWeightLearner(path, mock_hippocampus)
        assert learner2.weights["novelty"].weight == pytest.approx(0.42, abs=0.01)

    def test_save_load_preserves_rpe_stats(self, tmp_path, mock_hippocampus):
        from maxim.decisions.significance import SignificanceWeightLearner

        path = str(tmp_path / "weights.json")
        learner1 = SignificanceWeightLearner(path, mock_hippocampus)
        for rpe in [0.1, 0.2, 0.3]:
            learner1.update_rpe_stats(rpe)
        learner1.save()

        learner2 = SignificanceWeightLearner(path, mock_hippocampus)
        assert learner2.rpe_running_mean == pytest.approx(0.2, abs=0.01)

    def test_corrupt_file_triggers_random_init(self, tmp_path, mock_hippocampus):
        from maxim.decisions.significance import SignificanceWeightLearner

        path = str(tmp_path / "weights.json")
        with open(path, "w") as f:
            f.write("{invalid json")

        learner = SignificanceWeightLearner(path, mock_hippocampus)
        # Should have initialized with random weights
        assert len(learner.weights) > 0
        total = sum(w.weight for w in learner.weights.values())
        assert total == pytest.approx(1.0, abs=0.01)


class TestSignificanceWeightLearnerUtility:
    """Test utility harvesting and learning."""

    def test_track_promotion(self, weight_learner):
        scores = {"rpe_magnitude": 0.5, "novelty": 0.7}
        weight_learner.track_promotion("mem_1", scores)
        assert "mem_1" in weight_learner.tracked_memories

    def test_on_memory_deleted_cleans_up(self, weight_learner):
        weight_learner.track_promotion("mem_1", {"rpe_magnitude": 0.5})
        weight_learner.on_memory_deleted("mem_1")
        assert "mem_1" not in weight_learner.tracked_memories

    def test_detect_stuck_false_initially(self, weight_learner):
        assert weight_learner.detect_stuck() is False

    def test_restart_from_scratch_reinitializes(self, weight_learner):
        {n: w.weight for n, w in weight_learner.weights.items()}
        weight_learner.restart_from_scratch()
        # Weights should still sum to 1
        total = sum(w.weight for w in weight_learner.weights.values())
        assert total == pytest.approx(1.0, abs=0.01)

    def test_load_weights_into_heuristics(self, weight_learner):
        from maxim.decisions.significance import BASELINE_HEURISTICS

        heuristics = list(BASELINE_HEURISTICS)
        weight_learner.load_weights_into_heuristics(heuristics)
        for h in heuristics:
            if h.name in weight_learner.weights:
                assert h.weight == weight_learner.weights[h.name].weight


# ── Baseline Heuristics ──────────────────────────────────────────────────────


class TestBaselineHeuristics:
    """Test BASELINE_HEURISTICS configuration."""

    def test_six_heuristics_defined(self):
        from maxim.decisions.significance import BASELINE_HEURISTICS

        assert len(BASELINE_HEURISTICS) == 6

    def test_all_have_evaluate_functions(self):
        from maxim.decisions.significance import BASELINE_HEURISTICS

        for h in BASELINE_HEURISTICS:
            assert h.evaluate is not None

    def test_baseline_weights_sum_to_approximately_one(self):
        from maxim.decisions.significance import BASELINE_HEURISTICS

        total = sum(h.weight for h in BASELINE_HEURISTICS)
        assert total == pytest.approx(0.95, abs=0.1)
