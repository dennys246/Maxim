"""Unit tests for Nucleus Accumbens (NAc) - Causal inference and TD learning.

Tests the core functionality of causal link creation, observation,
prediction, and persistence.
"""

from __future__ import annotations

import pytest


class TestNAcObservation:
    """Test causal link creation and updates."""

    def test_observe_creates_new_link(self, nac, valence_positive):
        """First observation creates a new causal link."""
        link = nac.observe(
            event_type="tool",
            event_signature="internet_search",
            outcome_type="result",
            outcome_signature="success",
            outcome_valence=valence_positive,
            delta_seconds=2.0,
        )

        assert link is not None
        assert link.observation_count == 1
        assert link.confidence == 0.5  # Initial confidence
        assert link.event_signature == "internet_search"

    def test_observe_updates_existing_link(self, nac, valence_positive):
        """Repeated observations update the existing link."""
        # First observation
        nac.observe(
            event_type="tool",
            event_signature="look_around",
            outcome_type="result",
            outcome_signature="found_object",
            outcome_valence=valence_positive,
            delta_seconds=1.0,
            context={"mode": "exploration"},
        )

        # Same event-outcome pair
        link = nac.observe(
            event_type="tool",
            event_signature="look_around",
            outcome_type="result",
            outcome_signature="found_object",
            outcome_valence=valence_positive,
            delta_seconds=1.5,
            context={"mode": "exploration"},
        )

        assert link.observation_count == 2
        assert link.confidence > 0.5  # Should increase with observations

    def test_different_contexts_create_separate_links(self, nac, valence_positive):
        """Different contexts create separate causal links."""
        link1 = nac.observe(
            event_type="tool",
            event_signature="grasp",
            outcome_type="result",
            outcome_signature="success",
            outcome_valence=valence_positive,
            delta_seconds=1.0,
            context={"object": "mug"},
        )

        link2 = nac.observe(
            event_type="tool",
            event_signature="grasp",
            outcome_type="result",
            outcome_signature="success",
            outcome_valence=valence_positive,
            delta_seconds=1.0,
            context={"object": "plate"},  # Different context
        )

        # Should be different links
        assert link1.id != link2.id

    def test_observe_records_temporal_delta(self, nac, valence_positive):
        """Observations record temporal delays."""
        nac.observe(
            event_type="tool",
            event_signature="wait",
            outcome_type="result",
            outcome_signature="done",
            outcome_valence=valence_positive,
            delta_seconds=5.0,
        )
        nac.observe(
            event_type="tool",
            event_signature="wait",
            outcome_type="result",
            outcome_signature="done",
            outcome_valence=valence_positive,
            delta_seconds=3.0,
        )

        links = nac.get_links_for_event("wait")
        assert len(links) >= 1
        link = links[0]
        assert len(link.temporal_delta.observed_deltas) == 2
        assert link.temporal_delta.mean == 4.0


class TestNAcPrediction:
    """Test outcome prediction."""

    def test_predict_returns_none_for_unknown_event(self, nac):
        """Unknown events return None prediction."""
        prediction = nac.predict("tool", "never_seen_before")
        assert prediction is None

    def test_predict_returns_prediction_for_known_event(self, nac, valence_positive):
        """Known events return predictions."""
        # Create some observations
        for _ in range(3):
            nac.observe(
                event_type="tool",
                event_signature="reliable_tool",
                outcome_type="result",
                outcome_signature="success",
                outcome_valence=valence_positive,
                delta_seconds=1.0,
            )

        prediction = nac.predict("tool", "reliable_tool")

        assert prediction is not None
        assert prediction.event_signature == "reliable_tool"
        assert prediction.confidence > 0.3

    def test_predict_uses_highest_confidence_link(self, nac, valence_positive, valence_negative):
        """Prediction uses the highest-confidence link."""
        # Create many positive observations
        for _ in range(10):
            nac.observe(
                event_type="tool",
                event_signature="mixed_tool",
                outcome_type="result",
                outcome_signature="success",
                outcome_valence=valence_positive,
                delta_seconds=1.0,
            )

        # Create one negative observation
        nac.observe(
            event_type="tool",
            event_signature="mixed_tool",
            outcome_type="result",
            outcome_signature="failure",
            outcome_valence=valence_negative,
            delta_seconds=1.0,
        )

        prediction = nac.predict("tool", "mixed_tool")

        # Should predict based on more confident link (positive)
        assert prediction is not None
        assert prediction.predicted_value > 0.5

    def test_predict_respects_min_confidence_threshold(self, nac, valence_positive):
        """Low-confidence links are excluded from predictions."""
        from maxim.decisions.nac import NAc, NACConfig

        # Create NAc with high threshold
        strict_nac = NAc(config=NACConfig(min_confidence_threshold=0.9))

        # Single observation = low confidence
        strict_nac.observe(
            event_type="tool",
            event_signature="new_tool",
            outcome_type="result",
            outcome_signature="success",
            outcome_valence=valence_positive,
            delta_seconds=1.0,
        )

        prediction = strict_nac.predict("tool", "new_tool")
        assert prediction is None  # Below confidence threshold

    def test_cold_start_priors(self, nac):
        """Priors provide predictions when no links exist."""
        nac.set_prior("tool", "untested_tool", predicted_value=0.7, confidence=0.3)

        prediction = nac.predict("tool", "untested_tool")

        assert prediction is not None
        assert prediction.predicted_value == 0.7
        assert prediction.confidence == 0.3

    def test_predict_all_outcomes(self, nac, valence_positive, valence_negative):
        """predict_all_outcomes returns multiple possible outcomes."""
        # Same event, different outcomes
        nac.observe(
            event_type="tool",
            event_signature="risky_action",
            outcome_type="result",
            outcome_signature="success",
            outcome_valence=valence_positive,
            delta_seconds=1.0,
        )
        nac.observe(
            event_type="tool",
            event_signature="risky_action",
            outcome_type="result",
            outcome_signature="partial_success",
            outcome_valence=valence_positive,
            delta_seconds=2.0,
        )

        # Observe enough for confidence
        for _ in range(5):
            nac.observe(
                event_type="tool",
                event_signature="risky_action",
                outcome_type="result",
                outcome_signature="success",
                outcome_valence=valence_positive,
                delta_seconds=1.0,
            )

        predictions = nac.predict_all_outcomes("tool", "risky_action")

        assert len(predictions) >= 1


class TestNAcQueries:
    """Test link query methods."""

    def test_get_links_for_event(self, nac, valence_positive):
        """Query links by event signature."""
        nac.observe(
            event_type="tool",
            event_signature="specific_tool",
            outcome_type="result",
            outcome_signature="success",
            outcome_valence=valence_positive,
            delta_seconds=1.0,
        )

        links = nac.get_links_for_event("specific_tool")
        assert len(links) == 1
        assert links[0].event_signature == "specific_tool"

    def test_get_positive_outcomes(self, nac, valence_positive, valence_negative):
        """Query only positive outcome links."""
        nac.observe(
            event_type="tool",
            event_signature="tool_a",
            outcome_type="result",
            outcome_signature="success",
            outcome_valence=valence_positive,
            delta_seconds=1.0,
        )
        nac.observe(
            event_type="tool",
            event_signature="tool_a",
            outcome_type="result",
            outcome_signature="failure",
            outcome_valence=valence_negative,
            delta_seconds=1.0,
            context={"different": True},  # Different context for separate link
        )

        positive = nac.get_positive_outcomes("tool_a")
        negative = nac.get_negative_outcomes("tool_a")

        assert len(positive) >= 1
        assert len(negative) >= 1

    def test_get_links_for_memory(self, nac, valence_positive):
        """Query links by memory ID."""
        nac.observe(
            event_type="tool",
            event_signature="memory_linked",
            outcome_type="result",
            outcome_signature="success",
            outcome_valence=valence_positive,
            delta_seconds=1.0,
            memory_id="mem_123",
        )

        links = nac.get_links_for("mem_123")
        assert len(links) == 1

    def test_scan_links_for_keywords_substring_match(self, nac, valence_positive):
        """Narrative keywords find compound tool signatures via substring."""
        # Boost confidence so links pass the default 0.3 floor.
        for _ in range(10):
            nac.observe(
                event_type="tool",
                event_signature="tool:rusty_sword_slash",
                outcome_type="result",
                outcome_signature="success",
                outcome_valence=valence_positive,
                delta_seconds=1.0,
            )

        # Each fragment from AffordanceDecompositionStrategy("rusty_sword_slash")
        # should now find the compound signature.
        for kw in ["rusty", "sword", "slash"]:
            matches = nac.scan_links_for_keywords([kw])
            assert any(lk.event_signature == "tool:rusty_sword_slash" for lk in matches), (
                f"keyword {kw!r} did not match tool:rusty_sword_slash"
            )

    def test_scan_links_for_keywords_dedupes_and_sorts(self, nac, valence_positive):
        """Multiple keywords matching the same link return it once, sorted by confidence."""
        for _ in range(15):
            nac.observe(
                event_type="tool",
                event_signature="tool:fire_breath",
                outcome_type="result",
                outcome_signature="success",
                outcome_valence=valence_positive,
                delta_seconds=1.0,
            )
        for _ in range(5):
            nac.observe(
                event_type="tool",
                event_signature="tool:water_splash",
                outcome_type="result",
                outcome_signature="success",
                outcome_valence=valence_positive,
                delta_seconds=1.0,
            )

        # Both "fire" and "breath" hit the same link; "water" hits a different one.
        matches = nac.scan_links_for_keywords(["fire", "breath", "water"])
        sigs = [lk.event_signature for lk in matches]
        # Each signature appears at most once.
        assert len(sigs) == len(set(sigs))
        # Higher-confidence link comes first.
        assert sigs[0] == "tool:fire_breath"

    def test_scan_links_for_keywords_drops_short_kws(self, nac, valence_positive):
        """Stop-words below min_keyword_length don't match everything."""
        for _ in range(10):
            nac.observe(
                event_type="tool",
                event_signature="tool:a_very_long_tool_name",
                outcome_type="result",
                outcome_signature="success",
                outcome_valence=valence_positive,
                delta_seconds=1.0,
            )
        # "a" would substring-match "tool:a_very_long_tool_name" if not filtered.
        matches = nac.scan_links_for_keywords(["a", "to", "of"])
        assert matches == []

    def test_scan_links_for_keywords_respects_confidence_floor(self, nac, valence_positive):
        """Links below min_confidence are filtered out."""
        # Single observation: confidence stays at the bootstrap value (~0.5
        # base * RW step), well below 0.9.
        nac.observe(
            event_type="tool",
            event_signature="tool:rare_tool",
            outcome_type="result",
            outcome_signature="success",
            outcome_valence=valence_positive,
            delta_seconds=1.0,
        )
        matches = nac.scan_links_for_keywords(["rare"], min_confidence=0.9)
        assert matches == []


class TestNAcMaintenance:
    """Test maintenance operations."""

    def test_enforce_limits_removes_low_confidence_links(self, valence_positive):
        """Enforces max_links by removing lowest confidence."""
        from maxim.decisions.nac import NAc, NACConfig

        small_nac = NAc(config=NACConfig(max_links=5))

        # Create more links than allowed
        for i in range(10):
            small_nac.observe(
                event_type="tool",
                event_signature=f"tool_{i}",
                outcome_type="result",
                outcome_signature="success",
                outcome_valence=valence_positive,
                delta_seconds=1.0,
            )

        # Should have enforced limit
        assert len(small_nac) <= 5

    def test_decay_reduces_confidence(self, nac, valence_positive):
        """Decay reduces link confidence."""
        link = nac.observe(
            event_type="tool",
            event_signature="decaying",
            outcome_type="result",
            outcome_signature="success",
            outcome_valence=valence_positive,
            delta_seconds=1.0,
        )

        initial_confidence = link.confidence
        nac.decay_all(factor=0.5)

        assert link.confidence == initial_confidence * 0.5

    def test_decay_eligibility_reduces_traces(self, nac):
        """Eligibility traces decay toward zero each tick."""
        nac.update_eligibility("agent-1", "node-a", 1.0)
        nac.update_eligibility("agent-1", "node-b", 0.5)

        nac.decay_eligibility(factor=0.9)

        assert nac._eligibility[("agent-1", "node-a")] == pytest.approx(0.9)
        assert nac._eligibility[("agent-1", "node-b")] == pytest.approx(0.45)

    def test_decay_eligibility_prunes_below_threshold(self, nac):
        """Traces below 0.01 are removed entirely."""
        nac.update_eligibility("agent-1", "node-a", 0.02)

        nac.decay_eligibility(factor=0.4)  # 0.02 * 0.4 = 0.008 < 0.01

        assert ("agent-1", "node-a") not in nac._eligibility

    def test_decay_eligibility_multiple_ticks(self, nac):
        """After many ticks, traces converge to zero."""
        nac.update_eligibility("agent-1", "node-a", 1.0)

        for _ in range(100):
            nac.decay_eligibility(factor=0.9)

        # 0.9^100 ≈ 0.000027 — should be pruned
        assert ("agent-1", "node-a") not in nac._eligibility

    def test_decay_reward_biases_reduces_toward_zero(self, nac):
        """Reward biases decay toward zero each tick."""
        nac.credit_node("agent-1", "node-a", reward=1.0)
        initial = nac.reward_bias("agent-1", "node-a")
        assert initial > 0

        nac.decay_reward_biases()

        reduced = nac.reward_bias("agent-1", "node-a")
        assert reduced < initial

    def test_decay_reward_biases_prunes_near_zero(self, nac):
        """Biases below 0.001 are removed."""
        nac._reward_bias[("agent-1", "node-a")] = 0.0005

        pruned = nac.decay_reward_biases()

        assert pruned == 1
        assert ("agent-1", "node-a") not in nac._reward_bias

    def test_distribute_reward_skips_decayed_traces(self, nac):
        """After full decay, distribute_reward credits nothing."""
        nac.update_eligibility("agent-1", "node-a", 1.0)

        # Decay to zero
        for _ in range(200):
            nac.decay_eligibility(factor=0.9)

        credited = nac.distribute_reward("agent-1", reward=1.0)
        assert credited == []


class TestTemporalEligibility:
    """Tests for SCN-coupled temporal eligibility credit (Stage 2)."""

    def _make_nac(self, temporal_credit_weight=0.3):
        from maxim.decisions.nac import NAc, NACConfig

        return NAc(config=NACConfig(temporal_credit_weight=temporal_credit_weight))

    def _make_sig(self, **overrides):
        import time

        from maxim.time.temporal_signature import TemporalSignature

        defaults = {
            "timestamp": time.time(),  # Recent timestamp so anchors survive pruning
            "circadian_phase": 0.5,
            "weekly_phase": 0.3,
            "monthly_phase": 0.2,
            "annual_phase": 0.1,
        }
        defaults.update(overrides)
        return TemporalSignature(**defaults)

    def test_temporal_anchor_stored_with_eligibility(self):
        """update_eligibility with temporal_sig stores anchor."""
        nac = self._make_nac()
        sig = self._make_sig()
        nac.update_eligibility("agent-1", "node-a", 1.0, temporal_sig=sig)

        assert ("agent-1", "node-a") in nac._temporal_anchors
        orig_act, stored_sig = nac._temporal_anchors[("agent-1", "node-a")]
        assert orig_act == 1.0
        assert stored_sig is sig

    def test_no_temporal_sig_no_anchor(self):
        """update_eligibility without temporal_sig creates no anchor."""
        nac = self._make_nac()
        nac.update_eligibility("agent-1", "node-a", 1.0)

        assert ("agent-1", "node-a") not in nac._temporal_anchors

    def test_temporal_fallback_credits_decayed_node(self):
        """After fast-decay expires, temporal anchor still credits node."""
        nac = self._make_nac(temporal_credit_weight=0.5)
        sig = self._make_sig()
        nac.update_eligibility("agent-1", "node-a", 1.0, temporal_sig=sig)

        # Decay to zero
        for _ in range(200):
            nac.decay_eligibility(factor=0.9)

        # Fast-decay trace should be gone
        assert ("agent-1", "node-a") not in nac._eligibility

        # But temporal anchor should remain
        assert ("agent-1", "node-a") in nac._temporal_anchors

        # distribute_reward should still credit via temporal path
        credited = nac.distribute_reward("agent-1", reward=1.0)
        assert len(credited) > 0
        assert credited[0][0] == "node-a"

    def test_fast_decay_takes_priority_over_temporal(self):
        """When fast-decay trace exists, temporal path doesn't fire."""
        nac = self._make_nac(temporal_credit_weight=0.5)
        sig = self._make_sig()
        nac.update_eligibility("agent-1", "node-a", 1.0, temporal_sig=sig)

        # Don't decay — fast-decay trace is still alive
        credited = nac.distribute_reward("agent-1", reward=1.0)
        assert len(credited) == 1
        # Credit should be proportional to fast-decay strength (1.0),
        # not the reduced temporal weight
        assert credited[0][1] == pytest.approx(1.0)

    def test_temporal_credit_weight_scales_credit(self):
        """temporal_credit_weight scales the temporal fallback credit."""
        nac_low = self._make_nac(temporal_credit_weight=0.1)
        nac_high = self._make_nac(temporal_credit_weight=0.9)

        sig = self._make_sig()
        nac_low.update_eligibility("agent-1", "node-a", 1.0, temporal_sig=sig)
        nac_high.update_eligibility("agent-1", "node-a", 1.0, temporal_sig=sig)

        # Decay both fully
        for _ in range(200):
            nac_low.decay_eligibility(factor=0.9)
            nac_high.decay_eligibility(factor=0.9)

        # Both should credit, but amounts differ
        # (total is normalized to reward, but the temporal_strength differs)
        credited_low = nac_low.distribute_reward("agent-1", reward=1.0)
        credited_high = nac_high.distribute_reward("agent-1", reward=1.0)

        # Both should credit since it's the only node
        assert len(credited_low) > 0
        assert len(credited_high) > 0

    def test_env_var_overrides_temporal_credit_weight(self):
        """MAXIM_NAC_TEMPORAL_CREDIT_WEIGHT env var overrides config."""
        import os

        from maxim.decisions.nac import NAc, NACConfig

        os.environ["MAXIM_NAC_TEMPORAL_CREDIT_WEIGHT"] = "0.7"
        try:
            nac = NAc(config=NACConfig(temporal_credit_weight=0.3))
            assert nac.config.temporal_credit_weight == pytest.approx(0.7)
        finally:
            os.environ.pop("MAXIM_NAC_TEMPORAL_CREDIT_WEIGHT", None)

    def test_env_var_clamped(self):
        """Env var is clamped to [0.05, 1.0]."""
        import os

        from maxim.decisions.nac import NAc

        os.environ["MAXIM_NAC_TEMPORAL_CREDIT_WEIGHT"] = "5.0"
        try:
            nac = NAc()
            assert nac.config.temporal_credit_weight == pytest.approx(1.0)
        finally:
            os.environ.pop("MAXIM_NAC_TEMPORAL_CREDIT_WEIGHT", None)

    def test_agent_id_scoping(self):
        """Temporal credit is scoped to agent_id — no cross-agent leaking."""
        nac = self._make_nac(temporal_credit_weight=0.5)
        sig = self._make_sig()
        nac.update_eligibility("agent-A", "node-a", 1.0, temporal_sig=sig)
        nac.update_eligibility("agent-B", "node-b", 1.0, temporal_sig=sig)

        # Decay both
        for _ in range(200):
            nac.decay_eligibility(factor=0.9)

        # Agent A reward should only credit node-a
        credited = nac.distribute_reward("agent-A", reward=1.0)
        credited_ids = [nid for nid, _ in credited]
        assert "node-a" in credited_ids
        assert "node-b" not in credited_ids


class TestTemporalAnchorPruning:
    """Test anchor pruning edge cases."""

    def _make_nac(self, temporal_window_seconds: float = 300.0):
        from maxim.decisions.nac import NAc, NACConfig

        return NAc(NACConfig(temporal_window_seconds=temporal_window_seconds))

    def test_anchors_survive_in_short_session(self):
        """In a short sim (<5 min), ALL anchors survive — pruning never fires.

        Regression guard: anchors are pruned only when fast-decay expires
        AND age > temporal_window_seconds. With default 300s window,
        anchors created within the last 5 minutes are always retained.
        """
        from maxim.time.temporal_signature import TemporalSignature

        nac = self._make_nac(temporal_window_seconds=300.0)

        sig = TemporalSignature.now()
        nac.update_eligibility("agent", "node-a", 1.0, temporal_sig=sig)
        nac.update_eligibility("agent", "node-b", 0.8, temporal_sig=sig)

        # Run enough decay cycles to expire fast-decay traces
        for _ in range(200):
            nac.decay_eligibility(factor=0.9)

        # Fast-decay should be gone
        active = {k: v for k, v in nac._eligibility.items() if v > 0.01}
        assert not active, "Fast-decay traces should be expired"

        # But temporal anchors survive (session is <5 min old)
        assert ("agent", "node-a") in nac._temporal_anchors
        assert ("agent", "node-b") in nac._temporal_anchors

    def test_anchors_pruned_after_temporal_window(self):
        """Anchors ARE pruned when age exceeds temporal_window_seconds."""
        from maxim.time.temporal_signature import TemporalSignature

        nac = self._make_nac(temporal_window_seconds=0.0)  # Zero window = prune immediately

        sig = TemporalSignature.now()
        nac.update_eligibility("agent", "node-a", 1.0, temporal_sig=sig)

        # Decay until fast-decay expires
        for _ in range(200):
            nac.decay_eligibility(factor=0.9)

        # With 0s window, anchor should be pruned
        assert ("agent", "node-a") not in nac._temporal_anchors


class TestGoalRewardBias:
    """Tests for _goal_reward_bias (bidirectional, for ThoughtGate)."""

    def _make_nac(self):
        from maxim.decisions.nac import NAc

        return NAc()

    def test_credit_goal_positive(self):
        """Positive credit creates positive bias."""
        nac = self._make_nac()
        nac.credit_goal("escape", 1.0)
        assert nac.get_goal_reward_bias("escape") > 0

    def test_credit_goal_negative(self):
        """Negative credit creates negative bias (indirect pathway)."""
        nac = self._make_nac()
        nac.credit_goal("negotiate", -1.0)
        assert nac.get_goal_reward_bias("negotiate") < 0

    def test_credit_goal_none_is_noop(self):
        """credit_goal(None, ...) must not create a phantom None key."""
        nac = self._make_nac()
        nac.credit_goal(None, 1.0)
        assert nac.get_goal_reward_bias(None) == 0.0
        assert None not in nac._goal_reward_bias

    def test_get_goal_reward_bias_none_returns_zero(self):
        """get_goal_reward_bias(None) always returns 0.0."""
        nac = self._make_nac()
        nac.credit_goal("escape", 1.0)
        assert nac.get_goal_reward_bias(None) == 0.0

    def test_goal_bias_clamped_bidirectional(self):
        """Goal bias clamps to [-max, +max], unlike _reward_bias [0, max]."""
        nac = self._make_nac()
        cap = nac.config.max_reward_bias

        # Max out positive
        for _ in range(100):
            nac.credit_goal("good_goal", 1.0)
        assert nac.get_goal_reward_bias("good_goal") == pytest.approx(cap, abs=0.001)

        # Max out negative
        for _ in range(100):
            nac.credit_goal("bad_goal", -1.0)
        assert nac.get_goal_reward_bias("bad_goal") == pytest.approx(-cap, abs=0.001)

    def test_decay_goal_reward_biases(self):
        """Biases decay toward zero over time."""
        nac = self._make_nac()
        nac.credit_goal("escape", 1.0)
        initial = nac.get_goal_reward_bias("escape")
        assert initial > 0

        for _ in range(10):
            nac.decay_goal_reward_biases()

        decayed = nac.get_goal_reward_bias("escape")
        assert 0 < decayed < initial

    def test_decay_prunes_near_zero(self):
        """Biases below 0.001 are pruned on decay."""
        nac = self._make_nac()
        nac.credit_goal("ephemeral", 0.01)

        # Decay until pruned
        for _ in range(500):
            pruned = nac.decay_goal_reward_biases()
            if pruned > 0:
                break

        assert "ephemeral" not in nac._goal_reward_bias

    def test_serialization_roundtrip(self):
        """goal_reward_bias survives dump/load_state."""
        nac = self._make_nac()
        nac.credit_goal("escape", 1.0)
        nac.credit_goal("negotiate", -1.0)

        state = nac.dump()
        assert "goal_reward_bias" in state
        assert state["goal_reward_bias"]["escape"] > 0
        assert state["goal_reward_bias"]["negotiate"] < 0

        # Load into fresh NAc
        nac2 = self._make_nac()
        nac2.load_state(state)
        assert nac2.get_goal_reward_bias("escape") > 0
        assert nac2.get_goal_reward_bias("negotiate") < 0

    def test_load_state_backward_compatible(self):
        """Old snapshots without goal_reward_bias load cleanly."""
        nac = self._make_nac()
        old_state = {"links": {}, "outcome_index": {}, "priors": {}, "total_observations": 0, "reward_bias": {}}
        nac.load_state(old_state)
        assert nac._goal_reward_bias == {}
        assert nac.get_goal_reward_bias("anything") == 0.0


class TestNAcPersistence:
    """Test save/load roundtrip."""

    def test_save_load_preserves_links(self, nac, valence_positive, tmp_path):
        """Links survive save/load cycle."""
        # Create some links
        nac.observe(
            event_type="tool",
            event_signature="persistent_tool",
            outcome_type="result",
            outcome_signature="success",
            outcome_valence=valence_positive,
            delta_seconds=2.5,
        )

        path = tmp_path / "nac.json"
        nac.save(str(path))

        # Load into fresh instance
        from maxim.decisions.nac import NAc

        nac2 = NAc()
        nac2.load(str(path))

        assert len(nac2) == len(nac)
        links = nac2.get_links_for_event("persistent_tool")
        assert len(links) == 1
        assert links[0].temporal_delta.mean == 2.5

    def test_save_load_default_path_from_config(self, valence_positive, tmp_path):
        """save() and load() fall back to NACConfig.persistence_path when called with no args."""
        from maxim.decisions.nac import NAc, NACConfig

        path = tmp_path / "nac_default.json"
        config = NACConfig(persistence_path=str(path))
        nac = NAc(config=config)
        nac.observe(
            event_type="tool",
            event_signature="default_path_tool",
            outcome_type="result",
            outcome_signature="success",
            outcome_valence=valence_positive,
            delta_seconds=1.0,
        )

        nac.save()
        assert path.exists()

        nac2 = NAc(config=config)
        nac2.load()
        assert len(nac2.get_links_for_event("default_path_tool")) == 1

    def test_save_load_raise_without_path(self):
        """save() and load() raise ValueError when no path is set anywhere."""
        from maxim.decisions.nac import NAc

        nac = NAc()  # NACConfig.persistence_path defaults to None
        with pytest.raises(ValueError, match="persistence_path"):
            nac.save()
        with pytest.raises(ValueError, match="persistence_path"):
            nac.load()

    def test_save_load_preserves_stats(self, nac, valence_positive, tmp_path):
        """Statistics survive save/load cycle."""
        for i in range(5):
            nac.observe(
                event_type="tool",
                event_signature=f"tool_{i}",
                outcome_type="result",
                outcome_signature="success",
                outcome_valence=valence_positive,
                delta_seconds=1.0,
            )

        path = tmp_path / "nac.json"
        nac.save(str(path))

        from maxim.decisions.nac import NAc

        nac2 = NAc()
        nac2.load(str(path))

        assert nac2.stats()["total_observations"] == nac.stats()["total_observations"]


class TestNAcStats:
    """Test statistics collection."""

    def test_stats_returns_expected_keys(self, nac, valence_positive):
        """Stats returns all expected keys."""
        nac.observe(
            event_type="tool",
            event_signature="test",
            outcome_type="result",
            outcome_signature="success",
            outcome_valence=valence_positive,
            delta_seconds=1.0,
        )

        stats = nac.stats()

        assert "total_links" in stats
        assert "event_signatures" in stats
        assert "total_observations" in stats
        assert "pending_events" in stats

    def test_len_returns_total_links(self, nac, valence_positive):
        """__len__ returns total link count."""
        assert len(nac) == 0

        nac.observe(
            event_type="tool",
            event_signature="test1",
            outcome_type="result",
            outcome_signature="success",
            outcome_valence=valence_positive,
            delta_seconds=1.0,
        )

        assert len(nac) == 1


class TestContextSimilarity:
    """Regression guards for the directional ``_context_similarity``.

    The Stage 2 substrate P2 pre-merge review caught that the old
    implementation used ``len(keys_union)`` as the denominator, which
    silently diluted legitimate matches whenever the outcome side
    carried more keys than the pending event. The fix: ``len(ctx1)``
    (event-side only) so extra outcome-side keys don't hurt
    attribution.

    Call convention (enforced by all in-file call sites):
    ``_context_similarity(event_or_stored_link, outcome_or_query)``.
    """

    def test_full_match_event_inside_rich_outcome(self, nac):
        """Sparse event context fully contained in rich outcome = 1.0.

        This is the regression guard: pre-Stage-2 the union-of-keys
        denominator made this case ratio = 2/7 ≈ 0.29, below the 0.5
        threshold. Post-fix it's 2/2 = 1.0.
        """
        event_ctx = {"source": "embodiment", "entity": "rusty_sword"}
        outcome_ctx = {
            "source": "embodiment",
            "entity": "rusty_sword",
            "entity_type": "weapon",
            "failure_mode": "shatter",
            "composes": [],
            "sensor_readings": {"durability": 0.05},
            "intensity": 0.6,
        }
        assert nac._context_similarity(event_ctx, outcome_ctx) == 1.0

    def test_partial_match(self, nac):
        """One of two event keys matches -> 0.5."""
        event_ctx = {"source": "embodiment", "entity": "rusty_sword"}
        outcome_ctx = {"source": "embodiment", "entity": "longbow"}
        assert nac._context_similarity(event_ctx, outcome_ctx) == 0.5

    def test_no_match(self, nac):
        """Disjoint keys -> 0.0. All event keys present but values differ -> 0.0."""
        # Disjoint key sets
        assert nac._context_similarity({"a": 1, "b": 2}, {"c": 3, "d": 4}) == 0.0
        # Shared keys, different values (non-string)
        assert nac._context_similarity({"a": 1, "b": 2}, {"a": 9, "b": 8}) == 0.0

    def test_empty_context_neutral(self, nac):
        """Empty context on either side returns 0.5 neutral."""
        assert nac._context_similarity({}, {"a": 1}) == 0.5
        assert nac._context_similarity({"a": 1}, {}) == 0.5

    def test_case_insensitive_string_match(self, nac):
        """String values match case-insensitively at 0.8 weight."""
        event_ctx = {"source": "EMBODIMENT"}
        outcome_ctx = {"source": "embodiment"}
        assert nac._context_similarity(event_ctx, outcome_ctx) == 0.8

    def test_outcome_extra_keys_do_not_dilute(self, nac):
        """Adding extra keys to the outcome side MUST NOT lower similarity.

        This is the inverse regression guard: if a future refactor
        changes the denominator back to union-of-keys, this test fails
        because adding extra outcome keys would drop the ratio.
        """
        event_ctx = {"source": "embodiment", "entity": "sword"}
        baseline = nac._context_similarity(event_ctx, {"source": "embodiment", "entity": "sword"})
        enriched = nac._context_similarity(
            event_ctx,
            {
                "source": "embodiment",
                "entity": "sword",
                "extra_1": "x",
                "extra_2": "y",
                "extra_3": "z",
            },
        )
        assert baseline == enriched == 1.0, (
            f"outcome-side extra keys diluted the match: baseline={baseline}, enriched={enriched} — "
            "the union-of-keys bug has returned"
        )

    def test_record_outcome_full_matches_despite_rich_context(self, nac, valence_negative):
        """End-to-end: pending event with slim context links to outcome with rich context.

        Simulates the SEM pain cascade attribution path. If this fails
        the slim-context workaround in create_pain_nac_subscriber would
        have to come back.
        """
        nac.record_event(
            event_type="action",
            event_signature="slash:rusty_sword",
            context={"source": "embodiment", "entity": "rusty_sword"},
        )
        assert len(nac._pending_events) == 1

        nac.record_outcome_full(
            outcome_type="pain",
            outcome_signature="pain:embodiment:rusty_sword:shatter",
            outcome_valence=valence_negative,
            context={
                "source": "embodiment",
                "entity": "rusty_sword",
                "entity_type": "weapon",
                "failure_mode": "shatter",
                "composes": [],
                "sensor_readings": {"durability": 0.05},
                "intensity": 0.6,
            },
        )
        links = nac._links.get("slash:rusty_sword", [])
        assert len(links) == 1, f"rich-context outcome failed to link: {dict(nac._links)}"
        assert links[0].outcome_valence == valence_negative


class TestClusterRewardBiasDecayTauSplit:
    """Phase 1 of cluster_reward_bias_decay_tau_split.md.

    Pins the split between ``reward_bias_decay_tau`` (per-tick threshold
    modulation, default 50.0) and ``cluster_reward_bias_decay_tau``
    (Wire-A substrate-voice annotation, default 300.0). Before the split
    both decay paths shared a single tau, which made Wire-A's
    annotation decay too aggressively to be expressive at test time
    (Roy-3c-bisect A2 confirmation).
    """

    def test_decay_uses_dedicated_tau_not_reward_bias_tau(self):
        """``decay_cluster_reward_biases`` reads
        ``cluster_reward_bias_decay_tau``, NOT ``reward_bias_decay_tau``.

        Set the two taus to visibly different values and confirm the
        cluster bias decays per the cluster-specific tau. With
        reward_bias_decay_tau=10 (fast) and
        cluster_reward_bias_decay_tau=200 (slow), one decay tick
        shrinks the bias by ~1/200 = 0.5% — far less than the ~10%
        that a tau=10 decay would produce.
        """
        from maxim.decisions.nac import NAc, NACConfig

        config = NACConfig(
            reward_bias_decay_tau=10.0,
            cluster_reward_bias_decay_tau=200.0,
        )
        nac = NAc(config=config)
        nac.update_cluster_reward("agent", "c1", "tool:foo", reward=10.0)
        before = nac.cluster_reward_bias("agent", "c1", "tool:foo")
        assert before > 0.5  # Should be clamped near +1.0.

        nac.decay_cluster_reward_biases()
        after = nac.decay_cluster_reward_biases()  # noqa: F841
        # The cluster bias used tau=200 — after two ticks the bias
        # should still be > 99% of the start (decay factor 0.005/tick).
        observed = nac.cluster_reward_bias("agent", "c1", "tool:foo")
        expected_per_tick = 1.0 - (1.0 / 200.0)
        expected_after_two = before * (expected_per_tick**2)
        # Tight bound: within 1% of the cluster-tau-based prediction.
        assert observed == pytest.approx(expected_after_two, rel=0.01)
        # Sanity: if decay had used reward_bias_decay_tau=10, the bias
        # would be down to ~before * 0.81 after two ticks — observed
        # must be far above that.
        wrong_per_tick = 1.0 - (1.0 / 10.0)
        wrong_after_two = before * (wrong_per_tick**2)
        assert observed > wrong_after_two * 1.1

    def test_default_is_300(self):
        """Pin the default so a future re-tune is a reviewed change.

        The 300 value comes from the Phase 1 calibration math in
        cluster_reward_bias_decay_tau_split.md and was confirmed by
        the user 2026-05-26. Changing this default ships a behavioral
        shift to every NAc consumer; require an explicit reviewed PR.
        """
        from maxim.decisions.nac import NACConfig

        assert NACConfig().cluster_reward_bias_decay_tau == 300.0

    def test_env_override_applied(self):
        """``MAXIM_NAC_CLUSTER_REWARD_BIAS_DECAY_TAU`` overrides config.

        Mirrors the temporal_credit_weight env override pattern, but
        with stricter semantics: invalid/out-of-range values fall back
        to the default + WARN rather than silently clamping (see
        nac.py NAc.__init__ docstring + plan doc).
        """
        import os

        from maxim.decisions.nac import NAc, NACConfig

        os.environ["MAXIM_NAC_CLUSTER_REWARD_BIAS_DECAY_TAU"] = "500.0"
        try:
            nac = NAc(config=NACConfig())
            assert nac.config.cluster_reward_bias_decay_tau == pytest.approx(500.0)
        finally:
            os.environ.pop("MAXIM_NAC_CLUSTER_REWARD_BIAS_DECAY_TAU", None)

    def test_env_override_out_of_range_warns(self, caplog):
        """Out-of-range env values fall back to default and emit WARNING.

        Calibration knob: clamping silently to a bound would hide an
        operator misconfiguration, so we keep the default and surface
        the issue in logs.
        """
        import logging
        import os

        from maxim.decisions.nac import NAc, NACConfig

        # Out of range — too low (under 50).
        os.environ["MAXIM_NAC_CLUSTER_REWARD_BIAS_DECAY_TAU"] = "10.0"
        try:
            with caplog.at_level(logging.WARNING, logger="maxim.decisions.nac"):
                nac = NAc(config=NACConfig())
            assert nac.config.cluster_reward_bias_decay_tau == pytest.approx(300.0)
            assert any("out of range" in rec.message for rec in caplog.records if rec.levelno == logging.WARNING)
        finally:
            os.environ.pop("MAXIM_NAC_CLUSTER_REWARD_BIAS_DECAY_TAU", None)
            caplog.clear()

        # Out of range — too high (over 1000).
        os.environ["MAXIM_NAC_CLUSTER_REWARD_BIAS_DECAY_TAU"] = "5000.0"
        try:
            with caplog.at_level(logging.WARNING, logger="maxim.decisions.nac"):
                nac = NAc(config=NACConfig())
            assert nac.config.cluster_reward_bias_decay_tau == pytest.approx(300.0)
            assert any("out of range" in rec.message for rec in caplog.records if rec.levelno == logging.WARNING)
        finally:
            os.environ.pop("MAXIM_NAC_CLUSTER_REWARD_BIAS_DECAY_TAU", None)
            caplog.clear()

        # Non-numeric — falls back + WARN.
        os.environ["MAXIM_NAC_CLUSTER_REWARD_BIAS_DECAY_TAU"] = "not_a_number"
        try:
            with caplog.at_level(logging.WARNING, logger="maxim.decisions.nac"):
                nac = NAc(config=NACConfig())
            assert nac.config.cluster_reward_bias_decay_tau == pytest.approx(300.0)
            assert any("not numeric" in rec.message for rec in caplog.records if rec.levelno == logging.WARNING)
        finally:
            os.environ.pop("MAXIM_NAC_CLUSTER_REWARD_BIAS_DECAY_TAU", None)
            caplog.clear()

        # Boundary values (50 and 1000) — in range, no WARN.
        os.environ["MAXIM_NAC_CLUSTER_REWARD_BIAS_DECAY_TAU"] = "50.0"
        try:
            with caplog.at_level(logging.WARNING, logger="maxim.decisions.nac"):
                nac = NAc(config=NACConfig())
            assert nac.config.cluster_reward_bias_decay_tau == pytest.approx(50.0)
            assert not any(rec.levelno == logging.WARNING for rec in caplog.records)
        finally:
            os.environ.pop("MAXIM_NAC_CLUSTER_REWARD_BIAS_DECAY_TAU", None)
            caplog.clear()

        os.environ["MAXIM_NAC_CLUSTER_REWARD_BIAS_DECAY_TAU"] = "1000.0"
        try:
            with caplog.at_level(logging.WARNING, logger="maxim.decisions.nac"):
                nac = NAc(config=NACConfig())
            assert nac.config.cluster_reward_bias_decay_tau == pytest.approx(1000.0)
            assert not any(rec.levelno == logging.WARNING for rec in caplog.records)
        finally:
            os.environ.pop("MAXIM_NAC_CLUSTER_REWARD_BIAS_DECAY_TAU", None)
            caplog.clear()

        # Empty string — silent fall-back to default (matches missing env).
        os.environ["MAXIM_NAC_CLUSTER_REWARD_BIAS_DECAY_TAU"] = ""
        try:
            with caplog.at_level(logging.WARNING, logger="maxim.decisions.nac"):
                nac = NAc(config=NACConfig())
            assert nac.config.cluster_reward_bias_decay_tau == pytest.approx(300.0)
            assert not any(rec.levelno == logging.WARNING for rec in caplog.records)
        finally:
            os.environ.pop("MAXIM_NAC_CLUSTER_REWARD_BIAS_DECAY_TAU", None)
            caplog.clear()
